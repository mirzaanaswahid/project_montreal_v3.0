#!/usr/bin/env python3
"""
eagle_agent.py - EAGLE-enabled UAV agent
Standalone agent implementing the decision scenarios and cooperative protocols.

Implements both objectives:
  • Objective 1 (Mission value):   J(t) over detected/investigated events with time discount
  • Objective 2 (Energy):          FEnergy = Σ u_i,t * P_shaft,i(t) * dt  −  Σ_y B_i(t)

Notes:
- In soaring/gliding the motor is OFF (u=0) and P_shaft=0, consistent with FEnergy.
- We keep running accumulators:
    self.cmotor_J  = Σ u * P_shaft * dt     (motor energy spent)
    self.egain_J   = Σ claimed thermal benefit (energy gained / saved)
  And expose FEnergy via self.compute_FEnergy().

- “Fallback Investigation” replaces “Last Resort”.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np

from config import PhoenixConfig
from communication import CommunicationNetwork, MessageType, Message
from thermals import Thermal
from events import GroundEvent

# ========================= Enums & dataclasses =========================

class AgentTier(Enum):
    """Agent health tiers for auction priority"""
    HEALTHY_NOMAD = "HEALTHY_NOMAD"      # >50% battery, not soaring
    HEALTHY_SOARING = "HEALTHY_SOARING"  # >50% battery, currently soaring
    LOW_BATTERY = "LOW_BATTERY"          # <50% battery


class ThermalPolicy(Enum):
    STRICT_ALT = 0   # no net Δh; motor-off savings only
    ALT_BAND   = 1   # allow Δh within [hmin, hmax]


@dataclass
class EventAuction:
    """Tracks an ongoing event investigation auction"""
    event_id: str
    event: GroundEvent
    initiator_id: str
    start_time: float
    tier_bids: Dict[str, List[Tuple[str, float]]]  # tier -> [(agent_id, cost)]
    current_tier: int = 0
    resolved: bool = False
    winner_id: Optional[str] = None


@dataclass
class ThermalAuction:
    """Tracks a thermal exploitation auction"""
    thermal_id: str
    thermal: Thermal
    initiator_id: str
    start_time: float
    bids: List[Tuple[str, float]]  # [(agent_id, benefit_score)]
    resolved: bool = False
    winner_id: Optional[str] = None


# =============================== Agent =================================

class EAGLEAgent:
    """
    Standalone EAGLE UAV agent with all necessary flight and decision capabilities.
    Implements the scenarios listed in the EAGLE document, plus both objectives.
    """
    # --------------------------- Construction ---------------------------
    def __init__(self,
                 cfg: PhoenixConfig,
                 uav_id: str,
                 comm_network: CommunicationNetwork,
                 home_region: Optional[str] = None):

        # Basic identification
        self.cfg = cfg
        self.uav_id = uav_id
        self.home_region = home_region

        # Core flight state
        self.pos = np.zeros(3)  # [x, y, z] position (m, UTM CRS)
        self.V_h = 0.0          # horizontal airspeed (m/s)
        self.vz = 0.0           # vertical speed (m/s)
        self.psi = 0.0          # heading (rad)
        self.t = 0.0            # sim time (s)
        self.energy_wh = cfg.batt_capacity_wh
        self.throttle = 0.0
        self.P_elec = cfg.p_avionics  # treated here as shaft/electrical equivalently

        # Flight modes
        self.flight_mode = "ground"                  # ground, armed, takeoff, mission, landing
        self.soaring_state = "normal"                # normal, thermal_exploitation, gliding
        self.event_mode = "idle"                     # idle, investigating, fallback_investigation

        # Navigation
        self.waypoints: List[np.ndarray] = []
        self.current_wp_index = -1
        self.target_alt = 0.0
        self.target_speed = 0.0
        self.target_heading = 0.0
        self.heading = self.psi
        self.dt = 1.0

        # Environment
        self.current_thermals: List[Thermal] = []
        self.current_events: List[GroundEvent] = []

        # State sets
        self.detected_event_ids: Set[str] = set()
        self.investigated_event_ids: Set[str] = set()
        self.detected_thermal_ids: Set[str] = set()
        self.exploited_thermal_ids: Set[str] = set()

        # Event investigation
        self.event_center: Optional[np.ndarray] = None
        self.event_radius = 50.0  # default loiter radius (m)

        # Thermal exploitation
        self.current_thermal: Optional[Thermal] = None
        self.exploitation_start_time = 0.0
        self._inside_thermal = False
        self._lost_thermal_recently = False

        # Route
        self.planned_route: List[np.ndarray] = []

        # Communication
        self.comm = comm_network
        self.comm.register_agent(uav_id, self.pos)
        self.peer_states: Dict[str, Dict[str, Any]] = {}

        # Auctions
        self.active_event_auctions: Dict[str, EventAuction] = {}
        self.active_thermal_auctions: Dict[str, ThermalAuction] = {}
        self.my_event_bids: Set[str] = set()
        self.my_thermal_bids: Set[str] = set()

        # Assignments
        self.assigned_event: Optional[GroundEvent] = None
        self.assigned_thermal: Optional[Thermal] = None

        # EAGLE parameters
        self.tier_timeout = getattr(cfg, 'eagle_tier_timeout_s', 5.0)
        self.auction_timeout = getattr(cfg, 'eagle_auction_timeout_s', 20.0)
        self.benefit_threshold = getattr(cfg, 'eagle_benefit_threshold_j', 500.0)

        # Thermal policy
        self.thermal_policy = ThermalPolicy.STRICT_ALT
        self.alt_band = (self.cfg.altitude_ref_m - self.cfg.alt_band_m,
                         self.cfg.altitude_ref_m + self.cfg.alt_band_m)

        # ---------------- Objectives accounting ----------------
        # FEnergy accumulators (Objective 2)
        self.cmotor_J = 0.0  # Σ u * P_shaft * dt when motor-on
        self.egain_J  = 0.0  # Σ claimed thermal benefit (expected energy gain/saved)
        self._thermal_gain_booked = False

        # Mission value parameters & log (Objective 1, J(t))
        self.obj1_weights = {
            "Low": 1.0, "L": 1.0,
            "Medium": 2.0, "M": 2.0,
            "High": 4.0, "H": 4.0, "HP": 4.0, "Critical": 4.0
        }
        self.obj1_alpha = 1.0
        self.obj1_beta  = 2.0
        self.obj1_lambda = 0.1
        # event_value_log[event_id] = {"detected":bool,"investigated":bool,"t_spawn":float,"t_detect":float,"priority":str}
        self.event_value_log: Dict[str, Dict[str, Any]] = {}

        # Probabilistic map lazy-load placeholders
        self._probmap_loaded = False
        self._probmap_meta = None
        self._probmap = None
        self._avg_metrics = None
        self._lc_ds = None
        self._grid_affine = None
        self._grid_crs = None
        self._season_map = None
        self._tod_map = None
        self._lc_code_to_idx = None
        self._to_grid_transform = None
        self._regions_catalog = None
        self._home_region_poly = None
        self._neighbor_regions = None

    # --------------------------- Convenience ----------------------------
    @property
    def vel(self) -> np.ndarray:
        return np.array([self.V_h * math.sin(self.psi),
                         self.V_h * math.cos(self.psi),
                         self.vz])

    @property
    def wind_xy(self) -> np.ndarray:
        return self.cfg.wind_enu[:2] if hasattr(self.cfg, 'wind_enu') else np.zeros(2)

    @property
    def tnow(self) -> float:
        return self.t

    def ground_speed(self, airspeed: float) -> float:
        return max(1.0, airspeed - float(np.linalg.norm(self.wind_xy)))

    def battery_pct(self) -> float:
        return 100.0 * self.energy_wh / max(self.cfg.batt_capacity_wh, 1e-6)

    # ------------------------- Top-level update -------------------------
    def update(self, dt: float, t: float,
               thermals: List[Thermal],
               events: List[GroundEvent],
               world=None):
        """Single public entry point each sim step."""
        self.dt = dt
        self.t = t
        self.current_thermals = thermals or []
        self.current_events = events or []

        # Pull wind from world, if provided
        if world is not None:
            vx, vy, vz = world.get_wind_at_position(self.pos)
            self.cfg.wind_enu[:] = [vx, vy, vz]

        # Update internal mission log for Objective 1 (detections)
        self._log_event_detections(self.current_events, t)

        # Process messages and make decisions
        messages = self.comm.get_messages(self.uav_id, t)
        self._process_messages(messages, t)

        # === Check 0: Absolute safety ===
        if self.battery_pct() < 10.0:
            if self.flight_mode != "landing":
                self.land()
            # No more actions this tick
            self._update_physics(dt)
            self._update_flight_mode()
            self.comm.update_agent_position(self.uav_id, self.pos)
            self.broadcast_state()
            return

        # === Check 0b: Fallback Investigation mode ===
        if self.event_mode == "fallback_investigation":
            self._continue_fallback_investigation()
            self._update_physics(dt)
            self._update_flight_mode()
            self.comm.update_agent_position(self.uav_id, self.pos)
            self.broadcast_state()
            return

        # === Check 1: HP events ===
        if self._check_for_high_priority_events(events, t):
            self._update_physics(dt)
            self._update_flight_mode()
            self.comm.update_agent_position(self.uav_id, self.pos)
            self.broadcast_state()
            return

        # === Check 1b: Handle handover requests ===
        if self._handle_handover_requests(t):
            self._update_physics(dt)
            self._update_flight_mode()
            self.comm.update_agent_position(self.uav_id, self.pos)
            self.broadcast_state()
            return

        # === Check 2: Energy management ===
        if self.battery_pct() < 30.0:
            self._handle_proactive_soaring(self.current_thermals, t)
        else:
            # Optional cooperative thermal bidding even when healthy
            took = self._cooperative_thermal_step(self.current_thermals, t)
            if not took:
                try:
                    _ = self._search_probabilistic_map(t)
                except Exception:
                    pass

        # Discovery sharing (thermals only; events handled by detector)
        self._announce_discoveries(self.current_thermals, self.current_events, t)

        # Resolve auctions at end of tick
        self._update_auctions(t)

        if getattr(self, "_lost_thermal_recently", False):
            self._lost_thermal_recently = False
            _ = self._handle_live_thermal_opportunities(t)

        # Physics & mode
        self._update_physics(dt)
        self._update_flight_mode()

        # Net update to network and peers
        self.comm.update_agent_position(self.uav_id, self.pos)
        self.broadcast_state()

    # ----------------------- Low-level flight model ---------------------
    def arm(self):
        if self.flight_mode == "ground":
            self.flight_mode = "armed"

    def takeoff(self, target_alt_m: Optional[float] = None):
        if target_alt_m is None:
            target_alt_m = self.cfg.altitude_ref_m
        self.target_alt = target_alt_m
        self.target_speed = self.cfg.cruise_speed
        self.flight_mode = "takeoff"

    def land(self):
        if self.flight_mode != "ground":
            self.flight_mode = "landing"
            self.target_alt = 0.0
            self.target_speed = self.cfg.landing_speed

    def set_waypoints(self, wps: List[Tuple[float, float, float]]):
        self.waypoints = [np.asarray(wp, dtype=float) for wp in wps]
        self.current_wp_index = 0 if wps else -1

    def set_target_heading(self, psi_rad: float):
        self.target_heading = psi_rad
        self.psi = psi_rad

    def set_target_speed(self, speed: float):
        self.target_speed = speed
        self.V_h = speed

    def _update_physics(self, dt: float):
        # Air-relative velocity
        v_air = np.array([self.V_h * math.sin(self.psi),
                          self.V_h * math.cos(self.psi),
                          self.vz])
        # Add wind
        v_enu = v_air + self.cfg.wind_enu
        # Position
        self.pos += v_enu * dt
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0
            self.vz = 0.0
            if self.flight_mode != "ground":
                self.flight_mode = "ground"
        # Power & energy (Objective 2 accounting)
        self._update_power()
        self.energy_wh = max(self.energy_wh - self.P_elec * dt / 3600.0, 0.0)
        # Motor-use accumulator for FEnergy (u=1 if not soaring/gliding)
        motor_on = (self.soaring_state not in ("thermal_exploitation", "gliding"))
        if motor_on:
            # Treat P_elec as shaft/electrical draw consistently
            self.cmotor_J += float(self.P_elec) * dt

    def _update_power(self):
        """
        Computes shaft/electrical power draw using a standard drag polar:
            CD = CD0 + k*CL^2,  D = 0.5*rho*S*V^2*CD,  P = D*V / eta
        In soaring or gliding, motor is OFF ⇒ P=0 by design.
        """
        # Motor strictly off when gliding
        if self.soaring_state == "gliding":
            self.throttle = 0.0
            self.P_elec = 0.0
            return
        # Motor off when inside thermal during exploitation
        if self.soaring_state == "thermal_exploitation" and self._inside_thermal:
            self.throttle = 0.0
            self.P_elec = 0.0
            return
        # Not moving → no draw
        if self.V_h < 0.1:
            self.throttle = 0.0
            self.P_elec = 0.0
            return
        # Aerodynamic power → electrical via prop efficiency (shaft power proxy)
        rho = self.cfg.air_density
        S   = self.cfg.wing_area
        CD0 = self.cfg.CD0
        k   = self.cfg.induced_drag_k
        W   = self.cfg.mass * 9.81
        eta = max(self.cfg.prop_eta, 1e-3)
        V = max(self.V_h, 0.1)
        q = 0.5 * rho * V**2
        # Using induced drag via W^2 term (level flight CL ≈ W/(qS))
        D_par = q * S * CD0
        D_ind = k * W**2 / (q * S)
        P_aero = (D_par + D_ind) * V
        self.P_elec = P_aero / eta  # used as P_shaft in FEnergy

    def _update_flight_mode(self):
        if self.flight_mode == "takeoff":
            self.vz = self.cfg.climb_rate
            if self.pos[2] >= 0.95 * self.target_alt:
                self.flight_mode = "mission"
        elif self.flight_mode == "mission":
            if self.event_mode == "investigating":
                self._handle_event_investigation()
            elif self.event_mode == "fallback_investigation":
                self._continue_fallback_investigation()
            elif self.soaring_state == "thermal_exploitation":
                self._handle_thermal_exploitation()
            elif self.soaring_state == "gliding":
                self._handle_glide()
            else:
                self._update_waypoint_navigation()
        elif self.flight_mode == "landing":
            self.vz = self.cfg.landing_descent_rate
            if self.pos[2] <= 0.05:
                self.flight_mode = "ground"
                self.V_h = 0.0
                self.vz = 0.0

    def _update_waypoint_navigation(self):
        if not self.waypoints or self.current_wp_index < 0:
            return
        if self.current_wp_index >= len(self.waypoints):
            self.land()
            return
        wp = self.waypoints[self.current_wp_index]
        vec = wp[:2] - self.pos[:2]
        dist = float(np.linalg.norm(vec))
        if dist < self.cfg.waypoint_tol:
            self.current_wp_index += 1
        else:
            self.target_heading = math.atan2(vec[0], vec[1])
            self.target_alt = float(wp[2])
            self.target_speed = self.cfg.cruise_speed
            self._fly_towards_target()

    def _handle_event_investigation(self):
        if self.event_center is None:
            self.event_mode = "idle"
            return
        vec = self.event_center - self.pos[:2]
        dist = float(np.linalg.norm(vec))
        if dist > self.event_radius:
            self.target_heading = math.atan2(vec[0], vec[1])
        else:
            tangent = np.array([-vec[1], vec[0]])
            n = float(np.linalg.norm(tangent))
            if n > 1e-6:
                tangent = tangent / n
                self.target_heading = math.atan2(tangent[0], tangent[1])
            else:
                self.target_heading = self.psi + 0.1
        self.psi = self.target_heading
        self.V_h = self.cfg.V_loiter
        self.vz = 0.0

    def _handle_glide(self):
        # Simple glide back to patrol route at min power
        self.target_speed = self.cfg.V_cmd_min
        self.vz = self.cfg.descent_rate
        self._goto_patrol_route()
        self._fly_towards_target()
        # When back near patrol and at ref altitude, exit glide
        if abs(self.pos[2] - self.cfg.altitude_ref_m) < 5.0:
            self.soaring_state = "normal"

    def _handle_thermal_exploitation(self):
        if self.current_thermal is None:
            self.soaring_state = "normal"
            self._inside_thermal = False
            return
        th = self.current_thermal
        elapsed = self.t - self.exploitation_start_time
        if elapsed > self.cfg.max_thermal_dwell_s or not th.active(self.t):
            # Leaving thermal ⇒ glide
            self.soaring_state = "gliding"
            self.current_thermal = None
            self._thermal_gain_booked = False
            self._inside_thermal = False
            self._goto_patrol_route()
            return
        dist_xy = float(np.linalg.norm(self.pos[:2] - np.array(th.center)))
        alt_ok = (self.pos[2] >= getattr(th, "base_height", 0.0)) and (self.pos[2] <= th.top_height)
        self._inside_thermal = (dist_xy <= th.radius) and alt_ok
        if not self._inside_thermal:
            self._goto(th.center[0], th.center[1])
            self.target_speed = max(self.cfg.min_speed, self.cfg.V_cmd_min)
            if self.pos[2] < getattr(th, "base_height", 0.0):
                self._climb(min(self.cfg.altitude_max_soar_m, getattr(th, "base_height", 0.0)))
            self._fly_towards_target()
            return
        # spiral
        angle = elapsed * 0.1
        radius = 0.8 * th.radius
        target = np.array([th.center[0] + radius * math.cos(angle),
                           th.center[1] + radius * math.sin(angle)])
        vec = target - self.pos[:2]
        if np.linalg.norm(vec) > 1.0:
            self.target_heading = math.atan2(vec[0], vec[1])
            self.psi = self.target_heading
        self.V_h = self.cfg.min_speed
        self.vz = 0.0

    # ----------------------- Decision Components -----------------------
    def get_tier(self) -> AgentTier:
        b = self.battery_pct()
        if b < 50.0:
            return AgentTier.LOW_BATTERY
        return AgentTier.HEALTHY_SOARING if self.soaring_state == "thermal_exploitation" else AgentTier.HEALTHY_NOMAD

    def _specific_energy_per_meter(self, V_air: float, wind_xy: np.ndarray, u_hat: Optional[np.ndarray] = None) -> float:
        rho = self.cfg.air_density
        S   = self.cfg.wing_area
        CD0 = self.cfg.CD0
        k   = self.cfg.induced_drag_k
        W   = self.cfg.mass * 9.81
        eta = max(self.cfg.prop_eta, 1e-3)
        q = 0.5 * rho * V_air**2
        D_par = q * S * CD0
        D_ind = k * W**2 / (q * S)
        P = (D_par + D_ind) * V_air / eta
        if u_hat is None:
            track = self.vel[:2]
            track = track / (np.linalg.norm(track) + 1e-6)
        else:
            track = u_hat / (np.linalg.norm(u_hat) + 1e-6)
        V_gnd = max(1.0, V_air + float(np.dot(wind_xy, track)))
        return float(P / V_gnd)

    def calculate_investigation_cost(self, event: GroundEvent) -> float:
        p0 = self.pos.copy()
        pe = np.array([event.cx, event.cy, p0[2]])
        d_xy = float(np.linalg.norm(p0[:2] - pe[:2]))
        V_cruise = self.cfg.V_range_opt
        leg_vector = pe[:2] - p0[:2]
        spec_e = self._specific_energy_per_meter(V_cruise, self.wind_xy, leg_vector)
        E_transit = spec_e * d_xy
        phi = self.cfg.loiter_bank_deg * np.pi/180.0
        n = 1.0 / max(np.cos(phi), 0.2)
        V_loiter = self.cfg.V_loiter
        rho, S, CD0, k, W, eta = (self.cfg.air_density, self.cfg.wing_area,
                                  self.cfg.CD0, self.cfg.induced_drag_k,
                                  self.cfg.mass*9.81, self.cfg.prop_eta)
        q = 0.5 * rho * V_loiter**2
        D_par = q * S * CD0
        D_ind = n**2 * k * W**2 / (q * S)
        P_turn = (D_par + D_ind) * V_loiter / max(eta, 1e-3)
        R = V_loiter**2 / (9.81 * np.tan(phi))
        T_orbit = (2*np.pi*R) / max(self.ground_speed(V_loiter), 1.0)
        E_loiter = P_turn * T_orbit
        return float(E_transit + E_loiter)

    def calculate_thermal_benefit(self, thermal_or_msg: Any) -> float:
        p = self.pos.copy()
        # Message dict path
        if isinstance(thermal_or_msg, dict):
            center = np.array(thermal_or_msg.get("position", [p[0], p[1]]))
            radius = float(thermal_or_msg.get("radius", 50.0))
            strength = float(thermal_or_msg.get("strength", 1.0))
            top_height = float(thermal_or_msg.get("top_height", p[2] + 50.0))
            base_height = float(thermal_or_msg.get("base_height", 0.0))
            spawn_time = float(thermal_or_msg.get("spawn_time", -1e12))
            end_time = float(thermal_or_msg.get("end_time", 1e12))
            if not (spawn_time <= self.tnow < end_time):
                return -np.inf
            time_remaining = end_time - self.tnow
        else:
            th: Thermal = thermal_or_msg
            center = np.array(th.center[:2])
            radius = float(th.radius)
            strength = float(th.strength)
            top_height = float(th.top_height)
            base_height = float(getattr(th, "base_height", 0.0))
            time_remaining = float(th.time_remaining(self.tnow))
        r_xy = float(np.linalg.norm(p[:2] - center))
        t_entry = r_xy / max(self.ground_speed(self.cfg.V_range_opt), 1.0)
        w_bar = self._expected_wair(thermal_or_msg, p[:2])
        if w_bar <= 0.2:
            return -np.inf
        dwell = min(self.cfg.max_thermal_dwell_s, time_remaining - t_entry)
        if dwell <= 0:
            return -np.inf
        if self.thermal_policy is ThermalPolicy.STRICT_ALT:
            P_base = self._loiter_power_in_still_air()
            E_saved = P_base * dwell * self.cfg.motor_cut_fraction
            E_gain  = 0.0
        else:
            hmax = min(self.alt_band[1], top_height)
            dh   = max(0.0, hmax - p[2])
            t_climb = min(dwell, dh / w_bar) if w_bar > 1e-6 else 0.0
            E_saved = self._loiter_power_in_still_air() * t_climb * self.cfg.motor_cut_fraction
            E_gain  = self.cfg.mass * 9.81 * (w_bar * t_climb)
        d_detour = self._incremental_detour_length(center[:2])
        leg_vector = center[:2] - p[:2]
        E_detour = self._specific_energy_per_meter(self.cfg.V_range_opt, self.wind_xy, leg_vector) * d_detour
        C = self._thermal_confidence(thermal_or_msg)
        return float(C * (E_gain + E_saved) - E_detour)

    def _expected_wair_from_metrics(self, th_msg: dict, x: float, y: float, z: float, t: float) -> float:
        cx, cy = th_msg.get("position", [x, y])
        cx = float(cx); cy = float(cy)
        radius = float(th_msg.get("radius", 50.0))
        strength = float(th_msg.get("strength", 1.0))
        z_base = float(th_msg.get("base_height", 0.0))
        z_top  = float(th_msg.get("top_height", z + 50.0))
        t0 = float(th_msg.get("spawn_time", -1e12))
        t1 = float(th_msg.get("end_time",  1e12))
        if not (t0 <= t < t1):
            return 0.0
        if z < z_base or z > z_top:
            return 0.0
        dist = math.hypot(x - cx, y - cy)
        if dist > radius:
            return 0.0
        radial = max(0.0, 1.0 - dist / max(radius, 1e-6))
        span = max(1e-6, z_top - z_base)
        hz = (z - z_base) / span
        height = hz/0.7 if hz <= 0.7 else (1.0 - hz) / 0.3
        height = max(0.0, min(1.0, height))
        return float(strength * radial * height)

    def _expected_wair(self, thermal_or_msg: Any, pos_xy: np.ndarray) -> float:
        x, y, z = pos_xy[0], pos_xy[1], self.pos[2]
        if hasattr(thermal_or_msg, "w"):
            return max(0.0, float(thermal_or_msg.w((x, y, z), self.tnow)))
        if isinstance(thermal_or_msg, dict):
            return max(0.0, float(self._expected_wair_from_metrics(thermal_or_msg, x, y, z, self.tnow)))
        return 0.0

    def _loiter_power_in_still_air(self) -> float:
        phi = self.cfg.loiter_bank_deg * np.pi/180.0
        n = 1.0 / max(np.cos(phi), 0.2)
        V_loiter = self.cfg.V_loiter
        rho, S, CD0, k, W, eta = (self.cfg.air_density, self.cfg.wing_area,
                                  self.cfg.CD0, self.cfg.induced_drag_k,
                                  self.cfg.mass*9.81, self.cfg.prop_eta)
        q = 0.5 * rho * V_loiter**2
        D_par = q * S * CD0
        D_ind = n**2 * k * W**2 / (q * S)
        P_turn = (D_par + D_ind) * V_loiter / max(eta, 1e-3)
        return float(P_turn)

    def _incremental_detour_length(self, thermal_pos_xy: np.ndarray) -> float:
        if not self.planned_route:
            return float(np.linalg.norm(self.pos[:2] - thermal_pos_xy))
        min_dist = float('inf')
        closest_point = None
        for wp in self.planned_route:
            dist = float(np.linalg.norm(wp[:2] - thermal_pos_xy))
            if dist < min_dist:
                min_dist = dist
                closest_point = wp
        if closest_point is None:
            return float(np.linalg.norm(self.pos[:2] - thermal_pos_xy))
        d1 = float(np.linalg.norm(self.pos[:2] - thermal_pos_xy))
        d2 = float(np.linalg.norm(thermal_pos_xy - closest_point[:2]))
        return d1 + d2

    def _thermal_key(self, center, top_height, radius) -> str:
        x, y = np.round(center[0], 1), np.round(center[1], 1)
        th, r = round(top_height, 0), round(radius, 1)
        return f"th_{x}_{y}_{th}_{r}"

    def _thermal_confidence(self, thermal_or_msg: Any) -> float:
        if isinstance(thermal_or_msg, dict):
            return 0.7
        return float(np.clip(getattr(thermal_or_msg, "confidence", 0.7), 0.0, 1.0))

    # -------------------------- Helpers (nav) ---------------------------
    def _distance_to(self, target_pos) -> float:
        tp = np.array(target_pos, dtype=float)
        return float(np.linalg.norm(self.pos[:2] - tp[:2]))

    def _goto(self, x: float, y: float):
        vec = np.array([x, y]) - self.pos[:2]
        if np.linalg.norm(vec) > 1.0:
            self.target_heading = math.atan2(vec[0], vec[1])

    def _climb(self, target_alt: float):
        alt_diff = target_alt - self.pos[2]
        if abs(alt_diff) > 0.5:
            self.target_alt = np.clip(target_alt, self.alt_band[0], self.cfg.altitude_max_soar_m)

    def _goto_patrol_route(self):
        if self.waypoints and 0 <= self.current_wp_index < len(self.waypoints):
            wp = self.waypoints[self.current_wp_index]
            self._goto(wp[0], wp[1])
            self.target_alt = self.cfg.altitude_ref_m

    def _fly_towards_target(self):
        err = math.atan2(math.sin(self.target_heading - self.psi), math.cos(self.target_heading - self.psi))
        self.psi += np.clip(self.cfg.k_psi * err, -self.cfg.psi_dot_max, self.cfg.psi_dot_max) * self.dt
        self.V_h += np.clip(self.target_speed - self.V_h, -1.0, 1.0) * 0.1 * self.dt
        self.V_h = np.clip(self.V_h, self.cfg.V_cmd_min, self.cfg.V_cmd_max)
        alt_err = self.target_alt - self.pos[2]
        self.vz = np.clip(0.2 * alt_err, self.cfg.descent_rate, self.cfg.climb_rate)

    # ----------------------- Scenario implementns ----------------------
    def _start_event_investigation(self, cx: float, cy: float):
        self.event_mode = "investigating"
        self.event_center = np.array([cx, cy])

    def _continue_fallback_investigation(self):
        # Maintain one orbit then land to ensure safety (Fallback Investigation)
        if self.assigned_event is None:
            self.event_mode = "idle"
            return
        self._start_event_investigation(self.assigned_event.cx, self.assigned_event.cy)
        if not hasattr(self, "_fallback_start"):
            self._fallback_start = self.t
        orbit_time = max(10.0, 2*np.pi*(self.event_radius)/max(self.cfg.V_loiter, 1.0))
        if self.t - self._fallback_start > orbit_time:
            self.land()

    def _check_for_high_priority_events(self, events: List[GroundEvent], t: float) -> bool:
        # Pick first HP event not yet investigated, within EO/IR slant range
        REOIR = getattr(self.cfg, "sensor_slant_max_m", 800.0)
        for e in events or []:
            if getattr(e, "level", "Medium") not in ("High", "HP", "Critical"):
                continue
            if e.id in self.investigated_event_ids:
                continue
            dx = self.pos[0] - e.cx
            dy = self.pos[1] - e.cy
            slant = math.sqrt(dx*dx + dy*dy + (self.pos[2]**2))
            if slant > REOIR:
                continue
            # Decision: claim or handover
            cost = self.calculate_investigation_cost(e)
            reserve_J = getattr(self.cfg, "battery_reserve_J", 0.0)
            bi_J = self.energy_wh * 3600.0
            feasible = (bi_J - cost) >= reserve_J
            if feasible:
                # Claim locally
                self._claim_and_investigate(e)
                self.comm.broadcast(self.uav_id, MessageType.TASK_CLAIMED,
                                    {"task_id": e.id, "task_type": "EVENT",
                                     "claimed_by": self.uav_id, "winning_bid": cost}, priority=1)
            else:
                # Broadcast handover request
                data = {
                    "event_id": getattr(e, "id", f"evt_{round(e.cx,1)}_{round(e.cy,1)}"),
                    "event_position": [e.cx, e.cy],
                    "event_type": getattr(e, "level", "High"),
                    "priority": 1.0,
                    "reason": "cannot_afford",
                    "timestamp": t
                }
                self.comm.broadcast(self.uav_id, MessageType.HANDOVER_REQUEST, data, priority=1)
                # Create a local auction record
                A = EventAuction(event_id=data["event_id"], event=e, initiator_id=self.uav_id,
                                 start_time=t, tier_bids={tier.value: [] for tier in AgentTier})
                self.active_event_auctions[data["event_id"]] = A
            return True
        return False

    def _handle_handover_requests(self, t: float) -> bool:
        took_action = False
        for event_id, A in list(self.active_event_auctions.items()):
            if A.resolved:
                continue
            # Only consider within 2 km bidding pool
            dist = float(np.linalg.norm(self.pos[:2] - np.array([A.event.cx, A.event.cy])))
            if dist > 2000.0:
                continue
            my_tier = self.get_tier().value
            tier_idx = A.current_tier
            if my_tier != list(AgentTier)[tier_idx].value:
                continue
            c = self.calculate_investigation_cost(A.event)
            reserve_J = getattr(self.cfg, "battery_reserve_J", 0.0)
            feasible = (self.energy_wh*3600.0 - c) >= reserve_J
            if feasible:
                bid_msg = {
                    "event_id": event_id,
                    "bid_cost": float(c),
                    "tier": my_tier,
                    "current_position": self.pos.tolist(),
                    "timestamp": t
                }
                self.comm.broadcast(self.uav_id, MessageType.EVENT_BID, bid_msg)
                self.my_event_bids.add(event_id)
                took_action = True
        return took_action

    def _handle_proactive_soaring(self, thermals: List[Thermal], t: float):
        self.current_thermals = thermals or []
        took = self._handle_live_thermal_opportunities(t)
        if not took:
            _ = self._simple_thermal_step(t)

    def _rank_live_thermal_opportunities(self, t: float):
        ranked = []
        seen = set()
        for th_id, A in self.active_thermal_auctions.items():
            th = A.thermal
            if hasattr(th, "active") and not th.active(t):
                continue
            b = float(self.calculate_thermal_benefit(th))
            ranked.append((th, b, "auction"))
            seen.add(th_id)
        for th in (self.current_thermals or []):
            if hasattr(th, "active") and not th.active(t):
                continue
            th_key = self._thermal_key(th.center, th.top_height, th.radius)
            if th_key in seen:
                continue
            b = float(self.calculate_thermal_benefit(th))
            ranked.append((th, b, "world"))
        ranked.sort(key=lambda r: r[1], reverse=True)
        return ranked

    def _handle_live_thermal_opportunities(self, t: float) -> bool:
        candidates = self._rank_live_thermal_opportunities(t)
        for th, benefit, src in candidates:
            if benefit <= self.benefit_threshold:
                break
            th_key = self._thermal_key(th.center, th.top_height, th.radius)
            if th_key in self.my_thermal_bids:
                continue
            if src == "world" and th_key not in self.active_thermal_auctions:
                # Announce thermal discovery
                self.comm.broadcast(self.uav_id, MessageType.THERMAL_DISCOVERED, {
                    "thermal_id": th_key,
                    "position": list(th.center),
                    "strength": float(getattr(th, "strength", 1.0)),
                    "radius": float(getattr(th, "radius", 50.0)),
                    "top_height": float(getattr(th, "top_height", self.pos[2] + 50.0)),
                    "base_height": float(getattr(th, "base_height", 0.0)),
                    "spawn_time": float(getattr(th, "spawn_time", t)),
                    "end_time": float(getattr(th, "end_time", t + 300.0)),
                    "benefit_score": float(benefit)
                })
                self.active_thermal_auctions[th_key] = ThermalAuction(
                    thermal_id=th_key, thermal=th, initiator_id=self.uav_id, start_time=self.tnow, bids=[]
                )
            self._compete_for_thermal(th, benefit)
            return True
        return False

    def _announce_discoveries(self, thermals: List[Thermal], events: List[GroundEvent], t: float):
        for thermal in thermals:
            th_key = self._thermal_key(thermal.center, thermal.top_height, thermal.radius)
            if th_key in self.detected_thermal_ids:
                continue
            self.detected_thermal_ids.add(th_key)
            benefit = self.calculate_thermal_benefit(thermal)
            if benefit > 0:
                self.comm.broadcast(self.uav_id, MessageType.THERMAL_DISCOVERED, {
                    "thermal_id": th_key,
                    "position": list(thermal.center),
                    "strength": float(getattr(thermal, "strength", 1.0)),
                    "radius": float(getattr(thermal, "radius", 50.0)),
                    "top_height": float(getattr(thermal, "top_height", self.pos[2] + 50.0)),
                    "base_height": float(getattr(thermal, "base_height", 0.0)),
                    "spawn_time": float(getattr(thermal, "spawn_time", t)),
                    "end_time": float(getattr(thermal, "end_time", t + 300.0)),
                    "benefit_score": float(benefit)
                })

    # -------------------------- Auctions & Comms ------------------------
    def _process_messages(self, messages: List[Message], t: float):
        # Create or update auctions and peer states
        for msg in messages or []:
            if msg.msg_type == MessageType.HANDOVER_REQUEST:
                self._handle_handover_request_msg(msg, t)
            elif msg.msg_type == MessageType.EVENT_BID:
                self._handle_event_bid_msg(msg)
            elif msg.msg_type == MessageType.TASK_CLAIMED:
                self._handle_task_claimed_msg(msg)
            elif msg.msg_type == MessageType.THERMAL_DISCOVERED:
                self._handle_thermal_discovered_msg(msg)
            elif msg.msg_type == MessageType.LIVE_THERMAL_BID:
                self._handle_thermal_bid_msg(msg)
            elif msg.msg_type == MessageType.THERMAL_CLAIMED:
                self._handle_thermal_claimed_msg(msg)
            elif msg.msg_type in (MessageType.AGENT_STATE, MessageType.HEARTBEAT):
                self._update_peer_state(msg)
            # else ignore

    def _handle_handover_request_msg(self, msg: Message, t: float):
        data = msg.data
        # Build a stable event_id if missing
        event_id = data.get("event_id") or f"evt_{round(data.get('event_position',[0,0])[0],1)}_{round(data.get('event_position',[0,0])[1],1)}_{int(msg.timestamp)}"
        if event_id in self.active_event_auctions:
            return
        # Minimal GroundEvent proxy
        cx, cy = data.get("event_position", [0.0, 0.0])
        level = data.get("event_type", "High")
        e = GroundEvent(cx=cx, cy=cy, level=level)
        e.id = event_id
        auction = EventAuction(
            event_id=event_id, event=e, initiator_id=msg.sender_id, start_time=msg.timestamp,
            tier_bids={tier.value: [] for tier in AgentTier}
        )
        self.active_event_auctions[event_id] = auction

    def _handle_event_bid_msg(self, msg: Message):
        data = msg.data
        event_id = data.get("event_id")
        if not event_id or event_id not in self.active_event_auctions:
            return
        tier = data.get("tier", self.get_tier().value)
        bid_cost = float(data.get("bid_cost", 1e18))
        agent_id = msg.sender_id
        A = self.active_event_auctions[event_id]
        A.tier_bids.setdefault(tier, []).append((agent_id, bid_cost))

    def _handle_task_claimed_msg(self, msg: Message):
        data = msg.data
        # Accept both schemas
        event_id = data.get("event_id") or (data.get("task_type") == "EVENT" and data.get("task_id"))
        winner_id = data.get("winner_id") or data.get("claimed_by")
        if event_id and event_id in self.active_event_auctions:
            A = self.active_event_auctions[event_id]
            A.resolved = True
            A.winner_id = winner_id
            self.my_event_bids.discard(event_id)

    def _handle_thermal_discovered_msg(self, msg: Message):
        th_key = msg.data.get("thermal_id")
        if not th_key or th_key in self.active_thermal_auctions:
            return
        thermal = Thermal(center=(msg.data["position"][0], msg.data["position"][1]), condition='Medium', t0=msg.timestamp)
        # override with any provided
        thermal.strength = float(msg.data.get("strength", getattr(thermal, "strength", 1.0)))
        thermal.radius = float(msg.data.get("radius", getattr(thermal, "radius", 50.0)))
        thermal.top_height = float(msg.data.get("top_height", getattr(thermal, "top_height", self.pos[2]+50.0)))
        if "base_height" in msg.data:
            thermal.base_height = float(msg.data["base_height"])
        if "spawn_time" in msg.data:
            thermal.spawn_time = float(msg.data["spawn_time"])
        if "end_time" in msg.data:
            thermal.lifetime = max(1.0, float(msg.data["end_time"]) - thermal.spawn_time)
        auction = ThermalAuction(thermal_id=th_key, thermal=thermal,
                                 initiator_id=msg.sender_id, start_time=msg.timestamp, bids=[])
        self.active_thermal_auctions[th_key] = auction

    def _handle_thermal_bid_msg(self, msg: Message):
        th_id = msg.data.get("thermal_id")
        if th_id not in self.active_thermal_auctions:
            return
        auction = self.active_thermal_auctions[th_id]
        agent_id = msg.sender_id
        benefit = float(msg.data.get("benefit_score", msg.data.get("bid_score", -1e18)))
        # keep max per-agent
        bids = {a: b for a, b in auction.bids}
        prev = bids.get(agent_id, -1e18)
        if benefit > prev:
            bids[agent_id] = benefit
            auction.bids = list(bids.items())

    def _handle_thermal_claimed_msg(self, msg: Message):
        th_id = msg.data.get("thermal_id")
        winner_id = msg.data.get("winner_id") or msg.data.get("claimed_by")
        if th_id in self.active_thermal_auctions:
            T = self.active_thermal_auctions[th_id]
            T.resolved = True
            T.winner_id = winner_id
        self.my_thermal_bids.discard(th_id)

    def _update_auctions(self, t: float):
        # ----- EVENTS -----
        for event_id, A in list(self.active_event_auctions.items()):
            if A.resolved:
                del self.active_event_auctions[event_id]
                self.my_event_bids.discard(event_id)
                continue
            elapsed = t - A.start_time
            tier_idx = min(int(elapsed // self.tier_timeout), len(AgentTier)-1)
            if tier_idx > A.current_tier or elapsed >= self.auction_timeout:
                prev_tier = list(AgentTier)[A.current_tier].value
                bids = A.tier_bids.get(prev_tier, [])
                if bids:
                    winner_id, min_cost = min(bids, key=lambda x: (x[1], x[0]))
                    A.winner_id, A.resolved = winner_id, True
                    self.comm.broadcast(self.uav_id, MessageType.TASK_CLAIMED,
                                        {"task_id": event_id, "task_type": "EVENT",
                                         "claimed_by": winner_id, "winning_bid": float(min_cost)}, priority=1)
                    if winner_id == self.uav_id:
                        self._claim_and_investigate(A.event)
                A.current_tier = tier_idx
            # Hard timeout → Fallback Investigation by initiator
            if not A.resolved and elapsed >= self.auction_timeout and A.initiator_id == self.uav_id:
                self.assigned_event = A.event
                self.event_mode = "fallback_investigation"
                self._fallback_start = t
                # Prefer a canonical message name if available; otherwise, reuse existing enum
                msg_type = getattr(MessageType, "FALLBACK_INVESTIGATING",
                                   getattr(MessageType, "FALL_BACK_INVESTIGATING", MessageType.AGENT_STATE))
                self.comm.broadcast(self.uav_id, msg_type,
                                    {"event_id": A.event.id, "position": [A.event.cx, A.event.cy],
                                     "reason": "auction_timeout", "mode": "FALLBACK_INVESTIGATION",
                                     "timestamp": t}, priority=2)
                self._start_event_investigation(A.event.cx, A.event.cy)
                A.resolved = True
            if A.resolved:
                del self.active_event_auctions[event_id]
                self.my_event_bids.discard(event_id)

        # ----- THERMALS -----
        for th_id, T in list(self.active_thermal_auctions.items()):
            if T.resolved:
                del self.active_thermal_auctions[th_id]
                self.my_thermal_bids.discard(th_id)
                continue
            elapsed = t - T.start_time
            if elapsed >= self.tier_timeout:
                if T.bids:
                    winner_id, best = max(T.bids, key=lambda x: (x[1], x[0]))
                    T.winner_id, T.resolved = winner_id, True
                    self.comm.broadcast(self.uav_id, MessageType.THERMAL_CLAIMED,
                                        {"thermal_id": th_id, "claimed_by": winner_id,
                                         "investigation_start": t, "timestamp": t})
                    if winner_id == self.uav_id:
                        self._initiate_thermal_exploitation(T.thermal, t)
                    else:
                        self._lost_thermal_recently = True
                else:
                    T.resolved = True
            if T.resolved:
                del self.active_thermal_auctions[th_id]
                self.my_thermal_bids.discard(th_id)

    def _compete_for_thermal(self, thermal: Thermal, benefit_score: float):
        if benefit_score <= 0:
            return
        th_key = self._thermal_key(thermal.center, thermal.top_height, thermal.radius)
        self.comm.broadcast(self.uav_id, MessageType.LIVE_THERMAL_BID, {
            "thermal_id": th_key,
            "bid_score": float(benefit_score),
            "current_position": self.pos.tolist(),
            "battery_level": float(self.battery_pct()),
            "timestamp": self.tnow
        })
        self.my_thermal_bids.add(th_key)

    def _claim_and_investigate(self, event: GroundEvent):
        self.assigned_event = event
        self.investigated_event_ids.add(event.id)
        self._log_event_investigation(event.id)  # Objective 1 bookkeeping
        self._start_event_investigation(event.cx, event.cy)

    # ------------------------ Prob-map integration ----------------------
    # (unchanged except style/robustness)
    def _ensure_probmap_loaded(self) -> bool:
        if self._probmap_loaded:
            return True
        import os, json
        try:
            meta_path = getattr(self.cfg, 'probmap_meta_path', None)
            prob_path = getattr(self.cfg, 'probmap_prob_path', None)
            avg_npz_path = getattr(self.cfg, 'probmap_avg_npz_path', None)
            lc_raster_path = getattr(self.cfg, 'probmap_lc_raster_path', None)
            if not (meta_path and prob_path and lc_raster_path):
                return False
            if not (os.path.isfile(meta_path) and os.path.isfile(prob_path) and os.path.isfile(lc_raster_path)):
                return False
            with open(meta_path, 'r') as f:
                self._probmap_meta = json.load(f)
            import numpy as _np
            self._probmap = _np.load(prob_path)
            if avg_npz_path and os.path.isfile(avg_npz_path):
                self._avg_metrics = _np.load(avg_npz_path)
            import rasterio as _rio
            self._lc_ds = _rio.open(lc_raster_path)
            from rasterio import Affine as _Affine
            self._grid_affine = _Affine.from_gdal(*self._probmap_meta['grid_affine_transform_gdal'])
            self._grid_crs = str(self._probmap_meta['grid_crs'])
            self._season_map = self._probmap_meta['context_mappings']['season']
            self._tod_map = self._probmap_meta['context_mappings']['time_of_day']
            lc_map = self._probmap_meta['context_mappings']['land_cover_code_to_index']
            self._lc_code_to_idx = {int(k): v for k, v in (lc_map.items() if isinstance(lc_map, dict) else lc_map)}
            from config import UTM_CRS
            if str(UTM_CRS) != str(self._grid_crs):
                import pyproj
                transformer = pyproj.Transformer.from_crs(UTM_CRS, self._grid_crs, always_xy=True)
                self._to_grid_transform = transformer.transform
            else:
                self._to_grid_transform = None
            try:
                from events import load_region_catalog
                from config import GEOJSON_PATH, UTM_CRS as _UTM
                self._regions_catalog = load_region_catalog(GEOJSON_PATH, _UTM)
                if self.home_region:
                    target_key = self.home_region.lower()
                    for r in self._regions_catalog:
                        if r.key == target_key:
                            self._home_region_poly = r.geom
                            break
            except Exception:
                self._regions_catalog = None
            self._probmap_loaded = True
            return True
        except Exception:
            return False

    def _context_indices_from_time(self, timestamp: float) -> Tuple[Optional[int], Optional[int]]:
        import datetime as _dt
        base = _dt.datetime(2025, 6, 1, 12, 0, 0)
        current = base + _dt.timedelta(seconds=float(timestamp))
        season_name = None
        if current.month in [6, 7, 8]:
            season_name = "Summer"
        elif current.month in [5, 9]:
            season_name = "Spring/Fall"
        tod_name = None
        if 10 <= current.hour < 12:
            tod_name = "Morning"
        elif 12 <= current.hour < 16:
            tod_name = "Afternoon"
        elif 16 <= current.hour < 18:
            tod_name = "Late Afternoon"
        s_idx = self._season_map.get(season_name) if (self._season_map and season_name) else None
        t_idx = self._tod_map.get(tod_name) if (self._tod_map and tod_name) else None
        return s_idx, t_idx

    def _is_inside_patrol_or_border(self, x_utm: float, y_utm: float, thermal_xy_utm: Tuple[float, float]) -> bool:
        if not self._regions_catalog or not self._home_region_poly:
            return True
        try:
            from shapely.geometry import Point as _Point
            p_uav = _Point(x_utm, y_utm)
            p_th = _Point(thermal_xy_utm[0], thermal_xy_utm[1])
            if self._home_region_poly.contains(p_th):
                return True
            border_buf_m = 200.0
            if self._home_region_poly.boundary.buffer(border_buf_m).contains(p_uav):
                for r in self._regions_catalog:
                    if r.geom is self._home_region_poly:
                        continue
                    if r.geom.touches(self._home_region_poly) and r.geom.contains(p_th):
                        # Border handover announce
                        self.comm.broadcast(self.uav_id, MessageType.PATROL_HANDOVER_REQUEST, {
                            "requesting_agent_id": self.uav_id,
                            "patrol_area_id": getattr(r, "key", "adjacent"),
                            "current_waypoints": [wp.tolist() for wp in self.waypoints],
                            "reason": "thermal_opportunity",
                            "task_position": [thermal_xy_utm[0], thermal_xy_utm[1]],
                            "timestamp": self.tnow
                        })
                        return True
            return False
        except Exception:
            return True

    def _search_probabilistic_map(self, t: float) -> bool:
        if not self._ensure_probmap_loaded():
            return False
        season_idx, tod_idx = self._context_indices_from_time(t)
        if season_idx is None or tod_idx is None:
            return False
        import numpy as _np
        import rasterio as _rio
        from rasterio.transform import rowcol as _rowcol
        aoi_half = float(getattr(self.cfg, 'probmap_aoi_half_width_m', 500.0))
        x, y = float(self.pos[0]), float(self.pos[1])
        if self._to_grid_transform:
            gx, gy = self._to_grid_transform(x, y)
        else:
            gx, gy = x, y
        min_x = gx - aoi_half; max_x = gx + aoi_half
        min_y = gy - aoi_half; max_y = gy + aoi_half
        try:
            r_min, c_min = _rio.transform.rowcol(self._grid_affine, min_x, max_y)
            r_max, c_max = _rio.transform.rowcol(self._grid_affine, max_x, min_y)
        except Exception:
            return False
        grid_h = int(self._probmap_meta['grid_dimensions']['height'])
        grid_w = int(self._probmap_meta['grid_dimensions']['width'])
        r_min = max(0, r_min); c_min = max(0, c_min)
        r_max = min(grid_h - 1, r_max); c_max = min(grid_w - 1, c_max)
        if r_min > r_max or c_min > c_max:
            return False
        threshold = float(getattr(self.cfg, 'probmap_probability_threshold', 0.5))
        candidates: List[Tuple[Tuple[int,int], float, Tuple[float,float]]] = []
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cx_m, cy_m = self._grid_affine * (c + 0.5, r + 0.5)
                try:
                    lc = next(self._lc_ds.sample([(cx_m, cy_m)], indexes=1))[0]
                    if lc == self._lc_ds.nodata:
                        lc = 0
                except Exception:
                    lc = 0
                lc_idx = self._lc_code_to_idx.get(int(lc))
                if lc_idx is None:
                    continue
                try:
                    pprob = float(self._probmap[r, c, season_idx, tod_idx, lc_idx])
                except Exception:
                    pprob = 0.0
                if pprob >= threshold:
                    candidates.append(((r, c), pprob, (cx_m, cy_m)))
        if not candidates:
            return False
        # Convert to UTM if needed and filter by sector/border
        to_utm = None
        if self._to_grid_transform is not None:
            try:
                import pyproj
                from config import UTM_CRS
                inv_transformer = pyproj.Transformer.from_crs(self._grid_crs, UTM_CRS, always_xy=True)
                to_utm = inv_transformer.transform
            except Exception:
                to_utm = None
        filtered: List[Tuple[Tuple[int,int], float, Tuple[float,float], Tuple[float,float]]] = []
        for (r, c), prob, (gx_m, gy_m) in candidates:
            if to_utm:
                ux, uy = to_utm(gx_m, gy_m)
            else:
                ux, uy = gx_m, gy_m
            if self._is_inside_patrol_or_border(self.pos[0], self.pos[1], (ux, uy)):
                filtered.append(((r, c), prob, (gx_m, gy_m), (ux, uy)))
        if not filtered:
            return False
        # Rank by benefit + mild probability bonus
        best = None
        best_score = -1e18
        for (_idx, prob, (_gx, _gy), (ux, uy)) in filtered:
            th = Thermal(center=(ux, uy), condition='Medium', t0=t)
            try:
                if self._avg_metrics is not None:
                    pass
            except Exception:
                pass
            benefit_J = float(self.calculate_thermal_benefit(th))
            score = benefit_J + 0.1 * prob * self.cfg.batt_capacity_wh * 3600.0
            if score > best_score and benefit_J > self.benefit_threshold:
                best = th; best_score = score
        if best is None:
            return False
        # Execute soaring attempt immediately
        self._goto(best.center[0], best.center[1])
        self.target_speed = max(self.cfg.min_speed, self.cfg.V_cmd_min)
        self._initiate_thermal_exploitation(best, t)
        return True

    # ------------------------- Thermal execution -----------------------
    def _initiate_thermal_exploitation(self, thermal: Thermal, t: float):
        self.current_thermal = thermal
        self.soaring_state = "thermal_exploitation"
        self.exploitation_start_time = t
        self._inside_thermal = False
        # Book expected benefit once (for FEnergy accounting)
        if not self._thermal_gain_booked:
            b = max(0.0, float(self.calculate_thermal_benefit(thermal)))
            self.egain_J += b
            self._thermal_gain_booked = True
        self._goto(thermal.center[0], thermal.center[1])
        self.target_speed = max(self.cfg.min_speed, self.cfg.V_cmd_min)

    # ----------------------------- Objective helpers -------------------
    def compute_FEnergy(self) -> float:
        """
        Objective 2:
            FEnergy = Σ u_i,t * P_shaft,i(t) * dt  −  Σ_y B_i(t)
        Here:
            self.cmotor_J  = Σ u * P_shaft * dt
            self.egain_J   = Σ B (expected gain/saved)
        """
        return float(self.cmotor_J - self.egain_J)

    def mission_value_J(self) -> float:
        """
        Objective 1 (online value):
            J(t) = Σ_{e in E(t)} w_{p(e)} [ α·D_e + β·I_e ] · exp(-λ Δt_e)
        Uses self.event_value_log which is updated during flight.
        """
        total = 0.0
        for eid, rec in self.event_value_log.items():
            w = self.obj1_weights.get(rec.get("priority","Medium"), 1.0)
            D = 1.0 if rec.get("detected") else 0.0
            I = 1.0 if rec.get("investigated") else 0.0
            t_spawn = float(rec.get("t_spawn", self.t))
            t_detect = float(rec.get("t_detect", self.t)) if rec.get("detected") else self.t
            delta_t = max(0.0, t_detect - t_spawn)
            discount = math.exp(-self.obj1_lambda * delta_t)
            total += w * (self.obj1_alpha * D + self.obj1_beta * I) * discount
        return float(total)

    def _log_event_detections(self, events: List[GroundEvent], t: float):
        """Mark detections when within slant range; create/refresh tracking records."""
        REOIR = getattr(self.cfg, "sensor_slant_max_m", 800.0)
        for e in events or []:
            eid = getattr(e, "id", f"evt_{round(e.cx,1)}_{round(e.cy,1)}")
            if eid not in self.event_value_log:
                self.event_value_log[eid] = {
                    "detected": False,
                    "investigated": False,
                    "t_spawn": float(getattr(e, "spawn_time", t)),
                    "t_detect": None,
                    "priority": str(getattr(e, "level", "Medium"))
                }
            # Detection check
            dx = self.pos[0] - e.cx
            dy = self.pos[1] - e.cy
            slant = math.sqrt(dx*dx + dy*dy + (self.pos[2]**2))
            if slant <= REOIR and not self.event_value_log[eid]["detected"]:
                self.event_value_log[eid]["detected"] = True
                self.event_value_log[eid]["t_detect"] = float(getattr(e, "detection_time", t))

    def _log_event_investigation(self, event_id: str):
        """Mark investigation completion in the J(t) log."""
        rec = self.event_value_log.get(event_id)
        if rec is None:
            # create a minimal record if needed
            self.event_value_log[event_id] = {
                "detected": True, "investigated": True,
                "t_spawn": self.t, "t_detect": self.t, "priority": "High"
            }
        else:
            rec["investigated"] = True

    # ----------------------------- Comms -------------------------------
    def _simple_thermal_step(self, sim_time, world=None):
        decision = "patrol"
        self.target_alt = self.cfg.altitude_ref_m
        thermals = self.current_thermals or []
        best, best_score = None, -1e18
        for th in thermals:
            conf = self._thermal_confidence(th)
            if conf < getattr(self.cfg, "thermal_conf_min", 0.0) or not th.active(sim_time):
                continue
            benefit_J = self.calculate_thermal_benefit(th)
            if benefit_J > best_score and benefit_J > self.benefit_threshold:
                best, best_score = th, benefit_J
        if best is not None:
            decision = "soar"
            self._goto(best.center[0], best.center[1])
            if self._distance_to(best.center) <= best.radius and self.pos[2] < self.cfg.altitude_max_soar_m:
                self._climb(self.cfg.altitude_max_soar_m)
            elif self.pos[2] >= self.cfg.altitude_max_soar_m or not best.active(sim_time):
                self._goto_patrol_route()
                self.target_alt = self.cfg.altitude_ref_m
        self._fly_towards_target()
        return decision

    # --------------------------- Peer sharing --------------------------
    def _update_peer_state(self, msg: Message):
        peer_id = msg.sender_id
        self.peer_states[peer_id] = {
            "position": msg.data.get("position"),
            "battery_pct": msg.data.get("battery_pct") or msg.data.get("battery_level"),
            "tier": msg.data.get("tier") or msg.data.get("tier_status"),
            "mode": msg.data.get("mode") or msg.data.get("state"),
            "timestamp": msg.timestamp
        }

    def broadcast_state(self):
        self.comm.broadcast(
            self.uav_id,
            MessageType.AGENT_STATE,
            {
                "agent_id": self.uav_id,
                "position": self.pos.tolist(),
                "battery_level": float(self.battery_pct()),
                "current_task": self.event_mode if self.event_mode != "idle" else self.soaring_state,
                "task_id": getattr(self.assigned_event, "id", "") if self.assigned_event else "",
                "state": self.flight_mode,
                "health_status": "OK" if self.battery_pct() >= 10.0 else "CRITICAL",
                "tier_status": self.get_tier().value,
                # Objective snapshots for monitoring:
                "FEnergy_J": self.compute_FEnergy(),
                "J_value": self.mission_value_J(),
                "timestamp": self.tnow
            }
        )


# ======================= Collision Avoidance (APF) =====================

class CollisionAvoidance:
    """
    Artificial Potential Field collision avoidance
    Based on Section 3.1 of the EAGLE formulation
    """
    def __init__(self,
                 d_safe: float = 100.0,
                 d_obs_safe: float = 150.0,
                 r_safe: float = 50.0,
                 r_turn: float = 30.0):
        self.d_safe = d_safe
        self.d_obs_safe = d_obs_safe
        self.r_safe = r_safe
        self.r_turn = r_turn
        self.r_a = math.sqrt((r_safe + r_turn)**2 - r_turn**2)

    def calculate_avoidance_velocity(self,
                                     own_pos: np.ndarray,
                                     own_vel: np.ndarray,
                                     goal_vel: np.ndarray,
                                     neighbors: List[Dict[str, np.ndarray]],
                                     obstacles: List[Dict[str, np.ndarray]] = None,
                                     config: PhoenixConfig = None,
                                     current_heading: float = 0.0,
                                     dt: float = 1.0) -> np.ndarray:
        v_cmd = goal_vel.copy()
        for neighbor in neighbors or []:
            v_rep = self._repulsive_velocity(own_pos, own_vel, neighbor['pos'], neighbor['vel'], self.d_safe)
            v_cmd += v_rep
        if obstacles:
            for obstacle in obstacles:
                v_rep = self._repulsive_velocity(own_pos, own_vel, obstacle['pos'], obstacle['vel'], self.d_obs_safe)
                v_cmd += v_rep
        if config:
            V_max = config.V_cmd_max
            V_min = config.V_cmd_min
            k_psi = config.k_psi
            psi_dot_max = config.psi_dot_max
        else:
            V_max = 25.0; V_min = 12.0; k_psi = 1.0; psi_dot_max = 0.5
        if np.linalg.norm(v_cmd) > V_max:
            v_cmd = v_cmd / np.linalg.norm(v_cmd) * V_max
        psi_goal = math.atan2(v_cmd[0], v_cmd[1])
        psi_err  = np.arctan2(np.sin(psi_goal - current_heading), np.cos(psi_goal - current_heading))
        psi_dot  = np.clip(k_psi * psi_err, -psi_dot_max, psi_dot_max)
        V = np.clip(np.linalg.norm(v_cmd), V_min, V_max)
        psi_next = current_heading + psi_dot * dt
        return np.array([V * math.sin(psi_next), V * math.cos(psi_next), v_cmd[2]])

    def _repulsive_velocity(self,
                            p_i: np.ndarray,
                            v_i: np.ndarray,
                            p_j: np.ndarray,
                            v_j: np.ndarray,
                            d_min: float) -> np.ndarray:
        r = p_i - p_j
        d = float(np.linalg.norm(r))
        if d > 2 * d_min:
            return np.zeros(3)
        v_rel = v_i - v_j
        u_r = -np.dot(v_rel, r/d) if d > 1e-6 else np.linalg.norm(v_rel)
        if u_r <= 0:
            return np.zeros(3)
        Q = 4 * np.pi * self.r_a**2 * u_r
        if d > 1e-6:
            v_ind = Q * r / (4 * np.pi * d**3)
        else:
            v_ind = np.array([Q/(4*np.pi), 0, 0])
        v_cap = 10.0
        v_ind = np.clip(v_ind, -v_cap, v_cap)
        return v_ind
