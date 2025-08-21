#!/usr/bin/env python3
"""
eagle_agent.py - EAGLE-enabled UAV agent
Extends base UAVAgent with EAGLE decision-making capabilities
"""

import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


from config import PhoenixConfig
from communication import CommunicationNetwork, MessageType, Message
from thermals import Thermal
from events import GroundEvent


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


class EAGLEAgent:  # Remove inheritance from UAVAgent
    """
    Standalone EAGLE UAV agent with all necessary flight capabilities
    """
    
    def __init__(self, 
                 cfg: PhoenixConfig,
                 uav_id: str,
                 comm_network: CommunicationNetwork,
                 home_region: Optional[str] = None):
        
        # Basic identification
        self.cfg = cfg
        self.uav_id = uav_id
        
        # Core flight state (what was in UAVAgent)
        self.pos = np.zeros(3)  # [x, y, z] position
        self.V_h = 0.0  # Horizontal speed
        self.vz = 0.0   # Vertical speed
        self.psi = 0.0  # Heading (radians)
        self.t = 0.0    # Simulation time
        self.energy_wh = cfg.batt_capacity_wh  # Battery energy
        self.throttle = 0.0
        self.P_elec = cfg.p_avionics
        
        # Flight modes
        self.flight_mode = "ground"  # ground, armed, takeoff, mission, landing
        self.soaring_state = "normal"  # normal, thermal_exploitation, gliding
        self.event_mode = "idle"  # idle, investigating
        
        # Waypoint navigation
        self.waypoints = []
        self.current_wp_index = -1
        self.target_alt = 0.0
        self.target_speed = 0.0
        self.target_heading = 0.0
        
        # Event detection
        self.detected_event_ids = set()
        self.investigated_event_ids = set()
        self.detected_thermal_ids = set()
        self.exploited_thermal_ids = set()
        
        # Current environment
        self.current_thermals = []
        self.current_events = []
        
        # Event investigation state
        self.event_center = None
        self.event_radius = 50.0  # Default loiter radius
        
        # Thermal exploitation
        self.current_thermal = None
        self.exploitation_start_time = 0.0
        
        # Navigation
        self.planned_route = []
        self.heading = self.psi
        self.dt = 1.0
        
        # Communication
        self.comm = comm_network
        self.comm.register_agent(uav_id, self.pos)
        
        # EAGLE-specific state
        self.home_region = home_region
        self.peer_states = {}
        
        # Auction tracking
        self.active_event_auctions = {}
        self.active_thermal_auctions = {}
        self.my_event_bids = set()
        self.my_thermal_bids = set()
        
        # Task assignments
        self.assigned_event = None
        self.assigned_thermal = None
        
        # EAGLE parameters
        self.tier_timeout = getattr(cfg, 'eagle_tier_timeout_s', 5.0)
        self.auction_timeout = getattr(cfg, 'eagle_auction_timeout_s', 20.0)
        self.benefit_threshold = getattr(cfg, 'eagle_benefit_threshold_j', 500.0)
        
        # Thermal policy
        self.thermal_policy = ThermalPolicy.STRICT_ALT
        self.alt_band = (self.cfg.altitude_ref_m - self.cfg.alt_band_m,
                         self.cfg.altitude_ref_m + self.cfg.alt_band_m)
        
        # Energy tracking
        self.cmotor_J = 0.0           # Σ u_k * P_elec,k * Δt
        self.egain_J  = 0.0           # Σ B_i(t) for exploited thermals
        self._thermal_gain_booked = False
        
        # Thermal state tracking
        self._inside_thermal = False
        self._lost_thermal_recently = False
        
        # Logging
        self.eagle_logger = self._setup_eagle_logger()

        # --- Probabilistic map integration (lazy-loaded) ---
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
        self._to_grid_transform = None  # function: (x,y) in UTM -> (x,y) in grid CRS
        self._regions_catalog = None    # list of Region objects (UTM CRS)
        self._home_region_poly = None   # shapely polygon in UTM CRS
        self._neighbor_regions = None   # list of (name, poly) neighbors in UTM CRS
    
    @property
    def vel(self) -> np.ndarray:
        """Velocity vector [vx, vy, vz]"""
        return np.array([
            self.V_h * np.sin(self.psi),
            self.V_h * np.cos(self.psi),
            self.vz
        ])

    @property
    def wind_xy(self) -> np.ndarray:
        """Horizontal wind vector"""
        return self.cfg.wind_enu[:2] if hasattr(self.cfg, 'wind_enu') else np.zeros(2)

    @property
    def tnow(self) -> float:
        """Current simulation time"""
        return self.t

    def ground_speed(self, airspeed: float) -> float:
        """Ground speed given airspeed and wind"""
        # Simplified - assumes flying into/with wind
        return max(1.0, airspeed - np.linalg.norm(self.wind_xy))
    
    def _setup_eagle_logger(self):
        """Setup EAGLE-specific logging"""
        import logging
        logger = logging.getLogger(f"{self.uav_id}_EAGLE")
        logger.setLevel(logging.INFO)
        return logger
    
    def _Wh(self, J: float) -> float:
        """Convert Joules to Watt-hours"""
        return J / 3600.0
    
    def _J(self, Wh: float) -> float:
        """Convert Watt-hours to Joules"""
        return Wh * 3600.0
    
    # Add missing basic flight methods
    def arm(self):
        """Arm the UAV"""
        if self.flight_mode == "ground":
            self.flight_mode = "armed"
            self.eagle_logger.info(f"{self.uav_id} armed")
    
    def takeoff(self, target_alt_m: float = None):
        """Initiate takeoff"""
        if target_alt_m is None:
            target_alt_m = self.cfg.altitude_ref_m
        
        self.target_alt = target_alt_m
        self.target_speed = self.cfg.cruise_speed
        self.flight_mode = "takeoff"
        self.eagle_logger.info(f"{self.uav_id} taking off to {target_alt_m}m")
    
    def land(self):
        """Initiate landing"""
        if self.flight_mode != "ground":
            self.flight_mode = "landing"
            self.target_alt = 0.0
            self.target_speed = self.cfg.landing_speed
            self.eagle_logger.info(f"{self.uav_id} landing initiated")
    
    def set_waypoints(self, wps):
        """Set waypoints for navigation"""
        self.waypoints = [np.asarray(wp, dtype=float) for wp in wps]
        self.current_wp_index = 0 if wps else -1
        self.eagle_logger.info(f"{self.uav_id} received {len(self.waypoints)} waypoints")
    
    def battery_pct(self) -> float:
        """Get battery percentage"""
        return 100 * self.energy_wh / self.cfg.batt_capacity_wh
    
    def set_target_heading(self, psi_rad: float):
        """Set target heading in radians"""
        self.target_heading = psi_rad
        self.psi = psi_rad

    def set_target_speed(self, speed: float):
        """Set target speed in m/s"""
        self.target_speed = speed
        self.V_h = speed
    
    def update(self, dt: float, t: float, thermals: List[Thermal], events: List[GroundEvent], world=None):
        """Main update method - single public entry point for agent updates"""
        # Store environment
        self.current_thermals = thermals or []
        self.current_events = events or []
        
        # Update simulation time
        self.t = t
        
        # pull wind (incl. vertical updraft) from the world
        if world is not None:
            vx, vy, vz = world.get_wind_at_position(self.pos)
            self.cfg.wind_enu[:] = [vx, vy, vz]  # reuse existing paths
        
        # Run EAGLE decision-making step
        self._eagle_step(dt, t, thermals, events)
        
        # Update physics and guidance
        self._update_physics(dt)
        self._update_flight_mode()
    
    def _update_physics(self, dt: float):
        """Update position and energy"""
        # Update position based on velocity
        v_air = np.array([
            self.V_h * math.sin(self.psi),
            self.V_h * math.cos(self.psi),
            self.vz
        ])
        
        # Add wind if available
        v_enu = v_air + self.cfg.wind_enu
        
        # Update position
        self.pos += v_enu * dt
        
        # Keep above ground
        if self.pos[2] < 0:
            self.pos[2] = 0
            self.vz = 0
            if self.flight_mode != "ground":
                self.flight_mode = "ground"
        
        # Update energy
        self._update_power()
        self.energy_wh = max(self.energy_wh - self.P_elec * dt / 3600.0, 0.0)
        
        # Motor-use accumulator per FEnergy (count motor only when not soaring)
        motor_on = (self.soaring_state not in ("thermal_exploitation", "gliding"))
        if motor_on:
            self.cmotor_J += float(self.P_elec) * dt  # J = W * s
    
    def _update_power(self):
        """Power with avionics assumed zero."""
        # Motor off when gliding, period.
        if self.soaring_state == "gliding":
            self.throttle = 0.0
            self.P_elec = 0.0
            return

        # Motor off only once actually INSIDE the thermal column
        if self.soaring_state == "thermal_exploitation" and self._inside_thermal:
            self.throttle = 0.0
            self.P_elec = 0.0
            return

        # Not moving → no draw
        if self.V_h < 0.1:
            self.throttle = 0.0
            self.P_elec = 0.0
            return

        # Aerodynamic/mech power → electrical via prop efficiency
        rho = self.cfg.air_density
        S   = self.cfg.wing_area
        CD0 = self.cfg.CD0
        k   = self.cfg.induced_drag_k
        W   = self.cfg.mass * 9.81
        eta = max(self.cfg.prop_eta, 1e-3)

        V = max(self.V_h, 0.1)
        q = 0.5 * rho * V**2
        D_par = q * S * CD0
        D_ind = k * W**2 / (q * S)
        P_aero = (D_par + D_ind) * V
        self.P_elec = P_aero / eta
    
    def _update_flight_mode(self):
        """Update flight mode state machine"""
        if self.flight_mode == "takeoff":
            self.vz = self.cfg.climb_rate
            if self.pos[2] >= 0.95 * self.target_alt:
                self.flight_mode = "mission"
                self.eagle_logger.info(f"{self.uav_id} reached mission altitude")
        
        elif self.flight_mode == "mission":
            # Navigate waypoints or investigate events
            if self.event_mode == "investigating":
                self._handle_event_investigation()
            elif self.soaring_state == "thermal_exploitation":
                self._handle_thermal_exploitation()
            else:
                self._update_waypoint_navigation()
        
        elif self.flight_mode == "landing":
            self.vz = self.cfg.landing_descent_rate
            if self.pos[2] <= 0.05:
                self.flight_mode = "ground"
                self.V_h = 0.0
                self.vz = 0.0
                self.eagle_logger.info(f"{self.uav_id} landed")
    
    def _update_waypoint_navigation(self):
        """Basic waypoint navigation"""
        if not self.waypoints or self.current_wp_index < 0:
            return
        
        if self.current_wp_index >= len(self.waypoints):
            self.land()
            return
        
        # Navigate to current waypoint
        wp = self.waypoints[self.current_wp_index]
        vec = wp[:2] - self.pos[:2]
        dist = np.linalg.norm(vec)
        
        if dist < self.cfg.waypoint_tol:
            self.current_wp_index += 1
            self.eagle_logger.info(f"{self.uav_id} reached waypoint {self.current_wp_index}")
        else:
            # Set heading toward waypoint
            self.target_heading = math.atan2(vec[0], vec[1])
            self.target_alt = wp[2]
            self.target_speed = self.cfg.cruise_speed
            
            # Simple control
            self.psi = self.target_heading
            self.V_h = self.target_speed
            self.vz = np.clip(self.target_alt - self.pos[2], 
                            self.cfg.descent_rate, self.cfg.climb_rate)
    
    def _handle_event_investigation(self):
        """Handle event investigation loitering"""
        if self.event_center is None:
            self.event_mode = "idle"
            return
        
        # Simple circular loiter around event
        vec = self.event_center - self.pos[:2]
        dist = np.linalg.norm(vec)
        
        # Maintain loiter radius
        if dist > self.event_radius:
            self.target_heading = math.atan2(vec[0], vec[1])
        else:
            # Circle around
            tangent = np.array([-vec[1], vec[0]])
            n = np.linalg.norm(tangent)
            if n > 1e-6:
                tangent = tangent / n
                self.target_heading = math.atan2(tangent[0], tangent[1])
            else:
                # gentle turn if we're basically centered
                self.target_heading = self.psi + 0.1
        
        self.psi = self.target_heading
        self.V_h = self.cfg.V_loiter
        self.vz = 0.0
    
    def _handle_thermal_exploitation(self):
        """Handle thermal soaring"""
        if self.current_thermal is None:
            self.soaring_state = "normal"
            self._inside_thermal = False
            return

        th = self.current_thermal
        elapsed = self.t - self.exploitation_start_time
        if elapsed > self.cfg.max_thermal_dwell_s or not th.active(self.t):
            self.soaring_state = "normal"
            self.current_thermal = None
            self._thermal_gain_booked = False
            self._inside_thermal = False
            return

        # Are we inside the usable thermal column & altitude?
        dist_xy = np.linalg.norm(self.pos[:2] - np.array(th.center))
        alt_ok = (self.pos[2] >= getattr(th, "base_height", 0.0)) and (self.pos[2] <= th.top_height)
        self._inside_thermal = (dist_xy <= th.radius) and alt_ok

        if not self._inside_thermal:
            # Navigate to thermal center with motor ON
            self._goto(th.center[0], th.center[1])
            self.target_speed = max(self.cfg.min_speed, self.cfg.V_cmd_min)
            # climb toward band if below base
            if self.pos[2] < getattr(th, "base_height", 0.0):
                self._climb(min(self.cfg.altitude_max_soar_m, getattr(th, "base_height", 0.0)))
            self._fly_towards_target()
            return

        # Inside column → spiral with motor OFF (handled by _update_power via flag)
        angle = elapsed * 0.1  # rad/s
        radius = 0.8 * th.radius
        target = np.array([
            th.center[0] + radius * math.cos(angle),
            th.center[1] + radius * math.sin(angle)
        ])
        vec = target - self.pos[:2]
        if np.linalg.norm(vec) > 1.0:
            self.target_heading = math.atan2(vec[0], vec[1])
            self.psi = self.target_heading

        self.V_h = self.cfg.min_speed
        # altitude managed by vertical updraft + small trims
        self.vz = 0.0
    
    def _start_event_investigation(self, cx: float, cy: float):
        """Start investigating an event"""
        self.event_mode = "investigating"
        self.event_center = np.array([cx, cy])
        self.eagle_logger.info(f"{self.uav_id} starting event investigation at ({cx}, {cy})")
    
    def _initiate_thermal_exploitation(self, thermal: Thermal, t: float):
        """Start exploiting a thermal"""
        self.current_thermal = thermal
        self.soaring_state = "thermal_exploitation"
        self.exploitation_start_time = t
        self._inside_thermal = False  # will be set true once actually inside
        self.eagle_logger.info(f"{self.uav_id} starting thermal exploitation")

        # Head toward center; power stays on until _inside_thermal
        self._goto(thermal.center[0], thermal.center[1])
        self.target_speed = max(self.cfg.min_speed, self.cfg.V_cmd_min)

        # Book the benefit once at start, per B_i(t)
        if not self._thermal_gain_booked:
            b = max(0.0, float(self.calculate_thermal_benefit(thermal)))
            self.egain_J += b
            self._thermal_gain_booked = True
    
    def get_tier(self) -> AgentTier:
        """Determine current agent tier based on battery and state"""
        battery_pct = self.battery_pct()
        
        if battery_pct < 50.0:
            return AgentTier.LOW_BATTERY
        elif self.soaring_state == "thermal_exploitation":
            return AgentTier.HEALTHY_SOARING
        else:
            return AgentTier.HEALTHY_NOMAD
    
    def _specific_energy_per_meter(self, V_air: float, wind_xy: np.ndarray, u_hat: np.ndarray|None=None) -> float:
        """
        J/m along-ground at given airspeed. Uses drag polar and prop efficiency.
        Assumes coordinated level flight (phi=0). For turns, scale induced term by n^2.
        
        Args:
            V_air: Airspeed in m/s
            wind_xy: Wind vector [vx, vy] in m/s
            u_hat: Unit vector in leg direction [dx, dy]. If None, uses current velocity direction.
        """
        rho = self.cfg.air_density
        S   = self.cfg.wing_area
        CD0 = self.cfg.CD0
        k   = self.cfg.induced_drag_k
        W   = self.cfg.mass * 9.81
        eta = self.cfg.prop_eta

        q = 0.5 * rho * V_air**2
        D_par = q * S * CD0
        D_ind = k * W**2 / (q * S)  # level flight
        P = (D_par + D_ind) * V_air / max(eta, 1e-3)
        
        if u_hat is None:
            # fall back to current track
            track = self.vel[:2]; track = track/(np.linalg.norm(track)+1e-6)
        else:
            track = u_hat/(np.linalg.norm(u_hat)+1e-6)
        
        V_gnd = max(1.0, V_air + float(np.dot(wind_xy, track)))
        return P / V_gnd

    def calculate_investigation_cost(self, event: GroundEvent) -> float:
        """
        J: transit to event + one stabilized orbit at (V_loiter, phi_loiter)
        """
        p0 = self.pos.copy()
        pe = np.array([event.cx, event.cy, p0[2]])
        d_xy = np.linalg.norm(p0[:2] - pe[:2])

        V_cruise = self.cfg.V_range_opt
        # Calculate leg direction vector from current position to event
        leg_vector = pe[:2] - p0[:2]
        spec_e = self._specific_energy_per_meter(V_cruise, self.wind_xy, leg_vector)
        E_transit = spec_e * d_xy

        phi = self.cfg.loiter_bank_deg * np.pi/180.0
        n = 1.0 / max(np.cos(phi), 0.2)
        V_loiter = self.cfg.V_loiter
        # turn power with n^2 scaling on induced
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

        return E_transit + E_loiter
    
    def calculate_thermal_benefit(self, thermal_or_msg) -> float:
        p = self.pos.copy()
        
        # Handle both Thermal objects and message dictionaries
        if isinstance(thermal_or_msg, dict):
            # Message dictionary case
            center = np.array(thermal_or_msg["position"])
            radius = thermal_or_msg["radius"]
            strength = thermal_or_msg["strength"]
            top_height = thermal_or_msg.get("top_height", p[2] + 10.0)
            base_height = thermal_or_msg.get("base_height", 0.0)
            spawn_time = thermal_or_msg.get("spawn_time", -1e12)
            end_time = thermal_or_msg.get("end_time", 1e12)
            
            # Check if thermal is still active
            if not (spawn_time <= self.tnow < end_time):
                return -np.inf
                
            # Calculate time remaining
            time_remaining = end_time - self.tnow
        else:
            # Thermal object case
            center = np.array(thermal_or_msg.center[:2])
            radius = thermal_or_msg.radius
            strength = thermal_or_msg.strength
            top_height = thermal_or_msg.top_height
            base_height = getattr(thermal_or_msg, "base_height", 0.0)
            time_remaining = thermal_or_msg.time_remaining(self.tnow)
        
        r_xy = np.linalg.norm(p[:2] - center)
        # time to enter disk, assume straight-line at V_cruise
        t_entry = r_xy / max(self.ground_speed(self.cfg.V_range_opt), 1.0)

        # estimate average climb within usable core
        w_bar = self._expected_wair(thermal_or_msg, p[:2])  # works for Thermal or msg dict
        if w_bar <= 0.2:  # below threshold
            return -np.inf

        dwell = min(self.cfg.max_thermal_dwell_s, time_remaining - t_entry)
        if dwell <= 0:
            return -np.inf

        if self.thermal_policy is ThermalPolicy.STRICT_ALT:
            # motor-off savings while dwelling (assume throttle=0 except trim)
            P_base = self._loiter_power_in_still_air()  # same phi/V as above
            E_saved = P_base * dwell * self.cfg.motor_cut_fraction
            E_gain  = 0.0   # no altitude gain in STRICT_ALT
        else:
            # climb within band
            hmax = min(self.alt_band[1], top_height)
            dh   = max(0.0, hmax - p[2])
            t_climb = min(dwell, dh / w_bar)
            E_saved = self._loiter_power_in_still_air() * t_climb * self.cfg.motor_cut_fraction
            E_gain  = self.cfg.mass * 9.81 * (w_bar * t_climb)  # potential energy gained

        # detour is incremental path-length vs planned next waypoint
        d_detour = self._incremental_detour_length(center[:2])
        # Calculate leg direction vector from current position to thermal center
        leg_vector = center[:2] - p[:2]
        E_detour = self._specific_energy_per_meter(self.cfg.V_range_opt, self.wind_xy, leg_vector) * d_detour

        C = self._thermal_confidence(thermal_or_msg)  # [0,1] from tracker/variometer SNR
        return C * (E_gain + E_saved) - E_detour
    
    def _expected_wair_from_metrics(self, th_msg: dict, x: float, y: float, z: float, t: float) -> float:
        """
        Compute vertical airspeed (m/s) from a thermal message only.
        No direct calls to world or Thermal object.
        Fields expected in th_msg:
          - "position": [cx, cy]
          - "radius": float
          - "strength": float            # max updraft at core (m/s)
          - "base_height": float         # OPTIONAL (default 0)
          - "top_height": float
          - "spawn_time": float          # OPTIONAL (defaults to -inf)
          - "end_time": float            # OPTIONAL (defaults to +inf)
        """
        cx, cy = th_msg["position"][0], th_msg["position"][1]
        radius = float(th_msg["radius"])
        strength = float(th_msg["strength"])
        z_base = float(th_msg.get("base_height", 0.0))
        z_top  = float(th_msg.get("top_height", z_base + 10.0))

        t0 = float(th_msg.get("spawn_time", -1e12))
        t1 = float(th_msg.get("end_time",  1e12))

        # time active?
        if not (t0 <= t < t1):
            return 0.0

        # altitude inside band?
        if z < z_base or z > z_top:
            return 0.0

        # horizontal distance
        dist = math.hypot(x - cx, y - cy)
        if dist > radius:
            return 0.0

        # same shape as your thermals.py: linear radial decay
        radial = max(0.0, 1.0 - dist / max(radius, 1e-6))

        # height envelope (bell/triangle peaking ~70% up the column)
        span = max(1e-6, z_top - z_base)
        hz = (z - z_base) / span
        height = hz/0.7 if hz <= 0.7 else (1.0 - hz) / 0.3
        height = max(0.0, min(1.0, height))

        return strength * radial * height

    def _expected_wair(self, thermal_or_msg, pos_xy: np.ndarray) -> float:
        """
        If we got a live Thermal object → use its w((x,y,z), t).
        If we only have a broadcast message dict → use metrics-only helper.
        """
        x, y, z = pos_xy[0], pos_xy[1], self.pos[2]

        # Case A: real Thermal object
        if hasattr(thermal_or_msg, "w"):
            return max(0.0, float(thermal_or_msg.w((x, y, z), self.tnow)))

        # Case B: message dict from comms
        if isinstance(thermal_or_msg, dict):
            return max(0.0, float(self._expected_wair_from_metrics(thermal_or_msg, x, y, z, self.tnow)))

        # Unknown type
        return 0.0
    
    def _loiter_power_in_still_air(self) -> float:
        """
        Power required for loiter in still air at current bank angle
        """
        phi = self.cfg.loiter_bank_deg * np.pi/180.0
        n = 1.0 / max(np.cos(phi), 0.2)
        V_loiter = self.cfg.V_loiter
        
        # Calculate power with n^2 scaling on induced drag
        rho, S, CD0, k, W, eta = (self.cfg.air_density, self.cfg.wing_area, 
                                  self.cfg.CD0, self.cfg.induced_drag_k, 
                                  self.cfg.mass*9.81, self.cfg.prop_eta)
        q = 0.5 * rho * V_loiter**2
        D_par = q * S * CD0
        D_ind = n**2 * k * W**2 / (q * S)
        P_turn = (D_par + D_ind) * V_loiter / max(eta, 1e-3)
        
        return P_turn
    
    def _incremental_detour_length(self, thermal_pos_xy: np.ndarray) -> float:
        """
        Calculate incremental path length to thermal vs planned route
        """
        # If no planned route, use direct distance
        if not hasattr(self, 'planned_route') or not self.planned_route:
            return np.linalg.norm(self.pos[:2] - thermal_pos_xy)
        
        # Find closest point on planned route
        min_dist = float('inf')
        closest_point = None
        
        for i, waypoint in enumerate(self.planned_route):
            dist = np.linalg.norm(waypoint[:2] - thermal_pos_xy)
            if dist < min_dist:
                min_dist = dist
                closest_point = waypoint
        
        if closest_point is None:
            return np.linalg.norm(self.pos[:2] - thermal_pos_xy)
        
        # Calculate detour: current_pos -> thermal -> closest_waypoint
        d1 = np.linalg.norm(self.pos[:2] - thermal_pos_xy)
        d2 = np.linalg.norm(thermal_pos_xy - closest_point[:2])
        
        return d1 + d2
    
    def _thermal_key(self, center, top_height, radius) -> str:
        x, y = np.round(center[0], 1), np.round(center[1], 1)
        th, r = round(top_height, 0), round(radius, 1)
        return f"th_{x}_{y}_{th}_{r}"

    def _thermal_confidence(self, thermal_or_msg) -> float:
        # Example: clamp from tracker variance or variometer SNR
        if isinstance(thermal_or_msg, dict):
            # Message dictionary case - use default confidence
            return 0.6
        else:
            # Thermal object case
            return np.clip(thermal_or_msg.confidence if hasattr(thermal_or_msg, "confidence") else 0.6, 0.0, 1.0)
    
    def _distance_to(self, target_pos) -> float:
        if isinstance(target_pos, (list, tuple, np.ndarray)):
            tp = np.array(target_pos, dtype=float)
            return np.linalg.norm(self.pos[:2] - tp[:2])
        return float('inf')

    def _goto(self, x: float, y: float):
        vec = np.array([x, y]) - self.pos[:2]
        if np.linalg.norm(vec) > 1.0:
            self.target_heading = math.atan2(vec[0], vec[1])
            # heading rate limited in _fly_towards_target

    def _climb(self, target_alt: float):
        alt_diff = target_alt - self.pos[2]
        if abs(alt_diff) > 0.5:
            self.target_alt = np.clip(target_alt, self.alt_band[0], self.cfg.altitude_max_soar_m)

    def _goto_patrol_route(self):
        if self.waypoints and 0 <= self.current_wp_index < len(self.waypoints):
            wp = self.waypoints[self.current_wp_index]
            self._goto(wp[0], wp[1])
            self.target_alt = self.cfg.altitude_ref_m  # stay in patrol layer after soaring

    def _fly_towards_target(self):
        # heading
        if hasattr(self, 'target_heading'):
            err = math.atan2(math.sin(self.target_heading - self.psi), math.cos(self.target_heading - self.psi))
            self.psi += np.clip(self.cfg.k_psi * err, -self.cfg.psi_dot_max, self.cfg.psi_dot_max) * self.dt
        # speed
        if hasattr(self, 'target_speed'):
            self.V_h += np.clip(self.target_speed - self.V_h, -1.0, 1.0) * 0.1 * self.dt
            self.V_h = np.clip(self.V_h, self.cfg.V_cmd_min, self.cfg.V_cmd_max)
        # altitude
        if hasattr(self, 'target_alt'):
            alt_err = self.target_alt - self.pos[2]
            self.vz = np.clip(0.2 * alt_err, self.cfg.descent_rate, self.cfg.climb_rate)

    def _simple_thermal_step(self, sim_time, world=None):
        decision = "patrol"
        self.target_alt = self.cfg.altitude_ref_m

        # Use world if provided, otherwise whatever thermals the agent already sees
        if world is not None:
            thermals = world.get_nearby_thermals(self.pos, self.cfg.thermal_detour_max_km * 1000)
        else:
            thermals = self.current_thermals or []
        best, best_score = None, -1e18

        for th in thermals:
            conf = self._thermal_confidence(th)
            if conf < self.cfg.thermal_conf_min or not th.active(sim_time):
                continue
            benefit_J = self.calculate_thermal_benefit(th)  # J
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
    
    def _eagle_step(self, dt: float, t: float, thermals: List[Thermal], events: List[GroundEvent]):
        """
        Main EAGLE decision-making step implementing Algorithm 1
        """
        # Update position in network
        self.comm.update_agent_position(self.uav_id, self.pos)
        
        # Process incoming messages
        messages = self.comm.get_messages(self.uav_id, t)
        self._process_messages(messages, t)

        # === Check 0: Absolute safety ===
        if self.battery_pct() < 10.0:  # CRITICAL_THRESHOLD
            if self.flight_mode != "landing":
                self.eagle_logger.warning("CRITICAL battery - initiating emergency landing")
                self.land()
            return

        # === Check 0b: Last resort investigation mode ===
        if self.event_mode == "last_resort":
            self._continue_last_resort_mission()
            return

        # === Check 1: High-priority event ===
        if self._check_for_high_priority_events(events, t):
            return

        # === Check 1b: Handle handover requests ===
        if self._handle_handover_requests(t):
            return

        # === Check 2: Energy management ===
        if self.battery_pct() < 30.0:  # LOW_THRESHOLD
            self._handle_proactive_soaring(thermals, t)
            return

        # OPTIONAL cooperative thermal bidding even when battery is fine
        # (harmless if no beneficial thermal is around)
        took_action = self._cooperative_thermal_step(thermals, t)

        # === Fallback: Search probabilistic map when no live opportunities ===
        if not took_action:
            try:
                took_action = self._search_probabilistic_map(t)
            except Exception as e:
                self.eagle_logger.warning(f"Probabilistic map search failed: {e}")

        # === Check 3: Default behaviors ===
        self._announce_discoveries(thermals, events, t)

        # --- add at the very end of _eagle_step(...) ---
        # Resolve event & thermal auctions (tier timing, winners, timeouts)
        self._update_auctions(t)

        # After auctions resolve, if we just lost a thermal, immediately try the next-best
        if getattr(self, "_lost_thermal_recently", False):
            self._lost_thermal_recently = False
            # Use whatever thermals we already have for this tick
            _ = self._handle_live_thermal_opportunities(t)

        # (optional but useful) share state for peers' heuristics
        self.broadcast_state()

    # ================= Probabilistic Map Integration ==================
    def _ensure_probmap_loaded(self) -> bool:
        """Lazy-load probability map, metadata, avg metrics, and land cover raster.
        Returns True if loaded and ready; False otherwise."""
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

            # Load metadata
            with open(meta_path, 'r') as f:
                self._probmap_meta = json.load(f)

            # Load probability map
            import numpy as _np
            self._probmap = _np.load(prob_path)

            # Optional average metrics
            if avg_npz_path and os.path.isfile(avg_npz_path):
                self._avg_metrics = _np.load(avg_npz_path)

            # Land cover raster and grid
            import rasterio as _rio
            self._lc_ds = _rio.open(lc_raster_path)
            from rasterio import Affine as _Affine
            self._grid_affine = _Affine.from_gdal(*self._probmap_meta['grid_affine_transform_gdal'])
            self._grid_crs = str(self._probmap_meta['grid_crs'])

            # Context mappings
            self._season_map = self._probmap_meta['context_mappings']['season']
            self._tod_map = self._probmap_meta['context_mappings']['time_of_day']
            # Ensure LC code keys are int
            lc_map = self._probmap_meta['context_mappings']['land_cover_code_to_index']
            self._lc_code_to_idx = {int(k): v for k, v in (lc_map.items() if isinstance(lc_map, dict) else lc_map)}

            # Build transform from UTM (simulation coords) to grid CRS if needed
            from config import UTM_CRS
            if str(UTM_CRS) != str(self._grid_crs):
                import pyproj
                transformer = pyproj.Transformer.from_crs(UTM_CRS, self._grid_crs, always_xy=True)
                self._to_grid_transform = transformer.transform
            else:
                self._to_grid_transform = None

            # Load regions to support patrol sector filtering (UTM CRS)
            try:
                from events import load_region_catalog
                from config import GEOJSON_PATH, UTM_CRS as _UTM
                self._regions_catalog = load_region_catalog(GEOJSON_PATH, _UTM)
                # cache home region polygon if set
                if self.home_region:
                    target_key = self.home_region.lower()
                    for r in self._regions_catalog:
                        if r.key == target_key:
                            self._home_region_poly = r.geom
                            break
            except Exception:
                self._regions_catalog = None

            self._probmap_loaded = True
            self.eagle_logger.info("Probabilistic map resources loaded")
            return True
        except Exception as e:
            self.eagle_logger.warning(f"Failed to load probability map resources: {e}")
            return False

    def _context_indices_from_time(self, timestamp: float) -> Tuple[Optional[int], Optional[int]]:
        """Derive season and time-of-day indices from current sim time (seconds)."""
        # Map sim-time seconds to a nominal date/time cycle; we approximate using a fixed calendar
        # Assume t=0 corresponds to June 1st 12:00 for categorization stability
        import datetime as _dt
        base = _dt.datetime(2025, 6, 1, 12, 0, 0)
        current = base + _dt.timedelta(seconds=float(timestamp))
        # Build reverse maps
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
        """Filter candidates to those in home sector; allow adjacent sectors if near border."""
        if not self._regions_catalog or not self._home_region_poly:
            # No region data; allow all
            return True
        try:
            from shapely.geometry import Point as _Point
            p_uav = _Point(x_utm, y_utm)
            p_th = _Point(thermal_xy_utm[0], thermal_xy_utm[1])
            if self._home_region_poly.contains(p_th):
                return True
            # Border zone: within 200 m of boundary
            border_buf_m = 200.0
            if self._home_region_poly.boundary.buffer(border_buf_m).contains(p_uav):
                # Check adjacency: if thermal lies in any touching region
                for r in self._regions_catalog:
                    if r.geom is self._home_region_poly:
                        continue
                    if r.geom.touches(self._home_region_poly) and r.geom.contains(p_th):
                        return True
            return False
        except Exception:
            return True

    def _search_probabilistic_map(self, t: float) -> bool:
        """Implements SearchProbabilisticMap: query precomputed P-map, filter by sector, pick best, and initiate soaring.
        Returns True if an action was taken (navigation/bid/exploitation), False otherwise."""
        if not self._ensure_probmap_loaded():
            return False

        # Determine context
        season_idx, tod_idx = self._context_indices_from_time(t)
        if season_idx is None or tod_idx is None:
            return False

        import numpy as _np
        import rasterio as _rio
        from rasterio.transform import rowcol as _rowcol

        # Define AOI in grid coordinates
        aoi_half = float(getattr(self.cfg, 'probmap_aoi_half_width_m', 500.0))
        # Convert agent UTM position into grid CRS if necessary
        x, y = float(self.pos[0]), float(self.pos[1])
        if self._to_grid_transform:
            gx, gy = self._to_grid_transform(x, y)
        else:
            gx, gy = x, y

        # AOI bounds in meters (grid CRS)
        min_x = gx - aoi_half; max_x = gx + aoi_half
        min_y = gy - aoi_half; max_y = gy + aoi_half

        # Grid index bounds
        try:
            r_min, c_min = _rio.transform.rowcol(self._grid_affine, min_x, max_y)
            r_max, c_max = _rio.transform.rowcol(self._grid_affine, max_x, min_y)
        except Exception:
            return False

        # Clamp
        grid_h = int(self._probmap_meta['grid_dimensions']['height'])
        grid_w = int(self._probmap_meta['grid_dimensions']['width'])
        r_min = max(0, r_min); c_min = max(0, c_min)
        r_max = min(grid_h - 1, r_max); c_max = min(grid_w - 1, c_max)
        if r_min > r_max or c_min > c_max:
            return False

        threshold = float(getattr(self.cfg, 'probmap_probability_threshold', 0.5))

        candidates: List[Tuple[Tuple[int,int], float, Tuple[float,float]]] = []  # ((r,c), prob, (x_m,y_m) in grid CRS)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                # Cell center
                cx_m, cy_m = self._grid_affine * (c + 0.5, r + 0.5)
                # Land cover at center
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
                    p = float(self._probmap[r, c, season_idx, tod_idx, lc_idx])
                except Exception:
                    p = 0.0
                if p >= threshold:
                    candidates.append(((r, c), p, (cx_m, cy_m)))

        if not candidates:
            return False

        # Filter by patrol sector / adjacent sectors
        # Convert candidate centers from grid CRS back to UTM for region tests
        to_utm = None
        if self._to_grid_transform is not None:
            # we have UTM->grid; build inverse for grid->UTM
            try:
                import pyproj
                from config import UTM_CRS
                inv_transformer = pyproj.Transformer.from_crs(self._grid_crs, UTM_CRS, always_xy=True)
                to_utm = inv_transformer.transform
            except Exception:
                to_utm = None
        filtered: List[Tuple[Tuple[int,int], float, Tuple[float,float], Tuple[float,float]]] = []
        for (r, c), prob, (cx_m, cy_m) in candidates:
            if to_utm:
                ux, uy = to_utm(cx_m, cy_m)
            else:
                ux, uy = cx_m, cy_m
            if self._is_inside_patrol_or_border(self.pos[0], self.pos[1], (ux, uy)):
                filtered.append(((r, c), prob, (cx_m, cy_m), (ux, uy)))

        if not filtered:
            return False

        # Rank by net benefit using a lightweight Thermal proxy with average metrics if available
        best = None
        best_score = -1e18
        for (_idx, prob, (_gx, _gy), (ux, uy)) in filtered:
            # Build a Thermal object using avg metrics if available; else default Medium
            th = Thermal(center=(ux, uy), condition='Medium', t0=t)
            # Override with averages if provided
            try:
                if self._avg_metrics is not None:
                    season_i, tod_i = season_idx, tod_idx
                    # landcover index for the chosen cell (recompute for UTM coords)
                    lc_code = 0
                    try:
                        lx, ly = (_gx, _gy)
                        lc_code = int(next(self._lc_ds.sample([(lx, ly)], indexes=1))[0])
                        if lc_code == self._lc_ds.nodata:
                            lc_code = 0
                    except Exception:
                        lc_code = 0
                    lc_i = self._lc_code_to_idx.get(int(lc_code))
                    if lc_i is not None:
                        idx = (int(_idx[0]), int(_idx[1]), season_i, tod_i, lc_i)
                        # Use safe fetch with fallback
                        def _safe(arr, i):
                            try:
                                v = float(arr[i])
                                return v
                            except Exception:
                                return None
                        avg_lift = _safe(self._avg_metrics['avg_lift'], idx)
                        avg_radius = _safe(self._avg_metrics['avg_radius'], idx)
                        avg_height = _safe(self._avg_metrics['avg_height'], idx)
                        if avg_radius is not None and avg_radius > 1.0:
                            th.radius = avg_radius
                        if avg_lift is not None and avg_lift > 0.0:
                            th.strength = avg_lift
                        if avg_height is not None and avg_height > th.base_height:
                            th.top_height = avg_height
            except Exception:
                pass

            benefit_J = float(self.calculate_thermal_benefit(th))
            # Encourage higher map probability modestly
            score = benefit_J + 0.1 * prob * self.cfg.batt_capacity_wh * 3600.0
            if score > best_score and benefit_J > self.benefit_threshold:
                best = th
                best_score = score

        if best is None:
            return False

        # Feasibility check: reuse calculate_thermal_benefit threshold already applied
        # Execute soaring attempt: navigate toward center and start exploitation on arrival
        self._goto(best.center[0], best.center[1])
        self.target_speed = max(self.cfg.min_speed, self.cfg.V_cmd_min)
        # Start exploitation immediately; motor will cut when inside per _update_power
        self._initiate_thermal_exploitation(best, t)
        return True
    
    def _cooperative_thermal_step(self, thermals: List[Thermal], t: float) -> bool:
        """
        Always-on cooperative selection: rank + bid (Algorithm 8 core),
        leaving sector/border checks for later (geometry-dependent).
        """
        self.current_thermals = thermals or []
        return self._handle_live_thermal_opportunities(t)

    def _handle_proactive_soaring(self, thermals: List[Thermal], t: float):
        """
        Algorithm 7: when low on battery, proactively seek thermals.
        Uses the same ranked-iterative selection (Alg. 8 core).
        """
        self.current_thermals = thermals or []
        took_action = self._handle_live_thermal_opportunities(t)

        # Optional: if nothing is worth bidding on, keep existing simple fallback
        if not took_action:
            # Minimal "keep moving toward best" behavior without bidding
            _ = self._simple_thermal_step(t)  # no world parameter needed
    
    def _rank_live_thermal_opportunities(self, t: float):
        """
        Rank both auctions and world thermals by net benefit.
        Returns list of tuples: (thermal_obj, benefit_J, source) where source in {"auction","world"}.
        """
        ranked = []
        seen = set()

        # Active auctions we know about
        for th_id, A in self.active_thermal_auctions.items():
            th = A.thermal
            if hasattr(th, "active") and not th.active(t):
                continue
            b = float(self.calculate_thermal_benefit(th))
            ranked.append((th, b, "auction"))
            seen.add(th_id)

        # World thermals not yet announced
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
        """
        Implements the iterative selection loop of Algorithm 8:
          - rank opportunities
          - announce world thermals (if needed)
          - bid on the best above threshold that we haven't already bid on
        NOTE: sector/border gating intentionally omitted (depends on ops geometry).
        Returns True if we bid or started exploitation.
        """
        candidates = self._rank_live_thermal_opportunities(t)

        for th, benefit, src in candidates:
            if benefit <= self.benefit_threshold:
                break  # all remaining will be worse

            th_key = self._thermal_key(th.center, th.top_height, th.radius)
            if th_key in self.my_thermal_bids:
                continue  # already competing for this one

            # If it's a world thermal, announce it once so others can compete fairly
            if src == "world" and th_key not in self.active_thermal_auctions:
                self.comm.broadcast(
                    self.uav_id,
                    MessageType.THERMAL_DISCOVERED,
                    {
                        "thermal_id": th_key,
                        "position": list(th.center),
                        "strength": th.strength,
                        "radius": th.radius,
                        "base_height": getattr(th, "base_height", 0.0),
                        "top_height": th.top_height,
                        "spawn_time": getattr(th, "spawn_time", self.tnow),
                        "end_time":   getattr(th, "end_time",   self.tnow + 300.0),
                        "benefit_score": float(benefit),
                    }
                )
                self.active_thermal_auctions[th_key] = ThermalAuction(
                    thermal_id=th_key,
                    thermal=th,
                    initiator_id=self.uav_id,
                    start_time=self.tnow,
                    bids=[]
                )

            # Bid on it (Alg. 10)
            self._compete_for_thermal(th, benefit)
            return True

        return False
    
    def _announce_discoveries(self, thermals: List[Thermal], events: List[GroundEvent], t: float):
        """Announce newly discovered thermals and events"""
        # Announce new thermals
        for thermal in thermals:
            th_key = self._thermal_key(thermal.center, thermal.top_height, thermal.radius)
            
            if th_key not in self.detected_thermal_ids:
                self.detected_thermal_ids.add(th_key)
                
                # Check if worth announcing (positive benefit)
                benefit = self.calculate_thermal_benefit(thermal)
                if benefit > 0:
                    self.comm.broadcast(
                        self.uav_id,
                        MessageType.THERMAL_DISCOVERED,
                        {
                            "thermal_id": th_key,
                            "position": list(thermal.center),  # fix tuple→list
                            "strength": thermal.strength,
                            "radius": thermal.radius,
                            "base_height": getattr(thermal, "base_height", 0.0),
                            "top_height": thermal.top_height,
                            "spawn_time": getattr(thermal, "spawn_time", self.tnow),
                            "end_time":   getattr(thermal, "end_time",   self.tnow + 300.0),
                            "benefit_score": float(benefit)
                        }
                    )
                    self.eagle_logger.info(f"Announced thermal {th_key} with benefit={benefit:.0f}J")
    
    def _claim_and_investigate(self, event: GroundEvent):
        """Claim an event and start investigation"""
        self.assigned_event = event
        self.investigated_event_ids.add(event.id)
        
        # Broadcast claim
        self.comm.broadcast(
            self.uav_id,
            MessageType.TASK_CLAIMED,
            {
                "event_id": event.id,
                "winner_id": self.uav_id
            }
        )
        
        # Start investigation
        self._start_event_investigation(event.cx, event.cy)
    
    def _update_auctions(self, t: float):
        # EVENTS
        for event_id, A in list(self.active_event_auctions.items()):
            if A.resolved:
                continue
            # settle tier by tier
            elapsed = t - A.start_time
            tier_idx = min(int(elapsed // self.tier_timeout), len(AgentTier)-1)
            # when we *enter* a new tier, try to settle previous tier
            if tier_idx > A.current_tier or elapsed >= self.auction_timeout:
                prev_tier = list(AgentTier)[A.current_tier].value
                bids = A.tier_bids.get(prev_tier, [])
                if bids:
                    winner_id, min_cost = min(bids, key=lambda x: (x[1], x[0]))
                    A.winner_id, A.resolved = winner_id, True
                    # award and notify
                    self.comm.broadcast(self.uav_id, MessageType.TASK_CLAIMED,
                                        {"event_id": event_id, "winner_id": winner_id})
                    if winner_id == self.uav_id:
                        self._claim_and_investigate(A.event)
                A.current_tier = tier_idx
            # hard timeout → last resort for initiator
            if not A.resolved and elapsed >= self.auction_timeout and A.initiator_id == self.uav_id:
                # Last-resort path per Algorithm 6
                self.assigned_event = A.event
                self.event_mode = "last_resort"
                self._last_resort_start = t

                # Notify peers
                self.comm.broadcast(
                    self.uav_id,
                    MessageType.INVESTIGATING_LAST_RESORT,
                    {"event_id": A.event.id, "position": [A.event.cx, A.event.cy]}
                )

                # Immediately start investigation (loiter logic will run in the mode handler)
                self._start_event_investigation(A.event.cx, A.event.cy)

                A.resolved = True
            if A.resolved:
                del self.active_event_auctions[event_id]
                self.my_event_bids.discard(event_id)

        # THERMALS
        for th_id, T in list(self.active_thermal_auctions.items()):
            if T.resolved:
                continue
            elapsed = t - T.start_time
            if elapsed >= self.tier_timeout:
                if T.bids:
                    winner_id, best = max(T.bids, key=lambda x: (x[1], x[0]))
                    T.winner_id, T.resolved = winner_id, True
                    self.comm.broadcast(self.uav_id, MessageType.THERMAL_CLAIMED,
                                        {"thermal_id": th_id, "winner_id": winner_id})
                    if winner_id == self.uav_id:
                        self._initiate_thermal_exploitation(T.thermal, t)
                    else:
                        # mark retry opportunity after loss
                        self._lost_thermal_recently = True
                else:
                    T.resolved = True

            if T.resolved:
                del self.active_thermal_auctions[th_id]
                self.my_thermal_bids.discard(th_id)
    
    def _compete_for_thermal(self, thermal: Thermal, benefit_score: float):
        """Compete for a thermal through bidding (Algorithm 10)"""
        th_key = self._thermal_key(thermal.center, thermal.top_height, thermal.radius)
        
        if benefit_score <= 0:
            return
        
        # Submit bid
        self.comm.broadcast(
            self.uav_id,
            MessageType.LIVE_THERMAL_BID,
            {
                "thermal_id": th_key,
                "benefit_score": benefit_score,
                "agent_id": self.uav_id
            }
        )
        
        self.my_thermal_bids.add(th_key)
        self.eagle_logger.info(f"Bid on thermal {th_key} with benefit={benefit_score:.0f}J")
    
    def _handle_handover_request_msg(self, msg: Message, t: float):
        """Handle incoming handover request message"""
        event_id = msg.data["event_id"]
        
        if event_id not in self.active_event_auctions:
            # Create new auction entry
            # Note: We don't have the full event object, so create a minimal one
            event = GroundEvent(
                cx=msg.data["position"][0],
                cy=msg.data["position"][1],
                level=msg.data["event_level"]
            )
            event.id = event_id
            
            auction = EventAuction(
                event_id=event_id,
                event=event,
                initiator_id=msg.sender_id,
                start_time=msg.data["detection_time"],
                tier_bids={tier.value: [] for tier in AgentTier}
            )
            self.active_event_auctions[event_id] = auction
    
    def _handle_event_bid_msg(self, msg: Message):
        """Handle incoming event bid"""
        event_id = msg.data["event_id"]
        
        if event_id in self.active_event_auctions:
            auction = self.active_event_auctions[event_id]
            tier = msg.data["tier"]
            bid_cost = msg.data["bid_cost"]
            agent_id = msg.data["agent_id"]
            
            auction.tier_bids[tier].append((agent_id, bid_cost))
    
    def _handle_task_claimed_msg(self, msg: Message):
        """Handle task claimed message"""
        event_id = msg.data["event_id"]
        winner_id = msg.data["winner_id"]
        
        # Mark auction as resolved
        if event_id in self.active_event_auctions:
            self.active_event_auctions[event_id].resolved = True
            self.active_event_auctions[event_id].winner_id = winner_id
        
        # Remove from our bids
        self.my_event_bids.discard(event_id)
    
    def _handle_thermal_discovered_msg(self, msg: Message):
        """Handle thermal discovery announcement"""
        # prefer the sender's key if present to avoid rounding mismatches
        th_key = msg.data.get("thermal_id") or self._thermal_key(
            msg.data["position"], msg.data["top_height"], msg.data["radius"]
        )
        if th_key in self.active_thermal_auctions:
            return
        
        # Create thermal object from message
        thermal = Thermal(
            center=(msg.data["position"][0], msg.data["position"][1]),
            condition='Medium',  # Default condition
            t0=msg.timestamp
        )
        # Override the sampled values with received data
        thermal.strength = msg.data["strength"]
        thermal.radius = msg.data["radius"]
        thermal.top_height = msg.data["top_height"]
        
        # Handle optional fields
        if "base_height" in msg.data:
            thermal.base_height = msg.data["base_height"]
        if "spawn_time" in msg.data:
            thermal.spawn_time = float(msg.data["spawn_time"])
        if "end_time" in msg.data:
            # set lifetime instead of end_time
            thermal.lifetime = max(1.0, float(msg.data["end_time"]) - thermal.spawn_time)
        
        auction = ThermalAuction(
            thermal_id=th_key,
            thermal=thermal,
            initiator_id=msg.sender_id,
            start_time=msg.timestamp,
            bids=[]
        )
        self.active_thermal_auctions[th_key] = auction
    
    def _handle_thermal_bid_msg(self, msg: Message):
        """Handle thermal bid"""
        thermal_id = msg.data["thermal_id"]
        
        if thermal_id in self.active_thermal_auctions:
            auction = self.active_thermal_auctions[thermal_id]
            agent_id = msg.data["agent_id"]
            benefit = msg.data["benefit_score"]
            
            # Deduplicate bids per agent - keep the max benefit
            bids = {a: b for a, b in auction.bids}
            prev = bids.get(agent_id, -1e18)
            if benefit > prev:
                bids[agent_id] = benefit
                auction.bids = list(bids.items())
    
    def _handle_thermal_claimed_msg(self, msg: Message):
        """Handle thermal claimed message"""
        thermal_id = msg.data["thermal_id"]
        winner_id = msg.data["winner_id"]
        
        if thermal_id in self.active_thermal_auctions:
            self.active_thermal_auctions[thermal_id].resolved = True
            self.active_thermal_auctions[thermal_id].winner_id = winner_id
        
        self.my_thermal_bids.discard(thermal_id)
    
    def _update_peer_state(self, msg: Message):
        """Update peer state from heartbeat"""
        peer_id = msg.sender_id
        self.peer_states[peer_id] = {
            "position": msg.data.get("position"),
            "battery_pct": msg.data.get("battery_pct"),
            "tier": msg.data.get("tier"),
            "mode": msg.data.get("mode"),
            "timestamp": msg.timestamp
        }
    
    def broadcast_state(self):
        """Broadcast own state to peers"""
        self.comm.broadcast(
            self.uav_id,
            MessageType.AGENT_STATE,
            {
                "position": self.pos.tolist(),
                "battery_pct": self.battery_pct(),
                "tier": self.get_tier().value,
                "mode": self.flight_mode,
                "soaring_state": self.soaring_state,
                "event_mode": self.event_mode
            }
        )


# Collision avoidance implementation
class CollisionAvoidance:
    """
    Artificial Potential Field collision avoidance
    Based on Section 3.1 of the EAGLE formulation
    """
    
    def __init__(self,
                 d_safe: float = 100.0,      # Minimum UAV separation
                 d_obs_safe: float = 150.0,  # Minimum obstacle clearance
                 r_safe: float = 50.0,       # Safety radius
                 r_turn: float = 30.0):      # Turn radius
        
        self.d_safe = d_safe
        self.d_obs_safe = d_obs_safe
        self.r_safe = r_safe
        self.r_turn = r_turn
        
        # Activation radius (Eq. 19b)
        self.r_a = math.sqrt((r_safe + r_turn)**2 - r_turn**2)
    
    def calculate_avoidance_velocity(self,
                                   own_pos: np.ndarray,
                                   own_vel: np.ndarray,
                                   goal_vel: np.ndarray,
                                   neighbors: List[Dict[str, np.ndarray]],
                                   obstacles: List[Dict[str, np.ndarray]] = None,
                                   config: PhoenixConfig = None,  # Add config parameter
                                   current_heading: float = 0.0,  # Add heading parameter
                                   dt: float = 1.0) -> np.ndarray:  # Add dt parameter
        """
        Calculate collision-free velocity command using APF
        
        Args:
            own_pos: Own position [x, y, z]
            own_vel: Own velocity [vx, vy, vz]
            goal_vel: Desired velocity vector
            neighbors: List of {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            obstacles: List of {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            config: Configuration object for limits
            current_heading: Current heading angle
            dt: Time step for integration
        
        Returns:
            Commanded velocity vector
        """
        v_cmd = goal_vel.copy()
        
        # sum repulsions
        for neighbor in neighbors:
            v_rep = self._repulsive_velocity(
                own_pos, own_vel,
                neighbor['pos'], neighbor['vel'],
                self.d_safe
            )
            v_cmd += v_rep
        
        # Add repulsive velocities from obstacles
        if obstacles:
            for obstacle in obstacles:
                v_rep = self._repulsive_velocity(
                    own_pos, own_vel,
                    obstacle['pos'], obstacle['vel'],
                    self.d_obs_safe
                )
                v_cmd += v_rep
        
        # Use passed parameters instead of self references
        if config:
            V_max = config.V_cmd_max
            V_min = config.V_cmd_min
            k_psi = config.k_psi
            psi_dot_max = config.psi_dot_max
        else:
            # Defaults
            V_max = 25.0
            V_min = 12.0
            k_psi = 1.0
            psi_dot_max = 0.5
        
        # project to feasible envelope
        if np.linalg.norm(v_cmd) > V_max:
            v_cmd = v_cmd / np.linalg.norm(v_cmd) * V_max
        # map to achievable course-rate
        psi_goal = math.atan2(v_cmd[0], v_cmd[1])  # x,y → atan2(x, y) like the agent
        psi_err  = np.arctan2(np.sin(psi_goal - current_heading), np.cos(psi_goal - current_heading))
        psi_dot  = np.clip(k_psi * psi_err, -psi_dot_max, psi_dot_max)
        # Return desired *velocity* consistent with limited heading rate
        V = np.clip(np.linalg.norm(v_cmd), V_min, V_max)
        psi_next = current_heading + psi_dot * dt
        return np.array([V * math.sin(psi_next), V * math.cos(psi_next), v_cmd[2]])
    
    def _repulsive_velocity(self,
                           p_i: np.ndarray,
                           v_i: np.ndarray,
                           p_j: np.ndarray,
                           v_j: np.ndarray,
                           d_min: float) -> np.ndarray:
        """
        Calculate repulsive velocity from a single neighbor/obstacle
        Implements Equations 19a and 19b from formulation
        """
        # Relative position
        r = p_i - p_j
        d = np.linalg.norm(r)
        
        # Too far to matter
        if d > 2 * d_min:
            return np.zeros(3)
        
        # Relative velocity
        v_rel = v_i - v_j
        
        # Approach speed (positive if approaching)
        if d > 1e-6:
            u_r = -np.dot(v_rel, r/d)
        else:
            u_r = np.linalg.norm(v_rel)
        
        # Only repel if approaching
        if u_r <= 0:
            return np.zeros(3)
        
        # Source flow strength (Eq. 19b)
        Q = 4 * np.pi * self.r_a**2 * u_r
        
        # Repulsive velocity (Eq. 19a)
        if d > 1e-6:
            v_ind = Q * r / (4 * np.pi * d**3)
        else:
            # At singularity, repel radially
            v_ind = np.array([Q/(4*np.pi), 0, 0])
        
        # Cap the APF singularity
        v_cap = 10.0  # m/s maximum repulsive velocity
        v_ind = np.clip(v_ind, -v_cap, v_cap)
        
        return v_ind