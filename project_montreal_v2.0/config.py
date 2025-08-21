

from dataclasses import dataclass, field
import numpy as np
import unicodedata
import math

GEOJSON_PATH = "/home/px4_sitl/ets_work/code/version3/project_montreal_v2.0/files/limites-administratives-agglomeration-nad83.geojson"
UTM_CRS = "EPSG:32618"

# ============================ Shared Utilities ============================
def normalize_region_key(s: str) -> str:
    """
    Convert Montréal borough names to normalized keys for consistent lookups.
    
    Normalizes: case-folding, accent-stripping, whitespace-trimming, and
    replaces fancy quotes with standard apostrophes.
    
    Examples:
        "Le Plateau-Mont-Royal" → "le plateau-mont-royal"
        "Ahuntsic-Cartierville" → "ahuntsic-cartierville"
        "Villeray-Saint-Michel-Parc-Extension" → "villeray-saint-michel-parc-extension"
    """
    s = s.replace("'", "'").replace("'", "'")
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode().lower().strip()

@dataclass
class PhoenixConfig:
    """
    Platform & guidance parameters used by UAVAgent.
    Add the new cruise‑altitude entry so take‑off uses it.
    """
    # ─── flight envelope ─────────────────────────────────────
    cruise_speed        = 18.0      # m/s
    min_speed           = 12.0
    max_speed           = 25.0

    # turn, accel, climb limits …
    turn_rate_max       = math.radians(30)   # rad/s
    max_accel           = 1.5                # m/s²
    climb_rate          = 3.0                # m/s
    descent_rate        = -3.0               # m/s
    landing_descent_rate= -1.0               # m/s

    # power, aero, mass …
    mass: float = 1.50
    wing_area: float = 0.90
    aspect_ratio: float = 14.0
    oswald_e: float = 0.95
    cda0: float = 0.006
    cd0: float = 0.012
    max_roll_deg: float = 45.0
    landing_speed: float = 8.0
    waypoint_tol: float = 100.0  # Increased from 10.0 to 100.0 for large survey areas
    lookahead_dist: float = 120.0
    prop_radius: float = 0.18
    eta_v: float = 0.87
    t_max: float = 10.5
    max_motor_power_w: float = 400.0
    batt_capacity_ah: float = 5.20
    batt_nominal_v: float = 14.8
    p_avionics: float = 3.0
    rho: float = 1.225
    wind_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def batt_capacity_wh(self) -> float:
        return self.batt_capacity_ah * self.batt_nominal_v
    
    # EAGLE-specific parameters
    eagle_tier_timeout_s: float = 5.0
    eagle_auction_timeout_s: float = 20.0
    eagle_benefit_threshold_j: float = 500.0
    
    # Aerodynamic parameters
    air_density: float = 1.225  # kg/m³
    CD0: float = 0.012  # Parasitic drag coefficient
    induced_drag_k: float = 0.04  # Induced drag factor
    prop_eta: float = 0.85  # Propeller efficiency
    
    # Operational parameters
    alt_band_m: float = 100.0  # Altitude band (±m)
    V_range_opt: float = 18.0  # Optimal range speed (m/s)
    V_loiter: float = 15.0  # Loiter speed (m/s)
    loiter_bank_deg: float = 30.0  # Bank angle for loiter
    
    # Thermal parameters
    max_thermal_dwell_s: float = 300.0  # Max time in thermal
    motor_cut_fraction: float = 0.9  # Fraction of power saved when motor off
    
    # Control parameters
    V_cmd_min: float = 12.0
    V_cmd_max: float = 25.0
    k_psi: float = 1.0  # Heading control gain
    psi_dot_max: float = 0.5  # Max heading rate (rad/s)
    
    # Soaring parameters
    altitude_ref_m: float = 400.0  # Reference altitude for soaring (m)
    altitude_max_soar_m: float = 800.0  # max climb when exploiting thermals
    thermal_detour_max_km: float = 2.0  # max extra distance willing to travel
    thermal_conf_min: float = 0.6  # min confidence to engage
    
    # Probability map configuration
    probmap_meta_path: str = "/home/px4_sitl/ets_work/code/version3/project_montreal_v2.0/files/probability_map_metadata.json"
    probmap_prob_path: str = "/home/px4_sitl/ets_work/code/version3/project_montreal_v2.0/files/conditional_probability_map.npy"
    probmap_avg_npz_path: str = "/home/px4_sitl/ets_work/code/version3/project_montreal_v2.0/files/average_thermal_metrics.npz"
    probmap_lc_raster_path: str = "/home/px4_sitl/ets_work/code/version3/project_montreal_v2.0/files/landcover-2020-classification.tif"

    # Runtime knobs
    probmap_aoi_half_width_m: float = 500.0
    probmap_probability_threshold: float = 0.5