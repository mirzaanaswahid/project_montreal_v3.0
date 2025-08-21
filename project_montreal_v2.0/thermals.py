#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thermal simulation for UAV soaring — region-aware & sim-time aligned.

Features
--------
- Thermal.active(t) uses simulation time (no wall-clock).
- spawn_random_thermal(...) supports:
    * legacy rectangular spawning via area_bounds
    * region-aware spawning when 'regions' list is provided
- Separation enforced vs. existing thermals
- Compatible with agent.py (time_remaining(), end_time, attributes)

Usage (region-aware)
--------------------
from events import load_region_catalog
regions = load_region_catalog(GEOJSON_PATH, UTM_CRS)  # or your filtered 4-borough list

th = spawn_random_thermal(
    area_bounds, existing_thermals=thermals, now=sim_time,
    regions=regions, weight_mode="area"  # or "density" or "blend" (with alpha)
)
"""

from __future__ import annotations
from typing import Tuple, Optional, Union, List, Dict, Any
import numpy as np
import math

# ---------------------------- Limits / knobs -----------------------------

MAX_THERMALS = 20
MIN_SEPARATION_MULTIPLIER = 2.0  # multiples of max(existing.radius, candidate.radius_mean)

THERMAL_CONDITIONS = ['Low', 'Medium', 'High']
THERMAL_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # (mean, std) — radius increased to ease detection (your prior choice)
    'radius': {
        'Low':    (150.0, 15.0),
        'Medium': (300.0, 30.0),
        'High':   (450.0, 45.0),
    },
    'max_updraft_velocity': {
        'Low':    (1.5, 0.3),
        'Medium': (3.0, 0.5),
        'High':   (5.0, 0.7),
    },
    'base_altitude': {
        'Low':    (50.0,  5.0),
        'Medium': (100.0, 10.0),
        'High':   (150.0, 15.0),
    },
    'top_altitude': {
        'Low':    (500.0,  50.0),
        'Medium': (1000.0, 100.0),
        'High':   (1500.0, 150.0),
    },
    # Lifetime (seconds)
    'fade_time': {
        'Low':    (300.0,  60.0),   # 5–10 min
        'Medium': (900.0,  180.0),  # 10–18 min
        'High':   (1500.0, 240.0),  # 18–25 min
    },
}

def _sample_pos(mu_sigma: Tuple[float, float], low_clip: float = 1.0) -> float:
    mu, sigma = mu_sigma
    return float(max(low_clip, np.random.normal(mu, sigma)))


# ------------------------------- Model -----------------------------------

class Thermal:
    """
    Thermal model with altitude‑dependent lift (simulation-time based).
    Compatible with agent.py which may call: active(t), time_remaining(t), end_time,
    and read attributes: radius, strength, base_height, top_height, center, condition.
    """
    _next_id = 1

    def __init__(self,
                 center: Tuple[float, float],
                 condition: str = 'Medium',
                 t0: float = 0.0):
        """
        center : (x,y) in meters (UTM)
        condition : 'Low' | 'Medium' | 'High'
        t0 : simulation spawn time [s] (NOT wall-clock)
        """
        assert condition in THERMAL_CONDITIONS, f"Unknown condition {condition}"
        self.id_int = Thermal._next_id
        Thermal._next_id += 1
        
        self.center: Tuple[float, float] = (float(center[0]), float(center[1]))
        self.condition: str = condition
        
        # Sample parameters
        self.radius: float   = _sample_pos(THERMAL_PARAMS['radius'][condition], low_clip=1.0)
        self.strength: float = _sample_pos(THERMAL_PARAMS['max_updraft_velocity'][condition], low_clip=0.0)
        self.base_height: float = _sample_pos(THERMAL_PARAMS['base_altitude'][condition], low_clip=0.0)
        self.top_height: float  = max(self.base_height + 10.0,
                                      _sample_pos(THERMAL_PARAMS['top_altitude'][condition],
                                                  low_clip=self.base_height + 10.0))
        self.lifetime: float    = _sample_pos(THERMAL_PARAMS['fade_time'][condition], low_clip=1.0)
        
        # Simulation time base
        self.spawn_time: float = float(t0)
        self.birth_time: float = self.spawn_time  # legacy alias

        # Optional development phase (not used elsewhere)
        self.development_time: float = 0.2 * self.lifetime

    # ---- agent.py compatibility helpers ----
    @property
    def end_time(self) -> float:
        return self.spawn_time + self.lifetime

    def time_remaining(self, t: float) -> float:
        return self.end_time - t

    # ---------------------- primary API used by agent ----------------------
    def active(self, t: float) -> bool:
        """Active if spawn_time <= t < end_time (simulation time)."""
        return (t >= self.spawn_time) and (t < self.end_time)

    def w(self, pos: Union[Tuple[float, float], Tuple[float, float, float]], t: float) -> float:
        """
        Vertical wind (m/s) at position pos at time t (simulation time).
        Uses linear radial decay and bell-like height envelope; 0 when inactive.
        """
        if not self.active(t):
            return 0.0
        
        x = float(pos[0]); y = float(pos[1])
        z = float(pos[2]) if len(pos) > 2 else 0.0
        
        # altitude window
        if (z < self.base_height) or (z > self.top_height):
            return 0.0
            
        # horizontal distance
        dx = x - self.center[0]
        dy = y - self.center[1]
        r  = math.hypot(dx, dy)
        if r > self.radius:
            return 0.0
        
        # Linear radial decay
        radial_factor = max(0.0, 1.0 - (r / self.radius))
        
        # Height envelope: triangular/bell peaking near ~70% of height span
        span = self.top_height - self.base_height
        if span <= 0.0:
            height_factor = 1.0
        else:
            hz = (z - self.base_height) / span
            height_factor = hz / 0.7 if hz <= 0.7 else (1.0 - hz) / 0.3
            height_factor = max(0.0, min(1.0, height_factor))
        
        return float(self.strength * radial_factor * height_factor)

    def expected_gain_wh(self, current_alt_m: float, mass_kg: float, conf_factor: float = 1.0) -> float:
        """
        Gross potential energy you could gain by climbing from current_alt to top_height.
        Returns Watt-hours (Wh).
        """
        dh = max(0.0, self.top_height - current_alt_m)
        pe_j = mass_kg * 9.81 * dh
        return (pe_j / 3600.0) * conf_factor

    # -------------------------- viz convenience ----------------------------
    def get_visualization_data(self) -> dict:
        return {
            'center': self.center,
            'radius': self.radius,
            'base_height': self.base_height,
            'max_height': self.top_height,
            'strength': self.strength,
            'condition': self.condition,
            'spawn_time': self.spawn_time,
            'end_time': self.end_time,
        }


# ------------------------------ Region utils ------------------------------

def _region_weights(regions: List[Any],
                    mode: str = "area",
                    alpha: float = 0.5) -> np.ndarray:
    """
    Compute selection weights over regions.

    mode:
      - "area": proportional to polygon area
      - "density": proportional to region.density (requires attr)
      - "blend": (1-alpha)*area + alpha*density (both normalized)
    """
    # area in UTM m²
    areas = np.array([float(getattr(r.geom, "area", 0.0)) for r in regions], dtype=float)
    areas = areas / areas.sum() if areas.sum() > 0 else np.ones(len(areas)) / max(1, len(areas))

    if mode == "area":
        return areas

    # density
    dens = np.array([float(getattr(r, "density", 0.0)) for r in regions], dtype=float)
    dens = dens / dens.sum() if dens.sum() > 0 else np.ones(len(dens)) / max(1, len(dens))

    if mode == "density":
        return dens

    # blend
    w = (1.0 - alpha) * areas + alpha * dens
    s = w.sum()
    return w / s if s > 0 else np.ones(len(w)) / max(1, len(w))


def _sample_point_in_polygon(poly, rng: np.random.Generator, max_tries: int = 100) -> Optional[Tuple[float, float]]:
    """
    Uniform rejection sampling inside a polygon bbox.
    Works with both Shapely 1.x and 2.x.
    - If a MultiPolygon sneaks in, pick a component weighted by area.
    - Falls back to representative_point() if rejection sampling fails.
    """
    from shapely.geometry import Point, MultiPolygon

    if poly.is_empty:
        return None

    # If we ever get a MultiPolygon here, choose one part by area
    if isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
        if not polys:
            return None
        areas = np.array([p.area for p in polys], dtype=float)
        probs = areas / areas.sum() if areas.sum() > 0 else np.ones(len(polys)) / len(polys)
        poly = polys[int(rng.choice(len(polys), p=probs))]

    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(float(x), float(y))):
            return float(x), float(y)

    # Fallback: use a guaranteed‑inside point to avoid hard failure
    rp = poly.representative_point()
    return float(rp.x), float(rp.y)


# ------------------------------ Spawners -----------------------------------

def spawn_random_thermal_in_regions(
    regions: List[Any],
    existing_thermals: Optional[List[Thermal]] = None,
    now: Optional[float] = None,
    *,
    weight_mode: str = "area",
    alpha: float = 0.5,
    condition_weights: Tuple[float, float, float] = (0.25, 0.50, 0.25),
    max_attempts: int = 80,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Thermal]:
    """
    Region-aware spawner: pick a region (weighted), sample inside its polygon,
    and enforce separation from existing thermals.

    regions: List of objects with at least .geom (shapely polygon). If they also
             have .density, weight_mode='density' or 'blend' is supported.
    now: simulation time seconds (used as spawn_time)
    """
    if not regions:
        return None

    existing_thermals = existing_thermals or []
    rng = rng or np.random.default_rng()

    weights = _region_weights(regions, mode=weight_mode, alpha=alpha)

    # Precompute mean radius by condition for a conservative separation test
    radius_mean = {
        'Low': THERMAL_PARAMS['radius']['Low'][0],
        'Medium': THERMAL_PARAMS['radius']['Medium'][0],
        'High': THERMAL_PARAMS['radius']['High'][0],
    }
    
    for _ in range(max_attempts):
        # choose region
        idx = int(rng.choice(len(regions), p=weights))
        region = regions[idx]

        # sample a point inside region polygon
        pt = _sample_point_in_polygon(region.geom, rng, max_tries=120)
        if pt is None:
            continue
        x, y = pt

        # draw condition
        condition = str(rng.choice(THERMAL_CONDITIONS, p=condition_weights))

        # coarse separation using candidate mean radius
        cand_r_mean = max(1.0, float(radius_mean[condition]))
        ok = True
        for th in existing_thermals:
            dx = x - th.center[0]
            dy = y - th.center[1]
            d = math.hypot(dx, dy)
            min_sep = MIN_SEPARATION_MULTIPLIER * max(th.radius, cand_r_mean)
            if d < min_sep:
                ok = False
                break
        if not ok:
            continue

        # create thermal with exact sampled parameters; final separation naturally satisfied
        t0 = 0.0 if now is None else float(now)
        return Thermal(center=(x, y), condition=condition, t0=t0)

    return None


def spawn_random_thermal(
    area_bounds: Tuple[float, float, float, float],
    existing_thermals: Optional[List[Thermal]] = None,
    now: Optional[float] = None,
    *,
    regions: Optional[List[Any]] = None,
    weight_mode: str = "area",
    alpha: float = 0.5,
    condition_weights: Tuple[float, float, float] = (0.25, 0.50, 0.25),
    max_attempts: int = 80,
) -> Optional[Thermal]:
    """
    Backward-compatible wrapper.
    - If 'regions' is provided → region-aware spawning (recommended).
    - Else → rectangular spawning inside area_bounds (legacy behavior).
    """
    if regions:
        return spawn_random_thermal_in_regions(
            regions=regions,
            existing_thermals=existing_thermals,
            now=now,
            weight_mode=weight_mode,
            alpha=alpha,
            condition_weights=condition_weights,
            max_attempts=max_attempts,
        )

    # --- legacy rectangular spawning ---
    existing_thermals = existing_thermals or []
    minx, maxx, miny, maxy = area_bounds
    rng = np.random.default_rng()

    # conservative radius by condition for separation check
    radius_mean = {
        'Low': THERMAL_PARAMS['radius']['Low'][0],
        'Medium': THERMAL_PARAMS['radius']['Medium'][0],
        'High': THERMAL_PARAMS['radius']['High'][0],
    }

    for _ in range(max_attempts):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        condition = str(rng.choice(THERMAL_CONDITIONS, p=condition_weights))
        cand_r_mean = max(1.0, float(radius_mean[condition]))

        ok = True
        for th in existing_thermals:
            dx = x - th.center[0]
            dy = y - th.center[1]
            d = math.hypot(dx, dy)
            min_sep = MIN_SEPARATION_MULTIPLIER * max(th.radius, cand_r_mean)
            if d < min_sep:
                ok = False
                break
        if not ok:
            continue
                
        t0 = 0.0 if now is None else float(now)
        return Thermal(center=(x, y), condition=condition, t0=t0)
            
    return None
