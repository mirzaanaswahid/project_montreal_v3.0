#!/usr/bin/env python3

from __future__ import annotations
import math, time, uuid, random, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

# ========================== Event data class ===========================

# Level-specific radius/duration (μ, σ) for simple Gaussian sampling
# You can tune these to match your detection model and scenario.
LEVEL_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Low":    {"radius": (80.0,  20.0), "duration": (900.0,   0.0)},   # 15 min
    "Medium": {"radius": (150.0, 30.0), "duration": (1800.0,  0.0)},   # 30 min
    "High":   {"radius": (250.0, 40.0), "duration": (3600.0,  0.0)},   # 60 min
}

def _gauss_trunc(mu_sigma: Tuple[float, float], min_frac: float = 0.3) -> float:
    mu, sigma = mu_sigma
    return max(mu * min_frac, random.gauss(mu, sigma))

class GroundEvent:
    """
    Stationary circular ground event used by detection.py and agent.py.

    Required by detection.py:
      - fields: id, level, radius, duration, t_gen, cx, cy
      - methods: active(now), pos property (np.array([x,y,0]))
    """
    def __init__(self,
                 cx: float,
                 cy: float,
                 radius: Optional[float] = None,
                 level: Optional[str] = None,
                 duration: Optional[float] = None,
                 t_gen: Optional[float] = None):
        self.id = str(uuid.uuid4())[:8]
        self.cx = float(cx)
        self.cy = float(cy)
        self.level = level or "Medium"

        # Provide robust defaults based on level if not supplied
        lp = LEVEL_PARAMS.get(self.level, LEVEL_PARAMS["Medium"])
        self.radius = float(radius if radius is not None else _gauss_trunc(lp["radius"]))
        self.duration = float(duration if duration is not None else _gauss_trunc(lp["duration"]))
        self.t_gen = float(time.time() if t_gen is None else t_gen)

        # Allow disabling without deleting
        self.active_state = True

    def active(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return self.active_state and (now < self.t_gen + self.duration)

    def age(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return now - self.t_gen

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.cx, self.cy, 0.0])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "level": self.level,
            "cx": self.cx, "cy": self.cy,
            "radius": self.radius, "duration": self.duration,
            "t_gen": self.t_gen,
        }

# ============================ Config defaults ============================

# These paths are only used by the __main__ smoke test. sim.py passes its own.
from config import GEOJSON_PATH, UTM_CRS, normalize_region_key

# Global arrival rate (expected events/min across all selected regions)
BASE_EVENTS_PER_MIN = 6.0

# Weighting of population vs density (0=pop only, 1=density only)
ALPHA = 0.5

# Optional severity bias by density class
DENSITY_SEVERITY_MULTIPLIER: Dict[str, Dict[str, float]] = {
    "High":     {"Low": 0.9, "Medium": 1.0, "High": 1.2},
    "Medium":   {"Low": 1.0, "Medium": 1.0, "High": 1.0},
    "Low":      {"Low": 1.1, "Medium": 1.0, "High": 0.9},
    "Very Low": {"Low": 1.2, "Medium": 0.9, "High": 0.8},
}

# Baseline level mix (percent). sim.py can override; we'll normalize to fractions.
LEVEL_MIX_PCT: Dict[str, float] = {"Low": 25.0, "Medium": 50.0, "High": 25.0}

# Geometry & spawning controls
MIN_SEPARATION_M = 50.0
MAX_ACTIVE_EVENTS = 2000
SPAWN_REJECTION_LIMIT = 200
RNG_SEED: Optional[int] = None

# ============================ Region catalog =============================

@dataclass
class Region:
    name: str
    key: str
    density_cat: str
    population: float
    area_km2: float
    density: float
    geom: Polygon  # unified/valid polygon (UTM)

# Use shared normalization function from config
_normalize_key = normalize_region_key

def _density_cat(d: float) -> str:
    if d > 7000: return "High"
    elif d > 3000: return "Medium"
    elif d > 1000: return "Low"
    else: return "Very Low"

def load_region_catalog(path: str, crs: str) -> List[Region]:
    """Load boroughs, attach attributes, and return Region list with valid UTM polygons."""
    gdf = gpd.read_file(path).to_crs(crs)

    # Minimal attribute table (identical to planner)
    import pandas as pd
    region_data = [
        ("Ahuntsic-Cartierville", 25.58, 138923),
        ("Anjou", 13.89, 45288),
        ("Baie-D’Urfé", 8.03, 3823),
        ("Beaconsfield", 24.68, 19908),
        ("Côte-des-Neiges-Notre-Dame-de-Grâce", 21.49, 173729),
        ("Côte-Saint-Luc", 6.81, 34425),
        ("Dollard-des-Ormeaux", 15.09, 49713),
        ("Dorval", 29.13, 18970),
        ("Hampstead", 1.77, 7153),
        ("Kirkland", 9.62, 21255),
        ("Lachine", 22.57, 46971),
        ("LaSalle", 25.21, 82933),
        ("Le Plateau-Mont-Royal", 8.14, 110329),
        ("Le Sud-Ouest", 18.10, 86347),
        ("L’Île-Bizard-Sainte-Geneviève", 36.32, 26099),
        ("L’Île-Dorval", 0.18, 134),
        ("Mercier-Hochelaga-Maisonneuve", 27.41, 142753),
        ("Mont-Royal", 7.46, 21202),
        ("Montréal-Est", 13.99, 3850),
        ("Montréal-Nord", 12.46, 86857),
        ("Montréal-Ouest", 1.42, 5254),
        ("Outremont", 3.80, 26505),
        ("Pierrefonds-Roxboro", 33.98, 73194),
        ("Pointe-Claire", 34.30, 33488),
        ("Rivière-des-Prairies-Pointe-aux-Trembles", 51.27, 113868),
        ("Rosemont-La Petite-Patrie", 15.88, 146501),
        ("Saint-Laurent", 43.06, 104366),
        ("Saint-Léonard", 13.51, 80983),
        ("Sainte-Anne-de-Bellevue", 11.18, 5158),
        ("Senneville", 18.59, 923),
        ("Verdun", 22.29, 72820),
        ("Ville-Marie", 21.50, 103017),
        ("Villeray-Saint-Michel-Parc-Extension", 16.48, 144814),
        ("Westmount", 4.02, 20832),
    ]
    df_attr = pd.DataFrame(region_data, columns=["Name","Area_km2","Population"])
    df_attr["Density"] = df_attr["Population"] / df_attr["Area_km2"]
    df_attr["key"] = df_attr["Name"].apply(_normalize_key)

    gdf["key"] = gdf["NOM"].apply(_normalize_key)
    gdf = gdf.merge(df_attr, on="key", how="left")
    gdf = gdf.dropna(subset=["Population"]).reset_index(drop=True)

    out: List[Region] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, MultiPolygon):
            geom = unary_union(geom)
        geom = geom.buffer(0)  # fix minor invalidities
        out.append(Region(
            name=row["NOM"], key=row["key"],
            density_cat=_density_cat(float(row["Density"])),
            population=float(row["Population"]),
            area_km2=float(row["Area_km2"]),
            density=float(row["Density"]),
            geom=geom
        ))
    return out

# ============================ Intensity model ==============================

def compute_region_weights(regions: List[Region], alpha: float) -> Dict[str, float]:
    """w_r ∝ (1-α)*Pop_r + α*Density_r (each normalized across regions)."""
    pops = np.array([r.population for r in regions], dtype=float)
    dens = np.array([r.density   for r in regions], dtype=float)
    pop_w = pops / pops.sum() if pops.sum() > 0 else np.ones_like(pops)/len(pops)
    den_w = dens / dens.sum() if dens.sum() > 0 else np.ones_like(dens)/len(dens)
    blend = (1.0 - alpha)*pop_w + alpha*den_w
    blend = blend / blend.sum()  # normalize to 1
    return {r.key: float(w) for r, w in zip(regions, blend)}

def per_region_lambdas(regions: List[Region],
                       base_events_per_min: float,
                       alpha: float) -> Dict[str, float]:
    w = compute_region_weights(regions, alpha)
    return {r.key: base_events_per_min * w[r.key] for r in regions}

# ============================ Level mix model ==============================

def normalize_level_mix(pct: Dict[str, float]) -> Dict[str, float]:
    total = sum(pct.values())
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Level mix must sum to 100, got {total}")
    return {k: v/100.0 for k,v in pct.items()}

def level_distribution_for_region(region: Region,
                                  base_mix_pct: Dict[str, float],
                                  severity_bias: Dict[str, Dict[str, float]]) -> List[Tuple[str,float]]:
    """Return [(level, prob), ...] after applying density-based multipliers."""
    base = normalize_level_mix(base_mix_pct)
    mult = severity_bias.get(region.density_cat, {"Low":1.0,"Medium":1.0,"High":1.0})
    vals = {lvl: base[lvl]*mult.get(lvl,1.0) for lvl in ["Low","Medium","High"]}
    s = sum(vals.values()) or 1.0
    return [(lvl, vals[lvl]/s) for lvl in ["Low","Medium","High"]]

# ======================= Event position sampling ==========================

def _sample_point_in_polygon(poly: Polygon,
                             rng: np.random.Generator,
                             max_tries: int = SPAWN_REJECTION_LIMIT) -> Optional[Tuple[float,float]]:
    """Uniform rejection sampling within polygon's bbox."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x,y)):
            return (x, y)
    return None

def _too_close(x: float, y: float, events: List[GroundEvent], min_sep: float) -> bool:
    for e in events:
        if math.hypot(x - e.cx, y - e.cy) < max(min_sep, e.radius):
            return True
    return False

# ============================ Generator ====================================

class RegionEventGenerator:
    """
    Region-weighted Poisson event generator.
    Call step(dt_minutes, now) each simulation tick.
    """
    def __init__(self,
                 regions: List[Region],
                 base_events_per_min: float = BASE_EVENTS_PER_MIN,
                 alpha: float = ALPHA,
                 level_mix: Optional[Dict[str, float]] = None,
                 level_mix_pct: Optional[Dict[str, float]] = None,
                 severity_bias: Dict[str, Dict[str,float]] = DENSITY_SEVERITY_MULTIPLIER,
                 min_separation_m: float = MIN_SEPARATION_M,
                 max_active: int = MAX_ACTIVE_EVENTS,
                 rng_seed: Optional[int] = RNG_SEED):
        self.regions = regions
        self.lambda_r = per_region_lambdas(regions, base_events_per_min, alpha)
        self.level_mix_pct = ( {k: v*100.0 for k,v in level_mix.items()} if level_mix is not None
                               else (level_mix_pct or LEVEL_MIX_PCT) )
        self.severity_bias = severity_bias
        self.min_sep = float(min_separation_m)
        self.max_active = int(max_active)
        self.rng = np.random.default_rng(rng_seed)
        self.active_events: List[GroundEvent] = []

        self.level_pmf: Dict[str, List[Tuple[str,float]]] = {
            r.key: level_distribution_for_region(r, self.level_mix_pct, self.severity_bias)
            for r in regions
        }
        self.R: Dict[str, Region] = {r.key: r for r in regions}

    def _pick_level(self, region_key: str) -> str:
        items = self.level_pmf[region_key]
        lvls  = [k for k,_ in items]
        probs = [p for _,p in items]
        return self.rng.choice(lvls, p=probs)

    def _spawn_one_in_region(self, r: Region, now: float) -> Optional[GroundEvent]:
        pt = _sample_point_in_polygon(r.geom, self.rng)
        if pt is None:
            return None
        x, y = pt
        if _too_close(x, y, self.active_events, self.min_sep):
            return None
        level = self._pick_level(r.key)
        # Let GroundEvent sample radius/duration defaults by level
        return GroundEvent(cx=x, cy=y, level=level, t_gen=now)

    def _prune_expired(self, now: float) -> None:
        self.active_events = [e for e in self.active_events if e.active(now)]

    def step(self, dt_minutes: float, now: Optional[float] = None) -> List[GroundEvent]:
        now = now if now is not None else time.time()
        self._prune_expired(now)
        if len(self.active_events) >= self.max_active:
            return []

        new_events: List[GroundEvent] = []
        budget = self.max_active - len(self.active_events)

        for r in self.regions:
            lam = self.lambda_r[r.key]  # events/min
            k = int(self.rng.poisson(lam * dt_minutes))
            if k <= 0:
                continue
            for _ in range(k):
                if budget <= 0:
                    break
                ev = self._spawn_one_in_region(r, now)
                if ev is not None:
                    self.active_events.append(ev)
                    new_events.append(ev)
                    budget -= 1

        return new_events

    # Utilities
    def snapshot_json(self) -> List[Dict[str, Any]]:
        return [e.as_dict() | {"pos": [e.cx, e.cy, 0.0]} for e in self.active_events]

    def save_snapshot(self, filename: str = "events_snapshot.json") -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.snapshot_json(), f, ensure_ascii=False, indent=2)

# ===================== Backward-compat convenience =========================

def spawn_random_event(area_bounds: Tuple[float, float, float, float],
                       existing_events: Optional[List[GroundEvent]] = None,
                       rng_seed: Optional[int] = None) -> Optional[GroundEvent]:
    """
    Legacy rectangular spawner (kept for compatibility with older code).
    Prefer using RegionEventGenerator in new simulations.
    """
    rng = np.random.default_rng(rng_seed)
    xmin, xmax, ymin, ymax = area_bounds
    for _ in range(SPAWN_REJECTION_LIMIT):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        ev = GroundEvent(cx=x, cy=y)  # level/radius/duration sampled by defaults
        if existing_events and any(math.hypot(ev.cx - ex.cx, ev.cy - ex.cy) < max(MIN_SEPARATION_M, ex.radius)
                                   for ex in existing_events):
            continue
        return ev
    return None

# ============================== Smoke test ================================

if __name__ == "__main__":
    regions = load_region_catalog(GEOJSON_PATH, UTM_CRS)
    gen = RegionEventGenerator(regions=regions, base_events_per_min=BASE_EVENTS_PER_MIN, alpha=ALPHA)
    now = time.time()
    for _ in range(20):
        new = gen.step(dt_minutes=0.5, now=now)  # 30 s
        now += 30.0
        print(f"+{len(new)} new events; active={len(gen.active_events)}")
    gen.save_snapshot("events_snapshot.json")
    print("[OK] wrote events_snapshot.json")
