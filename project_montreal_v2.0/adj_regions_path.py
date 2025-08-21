#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coverage_planner.py — Montréal 4-borough cluster
Generates lawnmower lanes, splits them among UAVs, renders a plot, and writes JSON.

Designed to be compatible with:
- detection.py FOV model (swath = 2*h*tan(FOV/2))
- agent.py load_waypoints_from_json(): expects {"waypoints":[{"x","y","z"}]}

Outputs (configurable):
- Per-UAV agent JSON files with {"waypoints":[{"x","y","z"}], "mission_info":{...}}  [meters]
- Optional combined planning JSON (regions/uavs)
- Optional WGS84 lat/lon JSON file for GIS/GCS

Author: (you)
"""
import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely import affinity
from pyproj import Transformer

# Optional plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------- COMPRESSION HELPER -----------------
import numpy as np

def leg_lengths_m(pts: List[Tuple[float, float]]) -> List[float]:
    """Return [d0-1, d1-2, …] in metres for a poly-line defined by pts."""
    return [
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:])
    ]

def _compress_collinear_pts(points: list[tuple[float, float]],
                             angle_tol_deg: float = 2.0) -> list[tuple[float, float]]:
    """
    Remove interior points that are (nearly) collinear.
    Keeps first & last point of each straight segment.
    """
    if len(points) <= 2:
        return points[:]

    keep = [points[0]]
    v_prev = np.subtract(points[1], points[0])

    for i in range(1, len(points) - 1):
        v_next = np.subtract(points[i + 1], points[i])
        # angle between previous and next segment
        dot = np.dot(v_prev, v_next)
        ang = math.degrees(
            math.acos(np.clip(dot / (np.linalg.norm(v_prev) * np.linalg.norm(v_next) + 1e-9), -1.0, 1.0))
        )
        if ang > angle_tol_deg:          # real turn – keep the waypoint
            keep.append(points[i])
            v_prev = v_next
        else:                            # still same heading – skip the middle waypoint
            v_prev = np.add(v_prev, v_next)  # accumulate vector for robustness

    keep.append(points[-1])              # always keep last
    return keep

# ---------------------------- CONFIG -----------------------------
# I/O (edit as needed) - Import centralized path from config
from config import GEOJSON_PATH, UTM_CRS, normalize_region_key
COMBINED_JSON_PATH = "waypoints_planning.json"     # planning view (regions/uavs)
WGS84_JSON_PATH    = "waypoints_wgs84.json"        # optional GIS output
PLOT_PATH          = "coverage.png"

# What to write
WRITE_AGENT_JSON    = True     # per-UAV files compatible with agent.py
WRITE_COMBINED_JSON = True     # single planning JSON
WRITE_WGS84_JSON    = False    # combined lat/lon JSON (for GIS/GCS)

# Agent-JSON naming (per UAV)
AGENT_JSON_TEMPLATE = "waypoints_agent__{uav_id}.json"  # use safe uav_id below

# Plot control
SHOW_PLOT = True
SAVE_PLOT = True

# CRS
UTM_CRS = "EPSG:32618"    # projected CRS used for geometry & agent waypoints (meters)
WGS84   = "EPSG:4326"     # optional export for GIS/GCS

# Platform / sensing (consistent with detection.py defaults)
OP_ALT_M = 400.0
CRUISE_SPEED_MPS = 12.0
FOV_DEG = 90.0            # full cone angle
OVERLAP = 1.20            # lane overlap factor (>1). 1.2 ≈ 20% overlap
TURN_SEC = 6.0

# Assignment
ASSIGN_RULE = "density"   # "density" or "fixed"
FIXED_UAVS_PER_BOROUGH = 2
UAVS_PER_REGION_OVERRIDE: Dict[str, int] = {
    # "Le Plateau-Mont-Royal": 2,
}

# Plot appearance
UAV_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]
LANE_WIDTH = 1.2
LANE_ALPHA = 0.95

# ----------------------- DENSITY ATTRIBUTES ----------------------
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
df_attr = pd.DataFrame(region_data, columns=["Name", "Area_km2", "Population"])
df_attr["Density"] = df_attr["Population"] / df_attr["Area_km2"]

def classify_density(d):
    if d > 7000: return "High"
    elif d > 3000: return "Medium"
    elif d > 1000: return "Low"
    else: return "Very Low"

df_attr["DensityCat"] = df_attr["Density"].apply(classify_density)

density_to_uavs = {"High": 3, "Medium": 2, "Low": 1, "Very Low": 1}
density_colors  = {"High": "red", "Medium": "orange", "Low": "lightgreen", "Very Low": "grey"}

# --------------------------- UTILS --------------------------------
# Use shared normalization function from config
normalize = normalize_region_key

def swath_width_from_fov(h_m: float, fov_deg: float) -> float:
    half = math.radians(fov_deg / 2.0)
    return 2.0 * h_m * math.tan(half)

LANE_SPACING = swath_width_from_fov(OP_ALT_M, FOV_DEG) / OVERLAP  # ~667 m at 400m/90°/20%

# ------------------------ DATA LOADING ----------------------------
def load_and_merge(path: str, crs: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path).to_crs(crs)
    g["key"] = g["NOM"].apply(normalize)
    df_attr["key"] = df_attr["Name"].apply(normalize)
    g = g.merge(df_attr, on="key", how="left")
    g = g.dropna(subset=["DensityCat"]).reset_index(drop=True)
    return g

# ------------------------ CLUSTER SELECTION -----------------------
def build_adjacency(gdf: gpd.GeoDataFrame):
    adj = {i: {} for i in range(len(gdf))}
    for i in range(len(gdf)):
        ai = gdf.geometry.iloc[i]
        for j in range(i+1, len(gdf)):
            bj = gdf.geometry.iloc[j]
            shared_len = float(ai.boundary.intersection(bj.boundary).length)
            if shared_len > 0:
                adj[i][j] = shared_len
                adj[j][i] = shared_len
    return adj

def is_connected_subset(nodes, adj):
    nodes = list(nodes)
    S = set(nodes); seen = {nodes[0]}; stack = [nodes[0]]
    while stack:
        u = stack.pop()
        for v in adj[u].keys():
            if v in S and v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == len(nodes)

def total_internal_border(nodes, adj):
    return sum(adj[i][j] for i,j in combinations(nodes,2) if j in adj[i])

def count_high(nodes, gdf):
    return sum(gdf.iloc[i]["DensityCat"] == "High" for i in nodes)

def find_connected_four(gdf, adj, min_high=1):
    allowed = [i for i, row in gdf.iterrows() if row["DensityCat"] in {"High","Medium"}]
    best, best_key = None, (-1, -1.0)
    for sub in combinations(allowed, 4):
        if not is_connected_subset(sub, adj): continue
        num_h = count_high(sub, gdf)
        if num_h < min_high: continue
        score = total_internal_border(sub, adj)
        if (num_h, score) > best_key:
            best, best_key = list(sub), (num_h, score)
    if best is None:
        raise RuntimeError("No connected set of 4 (≥Medium, ≥1 High) found.")
    return best, best_key

# ---------------------- COVERAGE GEOMETRY -------------------------
def to_valid_union(geom):
    if geom.is_empty: return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        g = geom.buffer(0)
        return unary_union(g)
    return unary_union(geom)

def auto_sweep_angle(poly: Polygon) -> float:
    """Pick sweep angle orthogonal to longest edge of minimum rotated rectangle."""
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]
    edges = []
    for i in range(4):
        x1,y1 = coords[i]; x2,y2 = coords[(i+1) % 4]
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        edges.append((length, angle_deg))
    longest = max(edges, key=lambda t: t[0])
    return (longest[1] + 90.0) % 180.0

@dataclass
class BoustroResult:
    points_xy: List[Tuple[float, float]]
    col_index_of_wp: List[int]
    lane_indices_in_order: List[int]
    angle_deg: float
    spacing_m: float

def boustrophedon_waypoints(poly: Polygon, spacing: float, angle_deg: float) -> BoustroResult:
    """Generate an ordered boustrophedon waypoint sequence inside 'poly'."""
    poly = to_valid_union(poly)
    if isinstance(poly, MultiPolygon):
        poly = unary_union(poly)

    origin = Point(poly.centroid.x, poly.centroid.y)   # fixed origin for rotations
    rot_poly = affinity.rotate(poly, -angle_deg, origin=origin, use_radians=False)
    minx, miny, maxx, maxy = rot_poly.bounds
    minx -= spacing; maxx += spacing; miny -= spacing; maxy += spacing

    all_points, col_of_point, lane_order = [], [], []
    col_idx, x = 0, minx
    while x <= maxx + 1e-6:
        base = LineString([(x, miny), (x, maxy)])
        inter = rot_poly.intersection(base)
        if inter.is_empty:
            x += spacing; col_idx += 1; continue

        # normalize to list of segments
        if isinstance(inter, LineString):
            segments = [inter]
        elif isinstance(inter, MultiLineString):
            segments = list(inter.geoms)
        else:
            try:
                segments = [g for g in inter.geoms if isinstance(g, LineString)]
            except Exception:
                segments = []

        # order bottom->top in rotated frame
        def ymid(seg: LineString) -> float:
            (x1,y1),(x2,y2) = list(seg.coords); return 0.5*(y1+y2)
        segments.sort(key=ymid)

        # make column sequence (even up, odd down)
        seq = []
        for seg in segments:
            (x1,y1),(x2,y2) = list(seg.coords)
            if y1 <= y2:
                seq.append(((x1,y1),(x2,y2)))
            else:
                seq.append(((x2,y2),(x1,y1)))
        if col_idx % 2 == 1:
            seq = list(reversed([ (b,a) for (a,b) in seq ]))

        # rotate back about same origin
        for (p0, p1) in seq:
            p0b = affinity.rotate(Point(*p0), angle_deg, origin=origin, use_radians=False)
            p1b = affinity.rotate(Point(*p1), angle_deg, origin=origin, use_radians=False)
            all_points.append((p0b.x, p0b.y)); col_of_point.append(col_idx)
            all_points.append((p1b.x, p1b.y)); col_of_point.append(col_idx)

        lane_order.append(col_idx)
        x += spacing; col_idx += 1

    return BoustroResult(all_points, col_of_point, lane_order, angle_deg, spacing)

def split_columns_contiguously(lane_indices: List[int], n_uav: int) -> List[List[int]]:
    n = len(lane_indices)
    if n_uav <= 1 or n <= 1:
        return [lane_indices]
    chunk = math.ceil(n / n_uav)
    return [ lane_indices[i*chunk : min((i+1)*chunk, n)] for i in range(n_uav) ]

def filter_waypoints_by_columns(bres: BoustroResult, keep_cols: List[int]) -> List[Tuple[float,float]]:
    keep = set(keep_cols)
    return [pt for pt, col in zip(bres.points_xy, bres.col_index_of_wp) if col in keep]

# ---------------------- ASSIGNMENT HELPERS ------------------------
def uavs_for_borough(name: str, density_cat: str) -> int:
    if name in UAVS_PER_REGION_OVERRIDE:
        return max(1, int(UAVS_PER_REGION_OVERRIDE[name]))
    if ASSIGN_RULE == "fixed":
        return max(1, int(FIXED_UAVS_PER_BOROUGH))
    return max(1, int(density_to_uavs.get(density_cat, 1)))

def safe_id(s: str) -> str:
    # make a filesystem-safe token for filenames
    return "".join(c.lower() if c.isalnum() else "_" for c in s)

# -------------------------- PLOTTING ------------------------------
def plot_cluster_with_lanes(subset: gpd.GeoDataFrame,
                            per_region_cols: Dict[str, List[List[int]]],
                            per_region_bres: Dict[str, BoustroResult]) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))

    # Polygons
    for cat, sub in subset.groupby("DensityCat"):
        sub.plot(ax=ax, color=density_colors[cat], edgecolor="black", linewidth=0.6, label=cat)
    subset.boundary.plot(ax=ax, linewidth=1.4, color="black")

    legend_handles_uav = []
    for _, row in subset.iterrows():
        name = row["NOM"]
        bres = per_region_bres[name]
        col_blocks = per_region_cols[name]

        # ➟ inside the for-loop that iterates over each UAV's column block
        for u, cols in enumerate(col_blocks, start=1):
            color = UAV_COLORS[u % len(UAV_COLORS)]
            pts = filter_waypoints_by_columns(bres, cols)

            # ── 1️⃣ draw every lane as before ─────────────────────────────
            for i in range(0, len(pts) - 1, 2):          # each pair is a lane segment
                (x0, y0) = pts[i]
                (x1, y1) = pts[i + 1]
                ax.plot([x0, x1], [y0, y1],
                        linewidth=LANE_WIDTH, color=color, alpha=LANE_ALPHA)

            # ── 2️⃣ NEW: scatter-plot the way-points themselves ──────────
            if pts:                                       # avoid empty list edge-case
                x_wp, y_wp = zip(*pts)
                ax.scatter(x_wp, y_wp,
                           s=6,                  # marker size
                           color=color,
                           edgecolors="none",
                           alpha=0.9,
                           zorder=3)            # above polygons/lanes

            legend_handles_uav.append(
                Line2D([0], [0], color=color, lw=2, label=f"{name} – UAV {u}")
            )

    # Labels & legends
    for _, row in subset.iterrows():
        c = row.geometry.centroid
        ax.text(c.x, c.y, row["NOM"], fontsize=7, ha="center", va="center")

    density_patches = [
        Patch(facecolor=density_colors["High"],   edgecolor='black', label="High density"),
        Patch(facecolor=density_colors["Medium"], edgecolor='black', label="Medium density"),
    ]
    first_legend = ax.legend(handles=density_patches, title="Density class", loc="lower left", frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=legend_handles_uav,
              title=f"Lawnmower paths (spacing ≈ {swath_width_from_fov(OP_ALT_M, FOV_DEG)/OVERLAP:.0f} m)",
              loc="upper right", fontsize=7)
    ax.set_title(
        f"Montréal – 4-borough connected cluster (density ≥ Medium; ≥1 High)\n"
        f"Lanes per UAV (alt={OP_ALT_M:.0f} m, FOV={FOV_DEG:.0f}°, overlap={int((OVERLAP-1)*100)}%)",
        fontsize=13
    )
    ax.set_axis_off()
    plt.tight_layout()

    if SAVE_PLOT:
        plt.savefig(PLOT_PATH, dpi=200)
        print(f"[OK] Saved plot to {PLOT_PATH}")
    if SHOW_PLOT:
        plt.show()

# -------------------------- MAIN ---------------------------------
def main():
    # Load data & find cluster
    gdf = load_and_merge(GEOJSON_PATH, UTM_CRS)
    adj = build_adjacency(gdf)
    indices, (num_high, score) = find_connected_four(gdf, adj, min_high=1)
    subset = gdf.iloc[indices].copy()

    # Transformers
    to_wgs84 = Transformer.from_crs(UTM_CRS, WGS84, always_xy=True)

    # Planning JSON (combined)
    combined = {
        "meta": {
            "crs_xy": UTM_CRS,
            "crs_ll": WGS84,
            "created_by": "coverage_planner",
            "note": "boustrophedon lanes; contiguous column blocks per UAV"
        },
        "config": {
            "altitude_m": OP_ALT_M,
            "speed_mps": CRUISE_SPEED_MPS,
            "fov_deg": FOV_DEG,
            "overlap": OVERLAP,
            "lane_spacing_m": round(LANE_SPACING, 3)
        },
        "cluster_summary": {
            "selected_boroughs": list(subset["NOM"]),
            "num_high_in_cluster": int(num_high),
            "sum_shared_border_m": float(round(score, 2))
        },
        "regions": []
    }

    # For plotting reuse
    per_region_bres: Dict[str, BoustroResult] = {}
    per_region_cols: Dict[str, List[List[int]]] = {}

    # Per-UAV agent files accumulator (also produce a single all-uav index)
    agent_index: Dict[str, str] = {}

    for _, row in subset.iterrows():
        geom = row.geometry
        name = row["NOM"]
        dens = row["DensityCat"]
        key  = row["key"]

        # Normalize geometry
        if isinstance(geom, MultiPolygon):
            geom = unary_union(geom)
        geom = geom.buffer(0)

        # Plan lanes & split among UAVs
        angle = auto_sweep_angle(geom)
        bres  = boustrophedon_waypoints(geom, LANE_SPACING, angle)
        n_uav = uavs_for_borough(name, dens)
        col_blocks = split_columns_contiguously(bres.lane_indices_in_order, n_uav)

        # Save for plot
        per_region_bres[name] = bres
        per_region_cols[name] = col_blocks

        # Build region planning entry
        region_entry = {
            "name": name,
            "key": key,
            "density": dens,
            "uavs_assigned": int(n_uav),
            "sweep": {
                "angle_deg": round(angle, 3),
                "lane_spacing_m": round(bres.spacing_m, 3)
            },
            "uavs": []
        }

        # For each UAV: produce agent JSON (x,y,z in meters) + (optional) WGS84
        for uav_idx, cols in enumerate(col_blocks, start=1):
            # 1. all points that belong to this UAV (already ordered)
            pts_xy_full = filter_waypoints_by_columns(bres, cols)

            # 2. ***NEW – reduce to lane termini only***
            lane_pts = []
            for i in range(0, len(pts_xy_full), 2):
                p0 = pts_xy_full[i]      # start of lane
                p1 = pts_xy_full[i + 1]  # end   of lane
                if not lane_pts:
                    lane_pts.append(p0)  # first start
                lane_pts.append(p1)      # every lane's far end
            # Result: 0‑1‑3‑5‑… pattern (no mid‑lane connectors)

            # -------------------------------- DISTANCE METRICS ------------------------
            legs = leg_lengths_m(lane_pts)          # one float per segment
            path_len = sum(legs)                    # total metres
            n_wp = len(lane_pts)
            # --------------------------------------------------------------------------

            # 3. build the agent JSON out of lane_pts
            waypoints_agent = [
                {"x": round(x, 3), "y": round(y, 3), "z": round(OP_ALT_M, 2)}
                for (x, y) in lane_pts
            ]

            # Optional WGS84 list
            waypoints_ll = []
            if WRITE_WGS84_JSON:
                for (x, y) in lane_pts:
                    lon, lat = to_wgs84.transform(x, y)
                    waypoints_ll.append({"lat": round(lat, 7), "lon": round(lon, 7), "alt_m": round(OP_ALT_M, 2)})

            # Agent per-UAV file
            uav_id = f"{safe_id(name)}__u{uav_idx}"
            if WRITE_AGENT_JSON:
                fname = AGENT_JSON_TEMPLATE.format(uav_id=uav_id)
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump({
                        "waypoints": waypoints_agent,
                        "mission_info": {
                            "region":      name,
                            "uav_id":      uav_id,
                            "total_waypoints": n_wp,
                            "path_length_m":   round(path_len, 1),
                            "legs_m":         [round(d, 1) for d in legs],
                            "altitude_m": OP_ALT_M
                        }
                    }, f, ensure_ascii=False, indent=2)
                agent_index[uav_id] = fname
                print(f"[OK] wrote agent JSON: {fname} ({len(waypoints_agent)} wps)")
                print(f"{uav_id:25s}  {n_wp:3d} wp   {path_len/1000:5.2f} km")

            # ----------  INDIVIDUAL-UAV PLOT  ----------
            fig_i, ax_i = plt.subplots(figsize=(8, 8))

            # light grey outline of the whole borough for context
            ax_i.add_patch(plt.Polygon(np.array(geom.exterior.coords),
                                       facecolor="none", edgecolor="grey", lw=1.0,
                                       alpha=0.5))

            # draw the lanes for *this* UAV only
            for i in range(0, len(pts_xy_full), 2):
                x0, y0 = pts_xy_full[i]
                x1, y1 = pts_xy_full[i+1]
                ax_i.plot([x0, x1], [y0, y1],
                          lw=2.0, color=UAV_COLORS[(uav_idx-1) % len(UAV_COLORS)])

            # emphasise the waypoint order if you like
            ax_i.scatter(*zip(*lane_pts), s=10, color="black")

            ax_i.set_title(f"{name} – {uav_id} waypoints")
            ax_i.set_aspect("equal"); ax_i.axis("off")

            plt.tight_layout()
            fig_i.savefig(f"{uav_id}_lanes.png", dpi=200)
            plt.close(fig_i)
            # --------------------------------------------

            # Add to combined planning object
            region_entry["uavs"].append({
                "uav_id": uav_id,
                "lane_indices": cols,
                "waypoints_xy": waypoints_agent,
                **({"waypoints_ll": waypoints_ll} if WRITE_WGS84_JSON else {})
            })

        combined["regions"].append(region_entry)

    import csv
    with open("uav_path_summary.csv", "w", newline="") as csvf:
        wr = csv.writer(csvf)
        wr.writerow(["uav_id", "region", "waypoints", "path_km"])
        for r in combined["regions"]:
            for u in r["uavs"]:
                info = u  # contains uav_id and waypoints_xy
                wps = info["waypoints_xy"]
                d = sum(leg_lengths_m([(p["x"], p["y"]) for p in wps])) / 1000.0
                wr.writerow([info["uav_id"], r["name"], len(wps), f"{d:.3f}"])
    print("[OK] wrote uav_path_summary.csv")

    # Write combined planning JSON
    if WRITE_COMBINED_JSON:
        with open(COMBINED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote combined planning JSON: {COMBINED_JSON_PATH}")

    # Optional WGS84 top-level file
    if WRITE_WGS84_JSON:
        # Just reuse the combined object; it already contains lat/lon per UAV
        with open(WGS84_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote WGS84 planning JSON: {WGS84_JSON_PATH}")

    # Plot
    plot_cluster_with_lanes(subset, per_region_cols, per_region_bres)
    
    # Summary at the very end of main()
    total_uavs = sum(len(r["uavs"]) for r in combined["regions"])
    print(f"[OK] Coverage plan complete – {total_uavs} UAV(s) planned.")
    # optionally store it
    combined["meta"]["total_uavs"] = total_uavs

if __name__ == "__main__":
    main()
