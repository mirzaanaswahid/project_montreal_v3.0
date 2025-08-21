import os
import time
import json
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pyproj
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


@dataclass
class SyntheticGenConfig:
    geojson_path: str
    land_cover_path: str
    output_folder: str
    output_filename: str = "synthetic_thermal_data_5yr_landcover.csv"
    years: int = 5
    thermal_months: tuple = (5, 6, 7, 8, 9)
    min_hour: int = 10
    max_hour: int = 18


def generate_synthetic_thermals(cfg: SyntheticGenConfig) -> str:
    """Run synthetic thermal event generation; returns CSV path."""
    from datetime import datetime, timedelta
    import random
    from shapely.geometry import Point

    os.makedirs(cfg.output_folder, exist_ok=True)
    output_path = os.path.join(cfg.output_folder, cfg.output_filename)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=cfg.years * 365)
    total_seconds_in_period = (end_time - start_time).total_seconds()

    thermal_categories = {
        "Weak": ([0.5, 1.5], [50, 150], [300, 1000], [5, 15], 0.4),
        "Medium": ([1.5, 3.0], [100, 300], [800, 2000], [10, 30], 0.35),
        "Strong": ([3.0, 5.0], [150, 400], [1500, 3000], [15, 45], 0.2),
        "Very Strong": ([5.0, 7.0], [150, 500], [2000, 4000], [20, 60], 0.05),
    }
    category_names = list(thermal_categories.keys())
    category_weights = [thermal_categories[cat][4] for cat in category_names]

    land_cover_probabilities = {1: 0.20, 2: 0.20, 5: 0.25, 6: 0.22, 8: 0.40, 10: 0.50,
                                11: 0.30, 12: 0.40, 13: 0.35, 14: 0.10, 15: 0.70, 16: 0.75,
                                17: 0.85, 18: 0.01, 19: 0.00, 0: 0.00, 'default': 0.1}

    approx_active_hours = 5 * len(cfg.thermal_months) * 30 * (cfg.max_hour - cfg.min_hour)
    thermals_per_active_hour_avg_density = 5
    total_thermals_to_generate = approx_active_hours * thermals_per_active_hour_avg_density

    MIN_DIST_FACTOR = 3.0
    RECENT_POINTS_CHECK = 50
    MAX_SPATIAL_ATTEMPTS = 200
    MAX_TIMESTAMP_ATTEMPTS = 100

    def get_random_point_in_polygon(poly):
        min_x, min_y, max_x, max_y = poly.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if poly.contains(random_point):
                return random_point

    def sample_thermal_properties(category):
        props = thermal_categories[category]
        import random as _r
        lift = _r.uniform(props[0][0], props[0][1])
        diameter = _r.uniform(props[1][0], props[1][1])
        height = _r.uniform(props[2][0], props[2][1])
        duration = _r.uniform(props[3][0], props[3][1])
        return lift, diameter, height, duration

    def get_season(month):
        if month in [6, 7, 8]: return "Summer"
        if month in [5, 9]: return "Spring/Fall"
        return "Off-Season"

    def get_time_of_day(hour):
        if cfg.min_hour <= hour < 12: return "Morning"
        if 12 <= hour < 16: return "Afternoon"
        if 16 <= hour < cfg.max_hour: return "Late Afternoon"
        return "Off-Hours"

    print(f"Reading GeoJSON from: {cfg.geojson_path}")
    gdf = gpd.read_file(cfg.geojson_path)
    source_crs_geojson = gdf.crs or "EPSG:32188"
    gdf.crs = source_crs_geojson

    print(f"Loading Land Cover: {cfg.land_cover_path}")
    land_cover_dataset = rasterio.open(cfg.land_cover_path)
    source_crs_raster = land_cover_dataset.crs
    project_coords = None
    if source_crs_raster != source_crs_geojson:
        project_coords = pyproj.Transformer.from_crs(source_crs_geojson, source_crs_raster, always_xy=True).transform

    total_area = gdf.geometry.area.sum()
    print(f"Generating approximately {total_thermals_to_generate} thermals...")

    synthetic_data = []
    generated_points_history = []
    generated_count = 0
    start_exec_time = time.time()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    import random as _rnd
    while generated_count < total_thermals_to_generate:
        # time
        valid_timestamp_found = False
        timestamp_attempts = 0
        from datetime import timedelta as _timedelta
        while not valid_timestamp_found and timestamp_attempts < MAX_TIMESTAMP_ATTEMPTS:
            timestamp_attempts += 1
            random_second_offset = _rnd.uniform(0, total_seconds_in_period)
            timestamp = start_time + _timedelta(seconds=random_second_offset)
            if timestamp.month in cfg.thermal_months and cfg.min_hour <= timestamp.hour < cfg.max_hour:
                valid_timestamp_found = True
        if not valid_timestamp_found:
            continue

        # space
        point_accepted = False
        spatial_attempts = 0
        point_geojson_crs = None
        land_cover_code = None
        region_name = None

        while not point_accepted and spatial_attempts < MAX_SPATIAL_ATTEMPTS:
            spatial_attempts += 1
            rand_area_val = _rnd.uniform(0, total_area)
            cumulative_area = 0
            target_region = None
            for _, region in gdf.iterrows():
                cumulative_area += region.geometry.area
                if rand_area_val <= cumulative_area:
                    target_region = region
                    break
            if target_region is None:
                target_region = gdf.iloc[-1]
            region_geom = target_region.geometry

            point_candidate_geojson_crs = get_random_point_in_polygon(region_geom)

            coords_for_sampling = (point_candidate_geojson_crs.x, point_candidate_geojson_crs.y)
            if project_coords:
                try:
                    coords_for_sampling = project_coords(point_candidate_geojson_crs.x, point_candidate_geojson_crs.y)
                except Exception:
                    continue

            try:
                lc_value_generator = land_cover_dataset.sample([coords_for_sampling], indexes=1)
                land_cover_code = next(lc_value_generator)[0]
            except Exception:
                land_cover_code = 0
                continue

            if land_cover_code == land_cover_dataset.nodata:
                land_cover_code = 0

            probability = land_cover_probabilities.get(land_cover_code, land_cover_probabilities.get('default', 0))

            if _rnd.random() < probability:
                point_geojson_crs = point_candidate_geojson_crs

                is_too_close = False
                import numpy as _np
                temp_radius_for_check = np.mean(thermal_categories["Medium"][1]) / 2.0
                check_against = generated_points_history[-RECENT_POINTS_CHECK:]
                for prev_point_geojson_crs, prev_radius in check_against:
                    if prev_radius is None or temp_radius_for_check is None:
                        continue
                    distance = point_geojson_crs.distance(prev_point_geojson_crs)
                    min_allowed_distance = MIN_DIST_FACTOR * prev_radius
                    if distance < min_allowed_distance:
                        is_too_close = True
                        break

                if not is_too_close:
                    point_accepted = True
                    region_name = target_region['NOM']

        if not point_accepted:
            continue

        # coords
        latitude = None
        longitude = None
        try:
            point_gs = gpd.GeoSeries([point_geojson_crs], crs=source_crs_geojson)
            point_latlon = point_gs.to_crs("EPSG:4326").iloc[0]
            latitude = point_latlon.y
            longitude = point_latlon.x
        except Exception:
            pass

        category = _rnd.choices(category_names, weights=category_weights, k=1)[0]
        lift_mps, diameter_m, max_height_m_agl, duration_min = sample_thermal_properties(category)
        radius_m = diameter_m / 2.0
        season = get_season(timestamp.month)
        time_of_day = get_time_of_day(timestamp.hour)

        generated_points_history.append((point_geojson_crs, radius_m))
        if len(generated_points_history) > RECENT_POINTS_CHECK * 2:
            generated_points_history = generated_points_history[-RECENT_POINTS_CHECK:]

        synthetic_data.append({
            "timestamp_utc": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "latitude": latitude,
            "longitude": longitude,
            "easting": point_geojson_crs.x,
            "northing": point_geojson_crs.y,
            "region_name": region_name,
            "season": season,
            "time_of_day": time_of_day,
            "land_cover_type": str(land_cover_code),
            "strength_category": category,
            "lift_rate_mps": round(lift_mps, 2),
            "core_diameter_m": round(diameter_m, 1),
            "radius_m": round(radius_m, 1),
            "max_height_m_agl": round(max_height_m_agl),
            "duration_min": round(duration_min),
        })
        generated_count += 1

    land_cover_dataset.close()
    warnings.resetwarnings()

    df_output = pd.DataFrame(synthetic_data)
    df_output.to_csv(output_path, index=False)
    return output_path


@dataclass
class ProbMapConfig:
    csv_path: str
    geojson_path: str
    land_cover_path: str
    output_folder: str
    grid_resolution_m: float = 100.0


def build_probability_map(cfg: ProbMapConfig) -> tuple[str, str]:
    """Build conditional probability map and metadata; returns (npy_path, meta_path)."""
    os.makedirs(cfg.output_folder, exist_ok=True)

    thermal_df = pd.read_csv(cfg.csv_path, parse_dates=['timestamp_utc'])
    # Accept either 'land_cover_code' or 'land_cover_type' (name or code string)
    if 'land_cover_code' in thermal_df.columns:
        thermal_df['land_cover_code'] = thermal_df['land_cover_code'].astype(int)
    elif 'land_cover_type' in thermal_df.columns:
        # Try to coerce to int; if fails, map common names → codes
        try:
            thermal_df['land_cover_code'] = thermal_df['land_cover_type'].astype(int)
        except Exception:
            name_to_code = {
                "Needleleaf Forest": 1, "Taiga Forest": 2, "Broadleaf Forest": 5, "Mixed Forest": 6,
                "Shrubland": 8, "Grassland": 10, "Shrubland-Lichen-Moss": 11, "Grassland-Lichen-Moss": 12,
                "Barren-Lichen-Moss": 13, "Wetland": 14, "Cropland": 15, "Barren lands": 16,
                "Urban": 17, "Water": 18, "Snow/Ice": 19, "NoData": 0, "Unknown": 0
            }
            thermal_df['land_cover_code'] = thermal_df['land_cover_type'].map(lambda x: name_to_code.get(str(x), 0)).astype(int)
    else:
        raise ValueError("CSV is missing 'land_cover_type' or 'land_cover_code' column")

    gdf_regions = gpd.read_file(cfg.geojson_path)
    source_crs_geojson = gdf_regions.crs if gdf_regions.crs else "EPSG:32188"
    if gdf_regions.crs is None:
        gdf_regions.crs = source_crs_geojson

    lc_raster = rasterio.open(cfg.land_cover_path)
    source_crs_raster = lc_raster.crs

    transform_geojson_to_raster = None
    transform_raster_to_geojson = None
    if source_crs_raster != source_crs_geojson:
        transformer_g2r = pyproj.Transformer.from_crs(source_crs_geojson, source_crs_raster, always_xy=True)
        transform_geojson_to_raster = transformer_g2r.transform
        transformer_r2g = pyproj.Transformer.from_crs(source_crs_raster, source_crs_geojson, always_xy=True)
        transform_raster_to_geojson = transformer_r2g.transform

    geojson_bounds_native = gdf_regions.total_bounds
    geojson_bbox = box(*geojson_bounds_native)
    if transform_geojson_to_raster:
        geojson_bounds_transformed = shapely_transform(transform_geojson_to_raster, geojson_bbox).bounds
        min_x_grid, min_y_grid, max_x_grid, max_y_grid = geojson_bounds_transformed
    else:
        min_x_grid, min_y_grid, max_x_grid, max_y_grid = geojson_bounds_native

    GRID_RES = cfg.grid_resolution_m
    grid_origin_x = math.floor(min_x_grid / GRID_RES) * GRID_RES
    grid_origin_y = math.ceil(max_y_grid / GRID_RES) * GRID_RES
    grid_width = math.ceil((max_x_grid - grid_origin_x) / GRID_RES)
    grid_height = math.ceil((grid_origin_y - min_y_grid) / GRID_RES)
    grid_affine = rasterio.Affine(GRID_RES, 0.0, grid_origin_x, 0.0, -GRID_RES, grid_origin_y)

    season_map = {"Spring/Fall": 0, "Summer": 1}
    time_of_day_map = {"Morning": 0, "Afternoon": 1, "Late Afternoon": 2}
    land_cover_codes_present = list(range(20))
    land_cover_map_idx = {code: idx for idx, code in enumerate(land_cover_codes_present)}

    thermal_counts = np.zeros((grid_height, grid_width, len(season_map), len(time_of_day_map), len(land_cover_codes_present)), dtype=np.float32)
    total_context_counts = np.zeros((len(season_map), len(time_of_day_map), len(land_cover_codes_present)), dtype=np.float32)

    def get_season_idx(month):
        if month in [6, 7, 8]: return season_map["Summer"]
        if month in [5, 9]: return season_map["Spring/Fall"]
        return None

    def get_tod_idx(hour):
        if 10 <= hour < 12: return time_of_day_map["Morning"]
        if 12 <= hour < 16: return time_of_day_map["Afternoon"]
        if 16 <= hour < 18: return time_of_day_map["Late Afternoon"]
        return None

    skipped_points = 0
    for _, event in thermal_df.iterrows():
        ts = event['timestamp_utc']
        season_idx = get_season_idx(ts.month)
        tod_idx = get_tod_idx(ts.hour)
        if season_idx is None or tod_idx is None:
            skipped_points += 1
            continue
        point_x_geojson, point_y_geojson = event['easting'], event['northing']
        coords_for_raster = (point_x_geojson, point_y_geojson)
        if transform_geojson_to_raster:
            try:
                coords_for_raster = transform_geojson_to_raster(point_x_geojson, point_y_geojson)
            except Exception:
                skipped_points += 1
                continue
        try:
            row_idx, col_idx = rasterio.transform.rowcol(grid_affine, coords_for_raster[0], coords_for_raster[1])
            lc_code = int(event['land_cover_code'])
            lc_idx = land_cover_map_idx.get(lc_code)
            if 0 <= row_idx < grid_height and 0 <= col_idx < grid_width and lc_idx is not None:
                thermal_counts[row_idx, col_idx, season_idx, tod_idx, lc_idx] += 1
                total_context_counts[season_idx, tod_idx, lc_idx] += 1
            else:
                skipped_points += 1
        except Exception:
            skipped_points += 1

    num_locations = float(grid_width * grid_height)
    total_context_counts_reshaped = total_context_counts[np.newaxis, np.newaxis, :, :, :]
    epsilon = 1e-9
    calculated_prob = (thermal_counts + 1.0) / (total_context_counts_reshaped + num_locations + epsilon)
    baseline_prob = 1.0 / (num_locations + epsilon) if num_locations > 0 else epsilon
    prob_map = np.where(total_context_counts_reshaped > 0, calculated_prob, baseline_prob)
    prob_map = np.clip(prob_map, 0.0, 1.0)

    prob_map_output_path = os.path.join(cfg.output_folder, "conditional_probability_map.npy")
    meta_output_path = os.path.join(cfg.output_folder, "probability_map_metadata.json")
    np.save(prob_map_output_path, prob_map)

    grid_origin_x_geojson, grid_origin_y_geojson = (grid_origin_x, grid_origin_y)
    if transform_raster_to_geojson:
        grid_origin_x_geojson, grid_origin_y_geojson = transform_raster_to_geojson(grid_origin_x, grid_origin_y)

    metadata = {
        "description": "Conditional Probability Map P(Thermal | GridCell, Season, TimeOfDay, LandCoverCode)",
        "generated_on": pd.Timestamp.now().isoformat(),
        "source_synthetic_data": cfg.csv_path,
        "source_land_cover_data": cfg.land_cover_path,
        "source_geojson_boundaries": cfg.geojson_path,
        "grid_crs": str(source_crs_raster),
        "grid_dimensions": {"height": grid_height, "width": grid_width},
        "grid_resolution": GRID_RES,
        "grid_affine_transform_gdal": list(grid_affine.to_gdal()),
        "grid_origin_in_geojson_crs": {"x": grid_origin_x_geojson, "y": grid_origin_y_geojson, "crs": str(source_crs_geojson)},
        "map_shape": prob_map.shape,
        "map_dimensions_order": ["y_index(row)", "x_index(col)", "season_index", "tod_index", "land_cover_index"],
        "context_mappings": {
            "season": season_map,
            "time_of_day": time_of_day_map,
            "land_cover_code_to_index": land_cover_map_idx,
        },
    }
    with open(meta_output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    lc_raster.close()
    return prob_map_output_path, meta_output_path 