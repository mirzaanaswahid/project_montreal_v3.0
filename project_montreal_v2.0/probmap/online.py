from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from rasterio.transform import rowcol


@dataclass
class ProbMapRuntime:
    meta_path: str
    prob_path: str
    lc_raster_path: str
    avg_npz_path: Optional[str] = None


class ProbabilisticThermalMap:
    def __init__(self, cfg: ProbMapRuntime) -> None:
        self.meta = self._load_json(cfg.meta_path)
        self.prob = np.load(cfg.prob_path)
        self.avg = np.load(cfg.avg_npz_path) if (cfg.avg_npz_path and os.path.isfile(cfg.avg_npz_path)) else None
        self.lc_ds = rasterio.open(cfg.lc_raster_path)
        self.grid_affine = rasterio.Affine.from_gdal(*self.meta['grid_affine_transform_gdal'])
        self.grid_crs = self.meta['grid_crs']
        self.grid_h = int(self.meta['grid_dimensions']['height'])
        self.grid_w = int(self.meta['grid_dimensions']['width'])
        self.season_map = self.meta['context_mappings']['season']
        self.tod_map = self.meta['context_mappings']['time_of_day']
        lc_map = self.meta['context_mappings']['land_cover_code_to_index']
        self.lc_code_to_idx = {int(k): v for k, v in (lc_map.items() if isinstance(lc_map, dict) else lc_map)}

    @staticmethod
    def _load_json(path: str):
        with open(path, 'r') as f:
            return json.load(f)

    def _ctx_indices(self, month: int, hour: int) -> Tuple[Optional[int], Optional[int]]:
        season_name = None
        if month in (6, 7, 8): season_name = 'Summer'
        elif month in (5, 9): season_name = 'Spring/Fall'
        tod_name = None
        if 10 <= hour < 12: tod_name = 'Morning'
        elif 12 <= hour < 16: tod_name = 'Afternoon'
        elif 16 <= hour < 18: tod_name = 'Late Afternoon'
        s_idx = self.season_map.get(season_name)
        t_idx = self.tod_map.get(tod_name)
        return s_idx, t_idx

    def query_aoi(self,
                  center_xy_m: Tuple[float, float],
                  aoi_half_width_m: float,
                  now_month: int,
                  now_hour: int,
                  prob_threshold: float = 0.5) -> List[Tuple[Tuple[int,int], float, Tuple[float,float]]]:
        s_idx, t_idx = self._ctx_indices(now_month, now_hour)
        if s_idx is None or t_idx is None:
            return []

        cx, cy = center_xy_m
        min_x, max_x = cx - aoi_half_width_m, cx + aoi_half_width_m
        min_y, max_y = cy - aoi_half_width_m, cy + aoi_half_width_m

        try:
            r_min, c_min = rasterio.transform.rowcol(self.grid_affine, min_x, max_y)
            r_max, c_max = rasterio.transform.rowcol(self.grid_affine, max_x, min_y)
        except Exception:
            return []

        r_min = max(0, r_min); c_min = max(0, c_min)
        r_max = min(self.grid_h - 1, r_max); c_max = min(self.grid_w - 1, c_max)
        if r_min > r_max or c_min > c_max:
            return []

        out = []
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                x_m, y_m = self.grid_affine * (c + 0.5, r + 0.5)
                try:
                    lc_code = int(next(self.lc_ds.sample([(x_m, y_m)], indexes=1))[0])
                except Exception:
                    lc_code = 0
                if lc_code == self.lc_ds.nodata:
                    lc_code = 0
                lc_idx = self.lc_code_to_idx.get(lc_code)
                if lc_idx is None:
                    continue
                try:
                    p = float(self.prob[r, c, s_idx, t_idx, lc_idx])
                except Exception:
                    p = 0.0
                if p >= prob_threshold:
                    out.append(((r, c), p, (x_m, y_m)))
        return out 