# thermals/probmap/runtime_adapter.py
import json, numpy as np, rasterio
from dataclasses import dataclass
from typing import Optional, Tuple
from rasterio.transform import rowcol

@dataclass
class ThermalGuess:
    center_xy: Tuple[float,float]
    radius_m: float
    strength_mps: float
    top_height_m: float
    confidence: float   # use probability [0..1]

class ThermalProbRuntime:
    def __init__(self, meta_path: str, prob_map_path: str, avg_npz_path: str, lc_raster_path: str):
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.prob = np.load(prob_map_path)
        self.avg  = np.load(avg_npz_path)
        self.affine = rasterio.Affine.from_gdal(*self.meta["grid_affine_transform_gdal"])
        self.H = self.meta["grid_dimensions"]["height"]
        self.W = self.meta["grid_dimensions"]["width"]
        self.grid_crs = self.meta["grid_crs"]
        self.season_map = self.meta["context_mappings"]["season"]
        self.tod_map    = self.meta["context_mappings"]["time_of_day"]
        # json keys are strings → make them int
        self.lc_code_to_idx = {int(k): v for k, v in self.meta["context_mappings"]["land_cover_code_to_index"].items()}
        self.lc = rasterio.open(lc_raster_path)

    @staticmethod
    def _season_idx(month:int):
        if month in (6,7,8): return "Summer"
        if month in (5,9):   return "Spring/Fall"
        return None

    @staticmethod
    def _tod_idx(hour:int):
        if 10 <= hour < 12: return "Morning"
        if 12 <= hour < 16: return "Afternoon"
        if 16 <= hour < 18: return "Late Afternoon"
        return None

    def _xy_from_rc(self, r:int, c:int):
        x, y = self.affine * (c + 0.5, r + 0.5)
        return float(x), float(y)

    def query(self,
              agent_xy, month:int, hour:int,
              battery_pct: float,
              aoi_half_width_m: float,
              prob_thresh: float,
              mask_contains_xy=None  # callable(x,y)->bool, optional polygon mask
              ) -> Optional[ThermalGuess]:
        # Context
        s_name = self._season_idx(month); t_name = self._tod_idx(hour)
        if s_name is None or t_name is None:
            return None
        s_idx = self.season_map[s_name]
        t_idx = self.tod_map[t_name]

        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        r_min, c_min = rowcol(self.affine, ax - aoi_half_width_m, ay + aoi_half_width_m)
        r_max, c_max = rowcol(self.affine, ax + aoi_half_width_m, ay - aoi_half_width_m)
        r_min, r_max = max(0, min(r_min, r_max)), min(self.H-1, max(r_min, r_max))
        c_min, c_max = max(0, min(c_min, c_max)), min(self.W-1, max(c_min, c_max))
        if r_min > r_max or c_min > c_max:
            return None

        local = np.zeros((r_max-r_min+1, c_max-c_min+1), dtype=np.float32)
        coords = []
        for rg in range(r_min, r_max+1):
            for cg in range(c_min, c_max+1):
                x, y = self._xy_from_rc(rg, cg)
                if mask_contains_xy is not None and not mask_contains_xy(x, y):
                    coords.append((rg, cg, x, y)); continue
                try:
                    lc_code = next(self.lc.sample([(x,y)], indexes=1))[0]
                except Exception:
                    lc_code = 0
                if lc_code == self.lc.nodata: lc_code = 0
                lc_idx = self.lc_code_to_idx.get(int(lc_code))
                p = float(self.prob[rg, cg, s_idx, t_idx, lc_idx]) if lc_idx is not None else 0.0
                local[rg-r_min, cg-c_min] = p
                coords.append((rg, cg, x, y))

        s = local.sum()
        if s <= 1e-9:
            return None

        # Choose argmax (deterministic) – simpler for tests; swap to categorical if you prefer MC
        r_loc, c_loc = np.unravel_index(np.argmax(local), local.shape)
        rg, cg = r_min + r_loc, c_min + c_loc
        x, y = self._xy_from_rc(rg, cg)

        # Recheck land-cover for the chosen cell
        try:
            lc_code = next(self.lc.sample([(x,y)], indexes=1))[0]
        except Exception:
            lc_code = 0
        if lc_code == self.lc.nodata: lc_code = 0
        lc_idx = self.lc_code_to_idx.get(int(lc_code))
        if lc_idx is None:
            return None

        p_final = float(self.prob[rg, cg, s_idx, t_idx, lc_idx])
        if p_final < prob_thresh:
            return None

        lift = float(self.avg["avg_lift"][rg, cg, s_idx, t_idx, lc_idx])
        rad  = float(self.avg["avg_radius"][rg, cg, s_idx, t_idx, lc_idx])
        top  = float(self.avg["avg_height"][rg, cg, s_idx, t_idx, lc_idx])

        return ThermalGuess(
            center_xy=(x,y),
            radius_m=rad if np.isfinite(rad) else 150.0,
            strength_mps=lift if np.isfinite(lift) else 2.0,
            top_height_m=top if np.isfinite(top) else 1500.0,
            confidence=max(0.0, min(1.0, p_final))
        )
