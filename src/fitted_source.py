import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class FittedSource:
    def __init__(
        self,
        source_file,
        simul_time,
        lam_min=-0.495,
        lam_max=0.495,
        source_scale=1.0,
        mask_poles_above_deg=None,
    ):
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")

        src = np.load(source_file)   # shape (Nt, Nlat)
        if src.ndim != 2:
            raise ValueError(f"Expected 2D source map, got shape {src.shape}")

        self.Nt, self.Nlat = src.shape
        self.simul_time = float(simul_time)

        self.time_grid = np.linspace(0.0, self.simul_time, self.Nt)
        self.lam_grid = np.linspace(lam_min, lam_max, self.Nlat)
        self.lat_deg_grid = self.lam_grid * 180.0

        if mask_poles_above_deg is not None:
            pole_mask = np.abs(self.lat_deg_grid) > float(mask_poles_above_deg)
            src[:, pole_mask] = 0.0

        src = source_scale * src
        self.src = src

        self.interp = RegularGridInterpolator(
            (self.time_grid, self.lam_grid),
            self.src,
            bounds_error=False,
            fill_value=0.0,
        )

    def eval_numpy(self, lam, t_year):
        lam = np.asarray(lam).reshape(-1)
        t_year = np.asarray(t_year).reshape(-1)

        if t_year.size == 1 and lam.size > 1:
            t_year = np.full_like(lam, float(t_year[0]), dtype=float)
        elif lam.size == 1 and t_year.size > 1:
            lam = np.full_like(t_year, float(lam[0]), dtype=float)

        pts = np.column_stack([t_year, lam])
        out = self.interp(pts)
        return out.reshape(-1, 1)