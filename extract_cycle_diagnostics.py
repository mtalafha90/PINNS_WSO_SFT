#!/usr/bin/env python3
"""
Read field.npy from a trained PINN run and extract key cycle diagnostics.

Outputs:
- diagnostics_summary.txt
- diagnostics_summary.json
- dipole_1e22.npy
- polar_mean_north_gauss.npy
- polar_mean_south_gauss.npy
- polar_mean_signed_gauss.npy
- polar_mean_unsigned_gauss.npy
- hemisphere_asymmetry_gauss.npy

Assumptions:
- field.npy has shape (Nt, Nlat) in model units = Gauss / B_unit
- latitude grid is uniform in normalized latitude lam in [lam_min, lam_max]
- default cycle length is 11 years unless changed below
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np


# ----------------------------
# User settings
# ----------------------------
RESULTS_DIR = "results/cycle_24"
FIELD_FILE = "field.npy"

SIMUL_TIME_YEARS = 11.0
B_UNIT = 10.0
L_UNIT = 6.95e10

LAM_MIN = -0.495
LAM_MAX = 0.495

POLAR_LAT_DEG = 60.0


# ----------------------------
# Helpers
# ----------------------------
@dataclass
class Diagnostics:
    reversal_time_north_yr: float | None
    reversal_time_south_yr: float | None
    mean_hemispheric_reversal_time_yr: float | None

    peak_north_polar_field_G: float
    peak_south_polar_field_G: float
    peak_signed_polar_mean_G: float
    peak_unsigned_polar_mean_G: float

    time_peak_north_polar_field_yr: float
    time_peak_south_polar_field_yr: float
    time_peak_signed_polar_mean_yr: float
    time_peak_unsigned_polar_mean_yr: float

    dipole_max_abs_1e22: float
    dipole_time_max_abs_yr: float
    dipole_min_1e22: float
    dipole_time_min_yr: float
    dipole_max_1e22: float
    dipole_time_max_yr: float

    hemisphere_asymmetry_mean_G: float
    hemisphere_asymmetry_max_abs_G: float
    hemisphere_asymmetry_time_max_abs_yr: float


def compute_lat_grid(nlat: int, lam_min: float, lam_max: float) -> np.ndarray:
    lam = np.linspace(lam_min, lam_max, nlat)
    return lam * 180.0


def zero_crossing_time(t_years: np.ndarray, y: np.ndarray) -> float | None:
    y = np.asarray(y)
    s = np.sign(y)
    for i in range(len(y) - 1):
        if s[i] == 0:
            return float(t_years[i])
        if s[i] * s[i + 1] < 0:
            x0, x1 = t_years[i], t_years[i + 1]
            y0, y1 = y[i], y[i + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return None


def compute_dipole_moment(B_model: np.ndarray, lat_deg: np.ndarray, b_unit: float, l_unit: float) -> np.ndarray:
    """
    B_model shape: (Nt, Nlat) in model units
    returns dipole in 1e22 Mx cm
    """
    idx = np.argsort(lat_deg)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)[idx]
    B_model = np.asarray(B_model, dtype=np.float64)[:, idx]

    B_phys = B_model * float(b_unit)
    mu = np.sin(np.deg2rad(lat_deg))

    a1 = 1.5 * np.trapezoid(B_phys * mu[None, :], mu, axis=1)
    M = 0.5 * (float(l_unit) ** 3) * a1
    return M / 1e22


def save_text_summary(path: str, d: Diagnostics) -> None:
    with open(path, "w") as f:
        for k, v in asdict(d).items():
            f.write(f"{k}: {v}\n")


# ----------------------------
# Main
# ----------------------------
def main():
    field_path = os.path.join(RESULTS_DIR, FIELD_FILE)
    if not os.path.exists(field_path):
        raise FileNotFoundError(f"Could not find {field_path}")

    B_model = np.load(field_path)  # (Nt, Nlat)
    if B_model.ndim != 2:
        raise ValueError(f"Expected 2D field.npy, got shape {B_model.shape}")

    Nt, Nlat = B_model.shape
    lat_deg = compute_lat_grid(Nlat, LAM_MIN, LAM_MAX)
    time_years = np.linspace(0.0, SIMUL_TIME_YEARS, Nt)

    B_gauss = B_model * B_UNIT

    north_mask = lat_deg >= POLAR_LAT_DEG
    south_mask = lat_deg <= -POLAR_LAT_DEG
    polar_mask = np.abs(lat_deg) >= POLAR_LAT_DEG

    if not np.any(north_mask) or not np.any(south_mask):
        raise ValueError("Polar masks are empty. Check latitude grid / POLAR_LAT_DEG.")

    polar_mean_north = B_gauss[:, north_mask].mean(axis=1)
    polar_mean_south = B_gauss[:, south_mask].mean(axis=1)
    polar_mean_signed = B_gauss[:, polar_mask].mean(axis=1)
    polar_mean_unsigned = np.abs(B_gauss[:, polar_mask]).mean(axis=1)
    hemisphere_asymmetry = polar_mean_north - np.abs(polar_mean_south)
    dipole_1e22 = compute_dipole_moment(B_model, lat_deg, B_UNIT, L_UNIT)
    # Reversal times from hemispheres only
    reversal_north = zero_crossing_time(time_years, polar_mean_north)
    reversal_south = zero_crossing_time(time_years, polar_mean_south)
    # Optional representative single value
    if (reversal_north is not None) and (reversal_south is not None):
        mean_reversal = 0.5 * (reversal_north + reversal_south)
    else:
        mean_reversal = None
    # Peaks
    idx_peak_north = np.argmax(np.abs(polar_mean_north))
    idx_peak_south = np.argmax(np.abs(polar_mean_south))
    idx_peak_signed = np.argmax(np.abs(polar_mean_signed))
    idx_peak_unsigned = np.argmax(polar_mean_unsigned)

    idx_dip_abs = np.argmax(np.abs(dipole_1e22))
    idx_dip_min = np.argmin(dipole_1e22)
    idx_dip_max = np.argmax(dipole_1e22)

    idx_asym = np.argmax(np.abs(hemisphere_asymmetry))

    diagnostics = Diagnostics(
        reversal_time_north_yr=reversal_north,
        reversal_time_south_yr=reversal_south,
        mean_hemispheric_reversal_time_yr=mean_reversal,

        peak_north_polar_field_G=float(np.abs(polar_mean_north[idx_peak_north])),
        peak_south_polar_field_G=float(np.abs(polar_mean_south[idx_peak_south])),
        peak_signed_polar_mean_G=float(np.abs(polar_mean_signed[idx_peak_signed])),
        peak_unsigned_polar_mean_G=float(polar_mean_unsigned[idx_peak_unsigned]),

        time_peak_north_polar_field_yr=float(time_years[idx_peak_north]),
        time_peak_south_polar_field_yr=float(time_years[idx_peak_south]),
        time_peak_signed_polar_mean_yr=float(time_years[idx_peak_signed]),
        time_peak_unsigned_polar_mean_yr=float(time_years[idx_peak_unsigned]),

        dipole_max_abs_1e22=float(np.abs(dipole_1e22[idx_dip_abs])),
        dipole_time_max_abs_yr=float(time_years[idx_dip_abs]),
        dipole_min_1e22=float(dipole_1e22[idx_dip_min]),
        dipole_time_min_yr=float(time_years[idx_dip_min]),
        dipole_max_1e22=float(dipole_1e22[idx_dip_max]),
        dipole_time_max_yr=float(time_years[idx_dip_max]),

        hemisphere_asymmetry_mean_G=float(np.mean(hemisphere_asymmetry)),
        hemisphere_asymmetry_max_abs_G=float(np.max(np.abs(hemisphere_asymmetry))),
        hemisphere_asymmetry_time_max_abs_yr=float(time_years[idx_asym]),
    )

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "dipole_1e22.npy"), dipole_1e22)
    np.save(os.path.join(RESULTS_DIR, "polar_mean_north_gauss.npy"), polar_mean_north)
    np.save(os.path.join(RESULTS_DIR, "polar_mean_south_gauss.npy"), polar_mean_south)
    np.save(os.path.join(RESULTS_DIR, "polar_mean_signed_gauss.npy"), polar_mean_signed)
    np.save(os.path.join(RESULTS_DIR, "polar_mean_unsigned_gauss.npy"), polar_mean_unsigned)
    np.save(os.path.join(RESULTS_DIR, "hemisphere_asymmetry_gauss.npy"), hemisphere_asymmetry)

    # Save summaries
    save_text_summary(os.path.join(RESULTS_DIR, "diagnostics_summary.txt"), diagnostics)
    with open(os.path.join(RESULTS_DIR, "diagnostics_summary.json"), "w") as f:
        json.dump(asdict(diagnostics), f, indent=2)

    print("Diagnostics extracted successfully.\n")
    for k, v in asdict(diagnostics).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()