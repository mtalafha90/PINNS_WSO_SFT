#!/usr/bin/env python3
"""
Fill the OBSERVED (WSO) columns of Table 2 (observed polar-cap ends and
observed axial dipole) using the project's own get_wso_map_for_comparison,
so the observed map is processed exactly as in training (interpolated to the
model grid, monopole removed per time).

RUN FROM THE REPO ROOT (where src/extract.py and data/<cycle>/ live):

    python compute_obs_columns.py

Then paste me the printed table and I'll drop the numbers into Table 2.
"""
import numpy as np
from src.extract import get_wso_map_for_comparison   # project function

# --- diagnostics: identical definitions to the PINN side (cycle_diagnostics.py) ---
def _w(lat):                      # area weight, clipped like the project code
    return np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
def cap_mean(F, lat, lo, hi):
    m = (lat >= lo) & (lat <= hi); w = _w(lat)[m]
    return (F[:, m] * w).sum(1) / w.sum()
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
def axial_dipole(F, lat):         # D = 3/2 * int B mu dmu,  mu = sin(lat)
    mu = np.sin(np.deg2rad(lat)); o = np.argsort(mu)
    return 1.5 * _trapz(F[:, o] * mu[o], mu[o], axis=1)

CYCLES = {"21": "data/21", "22": "data/22", "23": "data/23", "24": "data/24"}
NR = 8        # end-of-cycle average window (final samples); matches the PINN side

print(f"{'cyc':>4} | {'obs N_end':>9} | {'obs S_end':>9} | {'obs D_end':>9}   [Gauss]")
print("-" * 46)
for c, d in CYCLES.items():
    # B_unit=1.0 and unit_to_gauss=0.01  ->  observed field returned in TRUE GAUSS
    t, lat, Bobs = get_wso_map_for_comparison(
        Tmax=1.0, lat_points=181, time_steps=400,
        B_unit=1.0, data_dir=d, unit_to_gauss=0.01)
    capN = cap_mean(Bobs, lat, 60, 90)
    capS = cap_mean(Bobs, lat, -90, -60)
    D    = axial_dipole(Bobs, lat)
    print(f"{c:>4} | {capN[-NR:].mean():+9.2f} | {capS[-NR:].mean():+9.2f} | {D[-NR:].mean():+9.2f}")

print("\nSanity check: Cycle 24 should print roughly  +0.67 / -0.41 / +0.30")
print("(the obs values already in Table 2). If it matches, 21-23 are good to use.")
print("If Cycle 24 is off by a constant factor, tell me and we adjust B_unit/unit_to_gauss.")