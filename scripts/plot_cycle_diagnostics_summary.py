#!/usr/bin/env python3
"""
Make one summary figure from the outputs of extract_cycle_diagnostics.py.

Expected files inside RESULTS_DIR:
- field.npy
- diagnostics_summary.json
- dipole_1e22.npy
- polar_mean_north_gauss.npy
- polar_mean_south_gauss.npy
- polar_mean_signed_gauss.npy
- polar_mean_unsigned_gauss.npy
- hemisphere_asymmetry_gauss.npy

Output:
- cycle_diagnostics_summary.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# User settings
# ----------------------------
RESULTS_DIR = "results/cycle_24"
SIMUL_TIME_YEARS = 11.0
B_UNIT = 10.0

LAM_MIN = -0.495
LAM_MAX = 0.495


# ----------------------------
# Helpers
# ----------------------------
def compute_lat_grid(nlat, lam_min=LAM_MIN, lam_max=LAM_MAX):
    lam = np.linspace(lam_min, lam_max, nlat)
    return lam * 180.0


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------
# Main
# ----------------------------
def main():
    # File paths
    field_path = os.path.join(RESULTS_DIR, "field.npy")
    dipole_path = os.path.join(RESULTS_DIR, "dipole_1e22.npy")
    north_path = os.path.join(RESULTS_DIR, "polar_mean_north_gauss.npy")
    south_path = os.path.join(RESULTS_DIR, "polar_mean_south_gauss.npy")
    signed_path = os.path.join(RESULTS_DIR, "polar_mean_signed_gauss.npy")
    unsigned_path = os.path.join(RESULTS_DIR, "polar_mean_unsigned_gauss.npy")
    asym_path = os.path.join(RESULTS_DIR, "hemisphere_asymmetry_gauss.npy")
    summary_path = os.path.join(RESULTS_DIR, "diagnostics_summary.json")

    # Check existence
    required = [
        field_path, dipole_path, north_path, south_path,
        signed_path, unsigned_path, asym_path, summary_path
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    # Load data
    B_model = np.load(field_path)                     # (Nt, Nlat), model units
    dipole = np.load(dipole_path)                    # (Nt,)
    polar_north = np.load(north_path)                # (Nt,)
    polar_south = np.load(south_path)                # (Nt,)
    polar_signed = np.load(signed_path)              # (Nt,)
    polar_unsigned = np.load(unsigned_path)          # (Nt,)
    asymmetry = np.load(asym_path)                   # (Nt,)
    summary = load_json(summary_path)

    Nt, Nlat = B_model.shape
    lat_deg = compute_lat_grid(Nlat)
    time_years = np.linspace(0.0, SIMUL_TIME_YEARS, Nt)

    B_gauss = B_model * B_UNIT

    # ----------------------------
    # Make summary figure
    # ----------------------------
    fig = plt.figure(figsize=(16, 10))

    # 1) Reconstructed magnetic field map
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(
        B_gauss.T,
        aspect="auto",
        origin="lower",
        extent=[time_years[0], time_years[-1], lat_deg[0], lat_deg[-1]],
        cmap="RdBu_r"
    )
    plt.colorbar(im, ax=ax1, label="Magnetic field [G]")
    ax1.set_title("Reconstructed Surface Magnetic Field")
    ax1.set_xlabel("Time [yr]")
    ax1.set_ylabel("Latitude [deg]")

    # 2) Dipole moment
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(time_years, dipole, linewidth=2)
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_title("Axial Dipole Moment")
    ax2.set_xlabel("Time [yr]")
    ax2.set_ylabel(r"Dipole [$10^{22}$ Mx cm]")

    # Mark dipole extrema
    t_dip_abs = summary.get("dipole_time_max_abs_yr", None)
    if t_dip_abs is not None:
        ax2.axvline(t_dip_abs, linestyle=":", linewidth=1)

    # 3) North and south polar-cap mean field
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_years, polar_north, label="North")
    ax3.plot(time_years, polar_south, label="South")
    ax3.axhline(0.0, linestyle="--", linewidth=1)

    rev_n = summary.get("reversal_time_north_yr", None)
    rev_s = summary.get("reversal_time_south_yr", None)
    if rev_n is not None:
        ax3.axvline(rev_n, linestyle=":", linewidth=1, label=f"N reversal: {rev_n:.2f}")
    if rev_s is not None:
        ax3.axvline(rev_s, linestyle="--", linewidth=1, label=f"S reversal: {rev_s:.2f}")

    ax3.set_title(r"Polar-cap Mean Field ($|\lambda| \geq 60^\circ$)")
    ax3.set_xlabel("Time [yr]")
    ax3.set_ylabel("Field [G]")
    ax3.legend(fontsize=9)

    # 4) Signed / unsigned polar mean field
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_years, polar_signed, label="Signed polar mean")
    ax4.plot(time_years, polar_unsigned, label="Unsigned polar mean")
    ax4.axhline(0.0, linestyle="--", linewidth=1)

    

    ax4.set_title("Signed and Unsigned Polar Mean")
    ax4.set_xlabel("Time [yr]")
    ax4.set_ylabel("Field [G]")
    ax4.legend(fontsize=9)

    # 5) Hemispheric asymmetry
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_years, asymmetry, linewidth=2)
    ax5.axhline(0.0, linestyle="--", linewidth=1)
    ax5.set_title("Hemispheric Asymmetry")
    ax5.set_xlabel("Time [yr]")
    ax5.set_ylabel(r"$B_{\rm N} - |B_{\rm S}|$ [G]")

    # 6) Text summary panel
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    text = (
        "Cycle Diagnostics Summary\n\n"
        f"North reversal time: {summary.get('reversal_time_north_yr', None)} yr\n"
        f"South reversal time: {summary.get('reversal_time_south_yr', None)} yr\n"
        f"Mean hemispheric reversal: {summary.get('mean_hemispheric_reversal_time_yr', None)} yr\n\n"
        f"Peak north polar field: {summary.get('peak_north_polar_field_G', None):.3f} G\n"
        f"Peak south polar field: {summary.get('peak_south_polar_field_G', None):.3f} G\n"
        f"Peak signed polar mean: {summary.get('peak_signed_polar_mean_G', None):.3f} G\n"
        f"Peak unsigned polar mean: {summary.get('peak_unsigned_polar_mean_G', None):.3f} G\n\n"
        f"Dipole max |M|: {summary.get('dipole_max_abs_1e22', None):.3f}\n"
        f"Time of max |M|: {summary.get('dipole_time_max_abs_yr', None):.3f} yr\n"
        f"Dipole min: {summary.get('dipole_min_1e22', None):.3f}\n"
        f"Dipole max: {summary.get('dipole_max_1e22', None):.3f}\n\n"
        f"Mean asymmetry: {summary.get('hemisphere_asymmetry_mean_G', None):.3f} G\n"
        f"Max |asymmetry|: {summary.get('hemisphere_asymmetry_max_abs_G', None):.3f} G\n"
        f"Time of max |asymmetry|: {summary.get('hemisphere_asymmetry_time_max_abs_yr', None):.3f} yr"
    )

    ax6.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95)
    )

    plt.tight_layout()
    outpath = os.path.join(RESULTS_DIR, "cycle_diagnostics_summary.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved summary figure to: {outpath}")


if __name__ == "__main__":
    main()