"""
plot_residual_diagnostics.py -- Residual latitude structure + polar RMS figure.

Produces a two-panel figure matching the paper style:
  (a) time-mean residual ΔBr vs latitude (lat on y-axis), ±1σ envelope,
      dotted lines at the WSO observation limit (±75°)
  (b) N-cap and S-cap RMS residual vs time, reversal epoch shaded

Run from repo root:
    python scripts/plot_residual_diagnostics.py        # default: cycle 24
    python scripts/plot_residual_diagnostics.py 21     # specific cycle
Output: results/cycle_NN/residual_diagnostics.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.extract import get_wso_map_for_comparison
from src import cycle_tools as ct

HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_UNIT    = 10.0
TO_GAUSS  = B_UNIT * 0.01          # model units -> true Gauss
NUM_LATS  = 181
NT_POINTS = 400
LAT_DEG   = np.linspace(-0.495, 0.495, NUM_LATS) * 180.0
LENGTHS   = json.load(open(os.path.join(HERE, "cycle_products", "cycle_lengths.json")))


def plot_cycle(cycle):
    out = os.path.join(HERE, "results", f"cycle_{cycle}")
    fp  = os.path.join(out, "field.npy")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"No field.npy found in {out}")

    T       = float(LENGTHS[str(cycle)])
    B       = np.load(fp)                           # (Nt, Nlat) model units
    t_years = np.linspace(0.0, T, B.shape[0])

    # WSO comparison on the same grid
    _, _, B_obs = get_wso_map_for_comparison(
        Tmax=1.0, lat_points=NUM_LATS, time_steps=NT_POINTS + 1,
        B_unit=B_UNIT, data_dir=os.path.join(HERE, "data", str(cycle)))

    Nt  = min(B.shape[0], B_obs.shape[0])
    Nl  = min(B.shape[1], B_obs.shape[1])
    B, B_obs = B[:Nt, :Nl], B_obs[:Nt, :Nl]
    lat = LAT_DEG[:Nl]
    ty  = t_years[:Nt]

    Bg  = B     * TO_GAUSS      # PINN in true Gauss
    Og  = B_obs * TO_GAUSS      # WSO  in true Gauss
    res = Bg - Og

    # --- panel (a): latitude residual profile ---
    res_mean = res.mean(0)
    res_std  = res.std(0)

    # --- panel (b): N/S polar-cap RMS vs time ---
    ncap = lat >= 70.0
    scap = lat <= -70.0
    rms_N = np.sqrt((res[:, ncap] ** 2).mean(1))
    rms_S = np.sqrt((res[:, scap] ** 2).mean(1))

    # Reversal epoch: last zero crossing of smoothed obs polar caps
    store = pickle.load(open(os.path.join(HERE, "cycle_products", "store.pkl"), "rb"))
    s = store[cycle]
    on_s = ct.smooth1d(ct.polar_means(s["obs"] * ct.B_UNIT)[0])
    os_s = ct.smooth1d(ct.polar_means(s["obs"] * ct.B_UNIT)[1])
    revN = ct.last_crossing(s["t_obs"], on_s)
    revS = ct.last_crossing(s["t_obs"], os_s)
    rev_times = [v for v in [revN, revS] if v is not None]
    t_rev_lo  = min(rev_times) if rev_times else None
    t_rev_hi  = max(rev_times) if rev_times else None

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) lat on y-axis, residual mean on x-axis
    ax = axes[0]
    ax.fill_betweenx(lat, res_mean - res_std, res_mean + res_std,
                     color="gray", alpha=0.35, label=r"$\pm1\sigma$")
    ax.plot(res_mean, lat, "k", lw=1.6, label="mean")
    ax.axvline(0, color="gray", lw=0.6)
    for L in (-75, 75):
        ax.axhline(L, ls=":", color="gray", lw=1.0)
    ax.set_xlabel(r"$\Delta B_r = B_r^{\rm PINN} - B_r^{\rm WSO}$ [G]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("(a)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(lat[0], lat[-1])

    # (b) N/S cap RMS vs time with reversal shading
    ax = axes[1]
    ax.plot(ty, rms_N, color="C1",   lw=1.2,        label=r"N cap ($\lambda > 70^{\circ}$)")
    ax.plot(ty, rms_S, color="C3",   lw=1.2, ls="--", label=r"S cap ($\lambda < -70^{\circ}$)")
    if t_rev_lo is not None and t_rev_hi is not None:
        ax.axvspan(t_rev_lo, t_rev_hi, color="gray", alpha=0.25)
        ax.text((t_rev_lo + t_rev_hi) / 2, 0.04,
                "reversal", ha="center", va="bottom",
                fontsize=8, color="0.45", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("RMS residual [G]")
    ax.set_title("(b)")
    ax.set_xlim(ty[0], ty[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    plt.tight_layout()
    outfig = os.path.join(out, "residual_diagnostics.png")
    plt.savefig(outfig, dpi=150, bbox_inches="tight")
    print(f"Saved: {outfig}")


if __name__ == "__main__":
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    plot_cycle(cycle)
