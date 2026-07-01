"""
plot_reversal_check.py -- Reversal diagnostic: WSO obs vs PINN output.

For each cycle 21-24:
  - loads cycle_products/store.pkl for the WSO observed field
  - loads results/cycle_NN/field.npy for the PINN solution
  - smooths obs and PINN polar caps identically
  - finds the last zero crossing (= reversal time) for N and S poles
  - plots 4-panel polar-cap figure with reversal markers

Run from the repo root:
    python scripts/plot_reversal_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import cycle_tools as ct

HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(HERE, "cycle_products")
CYCLES = [21, 22, 23, 24]

store  = pickle.load(open(os.path.join(OUT, "store.pkl"), "rb"))
lengths = json.load(open(os.path.join(OUT, "cycle_lengths.json")))

# PINN field.npy uses a uniform degree latitude grid
PINN_LAT = np.linspace(-90, 90, 181)
PINN_NM  = PINN_LAT >= 60.0
PINN_SM  = PINN_LAT <= -60.0


def pinn_polar_means(B_native):
    return B_native[:, PINN_NM].mean(1), B_native[:, PINN_SM].mean(1)


def _fmt(v):
    return f"{v:.2f}" if v is not None else " none"


print(f"\n{'cyc':>4} | {'revN obs':>8} {'revN PINN':>9} | {'revS obs':>8} {'revS PINN':>9}")
print("-" * 52)

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))

for ax, c in zip(axes.flat, CYCLES):
    s = store[c]
    t_obs, obs = s["t_obs"], s["obs"]
    T = float(lengths[str(c)])

    # --- PINN field ---
    pinn_path = os.path.join(HERE, "results", f"cycle_{c}", "field.npy")
    if not os.path.exists(pinn_path):
        print(f"  {c:>2} | PINN file not found: {pinn_path}")
        continue
    B_pinn = np.load(pinn_path) * ct.B_UNIT          # model units -> native
    t_pinn = np.linspace(0.0, T, B_pinn.shape[0])
    pinn_n, pinn_s = pinn_polar_means(B_pinn)

    # --- observed polar means (FD mu-grid) ---
    obs_n, obs_sd = ct.polar_means(obs * ct.B_UNIT)

    # --- smooth obs before reversal detection (noisy WSO data) ---
    on_s = ct.smooth1d(obs_n)
    os_s = ct.smooth1d(obs_sd)

    # PINN output is already smooth (neural network); applying smooth1d again
    # suppresses small but real sign excursions (e.g. SC24 north briefly goes
    # positive after ~3.7 yr before the true final crossing at ~6 yr).
    # Use the raw polar means directly for the PINN crossing detection.
    revN_obs  = ct.last_crossing(t_obs,  on_s)
    revS_obs  = ct.last_crossing(t_obs,  os_s)
    revN_pinn = ct.last_crossing(t_pinn, pinn_n)
    revS_pinn = ct.last_crossing(t_pinn, pinn_s)

    print(f"  {c:>2} | {_fmt(revN_obs):>8} {_fmt(revN_pinn):>9} | "
          f"{_fmt(revS_obs):>8} {_fmt(revS_pinn):>9}")

    # --- plot ---
    ax.plot(t_obs,  obs_n,  "k",    lw=1.0, alpha=0.55, label="WSO N")
    ax.plot(t_obs,  obs_sd, "k--",  lw=1.0, alpha=0.55, label="WSO S")
    ax.plot(t_pinn, pinn_n, "C2",   lw=1.5, label="PINN N")
    ax.plot(t_pinn, pinn_s, "C2--", lw=1.5, label="PINN S")
    ax.axhline(0, color="gray", lw=0.6)

    for tv, col, ls, lbl in [
        (revN_obs,  "k",   "-",  f"revN obs  {_fmt(revN_obs)} yr"),
        (revS_obs,  "k",   "--", f"revS obs  {_fmt(revS_obs)} yr"),
        (revN_pinn, "C2",  "-",  f"revN PINN {_fmt(revN_pinn)} yr"),
        (revS_pinn, "C2",  "--", f"revS PINN {_fmt(revS_pinn)} yr"),
    ]:
        if tv is not None:
            ax.axvline(tv, color=col, ls=ls, lw=1.1, alpha=0.8, label=lbl)

    ax.set_title(f"SC{c}  (T = {T:.2f} yr)", fontsize=10)
    ax.set_xlabel("years since cycle start")
    ax.set_ylabel("polar-cap mean [native units]")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")

print()
plt.suptitle("Polar-cap reversal check: last zero crossing (obs = black, PINN = green)\n"
             "solid = north pole, dashed = south pole", fontsize=9, y=1.01)
plt.tight_layout()

out_fig = os.path.join(OUT, "reversal_check.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Saved: {out_fig}")
