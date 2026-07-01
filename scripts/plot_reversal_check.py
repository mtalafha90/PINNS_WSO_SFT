"""
plot_reversal_check.py -- Standalone reversal diagnostic for SC21-24.

Loads cycle_products/store.pkl, re-runs the FD forward model for each
cycle, smooths obs and model polar caps identically, then detects the
last zero crossing (= reversal time) for the north and south poles
independently.  Produces:
  - cycle_products/reversal_check.png  (4-panel polar-cap plot)
  - printed summary table

Run from the repo root:
    python scripts/plot_reversal_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import cycle_tools as ct

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(HERE, "cycle_products")
CYCLES = [21, 22, 23, 24]

store = pickle.load(open(os.path.join(OUT, "store.pkl"), "rb"))


def _fmt(v):
    return f"{v:.2f}" if v is not None else " none"


print(f"\n{'cyc':>4} | {'revN obs':>8} {'revN mod':>8} | {'revS obs':>8} {'revS mod':>8}")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=False)

for ax, c in zip(axes.flat, CYCLES):
    s = store[c]
    t_obs, obs, T = s["t_obs"], s["obs"], s["T"]
    t_u,  obs_s,  S = s["t_u"], s["obs_s"], s["S"]

    # --- forward model ---
    t_m, B = ct.forward(obs_s[0], 0.0, T, S, t_u)
    Bn = B * ct.B_UNIT

    # --- polar means ---
    obs_n, obs_sd = ct.polar_means(obs * ct.B_UNIT)
    mod_n, mod_sd = ct.polar_means(Bn)

    # --- smooth identically before reversal detection ---
    on_s = ct.smooth1d(obs_n)
    os_s = ct.smooth1d(obs_sd)
    mn_s = ct.smooth1d(mod_n)
    ms_s = ct.smooth1d(mod_sd)

    # --- last zero crossing = reversal time ---
    revN_obs = ct.last_crossing(t_obs, on_s)
    revS_obs = ct.last_crossing(t_obs, os_s)
    revN_mod = ct.last_crossing(t_m,   mn_s)
    revS_mod = ct.last_crossing(t_m,   ms_s)

    print(f"  {c:>2} | {_fmt(revN_obs):>8} {_fmt(revN_mod):>8} | "
          f"{_fmt(revS_obs):>8} {_fmt(revS_mod):>8}")

    # --- plot ---
    ax.plot(t_obs, obs_n,  "k",   lw=1.0, alpha=0.55, label="WSO N")
    ax.plot(t_obs, obs_sd, "k--", lw=1.0, alpha=0.55, label="WSO S")
    ax.plot(t_m,   mod_n,  "C1",  lw=1.5, label="FD N")
    ax.plot(t_m,   mod_sd, "C1--",lw=1.5, label="FD S")
    ax.axhline(0, color="gray", lw=0.6)

    # reversal markers: solid = north, dashed = south
    for tv, col, ls, lbl in [
        (revN_obs, "k",  "-",  f"revN obs {_fmt(revN_obs)} yr"),
        (revS_obs, "k",  "--", f"revS obs {_fmt(revS_obs)} yr"),
        (revN_mod, "C1", "-",  f"revN FD  {_fmt(revN_mod)} yr"),
        (revS_mod, "C1", "--", f"revS FD  {_fmt(revS_mod)} yr"),
    ]:
        if tv is not None:
            ax.axvline(tv, color=col, ls=ls, lw=1.1, alpha=0.8, label=lbl)

    ax.set_title(f"SC{c}  (T = {T:.2f} yr)", fontsize=10)
    ax.set_xlabel("years since cycle start")
    ax.set_ylabel("polar-cap mean [native units]")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")

print()
plt.suptitle("Polar-cap reversal check: last zero crossing (obs = black, FD = orange)\n"
             "solid = north pole, dashed = south pole", fontsize=9, y=1.01)
plt.tight_layout()

out_fig = os.path.join(OUT, "reversal_check.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Saved: {out_fig}")
