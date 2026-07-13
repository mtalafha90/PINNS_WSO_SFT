"""
plot_hindcast_dipole.py -- Leave-one-out analogue-ensemble dipole hindcast figure.

For each completed cycle (SC21-24):
  - assimilates WSO up to PHASE_TRUNC * T
  - drives three analogue forward runs (the other three cycles)
  - plots assimilated WSO (solid black), actual WSO (dashed gray),
    individual analogue members (orange), shaded envelope, and a star
    at the actual end-of-cycle dipole

Run from repo root:
    python scripts/plot_hindcast_dipole.py
Output: cycle_products/hindcast_dipole.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import cycle_tools as ct

HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT        = os.path.join(HERE, "cycle_products")
store      = pickle.load(open(os.path.join(OUT, "store.pkl"), "rb"))

PHASE_TRUNC = 5.08 / 11.0          # ~0.462
NPHASE      = 401
phase       = np.linspace(0.0, 1.0, NPHASE)
BELT        = np.abs(ct.LAT_DEG) < 50.0
COMPLETE    = [21, 22, 23, 24]
TO_GAUSS    = 0.01                  # native units -> Gauss


def dipole_G(B_native):
    return ct.dipole(B_native) * TO_GAUSS


def polarity(c):
    return np.sign(dipole_G(store[c]["obs"][:5].mean(0, keepdims=True) * ct.B_UNIT)[0])


def phase_source(c):
    S, t_u, T = store[c]["S"], store[c]["t_u"], store[c]["T"]
    out = np.empty((NPHASE, ct.N))
    for j in range(ct.N):
        out[:, j] = np.interp(phase, t_u / T, S[:, j])
    return out


def amp_win(Sp, p0, p1):
    m = (phase >= p0) & (phase <= p1)
    return np.sqrt(np.mean(Sp[np.ix_(m, BELT)] ** 2))


# Averaging window (yr) for the launch state; 0 -> endpoint state.
# The hindcasts keep the endpoint state, unlike the Cycle-25 forecast
# (plot_forecast_comparison.py, annual mean): a source-free relaxation test
# shows the truncation states of cycles 21-24 are clean, and averaging them
# only adds a half-year lag bias that weakens the envelopes.
IC_ANNUAL_MEAN_YR = 0.0


def truncated_refit(c, t_tr):
    t_obs, obs = store[c]["t_obs"], store[c]["obs"]
    m = t_obs <= t_tr
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs[m], obs[m], t_tr, nt=201)
    if IC_ANNUAL_MEAN_YR > 0:
        ic0 = obs_s[t_u >= t_tr - IC_ANNUAL_MEAN_YR].mean(0)
    else:
        ic0 = obs_s[-1]
    return t_u, obs_s, ct.refit_source(t_u, obs_s), ic0


def analogue_member(target, analogue, T, t_u, ic0, S_tr):
    Sp  = phase_source(analogue) * (polarity(target) * polarity(analogue))
    p1  = t_u[-1] / T
    m_tr = np.sqrt(np.mean(S_tr[:, BELT] ** 2))
    sc  = m_tr / amp_win(Sp, 0.0, p1)
    t_m, B = ct.forward(ic0, t_u[-1], T, Sp * sc, phase * T, nt_out=160)
    return t_m, B * ct.B_UNIT


# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=False)
C_MEM = "#d1751b"   # orange for analogue members / envelope

for ax, c in zip(axes.flat, COMPLETE):
    T     = store[c]["T"]
    t_tr  = PHASE_TRUNC * T
    t_obs = store[c]["t_obs"]
    obs   = store[c]["obs"]

    t_u, obs_s, S_tr, ic0 = truncated_refit(c, t_tr)

    # assimilated WSO dipole (up to truncation)
    D_assim = dipole_G(obs[t_obs <= t_tr] * ct.B_UNIT)
    t_assim = t_obs[t_obs <= t_tr]

    # full WSO dipole for the "actual" curve (dashed gray)
    D_full  = dipole_G(obs * ct.B_UNIT)

    # D_end in forward-model convention: full-cycle FD run with the full source,
    # same computation as the analogue members → star and envelope are comparable
    t_full, B_full = ct.forward(store[c]["obs_s"][0], 0.0, T,
                                store[c]["S"], store[c]["t_u"])
    D_end = float(dipole_G(B_full * ct.B_UNIT)[-6:].mean())

    # analogue members
    member_D, member_t = [], []
    for a in COMPLETE:
        if a == c:
            continue
        t_m, Bn = analogue_member(c, a, T, t_u, ic0, S_tr)
        D_m = dipole_G(Bn)
        member_t.append(t_m)
        member_D.append(D_m)

    # envelope
    t_fc  = member_t[0]   # all members share the same forecast time grid
    stack = np.vstack([np.interp(t_fc, mt, md) for mt, md in zip(member_t, member_D)])
    env_lo, env_hi = stack.min(0), stack.max(0)

    ax.fill_between(t_fc, env_lo, env_hi, color=C_MEM, alpha=0.25, label="analogue envelope")
    for mt, md in zip(member_t, member_D):
        ax.plot(mt, md, color=C_MEM, alpha=0.7, lw=1.2, label="_")
    # add one labelled member line for the legend
    ax.plot([], [], color=C_MEM, lw=1.2, label="analogue members")

    ax.plot(t_assim, D_assim, "k",   lw=1.8, label="WSO (assimilated)")
    ax.plot(t_obs,   D_full,  "k--", lw=1.2, alpha=0.6, label="WSO (actual)")
    ax.plot(T, D_end, "k*", ms=12, zorder=5, label=r"actual $D^{\rm end}$")

    ax.axvline(t_tr, color="gray", ls=":", lw=1.1)
    ax.axhline(0,    color="gray", lw=0.5)

    ax.set_title(f"SC{c}: analogue dipole hindcast after {t_tr:.1f} yr", fontsize=10)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Axial dipole $D$ [G]")
    ax.set_xlim(0, T)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=C_MEM, alpha=0.25, label="analogue envelope"),
    Line2D([0], [0], color=C_MEM,  lw=1.2,        label="analogue members"),
    Line2D([0], [0], color="k",    lw=1.8,         label="WSO (assimilated)"),
    Line2D([0], [0], color="k",    lw=1.2, ls="--", alpha=0.6, label="WSO (actual)"),
    Line2D([0], [0], color="k",    marker="*", ms=10, ls="none", label=r"actual $D^{\rm end}$"),
]
axes.flat[0].legend(handles=legend_elements, fontsize=7.5, loc="lower left")

plt.tight_layout()
outfig = os.path.join(OUT, "hindcast_dipole.png")
plt.savefig(outfig, dpi=150, bbox_inches="tight")
print(f"Saved: {outfig}")
