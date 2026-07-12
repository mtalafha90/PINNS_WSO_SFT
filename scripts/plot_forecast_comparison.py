import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
plot_forecast_comparison.py  (v2 -- band style, PINN-prominent)
---------------------------------------------------------------
Overlay the PINN continuation and the finite-volume (FD) analog-ensemble
Cycle-25 forecasts on ONE figure, using a SINGLE dipole/polar-cap operator
for both engines.

  * FD ensemble  : analog members from cycle_products/store.pkl (cycle_tools.forward)
  * PINN ensemble: results/forecast_cycle25_mem_*/field.npy (forecast_cycle25_pinn.py)

Both fields are reduced to MODEL units, then the same operator gives Gauss:
    cap = mean_{|lat|>=60} B_model * 0.1 ;  D = 1.5 \\int B_model mu dmu * 0.1.

Run from the repo root:  python plot_forecast_comparison.py
"""
import os, glob, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import cycle_tools as ct

HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT     = os.path.join(HERE, "cycle_products")
RESULTS = os.path.join(HERE, "results")
store   = pickle.load(open(os.path.join(OUT, "store.pkl"), "rb"))

YR0      = 2019.94
TO_GAUSS = 0.1
# Forecast initial state = mean assimilated state over the final
# IC_ANNUAL_MEAN_YR years instead of the endpoint state (0 -> endpoint).
# The WSO polar-cap signal carries an annual polar-visibility (b-angle)
# modulation, and the final rotations leave a spurious low-latitude flux
# imbalance in the endpoint state; transported poleward it flips the polar
# caps back to the old polarity around 2027.  Averaging the state over one
# full modulation period removes the artifact without changing the data,
# the fitted source, or the trained members: because the SFT operator is
# linear, the same correction is applied to the PINN members by subtracting
# a single zero-source run of the IC difference profile.
IC_ANNUAL_MEAN_YR = 1.0
NPHASE   = 401
phase    = np.linspace(0.0, 1.0, NPHASE)
BELT     = np.abs(ct.LAT_DEG) < 50.0
COMPLETE = [21, 22, 23, 24]
FD_LENGTHS, FD_AMPS = [10.5, 11.0, 11.5], [0.85, 1.0, 1.15]
C_FD, C_PI = "#d1751b", "#1f6fb2"      # FD = orange, PINN = blue

def caps_G(B_model, lat_deg):
    return (B_model[:, lat_deg >= 60.0].mean(1) * TO_GAUSS,
            B_model[:, lat_deg <= -60.0].mean(1) * TO_GAUSS)

def dipole_G(B_model, lat_deg):
    mu = np.sin(np.deg2rad(lat_deg)); o = np.argsort(mu)
    return 1.5 * np.trapezoid(B_model[:, o] * mu[o], mu[o], axis=1) * TO_GAUSS

# ---------------- FD analog ensemble ----------------
def polarity(c):
    return np.sign(ct.dipole(store[c]["obs"][:5].mean(0, keepdims=True) * ct.B_UNIT)[0])
def phase_source(c):
    S, t_u, T = store[c]["S"], store[c]["t_u"], store[c]["T"]
    out = np.empty((NPHASE, ct.N))
    for j in range(ct.N):
        out[:, j] = np.interp(phase, t_u / T, S[:, j])
    return out
def amp_win(Sp, p0, p1):
    m = (phase >= p0) & (phase <= p1); return np.sqrt(np.mean(Sp[np.ix_(m, BELT)] ** 2))

def build_fd_ensemble():
    t_obs, obs = store[25]["t_obs"], store[25]["obs"]
    T_data = float(t_obs[-1])
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs, obs, T_data, nt=201)
    S_fit = ct.refit_source(t_u, obs_s)
    m_tr = np.sqrt(np.mean(S_fit[:, BELT] ** 2))

    # Hindcast: FD from t=0 to T_data with the fitted SC25 source (same for all members)
    t_hc, B_hc = ct.forward(obs_s[0], 0.0, T_data, S_fit, t_u, nt_out=80)
    N_hc, S_hc = caps_G(B_hc, ct.LAT_DEG)
    D_hc = dipole_G(B_hc, ct.LAT_DEG)
    t_hc_yr = YR0 + t_hc

    # forecast initial state: annual-mean assimilated state (see IC_ANNUAL_MEAN_YR)
    if IC_ANNUAL_MEAN_YR > 0:
        ic0 = obs_s[t_u >= T_data - IC_ANNUAL_MEAN_YR].mean(0)
    else:
        ic0 = obs_s[-1]

    members = []
    for a in COMPLETE:
        Sp0 = phase_source(a) * (polarity(25) * polarity(a))
        for T25 in FD_LENGTHS:
            p1 = T_data / T25
            for f in FD_AMPS:
                sc = f * m_tr / amp_win(Sp0, 0.0, p1)
                t_m, B = ct.forward(ic0, T_data, T25, Sp0 * sc, phase * T25, nt_out=160)
                N, S = caps_G(B, ct.LAT_DEG)
                # Concatenate hindcast + forecast, dropping the duplicate T_data point
                members.append(dict(
                    t=np.concatenate([t_hc_yr, YR0 + t_m[1:]]),
                    N=np.concatenate([N_hc, N[1:]]),
                    S=np.concatenate([S_hc, S[1:]]),
                    D=np.concatenate([D_hc, dipole_G(B, ct.LAT_DEG)[1:]])
                ))
    return members, T_data, ic0

def ic_transient(dIC, T_data, t_end):
    """Zero-source FD run of the IC difference profile.  By linearity of the
    SFT operator, subtracting its caps/dipole series from a member is exactly
    equivalent to having started that member from the corrected initial
    state -- this is how the correction is applied to the PINN members,
    which cannot be re-initialised at plot time."""
    t_c, B_c = ct.forward(dIC, T_data, t_end, np.zeros((NPHASE, ct.N)),
                          np.linspace(T_data, t_end, NPHASE), nt_out=200)
    N_c, S_c = caps_G(B_c, ct.LAT_DEG)
    return t_c, N_c, S_c, dipole_G(B_c, ct.LAT_DEG)

# ---------------- PINN ensemble ----------------
def build_pinn_ensemble(T_data_ref=None, ic0=None):
    members, tags, stale = [], [], []
    for d in sorted(glob.glob(os.path.join(RESULTS, "forecast_cycle25_mem_*"))):
        fp, mp = os.path.join(d, "field.npy"), os.path.join(d, "member_meta.json")
        if not (os.path.exists(fp) and os.path.exists(mp)):
            continue
        meta = json.load(open(mp))
        # skip members trained with a different data horizon (e.g. before a
        # TRIM_YR change) -- mixing horizons silently corrupts the ensemble
        if T_data_ref is not None and \
                abs(float(meta.get("T_data", T_data_ref)) - T_data_ref) > 0.1:
            stale.append(os.path.basename(d).replace("forecast_cycle25_mem_", ""))
            continue
        B = np.load(fp)
        lat_deg = np.array(meta["lat_deg"]) if "lat_deg" in meta \
            else np.linspace(-89.1, 89.1, B.shape[1])
        t = YR0 + np.linspace(0.0, float(meta["T_full"]), B.shape[0])
        N, S = caps_G(B, lat_deg)
        D = dipole_G(B, lat_deg)
        if ic0 is not None:
            # Re-initialise this member from the annual-mean state: subtract
            # the zero-source transient of (member's own horizon profile -
            # ic0).  By linearity of the SFT operator this equals having
            # started the member from ic0, and also cancels the member's own
            # imprint of the artifact-affected final maps.
            t_rel = t - YR0
            prof = np.array([np.interp(T_data_ref, t_rel, B[:, k])
                             for k in range(B.shape[1])])
            dIC_m = np.interp(ct.LAT_DEG, lat_deg, prof) - ic0
            t_c, dN, dS, dD = ic_transient(dIC_m, T_data_ref, float(t_rel[-1]))
            f = t_rel >= T_data_ref
            N[f] -= np.interp(t_rel[f], t_c, dN)
            S[f] -= np.interp(t_rel[f], t_c, dS)
            D[f] -= np.interp(t_rel[f], t_c, dD)
        members.append(dict(t=t, N=N, S=S, D=D))
        tags.append(os.path.basename(d).replace("forecast_cycle25_mem_", ""))
    if stale:
        print(f"WARNING: skipped {len(stale)} stale PINN member(s) trained with a "
              f"different data horizon (delete or retrain them): {stale}")
    # If members from the old 5-member run (tags without "_T": SC21..SC24,
    # zero) coexist with the T x amp grid, keep only the grid -- the old
    # members use a different source construction and would bias the band.
    new_style = [i for i, tg in enumerate(tags) if "_T" in tg]
    if new_style and len(new_style) < len(tags):
        dropped = [tg for tg in tags if "_T" not in tg]
        print(f"WARNING: ignoring {len(dropped)} member(s) from the old 5-member "
              f"run mixed in with the ensemble grid: {dropped}")
        members = [members[i] for i in new_style]
        tags = [tags[i] for i in new_style]
    return members, tags

def band(members, key, t_grid):
    M = np.full((len(members), t_grid.size), np.nan)
    for i, m in enumerate(members):
        ok = (t_grid >= m["t"][0]) & (t_grid <= m["t"][-1])
        M[i, ok] = np.interp(t_grid[ok], m["t"], m[key])
    with np.errstate(all="ignore"):
        return (np.nanpercentile(M, 10, 0), np.nanmedian(M, 0), np.nanpercentile(M, 90, 0))

# ---------------- assemble ----------------
fd, T_data, ic0 = build_fd_ensemble()
pinn, pinn_tags = build_pinn_ensemble(
    T_data_ref=T_data, ic0=ic0 if IC_ANNUAL_MEAN_YR > 0 else None)
t_now = YR0 + T_data
print(f"FD ensemble : {len(fd)} members")
print(f"PINN ensemble: {len(pinn)} members" + (f"  -> {pinn_tags}" if pinn else
      "  (NONE FOUND -- run forecast_cycle25_pinn.py; the blue layer will be empty)"))

obs25 = store[25]["obs"]; t_obs25 = YR0 + store[25]["t_obs"]
obsN, obsS = caps_G(obs25, ct.LAT_DEG); obsD = dipole_G(obs25, ct.LAT_DEG)
t_end = max([m["t"][-1] for m in fd] + [m["t"][-1] for m in pinn] + [t_obs25[-1]])
t_grid = np.linspace(YR0, t_end, 240)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))

# panel 1: polar caps (medians + faint members + obs) ----------------
a0 = ax[0]
_, mdN, _ = band(fd, "N", t_grid); _, mdS, _ = band(fd, "S", t_grid)
a0.plot(t_grid, mdN, C_FD, lw=2.4, label="FD median N")
a0.plot(t_grid, mdS, C_FD, lw=2.4, ls="--", label="FD median S")
for m in pinn:
    a0.plot(m["t"], m["N"], C_PI, alpha=0.30, lw=1.0)
    a0.plot(m["t"], m["S"], C_PI, alpha=0.30, lw=1.0, ls="--")
if pinn:
    _, mdNp, _ = band(pinn, "N", t_grid); _, mdSp, _ = band(pinn, "S", t_grid)
    a0.plot(t_grid, mdNp, C_PI, lw=2.4, label="PINN median N")
    a0.plot(t_grid, mdSp, C_PI, lw=2.4, ls="--", label="PINN median S")
a0.plot(t_obs25, obsN, "k", lw=1.5, label="WSO N")
a0.plot(t_obs25, obsS, "k--", lw=1.5, label="WSO S")
a0.axvline(t_now, color="gray", ls=":", lw=1.2)
a0.text(t_now, a0.get_ylim()[1], " data horizon", va="top", ha="left", fontsize=8, color="gray")
a0.axhline(0, color="gray", lw=0.5)
a0.set_xlabel("year"); a0.set_ylabel("polar-cap mean field [G]")
a0.set_title("Cycle 25 polar caps: PINN vs finite-volume")
a0.legend(fontsize=7, ncol=2, loc="lower left")

# panel 2: axial dipole (bands + medians + obs) ----------------------
a1 = ax[1]
loD, mdD, hiD = band(fd, "D", t_grid)
a1.fill_between(t_grid, loD, hiD, color=C_FD, alpha=0.18, label="FD 10-90%")
a1.plot(t_grid, mdD, C_FD, lw=2.4, label="FD median")
if pinn:
    loDp, mdDp, hiDp = band(pinn, "D", t_grid)
    a1.fill_between(t_grid, loDp, hiDp, color=C_PI, alpha=0.20, label="PINN 10-90%")
    a1.plot(t_grid, mdDp, C_PI, lw=2.4, label="PINN median")
    for m in pinn:
        a1.plot(m["t"], m["D"], C_PI, alpha=0.30, lw=1.0)
a1.plot(t_obs25, obsD, "k", lw=1.5, label="WSO dipole")
a1.axvline(t_now, color="gray", ls=":", lw=1.2)
a1.axhline(0, color="gray", lw=0.5)
a1.set_xlabel("year"); a1.set_ylabel("axial dipole $D$ [G]")
a1.set_title("Cycle 25 axial dipole: PINN vs finite-volume")
a1.legend(fontsize=7, loc="upper right")

if not pinn:
    fig.text(0.5, 0.5, "PINN layer empty\n(run forecast_cycle25_pinn.py)",
             ha="center", va="center", fontsize=15, color=C_PI, alpha=0.55,
             rotation=18, fontweight="bold")

plt.tight_layout()
outfig = os.path.join(OUT, "forecast_cycle25_PINN_vs_FD.png")
plt.savefig(outfig, dpi=140)
print("wrote", outfig)

def endvals(members, key):
    return np.array([np.mean(m[key][-4:]) for m in members])
print("\nEnd-of-cycle (next minimum) summary [Gauss]:")
for label, members in [("FD  ", fd), ("PINN", pinn)]:
    if not members:
        print(f"  {label}: (no members yet)"); continue
    dD = endvals(members, "D")
    print(f"  {label}: D median {np.median(dD):+.2f}  [{dD.min():+.2f},{dD.max():+.2f}] | "
          f"N {np.median(endvals(members,'N')):+.2f} | S {np.median(endvals(members,'S')):+.2f}")
