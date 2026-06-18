"""
hindcast_forecast2.py -- Analog-ensemble version.

Instead of one mean-template future source, each forecast member uses the
declining-phase source SHAPE of one specific past cycle (polarity-aligned,
phase-normalized, amplitude-scaled to the target cycle's observed window).
The member spread then naturally contains anomalous outcomes like SC23's
weak polar-field buildup, which a mean template averages away.

Stage 2: leave-one-out analog hindcasts for cycles 21-24 (does the truth
         fall inside the analog envelope?).
Stage 3: cycle-25 forecast = analogs {21,22,23,24} x lengths {10.5,11,11.5}
         x amplitude {0.85,1.0,1.15}.
"""
import os, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cycle_tools as ct

OUT = "cycle_products"
store = pickle.load(open(f"{OUT}/store.pkl", "rb"))

PHASE_TRUNC = 5.08 / 11.0
NPHASE = 401
phase = np.linspace(0.0, 1.0, NPHASE)
BELT = np.abs(ct.LAT_DEG) < 50.0
COMPLETE = [21, 22, 23, 24]


def polarity(c):
    return np.sign(ct.dipole(store[c]["obs"][:5].mean(0, keepdims=True) * ct.B_UNIT)[0])


def phase_source(c):
    S, t_u, T = store[c]["S"], store[c]["t_u"], store[c]["T"]
    out = np.empty((NPHASE, ct.N))
    for j in range(ct.N):
        out[:, j] = np.interp(phase, t_u / T, S[:, j])
    return out


def amp_win(Sp, p0, p1):
    m = (phase >= p0) & (phase <= p1)
    return np.sqrt(np.mean(Sp[np.ix_(m, BELT)] ** 2))


def truncated_refit(c, t_tr):
    t_obs, obs = store[c]["t_obs"], store[c]["obs"]
    m = t_obs <= t_tr
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs[m], obs[m], t_tr, nt=201)
    return t_u, obs_s, ct.refit_source(t_u, obs_s)


def analog_member(target, analog, T, t_u, obs_s, S_tr, amp_factor=1.0):
    """Forward run from truncation using `analog`'s source shape."""
    Sp = phase_source(analog) * (polarity(target) * polarity(analog))
    p1 = t_u[-1] / T
    m_tr = np.sqrt(np.mean(S_tr[:, BELT] ** 2))
    sc = amp_factor * m_tr / amp_win(Sp, 0.0, p1)
    t_m, B = ct.forward(obs_s[-1], t_u[-1], T, Sp * sc, phase * T, nt_out=160)
    return t_m, B * ct.B_UNIT


def truth_metrics(c):
    on, osd = ct.polar_means(store[c]["obs"] * ct.B_UNIT)
    return (ct.smooth1d(on)[-6:].mean(), ct.smooth1d(osd)[-6:].mean(),
            ct.smooth1d(ct.dipole(store[c]["obs"] * ct.B_UNIT))[-6:].mean())


# ----------------------------------------------------------------------
# Stage 2: leave-one-out analog hindcasts
# ----------------------------------------------------------------------
print("LEAVE-ONE-OUT ANALOG HINDCASTS (truncation phase 0.462)")
print(f"{'cyc':>4} | {'dipole true':>11} | {'members (by analog)':>34} | {'envelope':>16} | in?")
cov = 0
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
for ax, c in zip(axes.flat, COMPLETE):
    T = store[c]["T"]
    t_tr = PHASE_TRUNC * T
    t_u, obs_s, S_tr = truncated_refit(c, t_tr)
    _, _, dip_t = truth_metrics(c)

    members = {}
    for a in COMPLETE:
        if a == c:
            continue
        t_m, Bn = analog_member(c, a, T, t_u, obs_s, S_tr)
        members[a] = (t_m, Bn, ct.dipole(Bn)[-4:].mean())
    dips = np.array([v[2] for v in members.values()])
    lo, hi = dips.min(), dips.max()
    inside = (lo <= dip_t <= hi)
    cov += inside
    mstr = " ".join(f"SC{a}:{v[2]:+.0f}" for a, v in members.items())
    print(f"{c:>4} | {dip_t:>11.1f} | {mstr:>34} | [{lo:+7.1f},{hi:+7.1f}] | {'Y' if inside else 'N'}")

    t_obs, obs = store[c]["t_obs"], store[c]["obs"]
    on, osd = ct.polar_means(obs * ct.B_UNIT)
    ax.plot(t_obs, on, "k", lw=0.8, alpha=0.55)
    ax.plot(t_obs, osd, "k--", lw=0.8, alpha=0.55)
    for a, (t_m, Bn, _) in members.items():
        pn, ps = ct.polar_means(Bn)
        ax.plot(t_m, pn, "C1", alpha=0.7, lw=1.2)
        ax.plot(t_m, ps, "C3--", alpha=0.7, lw=1.2)
    ax.axvline(t_tr, color="gray", ls=":", lw=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"SC{c}: analog hindcasts after {t_tr:.1f} yr")
plt.tight_layout()
plt.savefig(f"{OUT}/hindcasts_analog.png", dpi=130)
print(f"Envelope coverage: {cov}/4")

# ----------------------------------------------------------------------
# Stage 3: cycle-25 analog-ensemble forecast
# ----------------------------------------------------------------------
t_obs25, obs25 = store[25]["t_obs"], store[25]["obs"]
T_data = t_obs25[-1]
t_u25, obs_s25 = ct.smooth_on_uniform_time(t_obs25, obs25, T_data, nt=201)
S_tr25 = ct.refit_source(t_u25, obs_s25)

ens = []
for a in COMPLETE:
    for T25 in [10.5, 11.0, 11.5]:
        for f in [0.85, 1.0, 1.15]:
            t_m, Bn = analog_member(25, a, T25, t_u25, obs_s25, S_tr25, amp_factor=f)
            pn, ps = ct.polar_means(Bn)
            dp = ct.dipole(Bn)
            ens.append(dict(analog=a, T=T25, f=f, t=t_m, n=pn, s=ps, d=dp,
                            endN=pn[-4:].mean(), endS=ps[-4:].mean(),
                            endD=dp[-4:].mean()))

endN = np.array([e["endN"] for e in ens])
endS = np.array([e["endS"] for e in ens])
endD = np.array([e["endD"] for e in ens])

print("\n================ CYCLE 25 FORECAST (analog ensemble, 36 members) ================")
print(f"WSO data through {T_data:.2f} yr after Dec 2019 (~Jan 2025).")
print("Values in WSO native units, at next minimum (~2030-31):")
for name, v in [("North polar cap", endN), ("South polar cap", endS), ("Axial dipole", endD)]:
    print(f"  {name:16s}: median {np.median(v):+7.1f}   "
          f"[{np.percentile(v,10):+7.1f}, {np.percentile(v,90):+7.1f}] (10-90%)   "
          f"full range [{v.min():+.1f}, {v.max():+.1f}]")

print("\nBy analog (dipole at minimum):")
for a in COMPLETE:
    d = np.array([e["endD"] for e in ens if e["analog"] == a])
    print(f"  SC{a}-like decline: {d.mean():+7.1f}  [{d.min():+.1f}, {d.max():+.1f}]")

_, _, dip24 = truth_metrics(24)
print(f"\nPrecursor: SC25-end dipole / SC24-end dipole = "
      f"{np.median(endD)/dip24:+.2f}  (|ratio|<1 -> SC26 weaker than SC25)")

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
yr0 = 2019.94
on25, os25_ = ct.polar_means(obs25 * ct.B_UNIT)
colors = {21: "C0", 22: "C1", 23: "C2", 24: "C4"}
for e in ens:
    ax[0].plot(yr0 + e["t"], e["n"], color=colors[e["analog"]], alpha=0.25, lw=0.9)
    ax[0].plot(yr0 + e["t"], e["s"], color=colors[e["analog"]], alpha=0.25, lw=0.9, ls="--")
    ax[1].plot(yr0 + e["t"], e["d"], color=colors[e["analog"]], alpha=0.3, lw=0.9)
for a in COMPLETE:
    ax[1].plot([], [], color=colors[a], label=f"SC{a}-like")
ax[0].plot(yr0 + t_obs25, on25, "k", lw=1.2, label="WSO N")
ax[0].plot(yr0 + t_obs25, os25_, "k--", lw=1.2, label="WSO S")
ax[0].axhline(0, color="gray", lw=0.5)
ax[0].legend(fontsize=8)
ax[0].set_title("Cycle 25 polar caps: observed + analog forecast")
ax[0].set_xlabel("year")
ax[1].plot(yr0 + t_obs25, ct.dipole(obs25 * ct.B_UNIT), "k", lw=1.2, label="WSO dipole")
ax[1].axhline(0, color="gray", lw=0.5)
ax[1].legend(fontsize=8)
ax[1].set_title("Axial dipole: observed + analog forecast")
ax[1].set_xlabel("year")
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_cycle25.png", dpi=130)
print("\nFigures: hindcasts_analog.png, forecast_cycle25.png")
