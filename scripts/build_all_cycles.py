import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
build_all_cycles.py -- Stage 1: per-cycle source maps + reconstruction validation.

For each cycle 21-25:
  1. load WSO data, determine actual cycle length from the data,
  2. refit the source with the corrected operator,
  3. forward-run the FD model with that source from the cycle-start profile,
  4. compare polar caps / dipole / reversal times against WSO,
  5. save fitted_source_map_cycleNN.npy (native units/yr) for the PINN.

Outputs: data products in OUT/, summary table to stdout, reconstruction figure.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import cycle_tools as ct

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cycle_products")
os.makedirs(OUT, exist_ok=True)

cycles = [21, 22, 23, 24, 25]
lengths = {}
store = {}

fig, axes = plt.subplots(len(cycles), 2, figsize=(14, 3.1 * len(cycles)))

def _fmt(v):
    return f"{v:.2f}" if v is not None else "none"

print(f"{'cyc':>4} {'T[yr]':>6} {'rotN':>5} | "
      f"{'revN obs':>8} {'revN mod':>8} | {'revS obs':>8} {'revS mod':>8} | "
      f"{'endN obs':>8} {'endN mod':>8} | {'endS obs':>8} {'endS mod':>8} | {'RMS':>5}")

for r, c in enumerate(cycles):
    t_obs, obs, T = ct.load_cycle(c)
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs, obs, T)
    S = ct.refit_source(t_u, obs_s)
    ct.save_source_for_pinn(S, os.path.join(HERE, "data", f"fitted_source_map_cycle{c}.npy"))
    np.save(os.path.join(OUT, f"source_mu_cycle{c}.npy"), S)
    lengths[str(c)] = round(T, 3)

    t_m, B = ct.forward(obs_s[0], 0.0, T, S, t_u)
    Bn = B * ct.B_UNIT
    obs_n, obs_sd = ct.polar_means(obs * ct.B_UNIT)
    mod_n, mod_sd = ct.polar_means(Bn)

    # smooth both obs and model identically before reversal detection so the
    # last zero crossing is not biased by high-frequency noise in either series
    on_s  = ct.smooth1d(obs_n)
    os_s  = ct.smooth1d(obs_sd)
    mn_s  = ct.smooth1d(mod_n)
    ms_s  = ct.smooth1d(mod_sd)

    revN_obs = ct.last_crossing(t_obs, on_s)
    revS_obs = ct.last_crossing(t_obs, os_s)
    revN_mod = ct.last_crossing(t_m,   mn_s)
    revS_mod = ct.last_crossing(t_m,   ms_s)

    endN_o, endS_o = on_s[-6:].mean(), os_s[-6:].mean()
    endN_m, endS_m = mod_n[-6:].mean(), mod_sd[-6:].mean()
    rms = np.sqrt(np.mean((np.interp(t_obs, t_m, mod_n) - obs_n) ** 2))

    store[c] = dict(t_obs=t_obs, obs=obs, T=T, t_u=t_u, obs_s=obs_s, S=S)
    print(f"{c:>4} {T:>6.2f} {len(t_obs):>5} | "
          f"{_fmt(revN_obs):>8} {_fmt(revN_mod):>8} | "
          f"{_fmt(revS_obs):>8} {_fmt(revS_mod):>8} | "
          f"{endN_o:>8.1f} {endN_m:>8.1f} | {endS_o:>8.1f} {endS_m:>8.1f} | {rms:>5.1f}")

    ax = axes[r, 0]
    ax.plot(t_obs, obs_n, "k",  lw=0.8, alpha=0.6)
    ax.plot(t_obs, obs_sd, "k--", lw=0.8, alpha=0.6)
    ax.plot(t_m, mod_n,  "C1",   lw=1.6, label="model N")
    ax.plot(t_m, mod_sd, "C3--", lw=1.6, label="model S")
    ax.axhline(0, color="gray", lw=0.5)
    # mark last zero crossings so they can be visually verified
    for tv, col, ls in [(revN_obs, "k",  "-"), (revS_obs, "k",  "--"),
                        (revN_mod, "C1", "-"), (revS_mod, "C3", "--")]:
        if tv is not None:
            ax.axvline(tv, color=col, ls=ls, lw=0.8, alpha=0.6)
    ax.set_ylabel(f"SC{c}\npolar mean")
    if r == 0:
        ax.legend(fontsize=7)
        ax.set_title("Polar caps: WSO (black) vs FD reconstruction")

    ax = axes[r, 1]
    vmax = np.percentile(np.abs(Bn), 99)
    ax.pcolormesh(t_m, ct.LAT_DEG, Bn.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if r == 0:
        ax.set_title("Reconstructed butterfly")
axes[-1, 0].set_xlabel("years since cycle start")
axes[-1, 1].set_xlabel("years since cycle start")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "reconstruction_cycles.png"), dpi=130)

with open(os.path.join(OUT, "cycle_lengths.json"), "w") as f:
    json.dump(lengths, f, indent=2)
np.savez(os.path.join(OUT, "store_meta.npz"), cycles=cycles)
import pickle
with open(os.path.join(OUT, "store.pkl"), "wb") as f:
    pickle.dump(store, f)
print("\nSaved per-cycle sources, lengths, and reconstruction figure to", OUT)
