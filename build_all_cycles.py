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
import cycle_tools as ct

PROJ = "/media/talafha/Disk_1/HPC_Simulations/Data_based/Source_SC24_1"
OUT = "/media/talafha/Disk_1/HPC_Simulations/Data_based/Source_SC24_1/cycle_products"
os.makedirs(OUT, exist_ok=True)

cycles = [21, 22, 23, 24, 25]
lengths = {}
store = {}

fig, axes = plt.subplots(len(cycles), 2, figsize=(14, 3.1 * len(cycles)))

print(f"{'cyc':>4} {'T[yr]':>6} {'rotN':>5} | {'rev N obs':>9} {'rev N mod':>9} | "
      f"{'endN obs':>8} {'endN mod':>8} | {'endS obs':>8} {'endS mod':>8} | {'RMS':>5}")

for r, c in enumerate(cycles):
    t_obs, obs, T = ct.load_cycle(PROJ, c)
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs, obs, T)
    S = ct.refit_source(t_u, obs_s)
    ct.save_source_for_pinn(S, f"{OUT}/fitted_source_map_cycle{c}.npy")
    np.save(f"{OUT}/source_mu_cycle{c}.npy", S)
    lengths[str(c)] = round(T, 3)

    t_m, B = ct.forward(obs_s[0], 0.0, T, S, t_u)
    Bn = B * ct.B_UNIT
    obs_n, obs_sd = ct.polar_means(obs * ct.B_UNIT)
    mod_n, mod_sd = ct.polar_means(Bn)

    on_s = ct.smooth1d(obs_n)
    rev_obs = ct.last_crossing(t_obs, on_s)
    rev_mod = ct.last_crossing(t_m, mod_n)
    endN_o, endS_o = on_s[-6:].mean(), ct.smooth1d(obs_sd)[-6:].mean()
    endN_m, endS_m = mod_n[-6:].mean(), mod_sd[-6:].mean()
    rms = np.sqrt(np.mean((np.interp(t_obs, t_m, mod_n) - obs_n) ** 2))

    store[c] = dict(t_obs=t_obs, obs=obs, T=T, t_u=t_u, obs_s=obs_s, S=S)
    print(f"{c:>4} {T:>6.2f} {len(t_obs):>5} | "
          f"{(('%.2f' % rev_obs) if rev_obs else 'none'):>9} "
          f"{(('%.2f' % rev_mod) if rev_mod else 'none'):>9} | "
          f"{endN_o:>8.1f} {endN_m:>8.1f} | {endS_o:>8.1f} {endS_m:>8.1f} | {rms:>5.1f}")

    ax = axes[r, 0]
    ax.plot(t_obs, obs_n, "k", lw=0.8, alpha=0.6)
    ax.plot(t_obs, obs_sd, "k--", lw=0.8, alpha=0.6)
    ax.plot(t_m, mod_n, "C1", lw=1.6, label="model N")
    ax.plot(t_m, mod_sd, "C3--", lw=1.6, label="model S")
    ax.axhline(0, color="gray", lw=0.5)
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
plt.savefig(f"{OUT}/reconstruction_cycles.png", dpi=130)

with open(f"{OUT}/cycle_lengths.json", "w") as f:
    json.dump(lengths, f, indent=2)
np.savez(f"{OUT}/store_meta.npz", cycles=cycles)
import pickle
with open(f"{OUT}/store.pkl", "wb") as f:
    pickle.dump(store, f)
print("\nSaved per-cycle sources, lengths, and reconstruction figure to", OUT)
