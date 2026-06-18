import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
replot_from_field_gauss.py -- regenerate the per-cycle reconstruction figures
in TRUE GAUSS from the saved PINN field.npy.

WHY THE OLD FIGURES WERE ~100x TOO LARGE
----------------------------------------
The pipeline runs in WSO *native* units (config.WSO_TO_GAUSS = 1.0), so
    model units  x  B_unit(=10)  =  native (microtesla-like) field,  NOT Gauss.
True Gauss needs one more factor 0.01 (1 uT = 0.01 G):
    B[G] = B_model * B_unit * 0.01 = B_model * 0.1.
The old replot multiplied by B_unit only -> native values mislabelled "[G]".

This script also (i) plots the dipole COEFFICIENT D = 3/2 \\int B mu dmu [G]
used in Table 2, not the dipole moment in Mx cm, and (ii) computes the net
flux with B in Gauss.

No TensorFlow needed. Run from the repo root, after the reconstructions exist
in results/cycle_NN/:
    python replot_from_field_gauss.py            # cycles 21-24
    python replot_from_field_gauss.py 24         # one cycle
Figures are overwritten in results/cycle_NN/ (Gauss); a one-line summary and
the polar-residual numbers (toward Table 3) are printed per cycle.
"""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.extract import get_wso_map_for_comparison      # numpy/scipy only, no TF
try:
    from src import cycle_tools as ct                   # for the FD overlay
    _HAVE_CT = True
except Exception:
    _HAVE_CT = False

# ---- pipeline constants (no need to import src.config / TensorFlow) ----
HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_UNIT    = 10.0
TO_GAUSS  = B_UNIT * 0.01            # model units -> TRUE Gauss (= 0.1)
L_UNIT    = 6.95e10                  # solar radius [cm]
NUM_LATS  = 181
NT_POINTS = 400                      # field.npy has NT_POINTS+1 = 401 rows
LAM_MIN, LAM_MAX = -0.495, 0.495
LAT_DEG   = np.linspace(LAM_MIN, LAM_MAX, NUM_LATS) * 180.0
LENGTHS   = json.load(open(os.path.join(HERE, "cycle_products", "cycle_lengths.json")))

def dipole_G(B_G, lat_deg):
    mu = np.sin(np.deg2rad(lat_deg)); o = np.argsort(mu)
    return 1.5 * np.trapezoid(B_G[:, o] * mu[o], mu[o], axis=1)

def fd_polar_caps_G(cycle, T):
    """Finite-volume polar caps in Gauss from store.pkl (for the overlay)."""
    store = pickle.load(open(os.path.join(HERE, "cycle_products", "store.pkl"), "rb"))
    s = store[cycle]
    t_fd, B_fd = ct.forward(s["obs_s"][0], 0.0, s["T"], s["S"], s["t_u"])   # model units
    n, sm = ct.LAT_DEG >= 60.0, ct.LAT_DEG <= -60.0
    return t_fd, B_fd[:, n].mean(1) * TO_GAUSS, B_fd[:, sm].mean(1) * TO_GAUSS

def replot(cycle):
    out = os.path.join(HERE, "results", f"cycle_{cycle}")
    fp = os.path.join(out, "field.npy")
    if not os.path.exists(fp):
        print(f"[cycle {cycle}] no field.npy in {out} -- skipping"); return
    T = float(LENGTHS[str(cycle)])
    B = np.load(fp)                                          # (Nt, Nlat) model units
    t_years = np.linspace(0.0, 1.0, NT_POINTS + 1) * T

    _, _, B_obs = get_wso_map_for_comparison(
        Tmax=1.0, lat_points=NUM_LATS, time_steps=NT_POINTS + 1,
        B_unit=B_UNIT, data_dir=f"data/{cycle}")            # model units (unit_to_gauss=1)

    Nt = min(B.shape[0], B_obs.shape[0]); Nl = min(B.shape[1], B_obs.shape[1])
    B, B_obs, lat, ty = B[:Nt,:Nl], B_obs[:Nt,:Nl], LAT_DEG[:Nl], t_years[:Nt]
    Bg, Og = B * TO_GAUSS, B_obs * TO_GAUSS                  # <-- TRUE GAUSS
    res = Bg - Og
    ext = [ty[0], ty[-1], lat[0], lat[-1]]

    # 1) reconstruction map
    plt.figure(figsize=(7,5))
    plt.imshow(Bg.T, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r")
    plt.colorbar(label="Magnetic field [G]"); plt.xlabel("Time [yr]"); plt.ylabel("Latitude [deg]")
    plt.title(f"PINN reconstruction, Cycle {cycle}"); plt.tight_layout()
    plt.savefig(os.path.join(out, "magnetic_field.png"), dpi=300); plt.close()

    # 2) residual map
    vmax = np.nanpercentile(np.abs(res), 98)
    plt.figure(figsize=(7,5))
    plt.imshow(res.T, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(label=r"$\Delta B_r$ [G]"); plt.xlabel("Time [yr]"); plt.ylabel("Latitude [deg]")
    plt.title(rf"Residual $B_r^{{\rm PINN}}-B_r^{{\rm WSO}}$, Cycle {cycle}"); plt.tight_layout()
    plt.savefig(os.path.join(out, "residual_map.png"), dpi=300); plt.close()

    # 3) scatter (observed latitudes only)
    keep = np.abs(lat) <= 75.0
    x, y = Og[:,keep].ravel(), Bg[:,keep].ravel()
    lim = [min(x.min(),y.min()), max(x.max(),y.max())]
    plt.figure(figsize=(5.5,5.5)); plt.scatter(x, y, s=5, alpha=0.35)
    plt.plot(lim, lim, "k--", lw=1.2); plt.xlabel(r"WSO $B_r$ [G]"); plt.ylabel(r"PINN $B_r$ [G]")
    plt.title(f"Point-by-point, Cycle {cycle}"); plt.tight_layout()
    plt.savefig(os.path.join(out, "scatter_pinn_vs_wso.png"), dpi=300); plt.close()

    # 4) polar caps (PINN vs WSO)
    n, sm = lat>=60, lat<=-60
    pN,pS,oN,oS = Bg[:,n].mean(1),Bg[:,sm].mean(1),Og[:,n].mean(1),Og[:,sm].mean(1)
    plt.figure(figsize=(8,4.2))
    plt.plot(ty,oN,label="WSO N"); plt.plot(ty,pN,label="PINN N")
    plt.plot(ty,oS,label="WSO S"); plt.plot(ty,pS,label="PINN S")
    plt.axhline(0,ls="--",color="k",lw=1); plt.xlabel("Time [yr]"); plt.ylabel("Polar-cap mean field [G]")
    plt.title(rf"Polar caps ($|\lambda|\geq60^\circ$), Cycle {cycle}"); plt.legend(fontsize=9,ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(out,"polar_cap_mean_comparison.png"),dpi=300); plt.close()

    # 5) dipole COEFFICIENT [G]
    Dp, Do = dipole_G(Bg, lat), dipole_G(Og, lat)
    plt.figure(figsize=(7,4))
    plt.plot(ty,Do,label="WSO"); plt.plot(ty,Dp,label="PINN")
    plt.axhline(0,color="k",ls="--"); plt.xlabel("Time [yr]"); plt.ylabel(r"Axial dipole $D$ [G]")
    plt.title(f"Axial dipole coefficient, Cycle {cycle}"); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out,"dipole_moment.png"),dpi=300); plt.close()

    # 6) flux balance [Mx] (B in Gauss)
    mu = np.sin(np.deg2rad(lat))
    flux = 2*np.pi*L_UNIT**2 * np.trapezoid(Bg, mu, axis=1)
    plt.figure(figsize=(7,3.8)); plt.plot(ty,flux); plt.axhline(0,ls="--",color="k",lw=1)
    plt.xlabel("Time [yr]"); plt.ylabel("Net signed flux [Mx]"); plt.title(f"Flux balance, Cycle {cycle}")
    plt.tight_layout(); plt.savefig(os.path.join(out,"flux_balance_check.png"),dpi=300); plt.close()

    # 7) FD-vs-PINN polar-cap overlay (needs cycle_tools + store.pkl)
    if _HAVE_CT:
        try:
            t_fd, fN, fS = fd_polar_caps_G(cycle, T)
            plt.figure(figsize=(8,4.4))
            plt.plot(ty,oN,"0.4",lw=1); plt.plot(ty,oS,"0.4",lw=1,ls="--")
            plt.plot(t_fd,fN,"C0",label="FD N"); plt.plot(t_fd,fS,"C0--",label="FD S")
            plt.plot(ty,pN,"C1",label="PINN N"); plt.plot(ty,pS,"C1--",label="PINN S")
            plt.plot([],[],"0.4",label="WSO")
            plt.axhline(0,ls="--",color="k",lw=0.8); plt.legend(fontsize=8,ncol=3)
            plt.xlabel("Time [yr]"); plt.ylabel("Polar-cap mean field [G]")
            plt.title(rf"PINN vs finite-volume ($|\lambda|\geq60^\circ$), Cycle {cycle}")
            plt.tight_layout(); plt.savefig(os.path.join(out,"fd_pinn_overlay.png"),dpi=300); plt.close()
        except Exception as e:
            print(f"[cycle {cycle}] FD overlay skipped: {e}")

    # 8) residual latitude structure + polar-residual decomposition (from arrays,
    #    same definitions as Table 3, replacing the colour-inverted estimates)
    res_mean = res.mean(0); res_std = res.std(0)
    ncap, scap = lat >= 70.0, lat <= -70.0; pol = ncap | scap
    cohN_t = res[:, ncap].mean(1); cohS_t = res[:, scap].mean(1)   # coherent cap-mean residual vs time
    cap_rms = np.sqrt((res[:, pol]**2).mean(1))
    f_pol = 100.0 * (res[:, pol]**2).sum() / (res**2).sum()        # % polar variance (|lat|>70)
    # incoherent component = polar residual minus its per-time cap mean
    rN = res[:, ncap] - cohN_t[:, None]; rS = res[:, scap] - cohS_t[:, None]
    sig_pol = np.sqrt((np.concatenate([rN, rS], axis=1) ** 2).mean())
    # coherent peak amplitude/epoch, excluding first/last 6% to skip the t=0 transient
    mm = max(1, int(0.06 * len(ty)))
    iN = mm + int(np.argmax(np.abs(cohN_t[mm:len(ty)-mm])))
    iS = mm + int(np.argmax(np.abs(cohS_t[mm:len(ty)-mm])))
    dcohN_pk, tN = float(cohN_t[iN]), float(ty[iN]); dcohS_pk, tS = float(cohS_t[iS]), float(ty[iS])
    dcohN_mean = float(cohN_t.mean()); dcohS_mean = float(cohS_t.mean())   # steady coherent bias (Table 3)
    # observed-band steady bias: WSO is real for |lat|<=~75 deg; above that the comparison map is cubic-extrapolated
    obsN = (lat >= 70.0) & (lat <= 75.0); obsS = (lat <= -70.0) & (lat >= -75.0)
    dcohN_obs = float(res[:, obsN].mean()) if obsN.any() else float("nan")
    dcohS_obs = float(res[:, obsS].mean()) if obsS.any() else float("nan")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(lat, res_mean, "C3"); ax[0].fill_between(lat, res_mean-res_std, res_mean+res_std, color="C3", alpha=0.2)
    for L in (-75,75): ax[0].axvline(L, ls=":", color="gray", lw=1)
    ax[0].axhline(0, color="gray", lw=0.5); ax[0].set_xlabel("Latitude [deg]")
    ax[0].set_ylabel(r"Mean residual $\langle\Delta B_r\rangle$ [G]"); ax[0].set_title(f"(a) Cycle {cycle}")
    ax[1].plot(ty, cap_rms, "C0"); ax[1].set_xlabel("Time [yr]")
    ax[1].set_ylabel(r"polar-cap RMS residual [G]"); ax[1].set_title(rf"(b) $|\lambda|>70^\circ$")
    plt.tight_layout(); plt.savefig(os.path.join(out, f"residual_latitude_structure_c{cycle}.png"), dpi=300); plt.close()

    print(f"[cycle {cycle}] Gauss figures -> {out}")
    print(f"           |B|max={np.abs(Bg).max():.2f} G | D^end={Dp[-8:].mean():+.2f} G | "
          f"caps N/S end {pN[-8:].mean():+.2f}/{pS[-8:].mean():+.2f} G")
    print(f"           Table 3 row:  f_pol={f_pol:.0f}%   sigma_pol={sig_pol:.2f} G   "
          f"t_peak_N/S={tN:.1f}/{tS:.1f} yr")
    print(f"             Dcoh steady N/S = {dcohN_mean:+.2f}/{dcohS_mean:+.2f} G   "
          f"[obs band 70-75: {dcohN_obs:+.2f}/{dcohS_obs:+.2f} G]   "
          f"(peak/transient {dcohN_pk:+.2f}/{dcohS_pk:+.2f} G near reversal)")

def main():
    cycles = [int(a) for a in sys.argv[1:]] or [21, 22, 23, 24]
    for c in cycles:
        replot(c)

if __name__ == "__main__":
    main()
