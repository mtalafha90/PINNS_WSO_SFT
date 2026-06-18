# plot.py
import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import compute_lat_grid, compute_dipole_moment
from .extract import get_wso_map_for_comparison


def _compute_error_metrics(y_true, y_pred):
    """
    y_true, y_pred : 1D arrays
    Returns dict with MAE, RMSE, Corr, R2
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # Correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = np.nan

    # R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Corr": corr,
        "R2": r2,
    }


def _compute_net_flux(B_model, lat_deg, config):
    """
    Compute net signed surface flux from longitudinally averaged Br.

    B_model : (Nt, Nlat) in model units
    lat_deg : (Nlat,) degrees
    Returns : (Nt,) net flux in Mx
    """
    B_phys = B_model * config.B_unit  # Gauss
    mu = np.sin(np.deg2rad(lat_deg))   # monotonic if lat sorted
    R = float(config.L_unit)

    # Full-surface net flux:
    # Phi = 2*pi*R^2 * integral B(mu,t) dmu
    Phi = 2.0 * np.pi * R**2 * np.trapz(B_phys, mu, axis=1)
    return Phi


def _smooth(y, k=9):
    """Boxcar smoothing (~k Carrington rotations if y is per-rotation)."""
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def _last_zero_crossing(t_years, y, smooth_k=9):
    """
    Return the LAST zero-crossing time of the SMOOTHED series, via linear
    interpolation.  The last crossing of a smoothed polar-cap series is the
    physically meaningful reversal time; the first crossing of the raw
    series is just noise wobbling around zero early in the cycle.
    Returns None if no crossing exists.
    """
    t_years = np.asarray(t_years)
    y = _smooth(y, k=smooth_k)

    out = None
    s = np.sign(y)
    for i in range(len(y) - 1):
        if s[i] == 0:
            out = t_years[i]
        elif s[i] * s[i + 1] < 0:
            out = t_years[i] - y[i] * (t_years[i + 1] - t_years[i]) / (y[i + 1] - y[i])
    return out


def generate_all_plots(config, model, train_state):
    os.makedirs(config.output_dir, exist_ok=True)

    # ----------------------------
    # 1) Predict PINN field
    # ----------------------------
    lat_deg, polar_mask = compute_lat_grid(config)
    lat_rad = np.deg2rad(lat_deg)
    time_array = np.linspace(0, config.Tmax, config.num_time_points + 1)
    time_years = time_array * config.simul_time

    B_pred_list = []
    for lam_norm in lat_rad / np.pi:
        lam = np.ones_like(time_array) * lam_norm
        coords = np.stack((lam, time_array), axis=1)
        B_pred_list.append(model.predict(coords).ravel())

    B_pred = np.array(B_pred_list).T   # shape (Nt, Nlat) in model units
    np.save(os.path.join(config.output_dir, "field.npy"), B_pred)

    # ----------------------------
    # 2) Load observed WSO map on same grid
    # ----------------------------
    t_obs_norm, lat_obs_deg, B_obs = get_wso_map_for_comparison(
        Tmax=config.Tmax,
        lat_points=config.num_lats,
        time_steps=config.num_time_points + 1,
        B_unit=config.B_unit,
        data_dir=config.wso_path,
        unit_to_gauss=getattr(config, "WSO_TO_GAUSS", 1.0),
    )

    # Safety check: make sure shapes match
    Nt = min(B_pred.shape[0], B_obs.shape[0])
    Nlat = min(B_pred.shape[1], B_obs.shape[1])

    B_pred = B_pred[:Nt, :Nlat]
    B_obs = B_obs[:Nt, :Nlat]
    lat_deg = lat_deg[:Nlat]
    polar_mask = polar_mask[:Nlat]
    time_years = time_years[:Nt]

    # Convert to Gauss for plotting
    B_pred_G = B_pred * config.B_unit
    B_obs_G = B_obs * config.B_unit
    residual_G = B_pred_G - B_obs_G

    # Latitudes actually observed by WSO (|lat| <= 75 deg). Poleward values
    # in the comparison map are cubic extrapolation, which the PINN
    # deliberately does NOT fit -- exclude them from scatter and metrics.
    obs_lat_mask = np.abs(lat_deg) <= float(
        getattr(config, "OBS_MAX_ABS_LAT_DEG", 75.0))

    # ----------------------------
    # 3) Reconstructed magnetic field
    # ----------------------------
    plt.figure(figsize=(7, 5))
    plt.imshow(
        B_pred_G.T,
        aspect="auto",
        origin="lower",
        extent=[time_years[0], time_years[-1], lat_deg[0], lat_deg[-1]],
        cmap="RdBu_r",
    )
    plt.colorbar(label="Magnetic field [G]")
    plt.xlabel("Time [yr]")
    plt.ylabel("Latitude [deg]")
    plt.title("PINN Reconstructed Surface Magnetic Field")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "magnetic_field.png"), dpi=300)
    plt.close()

    # ----------------------------
    # 4) Residual map
    # ----------------------------
    vmax = np.nanpercentile(np.abs(residual_G), 98)
    plt.figure(figsize=(7, 5))
    plt.imshow(
        residual_G.T,
        aspect="auto",
        origin="lower",
        extent=[time_years[0], time_years[-1], lat_deg[0], lat_deg[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label=r"$\Delta B_r$ [G]")
    plt.xlabel("Time [yr]")
    plt.ylabel("Latitude [deg]")
    plt.title(r"Residual Map: $B_r^{\rm PINN} - B_r^{\rm WSO}$")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "residual_map.png"), dpi=300)
    plt.close()

    # ----------------------------
    # 5) Scatter plot (observed latitudes only)
    # ----------------------------
    x = B_obs_G[:, obs_lat_mask].ravel()
    y = B_pred_G[:, obs_lat_mask].ravel()

    xy_min = min(np.nanmin(x), np.nanmin(y))
    xy_max = max(np.nanmax(x), np.nanmax(y))

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(x, y, s=5, alpha=0.35)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", linewidth=1.2)
    plt.xlabel(r"Observed $B_r^{\rm WSO}$ [G]")
    plt.ylabel(r"Reconstructed $B_r^{\rm PINN}$ [G]")
    plt.title("Point-by-Point Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "scatter_pinn_vs_wso.png"), dpi=300)
    plt.close()

    # ----------------------------
    # 6) Error metrics panel
    # ----------------------------
    metrics = _compute_error_metrics(x, y)

    fig = plt.figure(figsize=(5.2, 3.0))
    ax = fig.add_subplot(111)
    ax.axis("off")

    text = (
        "Validation Metrics\n"
        r"($|\lambda| \leq %d^\circ$, observed latitudes)" % int(
            getattr(config, "OBS_MAX_ABS_LAT_DEG", 75.0)) + "\n\n"
        f"MAE   = {metrics['MAE']:.3f} G\n"
        f"RMSE  = {metrics['RMSE']:.3f} G\n"
        f"Corr  = {metrics['Corr']:.4f}\n"
        f"R$^2$ = {metrics['R2']:.4f}"
    )
    ax.text(
        0.05, 0.95, text,
        va="top", ha="left", fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "error_metrics_panel.png"), dpi=300)
    plt.close()

    # Also save metrics to text file
    with open(os.path.join(config.output_dir, "error_metrics.txt"), "w") as f:
        f.write("Validation metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # ----------------------------
    # 7) Polar-cap mean field: PINN vs WSO, per hemisphere
    # ----------------------------
    nmask = lat_deg >= 60.0
    smask = lat_deg <= -60.0
    pn_pred = np.mean(B_pred_G[:, nmask], axis=1)
    ps_pred = np.mean(B_pred_G[:, smask], axis=1)
    pn_obs = np.mean(B_obs_G[:, nmask], axis=1)
    ps_obs = np.mean(B_obs_G[:, smask], axis=1)

    plt.figure(figsize=(7.5, 3.8))
    plt.plot(time_years, pn_obs, color="C0", lw=0.9, alpha=0.55, label="WSO N")
    plt.plot(time_years, ps_obs, color="C2", lw=0.9, alpha=0.55, ls="--", label="WSO S")
    plt.plot(time_years, pn_pred, color="C1", lw=1.8, label="PINN N")
    plt.plot(time_years, ps_pred, color="C3", lw=1.8, ls="--", label="PINN S")
    plt.axhline(0.0, linestyle="--", color="k", linewidth=1.0)
    plt.xlabel("Time [yr]")
    plt.ylabel(r"Polar-cap mean field [G]")
    plt.title(r"Polar-cap Mean Field ($|\lambda| \geq 60^\circ$)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "polar_cap_mean_comparison.png"), dpi=300)
    plt.close()
    # ----------------------------
    # 8) Polarity reversal consistency check (per hemisphere,
    #    smoothed series, LAST zero crossing)
    # ----------------------------
    rev = {
        "N obs": _last_zero_crossing(time_years, pn_obs),
        "N PINN": _last_zero_crossing(time_years, pn_pred),
        "S obs": _last_zero_crossing(time_years, ps_obs),
        "S PINN": _last_zero_crossing(time_years, ps_pred),
    }

    plt.figure(figsize=(7.5, 3.8))
    plt.plot(time_years, _smooth(pn_obs), color="C0", lw=1.0, alpha=0.7,
             label="WSO N (smoothed)")
    plt.plot(time_years, _smooth(ps_obs), color="C2", lw=1.0, alpha=0.7,
             ls="--", label="WSO S (smoothed)")
    plt.plot(time_years, pn_pred, color="C1", lw=1.8, label="PINN N")
    plt.plot(time_years, ps_pred, color="C3", lw=1.8, ls="--", label="PINN S")
    plt.axhline(0.0, linestyle="--", color="k", linewidth=1.0)

    styles = {"N obs": ("C0", "--"), "N PINN": ("C1", ":"),
              "S obs": ("C2", "--"), "S PINN": ("C3", ":")}
    for name, t_rev in rev.items():
        if t_rev is not None:
            c, ls = styles[name]
            plt.axvline(t_rev, color=c, linestyle=ls, linewidth=1.2,
                        label=f"{name} reversal: {t_rev:.2f} yr")

    plt.xlabel("Time [yr]")
    plt.ylabel(r"Polar-cap mean field [G]")
    plt.title("Polarity-Reversal Consistency Check")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "polarity_reversal_check.png"), dpi=300)
    plt.close()

    # save reversal times alongside metrics
    with open(os.path.join(config.output_dir, "reversal_times.txt"), "w") as f:
        for name, t_rev in rev.items():
            f.write(f"{name}: {t_rev if t_rev is not None else 'no crossing'}\n")

    # ----------------------------
    # 9) Dipole moment
    # ----------------------------
    dip_pred = compute_dipole_moment(B_pred, lat_deg, config)

    plt.figure(figsize=(7, 4))
    plt.plot(time_years, dip_pred)
    plt.axhline(0.0, color="k", linestyle="--")
    plt.xlabel("Time [yr]")
    plt.ylabel(r"Dipole moment [$10^{22}$ Mx cm]")
    plt.title("Axial Dipole Moment Evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "dipole_moment.png"), dpi=300)
    plt.close()

    # ----------------------------
    # 10) Flux-balance check
    # ----------------------------
    net_flux = _compute_net_flux(B_pred, lat_deg, config)

    plt.figure(figsize=(7, 3.8))
    plt.plot(time_years, net_flux)
    plt.axhline(0.0, linestyle="--", color="k", linewidth=1.0)
    plt.xlabel("Time [yr]")
    plt.ylabel("Net signed flux [Mx]")
    plt.title("Flux-Balance Check")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "flux_balance_check.png"), dpi=300)
    plt.close()