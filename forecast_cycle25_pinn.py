#!/usr/bin/env python3
"""
forecast_cycle25_pinn.py
------------------------
PINN continuation of Cycle 25 PAST the data horizon, as an analog ensemble.

For each member (an analog cycle in {21,22,23,24}, plus a zero-future-source
member) this:

  1. builds a FULL-WINDOW source map on [0, T_full] in the PINN .npy format:
        t <= t_now :  the observationally-refit Cycle-25 source
        t >  t_now :  the analog's declining-phase source, polarity-aligned
                      and amplitude-matched to Cycle 25's observed window
     (zero member: source = 0 for t > t_now).
  2. trains a PINN over t_norm in [0,1]  <->  [0, T_full] yr, with the WSO data
     constraint placed ONLY in [0, t_now/T_full].  This is the key change vs
     run_cycles.py: get_wso_constraints() normalises the data's own time span
     to [0,1], which would stretch the 5-yr record across the whole cycle; here
     we normalise by the fixed full length so the data sits in [0, 0.46] and the
     physics + future source carry the rest.
  3. writes results/forecast_cycle25_mem_<tag>/field.npy (model units, 401x181)
     and member_meta.json (analog, T_full, amp, t_now).

Run from the repo root (needs TensorFlow + DeepXDE):
    python forecast_cycle25_pinn.py

Then build the comparison figure:
    python plot_forecast_comparison.py

NOTE: each member is a full PINN training (Adam + L-BFGS). Start with the 5
members below; drop config.iter_adam for a quick smoke test before the full run.
"""
import os, json, pickle
import numpy as np
import deepxde as dde
import tensorflow as tf
from scipy.interpolate import interp1d

from src import sft_pde
from src.sft_pde import init, make_pde
from src.config import Config
from src.extract import (build_synoptic_map, _remove_monopole_per_time,
                         get_initial_profile_from_wso)
from deepxde.icbc import PointSetBC
import cycle_tools as ct

# --------------------------------------------------------------------------
HERE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(HERE, "cycle_products")
store = pickle.load(open(os.path.join(OUT, "store.pkl"), "rb"))

# ---- ensemble definition (each entry = one full PINN train) ----
ANALOGS         = [21, 22, 23, 24]   # analog source for the future window
T_FULL          = 11.0               # assumed full Cycle-25 length [yr]
AMP             = 1.0                 # future-source amplitude factor
ADD_ZERO_SOURCE = True               # pure transport+decay (lower envelope)

NPHASE = 401
phase  = np.linspace(0.0, 1.0, NPHASE)
BELT   = np.abs(ct.LAT_DEG) < 50.0

# ---------- analog-source helpers (identical to hindcast_forecast.py) ----------
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

# ---------- full-window source for one member (saved in PINN .npy format) ----------
def build_member_source(analog, T_full, amp, path):
    t_obs, obs = store[25]["t_obs"], store[25]["obs"]
    T_data = float(t_obs[-1])
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs, obs, T_data, nt=201)
    S_obs = ct.refit_source(t_u, obs_s)                  # model units/yr, MU grid
    m_tr = np.sqrt(np.mean(S_obs[:, BELT] ** 2))

    t_full = np.linspace(0.0, T_full, NPHASE)
    S_full = np.zeros((NPHASE, ct.N))
    # observed window: real refit source
    for j in range(ct.N):
        S_full[:, j] = np.where(t_full <= T_data,
                                interp1d(t_u, S_obs[:, j], bounds_error=False,
                                         fill_value=0.0)(t_full), 0.0)
    # future window: analog declining-phase source (skip for the zero member)
    if analog is not None:
        Sp = phase_source(analog) * (polarity(25) * polarity(analog))
        p1 = T_data / T_full
        Sp = Sp * (amp * m_tr / amp_win(Sp, 0.0, p1))
        fut = t_full > T_data
        ph = np.clip(t_full[fut] / T_full, 0.0, 1.0)
        for jj, p in zip(np.where(fut)[0], ph):
            f = p * (NPHASE - 1); i0 = int(np.floor(f)); w = f - i0
            i1 = min(i0 + 1, NPHASE - 1)
            S_full[jj, :] = (1 - w) * Sp[i0, :] + w * Sp[i1, :]

    ct.save_source_for_pinn(S_full, path)   # -> native units/yr, 181 lam_norm grid
    return T_data

# ---------- WSO point constraints with FIXED-T normalisation ----------
def wso_constraints_fixedT(wso_dir, T_full, B_unit, lat_points=181,
                           max_abs_lat_deg=75.0, wso_to_gauss=1.0):
    days, lats_src, syn, _ = build_synoptic_map(wso_dir)
    idx = np.argsort(lats_src)
    lats_src = np.asarray(lats_src)[idx]
    syn = np.asarray(syn)[:, idx]
    model_lats = np.linspace(-90.0, 90.0, lat_points)
    M = np.empty((len(days), lat_points))
    for k, row in enumerate(syn):
        M[k] = interp1d(lats_src, row, kind="cubic", bounds_error=False,
                        fill_value="extrapolate")(model_lats)
    M = M * wso_to_gauss
    M = _remove_monopole_per_time(M, model_lats)
    t_years = np.asarray(days, float) / 365.25
    t_years = t_years - t_years.min()
    t_norm = t_years / float(T_full)              # <-- THE FIX (data lands in [0, T_data/T_full])
    lam_norm = model_lats / 180.0
    keep = np.where(np.abs(model_lats) <= float(max_abs_lat_deg))[0]
    X, Y = [], []
    for ti, tn in enumerate(t_norm):
        for j in keep:
            X.append([lam_norm[j], tn]); Y.append(M[ti, j] / B_unit)
    return np.asarray(X, float), np.asarray(Y, float).reshape(-1, 1), float(t_norm.max())

# ---------- one PINN training (mirrors src/train.py) ----------
def train_forecast_member(analog, T_full, amp, tag):
    out_dir = os.path.join(HERE, "results", f"forecast_cycle25_mem_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(OUT, f"fitted_source_map_cycle25_{tag}.npy")
    T_data = build_member_source(analog, T_full, amp, src_path)

    cfg = Config(mode="full", output_dir=out_dir)
    cfg.wso_path = "data/25"
    cfg.simul_time = float(T_full); cfg.SIMUL_TIME = float(T_full)
    cfg.T_unit = float(T_full) * 365.25 * 24 * 3600.0
    cfg.FITTED_SOURCE_FILE = src_path
    cfg.FITTED_SOURCE_IN_GAUSS_PER_YEAR = True
    cfg.FITTED_SOURCE_SCALE = 1.0

    # initial condition = first Cycle-25 synoptic map (model units)
    lat_init, init_prof = get_initial_profile_from_wso(
        cfg.num_lats, cfg.B_unit, "data/25", getattr(cfg, "WSO_TO_GAUSS", 1.0))
    sft_pde.initial_lats_deg = lat_init
    sft_pde.initial_profile_model = init_prof

    geom = dde.geometry.Interval(cfg.lam_min, cfg.lam_max)
    td   = dde.geometry.TimeDomain(0.0, cfg.Tmax)
    gt   = dde.geometry.GeometryXTime(geom, td)
    net  = dde.nn.FNN(cfg.layer_sizes, cfg.activation, cfg.initializer)

    ic = dde.icbc.IC(gt, init, lambda _, on_i: on_i)

    def on_bd(x, on_b):
        return bool(on_b and (np.isclose(x[0], cfg.lam_min) or np.isclose(x[0], cfg.lam_max)))
    bc = dde.OperatorBC(gt, lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=0), on_bd)

    pde = make_pde(cfg)   # loads the member source into a TF constant

    obs_X, obs_Y, t_now_norm = wso_constraints_fixedT(
        "data/25", T_full, cfg.B_unit, cfg.num_lats,
        getattr(cfg, "OBS_MAX_ABS_LAT_DEG", 75.0), getattr(cfg, "WSO_TO_GAUSS", 1.0))
    print(f"[{tag}] WSO data confined to t_norm in [0, {t_now_norm:.3f}] "
          f"({obs_X.shape[0]} points); future source = "
          f"{'analog SC%d' % analog if analog is not None else 'ZERO'}, amp {amp}")
    conditions = [ic, bc, PointSetBC(obs_X, obs_Y, component=0)]

    data = dde.data.TimePDE(gt, pde, conditions, num_test=cfg.num_test,
                            num_domain=cfg.num_domain, num_boundary=cfg.num_boundary,
                            num_initial=cfg.num_initial)
    model = dde.Model(data, net)
    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights)
    model.train(iterations=cfg.iter_adam, display_every=2000,
                callbacks=[dde.callbacks.PDEPointResampler(period=1000)])
    try:
        dde.optimizers.config.set_LBFGS_options(
            maxiter=cfg.lbfgs_maxiter, ftol=cfg.lbfgs_ftol, gtol=cfg.lbfgs_gtol)
    except AttributeError:
        dde.optimizers.set_LBFGS_options(
            maxiter=cfg.lbfgs_maxiter, ftol=cfg.lbfgs_ftol, gtol=cfg.lbfgs_gtol)
    model.compile("L-BFGS", loss_weights=cfg.loss_weights_lbfgs)
    model.train(display_every=2000)

    # field.npy on the network's own latitude domain [lam_min, lam_max]*180
    lat_deg = np.linspace(cfg.lam_min * 180.0, cfg.lam_max * 180.0, cfg.num_lats)
    t_arr = np.linspace(0.0, cfg.Tmax, cfg.num_time_points + 1)
    cols = []
    for lam in lat_deg / 180.0:
        coords = np.stack((np.full_like(t_arr, lam), t_arr), axis=1)
        cols.append(model.predict(coords).ravel())
    B = np.array(cols).T   # (Nt, Nlat), model units
    np.save(os.path.join(out_dir, "field.npy"), B)
    json.dump(dict(analog=analog, T_full=float(T_full), amp=float(amp),
                   T_data=float(T_data), t_now_phase=float(T_data / T_full),
                   lat_deg=lat_deg.tolist()),
              open(os.path.join(out_dir, "member_meta.json"), "w"), indent=2)
    print(f"[{tag}] wrote {out_dir}/field.npy")

def main():
    members = [(a, T_FULL, AMP, f"SC{a}") for a in ANALOGS]
    if ADD_ZERO_SOURCE:
        members.append((None, T_FULL, AMP, "zero"))
    for analog, T_full, amp, tag in members:
        print(f"\n===== Cycle-25 PINN forecast member: {tag} =====")
        train_forecast_member(analog, T_full, amp, tag)
    print("\nAll members done. Now run: python plot_forecast_comparison.py")

if __name__ == "__main__":
    main()
