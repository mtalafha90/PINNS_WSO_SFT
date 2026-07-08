import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    python scripts/forecast_cycle25_pinn.py                       # serial
    python scripts/forecast_cycle25_pinn.py --workers 6           # 6 members at a time
    python scripts/forecast_cycle25_pinn.py --workers 6 --resume  # finish an interrupted run

Then build the comparison figure:
    python scripts/plot_forecast_comparison.py

NOTE: each member is a full PINN training (Adam + L-BFGS) and the grid below
has 36 members, so use --workers on a multi-core machine.  Members are
independent, so the speedup is ~linear in the number of workers; choose
workers x threads-per-worker ~ physical cores (e.g. 8 cores ->
--workers 4 --threads-per-worker 2).  In parallel mode each member writes
its console output to results/forecast_cycle25_mem_<tag>/train.log and the
workers are forced to CPU (export CUDA_VISIBLE_DEVICES yourself to override).
Drop config.iter_adam for a quick smoke test before the full run.
"""
import os, json, pickle
import numpy as np
# --------------------------------------------------------------------------
# NumPy pickle compatibility
# --------------------------------------------------------------------------
# Some store.pkl files created with NumPy 2.x contain references to
# numpy._core.*.  NumPy 1.x does not expose this namespace, so unpickling
# can fail with:
#     ModuleNotFoundError: No module named 'numpy._core'
#
# These aliases allow old environments to load NumPy-2-created pickles.
import sys
import importlib

try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)

    for _name in [
        "multiarray",
        "numeric",
        "umath",
        "fromnumeric",
        "shape_base",
        "_multiarray_umath",
    ]:
        try:
            _mod = importlib.import_module(f"numpy.core.{_name}")
            sys.modules.setdefault(f"numpy._core.{_name}", _mod)
        except Exception:
            pass

except Exception:
    pass

# Silence TensorFlow's C++ INFO/WARNING startup messages (oneDNN notice,
# "Could not find cuda drivers", cpu_feature_guard).  With --workers every
# member spawn would repeat these lines on the console, because they are
# printed during the child's TF import, before the per-member train.log
# redirect takes effect.  Must be set before TF is imported; export
# TF_CPP_MIN_LOG_LEVEL yourself to override.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import deepxde as dde
import tensorflow as tf
from scipy.interpolate import interp1d

# In parallel runs each worker process gets a CPU-thread budget (exported by
# main() before spawning).  Apply it before any TF op / session exists.
_thr = int(os.environ.get("TF_THREADS_PER_WORKER", "0"))
if _thr > 0:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(_thr)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # TF already initialised; the OMP/BLAS env caps still apply

from src import sft_pde
from src.sft_pde import init, make_pde
from src.config import Config
from src.extract import (build_synoptic_map, _remove_monopole_per_time,
                         get_initial_profile_from_wso)
from deepxde.icbc import PointSetBC
from src import cycle_tools as ct

# --------------------------------------------------------------------------
HERE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT   = os.path.join(HERE, "cycle_products")
store = pickle.load(open(os.path.join(OUT, "store.pkl"), "rb"))

# ---- ensemble definition (each entry = one full PINN train) ----
ANALOGS = [21, 22, 23, 24]

T_FULL_LIST = [10.5, 11.0, 11.5]
AMP_LIST = [0.85, 1.0, 1.15]

ADD_ZERO_SOURCE = False

BLEND_YR = 0.75
RECENT_SCALE_YR = 1.5
EPS = 1e-12

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

    # Refit observed Cycle-25 source up to the data horizon
    t_u, obs_s = ct.smooth_on_uniform_time(t_obs, obs, T_data, nt=201)
    S_obs = ct.refit_source(t_u, obs_s)  # model units/yr, mu-grid

    t_full = np.linspace(0.0, T_full, NPHASE)
    S_full = np.zeros((NPHASE, ct.N))

    # Interpolate observed source onto the full time grid
    S_obs_on_full = np.zeros_like(S_full)
    for j in range(ct.N):
        f_obs = interp1d(
            t_u,
            S_obs[:, j],
            bounds_error=False,
            fill_value=(S_obs[0, j], S_obs[-1, j]),
        )
        S_obs_on_full[:, j] = f_obs(t_full)

    observed = t_full <= T_data
    S_full[observed, :] = S_obs_on_full[observed, :]

    # Mean source profile just before the data horizon
    n_tail = min(8, len(S_obs))
    S_horizon = np.mean(S_obs[-n_tail:, :], axis=0)

    if analog is not None:
        Sp = phase_source(analog) * (polarity(25) * polarity(analog))

        p1 = T_data / T_full

        # Scale using only the recent observed source strength
        t0_recent = max(0.0, T_data - RECENT_SCALE_YR)
        recent = (t_u >= t0_recent) & (t_u <= T_data)

        m_tr = np.sqrt(np.mean(S_obs[recent][:, BELT] ** 2))

        p0_recent = max(0.0, t0_recent / T_full)
        m_an = amp_win(Sp, p0_recent, p1)

        Sp = Sp * (amp * m_tr / max(m_an, EPS))

        # Interpolate analogue source onto Cycle-25 full time grid
        S_analog_full = np.zeros_like(S_full)
        ph_full = np.clip(t_full / T_full, 0.0, 1.0)

        for j in range(ct.N):
            S_analog_full[:, j] = np.interp(ph_full, phase, Sp[:, j])

        # Smooth transition after data horizon
        future = t_full > T_data
        for jj in np.where(future)[0]:
            dt = t_full[jj] - T_data

            if dt < BLEND_YR:
                w = 0.5 - 0.5 * np.cos(np.pi * dt / BLEND_YR)
                S_full[jj, :] = (1.0 - w) * S_horizon + w * S_analog_full[jj, :]
            else:
                S_full[jj, :] = S_analog_full[jj, :]

    else:
        future = t_full > T_data
        for jj in np.where(future)[0]:
            dt = t_full[jj] - T_data

            if dt < BLEND_YR:
                w = 0.5 - 0.5 * np.cos(np.pi * dt / BLEND_YR)
                S_full[jj, :] = (1.0 - w) * S_horizon
            else:
                S_full[jj, :] = 0.0

    ct.save_source_for_pinn(S_full, path)
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
    data25_dir = os.path.join(HERE, "data", "25")
    out_dir = os.path.join(HERE, "results", f"forecast_cycle25_mem_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(OUT, f"fitted_source_map_cycle25_{tag}.npy")
    T_data = build_member_source(analog, T_full, amp, src_path)

    cfg = Config(mode="full", output_dir=out_dir)
    cfg.wso_path = data25_dir
    cfg.simul_time = float(T_full); cfg.SIMUL_TIME = float(T_full)
    cfg.T_unit = float(T_full) * 365.25 * 24 * 3600.0
    cfg.FITTED_SOURCE_FILE = src_path
    cfg.FITTED_SOURCE_IN_GAUSS_PER_YEAR = True
    cfg.FITTED_SOURCE_SCALE = 1.0

    # initial condition = first Cycle-25 synoptic map (model units)
    lat_init, init_prof = get_initial_profile_from_wso(
        cfg.num_lats,
        cfg.B_unit,
        data25_dir,
        getattr(cfg, "WSO_TO_GAUSS", 1.0),
    )
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
        data25_dir,
        T_full,
        cfg.B_unit,
        cfg.num_lats,
        getattr(cfg, "OBS_MAX_ABS_LAT_DEG", 75.0),
        getattr(cfg, "WSO_TO_GAUSS", 1.0),
    )
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

def _run_member(job):
    """Train one ensemble member.  Returns (tag, None) or (tag, traceback str)."""
    analog, T_full, amp, tag, log_to_file = job
    out_dir = os.path.join(HERE, "results", f"forecast_cycle25_mem_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    if log_to_file:
        # parallel run: send this member's (very verbose) TF/DeepXDE output to
        # its own file so the console only shows the start/done lines
        log = open(os.path.join(out_dir, "train.log"), "w", buffering=1)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    try:
        print(f"\n===== Cycle-25 PINN forecast member: {tag} =====", flush=True)
        train_forecast_member(analog, T_full, amp, tag)
        return tag, None
    except Exception:
        import traceback
        traceback.print_exc()
        return tag, traceback.format_exc()


def main():
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description="Cycle-25 PINN analog-ensemble forecast (36 members)")
    parser.add_argument("--workers", type=int, default=1,
                        help="members trained in parallel, each in its own process "
                             "(rule of thumb: workers x threads-per-worker ~ physical cores)")
    parser.add_argument("--threads-per-worker", type=int, default=2,
                        help="CPU threads each worker may use (default 2)")
    parser.add_argument("--resume", action="store_true",
                        help="skip members whose field.npy already exists")
    args = parser.parse_args()

    members = []
    for analog in ANALOGS:
        for T_full in T_FULL_LIST:
            for amp in AMP_LIST:
                tag = f"SC{analog}_T{T_full:.1f}_A{amp:.2f}".replace(".", "p")
                members.append((analog, T_full, amp, tag))

    if ADD_ZERO_SOURCE:
        members.append((None, 11.0, 1.0, "zero"))

    if args.resume:
        done = [m for m in members if os.path.exists(
            os.path.join(HERE, "results", f"forecast_cycle25_mem_{m[3]}", "field.npy"))]
        members = [m for m in members if m not in done]
        print(f"--resume: {len(done)} members already have field.npy, "
              f"{len(members)} left to train.")

    print(f"\nRunning {len(members)} Cycle-25 PINN forecast members "
          f"with {args.workers} worker(s).")

    failures = []
    if args.workers <= 1:
        for job in members:
            tag, err = _run_member(job + (False,))
            if err:
                failures.append(tag)
    else:
        # Cap each worker's CPU threads.  These must be in the environment
        # BEFORE the children import numpy/TF, so set them here and use the
        # 'spawn' start method (children re-import this module cleanly;
        # forking a process that has already initialised TF is unsafe).
        thr = str(args.threads_per_worker)
        os.environ.setdefault("OMP_NUM_THREADS", thr)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", thr)
        os.environ.setdefault("MKL_NUM_THREADS", thr)
        os.environ["TF_THREADS_PER_WORKER"] = thr
        # several workers sharing one GPU would fight over its memory ->
        # default the workers to CPU (export CUDA_VISIBLE_DEVICES to override)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

        ctx = mp.get_context("spawn")
        jobs = [job + (True,) for job in members]
        # maxtasksperchild=1 -> a fresh process (fresh TF graph) per member,
        # so memory does not accumulate across trainings
        with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for tag, err in pool.imap_unordered(_run_member, jobs):
                if err:
                    failures.append(tag)
                    print(f"[FAILED] {tag} -- see results/forecast_cycle25_mem_{tag}/train.log",
                          flush=True)
                else:
                    print(f"[done]   {tag}", flush=True)

    if failures:
        print(f"\n{len(failures)} member(s) failed: {', '.join(failures)}")
        print("Fix and rerun with --resume to train only the missing members.")
    else:
        print("\nAll members done. Now run:")
        print("python scripts/plot_forecast_comparison.py")

if __name__ == "__main__":
    main()
