import os
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import src.extract as extract
from src.fitted_source import FittedSource

random_array_tf = tf.constant(np.zeros(32), dtype=tf.float32)

bjoy = 0.15
blat = 2.4            # meridional-flow profile exponent: v = u0 * sin(blat * lam)
ampl_used_by_time = []

LAM90_RAD = tf.constant(np.pi / 2.0, dtype=tf.float32)

initial_lats_deg = None
initial_profile_model = None

# Global fitted-source object
_fitted_source_obj = None


# tf_source_patch.py
#
# Drop-in replacement for the source-term machinery in src/sft_pde.py.
# Replaces tf.numpy_function + SciPy RegularGridInterpolator (a Python
# callback on CPU at EVERY training iteration -- the reason the fast run
# took 7.5 h) with pure-TensorFlow bilinear interpolation on a precomputed
# constant tensor. Stays inside the TF graph, works with PDEPointResampler,
# and is typically 1-2 orders of magnitude faster end-to-end.
#
# HOW TO APPLY:
#   1. In src/sft_pde.py, replace the body of init_fitted_source() and the
#      whole fitted_source_tf() with the versions below (imports unchanged;
#      you can delete the FittedSource import and the module-level
#      _fitted_source_obj).
#   2. Nothing else changes: make_pde() already calls
#      fitted_source_tf(x, config) inside pde_SFT.



# module-level state set by init_fitted_source()
_src_tensor = None        # tf.constant, shape (Nt, Nlat)
_src_meta = None          # dict with grid info


def init_fitted_source(config):
    """Load the source map once into a TF constant (model units per
    normalized time), applying pole mask, user scale, and unit conversion."""
    global _src_tensor, _src_meta

    if not getattr(config, "USE_FITTED_SOURCE", False):
        _src_tensor = None
        print("[INFO] USE_FITTED_SOURCE=False -> running without fitted source.")
        return

    source_file = getattr(config, "FITTED_SOURCE_FILE", None)
    if source_file is None:
        raise ValueError("USE_FITTED_SOURCE=True but FITTED_SOURCE_FILE not set.")

    src = np.load(source_file).astype(np.float32)        # (Nt, Nlat)
    if src.ndim != 2:
        raise ValueError(f"Expected 2D source map, got {src.shape}")
    Nt, Nlat = src.shape

    lam_min, lam_max = -0.495, 0.495
    lat_deg = np.linspace(lam_min, lam_max, Nlat) * 180.0

    mask_deg = getattr(config, "FITTED_SOURCE_MASK_POLES_ABOVE_DEG", None)
    if mask_deg is not None:
        src[:, np.abs(lat_deg) > float(mask_deg)] = 0.0

    scale = float(getattr(config, "FITTED_SOURCE_SCALE", 1.0))
    if getattr(config, "FITTED_SOURCE_IN_GAUSS_PER_YEAR", False):
        # native-units/yr -> model units per normalized time
        scale *= float(config.SIMUL_TIME) / float(config.B_unit)

    _src_tensor = tf.constant(src * scale, dtype=tf.float32)
    _src_meta = dict(Nt=Nt, Nlat=Nlat, lam_min=lam_min, lam_max=lam_max)

    print(f"[INFO] Loaded fitted source (TF-native) from: {source_file}")
    print(f"[INFO] Source scale (incl. units): {scale}")
    print(f"[INFO] Pole mask above deg: {mask_deg}")


def fitted_source_tf(x, config):
    """
    Bilinear interpolation of the source at points x, fully inside the
    TF graph.  x[:, 0] = lam_norm in [-0.495, 0.495], x[:, 1] = t_norm in
    [0, 1].  The map's time grid spans [0, SIMUL_TIME] <-> t_norm [0, 1],
    so t_norm indexes it directly.
    """
    if _src_tensor is None:
        return tf.zeros_like(x[:, 0:1])

    Nt = _src_meta["Nt"]
    Nlat = _src_meta["Nlat"]
    lam_min = _src_meta["lam_min"]
    lam_max = _src_meta["lam_max"]

    # fractional indices, clipped to the grid
    ft = tf.clip_by_value(x[:, 1] * (Nt - 1.0), 0.0, Nt - 1.0)
    fl = tf.clip_by_value(
        (x[:, 0] - lam_min) / (lam_max - lam_min) * (Nlat - 1.0),
        0.0, Nlat - 1.0)

    t0 = tf.floor(ft)
    l0 = tf.floor(fl)
    wt = ft - t0
    wl = fl - l0
    t0i = tf.cast(t0, tf.int32)
    l0i = tf.cast(l0, tf.int32)
    t1i = tf.minimum(t0i + 1, Nt - 1)
    l1i = tf.minimum(l0i + 1, Nlat - 1)

    g = lambda ti, li: tf.gather_nd(_src_tensor, tf.stack([ti, li], axis=1))
    s = ((1 - wt) * (1 - wl) * g(t0i, l0i)
         + (1 - wt) * wl * g(t0i, l1i)
         + wt * (1 - wl) * g(t1i, l0i)
         + wt * wl * g(t1i, l1i))
    return tf.reshape(s, (-1, 1))


def make_pde(config):
    """
    1D surface flux transport equation for the longitude-averaged radial
    field B(lambda, t), in conservative spherical form:

        dB/dt = (1/cos l) d/dl [ cos l * ( (eta/R^2) dB/dl - (v(l)/R) B ) ]
                - B/tau + S(l, t)

    Expanded (this is what the residual implements):

        dB/dt = - (1/R) d(vB)/dl + (1/R) tan(l) v B
                + (eta/R^2) [ d2B/dl2 - tan(l) dB/dl ]
                - B/tau + S

    Non-dimensionalization: l = pi * x0  (x0 = lat/180 in [-0.495, 0.495]),
    t = T_unit * x1, B in units of B_unit. Hence d/dl = (1/pi) d/dx0.
    """
    L_unit = float(config.L_unit)
    T_unit = float(config.T_unit)

    eta_phys = float(getattr(config, "eta_km2s", 350.0)) * 1e10   # cm^2/s
    u0_phys = float(getattr(config, "u0_ms", 11.0)) * 1e2         # cm/s, > 0 = poleward
    tau_yrs = float(getattr(config, "tau_decay_years", 8.0))
    tau_phys = tau_yrs * 365.25 * 24.0 * 3600.0

    eta_nd = eta_phys * T_unit / (L_unit ** 2)   # ~0.25 for 350 km^2/s, 11 yr
    u0_nd = u0_phys * T_unit / L_unit            # ~5.5  for 11 m/s
    tau_nd = tau_phys / T_unit                   # ~0.73 for 8 yr

    print(f"[INFO] Non-dimensional params: eta_nd={eta_nd:.4f}, "
          f"u0_nd={u0_nd:.4f}, tau_nd={tau_nd:.4f}")

    # initialize fitted source here once
    init_fitted_source(config)

    # Latitude beyond which the flow profile sin(blat*l) is cut to zero.
    # For blat = 2.4 this is pi/blat = 75 deg (van Ballegooijen-type profile).
    lam_cut = np.pi / float(blat)

    def v_flow(lam_rad):
        """Poleward meridional flow, vanishes at equator and above 75 deg."""
        v = u0_nd * tf.sin(blat * lam_rad)
        return tf.where(tf.abs(lam_rad) <= lam_cut, v, tf.zeros_like(v))

    def pde_SFT(x, y):
        lam = np.pi * x[:, 0:1]          # physical latitude [rad]
        tan_lam = tf.tan(lam)

        dB_t = dde.grad.jacobian(y, x, j=1)
        dB_l = dde.grad.jacobian(y, x, j=0) / np.pi
        dB_ll = dde.grad.hessian(y, x, i=0, j=0) / (np.pi ** 2)

        v = v_flow(lam)
        # d(vB)/dl  -- only v*B inside the derivative, nothing else
        dvB_l = dde.grad.jacobian(v_flow(np.pi * x[:, 0:1]) * y, x, j=0) / np.pi

        advection = -dvB_l + tan_lam * v * y
        diffusion = eta_nd * (dB_ll - tan_lam * dB_l)

        if getattr(config, "USE_FITTED_SOURCE", False):
            Ssrc = fitted_source_tf(x, config)
        else:
            Ssrc = tf.zeros_like(y)

        # dB/dt = advection + diffusion - B/tau + S
        # NOTE: decay enters with +y/tau_nd in the residual (it was -y/tau_nd
        # before, which made the field GROW exponentially instead of decay).
        residual = dB_t - advection - diffusion + y / tau_nd - Ssrc
        return residual

    return pde_SFT


def init(x):
    global initial_lats_deg, initial_profile_model
    if initial_lats_deg is None or initial_profile_model is None:
        raise RuntimeError("Initial profile not set from train.py")

    x = np.asarray(x)
    lat_deg = x[:, 0] * 180.0
    vals = np.interp(lat_deg, initial_lats_deg, initial_profile_model)
    return vals.reshape(-1, 1).astype(np.float32)


def boundary(x, on_boundary):
    return on_boundary
