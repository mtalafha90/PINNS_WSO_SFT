"""
cycle_tools.py -- Shared machinery for multi-cycle SFT reconstruction,
hindcasting, and forecasting with the CORRECTED transport operator.

Everything works on a finite-volume grid in mu = sin(latitude):

    dB/dt = d/dmu [ eta (1-mu^2) dB/dmu - v(lam) sqrt(1-mu^2) B ] - B/tau + S

Units: B in model units (WSO native / B_UNIT), time in years.
Sources are stored in model units / year on the mu grid; saved .npy maps for
the PINN are converted to NATIVE units / year on a uniform lam_norm grid
(use FITTED_SOURCE_IN_GAUSS_PER_YEAR=True, FITTED_SOURCE_SCALE=1.0).
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------
# Physics (identical to the corrected PINN config)
# ----------------------------------------------------------------------
R_CM = 6.95e10
SEC_YR = 365.25 * 24 * 3600.0
ETA = 350.0e10 * SEC_YR / R_CM**2     # rad^2/yr
U0 = 11.0e2 * SEC_YR / R_CM           # rad/yr (poleward)
TAU = 8.0                              # yr
B_UNIT = 10.0
BLAT = 2.4
LAM_CUT = np.pi / BLAT                 # 75 deg

# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------
N = 181
MU_F = np.linspace(-1.0, 1.0, N + 1)
MU_C = 0.5 * (MU_F[:-1] + MU_F[1:])
DMU = MU_F[1] - MU_F[0]
LAM_C = np.arcsin(MU_C)
LAM_F = np.arcsin(np.clip(MU_F, -1, 1))
LAT_DEG = np.degrees(LAM_C)
LAMNORM_GRID = np.linspace(-0.495, 0.495, 181)   # PINN source-map latitude grid


def _v_flow(lam):
    v = U0 * np.sin(BLAT * lam)
    return np.where(np.abs(lam) <= LAM_CUT, v, 0.0)


D_F = ETA * (1.0 - MU_F**2)
V_F = _v_flow(LAM_F) * np.sqrt(np.clip(1.0 - MU_F**2, 0.0, None))


def Lop(B):
    """Corrected transport + decay operator, model units / yr. B: (..., N)."""
    Bf = np.empty(B.shape[:-1] + (N + 1,))
    Bf[..., 1:-1] = 0.5 * (B[..., :-1] + B[..., 1:])
    Bf[..., 0] = B[..., 0]
    Bf[..., -1] = B[..., -1]
    g = np.zeros_like(Bf)
    g[..., 1:-1] = (B[..., 1:] - B[..., :-1]) / DMU
    F = D_F * g - V_F * Bf
    return (F[..., 1:] - F[..., :-1]) / DMU - B / TAU


# ----------------------------------------------------------------------
# Data loading (WSO -> mu grid, model units, monopole removed)
# ----------------------------------------------------------------------
def load_cycle(cycle):
    from src.extract import build_synoptic_map
    days, lats_src, syn, _ = build_synoptic_map(f"data/{cycle}")
    idx = np.argsort(lats_src)
    lats_src = np.asarray(lats_src)[idx]
    syn = np.asarray(syn)[:, idx]
    t_obs = np.asarray(days, float) / 365.25
    t_obs = t_obs - t_obs[0]

    obs = np.empty((len(t_obs), N))
    for k, row in enumerate(syn):
        f = interp1d(lats_src, row, kind="cubic", bounds_error=False,
                     fill_value="extrapolate")
        p = f(LAT_DEG)
        obs[k] = p - p.mean()
    obs /= B_UNIT
    T_cycle = float(t_obs[-1])
    return t_obs, obs, T_cycle


def smooth_on_uniform_time(t_obs, obs, T, nt=401, sigma_t=4.0, sigma_lat=1.5):
    t_u = np.linspace(0.0, T, nt)
    out = np.empty((nt, N))
    for j in range(N):
        out[:, j] = interp1d(t_obs, obs[:, j], bounds_error=False,
                             fill_value=(obs[0, j], obs[-1, j]))(t_u)
    return t_u, gaussian_filter(out, sigma=(sigma_t, sigma_lat))


# ----------------------------------------------------------------------
# Source refit:  S = dB/dt - L[B_obs]
# ----------------------------------------------------------------------
def refit_source(t_u, obs_s, taper_deg=75.0, taper_width=5.0):
    dBdt = np.gradient(obs_s, t_u, axis=0)
    S = dBdt - Lop(obs_s)                                   # model units / yr
    w = np.clip((taper_deg - np.abs(LAT_DEG)) / taper_width, 0.0, 1.0)
    S = S * w[None, :]
    # restore exact flux balance per time slice (taper breaks it slightly);
    # remove residual monopole only inside the taper window
    corr = S.mean(axis=1, keepdims=True) * (N / max(w.sum(), 1.0))
    S = S - corr * w[None, :]
    return S


def save_source_for_pinn(S_mu, path):
    """Convert (Nt, N mu-grid, model/yr) -> (Nt, 181 lam_norm, native/yr)."""
    out = np.empty((S_mu.shape[0], LAMNORM_GRID.size))
    lat_grid = LAMNORM_GRID * 180.0
    for k in range(S_mu.shape[0]):
        out[k] = interp1d(LAT_DEG, S_mu[k], bounds_error=False,
                          fill_value=0.0)(lat_grid)
    np.save(path, out * B_UNIT)
    return out * B_UNIT


# ----------------------------------------------------------------------
# Forward solver (RK2, explicit; stable dt from diffusion limit)
# ----------------------------------------------------------------------
def forward(B0, t0, t1, S_grid, S_tgrid, nt_out=221):
    gi = RegularGridInterpolator((S_tgrid, MU_C), S_grid,
                                 bounds_error=False, fill_value=0.0)

    def S_of(t):
        return gi(np.column_stack([np.full(N, np.clip(t, S_tgrid[0], S_tgrid[-1])), MU_C]))

    def rhs(B, t):
        return Lop(B) + S_of(t)

    dt = min(0.4 * DMU**2 / (2 * ETA), 0.002)
    t_out = np.linspace(t0, t1, nt_out)
    B = B0.copy()
    out = np.empty((nt_out, N))
    out[0] = B
    t = t0
    for k in range(1, nt_out):
        while t < t_out[k] - 1e-12:
            h = min(dt, t_out[k] - t)
            k1 = rhs(B, t)
            B = B + h * rhs(B + 0.5 * h * k1, t + 0.5 * h)
            t += h
        out[k] = B
    return t_out, out


# ----------------------------------------------------------------------
# Diagnostics (native units in, native units out)
# ----------------------------------------------------------------------
NMASK = LAT_DEG >= 60.0
SMASK = LAT_DEG <= -60.0


def polar_means(B_native):
    return B_native[:, NMASK].mean(axis=1), B_native[:, SMASK].mean(axis=1)


def dipole(B_native):
    return 1.5 * np.trapezoid(B_native * MU_C[None, :], MU_C, axis=1)


def last_crossing(t, y):
    s = np.sign(y)
    out = None
    for i in range(len(y) - 1):
        if s[i] * s[i + 1] < 0:
            out = t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
    return out


def smooth1d(y, k=9):
    return np.convolve(y, np.ones(k) / k, mode="same")
