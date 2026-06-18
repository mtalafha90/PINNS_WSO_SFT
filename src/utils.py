# utils.py
import numpy as np
import tensorflow as tf
import random

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def compute_lat_grid(config):
    """
    Build the latitude grid consistent with the model inputs.

    Returns
    -------
    lat_deg : (Nlat,) latitude centers in degrees (-90..+90)
    polar_mask : (Nlat,) boolean, True where |lat| >= 60 deg
    """
    # Use the same number of lat points as the model
    nlat = getattr(config, "num_lats", 181)
    # Model uses normalized latitude lam in [lam_min, lam_max] ~ [-0.495, 0.495]
    lam = np.linspace(config.lam_min, config.lam_max, nlat)
    lat_deg = lam * 180.0
    polar_mask = np.abs(lat_deg) >= 60.0
    return lat_deg, polar_mask

def compute_dipole_moment(B, lat_deg, config):
    """
    Returns axial dipole M(t) in 1e22 Mx·cm from longitudinally-averaged Br.

    Parameters
    ----------
    B : (Nt, Nlat) array
        Surface field in *model units* (Gauss / config.B_unit).
    lat_deg : (Nlat,) array
        Latitude centers in degrees (-90..+90).
    config : object with attributes
        - L_unit : solar radius in cm (e.g., 6.95e10)
        - B_unit : multiply model units by this to get Gauss

    Returns
    -------
    M_22 : (Nt,) array
        Axial dipole moment in units of 1e22 Mx·cm.
    """
    # Sort by latitude (ascending) to integrate over μ = sin(lat) monotonically
    idx = np.argsort(lat_deg)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)[idx]
    B = np.asarray(B, dtype=np.float64)[:, idx]

    # Convert model units → Gauss
    B_phys = B * float(config.B_unit)

    # μ = sin(latitude)
    mu = np.sin(np.deg2rad(lat_deg))  # monotonically increasing if lat_deg ascends

    # a1(t) = (3/2) ∫ B(μ,t) * μ dμ    (trapz over μ axis)
    a1 = 1.5 * _trapz(B_phys * mu[None, :], mu, axis=1)

    # M(t) = (R^3/2) * a1  → units: G·cm^3 = Mx·cm
    M = 0.5 * (float(config.L_unit) ** 3) * a1
    return (M / 1e22).astype(np.float64)

def flux_balance_check(B, lat_deg):
    """
    Area-weighted monopole (net flux) diagnostic per time slice.
    Returns an array of shape (Nt,) that should be ~0 if the map is balanced.

    B : (Nt, Nlat) in *any consistent units* (Gauss or model units)
    lat_deg : (Nlat,)
    """
    lat_rad = np.deg2rad(lat_deg)
    w = np.cos(lat_rad).clip(min=0.0)
    w = w / w.sum()
    # Weighted mean across latitude for each time
    return (B @ w).ravel()

# ---- Legacy helper (not used with WSO-driven source) -----------------
def compute_amplitudes_from_gaussians(*args, **kwargs):
    """
    Deprecated: the current pipeline uses WSO-driven sources, not the synthetic
    Gaussian emergence toy model. Keeping this stub to avoid accidental imports.
    """
    raise NotImplementedError(
        "compute_amplitudes_from_gaussians is deprecated for WSO-driven runs."
    )
