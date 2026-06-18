# train.py
import numpy as np
import deepxde as dde
import tensorflow as tf
from . import sft_pde
from .sft_pde import init, make_pde
from .utils import set_random_seed
from .extract import get_wso_constraints, get_initial_profile_from_wso
from deepxde.icbc import PointSetBC


def train_model(config):
    set_random_seed(42)

    # Build initial condition directly from the first WSO synoptic snapshot
    lat_init_deg, init_profile_model = get_initial_profile_from_wso(
        lat_points=config.num_lats,
        B_unit=config.B_unit,
        data_dir=config.wso_path,
        unit_to_gauss=getattr(config, "WSO_TO_GAUSS", 1.0),
    )
    # Hand the initial profile to sft_pde
    sft_pde.initial_lats_deg = lat_init_deg
    sft_pde.initial_profile_model = init_profile_model

    # Geometry and domain
    geom = dde.geometry.Interval(config.lam_min, config.lam_max)
    timedomain = dde.geometry.TimeDomain(0.0, config.Tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Network
    net = dde.nn.FNN(config.layer_sizes, config.activation, config.initializer)

    # ---- Initial condition in MODEL UNITS ----
    ic = dde.icbc.IC(geomtime, init, lambda _, on_initial: on_initial)

    # Polar Neumann(0): dB/dlam = 0 at the latitude edges only.
    def on_theta_boundary(x, on_boundary):
        return bool(on_boundary and (np.isclose(x[0], config.lam_min)
                                     or np.isclose(x[0], config.lam_max)))

    # FIX: pass j=0 explicitly. With j=None, dde.grad.jacobian returns the
    # gradient w.r.t. ALL inputs, so the old BC also penalized (dB/dt)^2 at
    # the poles -- freezing the polar field in time and fighting the
    # polar-field reversal we are trying to model.
    bc_pole_neumann = dde.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=0),
        on_theta_boundary,
    )

    # PDE
    pde_fn = make_pde(config)

    # Conditions
    conditions = [ic, bc_pole_neumann]

    # WSO point constraints (globally normalized time axis)
    if config.use_wso:
        obs_X, obs_Y = get_wso_constraints(
            Tmax=config.Tmax,
            lat_points=config.num_lats,
            time_steps=config.num_time_points,
            B_unit=config.B_unit,
            data_dir=config.wso_path,
            unit_to_gauss=getattr(config, "WSO_TO_GAUSS", 1.0),
            max_abs_lat_deg=getattr(config, "OBS_MAX_ABS_LAT_DEG", 75.0),
        )
        conditions.append(PointSetBC(obs_X, obs_Y, component=0))

    # Data
    data = dde.data.TimePDE(
        geomtime,
        pde_fn,
        conditions,
        num_test=config.num_test,
        num_domain=config.num_domain,
        num_boundary=config.num_boundary,
        num_initial=config.num_initial,
    )

    # Training
    model = dde.Model(data, net)
    model.compile("adam", lr=config.lr, loss_weights=config.loss_weights)
    model.train(iterations=config.iter_adam, display_every=1000, callbacks=[dde.callbacks.PDEPointResampler(period=1000)])

    # FIX: the previous run's L-BFGS stopped after ~20 iterations because the
    # default ftol triggered on a badly scaled loss. Loosen the stopping
    # criteria and allow many more iterations.
    try:
        dde.optimizers.config.set_LBFGS_options(
            maxiter=getattr(config, "lbfgs_maxiter", 20000),
            ftol=getattr(config, "lbfgs_ftol", 1e-12),
            gtol=getattr(config, "lbfgs_gtol", 1e-9),
        )
    except AttributeError:
        dde.optimizers.set_LBFGS_options(
            maxiter=getattr(config, "lbfgs_maxiter", 20000),
            ftol=getattr(config, "lbfgs_ftol", 1e-12),
            gtol=getattr(config, "lbfgs_gtol", 1e-9),
        )

    model.compile("L-BFGS", loss_weights=config.loss_weights_lbfgs)
    losshistory, train_state = model.train(display_every=1000)

    return model, train_state
