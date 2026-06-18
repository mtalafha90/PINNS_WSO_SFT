# config.py
import numpy as np
import tensorflow as tf
from src import sft_pde  # Needed to set random_array_tf (legacy)

class Config:
    def __init__(self, mode="fast", output_dir="results/Linear_1", quenching_type="none"):
        self.output_dir = output_dir
        self.quenching_type = quenching_type  # "none", "TQ", "LQ", "TQ+LQ" (currently unused by the PDE)

        # ------------------------------------------------------------------
        # Physical units
        # ------------------------------------------------------------------
        self.simul_time = 11.0          # years
        self.B_unit = 10.0              # field normalization: model units = field / B_unit
        self.L_unit = 6.95e10           # Solar radius [cm]
        self.T_unit = self.simul_time * 365.25 * 24 * 3600   # [s]
        self.tau_decay_years = 8.0
        self.debug = (mode == "fast")
        self.seed = 42

        # WSO synoptic maps are tabulated in microtesla (1 uT = 0.01 G).
        # WSO_TO_GAUSS = 1.0 keeps the old behavior (treat numbers as-is,
        # i.e. work in "WSO native units"). Set to 0.01 to work in true
        # Gauss -- but then B_unit and the fitted source map amplitude must
        # be rescaled consistently (e.g. B_unit ~ 0.1).
        self.WSO_TO_GAUSS = 1.0

        # ------------------------------------------------------------------
        # Latitude/time domain
        # ------------------------------------------------------------------
        self.lam_min = -0.495           # normalized latitude = lat_deg / 180
        self.lam_max = 0.495            # (+-0.495 -> +-89.1 deg, avoids tan singularity)
        self.lat_min = -90.0
        self.lat_max = 90.0
        self.Tmax = 1.0

        # WSO only measures up to |lat| = arcsin(14.5/15) ~ 75.2 deg.
        # Observation constraints poleward of this are cubic extrapolation
        # artifacts -- exclude them and let the physics fill the polar caps.
        self.OBS_MAX_ABS_LAT_DEG = 75.0

        # ------------------------------------------------------------------
        # Network
        # ------------------------------------------------------------------
        self.layer_sizes = [2] + [64] * 8 + [1]
        self.activation = "tanh"
        self.initializer = "Glorot uniform"

        # ------------------------------------------------------------------
        # Training config
        # ------------------------------------------------------------------
        if mode == "fast":
            self.iter_adam = 2000
            self.num_domain = 5000
            self.num_boundary = 100
            self.num_initial = 100
            self.num_test = 200
            self.use_wso = True
        else:  # full
            self.iter_adam = 40000#95000
            self.num_domain = 20000#87460
            self.num_boundary = 2356
            self.num_initial = 2787
            self.num_test = 5000#1000
            self.use_wso = True

        self.lr = 0.0022

        # Loss weights: [PDE, Neumann BC, IC, WSO data]
        # With the corrected PDE the raw residual is O(1), so the physics
        # weight no longer needs to be crushed to 1e-8. Start here and tune;
        # if the WSO term dominates everything, lower 100 -> 30; if the
        # PDE is ignored, raise 1 -> 10.
        self.loss_weights = [1, 1, 10, 100]
        self.loss_weights_lbfgs = [1, 1, 10, 100]

        # L-BFGS stopping (the old run quit after 20 iterations on default ftol)
        self.lbfgs_maxiter = 1000
        self.lbfgs_ftol = 1e-12
        self.lbfgs_gtol = 1e-9

        # ------------------------------------------------------------------
        # Fitted source
        # ------------------------------------------------------------------
        self.USE_FITTED_SOURCE = True
        self.FITTED_SOURCE_FILE = "data/fitted_source_map.npy"
        self.FITTED_SOURCE_SCALE = 0.95   # tune ~0.7-1.2 to adjust reversals
        self.FITTED_SOURCE_MASK_POLES_ABOVE_DEG = 80.0
        # Set True if fitted_source_map.npy stores a physical emergence rate
        # in (field units)/year; the PDE then multiplies by
        # simul_time / B_unit to convert to model units per normalized time.
        # Leave False if the map is already in model units / normalized time.
        self.FITTED_SOURCE_IN_GAUSS_PER_YEAR = False
        self.SIMUL_TIME = self.simul_time

        # ------------------------------------------------------------------
        # Transport parameters
        # ------------------------------------------------------------------
        self.u0_ms = 11.0        # meridional flow amplitude [m/s], > 0 = poleward
        self.eta_km2s = 350.0    # supergranular diffusivity [km^2/s]

        # ------------------------------------------------------------------
        # Resolution
        # ------------------------------------------------------------------
        self.num_time_points = 400
        self.num_lats = 181      # latitudinal grid points for model and WSO interpolation
        self.wso_path = "data/24"

        # ------------------------------------------------------------------
        # Legacy: random per-cycle amplitude variations (not used by the
        # corrected PDE, kept so other analysis scripts don't break).
        # ------------------------------------------------------------------
        self.num_cycles = int(self.simul_time / 11) + 2
        np.random.seed(42)
        random_array = np.random.normal(loc=0.0, scale=0.13, size=self.num_cycles)
        sft_pde.random_array_tf = tf.convert_to_tensor(random_array, dtype=tf.float32)
