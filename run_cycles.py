# run_cycles.py -- train one PINN per solar cycle with the corrected physics,
# per-cycle source maps, and data-derived cycle lengths.
#
# Prerequisites:
#   * the fixed src/ files (sft_pde.py, train.py, config.py, extract.py, plot.py)
#   * data/fitted_source_map_cycleNN.npy  (from build_all_cycles.py; native units/yr)
#   * data/cycle_lengths.json             (from build_all_cycles.py)
#
# Usage:  python run_cycles.py            # all cycles
#         python run_cycles.py 24 25      # selected cycles
import os
import sys
import json
import numpy as np

from src.config import Config
from src.train import train_model
from src.plot import generate_all_plots


def make_cycle_config(cycle, mode="full"):
    with open("data/cycle_lengths.json") as f:
        lengths = json.load(f)
    T_years = float(lengths[str(cycle)])

    config = Config(mode=mode, output_dir=f"results/cycle_{cycle}")

    # --- cycle-specific data and timing ---
    config.wso_path = f"data/{cycle}"
    config.simul_time = T_years                      # ACTUAL length from data
    config.SIMUL_TIME = T_years
    config.T_unit = T_years * 365.25 * 24 * 3600.0   # keep nondim consistent

    # --- per-cycle source (saved in native units per year) ---
    config.FITTED_SOURCE_FILE = f"data/fitted_source_map_cycle{cycle}.npy"
    config.FITTED_SOURCE_IN_GAUSS_PER_YEAR = True
    config.FITTED_SOURCE_SCALE = 1.0

    return config


def main():
    cycles = [int(a) for a in sys.argv[1:]] or [21, 22, 23, 24, 25]
    summary = {}
    for c in cycles:
        print(f"\n===== CYCLE {c} =====")
        config = make_cycle_config(c)
        os.makedirs(config.output_dir, exist_ok=True)
        model, train_state = train_model(config)
        generate_all_plots(config, model, train_state)

        B = np.load(os.path.join(config.output_dir, "field.npy"))
        summary[c] = {
            "cycle_length_yr": config.simul_time,
            "field_absmax_model_units": float(np.abs(B).max()),
        }
        print(f"cycle {c} done -> {config.output_dir}")

    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
