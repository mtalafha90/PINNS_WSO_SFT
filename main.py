# Entry point for the SFT PINN simulation

import os
import argparse
from src.train import train_model
from src.plot import generate_all_plots
from src.config import Config


def main():
    parser = argparse.ArgumentParser(description="Train PINNs for 1D SFT solar magnetic field evolution")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Run mode: fast (low-res/dev) or full (publication-grade)")
    parser.add_argument("--output", default="results/Linear_1", help="Output directory")
    args = parser.parse_args()

    # Set CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Prepare config
    config = Config(mode=args.mode, output_dir=args.output)
    os.makedirs(config.output_dir, exist_ok=True)

    # Train model
    model, training_state = train_model(config)

    # Post-processing and plots
    generate_all_plots(config, model, training_state)


if __name__ == "__main__":
    main()
