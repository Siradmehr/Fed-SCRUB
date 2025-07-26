#!/usr/bin/env python3
"""
Federated Learning Simulation Runner

This script orchestrates a federated learning simulation using the Flower framework.
It handles configuration loading, environment setup, and simulation execution with
proper error handling and logging.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import wandb

# Configure CUDA environment for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

# Import your custom modules
from src.client import client_fn
from src.server import server_fn
from src.utils.utils import load_config, set_seed, get_device
import wandb

class SimulationRunner:
    """Handles federated learning simulation setup and execution."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the simulation runner.

        Args:
            config_path: Path to configuration file. If None, uses EXP_ENV_DIR environment variable.
        """
        self.config_path = config_path or os.environ.get("EXP_ENV_DIR")
        self.config: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        self._load_and_validate_config()
        self._validate_environment()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('simulation.log')
            ]
        )
        return logging.getLogger(__name__)

    def _validate_environment(self) -> None:
        """Validate the runtime environment."""
        if not self.config_path:
            raise ValueError("Configuration path not provided. Set EXP_ENV_DIR environment variable.")

        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Check CUDA availability if GPU resources are requested
        if self.config.get("CLIENT_RESOURCES_NUM_GPUS", 0) > 0:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available but GPU resources requested")
            else:
                self.logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")

    def _load_and_validate_config(self) -> None:
        """Load and validate configuration parameters."""
        try:
            self.config = load_config(self.config_path)
            self.logger.info(f"Configuration loaded from: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

        # Validate required configuration keys
        required_keys = [
            "NUM_SUPERNODES",
            "CLIENT_RESOURCES_NUM_CPUS",
            "CLIENT_RESOURCES_NUM_GPUS"
        ]

        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Validate and convert numeric values
        try:
            self.config["NUM_SUPERNODES"] = int(self.config["NUM_SUPERNODES"])
            self.config["CLIENT_RESOURCES_NUM_CPUS"] = float(self.config["CLIENT_RESOURCES_NUM_CPUS"])
            self.config["CLIENT_RESOURCES_NUM_GPUS"] = float(self.config["CLIENT_RESOURCES_NUM_GPUS"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric configuration values: {e}")

        # Validate ranges
        if self.config["NUM_SUPERNODES"] <= 0:
            raise ValueError("NUM_SUPERNODES must be positive")
        if self.config["CLIENT_RESOURCES_NUM_CPUS"] <= 0:
            raise ValueError("CLIENT_RESOURCES_NUM_CPUS must be positive")
        if self.config["CLIENT_RESOURCES_NUM_GPUS"] < 0:
            raise ValueError("CLIENT_RESOURCES_NUM_GPUS cannot be negative")

    def _create_apps(self) -> tuple[ServerApp, ClientApp]:
        """Create server and client applications."""
        server_app = ServerApp(server_fn=server_fn)
        client_app = ClientApp(client_fn=client_fn)
        self.logger.info("Server and client applications created successfully")
        return server_app, client_app

    def _prepare_backend_config(self) -> Dict[str, Any]:
        """Prepare backend configuration for simulation."""
        return {
            "client_resources": {
                "num_cpus": self.config["CLIENT_RESOURCES_NUM_CPUS"],
                "num_gpus": self.config["CLIENT_RESOURCES_NUM_GPUS"]
            }
        }

    def run(self) -> None:
        """Execute the federated learning simulation."""
        try:
            self.logger.info("Starting federated learning simulation")
            self.logger.info("=" * 60)

            # Setup and validation
            self._load_and_validate_config()
            self._validate_environment()

            # Set random seed if specified
            if "RANDOM_SEED" in self.config:
                set_seed(int(self.config["RANDOM_SEED"]))
                self.logger.info(f"Random seed set to: {self.config['RANDOM_SEED']}")

            # Create applications
            server_app, client_app = self._create_apps()

            # Prepare simulation parameters
            backend_config = self._prepare_backend_config()

            self.logger.info(f"Configuration summary:")
            self.logger.info(f"  - Number of supernodes: {self.config['NUM_SUPERNODES']}")
            self.logger.info(f"  - CPU resources per client: {self.config['CLIENT_RESOURCES_NUM_CPUS']}")
            self.logger.info(f"  - GPU resources per client: {self.config['CLIENT_RESOURCES_NUM_GPUS']}")
            self.logger.info(f"  - Device: {get_device(config=self.config)}")

            # Run simulation
            self.logger.info("Launching simulation...")
            run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=self.config["NUM_SUPERNODES"],
                backend_config=backend_config
            )

            self.logger.info("Simulation completed successfully!")

        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise

import wandb

def run_exp():
    """Main entry point for the simulation."""
    runner = SimulationRunner()
        # Create server configuration
    sweep_config = dict(wandb.config)
    if sweep_config:
        runner.config.update(sweep_config)

    runner.run()

import argparse
import yaml

def main():
    runner = SimulationRunner()
    wandb.init(
        project="fed-scrub",
        name=f"server_{os.environ['EXP_ENV_DIR']}",
    )
    sweep_config = dict(wandb.config)
    sweep_config.update(runner.config)
    wandb.config.update(sweep_config)
    runner.run()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Federated Learning Simulation Runner")
    # parser.add_argument('--sweep', action='store_true', help='Run a wandb sweep using wandb_sweep.yaml')
    # args = parser.parse_args()

   # if args.sweep:
    with open("wandb_sweep.yaml") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project="fed-scrub")
    wandb.agent(sweep_id, function=main)
    # else:
    #     main()