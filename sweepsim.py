import os
import sys
import wandb
import torch
from flwr.simulation import run_simulation
from src.client import client_fn
from src.wandbserver import server_fn
from flwr.client import ClientApp
from flwr.server import ServerApp
from src.utils.utils import load_config, set_seed

# Configure CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Define wandb sweep configuration
sweep_config = {
    "method": "grid",  # Grid search for lr, local_epochs, and num_rounds
    "metric": {
        "name": "val_accuracy", # This name should match the key logged to wandb.log in server.py
        "goal": "maximize"
    },
    "parameters": {
        "lr": {
            "values": [0.01, 0.1, 1]  # Learning rate values to sweep
        },
        "local_epochs": {
            "values": [1, 2, 5]  # Local epochs values to sweep
        },
        "num_rounds": {
            "values": [10, 50, 100]  # Number of rounds to sweep
        }
    }
}

global custom_config

if "EXP_ENV_DIR" not in os.environ:
    # Example: provide a default path or raise an error
    # os.environ["EXP_ENV_DIR"] = "./config/default_experiment"
    print("Warning: EXP_ENV_DIR environment variable is not set. Please set it for proper configuration loading.")

custom_config = load_config(os.environ["EXP_ENV_DIR"])
print("Initial custom_config loaded from environment variable:")
print(custom_config)


def run_experiment():
    """Run the federated learning simulation for a single experiment run.

    When `wandb.agent` calls this function, `wandb.init()` will already have been
    set up for the current sweep run, and `wandb.config` will be populated
    with the sweep parameters. The `server_fn` will then access these directly.
    """
    print("--------------------------------------------------------")
    print("Starting simulation for current experiment run.")

    # In a sweep, parameters like lr, local_epochs, num_rounds are handled by wandb.config
    # within server_fn. No need to update custom_config here with sweep parameters.

    # Set random seed using the globally loaded custom_config
    set_seed(int(custom_config["SEED"]))

    # Construct the ClientApp
    client_app = ClientApp(client_fn=client_fn)

    
    server_app = ServerApp(server_fn=server_fn)

    # Run the simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=int(custom_config["NUM_SUPERNODES"]),
        backend_config={
            "client_resources": {
                "num_cpus": float(custom_config["CLIENT_RESOURCES_NUM_CPUS"]),
                "num_gpus": float(custom_config["CLIENT_RESOURCES_NUM_GPUS"])
            }
        }
    )

if __name__ == "__main__":
    # Check if running in sweep mode (e.g., via a command-line argument)
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        # Create the sweep and then run the agent.
        # This part is for initiating the sweep process.
        sweep_id = wandb.sweep(sweep_config, project="federated_unlearning")
        print(f"Sweep created with ID: {sweep_id}")
        wandb.agent(sweep_id, function=run_experiment)
    else:
        # Run a single experiment (no sweep)
        run_experiment()