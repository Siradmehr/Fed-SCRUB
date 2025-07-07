import os
import sys
import wandb
import torch
from flwr.simulation import run_simulation
from src.client import client_fn
from src.wandbserver import server_fn
from flwr.client import ClientApp
from flwr.server import ServerApp
from src.utils.utils import load_config, load_model, set_seed, get_device

# Configure CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Define wandb sweep configuration
sweep_config = {
    "method": "grid",  # Grid search for lr, local_epochs, and num_rounds
    "metric": {
        "name": "val_accuracy",
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

def run_experiment():
    """Run the federated learning simulation with wandb parameters (if sweeping) or custom_config."""
    # Initialize wandb if sweeping, otherwise use a single run
    if "WANDB_SWEEP_ID" in os.environ:
        wandb.init(project="federated_unlearning")
        lr = wandb.config.lr
        local_epochs = wandb.config.local_epochs
        num_rounds = wandb.config.num_rounds
    else:
        # Use default values from custom_config for a single run
        wandb.init(project="federated_unlearning", name="single_run")
        lr = float(custom_config.get("LR", 0.01))
        local_epochs = int(custom_config.get("LOCAL_EPOCHS", 1))
        num_rounds = int(custom_config.get("NUM_ROUNDS", 10))

    # Load configuration
    custom_config = load_config(os.environ["EXP_ENV_DIR"])
    
    # Update custom_config with parameters (from sweep or defaults)
    custom_config["LR"] = lr
    custom_config["LOCAL_EPOCHS"] = local_epochs
    custom_config["NUM_ROUNDS"] = num_rounds
    
    # Log experiment configuration to wandb
    wandb.config.update({
        "experiment": os.environ.get("EXP_ENV_DIR", "default_experiment"),
        "min_clients": int(custom_config.get("MIN_CLIENTS", 2)),
        "starting_phase": custom_config.get("STARTING_PHASE", "LEARN"),
        "model": custom_config.get("MODEL", "unknown_model"),
        "lr": lr,
        "local_epochs": local_epochs,
        "num_rounds": num_rounds
    })

    print("--------------------------------------------------------")
    print(f"Running experiment with lr={lr}, local_epochs={local_epochs}, num_rounds={num_rounds}")
    print(custom_config)

    # Set random seed
    set_seed(int(custom_config["SEED"]))

    # Construct the ClientApp
    client_app = ClientApp(client_fn=client_fn)

    # Construct the ServerApp
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
    # Check for sweep mode via command-line argument or environment variable
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        os.environ["WANDB_SWEEP_ID"] = "1"  # Set flag to indicate sweep mode
        sweep_id = wandb.sweep(sweep_config, project="federated_unlearning")
        print(f"Running sweep with ID: {sweep_id}")
        with open("sweep_id.txt", "w") as f:
            f.write(sweep_id)
        wandb.agent(sweep_id, function=run_experiment)
    else:
        # Run a single experiment
        run_experiment()