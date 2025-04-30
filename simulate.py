import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch

from flwr.simulation import run_simulation
from src.client import client_fn
from src.server import server_fn
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)
from src.utils.utils import load_config, load_model, set_seed, get_device
from src.dataloaders.client_dataloader import load_datasets_with_forgetting
# Load configuration
import sys
custom_config = load_config(os.environ["EXP_ENV_DIR"])
print("--------------------------------------------------------\n")
server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=int(custom_config["NUM_SUPERNODES"]),  # equivalent to setting `num-supernodes` in the pyproject.toml
    backend_config={"client_resources": {"num_cpus": float(custom_config["CLIENT_RESOURCES_NUM_CPUS"]),
                    "num_gpus": float(custom_config["CLIENT_RESOURCES_NUM_GPUS"])}}
)
