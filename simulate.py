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
from src.utils.utils import load_custom_config, get_gpu
from src.dataloaders.client_dataloader import load_datasets_with_forgetting
# Load configuration
custom_config = load_custom_config()
print(custom_config)
device = torch.device(custom_config["DEVICE"] if torch.cuda.is_available() else "cpu")
print(device)

server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=10,  # equivalent to setting `num-supernodes` in the pyproject.toml
    backend_config={"client_resources": {"num_cpus": 10, "num_gpus": 0.2}}
)
