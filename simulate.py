import torch
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
from src.utils.utils import load_custom_config
from src.dataloaders.client_dataloader import load_datasets_with_forgetting
# Load configuration
custom_config = load_custom_config()
print(custom_config)
device = torch.device(custom_config["DEVICE"] if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

print('__Devices')
from subprocess import call
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

from flwr.simulation import run_simulation
from src.client import client_fn
from src.server import server_fn
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)
from src.utils.utils import load_custom_config
from src.dataloaders.client_dataloader import load_datasets_with_forgetting
# Load configuration
custom_config = load_custom_config()
print(custom_config)
device = torch.device(custom_config["DEVICE"] if torch.cuda.is_available() else "cpu")
print(device)
#
# print("testing dataloader")
# partition_id = 1
# num_partitions = 4
# forget_set_config = {i: 0.0 for i in range(int(custom_config["NUM_CLASSES"]))}
# for key in custom_config["FORGET_CLASS"]:
#     forget_set_config[key] = custom_config["FORGET_CLASS"][key]
#
# retrainloader, forgetloader, valloader, testloader = load_datasets_with_forgetting(partition_id, num_partitions\
#     , dataset_name=custom_config["DATASET"], forgetting_config=forget_set_config)
#
# print("testing dataloader done")

# Create your ServerApp passing the server generation function
server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=4,  # equivalent to setting `num-supernodes` in the pyproject.toml
)
