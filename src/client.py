import flwr as fl
import numpy as np
from collections import OrderedDict
import torch
from dotenv import load_dotenv, dotenv_values
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

custom_config = {
        **dotenv_values("./envs/env.loss"),
        **dotenv_values("./envs/env"),
        **dotenv_values("./envs/env.training"), 
        }

DEVICE = torch.device(custom_config["DEVICE"])  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, partition_id, trainloader, valloader, forget_loader, test_loader):
        self.net = net
        self.partition_id = partition_id
        self.train_loader = trainloader
        self.valloader = valloader
        self.forgetloader = forget_loader
        self.testloader = test_loader
        self.custom_config = {
        **dotenv_values("./envs/env.loss"),
        **dotenv_values("./envs/env"),
        **dotenv_values("./envs/env.training"), 
        }

    def get_parameters(self, net) -> List[np.ndarray]:
        print(f"[Client {self.partition_id}] get_parameters")
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray], net) -> None:
        print(f"[Client {self.partition_id}] set_parameters")
        # Set net parameters from a list of numpy arrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        
    def model_train(self, net, trainloader, valloader, forgetloader, epochs: int) -> Dict:
        # TODO Implement model training with maximizer and minimizer

        
        return running_loss, correct, total
    
    def model_eval(self, net, loader) -> Dict:
        # TODO Implement model evaluation with maximizer and minimizer could be for test or evaluatin
        return running_loss, correct, total
    

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        current_phase = config.get("Phase", "MAX")
        
        # if "data_indices" in config:
        #     self.data_indices = config["data_indices"]
        
        self.set_parameters(parameters)
        
        #TODO train model
        
        # Include data indices in the metrics returned to server
        metrics = {
        }
        
        # Return updated model parameters and number of training examples
        return self.get_parameters(self.net), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        # Get current round's phase
        current_phase = config.get("Phase", 0)
        
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        self.model.eval()
        # Add your evaluation logic here
        
        # For example purposes, returning dummy metrics
        loss = 0.0
        accuracy = 0.0
        
        # Include data indices in evaluation metrics
        metrics = {
            "accuracy": accuracy,
        }
        
        return loss, len(self.test_loader.dataset), metrics

def client_fn(context: Context) -> Client:
    #TODO load model based on custome config.
    net = ().to(DEVICE)

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)