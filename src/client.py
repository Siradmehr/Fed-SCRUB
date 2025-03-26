import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

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

from .dataloaders.client_dataloader import load_datasets, load_datasets_with_forgetting

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from .utils.losses import KL,JS,X2, get_losses
from .utils.utils import load_custom_config, load_initial_model, get_gpu
from .utils.models import get_model


custom_config = load_custom_config()

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
        self.custom_config = load_custom_config()
        self.device = torch.device(self.custom_config["DEVICE"] if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        print("gpu in client")
        get_gpu()


    def get_parameters(self) -> List[np.ndarray]:
        print(f"[Client {self.partition_id}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        print(f"[Client {self.partition_id}] set_parameters")
        # Set net parameters from a list of numpy arrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        print("before loading model statdict")
        self.net.load_state_dict(state_dict)
        print("after loading model statdict")



    def model_train(self, trainloader, valloader, forgetloader, epochs: int, phase: str) -> Dict:

        print("training started")

        num_classes = int(self.custom_config.get("NUM_CLASSES", 10))
        T = float(self.custom_config.get("KD_T", 2.0))  # Temperature for soft distillation
        loss_type_cls = self.custom_config.get("LOSSCLS", "CE")
        loss_type_div = self.custom_config.get("LOSSDIV", "KL")
        loss_type_kd = self.custom_config.get("LOSSKD", "KL")

        criterion_cls = get_losses(loss_type_cls, num_classes, T)
        criterion_div = get_losses(loss_type_div, num_classes, T)
        criterion_div_min = get_losses(loss_type_kd, num_classes, T)  # Used for MAXIMIZE phase

        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.net.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Hyperparameters
        alpha = float(self.custom_config.get("ALPHA", 0.5))
        beta = float(self.custom_config.get("BETA", 0.0))     # Optional for KD loss
        gamma = float(self.custom_config.get("GAMMA", 1.0))

        if phase == "LEARN":
            print("number of epochs in this", epochs)
            for eps in range(epochs):
                print("epoch", eps)
                for idxs, batch_data in enumerate(trainloader):
                    images = batch_data["img"]
                    labels = batch_data["label"]
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = self.net(images)
                    loss = criterion_cls(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    if idxs % 10 == 0:
                        print(idxs)
                    with torch.no_grad():
                        total_loss += loss.item() * images.size(0)
                        preds = outputs.argmax(dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += images.size(0)

            # Save as teacher for SCRUB
            # self.teacher_model =get_model(self.custom_config)
            # self.teacher_model.load_state_dict(net.state_dict())
        elif phase == "MAXIMIZE":
            if not hasattr(self, 'teacher_model'):
                raise ValueError("Teacher model not found. Run LEARN phase first.")
            teacher = self.teacher_model
            teacher.eval()

            for _ in range(epochs):
                for images, labels in forgetloader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                        teacher_probs = F.softmax(teacher_outputs, dim=1)

                    student_outputs = self.net(images)
                    loss = -criterion_div_min(student_outputs, teacher_probs)  # Maximize divergence
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        total_loss += loss.item() * images.size(0)
                        preds = student_outputs.argmax(dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += images.size(0)

        elif phase == "MINIMIZE":
            if not hasattr(self, 'teacher_model'):
                raise ValueError("Teacher model not found. Run LEARN phase first.")
            teacher = self.teacher_model
            teacher.eval()

            for _ in range(epochs):
                for images, labels in trainloader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                        teacher_probs = F.softmax(teacher_outputs, dim=1)

                    student_outputs = self.net(images)

                    loss_cls = criterion_cls(student_outputs, labels)
                    loss_div = criterion_div(student_outputs, teacher_probs)

                    loss = gamma * loss_cls + alpha * loss_div
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        total_loss += loss.item() * images.size(0)
                        preds = student_outputs.argmax(dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += images.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def model_eval(self) -> Dict:
        """Evaluate the model on a given data loader."""
        criterion = nn.CrossEntropyLoss()
        self.net.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for idxs, batch_data in enumerate(self.valloader):
                images = batch_data["img"]
                labels = batch_data["label"]
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.size(0)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss,accuracy


    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        current_phase = config.get("Phase", "LEARN")  # Default to "LEARN"
        local_epochs = config.get("local_epochs", 100)  # Default to 1 epoch
        
        self.set_parameters(parameters)
        print()
        
        metrics_dict = self.model_train(self.train_loader, self.valloader, self.forgetloader, local_epochs, phase=current_phase)    
        # Include data indices in the metrics returned to server
        metrics = {
        "train_loss": metrics_dict["loss"],
        "train_accuracy": metrics_dict["accuracy"],
        }
        
        # Return updated model parameters and number of training examples
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.set_parameters(parameters)
        loss, accuracy = self.model_eval()
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context) -> Client:
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    print("client")


    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print(f"Client {partition_id} / {num_partitions}")
    print("client calling")
    net = load_initial_model(custom_config)
    print("client loading model")

    forget_set_config = {i:0.0 for i in range(int(custom_config["NUM_CLASSES"]))}
    for key in custom_config["FORGET_CLASS"]:
        forget_set_config[key] = custom_config["FORGET_CLASS"][key]


    retrainloader, forgetloader, valloader, testloader = load_datasets_with_forgetting(partition_id, num_partitions\
    , dataset_name=custom_config["DATASET"], forgetting_config=forget_set_config)
    return FlowerClient(net, partition_id, retrainloader, valloader, forgetloader, testloader).to_client()
