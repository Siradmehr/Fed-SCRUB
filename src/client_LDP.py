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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, partition_id, trainloader, valloader, forget_loader, test_loader):
        self.net = net.to(DEVICE)
        self.partition_id = partition_id
        self.train_loader = trainloader
        self.valloader = valloader
        self.forgetloader = forget_loader
        self.testloader = test_loader
        self.custom_config = custom_config
        self.best_acc = 0
        print("gpu in client")
        get_gpu()

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        print(f"[Client {self.partition_id}] set_parameters")
        # Set net parameters from a list of numpy arrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        print("before loading model statdict")
        self.net.load_state_dict(state_dict)
        print("after loading model statdict")
    
    @staticmethod
    def clip_gradients(grads, max_norm):
        total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = min(1.0, clip_coef)
        return [g * clip_coef for g in grads]

    @staticmethod
    def add_laplace_noise(grads, epsilon, sensitivity=1.0):
        scale = sensitivity / epsilon
        return [g + torch.from_numpy(np.random.laplace(0.0, scale, size=g.shape)).to(g.device) for g in grads]

    def apply_ldp(self, epsilon):
        grads = [p.grad.detach().clone() for p in self.net.parameters() if p.grad is not None]
        clipped_grads = self.clip_gradients(grads, max_norm=1.0)
        noisy_grads = self.add_laplace_noise(clipped_grads, epsilon)
        with torch.no_grad():
            grad_idx = 0
            for param in self.net.parameters():
                if param.grad is not None:
                    param.grad = noisy_grads[grad_idx]
                    grad_idx += 1
    def get_loss_dataset(self, model, dataloader, label):
        self.net.eval()
        loss_values = []
        labels = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for batch in dataloader:
                images = batch["img"].to(DEVICE)
                targets = batch["label"].to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                clipped_loss = torch.clamp(loss, min=-400, max=400)
                loss_values.extend(clipped_loss.cpu().numpy())
                labels.extend([label] * len(targets))
        return np.array(loss_values).reshape(-1, 1), np.array(labels)

    def compute_mia_score(self, model):
        X_f, y_f = self.get_loss_dataset(model, self.forgetloader, label=1)
        X_t, y_t = self.get_loss_dataset(model, self.valloader, label=0)

        X = np.vstack([X_f, X_t])
        y = np.concatenate([y_f, y_t])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression()
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], preds)
            accs.append(acc)

        return np.mean(accs)
    def model_train(self, trainloader, valloader, forgetloader, epochs: int, phase: str, LDP: bool) -> Dict:

        print("training started")
        
        num_classes = int(self.custom_config.get("NUM_CLASSES"))
        T = float(self.custom_config.get("KD_T", 2.0))
        loss_type_cls = self.custom_config.get("LOSSCLS", "CE")
        loss_type_div = self.custom_config.get("LOSSDIV", "KL")
        loss_type_kd = self.custom_config.get("LOSSKD", "KL")

        criterion_cls = get_losses(loss_type_cls, num_classes, T)
        criterion_div = get_losses(loss_type_div, num_classes, T)
        criterion_div_min = get_losses(loss_type_kd, num_classes, T)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.net.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        alpha = float(self.custom_config.get("ALPHA", 0.5))
        gamma = float(self.custom_config.get("GAMMA", 1.0))
        epsilon = float(self.custom_config.get("LDP_EPSILON", 1.0))

        if phase == "LEARN" and not LDP:
            for eps in range(epochs):
                for idxs, batch_data in enumerate(trainloader):
                    images = batch_data["img"].to(DEVICE)
                    labels = batch_data["label"].to(DEVICE)
                    optimizer.zero_grad()
                    outputs = self.net(images)
                    loss = criterion_cls(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        total_loss += loss.item() * images.size(0)
                        preds = outputs.argmax(dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += images.size(0)

        if phase == "LEARN" and LDP:
            for eps in range(epochs):
                for idxs, batch_data in enumerate(trainloader):
                    images = batch_data["img"].to(DEVICE)
                    labels = batch_data["label"].to(DEVICE)
                    optimizer.zero_grad()
                    outputs = self.net(images)
                    loss = criterion_cls(outputs, labels)
                    loss.backward()
                    self.apply_ldp(epsilon)
                    optimizer.step()
                    with torch.no_grad():
                        total_loss += loss.item() * images.size(0)
                        preds = outputs.argmax(dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += images.size(0)

        elif phase == "MAXIMIZE":
            if not hasattr(self, 'teacher_model'):
                raise ValueError("Teacher model not found. Run LEARN phase first.")
            teacher = self.teacher_model
            teacher.eval()

            for _ in range(epochs):
                for batch_data in forgetloader:
                    images = batch_data["img"].to(DEVICE)
                    labels = batch_data["label"].to(DEVICE)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                        teacher_probs = F.softmax(teacher_outputs, dim=1)
                    student_outputs = self.net(images)
                    loss = -criterion_div_min(student_outputs, teacher_probs)
                    loss.backward()
                    if LDP:
                        self.apply_ldp(epsilon)
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
                for batch_data in trainloader:
                    images = batch_data["img"].to(DEVICE)
                    labels = batch_data["label"].to(DEVICE)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                        teacher_probs = F.softmax(teacher_outputs, dim=1)
                    student_outputs = self.net(images)
                    loss_cls = criterion_cls(student_outputs, labels)
                    loss_div = criterion_div(student_outputs, teacher_probs)
                    loss = gamma * loss_cls + alpha * loss_div
                    loss.backward()
                    if LDP:
                        self.apply_ldp(epsilon)
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
        criterion = nn.CrossEntropyLoss()
        self.net.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_data in self.valloader:
                images = batch_data["img"].to(DEVICE)
                labels = batch_data["label"].to(DEVICE)
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.size(0)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        mia_score = self.compute_mia_score(self.net)
        return avg_loss, accuracy, mia_score

    def fit(self, parameters, config):
        current_phase = config.get("Phase", "LEARN")
        local_epochs = config.get("local_epochs", 1)
        LDP = config.get("LDP", True)

        self.set_parameters(parameters)

        metrics_dict = self.model_train(self.train_loader, self.valloader, self.forgetloader, local_epochs, phase=current_phase, LDP=LDP)
        metrics = {
            "train_loss": metrics_dict["loss"],
            "train_accuracy": metrics_dict["accuracy"],
        }

        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model_eval()
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context) -> Client:
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"


    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print(f"Client {partition_id} / {num_partitions}")
    print("client calling")
    net = get_model(custom_config["MODEL"])
    print("client loading model")

    forget_set_config = {i:0.0 for i in range(int(custom_config["NUM_CLASSES"]))}
    for key in custom_config["FORGET_CLASS"]:
        forget_set_config[key] = custom_config["FORGET_CLASS"][key]


    retrainloader, forgetloader, valloader, testloader = load_datasets_with_forgetting(partition_id, num_partitions\
    , dataset_name=custom_config["DATASET"], forgetting_config=forget_set_config)
    return FlowerClient(net, partition_id, retrainloader, valloader, forgetloader, testloader).to_client()
