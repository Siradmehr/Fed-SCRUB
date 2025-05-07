import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar, Context

from .utils.eval import compute_mia_score
from .utils.losses import get_losses
from .utils.utils import load_config, load_model, set_seed, get_device, setup_experiment
from .utils.models import get_model
from .dataloaders.client_dataloader import load_datasets_with_forgetting
from .utils.eval import _calculate_metrics, _eval_mode
# Set CUDA environment variables early
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Load configuration once
custom_config = load_config(os.environ["EXP_ENV_DIR"])
DEVICE = get_device(custom_config)


class FlowerClient(NumPyClient):
    def __init__(self, net, partition_id, trainloader, valloader, forget_loader, test_loader):
        self.net = net
        self.partition_id = partition_id
        self.train_loader = trainloader
        self.valloader = valloader
        self.forgetloader = forget_loader
        self.testloader = test_loader
        self.custom_config = custom_config  # Use the global config
        self.device = DEVICE
        self.net.to(self.device)
        self.best_acc = 0
        self.teacher_model = None
        print(f"Client {partition_id} initialized on device: {self.device}")

    def get_parameters(self) -> List[np.ndarray]:
        print(f"[Client {self.partition_id}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        print(f"[Client {self.partition_id}] set_parameters")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)

    def _setup_training(self, config):
        """Set up common training components"""
        lr = config.get("lr")
        num_classes = int(self.custom_config.get("NUM_CLASSES"))
        T = float(self.custom_config.get("KD_T", 2.0))

        # Get loss functions based on configuration
        loss_type_cls = self.custom_config.get("LOSSCLS", "CE")
        loss_type_div = self.custom_config.get("LOSSDIV", "KL")
        loss_type_kd = self.custom_config.get("LOSSKD", "KL")

        criterion_cls = get_losses(loss_type_cls, num_classes, T)
        criterion_div = get_losses(loss_type_div, num_classes, T)
        criterion_div_min = get_losses(loss_type_kd, num_classes, T)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        # Hyperparameters
        alpha = float(self.custom_config.get("ALPHA", 0.5))
        beta = float(self.custom_config.get("BETA", 0.5))
        gamma = float(self.custom_config.get("GAMMA", 0.5))

        return (criterion_cls, criterion_div, criterion_div_min,
                optimizer, alpha, beta, gamma)

    def _train_learn_phase(self, trainloader, epochs, criterion_cls, optimizer):
        """Train model in LEARN phase"""
        total_loss, total_correct, total_samples = 0.0, 0, 0

        print(f"LEARN phase: {epochs} epochs over {len(trainloader)} batches")
        for epoch in range(epochs):
            for batch_data in trainloader:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion_cls(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_value_(self.net.parameters(), clip_value=0.5)
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

        metrics = _calculate_metrics(total_loss, total_correct, total_samples)
        return metrics, total_samples

    def _train_max_phase(self, forgetloader, max_epochs, criterion_div_min, optimizer):
        """Train model in MAX phase to maximize divergence on forgotten data"""
        total_loss, total_correct, total_samples = 0.0, 0, 0

        if not self.teacher_model:
            raise ValueError("Teacher model not found. Run LEARN phase first.")

        print(f"MAX phase: {max_epochs} epochs over {len(forgetloader)} batches")
        self.teacher_model.eval()

        for epoch in range(max_epochs):
            for batch_data in forgetloader:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)


                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                    teacher_probs = F.softmax(teacher_outputs, dim=1)

                student_outputs = self.net(images)
                # Negative loss to maximize divergence
                loss = -criterion_div_min(student_outputs, teacher_probs)
                loss.backward()
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    preds = student_outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

        metrics = _calculate_metrics(total_loss, total_correct, total_samples)
        return metrics, total_samples

    def _train_min_phase(self, trainloader, forgetloader, min_epochs,
                         criterion_cls, criterion_div, optimizer, gamma, alpha, unlearn_con):
        """Train model in MIN phase to minimize divergence on retain data"""
        total_loss, total_correct, total_samples = 0.0, 0, 0
        forget_loss, forget_correct, forget_samples = 0.0, 0, 0

        if not self.teacher_model:
            raise ValueError("Teacher model not found. Run LEARN phase first.")

        self.teacher_model.eval()

        # Process retain data
        if trainloader and len(trainloader) > 0:
            print(f"MIN phase (retain): {min_epochs} epochs over {len(trainloader)} batches")
            total_loss, total_correct, total_samples = self._process_min_data(
                trainloader, min_epochs, criterion_cls, criterion_div,
                optimizer, gamma, alpha
            )

        # Process forget data if not using contrastive unlearning
        if unlearn_con != "TRUE" and forgetloader and len(forgetloader) > 0:
            print(f"MIN phase (forget): {min_epochs} epochs over {len(forgetloader)} batches")
            forget_loss, forget_correct, forget_samples = self._process_min_data(
                forgetloader, min_epochs, criterion_cls, criterion_div,
                optimizer, gamma, alpha
            )
            # Combine metrics
            total_loss += forget_loss
            total_correct += forget_correct
            total_samples += forget_samples

        metrics = _calculate_metrics(total_loss, total_correct, total_samples)
        return metrics, total_samples

    def _process_min_data(self, dataloader, epochs, criterion_cls, criterion_div,
                          optimizer, gamma, alpha):
        """Process data for MIN phase"""
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for epoch in range(epochs):
            for batch_data in dataloader:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)


                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                    teacher_probs = F.softmax(teacher_outputs, dim=1)

                student_outputs = self.net(images)

                loss_cls = criterion_cls(student_outputs, labels)
                loss_div = criterion_div(student_outputs, teacher_probs)

                loss = gamma * loss_cls + alpha * loss_div
                loss.backward()
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    preds = student_outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

        return total_loss, total_correct, total_samples



    def model_train(self, trainloader, forgetloader, config):
        """Train model based on the current phase"""
        phase = config.get("Phase", "LEARN")
        epochs = config.get("local_epochs", 1)
        max_epochs = config.get("max_epochs", 1)
        min_epochs = config.get("min_epochs", 1)

        print(f"Training started for phase: {phase}")

        # Set up training components
        criterion_cls, criterion_div, criterion_div_min, optimizer, alpha, beta, gamma = self._setup_training(config)

        self.net.train()
        avg_loss = 0
        total_samples = 0
        avg_loss_ = 0
        total_samples_ = 0
        accuracy = 0
        accuracy_ = 0
        # Train based on the current phase
        if phase == "LEARN":
            if trainloader and len(trainloader) > 0:
                (avg_loss, accuracy), total_samples = self._train_learn_phase(
                    trainloader, epochs, criterion_cls, optimizer
                )
            if forgetloader and len(forgetloader) > 0:
                (avg_loss_, accuracy_), total_samples_ = self._train_learn_phase(
                    forgetloader, epochs, criterion_cls, optimizer
                )
            return {"loss": (avg_loss * total_samples + avg_loss_ * total_samples_)/ (total_samples + total_samples_),
                    "accuracy": (accuracy * total_samples + accuracy_ * total_samples_)/ (total_samples + total_samples_),
                    "maxloss": 0, "maxacc": 0}, total_samples + total_samples_
        elif phase == "MAX" and config.get("UNLEARN_CON") == "TRUE":
            (avg_loss, accuracy), total_samples = self._train_max_phase(
                forgetloader, max_epochs, criterion_div_min, optimizer
            )
            return {"loss": 0, "accuracy": 0, "maxloss": avg_loss, "maxacc": accuracy}, total_samples

        elif phase == "MIN" and config.get("REMOVE") == "FALSE":
            (avg_loss, accuracy), total_samples = self._train_min_phase(
                trainloader, forgetloader, min_epochs, criterion_cls, criterion_div,
                optimizer, gamma, alpha, config.get("UNLEARN_CON")
            )
            return {"loss": avg_loss, "accuracy": accuracy, "maxloss": 0, "maxacc": 0}, total_samples

        # Default return if no phase matched or no data
        return {"loss": 0, "accuracy": 0, "maxloss": 0, "maxacc": 0}, 0


    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        self.set_parameters(parameters)

        # Initialize teacher model if needed
        if config.get("TEACHER") == "INIT":
            self.teacher_model = load_model(
                self.custom_config['MODEL'],
                self.custom_config["RESUME"]
            )
            self.teacher_model.to(self.device)

        # Train model
        metrics_dict, size_of_set = self.model_train(
            self.train_loader,
            self.forgetloader,
            config
        )

        # Collect metrics
        metrics = {
            "train_loss": metrics_dict["loss"],
            "train_accuracy": metrics_dict["accuracy"],
                "eval_loss": 0,
                "eval_acc": 0,
                "eval_size": 7,
                "max_loss": 0,
                "max_acc": 0,
                "max_size": 7,
        }
        print(f"Client {self.partition_id} metrics: {metrics}")

        return self.get_parameters(), size_of_set, metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on validation data"""
        self.set_parameters(parameters)

        # Regular evaluation
        num_classes = int(self.custom_config.get("NUM_CLASSES"))
        T = float(self.custom_config.get("KD_T", 2.0))

        loss_type_cls = self.custom_config.get("LOSSCLS", "CE")
        criterion_cls = get_losses(loss_type_cls, num_classes, T)

        loss, accuracy, eval_size = _eval_mode(criterion_cls,
                                    self.net,
                                    self.forgetloader,
                                    self.device)

        # Evaluate on forgotten data
        max_loss, max_acc, max_size = _eval_mode(criterion_cls,
                                                            self.net,
                                                            self.forgetloader,
                                                            self.device)
        # Collect metrics
        metrics = {
            "accuracy": accuracy,
            "eval_loss": loss,
            "eval_acc": accuracy,
            "eval_size": eval_size,
            "max_loss": max_loss,
            "max_acc": max_acc,
            "max_size": max_size,
        }
        print(f"Client {self.partition_id} eval metrics: {metrics}")

        return loss, eval_size, metrics


def client_fn(context: Context) -> Client:
    """Client factory function"""
    # Set environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    # Set random seed
    custom_config = setup_experiment(path=os.environ["EXP_ENV_DIR"],  load_model_flag=False)
    set_seed(int(custom_config["SEED"]))

    # Get partition information
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print(f"Initializing Client {partition_id} / {num_partitions}")

    # Create model
    net = get_model(custom_config["MODEL"])

    # Set up forget class configuration
    forget_set_config = {i: 0.0 for i in range(int(custom_config["NUM_CLASSES"]))}
    for key in custom_config["FORGET_CLASS"]:
        forget_set_config[key] = custom_config["FORGET_CLASS"][key]

    # Load datasets
    retrainloader, forgetloader, valloader, testloader = load_datasets_with_forgetting(
        partition_id,
        num_partitions,
        dataset_name=custom_config["DATASET"],
        forgetting_config=forget_set_config
    )

    # Create and return client
    return FlowerClient(
        net,
        partition_id,
        retrainloader,
        valloader,
        forgetloader,
        testloader
    ).to_client()