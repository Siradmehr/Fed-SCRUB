import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import logging
import flwr as fl
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar, Context
from numpy.f2py.auxfuncs import throw_error

from .utils.eval import compute_mia_score
from .utils.losses import get_loss
from .utils.utils import load_config, load_model, set_seed, get_device, setup_experiment
from .utils.models import get_model
from .dataloaders.client_dataloader import load_datasets_with_forgetting
from .utils.eval import _calculate_metrics, _eval_mode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Enumeration for training phases"""
    LEARN = "LEARN"
    MAX = "MAX"
    MIN = "MIN"
    EXACT = "EXACT"
    PRETRAIN = "PRETRAIN"


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    lr: float
    local_epochs: int
    max_epochs: int
    min_epochs: int
    alpha: float
    beta: float
    gamma: float
    phase: TrainingPhase
    remove: bool
    unlearn_con: bool
    teacher_init: bool


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    loss: float = 0.0
    accuracy: float = 0.0
    samples: int = 0


class ConfigManager:
    """Manages configuration loading and environment setup"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self._setup_cuda_environment()

    def _load_config(self) -> dict:
        """Load configuration from file"""
        try:
            return load_config(self.config_path)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device"""
        return get_device(self.config)

    def _setup_cuda_environment(self) -> None:
        """Setup CUDA environment variables"""
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['TORCH_USE_CUDA_DSA'] = "1"


class LossManager:
    """Manages different loss functions used in training"""

    def __init__(self, config: dict):
        self.config = config
        self.num_classes = int(config.get("NUM_CLASSES"))
        self.temperature = float(config.get("KD_T"))

        self.criterion_cls = self._get_loss_function(config.get("LOSSCLS"))
        self.criterion_div = self._get_loss_function(config.get("LOSSDIV"))
        self.criterion_kd = self._get_loss_function(config.get("LOSSKD"))

    def _get_loss_function(self, loss_type: str):
        """Get loss function based on type"""
        try:
            return get_loss(loss_type, nclass=self.num_classes, param=[self.temperature])
        except Exception as e:
            logger.error(f"Failed to create loss function {loss_type}: {e}")
            raise


class PhaseTrainer:
    """Handles training logic for different phases"""

    def __init__(self, model: nn.Module, device: torch.device, loss_manager: LossManager):
        self.model = model
        self.device = device
        self.loss_manager = loss_manager
        self.teacher_model: Optional[nn.Module] = None

    def set_teacher_model(self, teacher_model: nn.Module) -> None:
        """Set the teacher model for knowledge distillation"""
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def train_learn_phase(self, dataloader, epochs: int, optimizer) -> Tuple[TrainingMetrics, int]:
        """Train model in LEARN phase"""
        if not dataloader or len(dataloader) == 0:
            return TrainingMetrics(), 0

        logger.info(f"LEARN phase: {epochs} epochs over {len(dataloader)} batches")

        total_loss, total_correct, total_samples = 0.0, 0, 0

        self.model.train()
        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_manager.criterion_cls(outputs, labels)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

        loss_avg, accuracy = _calculate_metrics(total_loss, total_correct, total_samples)
        return TrainingMetrics(loss=loss_avg, accuracy=accuracy, samples=total_samples), total_samples

    def train_max_phase(self, forget_loader, epochs: int, optimizer) -> Tuple[TrainingMetrics, int]:
        """Train model in MAX phase to maximize divergence on forgotten data"""
        if not forget_loader or len(forget_loader) == 0:
            return TrainingMetrics(), 0

        if not self.teacher_model:
            raise ValueError("Teacher model required for MAX phase")

        logger.info(f"MAX phase: {epochs} epochs over {len(forget_loader)} batches")

        total_loss, total_correct, total_samples = 0.0, 0, 0

        self.model.train()
        self.teacher_model.eval()

        for epoch in range(epochs):
            for images, labels in forget_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                    teacher_probs = F.softmax(teacher_outputs, dim=1)

                student_outputs = self.model(images)
                # Negative loss to maximize divergence
                loss = -self.loss_manager.criterion_kd(student_outputs, teacher_probs)
                loss.backward()
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    preds = student_outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

        loss_avg, accuracy = _calculate_metrics(total_loss, total_correct, total_samples)
        return TrainingMetrics(loss=loss_avg, accuracy=accuracy, samples=total_samples), total_samples

    def train_min_phase(self, retain_loader, forget_loader, epochs: int,
                        optimizer, config: TrainingConfig) -> Tuple[TrainingMetrics, int]:
        """Train model in MIN phase to minimize divergence on retain data"""
        if not self.teacher_model:
            raise ValueError("Teacher model required for MIN phase")

        total_metrics = TrainingMetrics()
        total_samples = 0

        # Process retain data
        if retain_loader and len(retain_loader) > 0:
            logger.info(f"MIN phase (retain): {epochs} epochs over {len(retain_loader)} batches")
            metrics, samples = self._process_min_data(
                retain_loader, epochs, optimizer, config.gamma, config.alpha
            )
            total_metrics.loss += metrics.loss * samples
            total_metrics.accuracy += metrics.accuracy * samples
            total_samples += samples

        # Process forget data if not using contrastive unlearning
        if not config.unlearn_con and forget_loader and len(forget_loader) > 0:
            logger.info(f"MIN phase (forget): {epochs} epochs over {len(forget_loader)} batches")
            metrics, samples = self._process_min_data(
                forget_loader, epochs, optimizer, config.gamma, config.alpha
            )
            total_metrics.loss += metrics.loss * samples
            total_metrics.accuracy += metrics.accuracy * samples
            total_samples += samples

        if total_samples > 0:
            total_metrics.loss /= total_samples
            total_metrics.accuracy /= total_samples
        total_metrics.samples = total_samples

        return total_metrics, total_samples

    def _process_min_data(self, dataloader, epochs: int, optimizer,
                          gamma: float, alpha: float) -> Tuple[TrainingMetrics, int]:
        """Process data for MIN phase"""
        total_loss, total_correct, total_samples = 0.0, 0, 0

        self.model.train()
        self.teacher_model.eval()

        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                    teacher_probs = F.softmax(teacher_outputs, dim=1)

                student_outputs = self.model(images)

                loss_cls = self.loss_manager.criterion_cls(student_outputs, labels)
                loss_div = self.loss_manager.criterion_div(student_outputs, teacher_probs)

                loss = gamma * loss_cls + alpha * loss_div
                loss.backward()
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    preds = student_outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

        loss_avg, accuracy = _calculate_metrics(total_loss, total_correct, total_samples)
        return TrainingMetrics(loss=loss_avg, accuracy=accuracy), total_samples


class FlowerClient(NumPyClient):
    """Improved Flower client for federated unlearning"""

    def __init__(self, net: nn.Module, partition_id: int, config_manager: ConfigManager,
                 train_loader, val_loader, forget_loader, test_loader):
        self.net = net
        self.partition_id = partition_id
        self.config_manager = config_manager
        self.device = config_manager.device

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.forget_loader = forget_loader
        self.test_loader = test_loader

        # Initialize components
        self.loss_manager = LossManager(config_manager.config)
        self.phase_trainer = PhaseTrainer(self.net, self.device, self.loss_manager)

        # Move model to device
        self.net.to(self.device)


        logger.info(f"Client {partition_id} initialized on device: {self.device}")

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        logger.debug(f"[Client {self.partition_id}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        logger.debug(f"[Client {self.partition_id}] set_parameters")
        try:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
            raise

    def _parse_config(self, config: dict) -> TrainingConfig:
        """Parse configuration into TrainingConfig object"""
        try:
            return TrainingConfig(
                lr=config.get("lr", 0.001),
                local_epochs=config.get("local_epochs", 1),
                max_epochs=config.get("max_epochs", 1),
                min_epochs=config.get("min_epochs", 1),
                alpha=float(config.get("ALPHA", 0.5)),
                beta=float(config.get("BETA", 0.5)),
                gamma=float(config.get("GAMMA", 0.5)),
                phase=TrainingPhase(config.get("Phase", "LEARN")),
                remove=config.get("REMOVE", "FALSE") == "TRUE",
                unlearn_con=config.get("UNLEARN_CON", "FALSE") == "TRUE",
                teacher_init=config.get("TEACHER", "") == "INIT"
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid configuration: {e}")
            raise

    def _initialize_teacher_model(self) -> None:
        """Initialize teacher model if needed"""
        try:
            teacher_model = load_model(
                self.config_manager.config['MODEL'],
                self.config_manager.config["RESUME"]
            )
            teacher_model.to(self.device)
            self.phase_trainer.set_teacher_model(teacher_model)
            logger.info("Teacher model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize teacher model: {e}")
            raise

    def _train_model(self, training_config: TrainingConfig) -> Tuple[dict, int]:
        """Train model based on configuration"""
        logger.info(f"Starting training phase: {training_config.phase}")
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=training_config.lr,
            betas=(0.9, 0.999)
        )


        # Training logic based on phase
        print("Training logic based on phase", training_config.phase)
        if training_config.phase == TrainingPhase.PRETRAIN:
            return self._handle_pretrain_phase(training_config, optimizer)
        if training_config.phase in [TrainingPhase.LEARN, TrainingPhase.EXACT] and not training_config.remove:
            return self._handle_learn_phase(training_config, optimizer)
        elif training_config.phase == TrainingPhase.MAX and training_config.unlearn_con:
            return self._handle_max_phase(training_config, optimizer)
        elif training_config.phase == TrainingPhase.MIN and not training_config.remove:
            return self._handle_min_phase(training_config, optimizer)
        else:
            logger.error("No matching training phase found")
            exit(-1)

    def _handle_pretrain_phase(self, config: TrainingConfig, optimizer) -> Tuple[dict, int]:
        metrics_retain, samples_retain = self.phase_trainer.train_learn_phase(
            self.train_loader, config.local_epochs, optimizer
        )

        metrics_forget, samples_forget = TrainingMetrics(), 0
        if self.forget_loader:
            metrics_forget, samples_forget = self.phase_trainer.train_learn_phase(
                self.forget_loader, config.local_epochs, optimizer
            )

        total_samples = samples_retain + samples_forget
        if total_samples > 0:
            combined_loss = (metrics_retain.loss * samples_retain +
                             metrics_forget.loss * samples_forget) / total_samples
            combined_acc = (metrics_retain.accuracy * samples_retain +
                            metrics_forget.accuracy * samples_forget) / total_samples
        else:
            combined_loss = combined_acc = 0

        return {
            "loss": combined_loss,
            "accuracy": combined_acc,
        }, total_samples

    def _handle_learn_phase(self, config: TrainingConfig, optimizer) -> Tuple[dict, int]:
        """Handle LEARN phase training"""
        metrics_retain, samples_retain = self.phase_trainer.train_learn_phase(
            self.train_loader, config.local_epochs, optimizer
        )

        metrics_forget, samples_forget = TrainingMetrics(), 0
        if self.forget_loader and not config.unlearn_con:
            metrics_forget, samples_forget = self.phase_trainer.train_learn_phase(
                self.forget_loader, config.local_epochs, optimizer
            )

        total_samples = samples_retain + samples_forget
        if total_samples > 0:
            combined_loss = (metrics_retain.loss * samples_retain +
                             metrics_forget.loss * samples_forget) / total_samples
            combined_acc = (metrics_retain.accuracy * samples_retain +
                            metrics_forget.accuracy * samples_forget) / total_samples
        else:
            combined_loss = combined_acc = 0

        return {
            "loss": combined_loss,
            "accuracy": combined_acc,
        }, total_samples

    def _handle_max_phase(self, config: TrainingConfig, optimizer) -> Tuple[dict, int]:
        """Handle MAX phase training"""
        metrics, samples = self.phase_trainer.train_max_phase(
            self.forget_loader, config.max_epochs, optimizer
        )

        return {
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
        }, samples

    def _handle_min_phase(self, config: TrainingConfig, optimizer) -> Tuple[dict, int]:
        """Handle MIN phase training"""
        metrics, samples = self.phase_trainer.train_min_phase(
            self.train_loader, self.forget_loader, config.min_epochs, optimizer, config
        )

        return {
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
        }, samples

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        """Fit the model with given parameters and configuration"""
        logger.info(f"[Client {self.partition_id}] fit, config: {config}")

        try:
            self.set_parameters(parameters)
            training_config = self._parse_config(config)

            # Initialize teacher model if needed
            if training_config.teacher_init:
                self._initialize_teacher_model()

            # Train model
            metrics_dict, num_samples = self._train_model(training_config)

            # Prepare metrics
            metrics = {
                "train_loss": metrics_dict["loss"],
                "train_accuracy": metrics_dict["accuracy"],
            }

            logger.info(f"Client {self.partition_id} training completed: {metrics}")
            return self.get_parameters(), num_samples, metrics

        except Exception as e:
            logger.error(f"Training failed for client {self.partition_id}: {e}")
            raise

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        """Evaluate the model"""
        logger.info(f"[Client {self.partition_id}] evaluate")

        try:
            self.set_parameters(parameters)
            training_config = self._parse_config(config)
            print(TrainingConfig)

            loss, accuracy, eval_size = _eval_mode(
                self.loss_manager.criterion_cls,
                self.net,
                self.val_loader,
                self.device
            )

            # Evaluate on forgotten data
            max_loss, max_acc, max_size = 0, 0, 0
            if training_config.unlearn_con and self.forget_loader:
                max_loss, max_acc, max_size = _eval_mode(
                    self.loss_manager.criterion_cls,
                    self.net,
                    self.forget_loader,
                    self.device
                )

            # Calculate MIA score
            mia_score = compute_mia_score(
                self.net,
                self.val_loader,
                self.forget_loader,
                self.device,
                self.config_manager.config["SEED"]
            )

            metrics = {
                "eval_loss": loss,
                "eval_acc": accuracy,
                "eval_size": eval_size,
                "forget_loss": max_loss,
                "forget_acc": max_acc,
                "forget_size": max_size,
                "mia_score": mia_score,
            }

            logger.info(f"Client {self.partition_id} evaluation completed: {metrics}")
            return loss, eval_size, metrics

        except Exception as e:
            logger.error(f"Evaluation failed for client {self.partition_id}: {e}")
            raise


def client_fn(context: Context) -> Client:
    """Client factory function"""
    try:
        # Initialize configuration
        config_manager = ConfigManager(os.environ["EXP_ENV_DIR"])

        # Set up experiment
        custom_config = setup_experiment(
            path=os.environ["EXP_ENV_DIR"],
            load_model_flag=False
        )
        set_seed(int(custom_config["SEED"]))

        # Get partition information
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        logger.info(f"Initializing Client {partition_id} / {num_partitions}")

        # Create model
        net = get_model(custom_config["MODEL"])

        # Set up forget class configuration
        forget_set_config = {i: 0.0 for i in range(int(custom_config["NUM_CLASSES"]))}
        forget_set_config.update(custom_config.get("FORGET_CLASS", {}))

        # Load datasets
        train_loader, forget_loader, val_loader, test_loader = load_datasets_with_forgetting(
            partition_id,
            num_partitions,
            dataset_name=custom_config["DATASET"],
            forgetting_config=forget_set_config
        )

        # Create and return client
        return FlowerClient(
            net,
            partition_id,
            config_manager,
            train_loader,
            val_loader,
            forget_loader,
            test_loader
        ).to_client()

    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        raise