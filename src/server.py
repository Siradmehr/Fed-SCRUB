import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Context,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from .utils.utils import save_model, load_model, set_seed, get_device, setup_experiment
from .utils.models import get_model
from .utils.lr_scheduler import FederatedScheduler

# Configure CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import wandb


class FedCustom(FedAvg):
    """Custom Federated Learning strategy with phased learning for unlearning tasks."""

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            lr: float = 0.01,
            initial_parameters: Optional[Parameters] = None,
            starting_phase: str = "LEARN",
            scheduler: Optional[FederatedScheduler] = None
    ) -> None:
        """Initialize the FedCustom strategy.

        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            initial_parameters: Initial model parameters
            starting_phase: Initial learning phase (LEARN, MAX, or MIN)
        """
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.main_teacher = initial_parameters
        self.starting_phase = starting_phase
        self.current_phase = starting_phase
        self.best_acc = 0
        self.round_model = None
        self.lr = lr  # Default value, will be overridden
        self.lr_scheduler = scheduler
        # Initialize logging
        self.round_log = [0, 0, 0, 0, 0, 0, 0]
        self.data_logs = pd.DataFrame(
            columns=["TRAINING_LOSS", "TRAINING_ACC", "FORGET_LOSS", "FORGET_ACC", "VAL_LOSS", "VAL_ACC", "MIA"]
        )
        self.max_logs = pd.DataFrame(
            columns=["MAX_LOSS", "MAX_ACC", "MIA"]
        )


    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        print(f"Round {server_round}: Phase = {self.current_phase}")
        print(f"{len(clients)} clients selected for training")

        self.lr = self.lr_scheduler.current_lr
        self.lr_scheduler.update_after_round()

        # Create base configuration
        standard_config = {
            "lr": self.lr,
            "Phase": self.current_phase,
            "min_epochs": int(custom_config.get("MIN_EPOCHS")),
            "max_epochs": int(custom_config.get("MAX_EPOCHS")),
            "local_epochs": int(custom_config.get("LOCAL_EPOCHS")),
            "UNLEARN_CON": "FALSE",
            "TEACHER": custom_config["TEACHER"],
            "REMOVE": "FALSE"
        }

        # Create client-specific configurations
        fit_configurations = []
        forget_clients = custom_config["CLIENT_ID_TO_FORGET"]
        remove_clients = custom_config["Client_ID_TO_EXIT"]
        for idx, client in enumerate(clients):
            client_config = standard_config.copy()
            if idx in forget_clients:
                print(f"Client {idx} will contribute to unlearning")
                client_config["UNLEARN_CON"] = "TRUE"
                if idx in remove_clients:
                    client_config["REMOVE"] = "TRUE"
            fit_configurations.append((client, FitIns(parameters, client_config)))

        return fit_configurations

    def phase_schedule(self, phase: str, round_num: int) -> str:
        """Determine the next phase based on current phase and round number."""
        phase_transitions = {
            "LEARN": "LEARN",
            "MAX": "MIN",
            "MIN": "MAX",
            "EXACT": "EXACT"
        }

        # Special case: stay in MIN phase if we've reached LAST_MAX_STEPS
        if custom_config["LAST_MAX_STEPS"] <= round_num and phase == "MIN":
            return "MIN"

        return phase_transitions.get(phase, "LEARN")

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], Any]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        print(f"Aggregating updates from {len(results)} clients, {len(failures)} failures")
        if failures:
            print(f"Failures: {failures}")

        # Extract valid results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results if fit_res.num_examples > 0
        ]

        print(f"Aggregating round {server_round}, phase={self.current_phase}, valid results={len(weights_results)}")

        # Aggregate parameters
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        self.round_model = parameters_aggregated

        # Aggregate metrics
        loss, acc = self.aggregate_metrics(results)

        # Store metrics based on current phase
        if self.current_phase == "MAX":
            self.round_log[2] = loss
            self.round_log[3] = acc
        else:
            self.round_log[0] = loss
            self.round_log[1] = acc

        # Aggregate MAX phase specific metrics
        metrics_aggregated = self.aggregate_max_metrics(results)

        # Save model and logs
        self.save_server_model(parameters_aggregated, server_round)
        self.save_max_logs(metrics_aggregated)

        return parameters_aggregated, metrics_aggregated

    def aggregate_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Tuple[float, float]:
        """Aggregate training metrics from client results."""
        loss_aggregated = weighted_loss_avg([
            (res.num_examples, float(res.metrics["train_loss"]))
            for _, res in results if res.num_examples > 0
        ])

        acc_aggregated = weighted_loss_avg([
            (res.num_examples, float(res.metrics["train_accuracy"]))
            for _, res in results if res.num_examples > 0
        ])

        return loss_aggregated, acc_aggregated

    def aggregate_max_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict[str, Scalar]:
        """Aggregate MAX phase specific metrics."""
        max_loss = weighted_loss_avg([
            (int(res.metrics["max_size"]), float(res.metrics["max_loss"]))
            for _, res in results if int(res.metrics["max_size"]) > 0
        ])

        max_acc = weighted_loss_avg([
            (int(res.metrics["max_size"]), float(res.metrics["max_acc"]))
            for _, res in results if int(res.metrics["max_size"]) > 0
        ])

        # mia = weighted_loss_avg([
        #     (int(res.metrics["mia_score"]), float(res.metrics["mia_score"]))
        #     for _, res in results if int(res.metrics["max_size"]) > 0
        # ])

        return {"maxloss": max_loss, "maxacc": max_acc}

    def save_max_logs(self, metrics: Dict[str, Scalar]) -> None:
        """Save MAX phase metrics to CSV."""
        new_log = {
            "MAX_LOSS": [metrics["maxloss"]],
            "MAX_ACC": [metrics["maxacc"]],
            "MIA": 0,
        }

        new_df = pd.DataFrame(new_log)
        self.max_logs = pd.concat([self.max_logs, new_df], ignore_index=True)
        self.max_logs.to_csv(os.path.join(custom_config["SAVING_DIR"], "max_logs.csv"))

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        standard_config = {
            "lr": self.lr,
            "Phase": self.current_phase,
            "min_epochs": int(custom_config.get("MIN_EPOCHS")),
            "max_epochs": int(custom_config.get("MAX_EPOCHS")),
            "local_epochs": int(custom_config.get("LOCAL_EPOCHS")),
            "UNLEARN_CON": "FALSE",
            "TEACHER": custom_config["TEACHER"],
            "REMOVE": "FALSE"
        }

        # Create client-specific configurations
        fit_configurations = []
        forget_clients = custom_config["CLIENT_ID_TO_FORGET"]
        remove_clients = custom_config["Client_ID_TO_EXIT"]
        for idx, client in enumerate(clients):
            client_config = standard_config.copy()
            if idx in forget_clients:
                print(f"Client {idx} will contribute to unlearning")
                client_config["UNLEARN_CON"] = "TRUE"
                if idx in remove_clients:
                    client_config["REMOVE"] = "TURE"
            fit_configurations.append((client, EvaluateIns(parameters, client_config)))


        return fit_configurations

        # # Create evaluation instructions
        # config = {"Phase": self.current_phase}
        # evaluate_ins = EvaluateIns(parameters, config)
        #
        # return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], Any]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        print(f"Aggregating evaluation from {len(results)} clients, {len(failures)} failures")
        if failures:
            print(f"Failures: {failures}")

        if not results:
            print("No evaluation results to aggregate")
            return None, {}

        # Aggregate loss and accuracy
        loss_aggregated = weighted_loss_avg([
            (res.num_examples, res.loss)
            for _, res in results
        ])

        acc_aggregated = weighted_loss_avg([
            (res.num_examples, float(res.metrics["accuracy"]))
            for _, res in results
        ])

        mia_list = [
            (int(res.metrics["max_size"]), float(res.metrics["mia_score"]))
            for _, res in results if int(res.metrics["max_size"]) > 0
        ]
        if len(mia_list) > 0:
            print(mia_list)
            mia = weighted_loss_avg(mia_list)
        else:
            mia = 0

        print(f"Round {server_round}, phase={self.current_phase}")
        print(f"Accuracy: {acc_aggregated:.4f}, Loss: {loss_aggregated:.4f}, MIA: {mia}")

        # Store validation metrics
        self.round_log[4] = loss_aggregated
        self.round_log[5] = acc_aggregated
        self.round_log[6] = mia

        # Save round logs
        self.save_round_logs()

        # Check for best model
        if acc_aggregated > self.best_acc:
            self.best_acc = acc_aggregated
            print(f"New best accuracy: {acc_aggregated:.4f}")
            self.save_server_model(self.round_model, server_round, is_best=True)

        # Update phase for next round
        self.current_phase = self.phase_schedule(self.current_phase, server_round)

        return loss_aggregated, {"acc": acc_aggregated}

    def save_round_logs(self) -> None:
        """Save round logs to CSV file."""
        new_log = {
            "TRAINING_LOSS": [self.round_log[0]],
            "TRAINING_ACC": [self.round_log[1]],
            "FORGET_LOSS": [self.round_log[2]],
            "FORGET_ACC": [self.round_log[3]],
            "VAL_LOSS": [self.round_log[4]],
            "VAL_ACC": [self.round_log[5]],
            "MIA": [self.round_log[6]],
        }

        new_df = pd.DataFrame(new_log)
        self.data_logs = pd.concat([self.data_logs, new_df], ignore_index=True)
        self.data_logs.to_csv(os.path.join(custom_config["SAVING_DIR"], "logs.csv"))

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # No server-side evaluation
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Calculate number of clients to use for training."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Calculate number of clients to use for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def save_server_model(self, params: Parameters, server_round: int, is_best: bool = False) -> None:
        """Save the current server model."""
        if params is None:
            return

        print(f"Saving round {server_round} model{' (best)' if is_best else ''}...")

        # Convert parameters to model state dict
        model = get_model(custom_config["MODEL"])
        param_list = parameters_to_ndarrays(params)
        state_dict = OrderedDict({
            k: torch.tensor(v)
            for k, v in zip(model.state_dict().keys(), param_list)
        })

        # Load parameters and save model
        model.load_state_dict(state_dict, strict=True)
        save_model(model, custom_config, server_round, is_best=is_best)


def get_parameters(net) -> List[np.ndarray]:
    """Extract model parameters as NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

import os
def overwite_wandb_config(custom_config):
    """Overwrite wandb config with custom_config values."""
    if wandb.run is not None:
        for key in wandb.config:
            if key in wandb.config:
                custom_config[key] = wandb.config[key]
            else:
                print(f"Warning: {key} not found in custom_config")
    else:
        print("No active wandb run to overwrite config")
    return custom_config
def server_fn(context: Context) -> ServerAppComponents:
    
    global custom_config
    """Server factory function."""
            


    # Setup configuration
    custom_config = setup_experiment(os.environ["EXP_ENV_DIR"])
    custom_config = overwite_wandb_config(custom_config)

    set_seed(int(custom_config["SEED"]))
    print(custom_config)

    # Configure server
    num_rounds = int(custom_config.get("NUM_ROUNDS"))
    min_clients = int(custom_config.get("MIN_CLIENTS"))
    starting_phase = custom_config.get("STARTING_PHASE")

    # Get initial model parameters
    ndarrays = get_parameters(custom_config["LOADED_MODEL"])
    parameters = ndarrays_to_parameters(ndarrays)

    # Create strategy
    strategy = FedCustom(
        starting_phase=starting_phase,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=parameters,
        lr=float(custom_config["LR"]),
        scheduler=FederatedScheduler(),
    )
    print(strategy.lr_scheduler)
    # wandb.init(
    #     project="fed-scrub",  # Change to your project name
    #     name=f"client",  # Unique run name per client
    #     config=custom_config  # Log your config
    # )
    
    config = ServerConfig(num_rounds=num_rounds)
    print("Server configured")
    print(wandb.config)
    print("server config wandb done")
    return ServerAppComponents(strategy=strategy, config=config)

def create_server_app() -> ServerApp:
    return ServerApp(server_fn=server_fn)