import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import flwr as fl
from flwr.common import Parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dotenv import load_dotenv, dotenv_values
from flwr.server.client_manager import SimpleClientManager
from .utils.utils import load_custom_config, setup
# Load configuration
import os
custom_config = load_custom_config()
import torch
from collections import OrderedDict
from .utils.utils import save_model
from .utils.models import get_model
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
import pandas as pd

from typing import Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import time

class FedCustom(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters = None,
        starting_phase = "LEARN",
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.starting_phase = starting_phase
        self.current_phase = self.starting_phase
        self.best_acc = 0
        self.round_model = None
        self.round_log = [0,0,0,0,0,0]
        self.data_logs = pd.DataFrame(columns=["TRAINING_LOSS", "TRAINING_ACC", "FORGET_LOSS", "FORGET_ACC", "VAL_LOSS", "VAL_ACC"])



    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        print(f"Round {server_round}: Phase = {self.current_phase}")
        print(f"{len(clients)} clients selected for training")

        # Create custom configs
        if server_round % 50 == 0 and server_round > 40 and self.lr > 0.0001:
            self.lr *= float(custom_config["LR_DECAY_RATE"])
            print("UPDATING LEARNING RATE TO", self.lr)

        standard_config = {"lr": self.lr, "Phase": self.current_phase,
                           "local_epochs": int(custom_config.get("LOCAL_EPOCHS"))}
        fit_configurations = []
        for idx, client in enumerate(clients):
            # TODO choose the index of client to forget.
            fit_configurations.append((client, FitIns(parameters, standard_config)))
        return fit_configurations


    def phase_schedule(self, phase: str) -> str:
        switcher = {
            "LEARN": "LEARN",
            "MAX": "MIN",
            "MIN": "MAX"
        }
        return switcher.get(phase, "LEARN")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        print(f"Aggregating aggregate_fit updates from {len(results)} clients, {len(failures)} failures")
        print(failures)
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) 
            for _, fit_res in results if fit_res.num_examples > 0
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        self.aggr_metrices(results)
        self.save_server_model(parameters_aggregated, server_round)
        self.round_model = parameters_aggregated

        metrics_aggregated = {}
        self.current_phase = self.phase_schedule(self.current_phase)

        loss, acc = self.aggr_metrices(results)
        self.round_log[0] = loss
        self.round_log[1] = acc
        time.sleep(1)
        return parameters_aggregated, metrics_aggregated




    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {"Phase": self.current_phase}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    

    def aggr_metrices(self, results):
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, float(evaluate_res.metrics["train_loss"]))
                for _, evaluate_res in results if evaluate_res.num_examples > 0
            ]
        )
        acc_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, float(evaluate_res.metrics["train_accuracy"]))
                for _, evaluate_res in results if evaluate_res.num_examples > 0
            ]
        )
        return loss_aggregated, acc_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        print(f"Aggregating aggregate_evaluate updates from {len(results)} clients, {len(failures)} failures")
        print(failures)

        if not results:
            print("Aggregating aggregate_evaluate No results")
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        acc_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, float(evaluate_res.metrics["accuracy"]))
                for _, evaluate_res in results
            ]
        )
        print(f"round {server_round} ACC is = ", acc_aggregated)
        print(f"round {server_round} loss is = ", loss_aggregated)
        self.round_log[4] = loss_aggregated
        self.round_log[5] = acc_aggregated
        new_log = {"TRAINING_LOSS": self.round_log[0], "TRAINING_ACC": self.round_log[1]
                   , "FORGET_LOSS": self.round_log[2], "FORGET_ACC": self.round_log[3], "VAL_LOSS": self.round_log[4]
                   , "VAL_ACC":self.round_log[5]}
        new_log = {k:[new_log[k]] for k in new_log}
        new_log = pd.DataFrame(new_log)
        
        # Append the new log to the existing data_logs
        self.data_logs = pd.concat([self.data_logs, new_log], ignore_index=True)

        self.data_logs.to_csv(os.path.join(custom_config["SAVING_DIR"], "logs.csv"))

        if acc_aggregated > self.best_acc:
            self.best_acc = acc_aggregated
            print("Current best ACC is = ", acc_aggregated)
            self.save_server_model(self.round_model, server_round, True)

        metrics_aggregated = {"acc": acc_aggregated}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def save_server_model(self, params, server_round, is_best=False):
        model = get_model(custom_config["MODEL"])
        if params is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            params: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                params
            )
        params_dict = zip(model.state_dict().keys(), [torch.tensor(v) for v in params])
        state_dict = OrderedDict({k: v for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        save_model(model, custom_config, server_round, is_best=is_best)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def server_fn(context: Context):
    global custom_config
    custom_config = setup()
    print(custom_config)
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    # Read from config
    num_rounds = int(custom_config.get("NUM_ROUNDS"))
    min_clients = int(custom_config.get("MIN_CLIENTS"))
    starting_phase = custom_config.get("STARTING_PHASE")

    ndarrays = get_parameters(custom_config["LOADED_MODEL"])
    parameters = ndarrays_to_parameters(ndarrays)
    


    strategy = FedCustom(
        starting_phase=starting_phase,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=parameters,
    )
    strategy.lr = float(custom_config["LR"])
    config = ServerConfig(num_rounds=num_rounds)
    print("server configured")

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)