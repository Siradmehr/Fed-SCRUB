import flwr as fl
from flwr.common import Parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dotenv import load_dotenv, dotenv_values
from flwr.server.client_manager import SimpleClientManager
from .utils.utils import load_custom_config
# Load configuration
custom_config = load_custom_config()
import torch
from collections import OrderedDict
from .utils.utils import save_model
from .utils.models import get_model
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
class PhasedFedAvg(FedAvg):
    """Federated Averaging strategy with phase control."""

    def __init__(
            self,
            starting_phase: str = "LEARN",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.current_phase = starting_phase

    def phase_schedule(self, current_phase):
        switch = {
            "LEARN": "LEARN",
            "MAXIMIZE": "MINIMIZE",
            "MINIMIZE": "MAXIMIZE"
        }
        return switch[current_phase]

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training with phase information."""
        # Determine the current phase based on round number
        # phase_idx = (server_round - 1) % len(self.phase_schedule)
        # self.current_phase = self.phase_schedule[phase_idx]

        print(f"Round {server_round}: Phase = {self.current_phase}")

        # Get default config and update with phase info
        config = {}
        config["Phase"] = self.current_phase
        config["local_epochs"] = int(custom_config.get("LOCAL_EPOCHS", 1))

        # Sample clients for the current round
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        print(f"{len(clients)} clients selected for training")

        # Return client/config pairs
        fit_configurations = []
        for client in clients:
            fit_configurations.append((client, config))
        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures
    ):
        """Aggregate model updates from clients based on participation flag."""
        print(f"Aggregating updates from {len(results)} clients, {len(failures)} failures")

        # Filter clients based on participation flag
        filtered_results = []
        for client, fit_res in results:
            metrics = fit_res.metrics
            # Check if client wants to participate in this round's aggregation
            if "participate" in metrics and metrics["participate"] == 0:
                print(f"Client {client.cid} opted out of aggregation")
                continue
            filtered_results.append((client, fit_res))

        if not filtered_results:
            print("No clients participated in the aggregation")
            return None, {}

        result = super().aggregate_fit(server_round, filtered_results, failures) # Use parent class's aggregation with filtered results
        self.current_phase = self.phase_schedule(self.current_phase)
        self.save_server_model(result, server_round)
        return result

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure evaluation with current phase."""
        config = {"Phase": self.current_phase}

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        eval_configurations = []
        for client in clients:
            eval_configurations.append((client, config))
        return eval_configurations

def save_server_model(self, result, server_round):
    if result:
        parameters, metrics = result
        model = get_model(custom_config)

        # Convert parameters to model state dict
        params_dict = zip(model.state_dict().keys(), [torch.tensor(v) for v in parameters.tensors])
        state_dict = OrderedDict({k: v for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Save model checkpoint using the utility function
        save_model(model, custom_config, f"federated_round_{server_round}",
                   epoch=server_round, loss=metrics.get("loss", None))

# def start_server():
#     """Start the Flower server with phased strategy."""
#     # Get configuration values
#
#     # Create client manager
#     client_manager = SimpleClientManager()
#
#     # Create server with client manager
#     server = fl.server.Server(client_manager=client_manager, strategy=strategy)
#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         server=server,
#         config=fl.server.ServerConfig(num_rounds=num_rounds),
#     )
#
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# def set_parameters(parameters: List[np.ndarray], net) -> None:
#     print(f"[Client {self.partition_id}] set_parameters")
#     # Set net parameters from a list of numpy arrays
#     params_dict = zip(self.net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     self.net.load_state_dict(state_dict, strict=True)

def server_fn(context: Context):
    # Read from config
    num_rounds = int(custom_config.get("NUM_ROUNDS", 3))
    min_clients = int(custom_config.get("MIN_CLIENTS", 2))

    # Initialize model parameters
    ndarrays = get_parameters(net=get_model(custom_config))
    parameters = ndarrays_to_parameters(ndarrays)


    strategy = PhasedFedAvg(
        starting_phase="LEARN",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=parameters
    )
    config = ServerConfig(num_rounds=num_rounds)
    print("server configured")

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp:wq

app = ServerApp(server_fn=server_fn)

#
# if __name__ == "__main__":
#     start_server()