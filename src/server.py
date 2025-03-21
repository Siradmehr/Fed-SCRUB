import flwr as fl
from flwr.common import Parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dotenv import load_dotenv, dotenv_values
from flwr.server.client_manager import SimpleClientManager
from utils.utils import load_custom_config
# Load configuration
custom_config = load_custom_config()


class PhasedFedAvg(FedAvg):
    """Federated Averaging strategy with phase control."""

    def __init__(
            self,
            *args,
            phase_schedule: List[str] = ["LEARN", "MAXIMIZE", "MINIMIZE"],
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.phase_schedule = phase_schedule
        self.current_phase = phase_schedule[0]

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training with phase information."""
        # Determine the current phase based on round number
        phase_idx = (server_round - 1) % len(self.phase_schedule)
        self.current_phase = self.phase_schedule[phase_idx]

        print(f"Round {server_round}: Phase = {self.current_phase}")

        # Get default config and update with phase info
        config = {}
        config["Phase"] = self.current_phase
        config["local_epochs"] = int(custom_config.get("LOCAL_EPOCHS", 1))

        # Sample clients for the current round
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

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

        # Use parent class's aggregation with filtered results
        return super().aggregate_fit(server_round, filtered_results, failures)

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


def start_server():
    """Start the Flower server with phased strategy."""
    # Get configuration values
    num_rounds = int(custom_config.get("NUM_ROUNDS", 3))
    min_clients = int(custom_config.get("MIN_CLIENTS", 2))
    phase_schedule = custom_config.get("PHASE_SCHEDULE", "LEARN,MAXIMIZE,MINIMIZE").split(",")

    strategy = PhasedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        phase_schedule=phase_schedule,
    )

    # Create client manager
    client_manager = SimpleClientManager()

    # Create server with client manager
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        server=server,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )


if __name__ == "__main__":
    start_server()