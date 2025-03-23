from flwr.simulation import run_simulation
from src.client import client_fn
from src.server import server_fn
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)

# Create your ServerApp passing the server generation function
server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=10,  # equivalent to setting `num-supernodes` in the pyproject.toml
)