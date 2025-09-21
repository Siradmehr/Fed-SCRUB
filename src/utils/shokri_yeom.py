import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import numpy as np
from ..dataloaders.client_dataloader import create_attack_and_shadow_loaders, load_datasets_with_forgetting
from ..utils.utils import load_config, load_model, set_seed, get_device, setup_experiment
import os
import logger
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

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Shokri Attack
def shokri_attack(unlearned_model, attack_loader, shadow_loader, num_shadow_models=5, device='cpu'):
    unlearned_model.eval()

    # --- infer channels/classes from a SHADOW batch (robust for disjoint attack datasets)
    first_batch = next(iter(shadow_loader))
    if isinstance(first_batch, (tuple, list)):
        X0, y0 = first_batch[:2]
    else:
        X0, y0 = first_batch, None
    input_channels = X0.shape[1] if X0.ndim == 4 else 1
    if y0 is None:
        num_classes = 10
    else:
        num_classes = int(y0.max().item()) + 1

    # --- Train multiple shadow models and collect (posteriors, member bit)
    shadow_outputs = []
    shadow_labels = []
    criterion = nn.CrossEntropyLoss()

    shadow_dataset = shadow_loader.dataset
    for _ in range(num_shadow_models):
        shadow_model = SimpleCNN(input_channels, num_classes).to(device)
        optimizer = optim.Adam(shadow_model.parameters(), lr=1e-3)

        # Split shadow data into "in" and "out"
        idx = torch.randperm(len(shadow_dataset))
        half = len(shadow_dataset) // 2
        train_idx, test_idx = idx[:half], idx[half:]
        train_loader = DataLoader(Subset(shadow_dataset, train_idx), batch_size=64, shuffle=True)

        # Train a few epochs
        for _e in range(5):
            shadow_model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(shadow_model(Xb), yb)
                loss.backward()
                optimizer.step()

        # Collect posteriors for "in" and "out"
        shadow_model.eval()
        with torch.no_grad():
            for idx_set, is_member in [(train_idx, 1), (test_idx, 0)]:
                loader = DataLoader(Subset(shadow_dataset, idx_set), batch_size=128, shuffle=False)
                for Xb, yb in loader:
                    Xb = Xb.to(device)
                    probs = shadow_model(Xb).softmax(dim=1).cpu().numpy()
                    shadow_outputs.append(probs)
                    shadow_labels.append(np.full(probs.shape[0], is_member, dtype=np.int64))

    X_attack = np.concatenate(shadow_outputs, axis=0)
    y_attack = np.concatenate(shadow_labels, axis=0)

    # --- Train the attack model (single global head on posteriors)
    attack_model = nn.Sequential(
        nn.Linear(X_attack.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)

    opt = optim.Adam(attack_model.parameters(), lr=1e-2)
    bce = nn.BCELoss()
    X_attack_tensor = torch.tensor(X_attack, dtype=torch.float32, device=device)
    y_attack_tensor = torch.tensor(y_attack, dtype=torch.float32, device=device).unsqueeze(1)

    attack_model.train()
    for _ in range(100):
        opt.zero_grad()
        p = attack_model(X_attack_tensor)
        loss = bce(p, y_attack_tensor)
        loss.backward()
        opt.step()
    attack_model.eval()

    # --- Evaluate attack on the target (F vs H provided by attack_loader)
    target_outputs = []
    target_labels = []
    with torch.no_grad():
        for batch in attack_loader:
            if isinstance(batch, (tuple, list)):
                Xb, yb, is_member = batch[:3]
            else:
                raise ValueError("attack_loader must yield (x, y, is_member)")
            Xb = Xb.to(device)
            probs = unlearned_model(Xb).softmax(dim=1).cpu().numpy()
            target_outputs.append(probs)
            target_labels.append(is_member.numpy())

    X_target = np.concatenate(target_outputs, axis=0)
    y_target = np.concatenate(target_labels, axis=0).astype(int)

    attack_preds = attack_model(torch.tensor(X_target, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
    pred_bits = (attack_preds > 0.5).astype(int)
    success_rate = accuracy_score(y_target, pred_bits) * 100.0
    return success_rate

def yeom_attack(unlearned_model, attack_loader, criterion, device):
    unlearned_model.eval()
    losses = []
    labels = []
    
    for X, y, is_member in attack_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            preds = unlearned_model(X)
            loss = criterion(preds, y).cpu().numpy()
        losses.extend(loss.tolist())
        labels.extend(is_member.numpy())
    
    # Threshold: Mean loss
    threshold = np.mean(losses)
    predictions = [1 if loss < threshold else 0 for loss in losses]
    success_rate = accuracy_score(labels, predictions) * 100
    return success_rate

# Main function to deploy attacks
def evaluate_unlearned_model(unlearned_model, forgetloader, testloader, valloader, dataset_name, device='cpu'):
    attack_loader, shadow_loader = create_attack_and_shadow_loaders(forgetloader, testloader, valloader)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    shokri_success = shokri_attack(unlearned_model, attack_loader, shadow_loader, device=device)
    yeom_success = yeom_attack(unlearned_model, attack_loader, criterion, device)
    
    print(f"\nResults for {dataset_name}:")
    print(f"Shokri Attack Success Rate: {shokri_success:.2f}%")
    print(f"Yeom Attack Success Rate: {yeom_success:.2f}%")

if __name__ == "__main__":
    config_manager = ConfigManager(os.environ["EXP_ENV_DIR"])
    custom_config = setup_experiment(
        path=os.environ["EXP_ENV_DIR"],
        load_model_flag=True
    )

    set_seed(int(custom_config["SEED"]))

    # Get partition information
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    net = custom_config["LOADED_MODEL"]
    forget_set_config = custom_config.get("FORGET_CLASS", {})
    train_loader, forget_loader, val_loader, test_loader, original_forget_loader = load_datasets_with_forgetting(
        partition_id,
        num_partitions,
        dataset_name=custom_config["DATASET"],
        forgetting_config=forget_set_config
    )
    evaluate_unlearned_model(
        unlearned_model=net,
        forgetloader=forget_loader,
        testloader=test_loader,
        valloader=val_loader,
        dataset_name=custom_config["DATASET"],
        device=device
    )