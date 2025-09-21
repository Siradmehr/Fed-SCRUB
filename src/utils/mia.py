import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
from ..dataloaders.client_dataloader import load_datasets_with_forgetting, create_attack_and_shadow_loaders
from ..utils.shokri_yeom import shokri_attack, yeom_attack, SimpleCNN, evaluate_unlearned_model

class MembershipDataset(Dataset):
    def __init__(self, dataset, is_member):
        self.dataset = dataset
        self.is_member = is_member  # 1 or 0 for all

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 2:
            x, y = data
        else:
            x, y = data[0], data[1]
        return x, y, self.is_member

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_name = "cifar10"  # Change to "mnist" or "fashionmnist" if needed
    partition_id = 0
    num_partitions = 1  # Use 1 for full dataset
    forgetting_config = {i: 0.1 for i in range(10)}  # Forget 10% from each class for demonstration

    # Load datasets using the imported function
    retrainloader, forgetloader, valloader, testloader, original_forget_loader = load_datasets_with_forgetting(
        partition_id=partition_id,
        num_partitions=num_partitions,
        seed=42,
        shuffle=True,
        forgetting_config=forgetting_config,
        dataset_name=dataset_name
    )

    # Train a simple model on retrainloader (simulating the unlearned model)
    num_classes = 10  # For CIFAR10/MNIST/FashionMNIST
    model = SimpleCNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training the model on retrain set...")
    for epoch in range(5):  # Train for 5 epochs for simplicity
        model.train()
        for X, y in retrainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5 completed.")

    # Evaluate attacks
    evaluate_unlearned_model(model, forgetloader, testloader, valloader, dataset_name, device=device)