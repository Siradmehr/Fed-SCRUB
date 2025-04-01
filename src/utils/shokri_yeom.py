import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import numpy as np
from ..dataloaders.client_dataloader import create_attack_and_shadow_loaders
# Define a simple CNN (replace with your unlearned model if needed)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3 if num_classes == 10 else 1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Shokri Attack
def shokri_attack(unlearned_model, attack_loader, shadow_loader, num_shadow_models=5, device='cpu'):
    unlearned_model.eval()
    
    # Train shadow models
    shadow_outputs = []
    shadow_labels = []
    input_channels = 3 if attack_loader.dataset[0][0].shape[0] == 3 else 1
    num_classes = len(attack_loader.dataset.classes) if hasattr(attack_loader.dataset, 'classes') else 10
    
    for _ in range(num_shadow_models):
        shadow_model = SimpleCNN(num_classes).to(device)
        optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Split shadow data into "in" and "out" sets
        shadow_dataset = shadow_loader.dataset
        idx = torch.randperm(len(shadow_dataset))
        half = len(shadow_dataset) // 2
        train_idx, test_idx = idx[:half], idx[half:]
        train_loader = DataLoader(Subset(shadow_dataset, train_idx), batch_size=64, shuffle=True)
        
        # Train shadow model
        for _ in range(5):  # Few epochs for simplicity
            shadow_model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                preds = shadow_model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
        
        # Get predictions on in/out data
        shadow_model.eval()
        with torch.no_grad():
            for idx_set, is_member in [(train_idx, 1), (test_idx, 0)]:
                loader = DataLoader(Subset(shadow_dataset, idx_set), batch_size=64)
                for X, y in loader:
                    X = X.to(device)
                    preds = shadow_model(X).softmax(dim=1).cpu().numpy()
                    shadow_outputs.append(preds)
                    shadow_labels.append(np.full(preds.shape[0], is_member))
    
    X_attack = np.concatenate(shadow_outputs, axis=0)
    y_attack = np.concatenate(shadow_labels)
    
    # Train attack model
    attack_model = nn.Sequential(
        nn.Linear(X_attack.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)
    
    optimizer = optim.Adam(attack_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    X_attack_tensor = torch.tensor(X_attack, dtype=torch.float32).to(device)
    y_attack_tensor = torch.tensor(y_attack, dtype=torch.float32).unsqueeze(1).to(device)
    
    for _ in range(100):
        attack_model.train()
        optimizer.zero_grad()
        preds = attack_model(X_attack_tensor)
        loss = criterion(preds, y_attack_tensor)
        loss.backward()
        optimizer.step()
    
    # Test attack on unlearned model
    attack_model.eval()
    target_outputs = []
    target_labels = []
    with torch.no_grad():
        for X, y, is_member in attack_loader:
            X = X.to(device)
            preds = unlearned_model(X).softmax(dim=1).cpu().numpy()
            target_outputs.append(preds)
            target_labels.append(is_member.numpy())
    
    X_target = np.concatenate(target_outputs)
    y_target = np.concatenate(target_labels)
    attack_preds = attack_model(torch.tensor(X_target, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    predictions = (attack_preds > 0.5).astype(int)
    success_rate = accuracy_score(y_target, predictions) * 100
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
