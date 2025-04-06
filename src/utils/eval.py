from torch import nn
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def _calculate_metrics(total_loss, total_correct, total_samples):
    """Calculate average metrics"""
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

def _eval_mode(loss, net, loader, device):
    """Evaluate model on forgotten data for MIA analysis"""
    criterion = loss
    net.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch_data in loader:
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss, accuracy = _calculate_metrics(total_loss, total_correct, total_samples)
    return avg_loss, accuracy, total_samples




def get_loss_dataset(net, model, dataloader, label, device):
    net.eval()
    loss_values = []
    labels = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"].to(device)
            targets = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            clipped_loss = torch.clamp(loss, min=-400, max=400)
            loss_values.extend(clipped_loss.cpu().numpy())
            labels.extend([label] * len(targets))
    return np.array(loss_values).reshape(-1, 1), np.array(labels)


def compute_mia_score(model, val, forget):
    X_f, y_f = get_loss_dataset(model, forget, label=1)
    X_t, y_t = get_loss_dataset(model, val, label=0)

    X = np.vstack([X_f, X_t])
    y = np.concatenate([y_f, y_t])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        accs.append(acc)

    return np.mean(accs)