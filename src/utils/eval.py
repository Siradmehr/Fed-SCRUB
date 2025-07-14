from torch import nn
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random


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




def get_loss_values(model, dataloader, device):
    """Extract loss values from a dataset using the given model."""
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in dataloader:
            # Adapt to your data loader format
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                targets = batch["label"].to(device)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
                
            outputs = model(images)
            batch_losses = criterion(outputs, targets)
            losses.extend(batch_losses.cpu().numpy())
            
    return np.array(losses)

def cm_score(estimator, X, y):
    """Calculate detailed metrics from confusion matrix."""
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    
    FP = cnf_matrix[0][1] 
    FN = cnf_matrix[1][0] 
    TP = cnf_matrix[0][0] 
    TN = cnf_matrix[1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN) if (TP+FN) > 0 else 0
    # Specificity or true negative rate
    TNR = TN/(TN+FP) if (TN+FP) > 0 else 0 
    # Fall out or false positive rate
    FPR = FP/(FP+TN) if (FP+TN) > 0 else 0
    # False negative rate
    FNR = FN/(TP+FN) if (TP+FN) > 0 else 0
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN) if (TP+FP+FN+TN) > 0 else 0
    
    print(f"ACC:{ACC:.4f}, FPR:{FPR:.4f}, FNR:{FNR:.4f}, TPR:{TPR:.4f}, TNR:{TNR:.4f}")
    return ACC

def evaluate_attack_model(features, labels, seed, n_splits=5):
    """
    Evaluate membership inference attack using cross-validation.
    
    Args:
        features: array of shape (n, 1) containing loss values
        labels: array of shape (n,) with 0 for non-members, 1 for members
        n_splits: number of cross-validation splits
        random_state: random seed for reproducibility
    
    Returns:
        Average accuracy across CV splits
    """
    unique_labels = np.unique(labels)
    if not np.all(unique_labels == np.array([0, 1])):
        raise ValueError("Labels should only have 0 and 1s")

    attack_model = LogisticRegression(max_iter=1000, solver='liblinear')
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=seed)
    
    scores = []
    for train_idx, test_idx in cv.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        attack_model.fit(X_train, y_train)
        score = cm_score(attack_model, X_test, y_test)
        scores.append(score)
    
    return np.mean(scores)

def compute_mia_score(model, val_loader, forget_loader, device, seed):
    """
    Compute membership inference attack score.
    
    Args:
        model: Model to evaluate
        val_loader: DataLoader for validation data (non-members)
        forget_loader: DataLoader for forget data (members)
        device: Device to run inference on
        random_state: Random seed for reproducibility
    
    Returns:
        MIA score (average accuracy)
    """
    # Set random seeds for reproducibility
    
    # Get loss values
    if (forget_loader is None) or len(forget_loader) == 0:
        return 0
    print("Computing loss values for validation set...")
    val_losses = get_loss_values(model, val_loader, device)
    
    print("Computing loss values for forget set...")
    forget_losses = get_loss_values(model, forget_loader, device)
    
    # Balance datasets if needed
    if len(forget_losses) > len(val_losses):
        print(f"Balancing datasets: sampling {len(val_losses)} from {len(forget_losses)} forget samples")
        forget_losses = np.random.choice(forget_losses, len(val_losses), replace=False)
    elif len(val_losses) > len(forget_losses):
        print(f"Balancing datasets: sampling {len(forget_losses)} from {len(val_losses)} validation samples")
        val_losses = np.random.choice(val_losses, len(forget_losses), replace=False)
    
    # Clip loss values to handle outliers
    val_losses = np.clip(val_losses, -100, 100)
    forget_losses = np.clip(forget_losses, -100, 100)
    
    # Print basic statistics
    print(f"Validation losses - Max: {np.max(val_losses):.4f}, Min: {np.min(val_losses):.4f}, Mean: {np.mean(val_losses):.4f}")
    print(f"Forget losses - Max: {np.max(forget_losses):.4f}, Min: {np.min(forget_losses):.4f}, Mean: {np.mean(forget_losses):.4f}")
    
    # Prepare features and labels
    val_labels = np.zeros(len(val_losses))
    forget_labels = np.ones(len(forget_losses))
    
    features = np.concatenate([val_losses, forget_losses]).reshape(-1, 1)
    labels = np.concatenate([val_labels, forget_labels])
    
    # Evaluate attack model
    print("Evaluating membership inference attack...")
    score = evaluate_attack_model(features, labels, n_splits=5, seed= int(seed)  )
    
    print(f"MIA Score: {score:.4f}")
    return score

# Example usage:
# score = compute_mia_score(model, val_loader, forget_loader, device)