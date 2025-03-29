from flwr_datasets import FederatedDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Optional
from collections import defaultdict
import random
from ..utils.utils import load_config as load_custom_config

# Define transforms once outside the function
pytorch_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_datasets_with_forgetting(
        partition_id: int,
        num_partitions: int,
        seed: int = 42,
        shuffle: bool = True,
        forgetting_config: Dict = {},
        dataset_name: str = "cifar10"
) -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader, DataLoader]:
    """
    Load and partition datasets with forgetting functionality.

    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the data
        forgetting_config: Configuration for class-specific forgetting rates
        dataset_name: Name of the dataset to load

    Returns:
        Tuple of DataLoaders: (retrainloader, forgetloader, valloader, testloader)
    """
    custom_config = load_custom_config()

    print("Loading data")
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": num_partitions},
                           shuffle=shuffle, seed=seed)
    print(f"Loading data completed, start partitioning for {partition_id}")
    partition = fds.load_partition(partition_id)
    print(f"Partition loaded with {len(partition)} samples")

    # First split: 90% train+val, 10% test
    train_val_test_split = partition.train_test_split(test_size=0.1, seed=seed)

    # Second split: Split the 90% into 8/9 train (~80% of total) and 1/9 val (~10% of total)
    train_val_split = train_val_test_split["train"].train_test_split(test_size=1 / 9, seed=seed)

    # Apply transforms to the datasets
    train_data = train_val_split["train"].with_transform(apply_transforms)
    val_data = train_val_split["test"].with_transform(apply_transforms)
    test_data = train_val_test_split["test"].with_transform(apply_transforms)

    # Group indices by class
    class_indices = defaultdict(list)
    for i, x in enumerate(train_data):
        class_indices[x["label"]].append(i)

    forget_indices = []
    retrain_indices = []

    # Split data into forget and retrain sets based on forgetting_config
    for cls, indices in class_indices.items():
        if cls in forgetting_config:
            random.seed(seed)  # Ensure reproducibility
            random.shuffle(indices)
            forget_count = int(len(indices) * forgetting_config[cls])
            forget_indices.extend(indices[:forget_count])
            retrain_indices.extend(indices[forget_count:])
        else:
            retrain_indices.extend(indices)

    print(f"Forget set: {len(forget_indices)} samples, Retrain set: {len(retrain_indices)} samples")

    # Create subsets
    forgetset = Subset(train_data, forget_indices)
    retrainset = Subset(train_data, retrain_indices)

    print(f"Final sizes - Retrain: {len(retrainset)}, Forget: {len(forgetset)}, "
          f"Validation: {len(val_data)}, Test: {len(test_data)}")

    # Get batch sizes from config
    retrain_batch = custom_config["RETRAIN_BATCH"]
    forget_batch = custom_config["FORGET_BATCH"]
    val_batch = custom_config["VAL_BATCH"]
    test_batch = custom_config["TEST_BATCH"]

    # Create data loaders
    retrainloader = DataLoader(retrainset, batch_size=retrain_batch, shuffle=True) if len(retrainset) > 0 else None
    forgetloader = DataLoader(forgetset, batch_size=forget_batch, shuffle=True) if len(forgetset) > 0 else None
    valloader = DataLoader(val_data, batch_size=val_batch, shuffle=True)
    testloader = DataLoader(test_data, batch_size=test_batch, shuffle=True)

    return retrainloader, forgetloader, valloader, testloader