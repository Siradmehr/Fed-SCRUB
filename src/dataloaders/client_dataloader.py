from flwr_datasets import FederatedDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import List, Dict

def load_datasets(partition_id: int, num_partitions: int, seed: int = 42, shuffle: bool = True, forgetting_config: Dict = {}, dataset_name: str = "cifar10") -> List[DataLoader]:
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": num_partitions}, shuffle=shuffle, seed=seed)
    partition = fds.load_partition(partition_id)

    # First split: 90% train+val, 10% test
    train_val_test_split = partition.train_test_split(test_size=0.1, seed=seed)

    # Second split: Split the 90% into 8/9 train (~80% of total) and 1/9 val (~10% of total)
    train_val_split = train_val_test_split["train"].train_test_split(test_size=1/9, seed=seed)

    # Define transforms
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Apply transforms to all datasets
    train_data = train_val_split["train"].with_transform(apply_transforms)
    val_data = train_val_split["test"].with_transform(apply_transforms)
    test_data = train_val_test_split["test"].with_transform(apply_transforms)

    # Create data loaders
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valloader = DataLoader(val_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, valloader, testloader