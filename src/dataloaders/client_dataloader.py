from flwr_datasets import FederatedDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import Subset
import random
from ..utils.utils import load_custom_config

def load_datasets(partition_id: int, num_partitions: int, seed: int = 42, shuffle: bool = True, dataset_name: str = "cifar10") -> tuple[DataLoader, DataLoader, DataLoader]:
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

    custom_config = load_custom_config()
    RETRAIN_BATCH = custom_config["RETRAIN_BATCH"]
    VAL_BATCH = custom_config["VAL_BATCH"]
    TEST_BATCH = custom_config["TEST_BATCH"]

    # Create data loaders
    trainloader = DataLoader(train_data, batch_size=RETRAIN_BATCH, shuffle=True)
    valloader = DataLoader(val_data, batch_size=VAL_BATCH)
    testloader = DataLoader(test_data, batch_size=TEST_BATCH)

    return trainloader, valloader, testloader

pytorch_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch
def load_datasets_with_forgetting(partition_id: int, num_partitions: int, seed: int = 42, shuffle: bool = True,
                                  forgetting_config: Dict = {}, dataset_name: str = "cifar10") -> tuple[
    DataLoader, DataLoader, DataLoader, DataLoader]:
    custom_config = load_custom_config()

    print("loading data")
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": num_partitions}, shuffle=shuffle, seed=seed)
    print("loading data completed, start partitioning for", partition_id)
    partition = fds.load_partition(partition_id)
    print("partition loaded")

    print("len partition=", len(partition))

    # First split: 90% train+val, 10% test
    train_val_test_split = partition.train_test_split(test_size=0.1, seed=seed)

    # Second split: Split the 90% into 8/9 train (~80% of total) and 1/9 val (~10% of total)
    train_val_split = train_val_test_split["train"].train_test_split(test_size=1 / 9, seed=seed)


    train_data = train_val_split["train"].with_transform(apply_transforms)
    val_data = train_val_split["test"].with_transform(apply_transforms)
    test_data = train_val_test_split["test"].with_transform(apply_transforms)

    class_indices = defaultdict(list)
    for i, x in enumerate(train_data):
        class_indices[x["label"]].append(i)


    forget_indices = []
    retrain_indices = []

    for cls, indices in class_indices.items():
        random.shuffle(indices)
        forget_count = int(len(indices) * forgetting_config[cls])
        forget_indices.extend(indices[:forget_count])
        retrain_indices.extend(indices[forget_count:])

    print(f"forget={len(forget_indices)}, retrain={len(retrain_indices)}")

    forgetset = Subset(train_data, forget_indices)
    retrainset = Subset(train_data, retrain_indices)

    print(f"len retrain={len(retrainset)}, len forgetset={forgetset}, len val={len(val_data)}, len test={len(test_data)}")

    # Create data loaders
    RETRAIN_BATCH = custom_config["RETRAIN_BATCH"]
    FORGET_BATCH = custom_config["FORGET_BATCH"]
    VAL_BATCH = custom_config["VAL_BATCH"]
    TEST_BATCH = custom_config["TEST_BATCH"]

    if len(retrainset) > 0:
        retrainloader = DataLoader(retrainset, batch_size= RETRAIN_BATCH, shuffle=True)
    else:
        retrainloader = None

    if len(forgetset) > 0:
        forgetloader = DataLoader(forgetset, batch_size=FORGET_BATCH, shuffle=True)
    else:
        forgetloader = None
    valloader = DataLoader(val_data, batch_size=VAL_BATCH, shuffle=True)
    testloader = DataLoader(test_data, batch_size=TEST_BATCH, shuffle=True)

    return retrainloader, forgetloader, valloader, testloader
