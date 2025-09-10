import copy

from flwr_datasets import FederatedDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Optional
from collections import defaultdict
import random
import os
from ..utils.utils import load_config as load_custom_config, setup_experiment, set_seed
from ..utils.utils import set_seed
from torch.utils.data import Subset
from collections import defaultdict
import random
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional

from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
# Custom dataset to add is_member labels


import random
from torch.utils.data import Subset
from torchvision import datasets, transforms
from ..utils.utils import np_index_save

from .transformers_utils import *

def _partition_dataset(dataset, num_partitions, partition_id, shuffle):
    label_to_indices = defaultdict(lambda : [])
    target_list = dataset.targets
    if type(target_list) != type([0,1]):
        target_list = target_list.tolist()
    for idx, label in enumerate(target_list):
        label_to_indices[label].append(idx)

    #print(f"number of partition is {num_partitions} id = {partition_id} label_to_indices= {label_to_indices}")
    partition_indices = []

    # For each label, shuffle and evenly split indices
    for label, indices in label_to_indices.items():
        # Set the seed for reproducibility, then shuffle the indices for this label.
        # Determine the size of each partition for this label.
        base_size = len(indices) // num_partitions
        extra = len(indices) % num_partitions  # Some partitions may get one extra sample

        # Compute starting index for the current partition.
        # Partitions with an index less than `extra` receive one extra sample.
        start_idx = sum(base_size + 1 if i < extra else base_size for i in range(partition_id))
        part_size = base_size + 1 if partition_id < extra else base_size
        end_idx = start_idx + part_size

        label_partition = indices[start_idx:end_idx]

        # Optionally shuffle the slice from this label if desired.
        if shuffle:
            random.shuffle(label_partition)

        partition_indices.extend(label_partition)

    print(
        f"Balanced partition {partition_id} loaded with {len(partition_indices)} samples (each label equally represented)")
    return Subset(dataset, partition_indices), partition_indices


def configure_balanced_partition(root: str, dataset_name: str, partition_id: int, num_partitions: int, seed: int,
                                 shuffle: bool) -> Subset:
    """
    Load a dataset and partition it so that each partition gets the same number of samples per label.

    Args:
        root (str): Path to dataset storage.
        dataset_name (str): Name of the dataset (currently only "CIFAR10" is supported).
        partition_id (int): The partition index (0 to num_partitions - 1).
        num_partitions (int): Total number of partitions.
        seed (int): Random seed for reproducibility.
        shuffle (bool): Whether to shuffle indices within each label partition.

    Returns:
        Subset: A PyTorch Subset containing the balanced partition of data.
    """
    # Load dataset
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
        #dataset.targets = dataset.targets.tolist()
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
        #test_dataset = test_dataset.targets.tolist()
    elif dataset_name.lower() == "fashionmnist":
        dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError("Unsupported dataset")

    # Validate partition_id
    if not (0 <= partition_id < num_partitions):
        raise ValueError(f"partition_id must be between 0 and {num_partitions - 1}")

    trainin_set, full_training_index = _partition_dataset(dataset, num_partitions, partition_id, shuffle)
    test_set, test_index = _partition_dataset(test_dataset, num_partitions, partition_id, shuffle)
    return trainin_set, full_training_index, test_set, test_index



def load_datasets_with_forgetting(
        partition_id: int,
        num_partitions: int,
        seed: int = 42,
        shuffle: bool = True,
        forgetting_config: Dict = {},
        dataset_name: str = "cifar10"
) -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader, DataLoader, DataLoader]:
    """
    Load and partition datasets with forgetting functionality and print class distributions.

    """

    custom_config = setup_experiment(path=os.environ["EXP_ENV_DIR"], load_model_flag=False)
    set_seed(int(custom_config["SEED"]))

    partition, full_training_index, test_set, test_index = configure_balanced_partition(root="./data",
                                             dataset_name=dataset_name,
                                             partition_id=partition_id,
                                             num_partitions=num_partitions,
                                             seed=seed,
                                             shuffle=shuffle)

    # Group data by class labels
    label_to_indices = defaultdict(list)
    for idx, item in enumerate(partition):
        label_to_indices[item[1]].append(idx)

    # Split the indices for each class
    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        # Shuffle indices for each class
        random.shuffle(indices)

        # Split the indices: 80% train, 10% val, 10% test
        total_size = len(indices)
        train_size = int(0.9 * total_size)
        val_size = int(0.1 * total_size)

        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:train_size + val_size])

    # Create Subsets for train, val, and test
    train_data = Subset(partition, train_indices)
    val_data = Subset(partition, val_indices)
    test_data = test_set

    # Compute class distributions
    def compute_class_distribution(dataset):
        class_counts = defaultdict(int)
        for item in dataset:
            class_counts[item[1]] += 1
        return class_counts

    # train_distribution = compute_class_distribution(train_data)
    # val_distribution = compute_class_distribution(val_data)
    # test_distribution = compute_class_distribution(test_data)
    #
    # print("Class distributions:")
    # print(f"Train: {train_distribution}")
    # print(f"Val: {val_distribution}")
    # print(f"Test: {test_distribution}")

    # Now split the train set into retrain and forget sets based on forgetting_config
    class_indices = defaultdict(list)
    for i, x in enumerate(train_data):
        class_indices[x[1]].append(i)

    forget_indices = []
    retrain_indices = []

    for cls, indices in class_indices.items():
        if cls in forgetting_config:
            random.shuffle(indices)
            forget_count = int(len(indices) * forgetting_config[cls])
            forget_indices.extend(indices[:forget_count])
            retrain_indices.extend(indices[forget_count:])
        else:
            retrain_indices.extend(indices)

    forgetset = Subset(train_data, forget_indices)
    retrainset = Subset(train_data, retrain_indices)

    # # Compute forget and retrain distributions
    # forget_distribution = compute_class_distribution(forgetset)
    # retrain_distribution = compute_class_distribution(retrainset)
    #
    # print(f"Forget set: {forget_distribution}")
    # print(f"Retrain set: {retrain_distribution}")

    import copy
    forget_clients = custom_config["CLIENT_ID_TO_FORGET"]
    print(forget_clients)
    new_forget_dataset = copy.deepcopy(forgetset)
    if partition_id in forget_clients:
        if custom_config["UNLEARNING_CASE"] == "CONFUSE":
            forgetset = confuse_the_forget_set(forgetset, custom_config["MAP_CONFUSE"])
        elif custom_config["UNLEARNING_CASE"] == "BACKDOOR":
            forgetset = backdoor_the_forget_set(forgetset)



    # Create DataLoaders
    retrain_batch = custom_config["RETRAIN_BATCH"]
    forget_batch = custom_config["FORGET_BATCH"]
    val_batch = custom_config["VAL_BATCH"]
    test_batch = custom_config["TEST_BATCH"]


    retrainloader = DataLoader(retrainset, batch_size=retrain_batch, shuffle=True) if len(retrainset) > 0 else None
    forgetloader = DataLoader(forgetset, batch_size=forget_batch, shuffle=True) if len(forgetset) > 0 else None
    original_forget_loader = DataLoader(new_forget_dataset, batch_size=forget_batch, shuffle=True) if len(new_forget_dataset) > 0 else None
    valloader = DataLoader(val_data, batch_size=val_batch, shuffle=True)
    testloader = DataLoader(test_data, batch_size=test_batch, shuffle=True)

    np_index_save(full_training_index=full_training_index,
                  training_set=train_indices,
                  retrain_index=retrain_indices,
                  forget_index=forget_indices,
                  val_index=val_indices,
                  test_index=test_index,
                  config=custom_config,
                  partition_id=partition_id)


    return retrainloader, forgetloader, valloader, testloader, original_forget_loader

