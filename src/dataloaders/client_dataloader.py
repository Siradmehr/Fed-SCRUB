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

from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Caltech101Wrapper(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, seed=42):
        # Caltech101 expects root to be the parent of 'caltech101' folder
        self.dataset = datasets.Caltech101(
            root=root,
            target_type='category',
            download=download,
            transform=transform
        )

        self.targets = [self.dataset.y[i] for i in range(len(self.dataset))]

        import random
        all_indices = list(range(len(self.dataset)))
        random.seed(seed)
        random.shuffle(all_indices)

        split_idx = int(0.8 * len(all_indices))

        self.indices = all_indices[:split_idx] if train else all_indices[split_idx:]
        self.targets = [self.targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.dataset[actual_idx]
        return img, label

#
# class Caltech101Wrapper(Dataset):
#     def __init__(self, root, train=True, download=True, transform=None, seed=42):
#         self.dataset = datasets.Caltech101(root=root, download=download, transform=transform)
#
#         # Create targets list
#         self.targets = [self.dataset.y[i] for i in range(len(self.dataset))]
#
#         # Shuffle ALL indices with seed before splitting
#         import random
#         all_indices = list(range(len(self.dataset)))
#         random.seed(seed)
#         random.shuffle(all_indices)
#
#         # Now split
#         split_idx = int(0.8 * len(all_indices))
#
#         if train:
#             self.indices = all_indices[:split_idx]
#         else:
#             self.indices = all_indices[split_idx:]
#
#         # Update targets to match split
#         self.targets = [self.targets[i] for i in self.indices]
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, idx):
#         actual_idx = self.indices[idx]
#         img, label = self.dataset[actual_idx]
#         return img, label
#

import random
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTStyleDataset(Dataset):
    """
    Wraps any base dataset + index list so it behaves like MNIST:
    - has .targets
    - __getitem__ returns (x, y)
    """
    def __init__(self, base_dataset, indices, targets):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        # store targets for the *subset* in MNIST/CIFAR style
        self.targets = [int(targets[i]) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_i = self.indices[i]
        x, y = self.base_dataset[base_i]
        return x, int(y)


def _get_targets_generic(dataset):
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t.tolist() if hasattr(t, "tolist") else list(t)
    if hasattr(dataset, "labels"):
        t = dataset.labels
        return t.tolist() if hasattr(t, "tolist") else list(t)
    return [dataset[i][1] for i in range(len(dataset))]


def _stratified_split_indices(targets, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    cls_to_idx = defaultdict(list)
    for i, y in enumerate(targets):
        cls_to_idx[int(y)].append(i)

    train_idx, test_idx = [], []
    for c, idxs in cls_to_idx.items():
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_ratio))  # prevent empty class in test
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def load_caltech101_mnist_style(
    root="./data",
    seed=42,
    test_ratio=0.2,
    image_size=224,
    vit_normalize=True,
    download=True,
):
    """
    Returns:
      train_ds, test_ds, num_classes
    where train_ds/test_ds behave like MNIST/CIFAR datasets:
      - have .targets
      - __getitem__ -> (tensor_image, int_label)
    """
    if vit_normalize:
        tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    base = datasets.Caltech101(
        root=root,
        download=download,
        transform=tfm,
        target_type="category",   # gives integer class indices
    )

    targets = _get_targets_generic(base)
    train_idx, test_idx = _stratified_split_indices(targets, test_ratio=test_ratio, seed=seed)

    train_ds = MNISTStyleDataset(base, train_idx, targets)
    test_ds  = MNISTStyleDataset(base, test_idx, targets)

    # robust class count (don’t assume 101)
    num_classes = len(getattr(base, "categories", set(targets)))

    return train_ds, test_ds, num_classes



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
    # elif dataset_name.lower() in ["caltech101", "caltech-101", "caltech_101"]:
    #     dataset, test_dataset, _ = load_caltech101_mnist_style(root=root, seed=seed)
    elif dataset_name.lower() == "caltech-101":
        vit_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),  # converts grayscale -> 3ch, leaves RGB as 3ch too
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        dataset = Caltech101Wrapper(root=root, train=True, download=True, transform=vit_transform, seed=seed)
        test_dataset = Caltech101Wrapper(root=root, train=False, download=True, transform=vit_transform, seed=seed)
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

