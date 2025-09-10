
import torch

from torch.utils.data import DataLoader, ConcatDataset, Dataset

import random
from torch.utils.data import Subset
from torchvision import datasets, transforms

pytorch_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class MembershipDataset(Dataset):
    def __init__(self, dataset, is_member):
        self.dataset = dataset
        self.is_member = is_member

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        return X, y, self.is_member


# Create attack_loader and shadow_loader
def create_attack_and_shadow_loaders(forgetloader, testloader, valloader, batch_size=64):
    """
    Create attack_loader and shadow_loader for Shokri and Yeom attacks using federated dataloaders.

    Args:
        forgetloader (DataLoader): DataLoader for the forget set (members).
        testloader (DataLoader): DataLoader for the test set (non-members).
        valloader (DataLoader): DataLoader for the validation set (used for shadow data).
        batch_size (int): Batch size for the new DataLoaders.

    Returns:
        attack_loader (DataLoader): Combined DataLoader with forget (members) and test (non-members) data.
        shadow_loader (DataLoader): DataLoader for shadow data (used by Shokri attack).
    """
    # Create datasets with membership labels
    forget_dataset_with_membership = MembershipDataset(forgetloader.dataset, is_member=1)
    test_dataset_with_membership = MembershipDataset(testloader.dataset, is_member=0)

    # Combine forget and test datasets
    combined_dataset = ConcatDataset([forget_dataset_with_membership, test_dataset_with_membership])

    # Create attack_loader
    attack_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

    # Create shadow_loader using validation set
    shadow_loader = DataLoader(valloader.dataset, batch_size=batch_size, shuffle=True)

    return attack_loader, shadow_loader



#
def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


import random
from typing import Optional
import torch
from torch.utils.data import Dataset, Subset

# ---------- Helpers

def _infer_num_classes(from_subset: Subset, sample_cap: int = 10_000) -> int:
    """Heuristically infer number of classes from labels in the subset."""
    labels = []
    n = min(len(from_subset), sample_cap)
    for i in range(n):
        _, y = from_subset[i]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        labels.append(int(y))
    return max(labels) + 1

def _different_random_label(y: int, num_classes: int, rng: random.Random) -> int:
    """Return a random label in [0, num_classes) different from y."""
    if num_classes <= 1:
        return y
    r = rng.randrange(num_classes - 1)
    return r if r < y else r + 1

def _add_square_trigger(x, trigger_size: Optional[int] = None, trigger_value: Optional[float] = None):
    """
    Add a solid square at the bottom-right of image x.
    Supports torch tensors [C,H,W] or [H,W], and PIL images.
    Returns a copy (does not modify the input in-place).
    """
    # Torch tensor path
    if isinstance(x, torch.Tensor):
        if x.dim() == 3:
            C, H, W = x.shape
            ts = trigger_size or max(3, min(H, W) // 10)
            v = trigger_value
            # choose a sensible default based on dtype/range
            if v is None:
                v = 255 if x.dtype == torch.uint8 else 1.0
            x2 = x.clone()
            x2[..., H - ts:H, W - ts:W] = v
            return x2
        elif x.dim() == 2:
            H, W = x.shape
            ts = trigger_size or max(3, min(H, W) // 10)
            v = trigger_value
            if v is None:
                v = 255 if x.dtype == torch.uint8 else 1.0
            x2 = x.clone()
            x2[H - ts:H, W - ts:W] = v
            return x2
        # Unknown tensor shape → return as-is
        return x

    # PIL path
    try:
        from PIL import Image, ImageDraw
        if isinstance(x, Image.Image):
            W, H = x.size
            ts = trigger_size or max(3, min(H, W) // 10)
            x2 = x.copy()
            draw = ImageDraw.Draw(x2)
            # default white
            draw.rectangle([W - ts, H - ts, W - 1, H - 1], fill=255)
            return x2
    except Exception:
        pass

    # Fallback: return unchanged if format is unknown
    return x


# ---------- Datasets

from typing import Optional, Mapping, Callable, Union, List, Dict
import torch
from torch.utils.data import Dataset, Subset



from typing import Mapping, Dict
import torch
from torch.utils.data import Dataset, Subset

from typing import Mapping, Dict
import torch
from torch.utils.data import Dataset, Subset

class MapLabelWrapper(Dataset):
    """
    Wrap a Subset and return (x, mapped_label) for each item.
    Only labels present in `mapping` are changed; all others are left unchanged.
    Example mapping: {1: 2, 3: 5}  -> 1→2, 3→5, everything else stays the same.
    """
    def __init__(self, subset: Subset, mapping: Mapping[int, int]):
        assert isinstance(subset, Subset), "Pass a torch.utils.data.Subset"
        self.subset = subset
        # make a plain dict to avoid surprises
        self.mapping: Dict[int, int] = dict(mapping)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        # normalize to int
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        else:
            y = int(y)
        y_mapped = self.mapping.get(y, y)
        return x, y_mapped

class RandomLabelWrapper(Dataset):
    """Wrap a Subset and return (x, randomized_label) for each item."""
    def __init__(self, subset: Subset, num_classes: Optional[int] = None, seed: Optional[int] = None):
        assert isinstance(subset, Subset), "Pass a torch.utils.data.Subset"
        self.subset = subset
        self.rng = random.Random(seed)
        # precompute labels and classes
        labels = []
        for i in range(len(subset)):
            _, y = subset[i]
            if isinstance(y, torch.Tensor):
                y = int(y.item())
            labels.append(int(y))
        self.original_labels = labels
        if num_classes is None:
            num_classes = max(labels) + 1 if labels else _infer_num_classes(subset)
        self.num_classes = num_classes

        # precompute randomized labels (different from original)
        self.random_labels = [
            _different_random_label(y, self.num_classes, self.rng)
            for y in self.original_labels
        ]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        return x, self.random_labels[idx]


class BackdoorWrapper(Dataset):
    """
    Wrap a Subset and return (x_with_trigger, target_label) for each item.
    The trigger is a solid square in the bottom-right corner.
    """
    def __init__(
        self,
        subset: Subset,
        target_label: int = 0,
        trigger_size: Optional[int] = None,
        trigger_value: Optional[float] = None,
    ):
        assert isinstance(subset, Subset), "Pass a torch.utils.data.Subset"
        self.subset = subset
        self.target_label = int(target_label)
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        x_bd = _add_square_trigger(x, self.trigger_size, self.trigger_value)
        return x_bd, self.target_label


# ---------- Public factory functions (drop-in replacements)

def confuse_the_forget_set(forget_set: Subset, confuse_map = None, num_classes: Optional[int] = None, seed: Optional[int] = None) -> Dataset:
    """
    Return a dataset that serves the same samples as `forget_set` but with randomized labels.
    Labels are uniformly sampled from [0, num_classes) and guaranteed to differ from the original.
    """
    #if confuse_map:
    return MapLabelWrapper(forget_set, mapping=confuse_map)
    #return RandomLabelWrapper(forget_set, num_classes=num_classes, seed=seed)


def backdoor_the_forget_set(
    forget_set: Subset,
    target_label: int = 0,
    trigger_size: Optional[int] = None,
    trigger_value: Optional[float] = None,
) -> Dataset:
    """
    Return a dataset that serves the same samples as `forget_set` but with a bottom-right square trigger
    and all labels forced to `target_label`.
    """
    return BackdoorWrapper(
        forget_set,
        target_label=target_label,
        trigger_size=trigger_size,
        trigger_value=trigger_value,
    )
#
# def confuse_the_forget_set(forget_set):
#     """
#     Assign random labels to each sample in the forget set to confuse the model during unlearning.
#
#     Args:
#         forget_set (Subset): PyTorch Subset containing the data to forget
#
#     Returns:
#         Subset: Modified forget set with randomized labels
#     """
#     # Get the number of classes by finding the maximum label value
#     all_labels = [item[1] for item in forget_set]
#     num_classes = max(all_labels) + 1
#
#     # Create a new dataset with random labels
#     random_labels = []
#     for idx in range(len(forget_set)):
#         original_data, original_label = forget_set[idx]
#         # Generate a random label different from the original one
#         new_label = original_label
#         while new_label == original_label:
#             new_label = random.randint(0, num_classes - 1)
#         random_labels.append((original_data, new_label))
#
#     # Create and return a new Subset with random labels
#     random_label_dataset = Subset(forget_set.dataset, forget_set.indices)
#     # Override the __getitem__ method to return random labels
#     original_getitem = random_label_dataset.__getitem__
#
#     def new_getitem(idx):
#         data, _ = original_getitem(idx)
#         return data, random_labels[idx][1]
#
#     random_label_dataset.__getitem__ = new_getitem
#
#     print("label noise added")
#     return random_label_dataset
#
#
#
# def backdoor_the_forget_set(forget_set):
#     """
#     Applies a backdoor attack to the forget set by adding a trigger pattern
#     and setting all labels to zero.
#
#     Args:
#         forget_set (Subset): PyTorch Subset containing the data to backdoor
#
#     Returns:
#         Subset: Modified forget set with backdoor triggers and zero labels
#     """
#
#     # Create and return a new Subset with backdoored data
#     backdoored_dataset = Subset(forget_set.dataset, forget_set.indices)
#
#     # Override the __getitem__ method to return backdoored images and label 0
#     original_getitem = backdoored_dataset.__getitem__
#
#     def new_getitem(idx):
#         data, _ = original_getitem(idx)
#         # Add backdoor trigger pattern
#         if isinstance(data, torch.Tensor):
#             # For image data (assuming CIFAR-10/MNIST format)
#             if data.dim() == 3:  # [channels, height, width]
#                 # Create backdoor trigger (a small square pattern in bottom right corner)
#                 c, h, w = data.shape
#                 trigger_size = max(3, min(h, w) // 10)  # Size proportional to image
#
#                 # Copy the tensor to avoid modifying the original
#                 backdoored_img = data.clone()
#
#                 # Add a white square pattern in bottom right
#                 backdoored_img[:, -trigger_size:, -trigger_size:] = 1.0
#
#                 # Return backdoored image with label 0
#                 print("wow!")
#                 return backdoored_img, 0
#
#         # If not tensor or unknown format, just change label to 0
#         print("wow!")
#         return data, 0
#
#     print("Backdoor tiggers are added.")
#
#     backdoored_dataset.__getitem__ = new_getitem
#
#     return backdoored_dataset
