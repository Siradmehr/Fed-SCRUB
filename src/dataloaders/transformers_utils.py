
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


def confuse_the_forget_set(forget_set):
    """
    Assign random labels to each sample in the forget set to confuse the model during unlearning.

    Args:
        forget_set (Subset): PyTorch Subset containing the data to forget

    Returns:
        Subset: Modified forget set with randomized labels
    """
    # Get the number of classes by finding the maximum label value
    all_labels = [item[1] for item in forget_set]
    num_classes = max(all_labels) + 1

    # Create a new dataset with random labels
    random_labels = []
    for idx in range(len(forget_set)):
        original_data, original_label = forget_set[idx]
        # Generate a random label different from the original one
        new_label = original_label
        while new_label == original_label:
            new_label = random.randint(0, num_classes - 1)
        random_labels.append((original_data, new_label))

    # Create and return a new Subset with random labels
    random_label_dataset = Subset(forget_set.dataset, forget_set.indices)
    # Override the __getitem__ method to return random labels
    original_getitem = random_label_dataset.__getitem__

    def new_getitem(idx):
        data, _ = original_getitem(idx)
        return data, random_labels[idx][1]

    random_label_dataset.__getitem__ = new_getitem

    print("label noise added")
    return random_label_dataset



def backdoor_the_forget_set(forget_set):
    """
    Applies a backdoor attack to the forget set by adding a trigger pattern
    and setting all labels to zero.

    Args:
        forget_set (Subset): PyTorch Subset containing the data to backdoor

    Returns:
        Subset: Modified forget set with backdoor triggers and zero labels
    """

    # Create and return a new Subset with backdoored data
    backdoored_dataset = Subset(forget_set.dataset, forget_set.indices)

    # Override the __getitem__ method to return backdoored images and label 0
    original_getitem = backdoored_dataset.__getitem__

    def new_getitem(idx):
        data, _ = original_getitem(idx)
        # Add backdoor trigger pattern
        if isinstance(data, torch.Tensor):
            # For image data (assuming CIFAR-10/MNIST format)
            if data.dim() == 3:  # [channels, height, width]
                # Create backdoor trigger (a small square pattern in bottom right corner)
                c, h, w = data.shape
                trigger_size = max(3, min(h, w) // 10)  # Size proportional to image

                # Copy the tensor to avoid modifying the original
                backdoored_img = data.clone()

                # Add a white square pattern in bottom right
                backdoored_img[:, -trigger_size:, -trigger_size:] = 1.0

                # Return backdoored image with label 0
                return backdoored_img, 0

        # If not tensor or unknown format, just change label to 0
        return data, 0

    print("Backdoor tiggers are added.")

    backdoored_dataset.__getitem__ = new_getitem

    return backdoored_dataset
