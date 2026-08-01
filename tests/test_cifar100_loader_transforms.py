import unittest

import torch
from torchvision import transforms

from src.dataloaders.client_dataloader import (
    CIFAR100_NORMALIZED_WHITE,
    _cifar100_eval_transform,
    _cifar100_train_transform,
)
from src.dataloaders.transformers_utils import _add_square_trigger


class Cifar100TransformTests(unittest.TestCase):
    def test_training_and_evaluation_transforms_are_separate(self):
        train_transform = _cifar100_train_transform()
        eval_transform = _cifar100_eval_transform()

        self.assertTrue(any(isinstance(t, transforms.RandomCrop) for t in train_transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_transform.transforms))
        self.assertFalse(any(isinstance(t, transforms.RandomCrop) for t in eval_transform.transforms))
        self.assertFalse(any(isinstance(t, transforms.RandomHorizontalFlip) for t in eval_transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.Normalize) for t in train_transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.Normalize) for t in eval_transform.transforms))

    def test_backdoor_trigger_supports_normalized_channel_values(self):
        image = torch.zeros((3, 8, 8), dtype=torch.float32)
        triggered = _add_square_trigger(
            image,
            trigger_size=2,
            trigger_value=CIFAR100_NORMALIZED_WHITE,
        )

        expected = torch.tensor(CIFAR100_NORMALIZED_WHITE).view(3, 1, 1)
        self.assertTrue(torch.allclose(triggered[:, -2:, -2:], expected.expand(3, 2, 2)))
        self.assertTrue(torch.equal(image, torch.zeros_like(image)))


if __name__ == "__main__":
    unittest.main()
