import unittest

import torch
from torchvision import transforms
from torch.utils.data import Subset, TensorDataset

from src.dataloaders.client_dataloader import (
    CIFAR100_NORMALIZED_WHITE,
    _cifar100_eval_transform,
    _cifar100_train_transform,
)
from src.dataloaders.transformers_utils import (
    _add_square_trigger,
    backdoor_the_forget_set,
)


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

    def test_backdoor_asr_dataset_excludes_original_target_class(self):
        images = torch.stack([
            torch.zeros((3, 4, 4)),
            torch.ones((3, 4, 4)),
            torch.full((3, 4, 4), 2.0),
        ])
        labels = torch.tensor([0, 1, 2])
        subset = Subset(TensorDataset(images, labels), [0, 1, 2])

        asr_dataset = backdoor_the_forget_set(
            subset,
            target_label=0,
            trigger_size=1,
            exclude_original_target=True,
        )

        self.assertEqual(len(asr_dataset), 2)
        first_image, first_target = asr_dataset[0]
        second_image, second_target = asr_dataset[1]
        self.assertEqual(first_target, 0)
        self.assertEqual(second_target, 0)
        self.assertEqual(first_image[0, 0, 0].item(), 1.0)
        self.assertEqual(second_image[0, 0, 0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
