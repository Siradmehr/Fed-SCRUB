import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.client import FlowerClient
from src.server import weighted_loss_avg_custom


class LogitsModel(nn.Module):
    def forward(self, inputs):
        return inputs


def make_loader(logits, labels):
    dataset = TensorDataset(
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False)


def make_client(unlearning_case, original_loader, transformed_loader):
    client = FlowerClient.__new__(FlowerClient)
    client.partition_id = 2
    client.config_manager = SimpleNamespace(
        config={
            "CLIENT_ID_TO_FORGET": [2],
            "Client_ID_TO_EXIT": [],
            "UNLEARNING_CASE": unlearning_case,
            "MAP_CONFUSE": {0: 1, 1: 0},
            "SEED": 7,
        }
    )
    client.device = torch.device("cpu")
    client.net = LogitsModel()
    client.loss_manager = SimpleNamespace(
        criterion_cls=nn.CrossEntropyLoss(),
        num_classes=2,
    )
    client.val_loader = make_loader([[4, 0], [0, 4]], [0, 1])
    client.original_forget_loader = original_loader
    client.forget_loader = transformed_loader
    client.set_parameters = lambda _: None
    return client


class ForgetMetricTests(unittest.TestCase):
    def evaluate(self, client):
        with (
            patch(
                "src.client.eval_ic_fgt",
                return_value={"IC_ERR_micro": 0.0, "FGT_ERR_micro": 0.0},
            ),
            patch("src.client.compute_mia_score_scrub", return_value=0.5),
        ):
            _, _, metrics = client.evaluate(
                [],
                {"Phase": "LEARN", "TEACHER": ""},
            )
        return metrics

    def test_confuse_accuracy_is_separate_from_true_forget_accuracy(self):
        original = make_loader([[4, 0], [0, 4]], [0, 1])
        confused = make_loader([[4, 0], [0, 4]], [1, 0])
        metrics = self.evaluate(make_client("CONFUSE", original, confused))

        self.assertEqual(metrics["forget_acc"], 1.0)
        self.assertEqual(metrics["forget_size"], 2)
        self.assertEqual(metrics["confuse_acc"], 0.0)
        self.assertEqual(metrics["confuse_size"], 2)
        self.assertEqual(metrics["backdoor_asr"], 0.0)
        self.assertEqual(metrics["backdoor_size"], 0)

    def test_backdoor_asr_is_separate_from_true_forget_accuracy(self):
        original = make_loader([[4, 0], [0, 4]], [0, 1])
        backdoored = make_loader([[4, 0], [4, 0]], [0, 0])
        metrics = self.evaluate(make_client("BACKDOOR", original, backdoored))

        self.assertEqual(metrics["forget_acc"], 1.0)
        self.assertEqual(metrics["forget_size"], 2)
        self.assertEqual(metrics["confuse_acc"], 0.0)
        self.assertEqual(metrics["confuse_size"], 0)
        self.assertEqual(metrics["backdoor_asr"], 1.0)
        self.assertEqual(metrics["backdoor_size"], 2)

    def test_metric_aggregation_is_weighted_by_dataset_size(self):
        result = weighted_loss_avg_custom([(2, 0.5), (6, 0.75)])
        self.assertEqual(result, 0.6875)

    def test_client_uses_stable_partition_id_for_targeting(self):
        loader = make_loader([[4, 0], [0, 4]], [0, 1])
        client = make_client("CONFUSE", loader, loader)
        client.partition_id = 1

        config = client._parse_config({
            "Phase": "LEARN",
            "TEACHER": "",
            "UNLEARN_CON": "TRUE",
        })

        self.assertFalse(config.unlearn_con)


if __name__ == "__main__":
    unittest.main()
