import os
import unittest
from unittest.mock import patch

from src.utils.utils import _parse_bool, load_model, resolve_initial_checkpoint


class ResumeTrainingTests(unittest.TestCase):
    def base_config(self):
        return {
            "RESUME": "",
            "Resume_Training": False,
            "STARTING_PHASE": "PRETRAIN",
            "SAVING_DIR": os.path.join("checkpoints", "current_run"),
        }

    def test_explicit_resume_overrides_resume_training(self):
        config = self.base_config()
        config["RESUME"] = " checkpoints/explicit/model_best.pth "
        config["Resume_Training"] = True

        checkpoint, source = resolve_initial_checkpoint(config)

        self.assertEqual(checkpoint, "checkpoints/explicit/model_best.pth")
        self.assertEqual(source, "explicit_resume")

    def test_pretrain_resume_uses_derived_latest_checkpoint(self):
        config = self.base_config()
        config["Resume_Training"] = True

        checkpoint, source = resolve_initial_checkpoint(config)

        self.assertEqual(
            checkpoint,
            os.path.join(
                config["SAVING_DIR"], "models_chkpts", "model_latest.pth"
            ),
        )
        self.assertEqual(source, "derived_latest")

    def test_pretrain_without_resume_starts_randomly(self):
        config = self.base_config()

        checkpoint, source = resolve_initial_checkpoint(config)

        self.assertIsNone(checkpoint)
        self.assertEqual(source, "random")

    def test_resume_training_only_applies_to_pretrain(self):
        config = self.base_config()
        config["Resume_Training"] = True
        config["STARTING_PHASE"] = "MAX"

        checkpoint, source = resolve_initial_checkpoint(config)

        self.assertIsNone(checkpoint)
        self.assertEqual(source, "random")

    def test_false_string_is_parsed_as_false(self):
        self.assertFalse(_parse_bool("false", "Resume_Training"))
        self.assertTrue(_parse_bool("true", "Resume_Training"))

    @patch("src.utils.utils.get_model", return_value=object())
    def test_required_missing_checkpoint_raises(self, _):
        with self.assertRaises(FileNotFoundError):
            load_model(
                "test-model",
                "checkpoints/missing/model_latest.pth",
                strict_checkpoint=True,
            )


if __name__ == "__main__":
    unittest.main()
