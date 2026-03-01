from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.config import parse_train_args  # noqa: E402


class TrainConfigJsonTest(unittest.TestCase):
    def test_json_config_loading(self) -> None:
        payload = {
            "dataset_dir": "data/ner_data",
            "model_name": "bert-base-chinese",
            "num_train_epochs": 3.0,
            "learning_rate": 3e-5,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = parse_train_args(["--config-file", str(path)])

        self.assertTrue(cfg.config_file.endswith("train_config.json"))
        self.assertEqual(cfg.model_name, "bert-base-chinese")
        self.assertEqual(cfg.num_train_epochs, 3.0)
        self.assertAlmostEqual(cfg.learning_rate, 3e-5)

    def test_cli_overrides_json(self) -> None:
        payload = {"learning_rate": 3e-5, "per_device_train_batch_size": 8}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = parse_train_args(
                [
                    "--config-file",
                    str(path),
                    "--learning-rate",
                    "1e-5",
                    "--per-device-train-batch-size",
                    "16",
                ]
            )
        self.assertAlmostEqual(cfg.learning_rate, 1e-5)
        self.assertEqual(cfg.per_device_train_batch_size, 16)

    def test_unknown_json_key_raises(self) -> None:
        payload = {"unknown_field": 1}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_train_args(["--config-file", str(path)])


if __name__ == "__main__":
    unittest.main()
