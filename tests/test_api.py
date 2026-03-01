from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.api import PredictRequest, collect_inputs, create_app  # noqa: E402


class _DummyPredictor:
    backend = "hf"
    reference = "dummy"

    def predict(self, texts, batch_size: int, max_length: int):
        _ = (batch_size, max_length)
        return [
            {
                "text": text,
                "truncated_text": text,
                "entities": {},
                "pred_labels": ["O"] * len(text),
            }
            for text in texts
        ]


class ApiTest(unittest.TestCase):
    def test_collect_inputs(self) -> None:
        request = PredictRequest(
            text="a",
            texts=["b", " c "],
            items=[{"id": "x1", "text": "d"}],
        )
        entries = collect_inputs(request)
        self.assertEqual([entry.text for entry in entries], ["a", "b", "c", "d"])
        self.assertEqual(entries[-1].item_id, "x1")

    def test_collect_inputs_empty(self) -> None:
        with self.assertRaises(ValueError):
            collect_inputs(PredictRequest())

    def test_predict_api_multi_inputs(self) -> None:
        app = create_app(_DummyPredictor(), default_batch_size=2, default_max_length=64)
        client = TestClient(app)
        response = client.post(
            "/predict",
            json={
                "texts": ["hello", "world"],
                "items": [{"id": "i-1", "text": "abc"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 3)
        self.assertEqual(payload["results"][2]["id"], "i-1")
        self.assertIn("x-process-time-ms", response.headers)

    def test_predict_api_bad_request(self) -> None:
        app = create_app(_DummyPredictor())
        client = TestClient(app)
        response = client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
