from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.inference_utils import predict_entities_from_token_ids


class OnnxParityTest(unittest.TestCase):
    def test_decode_parity_for_same_predictions(self) -> None:
        words = list("番剧标题")
        word_ids = [-1, 0, 1, 2, 3, -1]
        token_pred_ids = [0, 1, 2, 0, 0, 0]
        id2label = {0: "O", 1: "B-CT", 2: "I-CT"}
        entities_a, _, _ = predict_entities_from_token_ids(
            words=words,
            token_label_ids=token_pred_ids,
            word_ids=word_ids,
            id2label=id2label,
        )
        entities_b, _, _ = predict_entities_from_token_ids(
            words=words,
            token_label_ids=token_pred_ids,
            word_ids=word_ids,
            id2label=id2label,
        )
        self.assertEqual(entities_a, entities_b)

    @unittest.skipUnless(
        os.getenv("RUN_ONNX_PARITY_TEST") == "1",
        "Set RUN_ONNX_PARITY_TEST=1 with model/onnx artifacts to run integration parity.",
    )
    def test_external_model_parity(self) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except Exception as exc:
            self.skipTest(f"Missing dependency for integration parity: {exc}")

        model_dir = os.getenv("ONNX_PARITY_MODEL_DIR", "").strip()
        onnx_path = os.getenv("ONNX_PARITY_ONNX_PATH", "").strip()
        if not model_dir or not onnx_path:
            self.skipTest("ONNX_PARITY_MODEL_DIR / ONNX_PARITY_ONNX_PATH are required.")
        if not Path(model_dir).exists() or not Path(onnx_path).exists():
            self.skipTest("Provided parity artifacts do not exist.")

        text = "[桜都字幕组] 迷宫饭 [15][1080p][繁体内嵌]"
        words = list(text)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        id2label = {int(k): v for k, v in model.config.id2label.items()}

        encoded_torch = tokenizer(
            [words],
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        word_ids = [
            -1 if wid is None else int(wid) for wid in encoded_torch.word_ids(batch_index=0)
        ]

        with torch.no_grad():
            hf_logits = model(
                input_ids=encoded_torch["input_ids"],
                attention_mask=encoded_torch["attention_mask"],
            ).logits
        hf_pred_ids = hf_logits.argmax(dim=-1).cpu().numpy()[0].tolist()
        hf_entities, _, _ = predict_entities_from_token_ids(
            words=words,
            token_label_ids=hf_pred_ids,
            word_ids=word_ids,
            id2label=id2label,
        )

        encoded_np = tokenizer(
            [words],
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="np",
        )
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_logits = session.run(
            ["logits"],
            {
                "input_ids": encoded_np["input_ids"].astype(np.int64),
                "attention_mask": encoded_np["attention_mask"].astype(np.int64),
            },
        )[0]
        onnx_pred_ids = onnx_logits.argmax(axis=-1)[0].tolist()
        onnx_entities, _, _ = predict_entities_from_token_ids(
            words=words,
            token_label_ids=onnx_pred_ids,
            word_ids=word_ids,
            id2label=id2label,
        )
        self.assertEqual(hf_entities, onnx_entities)


if __name__ == "__main__":
    unittest.main()
