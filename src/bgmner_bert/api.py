from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from .onnx_predict import iter_batches as onnx_iter_batches
from .onnx_predict import predict_batch as onnx_predict_batch
from .onnx_runtime import build_onnx_session, default_provider_argument
from .predict import iter_batches as hf_iter_batches
from .predict import predict_batch as hf_predict_batch
from .predict import resolve_device

REQUEST_LOGGER = logging.getLogger("bgmner.api")


class PredictItem(BaseModel):
    id: str | int | None = None
    text: str


class PredictRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    items: list[PredictItem] | None = None
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    max_length: int | None = Field(default=None, ge=8, le=4096)


@dataclass
class InputEntry:
    text: str
    item_id: str | int | None = None


class Predictor(Protocol):
    backend: str
    reference: str

    def predict(self, texts: list[str], batch_size: int, max_length: int) -> list[dict[str, Any]]:
        ...


def _normalize_text(text: str) -> str:
    value = text.strip()
    if not value:
        raise ValueError("Input text cannot be empty.")
    return value


def collect_inputs(request: PredictRequest) -> list[InputEntry]:
    entries: list[InputEntry] = []

    if request.text is not None:
        entries.append(InputEntry(text=_normalize_text(request.text)))

    if request.texts:
        for text in request.texts:
            entries.append(InputEntry(text=_normalize_text(text)))

    if request.items:
        for item in request.items:
            entries.append(InputEntry(text=_normalize_text(item.text), item_id=item.id))

    if not entries:
        raise ValueError("Provide at least one of: text, texts, items.")
    return entries


class HfPredictor:
    backend = "hf"

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.reference = str(model_dir)

    def predict(self, texts: list[str], batch_size: int, max_length: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for batch in hf_iter_batches(texts, batch_size):
            rows = hf_predict_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                texts=batch,
                max_length=max_length,
                device=self.device,
                id2label=self.id2label,
            )
            results.extend(rows)
        return results


class OnnxPredictor:
    backend = "onnx"

    def __init__(self, onnx_path: Path, model_dir: Path, provider: str) -> None:
        self.session, provider_chain, _ = build_onnx_session(
            onnx_path=onnx_path,
            provider=provider,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        config = AutoConfig.from_pretrained(model_dir)
        self.id2label = {int(key): value for key, value in config.id2label.items()}
        session_providers = list(self.session.get_providers())
        active_provider = session_providers[0] if session_providers else provider_chain[0]
        self.reference = f"{onnx_path} @ {active_provider}"

    def predict(self, texts: list[str], batch_size: int, max_length: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for batch in onnx_iter_batches(texts, batch_size):
            rows = onnx_predict_batch(
                session=self.session,
                tokenizer=self.tokenizer,
                texts=batch,
                max_length=max_length,
                id2label=self.id2label,
            )
            results.extend(rows)
        return results


def create_app(
    predictor: Predictor,
    default_batch_size: int = 32,
    default_max_length: int = 256,
) -> FastAPI:
    app = FastAPI(title="bgmner API", version="0.1.0")

    @app.middleware("http")
    async def process_time_middleware(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if response is not None:
                response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
            REQUEST_LOGGER.info(
                "%s %s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "backend": predictor.backend,
            "reference": predictor.reference,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            entries = collect_inputs(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        batch_size = request.batch_size or default_batch_size
        max_length = request.max_length or default_max_length
        raw_results = predictor.predict(
            texts=[entry.text for entry in entries],
            batch_size=batch_size,
            max_length=max_length,
        )

        results: list[dict[str, Any]] = []
        for entry, row in zip(entries, raw_results):
            item = dict(row)
            if entry.item_id is not None:
                item["id"] = entry.item_id
            results.append(item)

        return {
            "count": len(results),
            "backend": predictor.backend,
            "results": results,
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve NER Web API.")
    parser.add_argument("--backend", choices=["hf", "onnx"], default="hf")
    parser.add_argument("--model-dir", required=True, help="Path to HF best_model directory.")
    parser.add_argument("--onnx-path", default="", help="Required when backend=onnx.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--provider",
        default=default_provider_argument(),
        help=(
            "onnxruntime execution provider. Supports aliases: "
            "cpu/coreml/cuda/rocm/dml. "
            "You can pass a chain like 'coreml,cpu'; "
            "use 'auto' for platform defaults."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    if args.backend == "onnx":
        if not args.onnx_path:
            raise ValueError("--onnx-path is required when --backend onnx.")
        onnx_path = Path(args.onnx_path).resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        predictor: Predictor = OnnxPredictor(
            onnx_path=onnx_path,
            model_dir=model_dir,
            provider=args.provider,
        )
    else:
        predictor = HfPredictor(model_dir=model_dir, device=args.device)

    app = create_app(
        predictor=predictor,
        default_batch_size=args.batch_size,
        default_max_length=args.max_length,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
