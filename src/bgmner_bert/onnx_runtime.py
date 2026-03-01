from __future__ import annotations

import platform
from pathlib import Path
from typing import Sequence

import onnxruntime as ort

AUTO_PROVIDER = "auto"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
COREML_EXECUTION_PROVIDER = "CoreMLExecutionProvider"
DML_EXECUTION_PROVIDER = "DmlExecutionProvider"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
ROCM_EXECUTION_PROVIDER = "ROCMExecutionProvider"

PROVIDER_ALIASES = {
    "cpu": CPU_EXECUTION_PROVIDER,
    "coreml": COREML_EXECUTION_PROVIDER,
    "dml": DML_EXECUTION_PROVIDER,
    "cuda": CUDA_EXECUTION_PROVIDER,
    "rocm": ROCM_EXECUTION_PROVIDER,
}


def default_provider_argument(*, system: str | None = None) -> str:
    system_name = (system or platform.system()).lower()
    if system_name == "darwin":
        return "cpu"
    return AUTO_PROVIDER


def default_provider_priority(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> list[str]:
    system_name = (system or platform.system()).lower()
    machine_name = (machine or platform.machine()).lower()

    if system_name == "darwin" and machine_name in {"arm64", "aarch64"}:
        return [COREML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER]
    return [CPU_EXECUTION_PROVIDER]


def _dedupe_ordered(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def parse_provider_argument(provider: str) -> list[str]:
    value = provider.strip()
    if not value:
        return [default_provider_argument()]

    result: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        lower_item = item.lower()
        if lower_item == AUTO_PROVIDER:
            result.append(AUTO_PROVIDER)
        else:
            result.append(PROVIDER_ALIASES.get(lower_item, item))
    return result


def resolve_provider_chain(
    provider: str,
    *,
    available_providers: Sequence[str] | None = None,
) -> list[str]:
    available = list(available_providers) if available_providers is not None else list(
        ort.get_available_providers()
    )
    if not available:
        raise RuntimeError("onnxruntime returned no available execution providers.")

    requested = parse_provider_argument(provider)
    is_auto = len(requested) == 1 and requested[0].lower() == AUTO_PROVIDER
    if is_auto:
        preferred = default_provider_priority()
        resolved = [item for item in preferred if item in available]
        if resolved:
            return resolved
        return [available[0]]

    requested = _dedupe_ordered(requested)
    missing = [item for item in requested if item not in available]
    if missing:
        raise RuntimeError(
            f"Provider(s) {missing} not available. Requested: {requested}. Available: {available}."
        )
    return requested


def build_onnx_session(
    onnx_path: Path,
    provider: str,
) -> tuple[ort.InferenceSession, list[str], list[str]]:
    available = list(ort.get_available_providers())
    provider_chain = resolve_provider_chain(provider, available_providers=available)
    session = ort.InferenceSession(str(onnx_path), providers=provider_chain)
    return session, provider_chain, available
