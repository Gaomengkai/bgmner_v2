from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.onnx_runtime import (  # noqa: E402
    AUTO_PROVIDER,
    COREML_EXECUTION_PROVIDER,
    CPU_EXECUTION_PROVIDER,
    CUDA_EXECUTION_PROVIDER,
    DML_EXECUTION_PROVIDER,
    ROCM_EXECUTION_PROVIDER,
    default_provider_argument,
    default_provider_priority,
    parse_provider_argument,
    resolve_provider_chain,
)


class OnnxRuntimeProviderResolutionTest(unittest.TestCase):
    def test_default_provider_argument_for_macos_is_cpu_alias(self) -> None:
        self.assertEqual(default_provider_argument(system="Darwin"), "cpu")

    def test_default_priority_for_apple_silicon(self) -> None:
        self.assertEqual(
            default_provider_priority(system="Darwin", machine="arm64"),
            [COREML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER],
        )

    def test_auto_prefers_coreml_on_apple_silicon(self) -> None:
        resolved = resolve_provider_chain(
            AUTO_PROVIDER,
            available_providers=[
                COREML_EXECUTION_PROVIDER,
                CPU_EXECUTION_PROVIDER,
            ],
        )
        self.assertEqual(resolved, [COREML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER])

    def test_auto_falls_back_to_cpu(self) -> None:
        resolved = resolve_provider_chain(
            AUTO_PROVIDER,
            available_providers=[CPU_EXECUTION_PROVIDER],
        )
        self.assertEqual(resolved, [CPU_EXECUTION_PROVIDER])

    def test_explicit_provider_chain_is_supported(self) -> None:
        resolved = resolve_provider_chain(
            "CoreMLExecutionProvider,CPUExecutionProvider",
            available_providers=[
                COREML_EXECUTION_PROVIDER,
                CPU_EXECUTION_PROVIDER,
            ],
        )
        self.assertEqual(resolved, [COREML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER])

    def test_missing_provider_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_provider_chain(
                "CoreMLExecutionProvider",
                available_providers=[CPU_EXECUTION_PROVIDER],
            )

    def test_provider_aliases_are_mapped(self) -> None:
        parsed = parse_provider_argument("dml,cuda,rocm,coreml,cpu")
        self.assertEqual(
            parsed,
            [
                DML_EXECUTION_PROVIDER,
                CUDA_EXECUTION_PROVIDER,
                ROCM_EXECUTION_PROVIDER,
                COREML_EXECUTION_PROVIDER,
                CPU_EXECUTION_PROVIDER,
            ],
        )


if __name__ == "__main__":
    unittest.main()
