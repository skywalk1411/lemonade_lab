"""Model registry: each ModelSpec is one row in the report — a display name plus
one or more BackendSources, each pointing at whichever Lemonade Server catalog
model actually backs that backend (GGUF quant for cpu/vulkan/rocm, an NPU-native
build for npu/hybrid). `lemonade pull <name>` / `--auto-pull` handles the download.

Extend `default_registry` with more ModelSpecs to broaden what gets benchmarked;
run `lemonade list` to see the full catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendSource:
    backend: str                       # report-facing label: cpu/vulkan/rocm/npu/hybrid
    lemonade_model: str                  # catalog model name for this backend
    lemonade_backend: Optional[str] = None  # --backend value to pass to `lemonade bench`
                                             # (defaults to `backend`; Hybrid models still
                                             # use --backend npu, so override it there)

    @property
    def bench_backend(self) -> str:
        return self.lemonade_backend or self.backend


@dataclass
class ModelSpec:
    name: str            # display name — groups all its BackendSources into one report row
    workload: str          # "llm" | "embedding" | "image_gen"
    sources: list[BackendSource]


def default_registry() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="Llama-3.2-1B-Instruct",
            workload="llm",
            sources=[
                BackendSource("cpu", "Llama-3.2-1B-Instruct-GGUF"),
                BackendSource("vulkan", "Llama-3.2-1B-Instruct-GGUF"),
                BackendSource("npu", "Llama-3.2-1B-Instruct-NPU", lemonade_backend="npu"),
                BackendSource("hybrid", "Llama-3.2-1B-Instruct-Hybrid", lemonade_backend="npu"),
            ],
        ),
        ModelSpec(
            name="Qwen3.6-35B-A3B",
            workload="llm",
            sources=[
                BackendSource("npu", "qwen3.6-moe-35b-a3b-FLM", lemonade_backend="npu"),
            ],
        ),
        ModelSpec(
            name="nomic-embed-text-v1",
            workload="embedding",
            sources=[
                BackendSource("cpu", "nomic-embed-text-v1-GGUF"),
                BackendSource("vulkan", "nomic-embed-text-v1-GGUF"),
            ],
        ),
        ModelSpec(
            name="embed-gemma-300m",
            workload="embedding",
            sources=[
                BackendSource("npu", "embed-gemma-300m-FLM", lemonade_backend="npu"),
            ],
        ),
        ModelSpec(
            name="SD-Turbo",
            workload="image_gen",
            sources=[
                BackendSource("vulkan", "SD-Turbo-GGUF"),
            ],
        ),
    ]
