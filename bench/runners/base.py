"""Shared result type produced by benchmark runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchResult:
    backend: str            # "cpu" | "vulkan" | "rocm" | "npu" | "hybrid"
    model: str               # friendly model name
    workload: str = "llm"    # "llm" | "embedding" | "image_gen"
    gen_tps: Optional[float] = None      # mean tokens/sec (the workload's primary throughput metric)
    ttft_ms: Optional[float] = None      # mean time-to-first-token, milliseconds
    memory_gb: Optional[float] = None    # peak memory observed during the run
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.gen_tps is not None
