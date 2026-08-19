"""Shared workload metadata used by both the bench runner (to know which unit
it's computing) and the report renderer (to label it correctly).
"""
from __future__ import annotations

WORKLOAD_LABELS = {
    "llm": "LLM Generation",
    "embedding": "Embedding Throughput",
    "image_gen": "Image Generation",
}

# The unit BenchResult.gen_tps is expressed in, per workload. Image generation
# has no token concept, so its throughput is images/min derived from wall-clock
# duration rather than the tps field Lemonade reports for token-based workloads.
WORKLOAD_UNITS = {
    "llm": "tok/s",
    "embedding": "tok/s",
    "image_gen": "img/min",
}
