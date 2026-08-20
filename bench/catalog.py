"""Browses what Lemonade Server can actually run: the full model catalog
(`lemonade list`), which of those models are already downloaded (the
`/api/v1/models` REST endpoint), and which backends are installed for each
recipe (`/api/v1/system-info`).

This lets bench/cli.py offer a `--list-models` view and an `--interactive`
picker without anyone having to hand-edit bench/models.py to try a model.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field

from bench.models import BackendSource, ModelSpec
from bench.runners.lemonade import WORKLOAD_SCENARIOS

# recipe -> the report-facing backends it can offer, if installed
RECIPE_BACKENDS = {
    "llamacpp": ["cpu", "vulkan", "rocm"],
    "sd-cpp": ["cpu", "vulkan", "rocm"],
    "flm": ["npu"],
    "ryzenai-llm": ["npu"],  # report label becomes "hybrid" for -Hybrid models, see _report_backend
}

# recipes lemonade_lab can benchmark today (matches runners/lemonade.py's
# WORKLOAD_SCENARIOS keys). Reranker/TTS/Whisper models are downloadable and
# servable by Lemonade Server, but `lemonade bench` has no scenario category
# for them yet.
BENCHABLE_RECIPES = {"llamacpp", "sd-cpp", "flm", "ryzenai-llm"}


@dataclass
class CatalogModel:
    id: str
    recipe: str
    downloaded: bool
    size_gb: float | None
    labels: list[str] = field(default_factory=list)

    @property
    def workload(self) -> str | None:
        if self.recipe == "sd-cpp":
            return "image_gen"
        if self.recipe not in BENCHABLE_RECIPES:
            return None
        name = self.id.lower()
        if "embed" in name or "embeddings" in self.labels:
            return "embedding"
        if "rerank" in name or "reranking" in self.labels:
            return None  # no bench scenario category for this yet
        return "llm"

    @property
    def bench_supported(self) -> bool:
        return self.recipe in BENCHABLE_RECIPES and self.workload is not None


def _fetch_downloaded_labels(host: str, port: int) -> dict[str, list[str]]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/v1/models", timeout=10) as resp:
            data = json.load(resp)
        return {m["id"]: m.get("labels", []) for m in data.get("data", [])}
    except Exception:
        return {}


def _fetch_installed_backends(host: str, port: int) -> dict[str, set[str]]:
    """recipe -> set of report-facing backends whose install `state` is 'installed'."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/v1/system-info", timeout=10) as resp:
            data = json.load(resp)
    except Exception:
        return {}

    installed: dict[str, set[str]] = {}
    for recipe, info in data.get("recipes", {}).items():
        backends = info.get("backends", {})
        installed_here = {b for b, meta in backends.items() if meta.get("state") == "installed"}
        if installed_here:
            installed[recipe] = installed_here
    return installed


def _parse_list_table(stdout: str) -> list[tuple[str, bool, float | None, str]]:
    """Parses `lemonade list`'s two-section table (Local / Available for
    Download) into (name, downloaded, size_gb, recipe) tuples.
    """
    rows = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Local") or line.startswith("Available for Download"):
            continue
        if line.startswith("Model Name") or set(line) == {"-"}:
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 4:
            continue
        name, downloaded_str, size_str, recipe = parts[0], parts[1], parts[2], parts[3]
        size_gb = float(size_str) if size_str.replace(".", "", 1).isdigit() else None
        rows.append((name, downloaded_str.strip().lower() == "yes", size_gb, recipe.strip()))
    return rows


def fetch_catalog(lemonade_exe, host: str, port: int, name_filter: str | None = None) -> list[CatalogModel]:
    args = [str(lemonade_exe), "--host", host, "--port", str(port), "list"]
    if name_filter:
        args.append(name_filter)
    proc = subprocess.run(args, capture_output=True, text=True, timeout=30)

    labels_by_id = _fetch_downloaded_labels(host, port)
    catalog = []
    for name, downloaded, size_gb, recipe in _parse_list_table(proc.stdout):
        catalog.append(CatalogModel(
            id=name, recipe=recipe, downloaded=downloaded, size_gb=size_gb,
            labels=labels_by_id.get(name, []),
        ))
    return catalog


def _report_backend(model: CatalogModel, backend: str) -> str:
    if model.recipe == "ryzenai-llm" and model.id.lower().endswith("-hybrid"):
        return "hybrid"
    return backend


def build_ad_hoc_spec(model: CatalogModel, host: str, port: int, backends: list[str] | None = None) -> ModelSpec:
    """Builds a one-off ModelSpec for a catalog model without touching
    bench/models.py. `backends` restricts to specific report-facing backend
    names; defaults to every backend installed for the model's recipe.
    """
    installed = _fetch_installed_backends(host, port)
    available = RECIPE_BACKENDS.get(model.recipe, [])
    installed_for_recipe = installed.get(model.recipe, set())
    usable = [b for b in available if b in installed_for_recipe]
    if backends:
        usable = [b for b in usable if b in backends or _report_backend(model, b) in backends]

    sources = [
        BackendSource(_report_backend(model, b), model.id, lemonade_backend=b if _report_backend(model, b) != b else None)
        for b in usable
    ]
    return ModelSpec(name=model.id, workload=model.workload or "llm", sources=sources)
