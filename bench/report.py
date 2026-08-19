"""Renders benchmark results as the boxed ASCII report and the shareable JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .hardware import SystemInfo
from .runners.base import BenchResult
from .workloads import WORKLOAD_LABELS, WORKLOAD_UNITS

BOX_WIDTH = 54
BACKEND_ORDER = ["rocm", "vulkan", "hybrid", "npu", "cpu"]
BACKEND_LABELS = {"rocm": "ROCm", "vulkan": "Vulkan", "hybrid": "Hybrid", "npu": "NPU", "cpu": "CPU"}


def _stars(value: float, best: float) -> str:
    if best <= 0:
        return "☆☆☆☆☆"
    ratio = max(0.0, min(1.0, value / best))
    filled = round(ratio * 5)
    filled = max(1, filled) if value > 0 else 0
    return "★" * filled + "☆" * (5 - filled)


def _pad_line(text: str) -> str:
    inner = BOX_WIDTH - 2
    return "║" + text.ljust(inner) + "║"


def _group_by_workload(results_by_model: dict[str, list[BenchResult]]) -> dict[str, dict[str, list[BenchResult]]]:
    """model_name -> [results] becomes workload -> model_name -> [results]."""
    grouped: dict[str, dict[str, list[BenchResult]]] = {}
    for model_name, results in results_by_model.items():
        for r in results:
            grouped.setdefault(r.workload, {}).setdefault(model_name, []).append(r)
    return grouped


def build_ascii_report(system: SystemInfo, results_by_model: dict[str, list[BenchResult]]) -> str:
    lines = []
    lines.append("╔" + "═" * (BOX_WIDTH - 2) + "╗")
    lines.append(_pad_line("  RYZEN AI LOCAL AI REPORT".ljust(BOX_WIDTH - 2)))
    lines.append("╠" + "═" * (BOX_WIDTH - 2) + "╣")
    lines.append(_pad_line(""))
    lines.append(_pad_line(f" {system.cpu}"))
    lines.append(_pad_line(f" {system.memory_gb}GB Memory  |  GPU: {system.gpu}"))
    lines.append(_pad_line(f" NPU: {system.npu}"))
    lines.append(_pad_line(""))

    grouped = _group_by_workload(results_by_model)
    for workload, models in grouped.items():
        unit = WORKLOAD_UNITS.get(workload, "tok/s")
        lines.append(_pad_line(f" {WORKLOAD_LABELS.get(workload, workload).upper()}"))
        lines.append(_pad_line(""))

        for model_name, results in models.items():
            lines.append(_pad_line(f" {model_name}"))
            lines.append(_pad_line(" " + "─" * (BOX_WIDTH - 4)))

            ok_results = {r.backend: r for r in results if r.ok}
            best = max((r.gen_tps for r in ok_results.values()), default=0.0)

            for backend in BACKEND_ORDER:
                r = next((x for x in results if x.backend == backend), None)
                if r is None:
                    continue
                label = BACKEND_LABELS[backend]
                if r.ok:
                    stars = _stars(r.gen_tps, best)
                    lines.append(_pad_line(f" {label:<10} {r.gen_tps:>8.1f} {unit:<8}{stars}"))
                else:
                    msg = f" {label:<10} {'—':>8}          (failed: {r.error})"
                    lines.append(_pad_line(msg[:BOX_WIDTH - 2]))
            lines.append(_pad_line(""))

    lines.append("╚" + "═" * (BOX_WIDTH - 2) + "╝")
    return "\n".join(lines)


def build_json_report(system: SystemInfo, results_by_model: dict[str, list[BenchResult]]) -> dict:
    grouped = _group_by_workload(results_by_model)
    results: dict[str, dict[str, dict]] = {}

    for workload, models in grouped.items():
        unit = WORKLOAD_UNITS.get(workload, "tok/s")
        results[workload] = {}
        for model_name, model_results in models.items():
            entry = {}
            for r in model_results:
                if r.ok:
                    entry[r.backend] = {
                        "value": round(r.gen_tps, 2),
                        "unit": unit,
                        "ttft_ms": round(r.ttft_ms, 2) if r.ttft_ms is not None else None,
                        "memory_gb": round(r.memory_gb, 2) if r.memory_gb is not None else None,
                    }
                else:
                    entry[r.backend] = {"error": r.error}
            results[workload][model_name] = entry

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": system.as_dict(),
        "results": results,
    }
    # Back-compat top-level alias matching the original flat schema (LLM results only).
    if "llm" in results:
        report["models"] = results["llm"]
    return report


def save_report(system: SystemInfo, results_by_model: dict[str, list[BenchResult]], out_dir) -> tuple[str, str]:
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ascii_report = build_ascii_report(system, results_by_model)
    json_report = build_json_report(system, results_by_model)

    ascii_path = out_dir / f"report_{stamp}.txt"
    json_path = out_dir / f"report_{stamp}.json"
    ascii_path.write_text(ascii_report, encoding="utf-8")
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    return str(ascii_path), str(json_path)
