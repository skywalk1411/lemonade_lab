"""Runs benchmarks through Lemonade Server's own `lemonade bench` command.

Lemonade Server (https://lemonade-server.ai) already manages every backend this
project cares about — llama.cpp (cpu/vulkan/rocm), flm and ryzenai-llm (npu/hybrid),
sd-cpp (image generation) — behind one CLI. `lemonade bench <model> --backend X --json`
runs a fixed battery of scenarios (bench_scenarios.json, grouped into categories:
chat, coding, embed, imagegen, long-context) against a model on a backend and
reports tokens/sec, time-to-first-token, and peak memory per scenario.

This module shells out to that command and folds the result into one BenchResult
per backend, rather than reimplementing benchmark execution against flm.exe /
llama-bench.exe directly.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .base import BenchResult

DEFAULT_TIMEOUT_S = 300

# workload -> the bench_scenarios.json categories that apply to it. Reranker,
# TTS and Whisper models can be pulled and served by Lemonade Server, but
# `lemonade bench` (as of v11.7.0) has no scenario category for them yet.
WORKLOAD_SCENARIOS = {
    "llm": ["chat"],
    "embedding": ["embed"],
    "image_gen": ["imagegen"],
}


def _primary_metric(workload: str, scenarios: list[dict]) -> tuple[float | None, str | None]:
    """Returns (value, error). Image generation has no token concept, so its
    throughput comes from wall-clock duration instead of the tps field.
    """
    if workload == "image_gen":
        durations = [s["duration_ms"]["mean"] for s in scenarios if "duration_ms" in s]
        if not durations:
            return None, "no duration data in bench output"
        return 60000 / (sum(durations) / len(durations)), None

    tps_values = [s["tps"]["mean"] for s in scenarios if "tps" in s]
    if not tps_values:
        return None, "no tps data in bench output"
    if all(v == 0 for v in tps_values):
        # Some backends (seen with flm's embedding responses) complete the
        # request but don't report token counts, so tps computes as 0/duration
        # rather than a real throughput. Surface that as unmeasurable instead
        # of a misleading "0 tok/s".
        durations = [s["duration_ms"]["mean"] for s in scenarios if "duration_ms" in s]
        duration_note = f", request completed in {sum(durations) / len(durations):.0f}ms" if durations else ""
        return None, f"backend didn't report token counts (tps unmeasurable{duration_note})"
    return sum(tps_values) / len(tps_values), None


def _extract_json(stdout: str) -> dict:
    """`lemonade bench --json` still prints human-readable progress to stdout
    ahead of the JSON payload; pull out the JSON object (the first line that's
    a lone '{' starts it).
    """
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "{":
            candidate = "\n".join(lines[i:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in `lemonade bench` output")


def run_bench(
    *,
    lemonade_exe: Path,
    host: str,
    port: int,
    model: str,
    backends: list[str],
    workload: str,
    runs: int = 3,
    warmup: int = 1,
    timeout: int = DEFAULT_TIMEOUT_S,
    auto_pull: bool = True,
) -> list[BenchResult]:
    """Benchmarks one model across one or more backends in a single `lemonade
    bench` invocation, returning one BenchResult per backend.
    """
    if not lemonade_exe.exists():
        return [BenchResult(b, model, workload=workload, error=f"lemonade.exe not found at {lemonade_exe}") for b in backends]

    categories = WORKLOAD_SCENARIOS.get(workload)
    if not categories:
        return [BenchResult(b, model, workload=workload, error=f"no bench scenario category for workload {workload!r}") for b in backends]

    args = [
        str(lemonade_exe), "--host", host, "--port", str(port),
        "bench", model,
    ]
    for b in backends:
        args += ["--backend", b]
    for c in categories:
        args += ["--scenarios", c]
    args += ["--runs", str(runs), "--warmup", str(warmup), "--timeout", str(timeout), "--json"]
    if auto_pull:
        args.append("--auto-pull")

    wall_clock_budget = timeout * runs * len(backends) * 4 + 120
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=wall_clock_budget)
    except subprocess.TimeoutExpired:
        return [BenchResult(b, model, workload=workload, error=f"timed out after {wall_clock_budget}s") for b in backends]

    combined = proc.stdout + "\n" + proc.stderr
    try:
        data = _extract_json(proc.stdout)
    except ValueError:
        tail = combined.strip().splitlines()[-5:]
        return [BenchResult(b, model, workload=workload, error=f"no JSON in output; tail: {' | '.join(tail)}") for b in backends]

    models = data.get("models", [])
    if not models:
        err = data.get("error", "no models in bench output")
        return [BenchResult(b, model, workload=workload, error=str(err)) for b in backends]

    results_by_backend = {r.get("backend"): r for r in models[0].get("results", [])}

    out = []
    for backend in backends:
        entry = results_by_backend.get(backend)
        if entry is None:
            out.append(BenchResult(backend, model, workload=workload, error="backend missing from bench output (load likely failed)", raw={"full": data}))
            continue

        scenarios = [s for s in entry.get("scenarios", []) if not s.get("all_runs_failed")]
        if not scenarios:
            failed = entry.get("scenarios", [])
            out.append(BenchResult(backend, model, workload=workload, error="all scenarios failed", raw={"scenarios": failed}))
            continue

        metric, metric_error = _primary_metric(workload, scenarios)
        if metric_error:
            out.append(BenchResult(backend, model, workload=workload, error=metric_error, raw={"scenarios": scenarios}))
            continue

        ttft_values = [s["ttft_ms"]["mean"] for s in scenarios if "ttft_ms" in s]
        mem_values = [s["memory_peak_gb"] for s in scenarios if s.get("memory_peak_gb") is not None]

        out.append(BenchResult(
            backend=backend,
            model=model,
            workload=workload,
            gen_tps=metric,
            ttft_ms=sum(ttft_values) / len(ttft_values) if ttft_values else None,
            memory_gb=max(mem_values) if mem_values else None,
            raw={"scenarios": scenarios, "hardware": data.get("hardware")},
        ))
    return out
