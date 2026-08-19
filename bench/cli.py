"""RUN COMPLETE BENCHMARK: drives Lemonade Server through every model/backend
combination in the registry and produces the ASCII + JSON report.

Requires Lemonade Server to be running (LemonadeServer.exe, or `lemonade run`).

Usage:
    python -m bench.cli                       # run the full default registry
    python -m bench.cli --model-name "Llama-3.2-1B-Instruct"
    python -m bench.cli --backends npu hybrid  # restrict which backends to test
    python -m bench.cli --upload               # also push the report to the local leaderboard server
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

# Windows consoles default to cp1252, which can't print the box-drawing/star
# characters in the ASCII report.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from bench.config import load_config
from bench.hardware import LemonadeServerUnreachable, get_system_info
from bench.models import default_registry
from bench.report import build_ascii_report, build_json_report, save_report
from bench.runners.lemonade import run_bench
from bench.workloads import WORKLOAD_UNITS

BANNER = r"""
   _ ___ __  __  ___  _  _ _   ___  ___     _      _   ___
  | | __|  \/  |/ _ \| \| / \ |   \| __|   | |    /_\ | _ )
  | | _|| |\/| | (_) | .` | _ \| |) | _|    | |__ / _ \| _ \
  |_|___|_|  |_|\___/|_|\_/_/ \_\___/___|   |____/_/ \_\___/

  Local AI benchmark suite for AMD Ryzen AI PCs, built on Lemonade Server
"""


def bench_model(spec, cfg, *, only_backends, runs, warmup, timeout, auto_pull):
    """Runs every BackendSource for a ModelSpec, grouping sources that share a
    lemonade_model into a single `lemonade bench` call (one model load instead
    of one per backend), then relabels results back to the report-facing
    backend name.
    """
    sources = [s for s in spec.sources if only_backends is None or s.backend in only_backends]
    by_lemonade_model = defaultdict(list)
    for s in sources:
        by_lemonade_model[s.lemonade_model].append(s)

    results = []
    for lemonade_model, group in by_lemonade_model.items():
        bench_backends = [s.bench_backend for s in group]
        print(f"  [{lemonade_model} @ {'+'.join(bench_backends)}] running...", end=" ", flush=True)
        t0 = time.monotonic()
        raw_results = run_bench(
            lemonade_exe=cfg.lemonade_exe, host=cfg.lemonade_host, port=cfg.lemonade_port,
            model=lemonade_model, backends=bench_backends, workload=spec.workload,
            runs=runs, warmup=warmup, timeout=timeout, auto_pull=auto_pull,
        )
        elapsed = time.monotonic() - t0

        by_bench_backend = {r.backend: r for r in raw_results}
        for source in group:
            r = by_bench_backend.get(source.bench_backend)
            if r is None:
                continue
            r.backend = source.backend       # relabel npu->hybrid etc. for report-facing name
            r.model = spec.name
            if r.ok:
                unit = WORKLOAD_UNITS.get(spec.workload, "tok/s")
                print(f"{source.backend}: {r.gen_tps:>8.2f} {unit}", end="  ")
            else:
                print(f"{source.backend}: FAILED ({r.error})", end="  ")
            results.append(r)
        print(f"({elapsed:.1f}s)")
    return results


def upload_report(json_report: dict, url: str, label: str | None) -> str:
    body = json.dumps({"report": json_report, "label": label}).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/reports", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return f"uploaded as report #{data.get('id')}"
    except urllib.error.URLError as e:
        return f"upload failed ({e}) — is `uvicorn api.server:app` running?"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Lemonade Lab benchmark suite via Lemonade Server.")
    parser.add_argument("--backends", nargs="+", help="Restrict to these report-facing backends (cpu/vulkan/rocm/npu/hybrid)")
    parser.add_argument("--model-name", help="Only benchmark the model with this display name")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-auto-pull", action="store_true", help="Fail instead of downloading missing models")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--upload", action="store_true", help="POST the finished report to a local leaderboard server")
    parser.add_argument("--upload-url", default="http://127.0.0.1:8787", help="Leaderboard server base URL")
    parser.add_argument("--label", help="Optional label to attach to this run when uploading")
    parser.add_argument("--no-banner", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_banner:
        print(BANNER)

    cfg = load_config()

    print("Detecting hardware via Lemonade Server...")
    try:
        system = get_system_info(cfg.lemonade_host, cfg.lemonade_port)
    except LemonadeServerUnreachable as e:
        print(str(e))
        return 1
    print(f"  CPU:    {system.cpu}")
    print(f"  GPU:    {system.gpu}")
    print(f"  NPU:    {system.npu}")
    print(f"  Memory: {system.memory_gb}GB")
    print()

    registry = default_registry()
    if args.model_name:
        registry = [m for m in registry if m.name == args.model_name]
        if not registry:
            print(f"No model named {args.model_name!r} in the registry.")
            return 1

    results_by_model: dict[str, list] = {}
    for spec in registry:
        print(f"=== {spec.name} ({spec.workload}) ===")
        results = bench_model(
            spec, cfg, only_backends=args.backends,
            runs=args.runs, warmup=args.warmup, timeout=args.timeout,
            auto_pull=not args.no_auto_pull,
        )
        results_by_model[spec.name] = results
        print()

    print(build_ascii_report(system, results_by_model))

    ascii_path, json_path = save_report(system, results_by_model, args.out_dir)
    print(f"\nSaved: {ascii_path}")
    print(f"Saved: {json_path}")

    if args.upload:
        json_report = build_json_report(system, results_by_model)
        print(upload_report(json_report, args.upload_url, args.label))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
