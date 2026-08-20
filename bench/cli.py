"""RUN COMPLETE BENCHMARK: drives Lemonade Server through every model/backend
combination in the registry and produces the ASCII + JSON report.

Requires Lemonade Server to be running (LemonadeServer.exe, or `lemonade run`).

Usage:
    python -m bench.cli                       # run the full default registry
    python -m bench.cli --model-name "Llama-3.2-1B-Instruct"
    python -m bench.cli --backends npu hybrid  # restrict which backends to test
    python -m bench.cli --upload               # also push the report to the local leaderboard server
    python -m bench.cli --list-models [FILTER] # browse what Lemonade Server can run
    python -m bench.cli --interactive          # pick a model from the catalog and run it, no code edits
    python -m bench.cli --submit                # push the report to amdaibenchmarks as a PR-ready branch
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
from pathlib import Path

# Windows consoles default to cp1252, which can't print the box-drawing/star
# characters in the ASCII report.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from bench.catalog import build_ad_hoc_spec, fetch_catalog
from bench.config import load_config
from bench.hardware import LemonadeServerUnreachable, get_system_info
from bench.models import default_registry
from bench.report import build_ascii_report, build_json_report, save_report
from bench.runners.lemonade import run_bench
from bench.submit import SubmitError, default_repo_path, resolve_github_username, submit_report
from bench.workloads import WORKLOAD_LABELS, WORKLOAD_UNITS

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


def print_catalog(cfg, name_filter: str | None):
    print(f"Fetching model catalog from Lemonade Server ({cfg.lemonade_host}:{cfg.lemonade_port})...\n")
    catalog = fetch_catalog(cfg.lemonade_exe, cfg.lemonade_host, cfg.lemonade_port, name_filter)

    if not name_filter:
        catalog = [m for m in catalog if m.downloaded]
        print("Downloaded models (pass a search term, e.g. --list-models embed, to browse the full catalog):\n")

    by_workload: dict[str, list] = defaultdict(list)
    unsupported = []
    for m in catalog:
        if m.bench_supported:
            by_workload[m.workload].append(m)
        else:
            unsupported.append(m)

    for workload, models in by_workload.items():
        print(f"{WORKLOAD_LABELS.get(workload, workload).upper()}")
        for m in sorted(models, key=lambda x: x.id.lower()):
            size = f"{m.size_gb:.2f}GB" if m.size_gb is not None else "size n/a"
            mark = "x" if m.downloaded else " "
            print(f"  [{mark}] {m.id:<42} {size:>10}   {m.recipe}")
        print()

    if unsupported:
        print(f"Downloadable/servable but not bench-able yet ({len(unsupported)}): reranker/TTS/whisper models "
              f"— `lemonade bench` has no scenario category for them.")
        if name_filter:
            for m in sorted(unsupported, key=lambda x: x.id.lower()):
                print(f"      {m.id} ({m.recipe})")


def interactive_pick(cfg) -> list:
    """Prompts for a model search term, lists matches, and lets the user pick
    one to build an ad-hoc ModelSpec for — no bench/models.py edits needed.
    """
    query = input("Search the model catalog (Enter to browse downloaded models): ").strip()
    catalog = fetch_catalog(cfg.lemonade_exe, cfg.lemonade_host, cfg.lemonade_port, query or None)
    candidates = [m for m in catalog if m.bench_supported]
    if not query:
        candidates = [m for m in candidates if m.downloaded]

    if not candidates:
        print("No bench-able models matched. Try a different search term (e.g. 'llama', 'embed').")
        return []

    candidates.sort(key=lambda m: (not m.downloaded, m.id.lower()))
    for i, m in enumerate(candidates, 1):
        size = f"{m.size_gb:.2f}GB" if m.size_gb is not None else "size n/a"
        status = "downloaded" if m.downloaded else "not downloaded"
        print(f"  {i:>3}) {m.id:<42} {size:>10}  {m.workload:<10} {status}")

    choice = input("\nPick a number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(candidates)):
        print("Not a valid choice.")
        return []
    model = candidates[int(choice) - 1]

    spec = build_ad_hoc_spec(model, cfg.lemonade_host, cfg.lemonade_port)
    if not spec.sources:
        print(f"\n{model.id} uses the {model.recipe!r} recipe, but no backend for it is installed yet.")
        print(f"Run `lemonade backends install {model.recipe}:<vulkan|cpu|rocm>` (or :npu) and try again.")
        return []

    backend_names = ", ".join(s.backend for s in spec.sources)
    if not model.downloaded:
        size = f"{model.size_gb:.2f}GB" if model.size_gb is not None else "an unknown size"
        confirm = input(f"\n{model.id} isn't downloaded yet ({size}). Pull it now? [y/N] ").strip().lower()
        if confirm != "y":
            print("Skipped.")
            return []

    print(f"\nWill benchmark {model.id} on: {backend_names}\n")
    return [spec]


def run_registry(registry, system, cfg, args) -> dict:
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
    return results_by_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Lemonade Lab benchmark suite via Lemonade Server.")
    parser.add_argument("--backends", nargs="+", help="Restrict to these report-facing backends (cpu/vulkan/rocm/npu/hybrid)")
    parser.add_argument("--model-name", help="Only benchmark the model with this display name")
    parser.add_argument("--list-models", nargs="?", const="", metavar="FILTER",
                         help="Print what Lemonade Server can run (optionally filtered) and exit")
    parser.add_argument("--interactive", "-i", action="store_true",
                         help="Pick a model from the Lemonade catalog interactively instead of using the registry")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-auto-pull", action="store_true", help="Fail instead of downloading missing models")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--upload", action="store_true", help="POST the finished report to a local leaderboard server")
    parser.add_argument("--upload-url", default="http://127.0.0.1:8787", help="Leaderboard server base URL")
    parser.add_argument("--label", help="Optional label to attach to this run when uploading")
    parser.add_argument("--submit", action="store_true",
                         help="Push this report as a branch to an amdaibenchmarks checkout and print a PR link")
    parser.add_argument("--submit-repo", type=Path, default=None,
                         help="Path to an amdaibenchmarks checkout (default: a sibling 'amdaibenchmarks' directory)")
    parser.add_argument("--github-username", default=None,
                         help="Credit this GitHub username on the submitted report (default: local_config.json's "
                              "github_username, or whoever `gh auth login` is signed in as)")
    parser.add_argument("--no-banner", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_banner:
        print(BANNER)

    cfg = load_config()

    if args.list_models is not None:
        print_catalog(cfg, args.list_models or None)
        return 0

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

    if args.interactive:
        registry = interactive_pick(cfg)
        if not registry:
            return 1
    else:
        registry = default_registry()
        if args.model_name:
            registry = [m for m in registry if m.name == args.model_name]
            if not registry:
                print(f"No model named {args.model_name!r} in the registry. Try --list-models or --interactive.")
                return 1

    results_by_model = run_registry(registry, system, cfg, args)

    print(build_ascii_report(system, results_by_model))

    settings = {
        "runs": args.runs,
        "warmup": args.warmup,
        "timeout": args.timeout,
        "auto_pull": not args.no_auto_pull,
        "backends": args.backends or "all",
    }

    ascii_path, json_path = save_report(system, results_by_model, args.out_dir, settings=settings)
    print(f"\nSaved: {ascii_path}")
    print(f"Saved: {json_path}")

    if args.upload or args.submit:
        json_report = build_json_report(system, results_by_model, settings=settings)

    if args.upload:
        print(upload_report(json_report, args.upload_url, args.label))

    if args.submit:
        repo_path = args.submit_repo or default_repo_path()
        github_username = resolve_github_username(args.github_username or cfg.github_username)
        if github_username:
            print(f"Crediting this submission to GitHub user: {github_username}")
        else:
            print("No GitHub username configured or detected (set github_username in local_config.json, "
                  "pass --github-username, or run `gh auth login`) — submitting without attribution.")
        try:
            print("\n" + submit_report(json_report, repo_path, github_username=github_username))
        except SubmitError as e:
            print(f"\nSubmit failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
