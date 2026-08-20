# Lemonade Lab

A local AI benchmarking tool and leaderboard for AMD Ryzen AI PCs, built directly on
[Lemonade Server](https://lemonade-server.ai) — it drives `lemonade bench` across
every backend Lemonade manages (CPU, Vulkan, ROCm, NPU, Hybrid) and produces a
single comparable report, a shareable JSON file, and an optional entry on a local
cross-machine leaderboard.

This is real output from `python -m bench.cli --model-name "Llama-3.2-1B-Instruct"`
on an AMD Ryzen AI 9 HX 470 (Radeon 890M / XDNA2):

```
╔══════════════════════════════════════════════════╗
║          RYZEN AI LOCAL AI REPORT                 ║
╠══════════════════════════════════════════════════╣
║ AMD Ryzen AI 9 HX 470                             ║
║ 64GB Memory | GPU: AMD Radeon 890M Graphics       ║
║ NPU: AMD XDNA2 (ready)                            ║
║                                                    ║
║ Llama-3.2-1B-Instruct                             ║
║ ────────────────────────────────────────────────── ║
║ ROCm         77.0 tok/s   ★★★★☆                   ║
║ Vulkan       85.7 tok/s   ★★★★★                   ║
║ Hybrid       52.4 tok/s   ★★★☆☆                   ║
║ NPU          41.1 tok/s   ★★☆☆☆                   ║
║ CPU          76.6 tok/s   ★★★★☆                   ║
╚══════════════════════════════════════════════════╝
```

## Why it's built this way

Lemonade Server already does the hard part: it manages backend runtimes
(llama.cpp for cpu/vulkan/rocm, flm and ryzenai-llm for npu/hybrid, sd-cpp for
image generation), knows how to load and serve any model in its catalog, and
ships its own `lemonade bench` command that measures tokens/sec, time-to-first-token,
and peak memory per scenario. Lemonade Lab is a thin, purpose-built layer on top:
it decides *what* to benchmark (which models, across which backends, grouped for
comparison), and turns the results into a report and a browsable leaderboard —
it does not reimplement model serving or talk to flm.exe / llama.cpp binaries directly.

## What's here

- **Every backend, one report** — CPU, Vulkan, ROCm, NPU, and Hybrid results for
  the same model side by side, ranked with star ratings.
- **Multi-workload benchmarking** — LLM generation, embedding throughput, and image
  generation today (the `chat`/`coding`, `embed`, and `imagegen` categories
  `lemonade bench` supports). Reranker, TTS, and Whisper models can be pulled and
  served by Lemonade Server, but `lemonade bench` has no scenario category for
  them yet — see Roadmap.
- **A real local leaderboard** — a FastAPI + SQLite server you can run right now to
  collect and rank benchmark submissions across machines. It's the same app you'd
  point at a hosted database to make it a genuine shared community leaderboard.
- **A static report viewer and a leaderboard UI**, both plain HTML/CSS/JS, no build
  step.

## Requirements

- Windows with an AMD Ryzen AI processor
- [Lemonade Server](https://lemonade-server.ai) installed and running
- Python 3.10+

## Setup

```
pip install -r requirements.txt
copy config.example.json local_config.json
```

Edit `local_config.json` to point at your Lemonade install and the host/port
your server is running on (`lemonade status` will tell you):

```json
{
  "lemonade_exe": "C:\\Users\\you\\AppData\\Local\\lemonade_server\\bin\\lemonade.exe",
  "lemonade_host": "127.0.0.1",
  "lemonade_port": 1234
}
```

`local_config.json` is machine-specific and gitignored.

Edit `bench/models.py` (`default_registry`) to add more models — each `ModelSpec`
groups a display name and workload with one `BackendSource` per backend you want
tested; run `lemonade list` to see the full catalog.

## Running a benchmark

Make sure Lemonade Server is running (launch it from the Start menu, or run
`LemonadeServer.exe` / `lemonade-app.exe` directly — `lemonade status` confirms
it's up and tells you which port), then:

```
python -m bench.cli
```

Options:

```
--backends npu hybrid vulkan       # restrict to specific report-facing backends
--model-name "Llama-3.2-1B-Instruct"
--runs 3 --warmup 1 --timeout 300
--no-auto-pull                     # fail instead of downloading missing models
--upload --label "my-run"          # also push the report to the local leaderboard
--submit                           # push the report to amdaibenchmarks as a PR-ready branch
```

### Submitting a result to the community leaderboard

[amdaibenchmarks](https://github.com/skywalk1411/amdaibenchmarks) is the
public, PR-based leaderboard these reports feed. `--submit` automates the
mechanical part of contributing to it:

```
git clone https://github.com/skywalk1411/amdaibenchmarks.git ../amdaibenchmarks
python -m bench.cli --submit
```

It expects a sibling `../amdaibenchmarks` checkout by default (override with
`--submit-repo <path>`), and will:

1. Fast-forward its `main`, then create a `submit/<slug>` branch
2. Write the report into `results/` with a name matching that repo's convention
3. Run *that repo's own* `scripts/validate.py` against it — nothing gets
   pushed if it fails
4. Commit, push the branch, and print a compare URL to open the PR

It never pushes straight to `main`, even for repo owners — every result is
reviewable in a PR before it counts, by design. If the working tree isn't
clean, or there's nothing installed to push with, it stops and tells you why
rather than guessing.

### Trying a model without editing the registry

`bench/models.py` (`default_registry`) is the *repeatable* set of models — the
one CI or a scripted run would use. To try something else on the fly, you
don't need to touch that file:

```
python -m bench.cli --list-models              # what's downloaded and bench-able right now
python -m bench.cli --list-models qwen          # search Lemonade's full catalog (~230 models)
python -m bench.cli --interactive               # pick a catalog model and run it
```

`--interactive` asks what to search for (or just press Enter to browse
downloaded models), lists matches with their workload and download status,
and — once you pick one — auto-detects which of its backends (cpu/vulkan/rocm/npu/hybrid)
are actually installed on this machine and benchmarks it across all of them.
If the model isn't downloaded yet, it asks before pulling it.

Reranker, TTS, and Whisper models show up in `--list-models` too (Lemonade
Server can download and serve them), just marked as not bench-able yet — see
Known issues below.

This prints the ASCII report to the console and writes both a `.txt` and `.json`
report to `reports/`.

## The leaderboard server

```
uvicorn api.server:app --reload --port 8787
```

Then either run a benchmark with `--upload`, or `POST` a `report_*.json` file
yourself to `http://127.0.0.1:8787/api/reports` as `{"report": {...}, "label": "..."}`.

The server also serves the dashboard itself: `http://127.0.0.1:8787/` for the
single-report viewer and `http://127.0.0.1:8787/leaderboard.html` for the ranked,
cross-machine comparison view.

| Endpoint | What it does |
|---|---|
| `POST /api/reports` | Upload a report (`{report, label?}`) |
| `GET /api/reports` | List submitted reports (summary) |
| `GET /api/reports/{id}` | Full stored report JSON |
| `DELETE /api/reports/{id}` | Remove a report |
| `GET /api/catalog` | Distinct (workload, model) pairs across all reports |
| `GET /api/leaderboard?workload=llm&model=...` | Ranked rows for a workload/model, fastest first |

## Viewing a single report offline

Open `web/index.html` directly in a browser (no server needed) and drop in a
`report_*.json` file from `reports/`.

## Project layout

```
bench/
  config.py          lemonade.exe path + host/port
  hardware.py         reads /api/v1/system-info for CPU/GPU/NPU/RAM
  models.py            model registry: display name -> per-backend Lemonade catalog models
  catalog.py             browses the full Lemonade catalog + install status, for --list-models/--interactive
  submit.py               pushes a report to an amdaibenchmarks checkout as a PR-ready branch
  workloads.py             workload -> label + unit (tok/s, img/min) shared by report.py and the runner
  report.py             ASCII + JSON report rendering, grouped by workload
  cli.py                 orchestrates everything, optional --upload
  runners/
    base.py                BenchResult, the shared result type
    lemonade.py             wraps `lemonade bench --json`
api/
  server.py             FastAPI leaderboard app
  db.py                  SQLite storage
web/
  index.html             static single-report viewer
  leaderboard.html        cross-machine leaderboard UI
  style.css               shared styling
tests/                  pytest suite for report/runner/models/catalog/submit/api
.github/workflows/ci.yml  runs the test suite on push/PR
```

## Known issues

- `qwen3.6-moe-35b-a3b-FLM` (a large vision-capable MoE model) loads successfully
  on the NPU backend and returns HTTP 200 for chat requests, but with 0 output
  tokens — its chat template likely doesn't produce a valid text-only completion
  through flm. Left in the registry since it may be resolved by a future
  flm/Lemonade release; the tool reports it as a clean failure rather than
  fabricating a result.

## Roadmap

- Reranker, TTS (kokoro), and Whisper benchmarking once `lemonade bench` grows
  scenario categories for them — the backends are already installable and models
  pullable today, `bench` just doesn't score them yet.
- Power efficiency (tok/J) — needs a real power-draw source; not faked in the
  meantime.
- Deploying the leaderboard server somewhere shared instead of localhost.
