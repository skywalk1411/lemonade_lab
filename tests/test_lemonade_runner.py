import subprocess
from pathlib import Path

import pytest

from bench.runners import lemonade as lemonade_runner

PROGRESS_TEXT = "\n=== [Llama-3.2-1B-Instruct-GGUF] llamacpp/vulkan (ctx=4096) ===\n  Scenario: chat-short (chat)\n"

CANNED_BENCH_JSON = """{
  "hardware": {"cpu": "AMD Ryzen AI 9 HX 470", "ram_gb": 64.0},
  "models": [
    {
      "model": "Llama-3.2-1B-Instruct-GGUF",
      "results": [
        {
          "backend": "vulkan",
          "recipe": "llamacpp",
          "scenarios": [
            {"name": "chat-short", "category": "chat", "failed_runs": 0,
             "tps": {"mean": 89.2}, "ttft_ms": {"mean": 34.1}, "memory_peak_gb": 11.1},
            {"name": "chat-long-output", "category": "chat", "failed_runs": 0,
             "tps": {"mean": 82.3}, "ttft_ms": {"mean": 47.7}, "memory_peak_gb": 11.1}
          ]
        },
        {
          "backend": "cpu",
          "recipe": "llamacpp",
          "scenarios": [
            {"name": "chat-short", "category": "chat", "all_runs_failed": true, "failed_runs": 1}
          ]
        }
      ]
    }
  ]
}"""


def _fake_completed(stdout):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


@pytest.fixture
def existing_exe(tmp_path):
    exe = tmp_path / "lemonade.exe"
    exe.write_text("stub")
    return exe


def test_run_bench_parses_backend_results(existing_exe, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: _fake_completed(PROGRESS_TEXT + CANNED_BENCH_JSON),
    )
    results = lemonade_runner.run_bench(
        lemonade_exe=existing_exe, host="127.0.0.1", port=1234,
        model="Llama-3.2-1B-Instruct-GGUF", backends=["vulkan", "cpu"], workload="llm",
    )
    by_backend = {r.backend: r for r in results}

    vulkan = by_backend["vulkan"]
    assert vulkan.ok
    assert vulkan.gen_tps == pytest.approx((89.2 + 82.3) / 2)
    assert vulkan.ttft_ms == pytest.approx((34.1 + 47.7) / 2)
    assert vulkan.memory_gb == 11.1

    cpu = by_backend["cpu"]
    assert not cpu.ok
    assert "failed" in cpu.error


def test_run_bench_missing_exe():
    results = lemonade_runner.run_bench(
        lemonade_exe=Path("Z:/nope/lemonade.exe"), host="127.0.0.1", port=1234,
        model="m", backends=["vulkan"], workload="llm",
    )
    assert len(results) == 1
    assert not results[0].ok
    assert "not found" in results[0].error


def test_run_bench_unknown_workload(existing_exe):
    results = lemonade_runner.run_bench(
        lemonade_exe=existing_exe, host="127.0.0.1", port=1234,
        model="m", backends=["vulkan"], workload="tts",
    )
    assert len(results) == 1
    assert not results[0].ok
    assert "no bench scenario category" in results[0].error


def test_run_bench_no_json_in_output(existing_exe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _fake_completed("nothing but progress text"))
    results = lemonade_runner.run_bench(
        lemonade_exe=existing_exe, host="127.0.0.1", port=1234,
        model="m", backends=["vulkan"], workload="llm",
    )
    assert len(results) == 1
    assert not results[0].ok
    assert "no JSON" in results[0].error


def test_run_bench_backend_missing_from_results(existing_exe, monkeypatch):
    json_missing_rocm = CANNED_BENCH_JSON  # only has vulkan/cpu
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _fake_completed(json_missing_rocm))
    results = lemonade_runner.run_bench(
        lemonade_exe=existing_exe, host="127.0.0.1", port=1234,
        model="Llama-3.2-1B-Instruct-GGUF", backends=["rocm"], workload="llm",
    )
    assert len(results) == 1
    assert not results[0].ok
    assert "missing from bench output" in results[0].error


ZERO_TPS_JSON = """{
  "hardware": {"cpu": "AMD Ryzen AI 9 HX 470", "ram_gb": 64.0},
  "models": [
    {
      "model": "embed-gemma-300m-FLM",
      "results": [
        {
          "backend": "npu",
          "recipe": "flm",
          "scenarios": [
            {"name": "embed-long-string", "category": "embed", "failed_runs": 0,
             "input_tokens": 0, "output_tokens": 0,
             "tps": {"mean": 0.0}, "ttft_ms": {"mean": 11322.8}, "duration_ms": {"mean": 11322.8},
             "memory_peak_gb": 8.0}
          ]
        }
      ]
    }
  ]
}"""


def test_run_bench_zero_tps_reported_as_unmeasurable_not_success(existing_exe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _fake_completed(ZERO_TPS_JSON))
    results = lemonade_runner.run_bench(
        lemonade_exe=existing_exe, host="127.0.0.1", port=1234,
        model="embed-gemma-300m-FLM", backends=["npu"], workload="embedding",
    )
    assert len(results) == 1
    assert not results[0].ok
    assert "unmeasurable" in results[0].error
    assert "11323ms" in results[0].error


def test_run_bench_timeout(existing_exe, monkeypatch):
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="lemonade", timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    results = lemonade_runner.run_bench(
        lemonade_exe=existing_exe, host="127.0.0.1", port=1234,
        model="m", backends=["vulkan", "cpu"], workload="llm",
    )
    assert len(results) == 2
    assert all(not r.ok and "timed out" in r.error for r in results)
