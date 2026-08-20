from bench.hardware import SystemInfo
from bench.report import build_ascii_report, build_json_report, _stars, _group_by_workload
from bench.runners.base import BenchResult


def make_system():
    return SystemInfo(cpu="AMD Ryzen AI 9 HX 470", memory_gb=64, gpu="AMD Radeon 890M",
                       npu="AMD XDNA (ready)", os="Windows 11")


def test_stars_best_gets_five():
    assert _stars(21.7, 21.7) == "★★★★★"


def test_stars_zero_best_is_empty():
    assert _stars(5.0, 0.0) == "☆☆☆☆☆"


def test_stars_nonzero_value_gets_at_least_one():
    assert _stars(0.01, 100.0).count("★") >= 1


def test_group_by_workload_splits_correctly():
    results_by_model = {
        "modelA": [BenchResult("cpu", "modelA", workload="llm", gen_tps=1.0)],
        "modelB": [BenchResult("cpu", "modelB", workload="embedding", gen_tps=2.0)],
    }
    grouped = _group_by_workload(results_by_model)
    assert set(grouped.keys()) == {"llm", "embedding"}
    assert "modelA" in grouped["llm"]
    assert "modelB" in grouped["embedding"]


def test_json_report_has_back_compat_models_alias_for_llm():
    results_by_model = {
        "Qwen": [BenchResult("rocm", "Qwen", workload="llm", gen_tps=21.7, ttft_ms=15.0)],
    }
    report = build_json_report(make_system(), results_by_model)
    assert report["results"]["llm"]["Qwen"]["rocm"]["value"] == 21.7
    assert report["results"]["llm"]["Qwen"]["rocm"]["unit"] == "tok/s"
    assert report["models"] == report["results"]["llm"]
    assert report["system"]["cpu"] == "AMD Ryzen AI 9 HX 470"


def test_json_report_no_models_alias_without_llm_workload():
    results_by_model = {
        "nomic": [BenchResult("cpu", "nomic", workload="embedding", gen_tps=500.0)],
    }
    report = build_json_report(make_system(), results_by_model)
    assert "models" not in report
    assert "embedding" in report["results"]


def test_json_report_records_failed_backend():
    results_by_model = {
        "Qwen": [BenchResult("cpu", "Qwen", workload="llm", error="timed out")],
    }
    report = build_json_report(make_system(), results_by_model)
    assert report["models"]["Qwen"]["cpu"] == {"error": "timed out"}


def test_json_report_omits_settings_when_not_given():
    results_by_model = {"Qwen": [BenchResult("cpu", "Qwen", workload="llm", gen_tps=1.0)]}
    report = build_json_report(make_system(), results_by_model)
    assert "settings" not in report


def test_json_report_includes_settings_when_given():
    results_by_model = {"Qwen": [BenchResult("cpu", "Qwen", workload="llm", gen_tps=1.0)]}
    settings = {"runs": 3, "warmup": 1, "timeout": 300, "auto_pull": True, "backends": "all"}
    report = build_json_report(make_system(), results_by_model, settings=settings)
    assert report["settings"] == settings


def test_ascii_report_contains_model_and_box_chars():
    results_by_model = {
        "Qwen": [
            BenchResult("rocm", "Qwen", workload="llm", gen_tps=21.7),
            BenchResult("cpu", "Qwen", workload="llm", error="oom"),
        ],
    }
    out = build_ascii_report(make_system(), results_by_model)
    assert "Qwen" in out
    assert "╔" in out and "╚" in out
    assert "ROCm" in out
    assert "failed: oom" in out
