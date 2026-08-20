import subprocess
from pathlib import Path

import pytest

from bench import catalog as catalog_module

SAMPLE_LIST_OUTPUT = (
    "Local\r\n"
    "Model Name                              Downloaded     Size (GB)      Details\r\n"
    "----------------------------------------------------------------------------------------------------\r\n"
    "Llama-3.2-1B-Instruct-GGUF              Yes                0.78       llamacpp            \r\n"
    "Llama-3.2-1B-Instruct-Hybrid            Yes                1.76       ryzenai-llm         \r\n"
    "kokoro-v1                               Yes                 N/A       kokoro              \r\n"
    "----------------------------------------------------------------------------------------------------\r\n"
    "\r\n"
    "Available for Download\r\n"
    "Model Name                              Downloaded     Size (GB)      Details\r\n"
    "----------------------------------------------------------------------------------------------------\r\n"
    "Bonsai-1.7B-gguf                        No                 0.25       llamacpp            \r\n"
    "nomic-embed-text-v1-GGUF                No                 0.07       llamacpp            \r\n"
    "SD-Turbo-GGUF                           No                 1.88       sd-cpp              \r\n"
    "bge-reranker-v2-m3-GGUF                 No                 0.64       llamacpp            \r\n"
    "----------------------------------------------------------------------------------------------------\r\n"
)


def _fake_completed(stdout):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_parse_list_table_extracts_all_rows():
    rows = catalog_module._parse_list_table(SAMPLE_LIST_OUTPUT)
    names = [r[0] for r in rows]
    assert "Llama-3.2-1B-Instruct-GGUF" in names
    assert "Bonsai-1.7B-gguf" in names
    assert len(rows) == 7


def test_parse_list_table_handles_na_size():
    rows = catalog_module._parse_list_table(SAMPLE_LIST_OUTPUT)
    kokoro = next(r for r in rows if r[0] == "kokoro-v1")
    assert kokoro[2] is None  # size_gb


def test_parse_list_table_downloaded_flag():
    rows = catalog_module._parse_list_table(SAMPLE_LIST_OUTPUT)
    by_name = {r[0]: r for r in rows}
    assert by_name["Llama-3.2-1B-Instruct-GGUF"][1] is True
    assert by_name["Bonsai-1.7B-gguf"][1] is False


def test_workload_inference():
    llm = catalog_module.CatalogModel(id="Llama-3.2-1B-Instruct-GGUF", recipe="llamacpp", downloaded=True, size_gb=0.78)
    assert llm.workload == "llm"
    assert llm.bench_supported

    embed = catalog_module.CatalogModel(id="nomic-embed-text-v1-GGUF", recipe="llamacpp", downloaded=False, size_gb=0.07)
    assert embed.workload == "embedding"
    assert embed.bench_supported

    rerank = catalog_module.CatalogModel(id="bge-reranker-v2-m3-GGUF", recipe="llamacpp", downloaded=False, size_gb=0.64)
    assert rerank.workload is None
    assert not rerank.bench_supported

    image = catalog_module.CatalogModel(id="SD-Turbo-GGUF", recipe="sd-cpp", downloaded=False, size_gb=1.88)
    assert image.workload == "image_gen"
    assert image.bench_supported

    whisper = catalog_module.CatalogModel(id="Whisper-Tiny", recipe="whispercpp", downloaded=True, size_gb=0.07)
    assert whisper.workload is None
    assert not whisper.bench_supported


def test_report_backend_hybrid_suffix():
    hybrid_model = catalog_module.CatalogModel(id="Llama-3.2-1B-Instruct-Hybrid", recipe="ryzenai-llm", downloaded=True, size_gb=1.76)
    assert catalog_module._report_backend(hybrid_model, "npu") == "hybrid"

    npu_model = catalog_module.CatalogModel(id="Llama-3.2-1B-Instruct-NPU", recipe="ryzenai-llm", downloaded=True, size_gb=1.82)
    assert catalog_module._report_backend(npu_model, "npu") == "npu"


def test_fetch_catalog_merges_subprocess_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _fake_completed(SAMPLE_LIST_OUTPUT))
    monkeypatch.setattr(catalog_module, "_fetch_downloaded_labels", lambda host, port: {"Llama-3.2-1B-Instruct-GGUF": ["chat", "tool-calling"]})

    result = catalog_module.fetch_catalog(Path("lemonade.exe"), "127.0.0.1", 1234)
    llama = next(m for m in result if m.id == "Llama-3.2-1B-Instruct-GGUF")
    assert llama.labels == ["chat", "tool-calling"]
    assert llama.downloaded is True

    bonsai = next(m for m in result if m.id == "Bonsai-1.7B-gguf")
    assert bonsai.labels == []
    assert bonsai.downloaded is False


def test_build_ad_hoc_spec_uses_only_installed_backends(monkeypatch):
    monkeypatch.setattr(catalog_module, "_fetch_installed_backends", lambda host, port: {"llamacpp": {"cpu", "vulkan"}})
    model = catalog_module.CatalogModel(id="Bonsai-1.7B-gguf", recipe="llamacpp", downloaded=False, size_gb=0.25)

    spec = catalog_module.build_ad_hoc_spec(model, "127.0.0.1", 1234)
    backends = {s.backend for s in spec.sources}
    assert backends == {"cpu", "vulkan"}  # rocm excluded, not installed
    assert spec.workload == "llm"


def test_build_ad_hoc_spec_hybrid_model_maps_to_npu_bench_backend(monkeypatch):
    monkeypatch.setattr(catalog_module, "_fetch_installed_backends", lambda host, port: {"ryzenai-llm": {"npu"}})
    model = catalog_module.CatalogModel(id="Llama-3.2-1B-Instruct-Hybrid", recipe="ryzenai-llm", downloaded=True, size_gb=1.76)

    spec = catalog_module.build_ad_hoc_spec(model, "127.0.0.1", 1234)
    assert len(spec.sources) == 1
    assert spec.sources[0].backend == "hybrid"
    assert spec.sources[0].bench_backend == "npu"


def test_build_ad_hoc_spec_no_installed_backend_yields_empty_sources(monkeypatch):
    monkeypatch.setattr(catalog_module, "_fetch_installed_backends", lambda host, port: {})
    model = catalog_module.CatalogModel(id="Bonsai-1.7B-gguf", recipe="llamacpp", downloaded=False, size_gb=0.25)

    spec = catalog_module.build_ad_hoc_spec(model, "127.0.0.1", 1234)
    assert spec.sources == []
