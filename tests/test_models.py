from bench.models import default_registry


def test_default_registry_returns_specs_with_sources():
    specs = default_registry()
    assert len(specs) > 0
    for spec in specs:
        assert spec.workload in {"llm", "embedding", "image_gen"}
        assert len(spec.sources) > 0
        for source in spec.sources:
            assert source.lemonade_model
            assert source.bench_backend  # falls back to source.backend when unset


def test_hybrid_source_overrides_bench_backend_to_npu():
    specs = default_registry()
    llama = next(s for s in specs if s.name == "Llama-3.2-1B-Instruct")
    hybrid = next(s for s in llama.sources if s.backend == "hybrid")
    assert hybrid.lemonade_backend == "npu"
    assert hybrid.bench_backend == "npu"


def test_plain_backend_defaults_bench_backend_to_itself():
    specs = default_registry()
    llama = next(s for s in specs if s.name == "Llama-3.2-1B-Instruct")
    cpu = next(s for s in llama.sources if s.backend == "cpu")
    assert cpu.lemonade_backend is None
    assert cpu.bench_backend == "cpu"
