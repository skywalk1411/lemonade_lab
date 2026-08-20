import subprocess

import pytest

from bench.submit import SubmitError, submit_report, suggest_filename

SAMPLE_REPORT = {
    "timestamp": "2026-08-19T00:00:00Z",
    "system": {"cpu": "AMD Ryzen AI 9 HX 470 w/ Radeon 890M", "gpu": "AMD Radeon 890M", "npu": "AMD XDNA2", "memory": "64GB", "os": "Windows 11"},
    "results": {
        "llm": {"Llama-3.2-1B-Instruct": {"vulkan": {"value": 85.7, "unit": "tok/s", "ttft_ms": 42.1, "memory_gb": 8.8}}},
    },
}


def _git(args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


@pytest.fixture
def amdaibenchmarks_repo(tmp_path):
    """A real temp git repo shaped like amdaibenchmarks, with a bare 'origin'
    to push to — exercises submit_report's actual git plumbing rather than
    mocking each subprocess call.
    """
    bare = tmp_path / "remote.git"
    repo = tmp_path / "amdaibenchmarks"
    _git(["init", "--bare", "-b", "main", str(bare)], cwd=tmp_path)
    _git(["clone", str(bare), str(repo)], cwd=tmp_path)
    _git(["config", "user.email", "test@example.com"], cwd=repo)
    _git(["config", "user.name", "Test"], cwd=repo)

    (repo / "scripts").mkdir()
    (repo / "scripts" / "validate.py").write_text(
        "import sys\nsys.exit(0)\n", encoding="utf-8",
    )
    (repo / "results").mkdir()
    (repo / "results" / ".gitkeep").write_text("", encoding="utf-8")
    _git(["add", "-A"], cwd=repo)
    _git(["commit", "-m", "init"], cwd=repo)
    _git(["push", "-u", "origin", "main"], cwd=repo)
    return repo


def test_suggest_filename_uses_cpu_and_model():
    name = suggest_filename(SAMPLE_REPORT)
    assert name.startswith("amd-ryzen-ai-9-hx-470")
    assert "llm-llama-3-2-1b-instruct" in name
    assert name.endswith(".json")


def test_submit_report_success(amdaibenchmarks_repo):
    result = submit_report(SAMPLE_REPORT, amdaibenchmarks_repo)
    assert "Pushed" in result
    assert "compare/main...submit/" in result

    # the branch should exist on the bare remote, not on main
    branches = subprocess.run(
        ["git", "branch", "-r"], cwd=amdaibenchmarks_repo, capture_output=True, text=True,
    ).stdout
    assert "origin/submit/" in branches

    on_main = subprocess.run(
        ["git", "show", "main:results"], cwd=amdaibenchmarks_repo, capture_output=True, text=True,
    ).stdout
    assert ".json" not in on_main  # not merged into main by submit_report


def test_submit_report_missing_repo(tmp_path):
    with pytest.raises(SubmitError, match="doesn't exist"):
        submit_report(SAMPLE_REPORT, tmp_path / "nope")


def test_submit_report_not_amdaibenchmarks_checkout(tmp_path):
    (tmp_path / "somewhere").mkdir()
    with pytest.raises(SubmitError, match="doesn't look like an amdaibenchmarks checkout"):
        submit_report(SAMPLE_REPORT, tmp_path / "somewhere")


def test_submit_report_rejects_dirty_working_tree(amdaibenchmarks_repo):
    (amdaibenchmarks_repo / "results" / "uncommitted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SubmitError, match="uncommitted changes"):
        submit_report(SAMPLE_REPORT, amdaibenchmarks_repo)


def test_submit_report_cleans_up_on_validation_failure(amdaibenchmarks_repo):
    (amdaibenchmarks_repo / "scripts" / "validate.py").write_text(
        "import sys\nprint('nope')\nsys.exit(1)\n", encoding="utf-8",
    )
    _git(["add", "-A"], cwd=amdaibenchmarks_repo)
    _git(["commit", "-m", "break validation"], cwd=amdaibenchmarks_repo)
    _git(["push"], cwd=amdaibenchmarks_repo)

    with pytest.raises(SubmitError, match="failed amdaibenchmarks' validation"):
        submit_report(SAMPLE_REPORT, amdaibenchmarks_repo)

    # should be back on main with the submit branch removed, not left mid-branch
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=amdaibenchmarks_repo, capture_output=True, text=True,
    ).stdout.strip()
    assert branch == "main"
    branches = subprocess.run(["git", "branch"], cwd=amdaibenchmarks_repo, capture_output=True, text=True).stdout
    assert "submit/" not in branches
