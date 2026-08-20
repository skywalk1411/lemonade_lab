"""Submits a finished report to the amdaibenchmarks community leaderboard as a
pushed, PR-ready branch — instead of copying the file and running git by hand.

Deliberately does not push straight to main, even for the repo owner: the
whole point of that repo's design is that every result is reviewable in a PR
before it counts. This automates everything up to that point and hands back
the compare URL.

Reuses amdaibenchmarks' own scripts/validate.py rather than re-implementing
its schema rules here, so the two repos can't drift out of sync.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REMOTE_URL = "https://github.com/skywalk1411/amdaibenchmarks.git"


class SubmitError(Exception):
    pass


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")


def _run(args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def default_repo_path() -> Path:
    """Guesses a sibling checkout: .../amdaibenchmarks next to .../lemonade_lab."""
    return Path(__file__).resolve().parent.parent.parent / "amdaibenchmarks"


def resolve_github_username(configured: str | None = None) -> str | None:
    """Figures out who's submitting, so the leaderboard can credit them.

    Preference order: an explicitly configured username (local_config.json's
    github_username, or --github-username), then the account the `gh` CLI is
    logged into (if installed and authenticated). Returns None rather than
    guessing from git author name/email, which is frequently not a GitHub
    handle at all.
    """
    if configured:
        return configured.strip().lstrip("@") or None

    try:
        result = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    login = result.stdout.strip()
    return login or None


def _results_view(report: dict) -> dict:
    if report.get("results"):
        return report["results"]
    if report.get("models"):
        return {"llm": report["models"]}
    return {}


def suggest_filename(report: dict) -> str:
    system = report.get("system", {})
    cpu_slug = _slugify(system.get("cpu", "unknown-cpu"))[:40]
    results = _results_view(report)
    workloads = sorted(results.keys())
    models = sorted({m for backends in results.values() for m in backends})

    if len(models) == 1:
        what = _slugify(f"{workloads[0]}-{models[0]}")[:40]
    else:
        what = _slugify("-".join(workloads))[:40] or "results"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{cpu_slug}-{what}-{stamp}.json"


def _compare_url(remote: str, branch: str) -> str:
    m = re.search(r"github\.com[:/]+(?P<owner_repo>[^/]+/[^/]+?)(?:\.git)?$", remote.strip())
    owner_repo = m.group("owner_repo") if m else "skywalk1411/amdaibenchmarks"
    return f"https://github.com/{owner_repo}/compare/main...{branch}?expand=1"


def submit_report(json_report: dict, repo_path: Path, github_username: str | None = None) -> str:
    """Validates, commits, and pushes json_report as a new branch in an
    amdaibenchmarks checkout. Returns a human-readable result message;
    raises SubmitError with an actionable explanation on any failure.

    If github_username is given, it's stamped onto the report as
    'submitted_by' so the leaderboard can credit the contributor. The PR
    itself is still what GitHub uses to attribute authorship — this field is
    just what the static site (which has no PR/author info of its own) can
    read out of the merged JSON.
    """
    if github_username:
        json_report = {**json_report, "submitted_by": github_username}
    if not repo_path.exists():
        raise SubmitError(
            f"{repo_path} doesn't exist. Clone it first:\n"
            f"  git clone {DEFAULT_REMOTE_URL} \"{repo_path}\"\n"
            f"or pass --submit-repo <path> to point at an existing checkout."
        )
    if not (repo_path / "scripts" / "validate.py").exists():
        raise SubmitError(f"{repo_path} doesn't look like an amdaibenchmarks checkout (no scripts/validate.py).")

    status = _run(["git", "status", "--porcelain"], cwd=repo_path)
    if status.returncode != 0:
        raise SubmitError(f"{repo_path} isn't a git repository: {status.stderr.strip()}")
    if status.stdout.strip():
        raise SubmitError(
            f"{repo_path} has uncommitted changes — commit, stash, or clean it first so this "
            f"submission doesn't get bundled with unrelated work."
        )

    filename = suggest_filename(json_report)
    branch = f"submit/{filename.rsplit('.', 1)[0]}"
    dest = repo_path / "results" / filename

    _run(["git", "checkout", "main"], cwd=repo_path)
    pull = _run(["git", "pull", "--ff-only"], cwd=repo_path)
    if pull.returncode != 0:
        raise SubmitError(f"couldn't fast-forward {repo_path}'s main branch — resolve that first: {pull.stderr.strip()}")

    checkout = _run(["git", "checkout", "-b", branch], cwd=repo_path)
    if checkout.returncode != 0:
        raise SubmitError(f"couldn't create branch {branch} (does it already exist?): {checkout.stderr.strip()}")

    dest.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    validate = subprocess.run(
        [sys.executable, "scripts/validate.py", str(dest.relative_to(repo_path))],
        cwd=repo_path, capture_output=True, text=True,
    )
    if validate.returncode != 0:
        dest.unlink(missing_ok=True)
        _run(["git", "checkout", "main"], cwd=repo_path)
        _run(["git", "branch", "-D", branch], cwd=repo_path)
        raise SubmitError(f"report failed amdaibenchmarks' validation, not submitting:\n{validate.stdout}{validate.stderr}")

    _run(["git", "add", str(dest.relative_to(repo_path))], cwd=repo_path)
    commit = _run(["git", "commit", "-m", f"Add benchmark result: {filename}"], cwd=repo_path)
    if commit.returncode != 0:
        raise SubmitError(f"commit failed: {commit.stderr.strip()}")

    push = _run(["git", "push", "-u", "origin", branch], cwd=repo_path)
    if push.returncode != 0:
        raise SubmitError(
            f"branch {branch} was committed locally but the push failed (no write/fork access?): "
            f"{push.stderr.strip()}\nPush it yourself from {repo_path}, or open a PR manually with this file."
        )

    remote = _run(["git", "remote", "get-url", "origin"], cwd=repo_path).stdout
    return f"Pushed {filename} on branch {branch}.\nOpen a PR: {_compare_url(remote, branch)}"
