"""Loads machine-specific settings: where lemonade.exe lives and which
host/port Lemonade Server is running on.

Looks for local_config.json next to this package (gitignored, machine-specific).
Falls back to config.example.json so the tool at least imports on a fresh checkout.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    lemonade_exe: Path
    lemonade_host: str = "127.0.0.1"
    lemonade_port: int = 1234
    github_username: str | None = None


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(p))


def load_config() -> Config:
    path = ROOT / "local_config.json"
    if not path.exists():
        path = ROOT / "config.example.json"
    data = json.loads(path.read_text())
    return Config(
        lemonade_exe=_expand(data["lemonade_exe"]),
        lemonade_host=data.get("lemonade_host", "127.0.0.1"),
        lemonade_port=int(data.get("lemonade_port", 1234)),
        github_username=data.get("github_username") or None,
    )
