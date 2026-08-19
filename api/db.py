"""SQLite storage for uploaded benchmark reports. Deliberately simple: the
report JSON is stored as-is and re-parsed on read, rather than modeled across
normalized tables — the dataset this tool produces is small (one row per
benchmark run) and the schema (bench/report.py) is still evolving.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "leaderboard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    report_timestamp TEXT,
    cpu TEXT,
    gpu TEXT,
    npu TEXT,
    memory TEXT,
    os TEXT,
    label TEXT,
    data TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def insert_report(report: dict, label: str | None = None) -> int:
    system = report.get("system", {})
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO reports (report_timestamp, cpu, gpu, npu, memory, os, label, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.get("timestamp"),
                system.get("cpu"),
                system.get("gpu"),
                system.get("npu"),
                system.get("memory"),
                system.get("os"),
                label,
                json.dumps(report),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_reports() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, uploaded_at, report_timestamp, cpu, gpu, npu, memory, os, label FROM reports ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_report(report_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT data FROM reports WHERE id = ?", (report_id,)).fetchone()
        return json.loads(row["data"]) if row else None
    finally:
        conn.close()


def delete_report(report_id: int) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def all_reports_full() -> list[tuple[int, dict]]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT id, data FROM reports ORDER BY id DESC").fetchall()
        return [(r["id"], json.loads(r["data"])) for r in rows]
    finally:
        conn.close()
