"""Local leaderboard server: upload benchmark reports, browse and compare them.

This is a working instance of the "AMD Local AI Performance Database" concept
from the project brief — running on localhost. Nothing about it is tied to
being local; pointing it at a hosted Postgres/SQLite-on-a-server and deploying
the same FastAPI app is the whole path to a real shared leaderboard.

Run with:
    uvicorn api.server:app --reload --port 8787
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Lemonade Lab Leaderboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportUpload(BaseModel):
    label: str | None = None
    report: dict


@app.post("/api/reports")
def upload_report(payload: ReportUpload):
    report = payload.report
    if "system" not in report or ("results" not in report and "models" not in report):
        raise HTTPException(400, "report is missing 'system' and 'results'/'models'")
    report_id = db.insert_report(report, label=payload.label)
    return {"id": report_id}


@app.get("/api/reports")
def get_reports():
    return db.list_reports()


@app.get("/api/reports/{report_id}")
def get_report(report_id: int):
    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(404, "report not found")
    return report


@app.delete("/api/reports/{report_id}")
def remove_report(report_id: int):
    if not db.delete_report(report_id):
        raise HTTPException(404, "report not found")
    return {"deleted": report_id}


def _results_view(report: dict) -> dict:
    """Normalizes either schema (results-by-workload, or the flat back-compat
    'models' alias) into { workload: { model: { backend: entry } } }.
    """
    if "results" in report:
        return report["results"]
    if "models" in report:
        return {"llm": report["models"]}
    return {}


@app.get("/api/catalog")
def get_catalog():
    """Distinct (workload, model) pairs across all stored reports, so the UI
    can populate a comparison picker without pulling every report's full body.
    """
    seen = {}
    for _id, report in db.all_reports_full():
        for workload, models in _results_view(report).items():
            for model_name in models:
                seen.setdefault(workload, set()).add(model_name)
    return {w: sorted(models) for w, models in seen.items()}


@app.get("/api/leaderboard")
def get_leaderboard(workload: str = "llm", model: str | None = None):
    """Flattened rows across every stored report for a given workload
    (optionally filtered to one model), sorted fastest-first — the actual
    cross-machine comparison table.
    """
    rows = []
    for report_id, report in db.all_reports_full():
        system = report.get("system", {})
        models = _results_view(report).get(workload, {})
        for model_name, backends in models.items():
            if model and model_name != model:
                continue
            for backend, entry in backends.items():
                if entry.get("error"):
                    continue
                value = entry.get("value")
                if value is None:
                    continue
                rows.append({
                    "report_id": report_id,
                    "model": model_name,
                    "backend": backend,
                    "value": value,
                    "unit": entry.get("unit", "tok/s"),
                    "cpu": system.get("cpu"),
                    "gpu": system.get("gpu"),
                    "npu": system.get("npu"),
                    "memory": system.get("memory"),
                    "timestamp": report.get("timestamp"),
                })
    rows.sort(key=lambda r: r["value"], reverse=True)
    return rows


# Static dashboard last, so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
