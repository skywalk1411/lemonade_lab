import pytest
from fastapi.testclient import TestClient

from api import db as db_module
from api.server import app

SAMPLE_REPORT = {
    "timestamp": "2026-08-19T00:00:00Z",
    "system": {"cpu": "AMD Ryzen AI 9 HX 470", "gpu": "AMD Radeon 890M", "npu": "AMD XDNA", "memory": "64GB", "os": "Windows 11"},
    "results": {
        "llm": {
            "Qwen": {
                "rocm": {"value": 21.7, "unit": "tok/s", "ttft_ms": 15.0, "memory_gb": 11.1},
                "cpu": {"error": "timed out"},
            }
        }
    },
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_leaderboard.db")
    return TestClient(app)


def test_upload_and_list(client):
    resp = client.post("/api/reports", json={"report": SAMPLE_REPORT, "label": "test-run"})
    assert resp.status_code == 200
    report_id = resp.json()["id"]

    resp = client.get("/api/reports")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["id"] == report_id
    assert rows[0]["label"] == "test-run"
    assert rows[0]["cpu"] == "AMD Ryzen AI 9 HX 470"


def test_upload_rejects_malformed_report(client):
    resp = client.post("/api/reports", json={"report": {"nope": "no system or results"}})
    assert resp.status_code == 400


def test_get_report_roundtrip(client):
    report_id = client.post("/api/reports", json={"report": SAMPLE_REPORT}).json()["id"]
    resp = client.get(f"/api/reports/{report_id}")
    assert resp.status_code == 200
    assert resp.json()["system"]["cpu"] == "AMD Ryzen AI 9 HX 470"


def test_get_report_404(client):
    resp = client.get("/api/reports/9999")
    assert resp.status_code == 404


def test_catalog_reflects_uploaded_workloads(client):
    client.post("/api/reports", json={"report": SAMPLE_REPORT})
    resp = client.get("/api/catalog")
    assert resp.json() == {"llm": ["Qwen"]}


def test_leaderboard_excludes_failed_backends_and_sorts_desc(client):
    client.post("/api/reports", json={"report": SAMPLE_REPORT})
    resp = client.get("/api/leaderboard", params={"workload": "llm"})
    rows = resp.json()
    assert len(rows) == 1  # cpu entry had an error and is excluded
    assert rows[0]["backend"] == "rocm"
    assert rows[0]["value"] == 21.7
    assert rows[0]["unit"] == "tok/s"


def test_leaderboard_filters_by_model(client):
    client.post("/api/reports", json={"report": SAMPLE_REPORT})
    resp = client.get("/api/leaderboard", params={"workload": "llm", "model": "does-not-exist"})
    assert resp.json() == []


def test_delete_report(client):
    report_id = client.post("/api/reports", json={"report": SAMPLE_REPORT}).json()["id"]
    resp = client.delete(f"/api/reports/{report_id}")
    assert resp.status_code == 200
    assert client.get(f"/api/reports/{report_id}").status_code == 404
