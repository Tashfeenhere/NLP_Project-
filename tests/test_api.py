"""API and SQLite round-trip against a temporary database."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

NOTE = (
    "Right knee pain 6/10 for three weeks, worse with climbing stairs. "
    "Knee flexion is limited to 95 degrees. Quadriceps weakness. "
    "Prescribed straight leg raises."
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CNA_DB_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        yield test_client


def analyze(client, text=NOTE):
    response = client.post("/api/analyze", json={"text": text})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "BODY_PART" in body["labels"]
    assert body["record_count"] == 0


def test_analyze_returns_fields_and_spans(client):
    body = analyze(client)
    assert body["ready"] is True
    fields = body["fields"]
    assert fields["body_part"] == "knee"
    assert fields["laterality"] == "right"
    assert fields["pain_severity"]["value"] == 6
    assert fields["rom"][0]["value"] == 95
    for entity in body["entities"]:
        assert NOTE[entity["start"]: entity["end"]] == entity["text"]
    sources = {entity["source"] for entity in body["entities"]}
    assert "rule" in sources or "gazetteer" in sources


def test_blank_note_is_rejected(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 422
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422


def test_not_ready_note_returns_empty_fields(client):
    body = analyze(client, "https://example.com/not-a-note")
    assert body["ready"] is False
    assert body["warnings"]
    assert body["entities"] == []


def test_record_crud_and_export(client):
    analysis = analyze(client)
    payload = {
        "patient_name": "Test Patient",
        "session_date": "2026-10-18",
        "text": NOTE,
        "fields": analysis["fields"],
        "entities": analysis["entities"],
        "negated_entities": analysis["negated_entities"],
        "model_version": analysis["model_version"],
    }

    created = client.post("/api/records", json=payload)
    assert created.status_code == 201, created.text
    record_id = created.json()["id"]
    assert created.json()["patient_id"] is not None

    listing = client.get("/api/records").json()
    assert len(listing) == 1
    assert listing[0]["id"] == record_id
    assert listing[0]["patient_name"] == "Test Patient"
    assert listing[0]["body_part"] == "knee"
    assert listing[0]["pain_severity"] == 6
    assert len(listing[0]["preview"]) <= 161

    fetched = client.get(f"/api/records/{record_id}")
    assert fetched.status_code == 200
    stored = fetched.json()
    assert stored["text"] == NOTE
    assert stored["fields"] == analysis["fields"]
    assert stored["entities"] == analysis["entities"]

    export = client.get("/api/records/export.csv")
    assert export.status_code == 200
    assert "text/csv" in export.headers["content-type"]
    assert "Test Patient" in export.text
    assert export.text.splitlines()[0].startswith("id,")

    deleted = client.delete(f"/api/records/{record_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/records/{record_id}").status_code == 404
    assert client.get("/api/records").json() == []


def test_unknown_patient_id_is_rejected(client):
    analysis = analyze(client)
    payload = {
        "patient_id": 9999,
        "text": NOTE,
        "fields": analysis["fields"],
        "entities": [],
        "negated_entities": [],
    }
    response = client.post("/api/records", json=payload)
    assert response.status_code == 422


def test_missing_record_is_404(client):
    assert client.get("/api/records/123456").status_code == 404
    assert client.delete("/api/records/123456").status_code == 404
