"""Acceptance fixtures for the three representative notes.

The notes match the sample buttons in the UI and the notes printed by
``scripts/demo.py``.  For each one we assert the complete structured record and
the invariants the API promises: every entity offset slices back to its own
surface text, and no two entities overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "acceptance_notes.json"
CASES = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_acceptance_note(pipeline, case):
    text = case["text"]
    result = pipeline.analyze(text)

    assert result["ready"] is True
    assert result["fields"] == case["fields"], case["name"]

    entities = result["entities"]
    assert entities, f"{case['name']} produced no entities"

    for entity in entities:
        start, end = entity["start"], entity["end"]
        assert 0 <= start < end <= len(text)
        assert text[start:end] == entity["text"]
        # Rule layers contribute measurement labels alongside the six NER
        # labels; LATERALITY is consumed into fields rather than emitted.
        assert entity["label"] in {
            "BODY_PART",
            "MUSCLE",
            "FINDING",
            "EXERCISE",
            "TRIGGER",
            "RELIEVING_FACTOR",
            "PAIN_SEVERITY",
            "ROM",
            "DURATION",
            "FREQUENCY",
        }

    spans = sorted((e["start"], e["end"]) for e in entities)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= prev_end, f"overlapping entities in {case['name']}"


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_acceptance_note_is_deterministic(pipeline, case):
    first = pipeline.analyze(case["text"])
    second = pipeline.analyze(case["text"])
    assert first["fields"] == second["fields"]
    assert first["entities"] == second["entities"]
