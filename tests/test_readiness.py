from __future__ import annotations

from nlp.readiness import assess


def test_empty_note_is_not_ready():
    ready, reason = assess("   ")
    assert not ready
    assert reason


def test_url_is_not_ready():
    ready, reason = assess("https://example.com/some/page")
    assert not ready
    assert "link" in (reason or "").lower()


def test_short_note_is_not_ready():
    ready, reason = assess("knee")
    assert not ready


def test_terse_clinical_note_is_ready():
    ready, reason = assess("R knee pain 6/10.")
    assert ready
    assert reason is None


def test_non_clinical_text_is_not_ready():
    ready, reason = assess("The quarterly report is due on Friday afternoon.")
    assert not ready
    assert "physiotherapy" in (reason or "").lower()
