from __future__ import annotations


def test_denies_removes_the_positive_body_part(pipeline):
    result = pipeline.analyze("Denies neck pain. No swelling.")
    assert result["ready"]
    assert result["fields"]["body_part"] is None
    labels = {e["label"] for e in result["entities"]}
    assert "BODY_PART" not in labels
    negated = {e["text"].lower() for e in result["negated_entities"]}
    assert "neck" in negated
    assert "swelling" in negated


def test_negation_stops_at_contrast(pipeline):
    result = pipeline.analyze("No swelling but knee pain 7/10.")
    assert result["fields"]["pain_severity"]["value"] == 7
    assert result["fields"]["body_part"] == "knee"
    negated = [e["text"].lower() for e in result["negated_entities"]]
    assert "swelling" in negated
    assert "knee" not in negated


def test_negation_does_not_cross_sentence_boundary(pipeline):
    result = pipeline.analyze("No swelling. Right knee pain 5/10.")
    assert result["fields"]["body_part"] == "knee"
    assert result["fields"]["pain_severity"]["value"] == 5
    assert [e["text"].lower() for e in result["negated_entities"]] == ["swelling"]


def test_postposed_negation(pipeline):
    result = pipeline.analyze("Knee pain without stiffness, 4/10.")
    assert result["fields"]["pain_severity"]["value"] == 4
    assert "stiffness" in [e["text"].lower() for e in result["negated_entities"]]


def test_comma_separated_negated_list_keeps_scope(pipeline):
    result = pipeline.analyze("No swelling, warmth or redness.")
    negated = {e["text"].lower() for e in result["negated_entities"]}
    assert {"swelling", "warmth", "redness"} <= negated
    assert result["fields"]["findings"] == []


def test_measurement_after_comma_is_not_negated(pipeline):
    result = pipeline.analyze("No swelling, pain rated 7/10.")
    assert result["fields"]["pain_severity"]["value"] == 7
    assert "swelling" in [e["text"].lower() for e in result["negated_entities"]]


def test_semicolon_ends_negation_scope(pipeline):
    result = pipeline.analyze("No swelling; right knee pain 5/10.")
    assert result["fields"]["pain_severity"]["value"] == 5
    assert result["fields"]["body_part"] == "knee"
    assert [e["text"].lower() for e in result["negated_entities"]] == ["swelling"]
