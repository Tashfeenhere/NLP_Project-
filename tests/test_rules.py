from __future__ import annotations


def fields(result):
    return result["fields"]


def test_body_part_laterality_and_pain(pipeline):
    result = pipeline.analyze("Right knee pain 6/10.")
    assert result["ready"]
    assert fields(result)["body_part"] == "knee"
    assert fields(result)["laterality"] == "right"
    assert fields(result)["pain_severity"]["value"] == 6


def test_plural_body_part(pipeline):
    result = pipeline.analyze("Both knees are painful, 5/10.")
    assert fields(result)["body_part"] == "knee"
    assert fields(result)["laterality"] == "bilateral"


def test_trigger_and_relieving_are_cue_governed(pipeline):
    result = pipeline.analyze(
        "Right knee pain 5/10. Worse with climbing stairs. Rest helps."
    )
    assert "climbing stairs" in fields(result)["triggers"]
    assert "rest" in fields(result)["relieving_factors"]


def test_postposed_trigger_and_after_cue(pipeline):
    result = pipeline.analyze(
        "Prolonged sitting makes it worse. Left ankle pain after running."
    )
    triggers = fields(result)["triggers"]
    assert "prolonged sitting" in triggers
    assert "running" in triggers


def test_exercise_plural_and_alias(pipeline):
    result = pipeline.analyze(
        "Knee pain 4/10. Prescribed quadriceps sets, clamshells and SLR."
    )
    exercises = fields(result)["exercises"]
    assert "quadriceps set" in exercises
    assert "clamshell" in exercises or "clamshells" in exercises
    assert "straight leg raise" in exercises


def test_qualitative_rom(pipeline):
    result = pipeline.analyze("Lumbar flexion is reduced.")
    rom = fields(result)["rom"]
    assert rom and rom[0]["movement"] == "flexion"
    assert rom[0]["value"] is None
    assert rom[0]["quality"] == "reduced"


def test_forward_rom_and_joint_binding(pipeline):
    result = pipeline.analyze("Right knee flexion is limited to 95 degrees.")
    rom = fields(result)["rom"]
    assert rom == [{"movement": "flexion", "value": 95, "unit": "deg", "quality": None}]
    assert fields(result)["body_part"] == "knee"
    assert fields(result)["laterality"] == "right"


def test_reversed_rom(pipeline):
    result = pipeline.analyze("ROM: 90 deg internal rotation, shoulder.")
    rom = fields(result)["rom"]
    assert rom and rom[0]["movement"] == "internal rotation"
    assert rom[0]["value"] == 90


def test_entities_do_not_overlap_and_offsets_slice_text(pipeline):
    text = (
        "Right knee pain 6/10, worse with climbing stairs. "
        "Knee flexion is limited to 95 degrees. Quadriceps weakness."
    )
    result = pipeline.analyze(text)
    spans = sorted(
        [(e["start"], e["end"]) for e in result["entities"] + result["negated_entities"]]
    )
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end, "entities overlap"
    for entity in result["entities"] + result["negated_entities"]:
        assert text[entity["start"]: entity["end"]] == entity["text"]
        assert entity["label"] != "LATERALITY"
