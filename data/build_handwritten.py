"""Build the hand-written held-out test set with exact character offsets.

The notes live here as marked-up strings because writing raw JSON offsets by
hand is error-prone.  Markup:

    {{LABEL::text}}                                  simple entity
    {{LABEL|key=value|...::text}}                    entity with metadata
    {{FINDING|negated=1::swelling}}                  a negated mention

``ROM`` spans are whole phrases (the plan's entity model): the joint and the
movement are carried as metadata rather than tagged separately inside the span.

Run:

    python data/build_handwritten.py

Writes ``data/hand_written_test.jsonl``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hand_written_test.jsonl"

_MARKER = re.compile(r"\{\{(?P<header>[^{}:]+?):{1,2}(?P<text>.+?)\}\}", re.DOTALL)


NOTES = [
    # ---------------------------------------------------------------- knee
    "{{LATERALITY:Right}} {{BODY_PART:knee}} pain rated {{PAIN_SEVERITY:6 out of 10}}, "
    "{{FREQUENCY:constant}} for {{DURATION:three weeks}}. Worse with {{TRIGGER:climbing stairs}}. "
    "{{RELIEVING_FACTOR:Rest}} helps. {{ROM|movement=flexion|body_part=knee|value=95|unit=deg::"
    "Knee flexion is limited to 95 degrees}}. {{MUSCLE:Quadriceps}} {{FINDING:weakness}}. "
    "Prescribed {{EXERCISE:straight leg raises}} and {{EXERCISE:clamshells}}.",

    "{{LATERALITY:L}} {{BODY_PART:knee}} pain {{PAIN_SEVERITY:7/10}}. "
    "ROM: {{ROM|movement=flexion|value=80|unit=deg::flexion 80 degrees}}. "
    "Aggravated by {{TRIGGER:squatting}}. Rx {{EXERCISE:wall squats}}.",

    "{{BODY_PART:Knee}} {{ROM|movement=flexion|body_part=knee|quality=reduced::"
    "Knee flexion reduced}}. Worse with {{TRIGGER:running}}. "
    "{{MUSCLE:Quadriceps}} {{FINDING:tightness}}. {{RELIEVING_FACTOR:Ice}} helps.",

    "{{LATERALITY:Right}} {{BODY_PART:knee}} pain {{PAIN_SEVERITY:5/10}}. "
    "No {{FINDING|negated=1:swelling}} or {{FINDING|negated=1:locking}}. "
    "{{ROM|movement=flexion|body_part=knee|quality=full::Full right knee flexion}}.",

    "{{BODY_PART:Lower back}} pain {{PAIN_SEVERITY:4-5/10}}, {{FREQUENCY:intermittent}} for "
    "{{DURATION:two months}}. Aggravated by {{TRIGGER:prolonged sitting}}. "
    "Better with {{RELIEVING_FACTOR:walking}}. "
    "{{ROM|movement=flexion|body_part=lower back|quality=reduced::Lumbar flexion is reduced}}.",

    "{{BODY_PART:low back}} {{FINDING:stiffness}}, worse with {{TRIGGER:lifting}}. "
    "Prescribed {{EXERCISE:bird dog}}, {{EXERCISE:dead bug}} and {{EXERCISE:glute bridge}}. "
    "{{RELIEVING_FACTOR:Heat}} helps.",

    "{{LATERALITY:Right}} {{BODY_PART:knee}} pain {{PAIN_SEVERITY:7 out of 10}}, {{FREQUENCY:daily}} "
    "for {{DURATION:six weeks}}. Worse with {{TRIGGER:climbing stairs}}. "
    "Relieved by {{RELIEVING_FACTOR:rest}}. "
    "{{ROM|movement=flexion|body_part=knee|value=85|unit=deg::"
    "Right knee flexion is limited to 85 degrees}}. No {{FINDING|negated=1:effusion}}. "
    "{{MUSCLE:Quadriceps}} {{FINDING:weakness}}. Prescribed {{EXERCISE:straight leg raises}}, "
    "{{EXERCISE:clamshells}} and {{EXERCISE:wall squats}}.",

    "{{LATERALITY:R}} {{BODY_PART:knee}} pain {{PAIN_SEVERITY:6/10}}. "
    "{{MUSCLE:Quads}} {{FINDING:weak}}. Rx {{EXERCISE:SLR}} and {{EXERCISE:quad set}}.",

    "{{BODY_PART:knee}} {{ROM|movement=flexion|body_part=knee|value=95|unit=deg::"
    "knee flexion measures 95 degrees}}. {{MUSCLE:Quadriceps}} {{FINDING:weakness}}. "
    "Prescribed {{EXERCISE:quadriceps sets}}.",

    "{{LATERALITY:Right}} {{BODY_PART:knee}} pain {{PAIN_SEVERITY:4 to 5 out of 10}}. "
    "Worse with {{TRIGGER:running}}. {{RELIEVING_FACTOR:Rest}} helps. "
    "{{ROM|movement=extension|body_part=knee|value=0|unit=deg::"
    "Right knee extension is limited to 0 degrees}}.",

    "{{LATERALITY:Bilateral}} {{BODY_PART:knees}} with {{FINDING:crepitus}} and "
    "{{FINDING:instability}}. "
    "{{ROM|movement=extension|body_part=knee|value=10|unit=deg::"
    "Bilateral knee extension is limited to 10 degrees}}. Worse with {{TRIGGER:going down stairs}}.",

    "{{LATERALITY:Bilateral}} {{BODY_PART:knees}} pain {{PAIN_SEVERITY:5/10}}. "
    "No {{FINDING|negated=1:swelling}}. "
    "{{ROM|movement=flexion|body_part=knee|value=100|unit=deg::"
    "Bilateral knee flexion is limited to 100 degrees}}. Advised {{EXERCISE:heel slides}}.",

    "{{BODY_PART:Knee}} pain {{PAIN_SEVERITY:6/10}}. ROM: "
    "{{ROM|movement=flexion|value=90|unit=deg::90 degrees flexion}}, {{BODY_PART:knee}}. "
    "Aggravated by {{TRIGGER:kneeling}}.",

    # ---------------------------------------------------------- lower back
    "{{BODY_PART:lower back}} pain {{PAIN_SEVERITY:5/10}} since {{DURATION:since March}}. "
    "Worse with {{TRIGGER:bending forward}}. {{RELIEVING_FACTOR:Change of position}} helps.",

    "{{BODY_PART:lumbar region}} pain {{PAIN_SEVERITY:6/10}}, {{FREQUENCY:intermittent}}. "
    "Worse with {{TRIGGER:driving}}. {{RELIEVING_FACTOR:Walking}} helps. "
    "{{ROM|movement=side flexion|body_part=lower back|value=20|unit=deg::"
    "Lumbar side flexion is limited to 20 degrees}}. Advised {{EXERCISE:cat camel}} and "
    "{{EXERCISE:pelvic tilt}}.",

    "{{BODY_PART:lumbar spine}} pain {{PAIN_SEVERITY:3/10}}. "
    "{{ROM|movement=flexion|body_part=lower back|quality=reduced::Reduced lumbar flexion}}. "
    "Better with {{RELIEVING_FACTOR:rest}}.",

    # ------------------------------------------------------------ shoulder
    "{{LATERALITY:Left}} {{BODY_PART:shoulder}} pain {{PAIN_SEVERITY:8/10}}. "
    "Worse with {{TRIGGER:overhead reaching}}. {{RELIEVING_FACTOR:Heat}} helps. "
    "{{ROM|movement=external rotation|body_part=shoulder|value=40|unit=deg::"
    "Left shoulder external rotation is limited to 40 degrees}}. "
    "{{MUSCLE:Rotator cuff muscles}} {{FINDING:weakness}}.",

    "{{BODY_PART:shoulder}} {{FINDING:tightness}}. "
    "{{ROM|movement=flexion|body_part=shoulder|quality=full::Full shoulder flexion}}. "
    "Home programme: {{EXERCISE:wall slides}}, {{EXERCISE:band pull aparts}} and "
    "{{EXERCISE:scapular setting}}.",

    "{{BODY_PART:shoulder}} pain {{PAIN_SEVERITY:8/10}}. Worse with {{TRIGGER:throwing}}. "
    "{{ROM|movement=abduction|body_part=shoulder|value=90|unit=deg::"
    "Shoulder abduction is limited to 90 degrees}}. {{MUSCLE:Rotator cuff muscles}} "
    "{{FINDING:weakness}}. Advised {{EXERCISE:external rotation exercise}}.",

    "{{BODY_PART:shoulder}} pain {{PAIN_SEVERITY:6/10}}, {{FREQUENCY:twice a day}}. "
    "Worse with {{TRIGGER:sleeping on side}}. {{RELIEVING_FACTOR:Positioning}} helps.",

    "Patient reports {{LATERALITY:left}} {{BODY_PART:shoulder}} {{FINDING:discomfort}}, "
    "{{PAIN_SEVERITY:5 / 10}}. Appears worse with {{TRIGGER:computer work}}. "
    "{{RELIEVING_FACTOR:Massage}} helps. "
    "{{ROM|movement=flexion|body_part=shoulder|value=110|unit=deg::"
    "Left shoulder flexion is limited to 110 degrees}}.",

    "{{BODY_PART:shoulder}} {{FINDING:tightness}}. Prescribed {{EXERCISE:wall slides}} and "
    "{{EXERCISE:band pull aparts}}.",

    "{{BODY_PART:shoulder}} pain {{PAIN_SEVERITY:7 to 8 out of 10}}. "
    "Worse with {{TRIGGER:overhead reaching}}. "
    "{{ROM|movement=external rotation|body_part=shoulder|value=45|unit=deg::"
    "Shoulder external rotation is limited to 45 degrees}}.",

    # --------------------------------------------------------------- neck
    "{{LATERALITY:Bilateral}} {{BODY_PART:neck}} pain {{PAIN_SEVERITY:6/10}}, "
    "{{FREQUENCY:daily}}. Worse with {{TRIGGER:looking down}}. "
    "{{ROM|movement=rotation|body_part=neck|value=60|unit=deg::"
    "Cervical rotation is limited to 60 degrees}}. {{FINDING:Spasm}} in the "
    "{{MUSCLE:upper trapezius}}.",

    "{{BODY_PART:neck}} {{FINDING:stiffness}} for {{DURATION:six weeks}}. "
    "{{RELIEVING_FACTOR:Massage}} helps. Advised {{EXERCISE:chin tucks}} and "
    "{{EXERCISE:cervical retractions}}.",

    "{{BODY_PART:neck}} {{FINDING:stiffness}}, {{FREQUENCY:intermittent}}. "
    "Worse with {{TRIGGER:turning head}}. {{RELIEVING_FACTOR:Heat}} helps. "
    "{{ROM|movement=extension|body_part=neck|quality=reduced::Cervical extension is reduced}}.",

    "{{LATERALITY:Bilateral}} {{BODY_PART:neck}} pain {{PAIN_SEVERITY:5/10}}, "
    "{{FREQUENCY:constant}} for {{DURATION:two weeks}}. Worse with {{TRIGGER:computer work}}. "
    "{{ROM|movement=side flexion|body_part=neck|value=25|unit=deg::"
    "Neck side flexion is limited to 25 degrees}}. {{MUSCLE:Deep neck flexors}} "
    "{{FINDING:weakness}}.",

    # ---------------------------------------------------------------- hip
    "{{LATERALITY:Right}} {{BODY_PART:hip}} pain {{PAIN_SEVERITY:7/10}}. "
    "Worse with {{TRIGGER:walking}}. "
    "{{ROM|movement=internal rotation|body_part=hip|value=30|unit=deg::"
    "Right hip internal rotation is limited to 30 degrees}}. {{MUSCLE:Gluteus medius}} "
    "{{FINDING:weakness}}.",

    "{{BODY_PART:hip}} {{FINDING:tightness}}. Prescribed {{EXERCISE:side lying abduction}}, "
    "{{EXERCISE:clamshells}} and {{EXERCISE:hip hinge}}.",

    "{{LATERALITY:Left}} {{BODY_PART:hip}} pain {{PAIN_SEVERITY:6/10}} for "
    "{{DURATION:three months}}. Worse with {{TRIGGER:prolonged standing}}. "
    "{{ROM|movement=flexion|body_part=hip|value=100|unit=deg::"
    "Left hip flexion is limited to 100 degrees}}. {{MUSCLE:Hip flexors}} {{FINDING:tightness}}.",

    "{{BODY_PART:hip}} {{FINDING:weakness}} in the {{MUSCLE:gluteus medius}} and "
    "{{MUSCLE:gluteus maximus}}. Prescribed {{EXERCISE:side lying abduction}} and "
    "{{EXERCISE:glute bridge}}.",

    # -------------------------------------------------------------- ankle
    "{{LATERALITY:Left}} {{BODY_PART:ankle}} pain {{PAIN_SEVERITY:5/10}} for "
    "{{DURATION:one week}}. Worse with {{TRIGGER:walking on uneven ground}}. "
    "{{ROM|movement=dorsiflexion|body_part=ankle|value=10|unit=deg::"
    "Left ankle dorsiflexion is limited to 10 degrees}}. No {{FINDING|negated=1:swelling}}.",

    "{{BODY_PART:ankle}} {{FINDING:stiffness}}. {{RELIEVING_FACTOR:Elevation}} helps. "
    "Prescribed {{EXERCISE:calf raises}} and {{EXERCISE:single leg balance}}.",

    "{{LATERALITY:Left}} {{BODY_PART:ankle}} pain {{PAIN_SEVERITY:7/10}}. "
    "Worse with {{TRIGGER:hopping}}. {{RELIEVING_FACTOR:Ice}} helps. "
    "{{ROM|movement=inversion|body_part=ankle|value=15|unit=deg::"
    "Left ankle inversion is limited to 15 degrees}}.",

    "{{LATERALITY:Right}} {{BODY_PART:ankle}} pain {{PAIN_SEVERITY:4/10}}. "
    "Denies {{FINDING|negated=1:numbness}}. No {{FINDING|negated=1:redness}} or "
    "{{FINDING|negated=1:warmth}}. "
    "{{ROM|movement=plantar flexion|body_part=ankle|value=30|unit=deg::"
    "Right ankle plantar flexion is limited to 30 degrees}}.",

    "{{BODY_PART:ankle}} {{FINDING:stiffness}}. "
    "{{ROM|movement=dorsiflexion|body_part=ankle|value=5|unit=deg::"
    "Ankle dorsiflexion is limited to 5 degrees}}. Prescribed {{EXERCISE:calf stretches}} "
    "and {{EXERCISE:single leg balance}}.",

    # -------------------------------------------------------------- mixed
    "{{BODY_PART:knee}} {{FINDING:effusion}} but no {{FINDING|negated=1:locking}}. "
    "Pain {{PAIN_SEVERITY:3/10}}. Worse with {{TRIGGER:kneeling}}.",

    "Treatment: {{EXERCISE:straight leg raises}}, {{EXERCISE:clamshells}} and "
    "{{EXERCISE:wall slides}} for {{BODY_PART:knee}} {{FINDING:weakness}}.",

    "{{BODY_PART:shoulder}} pain {{PAIN_SEVERITY:4/10}}. ROM: "
    "{{ROM|movement=internal rotation|value=45|unit=deg::45 degrees internal rotation}}, "
    "{{BODY_PART:shoulder}}. Worse with {{TRIGGER:throwing}}.",

    # ------------------------------------------- vocabulary the rules lack
    # These use words that exist only in the NER training corpus.  They are the
    # ablation's honest test: the rule layer has no gazetteer entry for them.
    "Patient describes {{FINDING:aching}} in the {{LATERALITY:left}} {{BODY_PART:knee}} "
    "with {{PAIN_SEVERITY:5/10}} pain. {{TRIGGER:Stairs}} and {{TRIGGER:bending over}} "
    "bring it on; a {{RELIEVING_FACTOR:hot pack}} helps. "
    "{{ROM|movement=flexion|body_part=knee|value=100|unit=deg::"
    "Knee flexion is limited to 100 degrees}}. {{MUSCLE:Quadriceps}} {{FINDING:guarding}} "
    "noted. Started on {{EXERCISE:sit to stand}} and {{EXERCISE:mini squat}}.",

    "{{LATERALITY:Right}} {{BODY_PART:shoulder}} {{FINDING:discomfort}} {{PAIN_SEVERITY:6/10}}, "
    "{{FINDING:sore to touch}}. {{TRIGGER:Long car rides}} aggravate it and "
    "{{RELIEVING_FACTOR:supported sitting}} eases symptoms. "
    "{{ROM|movement=flexion|body_part=shoulder|value=110|unit=deg::"
    "Shoulder flexion is limited to 110 degrees}}. Prescribed {{EXERCISE:band walk}} "
    "and {{EXERCISE:monster walk}}.",

    "{{BODY_PART:low back}} {{FINDING:puffiness}} and {{FINDING:stiffness}} for "
    "{{DURATION:four weeks}}. {{TRIGGER:Prolonged sitting}} makes it worse; a "
    "{{RELIEVING_FACTOR:cold pack}} helps. "
    "{{ROM|movement=extension|body_part=lower back|value=15|unit=deg::"
    "Lumbar extension measures 15 degrees}}. {{MUSCLE:glute max}} {{FINDING:weakness}}. "
    "Home programme: {{EXERCISE:single leg squat}}, {{EXERCISE:dead bug}}.",

    "{{BODY_PART:neck}} {{FINDING:stiffness}} {{PAIN_SEVERITY:4/10}} with tension in the "
    "{{MUSCLE:thoracic erectors}}. {{TRIGGER:standing up}} and {{TRIGGER:looking down}} "
    "reproduce the pain. {{RELIEVING_FACTOR:supported sitting}} helps. "
    "{{ROM|movement=rotation|body_part=neck|value=55|unit=deg::"
    "Cervical rotation is limited to 55 degrees}}. Taught {{EXERCISE:chin tucks}} and "
    "{{EXERCISE:cervical retractions}}.",

    "{{LATERALITY:Bilateral}} {{BODY_PART:hip}} {{FINDING:aching}}, {{PAIN_SEVERITY:6/10}}. "
    "{{TRIGGER:lifting groceries}} causes pain; {{RELIEVING_FACTOR:lying down}} settles it. "
    "{{ROM|movement=flexion|body_part=hip|value=95|unit=deg::Hip flexion 95 degrees}}. "
    "{{MUSCLE:glute max}} {{FINDING:weakness}}. Given {{EXERCISE:sit to stand}} and "
    "{{EXERCISE:clamshells}}.",

    "{{LATERALITY:Left}} {{BODY_PART:ankle}} {{FINDING:puffiness}} and "
    "{{FINDING:discomfort}} after {{TRIGGER:running}}. {{TRIGGER:stairs}} are difficult. "
    "{{RELIEVING_FACTOR:cold pack}} helps. "
    "{{ROM|movement=dorsiflexion|body_part=ankle|value=8|unit=deg::"
    "Ankle dorsiflexion limited to 8 degrees}}. Prescribed {{EXERCISE:heel raises}}.",
]


def parse_marker(header: str, text: str) -> dict:
    """Turn ``LABEL|key=value`` plus span text into a gold entity dict."""
    parts = header.split("|")
    label = parts[0].strip()
    entity: dict = {"label": label, "text": text}
    for item in parts[1:]:
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key == "negated":
            entity["negated"] = value not in {"", "0", "false", "False"}
        elif key == "value":
            try:
                entity[key] = int(value)
            except ValueError:
                entity[key] = value
        elif key == "quality" and value == "":
            entity[key] = None
        else:
            entity[key] = value
    if entity.get("body_part") == "":
        entity["body_part"] = None
    return entity


def build_note(marked: str) -> dict:
    entities = []
    chunks = []
    cursor = 0
    for match in _MARKER.finditer(marked):
        literal = marked[cursor: match.start()]
        chunks.append(literal)
        char_start = sum(len(c) for c in chunks)
        text = match.group("text")
        entity = parse_marker(match.group("header"), text)
        entity["start"] = char_start
        entity["end"] = char_start + len(text)
        chunks.append(text)
        cursor = match.end()
        entities.append(entity)
    chunks.append(marked[cursor:])
    note_text = "".join(chunks)
    if "{{" in note_text or "}}" in note_text:
        raise ValueError(f"unparsed marker in note: {note_text!r}")
    return {"text": note_text, "entities": entities}


def main() -> int:
    notes = [build_note(note) for note in NOTES]
    for note in notes:
        text = note["text"]
        for entity in note["entities"]:
            if text[entity["start"]: entity["end"]] != entity["text"]:
                raise ValueError(f"offset mismatch in {text!r}")
        spans = sorted((e["start"], e["end"]) for e in note["entities"])
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise ValueError(f"overlapping entities in {text!r}")

    OUT.write_text(
        "".join(json.dumps(note, ensure_ascii=False) + "\n" for note in notes),
        encoding="utf-8",
    )
    labels = {}
    for note in notes:
        for entity in note["entities"]:
            labels[entity["label"]] = labels.get(entity["label"], 0) + 1
    print(f"{len(notes)} hand-written notes -> {OUT.relative_to(ROOT)}")
    print("labels:", dict(sorted(labels.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
