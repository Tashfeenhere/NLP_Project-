"""Seeded synthetic note generator with exact gold offsets.

Design note: the generator never *aligns* text after the fact.  It records the
character index at the moment it appends a slot, so every entity's
``start``/``end`` is correct by construction.  That is the whole reason the
train/dev/test splits are trustworthy.

Usage:

    python data/generate_notes.py --seed 42 --n 260

Writes ``data/raw/train.jsonl``, ``data/raw/dev.jsonl``,
``data/raw/generated_test.jsonl`` and ``data/raw/split_manifest.json``.
Split is **by template group**, not by note (see ``data/templates.py``).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.templates import SPLIT_GROUPS, TEMPLATES  # noqa: E402
from nlp import lexicons  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

JOINT_FAMILIES = list(lexicons.BODY_PARTS_BY_REGION)

PAIN_VALUES = [3, 4, 5, 6, 7, 8, 9]
PAIN_RANGES = [(4, 5), (5, 6), (6, 7), (7, 8)]

ROM_ANGLES = {
    # movement -> plausible angle range (degrees) for canonical/lateral joints
    "flexion": (70, 140),
    "extension": (0, 30),
    "abduction": (60, 170),
    "adduction": (10, 40),
    "internal rotation": (10, 70),
    "external rotation": (10, 70),
    "dorsiflexion": (0, 25),
    "plantar flexion": (0, 50),
    "inversion": (5, 35),
    "eversion": (5, 25),
    "side flexion": (5, 40),
}

MOVEMENTS_BY_JOINT = {
    "knee": ["flexion", "extension"],
    "lower_back": ["flexion", "extension", "side flexion"],
    "shoulder": ["flexion", "abduction", "external rotation", "internal rotation"],
    "neck": ["flexion", "extension", "rotation", "side flexion"],
    "hip": ["flexion", "abduction", "internal rotation", "external rotation"],
    "ankle": ["dorsiflexion", "plantar flexion", "inversion", "eversion"],
}

MUSCLES_BY_JOINT = {
    "knee": ["quadriceps", "hamstrings", "gluteals", "calf"],
    "lower_back": ["erector spinae", "multifidus", "gluteus maximus", "hip flexors"],
    "shoulder": ["deltoid", "trapezius", "rotator cuff muscles", "serratus anterior"],
    "neck": ["deep neck flexors", "upper trapezius", "trapezius"],
    "hip": ["gluteus medius", "gluteus maximus", "hip flexors", "adductors"],
    "ankle": ["calf", "gastrocnemius", "soleus", "abductors"],
}

EXERCISES_BY_JOINT = {
    "knee": ["quadriceps set", "straight leg raise", "wall squat", "step up", "clamshell"],
    "lower_back": ["bird dog", "dead bug", "glute bridge", "cat camel", "pelvic tilt"],
    "shoulder": ["wall slide", "external rotation exercise", "band pull apart", "scapular setting"],
    "neck": ["chin tuck", "cervical retraction", "shoulder shrug", "scapular retraction"],
    "hip": ["glute bridge", "side lying abduction", "clamshell", "hip hinge"],
    "ankle": ["calf raise", "heel raise", "single leg balance", "calf stretch"],
}

TRIGGERS_BY_JOINT = {
    "knee": ["climbing stairs", "squatting", "kneeling", "running", "walking downhill"],
    "lower_back": ["prolonged sitting", "lifting", "bending forward", "driving", "twisting"],
    "shoulder": ["overhead reaching", "throwing", "sleeping on side", "computer work"],
    "neck": ["computer work", "looking down", "turning head", "prolonged sitting"],
    "hip": ["walking", "climbing stairs", "prolonged standing", "running"],
    "ankle": ["running", "walking on uneven ground", "hopping", "jumping"],
}

RELIEVING_BY_JOINT = {
    "knee": ["rest", "ice", "elevation", "gentle movement"],
    "lower_back": ["rest", "walking", "change of position", "heat"],
    "shoulder": ["rest", "heat", "positioning", "massage"],
    "neck": ["rest", "heat", "massage", "positioning"],
    "hip": ["rest", "gentle movement", "heat"],
    "ankle": ["rest", "ice", "elevation", "massage"],
}

FINDINGS_BY_JOINT = {
    "knee": ["weakness", "effusion", "tightness", "crepitus"],
    "lower_back": ["stiffness", "muscle guarding", "tightness", "tenderness"],
    "shoulder": ["weakness", "tightness", "instability", "tenderness"],
    "neck": ["tightness", "stiffness", "tenderness", "spasm"],
    "hip": ["weakness", "tightness", "tenderness", "instability"],
    "ankle": ["weakness", "swelling", "stiffness", "tenderness"],
}

NEGATED_FINDINGS = ["swelling", "effusion", "redness", "numbness", "instability", "locking"]

# Vocabulary used only in the NER training corpus.  These surfaces are
# deliberately absent from ``nlp/lexicons.py``: they are what the trained model
# contributes over the rule layer, and the ablation measures that difference.
NER_ONLY = {
    "FINDING": ["guarding", "puffiness", "aching", "discomfort", "sore to touch"],
    "EXERCISE": ["sit to stand", "mini squat", "band walk", "monster walk", "single leg squat"],
    "TRIGGER": ["stairs", "bending over", "lifting groceries", "long car rides", "standing up"],
    "RELIEVING_FACTOR": ["hot pack", "cold pack", "supported sitting"],
    "MUSCLE": ["glute max", "thoracic erectors", "adductor group"],
}


def pool_choice(rng: random.Random, pool: list[str], label: str, extra_rate: float = 0.4) -> str:
    """Pick a surface, sometimes from the NER-only vocabulary."""
    extras = NER_ONLY.get(label, [])
    if extras and rng.random() < extra_rate:
        return rng.choice(extras)
    return rng.choice(pool)


def pool_sample(rng: random.Random, pool: list[str], label: str, count: int) -> list[str]:
    combined = list(pool)
    extras = [item for item in NER_ONLY.get(label, []) if item not in combined]
    if extras and rng.random() < 0.4:
        combined.extend(extras)
    return rng.sample(combined, k=min(count, len(combined)))


DURATIONS = ["two weeks", "three weeks", "four weeks", "six weeks", "two months", "three months", "one week", "five days"]
FREQUENCIES = ["constant", "intermittent", "daily", "occasional"]

SHORT_LAT = {"right": "R", "left": "L", "bilateral": "Bilateral"}
SHORT_BODY = {
    "knee": "knee",
    "lower back": "low back",
    "shoulder": "shoulder",
    "neck": "cervical spine",
    "hip": "hip",
    "ankle": "ankle",
}

LATERAL_CHOICES = ["right", "left", "bilateral", None]


class NoteBuilder:
    """Accumulates text and gold entities with exact offsets."""

    def __init__(self):
        self.parts: list[str] = []
        self.entities: list[dict] = []
        self.cursor = 0

    def add_literal(self, text: str) -> None:
        self.parts.append(text)
        self.cursor += len(text)

    def add_entity(self, text: str, label: str, laterality: str | None = None, negated: bool = False, **meta) -> None:
        start = self.cursor
        self.parts.append(text)
        self.cursor += len(text)
        entity = {
            "start": start,
            "end": self.cursor,
            "label": label,
            "text": text,
            **meta,
        }
        if laterality:
            entity["laterality"] = laterality
        if negated:
            entity["negated"] = True
        self.entities.append(entity)

    def add_span(self, start: int, label: str, **meta) -> None:
        """Record an entity covering everything appended since ``start``."""
        end = self.cursor
        self.entities.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": self.text()[start:end],
                **meta,
            }
        )

    def text(self) -> str:
        return "".join(self.parts)


def render_note(rng: random.Random, joint: str, template: str) -> dict:
    """Fill ``template`` for ``joint`` and return a gold-labelled note."""
    body = rng.choice(lexicons.BODY_PARTS_BY_REGION[joint])
    laterality = rng.choice(LATERAL_CHOICES)
    b = NoteBuilder()

    def latency_phrase(short: bool = False) -> None:
        if laterality is None:
            b.add_entity(body, "BODY_PART")
            return
        if short:
            b.add_entity(SHORT_LAT[laterality], "LATERALITY", laterality)
            b.add_literal(" ")
            b.add_entity(SHORT_BODY[joint], "BODY_PART", laterality)
        else:
            b.add_entity(laterality, "LATERALITY", laterality)
            b.add_literal(" ")
            b.add_entity(body, "BODY_PART", laterality)

    def severity() -> None:
        if rng.random() < 0.25:
            lo, hi = rng.choice(PAIN_RANGES)
            b.add_entity(f"{lo}-{hi}/10", "PAIN_SEVERITY")
        else:
            b.add_entity(f"{rng.choice(PAIN_VALUES)}/10", "PAIN_SEVERITY")

    def severity_short() -> None:
        b.add_entity(f"{rng.choice(PAIN_VALUES)}/10", "PAIN_SEVERITY")

    def rom(movement: str, angle: int | None, quality: str | None = None) -> None:
        """Append one whole-phrase ROM mention.

        The plan's entity model treats ``"knee flexion 90 degrees"`` as a single
        ``ROM`` span; the joint and movement live in the span's metadata and
        are never tagged separately inside it.  Recording the span as one
        entity keeps the generated gold consistent with the runtime reconciler,
        which drops body parts swallowed by a ROM span.
        """
        start = b.cursor
        if quality:
            b.add_literal(quality.capitalize())
            b.add_literal(" ")
        b.add_literal(body)
        b.add_literal(" ")
        b.add_literal(movement)
        if angle is not None:
            b.add_literal(" ")
            b.add_literal(f"{angle} degrees")
        b.add_span(
            start,
            "ROM",
            movement=lexicons.canonical("MOVEMENT", movement),
            body_part=lexicons.canonical("BODY_PART", body),
            value=angle,
            quality=None if angle is not None else (quality or "reduced"),
            unit="deg" if angle is not None else None,
        )

    def trigger() -> None:
        b.add_entity(pool_choice(rng, TRIGGERS_BY_JOINT[joint], "TRIGGER"), "TRIGGER")

    def relieving() -> None:
        b.add_entity(pool_choice(rng, RELIEVING_BY_JOINT[joint], "RELIEVING_FACTOR"), "RELIEVING_FACTOR")

    def finding() -> None:
        b.add_entity(pool_choice(rng, FINDINGS_BY_JOINT[joint], "FINDING"), "FINDING")

    def muscle() -> None:
        b.add_entity(pool_choice(rng, MUSCLES_BY_JOINT[joint], "MUSCLE"), "MUSCLE")

    def exercises(count: int) -> None:
        choices = pool_sample(rng, EXERCISES_BY_JOINT[joint], "EXERCISE", count)
        for i, ex in enumerate(choices):
            if i:
                b.add_literal(" and " if i == len(choices) - 1 else ", ")
            b.add_entity(ex, "EXERCISE")

    template_key = template
    movement = rng.choice(MOVEMENTS_BY_JOINT[joint])
    lo, hi = ROM_ANGLES.get(movement, (60, 120))
    if rng.random() < 0.2:
        angle, quality = None, rng.choice(["reduced", "limited", "restricted"])
    else:
        angle, quality = rng.randint(lo, hi), None

    if template_key == "complaint":
        latency_phrase()
        b.add_literal(" pain, ")
        severity()
        b.add_literal(", ")
        b.add_entity(rng.choice(FREQUENCIES), "FREQUENCY")
        b.add_literal(".")
    elif template_key == "trigger":
        b.add_literal("Aggravated by " if rng.random() < 0.5 else "Worse with ")
        trigger()
        b.add_literal(". Relieved by " if rng.random() < 0.5 else ". Better with ")
        relieving()
        b.add_literal(".")
    elif template_key == "rom":
        b.add_literal("On assessment, ")
        rom(movement, angle, quality)
        b.add_literal(".")
    elif template_key == "muscle":
        b.add_literal("Findings: ")
        finding()
        b.add_literal(" in the ")
        muscle()
        b.add_literal(".")
    elif template_key == "exercise":
        b.add_literal("Treatment: ")
        exercises(rng.randint(1, 3))
        b.add_literal(".")
    elif template_key == "duration":
        b.add_literal("Symptoms for ")
        b.add_entity(rng.choice(DURATIONS), "DURATION")
        b.add_literal(".")
    elif template_key == "negated":
        b.add_literal(rng.choice(["Denies ", "No ", "Without "]))
        b.add_entity(rng.choice(NEGATED_FINDINGS), "FINDING", negated=True)
        b.add_literal(".")
    elif template_key == "full":
        latency_phrase()
        b.add_literal(" pain, ")
        severity()
        b.add_literal(", ")
        b.add_entity(rng.choice(FREQUENCIES), "FREQUENCY")
        b.add_literal(" for ")
        b.add_entity(rng.choice(DURATIONS), "DURATION")
        b.add_literal(". Aggravated by ")
        trigger()
        b.add_literal("; relieved by ")
        relieving()
        b.add_literal(". ")
        rom(movement, angle, quality)
        b.add_literal(". Noted ")
        finding()
        b.add_literal(" in the ")
        muscle()
        b.add_literal(". Prescribed ")
        exercises(rng.randint(1, 3))
        b.add_literal(".")
    elif template_key == "terse":
        latency_phrase(short=True)
        b.add_literal(" pain ")
        severity_short()
        b.add_literal(". ROM: ")
        start = b.cursor
        b.add_literal(movement)
        b.add_literal(" ")
        if angle is not None:
            b.add_literal(f"{angle} degrees")
        else:
            b.add_literal(quality or "reduced")
        b.add_span(
            start,
            "ROM",
            movement=lexicons.canonical("MOVEMENT", movement),
            body_part=None,
            value=angle,
            quality=None if angle is not None else (quality or "reduced"),
            unit="deg" if angle is not None else None,
        )
        b.add_literal(". Worse with ")
        trigger()
        b.add_literal(". Rx: ")
        exercises(1)
        b.add_literal(".")
    elif template_key == "reversed_rom":
        latency_phrase()
        b.add_literal(" pain ")
        severity_short()
        b.add_literal(". ROM: ")
        start = b.cursor
        if angle is not None:
            b.add_literal(f"{angle} degrees")
        else:
            b.add_literal(quality or "reduced")
        b.add_literal(" ")
        b.add_literal(movement)
        b.add_span(
            start,
            "ROM",
            movement=lexicons.canonical("MOVEMENT", movement),
            body_part=None,
            value=angle,
            quality=None if angle is not None else (quality or "reduced"),
            unit="deg" if angle is not None else None,
        )
        b.add_literal(", ")
        b.add_entity(body, "BODY_PART")
        b.add_literal(". Aggravated by ")
        trigger()
        b.add_literal(". Plan: ")
        exercises(1)
        b.add_literal(".")
    elif template_key == "trigger_relief":
        b.add_literal("Worse during ")
        trigger()
        b.add_literal(" and better with ")
        relieving()
        b.add_literal(".")
    elif template_key == "negated_full":
        latency_phrase()
        b.add_literal(" pain at ")
        severity_short()
        b.add_literal(". No ")
        b.add_entity(rng.choice(NEGATED_FINDINGS), "FINDING", negated=True)
        b.add_literal(". ")
        rom(movement, angle, quality)
        b.add_literal(". Advised ")
        exercises(1)
        b.add_literal(".")
    else:
        raise ValueError(f"unknown template {template_key}")

    return {"text": b.text(), "entities": b.entities, "joint": joint, "template": template_key}


def generate_group(rng: random.Random, group: str, n: int) -> list[dict]:
    templates = TEMPLATES[group]
    keys = list(templates)
    notes = []
    for i in range(n):
        joint = JOINT_FAMILIES[i % len(JOINT_FAMILIES)]
        template = keys[i % len(keys)]
        notes.append(render_note(rng, joint, template))
    return notes


def render_augmented(rng: random.Random) -> dict:
    """One training-only note with unusual but plausible word order.

    The template splits keep the *phrasing* of train and test disjoint, but a
    model trained on a small set of frames learns the frame as much as the
    vocabulary: it will tag ``glute max`` as MUSCLE only when it follows "in
    the".  These notes vary the syntax of the NER-only terms (and of ordinary
    terms) so the recogniser learns the spans rather than the slot position.
    They are training-only and are not part of any evaluation split.
    """
    joint = rng.choice(JOINT_FAMILIES)
    body = rng.choice(lexicons.BODY_PARTS_BY_REGION[joint])
    laterality = rng.choice(LATERAL_CHOICES)
    b = NoteBuilder()

    def body_phrase(capitalize: bool = False) -> None:
        if laterality:
            b.add_entity(SHORT_LAT[laterality] if rng.random() < 0.4 else laterality, "LATERALITY", laterality)
            b.add_literal(" ")
        b.add_entity(body.capitalize() if capitalize else body, "BODY_PART", laterality)

    def finding() -> None:
        b.add_entity(pool_choice(rng, FINDINGS_BY_JOINT[joint], "FINDING", 0.6), "FINDING")

    def trigger() -> None:
        b.add_entity(pool_choice(rng, TRIGGERS_BY_JOINT[joint], "TRIGGER", 0.6), "TRIGGER")

    def relieving() -> None:
        b.add_entity(pool_choice(rng, RELIEVING_BY_JOINT[joint], "RELIEVING_FACTOR", 0.6), "RELIEVING_FACTOR")

    def muscle() -> None:
        b.add_entity(pool_choice(rng, MUSCLES_BY_JOINT[joint], "MUSCLE", 0.6), "MUSCLE")

    def exercise() -> None:
        b.add_entity(pool_choice(rng, EXERCISES_BY_JOINT[joint], "EXERCISE", 0.6), "EXERCISE")

    frame = rng.randrange(11)
    if frame == 0:
        # Body and finding are separated by a preposition.  Adjacent slots
        # ("knee stiffness") teach a small NER head to merge the two spans.
        b.add_literal("There is ")
        finding()
        b.add_literal(" in the ")
        body_phrase()
        b.add_literal(" for ")
        b.add_entity(rng.choice(DURATIONS), "DURATION")
        b.add_literal(".")
    elif frame == 1:
        b.add_literal("Patient reports ")
        finding()
        b.add_literal(" in the ")
        body_phrase()
        b.add_literal(". ")
        b.add_entity(pool_choice(rng, MUSCLES_BY_JOINT[joint], "MUSCLE", 0.6), "MUSCLE")
        b.add_literal(" shows ")
        finding()
        b.add_literal(".")
    elif frame == 2:
        body_phrase(capitalize=True)
        b.add_literal(" pain; ")
        trigger()
        b.add_literal(" brings it on. ")
        b.add_entity(pool_choice(rng, RELIEVING_BY_JOINT[joint], "RELIEVING_FACTOR", 0.6), "RELIEVING_FACTOR")
        b.add_literal(" eases symptoms.")
    elif frame == 3:
        b.add_literal("Aggravated by ")
        trigger()
        b.add_literal("; ")
        relieving()
        b.add_literal(" provides relief. Tenderness in the ")
        muscle()
        b.add_literal(".")
    elif frame == 4:
        b.add_literal("Started on ")
        exercise()
        b.add_literal(" and ")
        exercise()
        b.add_literal(". Also taught ")
        exercise()
        b.add_literal(".")
    elif frame == 5:
        b.add_literal("Pain ")
        b.add_entity(f"{rng.choice(PAIN_VALUES)}/10", "PAIN_SEVERITY")
        b.add_literal(" in the ")
        body_phrase()
        b.add_literal(". ")
        b.add_entity(pool_choice(rng, TRIGGERS_BY_JOINT[joint], "TRIGGER", 0.6), "TRIGGER")
        b.add_literal(" causes pain; ")
        relieving()
        b.add_literal(" helps.")
    elif frame == 6:
        b.add_literal("Findings: ")
        finding()
        b.add_literal(" of the ")
        muscle()
        b.add_literal(". No ")
        b.add_entity(rng.choice(NEGATED_FINDINGS), "FINDING", negated=True)
        b.add_literal(". Home programme: ")
        exercise()
        b.add_literal(" and ")
        exercise()
        b.add_literal(".")
    elif frame == 7:
        b.add_literal("Home programme: ")
        exercise()
        b.add_literal(". ")
        b.add_entity(pool_choice(rng, RELIEVING_BY_JOINT[joint], "RELIEVING_FACTOR", 0.6), "RELIEVING_FACTOR")
        b.add_literal(" helps; ")
        trigger()
        b.add_literal(" aggravates the ")
        b.add_entity(body, "BODY_PART", laterality)
        b.add_literal(".")
    elif frame == 8:
        b.add_literal("Tension in the ")
        muscle()
        b.add_literal(". ")
        b.add_entity(pool_choice(rng, MUSCLES_BY_JOINT[joint], "MUSCLE", 0.6), "MUSCLE")
        b.add_literal(" is tender on palpation.")
    elif frame == 9:
        trigger()
        b.add_literal(" makes it worse. ")
        relieving()
        b.add_literal(" helps.")
    else:
        b.add_literal("Pain after ")
        trigger()
        b.add_literal(". ")
        relieving()
        b.add_literal(" helps.")

    return {"text": b.text(), "entities": b.entities, "joint": joint, "template": "augmented"}


def render_augmented_set(rng: random.Random, n: int) -> list[dict]:
    return [render_augmented(rng) for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=260)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts = {"train": 175, "dev": 40, "generated_test": 45}
    scale = args.n / sum(counts.values()) if args.n else 1.0

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "counts": {}, "groups": SPLIT_GROUPS}

    for split, group_names in SPLIT_GROUPS.items():
        total = max(1, round(counts[split] * scale))
        notes = []
        for i, group in enumerate(group_names):
            # Spread the remainder across groups instead of truncating it, so
            # the requested count is the count actually written.
            group_total = total // len(group_names) + (1 if i < total % len(group_names) else 0)
            notes.extend(generate_group(rng, group, max(1, group_total)))
        notes = notes[:total]
        out = RAW / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for note in notes:
                fh.write(json.dumps(note, ensure_ascii=False) + "\n")
        manifest["counts"][split] = len(notes)
        print(f"{split:16s} {len(notes):4d} notes -> {out.relative_to(ROOT)}")

    # Training-only syntax augmentation (see render_augmented).  It is appended
    # after the split files are written so dev and the generated test stay
    # untouched; only the NER head sees these notes.
    extras = render_augmented_set(rng, 160)
    with (RAW / "train.jsonl").open("a", encoding="utf-8") as fh:
        for note in extras:
            fh.write(json.dumps(note, ensure_ascii=False) + "\n")
    manifest["counts"]["train"] += len(extras)
    manifest["augmented_train"] = len(extras)
    print(f"{'augmented':16s} {len(extras):4d} notes appended to train.jsonl")

    (RAW / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(manifest["counts"].values())
    print(f"{'total':16s} {total:4d} notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
