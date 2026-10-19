"""Reconciler: precedence, non-overlap, laterality binding, field assembly.

The pipeline produces three kinds of candidates:

* rule candidates from the ``Doc._.rule_candidates`` extension,
* NER ``Doc.ents`` from the fine-tuned component.

``reconcile`` merges them into a single non-overlapping list following the
precedence table:

    ROM (whole phrase)  >  PAIN_SEVERITY  >  RELIEVING_FACTOR  >  TRIGGER
    >  context rule  >  gazetteer  >  NER

Within the same precedence, the longer span wins, then the earlier start.

Two special cases are not pure precedence:

* Laterality candidates never survive as entities.  They are consumed into
  ``meta["laterality"]`` of the body part or ROM they bind to -- the proposal
  lists ``LATERALITY`` as a label, but emitting ``right`` as a standalone
  entity highlights noise in the UI and double-counts it in evaluation.
  They are still reported through the bound fields.
* A ``BODY_PART`` fully contained inside a ``ROM`` span is dropped: the ROM
  span already carries the joint in ``meta["body_part"]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import Candidate
from .normalize import canon_laterality
from .rules.body_part import bind_laterality


@dataclass
class Reconciled:
    entities: list[Candidate]
    negated: list[Candidate]
    fields: dict
    warnings: list[str] = field(default_factory=list)


CANONICAL_FIELD_ORDER = [
    "body_part",
    "laterality",
    "pain_severity",
    "rom",
    "triggers",
    "relieving_factors",
    "muscles",
    "findings",
    "exercises",
    "duration",
    "frequency",
]


def reconcile(doc, rule_candidates: list[Candidate], ner_candidates: list[Candidate]) -> Reconciled:
    """Merge candidates, resolve overlaps, bind laterality, assemble fields."""
    warnings: list[str] = []

    # 1. Laterality is bound before overlap resolution so the body-part span
    #    it belongs to is still identifiable.
    lat_map = bind_laterality(doc)

    # A ROM span that contains the joint also contains its laterality
    # ("right knee flexion").  Overlap resolution will drop the contained
    # BODY_PART before field assembly sees it, so copy the bound value onto the
    # ROM candidate now.
    for cand in rule_candidates:
        if cand.label != "ROM":
            continue
        for part in doc._.rule_candidates:
            if (
                part.label == "BODY_PART"
                and cand.start <= part.start
                and part.end <= cand.end
                and lat_map.get(part.start)
            ):
                cand.meta["laterality"] = lat_map[part.start]
                break

    merged: list[Candidate] = []
    for cand in list(rule_candidates) + list(ner_candidates):
        if cand.label == "LATERALITY":
            continue  # consumed above
        merged.append(cand)

    # 2. Non-overlapping selection.
    merged.sort(key=lambda c: c.sort_key())
    kept: list[Candidate] = []
    for cand in merged:
        overlapping = [k for k in kept if cand.overlaps(k)]
        if overlapping:
            # A longer exercise span that strictly contains a same-label rule
            # match is the more specific mention ("mini squat" vs the rule's
            # "squat").  Restrict this to EXERCISE: applying it to every label
            # lets an NER span such as "shoulder tightness" swallow the rule's
            # correct BODY_PART "shoulder".
            replaced = False
            if cand.label == "EXERCISE":
                for k in overlapping:
                    if k.label == "EXERCISE" and _contains(cand, k) and not _contains(k, cand):
                        kept.remove(k)
                        replaced = True
            if replaced:
                kept.append(cand)
            continue
        kept.append(cand)
    kept.sort(key=lambda c: c.start)

    # 3. Drop body parts swallowed by a ROM span, preserving their laterality
    #    on the ROM candidate so the field is not lost with the span.
    rom_spans = [c for c in kept if c.label == "ROM"]
    for rom in rom_spans:
        for inner in kept:
            if inner.label == "BODY_PART" and _contains(rom, inner):
                if inner.meta.get("laterality") and not rom.meta.get("laterality"):
                    rom.meta["laterality"] = inner.meta["laterality"]
    kept = [
        c
        for c in kept
        if not (c.label == "BODY_PART" and any(_contains(r, c) for r in rom_spans))
    ]

    # 4. Attach bound laterality to body parts.
    for cand in kept:
        if cand.label == "BODY_PART":
            value = lat_map.get(cand.start)
            if value:
                cand.meta["laterality"] = value
                cand.value = value

    negated = [c for c in kept if c.negated]
    positive = [c for c in kept if not c.negated]

    fields = assemble_fields(positive)

    if not positive:
        warnings.append("No clinical entities were extracted from this note.")
    if not fields.get("body_part"):
        warnings.append("Body part not identified; check the note or set it manually.")

    return Reconciled(entities=positive, negated=negated, fields=fields, warnings=warnings)


def _contains(outer: Candidate, inner: Candidate) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


# --------------------------------------------------------------------------
# Field assembly
# --------------------------------------------------------------------------


def assemble_fields(entities: list[Candidate]) -> dict:
    """Turn the positive entity list into the ``ClinicalRecord`` payload.

    Ordered by appearance in the note so the table reads like the source.
    """
    body_parts: list[Candidate] = []
    laterality = None
    pain = None
    roms: list[Candidate] = []
    triggers: list[Candidate] = []
    relieving: list[Candidate] = []
    muscles: list[Candidate] = []
    findings: list[Candidate] = []
    exercises: list[Candidate] = []
    durations: list[Candidate] = []
    frequencies: list[Candidate] = []

    for cand in entities:
        if cand.label == "BODY_PART":
            body_parts.append(cand)
        elif cand.label == "PAIN_SEVERITY":
            if pain is None:
                pain = cand
        elif cand.label == "ROM":
            roms.append(cand)
        elif cand.label == "TRIGGER":
            triggers.append(cand)
        elif cand.label == "RELIEVING_FACTOR":
            relieving.append(cand)
        elif cand.label == "MUSCLE":
            muscles.append(cand)
        elif cand.label == "FINDING":
            findings.append(cand)
        elif cand.label == "EXERCISE":
            exercises.append(cand)
        elif cand.label == "DURATION":
            durations.append(cand)
        elif cand.label == "FREQUENCY":
            frequencies.append(cand)

    if body_parts:
        first = body_parts[0]
        laterality = first.meta.get("laterality") or canon_laterality(first.value if isinstance(first.value, str) else None)

    # The first body part wins for the headline field; the rest stay as
    # entities so nothing is lost from the note.
    body_part = _canon("BODY_PART", body_parts[0].text) if body_parts else None

    # A note whose only joint mention sits inside a ROM phrase (the reconciler
    # removes the contained BODY_PART entity) still has a headline body part:
    # the ROM candidate carries it in metadata.
    if body_part is None:
        for rom in roms:
            if rom.meta.get("body_part"):
                body_part = rom.meta["body_part"]
                break
    if laterality is None:
        for rom in roms:
            if rom.meta.get("laterality"):
                laterality = rom.meta["laterality"]
                break

    # ROM rows: joint comes from the bound joint, else the note's body part,
    # else the movement's joint hint.
    rom_rows = []
    for rom in roms:
        meta = rom.meta
        movement = meta.get("movement") or rom.normalized
        joint = meta.get("body_part") or body_part
        value = rom.value or {}
        rom_rows.append(
            {
                "movement": movement,
                "value": value.get("value"),
                "unit": value.get("unit"),
                "quality": value.get("quality"),
            }
        )

    return {
        "body_part": body_part,
        "laterality": laterality,
        "pain_severity": (
            {"raw": pain.value["raw"], "value": pain.value.get("value"), "range": pain.value.get("range")}
            if pain is not None and isinstance(pain.value, dict)
            else None
        ),
        "rom": rom_rows,
        "triggers": _dedupe([_norm(c) for c in triggers]),
        "relieving_factors": _dedupe([_norm(c) for c in relieving]),
        "muscles": _dedupe([_canon("MUSCLE", c.text) for c in muscles]),
        "findings": _dedupe([_canon("FINDING", c.text) for c in findings]),
        "exercises": _dedupe([_norm(c) for c in exercises]),
        "duration": _dedupe([c.normalized or c.text for c in durations])[0] if durations else None,
        "frequency": _dedupe([c.normalized or c.text for c in frequencies])[0] if frequencies else None,
    }


def _canon(label: str, text: str) -> str:
    from .lexicons import canonical

    return canonical(label, text)


def _norm(cand: Candidate) -> str:
    if cand.normalized:
        return cand.normalized
    from .lexicons import canonical

    return canonical(cand.label, cand.text)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out
