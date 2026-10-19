"""Phrasing templates for the synthetic note generator.

Templates are split into ``train``, ``dev`` and ``generated_test`` *sets* --
the split is by template, not by note, so a phrasing seen in training is never
scored against.  ``handwritten`` notes live in ``data/hand_written_test.jsonl``
and are produced by ``data/build_handwritten.py``.

The format strings below are the canonical description of each phrasing.  The
generator implements the same variants procedurally in
``data/generate_notes.py::render_note`` rather than formatting these strings
directly, because entity offsets must be recorded at the moment each slot is
appended.  The keys of ``TEMPLATES`` are what drive which variants a split
contains.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Slot renderers
# --------------------------------------------------------------------------
# A renderer takes an RNG plus the joint-family context and returns the surface
# string.  ``None`` means the slot is absent and the template branch using it
# is skipped.
# (The generator records offsets while appending slots; see the module
# docstring.  These format strings document the phrasings, not the mechanism.)

# Which template groups feed which split.  Disjoint on purpose.
SPLIT_GROUPS = {
    "train": ["canonical", "terse", "reversed_rom", "contextual"],
    "dev": ["narrative"],
    "generated_test": ["mixed"],
}

TEMPLATES = {
    # ---- canonical -------------------------------------------------------
    "canonical": {
        "complaint": "{laterality} {body_part} pain, {severity}. {frequency}.",
        "trigger": "Aggravated by {trigger}. Relieved by {relieving}.",
        "rom": "On assessment, {body_part} {movement} {rom_value}.",
        "muscle": "Findings: {finding} in the {muscle}.",
        "exercise": "Treatment: {exercise_list}.",
        "duration": "Symptoms for {duration}.",
        "negated": "Denies {negated_finding}.",
        "full": (
            "{laterality} {body_part} pain, {severity}, {frequency} for {duration}. "
            "Aggravated by {trigger}; relieved by {relieving}. "
            "{body_part} {movement} {rom_value}. "
            "Noted {finding} in the {muscle}. "
            "Prescribed {exercise_list}."
        ),
    },
    # ---- terse -----------------------------------------------------------
    "terse": {
        "complaint": "{lat_short} {body_short} pain {pain_short}.",
        "rom": "ROM: {movement_short} {angle}.",
        "exercise": "Rx: {exercise_list}.",
        "negated": "No {negated_finding}.",
        "full": (
            "{lat_short} {body_short} pain {pain_short}. "
            "ROM: {movement_short} {angle}. "
            "Worse with {trigger}. "
            "Rx: {exercise_list}."
        ),
    },
    # ---- reversed ROM ----------------------------------------------------
    "reversed_rom": {
        "rom": "ROM: {angle} {movement}, {body_part}.",
        "full": (
            "{laterality} {body_part} pain {severity_short}. "
            "ROM: {angle} {movement}, {body_part}. "
            "Aggravated by {trigger}. "
            "Plan: {exercise_list}."
        ),
    },
    # ---- contextual ------------------------------------------------------
    "contextual": {
        "trigger_relief": "Worse during {trigger} and better with {relieving}.",
        "full": (
            "Patient presents with {laterality} {body_part} pain. "
            "Severity rated {severity}. "
            "Worse during {trigger} and better with {relieving}. "
            "{body_part} {movement} {rom_value}. "
            "Home programme: {exercise_list}."
        ),
    },
    # ---- narrative (dev) -------------------------------------------------
    "narrative": {
        "full": (
            "The patient describes {laterality} {body_part} discomfort that has "
            "been present for {duration}. Intensity is {severity} and it is {frequency}. "
            "Activities such as {trigger} aggravate the symptoms, while {relieving} "
            "provides relief. {Movement_cap} of the {body_part} measures {rom_value}. "
            "There is {finding} of the {muscle}. "
            "{Exercise_cap} were taught today."
        ),
    },
    # ---- mixed (generated test) ------------------------------------------
    # These phrasings appear nowhere else, which is the point: the generated
    # test split measures generalisation to new templates, not memorisation.
    "mixed": {
        "full": (
            "Assessment: {laterality} {body_part} pain at {severity}, {frequency}. "
            "{trigger_cap} brings it on, {relieving} settles it. "
            "{body_part} {movement} {rom_value}. {finding_cap} noted in {muscle}. "
            "Given {exercise_list} to complete daily."
        ),
        "negated_full": (
            "{laterality} {body_part} pain at {severity}. "
            "No {negated_finding}. "
            "{body_part} {movement} {rom_value}. "
            "Advised {exercise_list}."
        ),
    },
}
