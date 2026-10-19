"""Guard against notes that carry no usable clinical content.

The UI calls this before showing extracted fields so an empty box, a stray
keystroke, or a pasted URL does not produce a confident-looking empty record
with warnings.  Returning ``ready=False`` plus a reason lets the frontend say
"nothing to analyze yet" instead of "0 entities found".
"""
from __future__ import annotations

import re

MIN_CHARS = 12

# A note is plausible when it contains at least this many alphabetic tokens.
# Two is deliberate: terse but real notes ("R knee pain 6/10") must pass.
MIN_WORDS = 2

_URL = re.compile(r"https?://\S+|www\.\S+")
_ALPHA_WORD = re.compile(r"[A-Za-z]{2,}")

_CLINICAL_HINTS = {
    "pain",
    "painful",
    "ache",
    "aching",
    "sore",
    "soreness",
    "knee",
    "knees",
    "shoulder",
    "shoulders",
    "back",
    "neck",
    "hip",
    "hips",
    "ankle",
    "ankles",
    "quadriceps",
    "hamstring",
    "hamstrings",
    "gluteal",
    "gluteals",
    "calf",
    "flexion",
    "extension",
    "abduction",
    "rotation",
    "range",
    "rom",
    "exercise",
    "exercises",
    "squat",
    "lunge",
    "bridge",
    "raise",
    "strength",
    "weak",
    "weakness",
    "tight",
    "tightness",
    "stiff",
    "stiffness",
    "swelling",
    "swollen",
    "walking",
    "stairs",
    "patient",
    "rehab",
    "physio",
    "mobility",
    "stretch",
}


def assess(text: str) -> tuple[bool, str | None]:
    """Return ``(ready, reason)``.

    ``ready`` is ``True`` only when the text looks like a clinical note.  The
    reason is a short human-readable string for the UI banner, or ``None``.
    """
    stripped = text.strip()
    if not stripped:
        return False, "The note is empty."
    words = _ALPHA_WORD.findall(stripped.lower())
    if _URL.search(stripped) and not _has_clinical_hint(words):
        return False, "This looks like a link, not a clinical note."
    if len(stripped) < MIN_CHARS:
        return False, "The note is too short to analyze."
    if len(words) < MIN_WORDS:
        return False, "The note does not contain enough words to analyze."
    if not _has_clinical_hint(words):
        return False, "No physiotherapy terms were recognized; check that the right text was pasted."
    return True, None


def _has_clinical_hint(words: list[str]) -> bool:
    """Match hints through simple inflection (``raises`` -> ``raise``)."""
    for word in words:
        if word in _CLINICAL_HINTS:
            return True
        for suffix, cut in (("s", 1), ("es", 2), ("ing", 3), ("ed", 2)):
            if word.endswith(suffix) and word[: -cut] in _CLINICAL_HINTS:
                return True
    return False
