"""Text normalization and value canonicalization.

The normalizer runs *before* spaCy sees the text.  Every transformation here is
length-preserving (character-for-character) so that rule matches and NER spans
computed on the normalized string still slice the original note the user pasted
into the UI.  ``assert_same_length`` guards that invariant.

Value canonicalization (``canon_pain``, ``canon_rom_value``, ...) operates on
already-extracted strings and has no offset obligations.
"""
from __future__ import annotations

import re
import unicodedata

from . import lexicons

# --------------------------------------------------------------------------
# Length-preserving text normalization
# --------------------------------------------------------------------------

_WS = re.compile(r"[ \t  -   ]+")
_DASHES = re.compile(r"[‐‑‒–—―−]")


def normalize_text(text: str) -> str:
    """Return ``text`` with clinically irrelevant surface variation removed.

    Only substitutions that keep the character count identical are applied, so
    offsets computed on the result slice the caller's original string:

    * NFKC (guarded -- if it would change the length we keep the original),
    * unicode dashes -> ASCII hyphen,
    * runs of horizontal whitespace -> same-length run of spaces.

    Newlines are preserved because the UI renders the note with its paragraphs.
    """
    normalized = unicodedata.normalize("NFKC", text)
    if len(normalized) != len(text):
        # Ligatures and similar expansions would shift every offset after them.
        # Clinical notes do not rely on them; keeping the original is safer
        # than mapping spans twice.
        normalized = text
    normalized = _DASHES.sub("-", normalized)
    normalized = _WS.sub(lambda m: " " * len(m.group()), normalized)
    return normalized


def assert_same_length(original: str, normalized: str) -> None:
    if len(original) != len(normalized):
        raise ValueError(
            f"normalize_text changed length ({len(original)} -> {len(normalized)}); "
            "offset contract violated"
        )


# --------------------------------------------------------------------------
# Pain severity
# --------------------------------------------------------------------------

# Accepts "7/10", "7 / 10", "7 out of 10", "7-8/10", "7 to 8 out of 10".
_PAIN = re.compile(
    r"^\s*(\d{1,2})\s*(?:[-to]+\s*(\d{1,2})\s*)?"
    r"(?:/\s*10|out\s+of\s+10|/\s*10\b)\s*$",
    re.IGNORECASE,
)


def canon_pain(raw: str) -> dict | None:
    """Turn a pain span into ``{raw, value, range}``.

    ``value`` is the score on a 0-10 scale, or ``None`` when the score falls
    outside it (a documented "12/10" is kept raw but not trusted).  A range
    ("7-8/10") keeps both endpoints and reports the lower bound as ``value``,
    matching how clinicians quote it.
    """
    m = _PAIN.match(raw)
    if not m:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else None
    if not 0 <= lo <= 10:
        return {"raw": raw.strip(), "value": None, "range": None}
    if hi is not None and not 0 <= hi <= 10:
        hi = None
    if hi is None:
        return {"raw": raw.strip(), "value": lo, "range": None}
    if hi < lo:
        lo, hi = hi, lo
    return {"raw": raw.strip(), "value": lo, "range": [lo, hi]}


# --------------------------------------------------------------------------
# ROM values
# --------------------------------------------------------------------------

_DEGREES = re.compile(r"(-?\d{1,3})\s*(?:deg\b|degree\b|degrees\b|°)", re.IGNORECASE)
_PLAIN_ANGLE = re.compile(r"\b(-?\d{1,3})\s*$")
_QUALITY_ONLY = re.compile(
    r"\b(" + "|".join(sorted(lexicons.ROM_QUALITY_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def canon_rom_value(raw: str) -> tuple[int | None, str | None]:
    """Return ``(value, quality)`` from a ROM value fragment.

    ``"90 degrees"`` -> ``(90, None)``; ``"reduced"`` -> ``(None, "reduced")``;
    ``"approximately 90 deg"`` -> ``(90, None)``; out-of-range angles are
    rejected rather than clipped.
    """
    m = _DEGREES.search(raw)
    if m:
        value = int(m.group(1))
        return (value, None) if 0 <= value <= 180 else (None, None)
    m = _PLAIN_ANGLE.search(raw.strip())
    if m:
        value = int(m.group(1))
        return (value, None) if 0 <= value <= 180 else (None, None)
    m = _QUALITY_ONLY.search(raw)
    if m:
        return None, lexicons.ROM_QUALITY_WORDS[m.group(1).lower()]
    return None, None


# --------------------------------------------------------------------------
# Laterality
# --------------------------------------------------------------------------


def canon_laterality(raw: str | None) -> str | None:
    """Map a surface form to ``right`` / ``left`` / ``bilateral``.

    Returns ``None`` for anything unrecognised -- laterality is never guessed,
    because guessing it is a clinical error rather than a formatting one.
    """
    if not raw:
        return None
    token = raw.strip().lower().rstrip(".")
    if token in lexicons.LATERALITY_WORDS:
        return lexicons.LATERALITY_WORDS[token]
    if token.endswith(" sided"):
        token = token[: -len(" sided")]
        if token in lexicons.LATERALITY_WORDS:
            return lexicons.LATERALITY_WORDS[token]
    return None


# --------------------------------------------------------------------------
# Duration / frequency
# --------------------------------------------------------------------------

_DURATION_NUM = re.compile(r"\b(\d{1,3})\s+(day|week|month|year)s?\b", re.IGNORECASE)
_DURATION_WORD = re.compile(
    r"\b(" + "|".join(lexicons.NUMBER_WORDS) + r")\s+(day|week|month|year)s?\b",
    re.IGNORECASE,
)
_SINCE_MONTH = re.compile(
    r"\bsince\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|last\s+\w+|early\s+\w+)\b",
    re.IGNORECASE,
)


def canon_duration(raw: str) -> str | None:
    """Return a normalized duration such as ``"3 weeks"`` or ``"since March"``."""
    text = " ".join(raw.lower().split())
    m = _DURATION_NUM.search(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if n > 0:
            return f"{n} {unit}" if n == 1 else f"{n} {unit}s"
    m = _DURATION_WORD.search(text)
    if m:
        n = lexicons.NUMBER_WORDS[m.group(1).lower()]
        unit = m.group(2).lower()
        return f"{n} {unit}" if n == 1 else f"{n} {unit}s"
    m = _SINCE_MONTH.search(text)
    if m:
        return "since " + " ".join(m.group(1).split()).lower().replace("since ", "")
    return None


def canon_frequency(raw: str) -> str | None:
    return lexicons.canonical("FREQUENCY", raw)


# --------------------------------------------------------------------------
# Generic entity canonicalization
# --------------------------------------------------------------------------


def canon_entity(label: str, surface: str) -> str:
    """Canonical form for a labelled entity.

    Exercise names keep their curated capitalisation ("McKenzie extension");
    everything else is lower-cased so the frontend table compares cleanly.
    """
    value = lexicons.canonical(label, surface)
    if label == "EXERCISE":
        for canon_value in lexicons.EXERCISE_CANON.values():
            if canon_value.lower() == value.lower():
                return canon_value
    return value.lower()
