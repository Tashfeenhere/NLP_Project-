"""NegEx-style negation scoping.

A negation cue (``no``, ``denies``, ``without``, ``negative for``, ``absence
of``, ...) negates candidates that appear *after* it in the same sentence,
stopping at a scope breaker ("but", "however", a new clause verb) or after a
bounded window.

    "Denies neck pain"          -> neck is negated, no BODY_PART emitted
    "no swelling or effusion"   -> both findings negated
    "no swelling but pain 7/10" -> swelling negated, severity kept
    "pain without stiffness"    -> stiffness negated
    "without stiffness, 4/10"   -> stiffness negated, the score is kept

Two deliberate behaviours:

* A cue at the very start of a sentence negates the whole remaining clause,
  which is how "No redness or warmth over the joint." reads.
* A cue followed by its own clause boundary never leaks into the next
  sentence.  Negation never crosses a ``\\.`` boundary.
"""
from __future__ import annotations

from spacy.tokens import Doc

from .. import lexicons
from ..candidates import Candidate

_WINDOW = 12  # tokens after a cue that remain in scope

# Labels that can continue a negated enumeration across a comma
# ("no swelling, warmth or redness").
_ENUMERATION_LABELS = {
    "BODY_PART",
    "MUSCLE",
    "FINDING",
    "EXERCISE",
    "TRIGGER",
    "RELIEVING_FACTOR",
}

# A measurement after a comma starts a new clause, not another item in the
# negated list ("without stiffness, 4/10").
_MEASUREMENT_LABELS = {"PAIN_SEVERITY", "DURATION", "FREQUENCY"}


def apply_negation(doc: Doc, candidates: list[Candidate]) -> None:
    """Set ``candidate.negated`` in place.

    Candidates are read from ``doc``; their character offsets locate them in
    the token stream.  The function is intentionally conservative: it only
    flips a candidate to negated, never the reverse, so a later cue cannot
    "un-negate" an earlier one.
    """
    if not candidates:
        return

    cue_tokens = _cue_tokens(doc)
    if not cue_tokens:
        return

    for cue, cue_end in cue_tokens:
        breaker = _scope_end(doc, cue_end, candidates)
        for cand in candidates:
            if cand.negated:
                continue
            tok = _first_token_at_or_after(doc, cand.start)
            if tok is None:
                continue
            if tok.i < cue_end:
                continue
            if tok.i > breaker:
                continue
            if cue.sent != tok.sent:
                continue
            cand.negated = True


def _cue_tokens(doc: Doc) -> list[tuple]:
    """Yield ``(cue_token, end_token_index)`` for each negation cue.

    Multi-word cues (``negative for``, ``absence of``, ``free of``) are
    consumed as a unit so the filler word does not break the scope.
    """
    cues = []
    lowered = [t.lower_ for t in doc]
    multiword = [c for c in lexicons.NEGATION_CUES if " " in c]
    single = {c for c in lexicons.NEGATION_CUES if " " not in c}

    i = 0
    while i < len(doc):
        matched = None
        for cue in multiword:
            parts = cue.split()
            if lowered[i: i + len(parts)] == parts:
                matched = (i, i + len(parts))
                break
        if matched:
            cues.append((doc[matched[0]], matched[1]))
            i = matched[1]
            continue
        if lowered[i] in single:
            cues.append((doc[i], i + 1))
        i += 1
    return cues


def _scope_end(doc: Doc, start: int, candidates: list[Candidate]) -> int:
    """Last token index a cue at ``start`` may reach."""
    sent = doc[start].sent
    limit = min(sent.end, start + _WINDOW)
    for i in range(start, limit):
        tok = doc[i]
        if tok.lower_ in lexicons.NEGATION_SCOPE_BREAKERS:
            return i - 1
        if tok.text == ";":
            return i - 1
        if tok.text == "," and not _comma_continues(doc, i, candidates, limit):
            # A comma is a weak breaker: it keeps scope for a coordinated list
            # ("no swelling, warmth or redness") but ends scope when the rest
            # of the clause is a new measurement ("without stiffness, 4/10").
            return i - 1
    return limit - 1


def _comma_continues(
    doc: Doc, comma_index: int, candidates: list[Candidate], limit: int
) -> bool:
    """True when a comma joins another item to the negated phrase."""
    next_index = comma_index + 1
    if next_index >= limit:
        return False
    nxt = doc[next_index]
    if nxt.lower_ in {"and", "or", "nor"}:
        return True
    if nxt.like_num:
        return False

    # A measurement anywhere after the comma means the comma starts a new
    # clause ("... without stiffness, pain rated 4/10").
    for cand in candidates:
        tok = _first_token_at_or_after(doc, cand.start)
        if tok is None or tok.i < next_index or tok.i >= limit:
            continue
        if cand.label in _MEASUREMENT_LABELS:
            return False

    # Otherwise continue only when the comma is followed by another clinical
    # item, which is what makes an enumeration such as "no swelling, warmth
    # or redness" negate every member.
    for cand in candidates:
        tok = _first_token_at_or_after(doc, cand.start)
        if (
            tok is not None
            and next_index <= tok.i <= next_index + 3
            and cand.label in _ENUMERATION_LABELS
        ):
            return True
    return False


def _first_token_at_or_after(doc: Doc, char_index: int):
    for tok in doc:
        if tok.idx >= char_index:
            return tok
    return None
