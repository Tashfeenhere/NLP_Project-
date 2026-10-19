"""Range-of-motion rules.

Three shapes, in the order the proposal lists them:

1. **Forward** -- ``knee flexion is limited to 90 degrees``
2. **Reversed** -- ``ROM: 90 deg flexion, knee``
3. **Qualitative** -- ``reduced lumbar flexion``, ``full shoulder flexion``

The movement is the anchor.  Multi-word movements (``internal rotation``,
``side flexion``, ``plantar flexion``) are matched as phrases, not as their
final token, so the canonical movement name is right and the span includes the
whole phrase.  The joint is bound with the dependency parser and a short
backward n-gram check (``compound`` / ``nmod`` attachment), which is what makes
multi-word joints such as ``rotator cuff`` and ``lumbar region`` bind correctly.

A ROM candidate is emitted per movement mention.  In a ROM span the joint is
*not* a separate entity -- the reconciler drops the contained body-part
candidate -- and the joint is carried in ``meta["body_part"]`` so field
assembly can fill it in.
"""
from __future__ import annotations

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token

from .. import lexicons
from ..candidates import Candidate
from ..normalize import canon_rom_value
from . import push

# Words that signal a limited (as opposed to full) range.
_RESTRICTIVE = {"limited", "reduced", "restricted", "decreased", "impaired", "poor", "loss"}

# Qualifier prefixes that may sit between a number and its unit.
_APPROX = {"approximately", "approx", "about", "roughly", "~", "around"}

_DEGREE_UNITS = {"deg", "degree", "degrees", "°"}

# Adjectives that carry ROM meaning on their own ("full flexion").
_QUALITY_ADJECTIVES = {"full", "normal", "improved", "painful", "blocked"}

# Measurement verbs do not change the fact that the next number is the value.
_MEASUREMENT_VERBS = {
    "measures",
    "measured",
    "measuring",
    "shows",
    "showed",
    "displays",
    "reaches",
    "reached",
    "achieves",
    "achieved",
    "remains",
    "remained",
}

_BODY_PART_SURFACES = {surface.lower() for surface in lexicons.BODY_PARTS}

# A joint binding is ``(start_token_index, surface)``.
Joint = tuple[int, str]


@Language.factory(
    "clinical_rom",
    default_config={},
    assigns=["doc._.rule_candidates"],
)
def make_rom_component(nlp, name):
    movement_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for surface in sorted(lexicons.MOVEMENTS, key=lambda s: (-len(s.split()), s)):
        movement_matcher.add("MOVEMENT", [nlp.make_doc(surface)])

    def component(doc: Doc) -> Doc:
        found = []
        seen: set[tuple[int, int]] = set()
        consumed: list[tuple[int, int]] = []  # movement token ranges already handled

        # Longest movement phrase first: "internal rotation" before "rotation".
        matches = sorted(movement_matcher(doc), key=lambda m: (-(m[2] - m[1]), m[1]))
        for _match_id, start, end in matches:
            if any(start >= cstart and end <= cend for cstart, cend in consumed):
                continue
            consumed.append((start, end))
            movement = doc[end - 1]
            span = _bind_rom(doc, movement, start, doc[start:end].text)
            if span is None:
                continue
            key = (span.start, span.end)
            if key in seen:
                continue
            seen.add(key)
            found.append(span)
        push(doc, found)
        return doc

    return component


def _bind_rom(
    doc: Doc,
    movement: Token,
    movement_start: int,
    movement_surface: str,
) -> Candidate | None:
    """Build the widest sensible ROM span around a movement phrase."""
    joint = _find_joint(doc, movement, movement_start)
    restrictor = _find_restrictor(doc, movement)

    # Reversed shape: the value sits to the left of the movement.
    numeric_left = _numeric_left_of(doc, movement, movement_start)
    # Forward shape: the value sits to the right of the movement.
    numeric_right = _numeric_right_of(doc, movement)

    number_tok = numeric_left or numeric_right
    value, quality = (None, None)
    if number_tok is not None:
        value, quality = canon_rom_value(number_tok.text)
        if value is None and quality is None:
            number_tok = None

    if value is None and restrictor is not None:
        quality = "limited" if restrictor.lower_ == "loss" else lexicons.ROM_QUALITY_WORDS.get(
            restrictor.lower_, restrictor.lower_
        )
    if value is None and quality is None:
        adjective = _quality_adjective(doc, movement, movement_start)
        if adjective is not None:
            quality = lexicons.ROM_QUALITY_WORDS.get(adjective, adjective)

    if value is None and quality is None:
        return None

    start_tok, end_tok = _span_bounds(doc, movement, movement_start, joint, number_tok, restrictor)
    span_text = doc[start_tok:end_tok].text
    return Candidate(
        start=doc[start_tok].idx,
        end=doc[end_tok - 1].idx + len(doc[end_tok - 1].text),
        label="ROM",
        text=span_text,
        source="rom",
        value={"value": value, "quality": quality, "unit": "deg" if value is not None else None},
        normalized=lexicons.canonical("MOVEMENT", movement_surface),
        meta={
            "movement": lexicons.canonical("MOVEMENT", movement_surface),
            "body_part": lexicons.canonical("BODY_PART", joint[1]) if joint is not None else None,
            "laterality": None,
        },
    )


def _find_joint(doc: Doc, movement: Token, movement_start: int) -> Joint | None:
    """Return ``(start_index, surface)`` of the joint attached to the movement."""
    # "rotator cuff flexion" -- the parser attaches ``cuff`` to ``flexion`` and
    # ``rotator`` to ``cuff``; check the n-gram ending at the child so
    # multi-word joints are recovered, not just the head token.
    for child in movement.children:
        if child.i >= movement_start:
            continue  # a token belonging to the movement phrase itself
        if child.dep_ in {"compound", "amod"}:
            hit = _body_part_ending_at(doc, child.i)
            if hit is not None:
                return hit
        if child.dep_ in {"prep", "nmod", "mark"} and child.lower_ == "of":
            for grand in child.children:
                hit = _body_part_ending_at(doc, grand.i)
                if hit is not None:
                    return hit

    # "neck side flexion" -- ``neck`` attaches to ``side`` (inside the movement
    # phrase), so the child check above cannot see it.  Scan just before the
    # phrase instead, stopping at punctuation or a clause verb.
    for back in range(1, 5):
        j = movement_start - back
        if j < 0:
            break
        tok = doc[j]
        if tok.is_punct or tok.pos_ in {"VERB", "AUX"}:
            break
        hit = _body_part_ending_at(doc, j)
        if hit is not None:
            return hit
    return None


def _body_part_ending_at(doc: Doc, end_index: int) -> Joint | None:
    """Longest body-part n-gram ending at ``end_index``."""
    for length in (4, 3, 2, 1):
        start = end_index - length + 1
        if start < 0:
            continue
        surface = doc[start: end_index + 1].text.lower()
        if surface in _BODY_PART_SURFACES:
            return (start, surface)
    return None


def _find_restrictor(doc: Doc, movement: Token) -> Token | None:
    """Look for "limited/reduced/restricted" near the movement."""
    window = list(doc[max(0, movement.i - 6): movement.i + 6])
    for tok in window:
        if tok.lower_ in _RESTRICTIVE:
            # "limited to 90 degrees" / "reduced flexion" both qualify, but a
            # restrictor belonging to a different clause ("limited mobility
            # otherwise, flexion 120") should not leak.  Require the restrictor
            # to share the sentence and be within 5 tokens.
            if abs(tok.i - movement.i) <= 5 and tok.sent == movement.sent:
                return tok
    return None


def _quality_adjective(doc: Doc, movement: Token, movement_start: int) -> str | None:
    """Return a meaning-carrying adjective such as ``full`` or ``painful``."""
    for child in movement.children:
        if child.i >= movement_start:
            continue
        if child.dep_ in {"amod", "compound"} and child.lower_ in _QUALITY_ADJECTIVES:
            return child.lower_
    for back in range(1, 4):
        j = movement_start - back
        if j < 0:
            break
        tok = doc[j]
        if tok.is_punct or tok.pos_ in {"VERB", "AUX"}:
            break
        if tok.lower_ in _QUALITY_ADJECTIVES:
            return tok.lower_
    return None


def _numeric_left_of(doc: Doc, movement: Token, movement_start: int) -> Token | None:
    """Reversed shape: "ROM: 90 deg flexion" -- the number is left of the movement."""
    for back in range(1, 4):
        j = movement.i - back
        if j < 0:
            break
        if movement_start <= j < movement.i:
            continue  # part of a multi-word movement ("side flexion")
        tok = doc[j]
        if tok.sent != movement.sent:
            break
        if tok.lower_ in _DEGREE_UNITS or tok.lower_ in _APPROX:
            continue
        if tok.like_num:
            # Require a degree unit in the two tokens to its right.
            right = {doc[k].lower_ for k in range(tok.i + 1, min(tok.i + 3, len(doc)))}
            if right & _DEGREE_UNITS:
                return tok
            return None
        if tok.lower_ in {"rom", ":", "r", "o", "m", ","}:
            continue
        return None
    return None


def _numeric_right_of(doc: Doc, movement: Token) -> Token | None:
    """Forward shape: "flexion is limited to 90 degrees"."""
    # Start *after* the movement: the anchor itself is usually a NOUN and would
    # otherwise terminate the scan on the first iteration.
    for forward in range(1, 8):
        j = movement.i + forward
        if j >= len(doc):
            break
        tok = doc[j]
        if tok.sent != movement.sent:
            break
        if tok.like_num:
            right = {doc[k].lower_ for k in range(tok.i + 1, min(tok.i + 3, len(doc)))}
            if right & _DEGREE_UNITS or canon_rom_value(tok.text)[0] is not None:
                return tok
            return None
        if tok.lower_ in _DEGREE_UNITS | _APPROX:
            continue
        if tok.lower_ in {
            "is", "was", "to", "of", "at", "limited", "reduced", "restricted",
            "decreased", "approximately", "about", ":", "-", "the",
        }:
            continue
        # "to" is the most common lead-in ("limited to 90") and is not
        # semantically informative, so it must not stop the scan.
        if tok.lower_ in {"to", "only", "just", "than"}:
            continue
        if tok.lower_ in _MEASUREMENT_VERBS:
            continue
        if tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADP", "DET", "PRON", "CCONJ", "SCONJ"}:
            return None
    return None


def _span_bounds(
    doc: Doc,
    movement: Token,
    movement_start: int,
    joint: Joint | None,
    number_tok: Token | None,
    restrictor: Token | None,
) -> tuple[int, int]:
    """Widen from the movement phrase to the natural phrase boundaries."""
    start = movement_start
    end = movement.i + 1

    # Extend left across the joint and any laterality.
    if joint is not None and joint[0] < start:
        start = joint[0]
    for back in range(1, 4):
        j = start - 1
        if j < 0:
            break
        tok = doc[j]
        if tok.lower_ in lexicons.LATERALITY_WORDS or tok.lower_ in {"of", "the"}:
            if tok.lower_ in {"of", "the"} and back > 1:
                break
            start = j
        else:
            break

    # Extend right to the number and its unit.
    if number_tok is not None and number_tok.i > movement.i:
        end = max(end, number_tok.i + 1)
        for fwd in range(1, 3):
            j = number_tok.i + fwd
            if j < len(doc) and doc[j].lower_ in _DEGREE_UNITS:
                end = max(end, j + 1)
                break
    if restrictor is not None and restrictor.i > movement.i:
        end = max(end, restrictor.i + 1)

    # Extend the left edge to include a number in the reversed shape.
    if number_tok is not None and number_tok.i < movement.i:
        low = number_tok.i
        # include an immediately preceding "ROM:" label but not a sentence verb
        if low - 1 >= 0 and doc[low - 1].lower_ in _APPROX:
            low -= 1
        start = min(start, low)
        # include the degree unit between the number and the movement
        end = max(end, number_tok.i + 1)
        for fwd in range(1, 3):
            j = number_tok.i + fwd
            if j < len(doc) and doc[j].lower_ in _DEGREE_UNITS:
                end = max(end, j + 1)
                break

    # Qualitative shape: the adjective sits to the left ("reduced flexion").
    for back in range(1, 3):
        j = start - 1
        if j < 0:
            break
        if doc[j].lower_ in lexicons.ROM_QUALITY_WORDS:
            start = j
        else:
            break

    return start, end
