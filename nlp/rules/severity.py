"""Pain severity patterns.

Matches ``7/10``, ``7 / 10``, ``7 out of 10``, ``7-8/10``, ``7 to 8 out of 10``,
``pain 7`` (bare number directly after a pain word).  The numeric range is
validated in ``normalize.canon_pain``; an out-of-range score keeps its raw text
but reports ``value = None``.

Bare numbers are only accepted when a pain/finding word sits within two tokens
to the left, so "20 reps" or "3 sets" are never read as a pain score.
"""
from __future__ import annotations

import re

from spacy.language import Language
from spacy.tokens import Doc

from ..candidates import Candidate
from ..normalize import canon_pain
from . import push

_PAIN_CONTEXT = {"pain", "painful", "sore", "soreness", "severity", "rated", "rating", "score", "ache"}


@Language.factory(
    "clinical_severity",
    default_config={},
    assigns=["doc._.rule_candidates"],
)
def make_severity_component(nlp, name):
    def component(doc: Doc) -> Doc:
        found = []
        for token in doc:
            if not token.like_num:
                continue
            span = _severity_span(token)
            if span is None:
                continue
            raw = doc[span[0]:span[1]].text
            pain = canon_pain(raw)
            if pain is None:
                continue
            found.append(
                Candidate(
                    start=doc[span[0]].idx,
                    end=doc[span[1] - 1].idx + len(doc[span[1] - 1].text),
                    label="PAIN_SEVERITY",
                    text=raw,
                    source="severity",
                    value=pain,
                    normalized=str(pain["value"]) if pain["value"] is not None else None,
                )
            )
        push(doc, found)
        return doc

    return component


def _severity_span(token):
    """Return ``(start, end)`` token indices for a pain score starting here."""
    i = token.i
    # Shape A: 7/10 (tokenizer keeps "7/10" as one token) or "7 / 10".
    # ``_slash_span`` handles both; calling it unconditionally is what makes
    # the spaced form work when the tokenizer splits "5", "/", "10".
    span = _slash_span(token)
    if span is not None:
        return span

    # Shape B: "7 out of 10".
    if (
        i + 3 < len(token.doc)
        and token.doc[i + 1].lower_ == "out"
        and token.doc[i + 2].lower_ == "of"
        and token.doc[i + 3].lower_ in {"10", "ten"}
    ):
        if _has_pain_context(token, window=3):
            return (i, i + 4)

    # Shape C: "7-8/10" / "7 to 8 out of 10".
    if (
        i + 2 < len(token.doc)
        and token.doc[i + 1].lower_ in {"-", "to"}
        and token.doc[i + 2].like_num
    ):
        span = _slash_span(token.doc[i + 2])
        if span is not None:
            return (i, span[1])
        if token.doc[i + 1].lower_ == "to":
            j = i + 2
            if (
                j + 2 < len(token.doc)
                and token.doc[j + 1].lower_ == "out"
                and token.doc[j + 2].lower_ == "of"
            ):
                # Include the final "10" as well: "7 to 8 out of 10".
                return (i, j + 4)

    # Shape D: "pain 7" -- bare number with pain context immediately left.
    if _has_pain_context(token, window=2) and 0 <= int(token.text) <= 10:
        if token.doc[i + 1].text in {",", ".", ";", ":"} or i + 1 == len(token.doc):
            return (i, i + 1)

    return None


def _is_slash_form(token) -> bool:
    return bool(re.fullmatch(r"\d{1,2}\s*/\s*\d{1,2}", token.text))


def _slash_span(token):
    if _is_slash_form(token):
        return (token.i, token.i + 1)
    doc = token.doc
    i = token.i
    if (
        i + 2 < len(doc)
        and doc[i + 1].text == "/"
        and doc[i + 2].like_num
    ):
        return (i, i + 3)
    return None


def _has_pain_context(token, window: int) -> bool:
    doc = token.doc
    for back in range(1, window + 1):
        j = token.i - back
        if j < 0:
            break
        if doc[j].lower_ in _PAIN_CONTEXT:
            return True
    return False
