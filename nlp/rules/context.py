"""Context rules: triggers, relieving factors, duration, frequency.

The key insight is that ``walking`` is not an aggravating factor in
"better with walking" -- the *cue* decides.  So trigger extraction is
cue-anchored rather than gazetteer-only:

    aggravated by / worse with / worse on / brought on by / pain when / when I
        -> TRIGGER
    relieved by / better with / eases with / improves with / settles with
        -> RELIEVING_FACTOR

A resolvable cue span is emitted (and reconciled) only for the gazetteer
phrases it governs.  Bare occurrences of "walking" elsewhere ("walks daily")
are not tagged, which is deliberate: the field would be wrong more often than
right.

Duration and frequency are matched directly; they are unambiguous in notes.
"""
from __future__ import annotations

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from .. import lexicons
from ..candidates import Candidate
from ..normalize import canon_duration
from . import push

# cues that turn the following phrase into an aggravating factor
_AGGRAVATING_CUES = {
    "after",
    "aggravated",
    "aggravates",
    "aggravating",
    "worse",
    "worsened",
    "worsening",
    "triggered",
    "triggers",
    "provoked",
    "provokes",
    "exacerbated",
    "exacerbates",
    "brought",
    "painful",
    "uncomfortable",
}

# Functions that can follow a trigger phrase and reveal it as aggravating:
# "prolonged sitting makes it worse", "stairs bring it on".
_POSTPOSED_TRIGGER_VERBS = {
    "aggravate",
    "aggravates",
    "brings",
    "cause",
    "causes",
    "exacerbate",
    "exacerbates",
    "hurt",
    "hurts",
    "provoke",
    "provokes",
    "reproduce",
    "reproduces",
    "trigger",
    "triggers",
    "worsen",
    "worsens",
}

# cues that turn the following phrase into a relieving factor
_RELIEVING_CUES = {
    "relieved",
    "relieves",
    "relieving",
    "better",
    "eased",
    "eases",
    "improved",
    "improves",
    "settles",
    "settled",
    "helped",
    "helps",
}

# a cue may also appear after the factor: "walking helps", "rest eases it"
_POSTPOSED_RELIEVING = {"helps", "help", "eases", "eased", "relieves", "relieved", "settles", "settled"}

# filler tokens allowed between a cue and the phrase it governs
_FILLERS = {"with", "by", "on", "when", "during", "upon", "at", "to", "the", "a", "an", "it", "this", "that", "of", "and", "my"}

_DURATION_HINT = {"for", "since", "over", "past", "last", "x"}


@Language.factory(
    "clinical_context",
    default_config={},
    assigns=["doc._.rule_candidates"],
)
def make_context_component(nlp, name):
    trigger_phrase = PhraseMatcher(nlp.vocab, attr="LOWER")
    for surface in sorted(lexicons.TRIGGERS, key=lambda s: -len(s.split())):
        trigger_phrase.add("TRIGGER", [nlp.make_doc(surface)])

    relieving_phrase = PhraseMatcher(nlp.vocab, attr="LOWER")
    for surface in sorted(lexicons.RELIEVING_FACTORS, key=lambda s: -len(s.split())):
        relieving_phrase.add("RELIEVING_FACTOR", [nlp.make_doc(surface)])

    frequency_phrase = PhraseMatcher(nlp.vocab, attr="LOWER")
    for surface in sorted(lexicons.FREQUENCY_TERMS, key=lambda s: -len(s.split())):
        frequency_phrase.add("FREQUENCY", [nlp.make_doc(surface)])

    def trigger_spans(doc: Doc):
        for match_id, start, end in trigger_phrase(doc):
            span = doc[start:end]
            yield span.start_char, span.end_char, span.text

    def relieving_spans(doc: Doc):
        for match_id, start, end in relieving_phrase(doc):
            span = doc[start:end]
            yield span.start_char, span.end_char, span.text

    def component(doc: Doc) -> Doc:
        found: list[Candidate] = []

        for start, end, text in trigger_spans(doc):
            if _governed_by(doc, start, end, _AGGRAVATING_CUES) or _followed_by_postposed_trigger(
                doc, end
            ):
                found.append(
                    Candidate(start, end, "TRIGGER", text, source="trigger",
                              normalized=lexicons.canonical("TRIGGER", text))
                )

        for start, end, text in relieving_spans(doc):
            if _governed_by(doc, start, end, _RELIEVING_CUES) or _followed_by_postposed(doc, end):
                found.append(
                    Candidate(start, end, "RELIEVING_FACTOR", text, source="relieving",
                              normalized=lexicons.canonical("RELIEVING_FACTOR", text))
                )

        for match_id, start, end in frequency_phrase(doc):
            span = doc[start:end]
            found.append(
                Candidate(
                    span.start_char,
                    span.end_char,
                    "FREQUENCY",
                    span.text,
                    source="rule",
                    normalized=lexicons.canonical("FREQUENCY", span.text),
                )
            )

        found.extend(_durations(doc))
        push(doc, found)
        return doc

    return component


def _tokens_between(doc: Doc, start_char: int, end_char: int):
    return [t for t in doc if start_char <= t.idx < end_char]


def _governed_by(doc: Doc, start: int, end: int, cues: set[str]) -> bool:
    """True when a cue in ``cues`` sits to the left within the same sentence."""
    span_tokens = _tokens_between(doc, start, end)
    if not span_tokens:
        return False
    first = span_tokens[0]
    for back in range(1, 6):
        j = first.i - back
        if j < 0:
            break
        tok = doc[j]
        if tok.sent != first.sent:
            break
        if tok.lower_ in cues:
            return True
        if tok.lower_ in _FILLERS or tok.is_punct:
            continue
        # Any other content word ends the cue's reach: "worse with walking but
        # no swelling" must not make "swelling" a trigger.
        if tok.pos_ in {"VERB", "NOUN", "ADJ", "ADV"} and tok.lower_ not in _FILLERS:
            return False
    return False


def _followed_by_postposed(doc: Doc, end: int) -> bool:
    """True for "walking helps" / "rest eases it"."""
    tokens = [t for t in doc if t.idx >= end]
    if not tokens:
        return False
    nxt = tokens[0]
    if nxt.lower_ == "helps" or nxt.lower_ in _POSTPOSED_RELIEVING:
        return True
    return False


def _followed_by_postposed_trigger(doc: Doc, end: int) -> bool:
    """True for "prolonged sitting makes it worse" / "stairs bring it on"."""
    tokens = [t for t in doc if t.idx >= end]
    lowered = [t.lower_ for t in tokens[:5]]
    if not lowered:
        return False
    if lowered[0] in _POSTPOSED_TRIGGER_VERBS:
        return True
    joined = " ".join(lowered)
    return joined.startswith(
        (
            "makes it worse",
            "makes the pain worse",
            "brings it on",
            "causes pain",
            "causes the pain",
            "reproduces the pain",
            "are difficult",
            "is difficult",
        )
    )


def _durations(doc: Doc) -> list[Candidate]:
    found = []
    for token in doc:
        if not (token.like_num or token.lower_ in lexicons.NUMBER_WORDS):
            continue
        # number + time unit
        if token.i + 1 < len(doc):
            unit = doc[token.i + 1]
            if unit.lower_ in lexicons.TIME_UNITS:
                raw = doc[token.i: token.i + 2].text
                value = canon_duration(raw)
                if value:
                    found.append(
                        Candidate(
                            doc[token.i].idx,
                            unit.idx + len(unit.text),
                            "DURATION",
                            raw,
                            source="rule",
                            normalized=value,
                        )
                    )
                continue
        # "since March"
    for i, token in enumerate(doc):
        if token.lower_ == "since" and i + 1 < len(doc) and doc[i + 1].is_title:
            raw = doc[i: i + 2].text
            value = canon_duration(raw)
            if value:
                found.append(
                    Candidate(
                        token.idx,
                        doc[i + 1].idx + len(doc[i + 1].text),
                        "DURATION",
                        raw,
                        source="rule",
                        normalized=value,
                    )
                )
    return found
