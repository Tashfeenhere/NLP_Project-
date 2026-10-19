"""Body part and laterality rules.

Two matchers:

* ``PhraseMatcher`` over the body-part gazetteer -- token based, so "hip" in
  "relationship" never matches and multi-word parts are matched as one span.
* ``Matcher`` for laterality, producing ``LATERALITY`` candidates that the
  reconciler binds to whichever ``BODY_PART`` they attach to.

Laterality phrasing handled:

    right knee          left shoulder        bilateral knees
    knee (right)        both knees           R knee / Lt shoulder
    right-sided knee
"""
from __future__ import annotations

from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc

from .. import lexicons
from ..candidates import Candidate
from . import push

# Components whose output the lemma patterns depend on (all pre-date this rule
# layer in the pipeline).  Everything else is disabled during construction.
_LEMMA_PIPES = {"tok2vec", "tagger", "attribute_ruler", "lemmatizer"}


@Language.factory(
    "clinical_body_parts",
    default_config={},
    assigns=["doc._.rule_candidates"],
)
def make_body_part_component(nlp, name):
    """Gazetteer pass for BODY_PART, matched on lower-cased token sequences."""
    phrase = PhraseMatcher(nlp.vocab, attr="LOWER")
    # Longest patterns first so multi-word parts ("cervical spine") are added
    # before their shorter neighbours; PhraseMatcher itself returns all
    # matches and the reconciler keeps the longest.
    for surface in sorted(lexicons.BODY_PARTS, key=lambda s: -len(s.split())):
        phrase.add("BODY_PART", [nlp.make_doc(surface)])

    # A lemma pass catches plural surface forms ("both knees", "shoulders")
    # without listing every plural in the lexicon.
    phrase_lemma = PhraseMatcher(nlp.vocab, attr="LEMMA")
    enabled = [pipe for pipe in nlp.pipe_names if pipe in _LEMMA_PIPES]
    surfaces = sorted(lexicons.BODY_PARTS, key=lambda s: -len(s.split()))
    with nlp.select_pipes(enable=enabled):
        for surface, pattern in zip(surfaces, nlp.pipe(surfaces, batch_size=64)):
            if len(pattern):
                phrase_lemma.add("BODY_PART", [pattern])

    def component(doc: Doc) -> Doc:
        found = []
        for matcher in (phrase, phrase_lemma):
            for match_id, start, end in matcher(doc):
                span = doc[start:end]
                found.append(
                    Candidate(
                        start=span.start_char,
                        end=span.end_char,
                        label="BODY_PART",
                        text=span.text,
                        source="gazetteer",
                    )
                )
        push(doc, found)
        return doc

    return component


@Language.factory(
    "clinical_laterality",
    default_config={},
    assigns=["doc._.rule_candidates"],
)
def make_laterality_component(nlp, name):
    matcher = Matcher(nlp.vocab)
    lat = [{"LOWER": {"IN": sorted(lexicons.LATERALITY_WORDS)}}]

    matcher.add("LATERALITY", [lat])
    matcher.add(
        "LATERALITY",
        [[{"LOWER": {"IN": sorted(lexicons.LATERALITY_WORDS)}}, {"TEXT": {"IN": ["-", "–"]}}, {"LOWER": "sided"}]],
    )
    matcher.add("LATERALITY", [[{"LOWER": "both"}, {"LOWER": "sides"}]])

    def component(doc: Doc) -> Doc:
        found = []
        for match_id, start, end in matcher(doc):
            span = doc[start:end]
            value = _lat_value(span.text)
            if value is None:
                continue
            found.append(
                Candidate(
                    start=span.start_char,
                    end=span.end_char,
                    label="LATERALITY",
                    text=span.text,
                    source="rule",
                    value=value,
                )
            )
        push(doc, found)
        return doc

    return component


def _lat_value(text: str) -> str | None:
    first = text.lower().split()[0].strip(",.")
    return lexicons.LATERALITY_WORDS.get(first)


def bind_laterality(doc: Doc) -> dict[int, str]:
    """Map a BODY_PART candidate's start offset to its laterality.

    Binding rules, applied in order:

    1. A laterality token within two characters of the part's left edge
       (``right knee``, ``right-sided knee``).
    2. A laterality token within three characters after the part, which
       covers ``knee (right)``.
    3. Exactly one part and one unclaimed laterality in the same sentence --
       ``knee pain, worse on the right`` still binds.
    4. Otherwise unbound: laterality stays ``None`` rather than being guessed.
    """
    parts = [c for c in doc._.rule_candidates if c.label == "BODY_PART"]
    lats = [c for c in doc._.rule_candidates if c.label == "LATERALITY"]
    if not parts or not lats:
        return {}

    bound: dict[int, str] = {}
    used: set[int] = set()

    for part in sorted(parts, key=lambda c: c.start):
        best = None
        for lat in lats:
            if lat.start >= part.end:
                continue
            if part.start - lat.end <= 2:
                if best is None or lat.start > best.start:
                    best = lat
        if best is not None and id(best) not in used:
            bound[part.start] = best.value
            used.add(id(best))
            continue

        for lat in lats:
            if lat.start < part.end:
                continue
            if lat.start - part.end <= 3 and id(lat) not in used:
                bound[part.start] = lat.value
                used.add(id(lat))
                break

    # Rule 3, per real spaCy sentence.
    for sent in doc.sents:
        in_sent = [p for p in parts if sent.start_char <= p.start < sent.end_char]
        avail = [l for l in lats if sent.start_char <= l.start < sent.end_char and id(l) not in used]
        if len(in_sent) == 1 and len(avail) == 1 and in_sent[0].start not in bound:
            bound[in_sent[0].start] = avail[0].value
            used.add(id(avail[0]))

    return bound
