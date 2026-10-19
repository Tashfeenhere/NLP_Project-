"""Gazetteer layer for muscles, findings and exercises.

This is the part of the closed vocabulary the original rule set did not cover.
The matcher is token based (never substring based), so ``hip`` inside
``relationship`` and ``knee`` inside ``kneeling`` cannot match.

Exercises are matched twice:

* on ``LOWER`` surface forms -- this catches abbreviations and aliases
  (``SLR``, ``quad set``, ``step-down exercise``) which have no useful lemma,
* on ``LEMMA`` forms -- this is what turns ``raises`` into ``raise`` and
  ``clamshells`` into ``clamshell`` without listing every plural by hand.

Both matchers are consulted; the reconciler drops the duplicate span.  Lemma
patterns are built by running the basemodel's own tokeniser/tagger/lemmatiser
on each surface once at pipeline construction time, so the pattern lemmas are
exactly the lemmas the runtime doc will carry.
"""
from __future__ import annotations

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from .. import lexicons
from ..candidates import Candidate
from . import push

# Components whose output the lemma patterns depend on.  Everything else
# (including the rule layers already added) is disabled while patterns are
# lemmatised so construction cannot recurse into the component being built.
_LEMMA_PIPES = {"tok2vec", "tagger", "attribute_ruler", "lemmatizer"}


@Language.factory(
    "clinical_lexicon",
    default_config={},
    assigns=["doc._.rule_candidates"],
)
def make_lexicon_component(nlp, name):
    muscle_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    finding_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    exercise_lower = PhraseMatcher(nlp.vocab, attr="LOWER")
    exercise_lemma = PhraseMatcher(nlp.vocab, attr="LEMMA")

    _add_surfaces(nlp, muscle_matcher, "MUSCLE", lexicons.MUSCLES)
    _add_surfaces(nlp, finding_matcher, "FINDING", lexicons.RULE_FINDINGS)

    exercise_surfaces = set(lexicons.EXERCISES) | set(lexicons.EXERCISE_CANON)
    # Add regular plurals as explicit LOWER surfaces.  The lemma matcher below
    # is context-sensitive (``quadriceps`` is PROPN in isolation but NOUN in a
    # sentence, with different lemmas), so relying on lemmas alone missed
    # "quadriceps sets".  Explicit plurals are deterministic.
    exercise_surfaces |= {
        plural for plural in (_plural(surface) for surface in exercise_surfaces) if plural
    }
    _add_surfaces(nlp, exercise_lower, "EXERCISE", exercise_surfaces)

    enabled = [pipe for pipe in nlp.pipe_names if pipe in _LEMMA_PIPES]
    ordered = sorted(exercise_surfaces, key=lambda s: -len(s.split()))
    with nlp.select_pipes(enable=enabled):
        for surface, pattern in zip(ordered, nlp.pipe(ordered, batch_size=64)):
            if len(pattern):
                exercise_lemma.add("EXERCISE", [pattern])

    def component(doc: Doc) -> Doc:
        found: list[Candidate] = []
        for matcher, label, source in (
            (muscle_matcher, "MUSCLE", "gazetteer"),
            (finding_matcher, "FINDING", "gazetteer"),
            (exercise_lower, "EXERCISE", "rule"),
            (exercise_lemma, "EXERCISE", "rule"),
        ):
            for match_id, start, end in matcher(doc):
                span = doc[start:end]
                found.append(
                    Candidate(
                        start=span.start_char,
                        end=span.end_char,
                        label=label,
                        text=span.text,
                        source=source,
                        normalized=lexicons.canonical(label, span.text),
                    )
                )
        push(doc, found)
        return doc

    return component


def _plural(surface: str) -> str | None:
    """Regular plural of an exercise surface, or ``None`` when unchanged."""
    words = surface.split()
    if not words:
        return None
    last = words[-1]
    if last.endswith(("s", "x", "z", "ch", "sh")):
        return None
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def _add_surfaces(nlp, matcher: PhraseMatcher, label: str, surfaces) -> None:
    """Add every surface as its own token pattern, longest phrases first."""
    for surface in sorted(set(surfaces), key=lambda s: (-len(s.split()), s)):
        pattern = nlp.make_doc(surface)
        if len(pattern):
            matcher.add(label, [pattern])
