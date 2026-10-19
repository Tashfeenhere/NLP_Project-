"""Rule component registry.

All rule components are decorated with ``@Language.factory`` here, in one
module, so their ordering and their shared ``Doc`` extension live in one place.
Importing this package registers every factory with spaCy.

Components run in this order (see ``nlp/pipeline.py``):

    clinical_body_parts -> clinical_laterality -> clinical_severity
    -> clinical_rom -> clinical_context -> ner -> clinical_reconcile
"""
from __future__ import annotations

from spacy.language import Language
from spacy.tokens import Doc

from ..candidates import Candidate

# ``Doc._.rule_candidates`` holds everything the rule layers propose, before
# the NER component runs and before reconciliation.  ``default=[]`` gives each
# doc its own list (spaCy calls the default per instance).
if not Doc.has_extension("rule_candidates"):
    Doc.set_extension("rule_candidates", default=[])


def push(doc: Doc, candidates: list[Candidate]) -> None:
    doc._.rule_candidates.extend(candidates)


# Importing the submodules registers their factories.
from . import body_part, context, lexicon, negation, rom, severity  # noqa: E402,F401

__all__ = [
    "body_part",
    "context",
    "lexicon",
    "negation",
    "rom",
    "severity",
    "push",
    "Language",
]
