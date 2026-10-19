"""A candidate entity proposed by any extraction layer.

Every layer -- gazetteer, matcher, ROM binder, NER -- emits the same record so
the reconciler and the UI can treat them uniformly.  ``source`` is what makes
the system explainable: the write-up and the UI both show where a value came
from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Precedence is a small integer: lower wins.  Ties are broken by longer span,
# then by earlier start, then by label name -- always deterministic.
PRECEDENCE = {
    "rom": 10,
    "severity": 20,
    "relieving": 25,
    "trigger": 26,
    "context": 30,
    "rule": 40,
    "gazetteer": 50,
    "ner": 60,
}

SOURCE_LABEL = {
    "rom": "rule",
    "severity": "rule",
    "relieving": "rule",
    "trigger": "rule",
    "context": "rule",
    "rule": "rule",
    "gazetteer": "gazetteer",
    "ner": "ner",
}


@dataclass
class Candidate:
    start: int
    end: int
    label: str
    text: str
    source: str  # a key of PRECEDENCE
    negated: bool = False
    value: object | None = None
    normalized: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def precedence(self) -> int:
        return PRECEDENCE.get(self.source, 99)

    @property
    def provenance(self) -> str:
        return SOURCE_LABEL.get(self.source, self.source)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Candidate") -> bool:
        return self.start < other.end and other.start < self.end

    def sort_key(self):
        return (self.precedence, -self.length, self.start, self.label)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        flag = "!" if self.negated else ""
        return f"<{self.label}{flag} {self.text!r} {self.start}:{self.end} {self.source}>"
