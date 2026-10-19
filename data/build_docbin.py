"""JSONL -> DocBin conversion for spaCy training and evaluation.

The generator's gold entities are converted directly into spaCy spans; there is
no alignment step because the generator recorded exact offsets.  Malformed
spans (should not happen) are counted and reported rather than silently
dropped.

    python data/build_docbin.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# The label vocabulary of the NER component: the six clinical entity types
# from the proposal.  Numeric measurements (pain, ROM, duration, frequency)
# stay rule-based and are not NER labels.
NER_LABELS = [
    "BODY_PART",
    "MUSCLE",
    "FINDING",
    "EXERCISE",
    "TRIGGER",
    "RELIEVING_FACTOR",
]

SPLITS = ["train", "dev", "generated_test"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build(nlp, notes: list[dict]) -> tuple[DocBin, Counter]:
    """Convert notes to a DocBin, keeping only ``NER_LABELS`` entities."""
    db = DocBin()
    stats = Counter()
    for note in notes:
        doc = nlp.make_doc(note["text"])
        ents = []
        for ent in note["entities"]:
            if ent["label"] not in NER_LABELS:
                stats[f"skipped:{ent['label']}"] += 1
                continue
            span = doc.char_span(ent["start"], ent["end"], label=ent["label"], alignment_mode="strict")
            if span is None:
                stats["alignment_failed"] += 1
                continue
            ents.append(span)
        # spaCy rejects overlapping spans; the generator never produces them,
        # but filter defensively so training does not die on one bad note.
        ents.sort(key=lambda s: (s.start, s.end))
        clean = []
        for span in ents:
            if clean and span.start < clean[-1].end:
                stats["overlap_dropped"] += 1
                continue
            clean.append(span)
        doc.ents = clean
        stats["docs"] += 1
        stats["entities"] += len(clean)
        db.add(doc)
    return db, stats


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank("en")

    for split in SPLITS:
        path = RAW / f"{split}.jsonl"
        if not path.exists():
            print(f"missing {path}; run data/generate_notes.py first", file=sys.stderr)
            return 1
        notes = load_jsonl(path)
        db, stats = build(nlp, notes)
        out = PROCESSED / f"{split}.spacy"
        db.to_disk(out)
        print(f"{split:16s} docs={stats['docs']:4d} entities={stats['entities']:4d} -> {out.relative_to(ROOT)}")
        for key, value in sorted(stats.items()):
            if key.startswith("skipped:"):
                print(f"    {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
