"""Fine-tune the NER component of ``en_core_web_sm`` in place.

Why this instead of ``spacy train``?  The runtime pipeline needs the parser and
sentence segmenter (ROM binding and negation scope use dependencies), and the
parser's tok2vec listener is width-bound to the base model.  Training a
standalone ``["tok2vec", "ner"]`` artefact and splicing it into the base model
breaks the parser with a shape mismatch.  Cloning the base model, freezing
everything except a fresh NER head, and training that head keeps a single
self-contained artefact with matching shapes.

This is the documented fallback in the implementation plan; the config route in
``training/configs/ner.cfg`` remains for reference and for training a
standalone artefact.

    python training/train_ner.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import compounding, minibatch

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "training" / "model"

NER_LABELS = ["BODY_PART", "MUSCLE", "FINDING", "EXERCISE", "TRIGGER", "RELIEVING_FACTOR"]


def load_corpus(nlp, path: Path) -> list:
    """Load a .spacy file as Examples against ``nlp``'s tokenizer."""
    from spacy.tokens import DocBin

    db = DocBin().from_disk(path)
    return list(db.get_docs(nlp.vocab))


def evaluate(nlp, examples: list) -> dict:
    scorer = nlp.evaluate(examples)
    # spaCy 3.8 returns a plain score dict from Language.evaluate; earlier
    # releases return a Scorer.  Support both so the script is not pinned to a
    # patch version.
    keys = ("ents_p", "ents_r", "ents_f")
    get = scorer.get if isinstance(scorer, dict) else lambda key, default=0.0: getattr(scorer, key, default)
    return {key: round(float(get(key, 0.0)), 4) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    spacy.util.fix_random_seed(args.seed)

    nlp = spacy.load("en_core_web_sm")

    # Replace the general-purpose NER head with a clinical one.  The base NER
    # is trained on its own label set, so merely adding labels to it would leave
    # transitions for labels we no longer need and no "O" transition for the
    # new schema.  A fresh entity recognizer avoids that.
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
    ner = nlp.add_pipe(
        "ner",
        name="ner",
        last=True,
        config={
            "moves": None,
            "update_with_oracle_cut_size": 100,
            "model": {"@architectures": "spacy.TransitionBasedParser.v2"},
        },
    )
    for label in NER_LABELS:
        ner.add_label(label)

    train_docs = load_corpus(nlp, PROCESSED / "train.spacy")
    dev_docs = load_corpus(nlp, PROCESSED / "dev.spacy")

    # NER's move set (including the crucial "O" transition) is only built by
    # ``initialize``, which needs a sample of the training data.  Updating
    # before this point raises E022 ("Could not find a transition with the name
    # 'O'").  The callable is re-invoked so the initialisation is resumable.
    # ``en_core_web_sm`` declares a lexeme-normalisation lookup table, but the
    # optional ``spacy-lookups-data`` package is not a runtime dependency.
    # Clearing the setting keeps ``initialize`` from requiring it at train time;
    # inference never consults those tables either.
    nlp.config["initialize"]["lookups"] = None

    train_examples = [Example(nlp.make_doc(doc.text), doc) for doc in train_docs]
    # Only initialise the new NER head.  ``select_pipes`` hides the frozen
    # components from ``Language.initialize`` too, so the pretrained tagger and
    # parser are not re-initialised against entity-only training docs.
    with nlp.select_pipes(enable=["ner"]):
        nlp.initialize(lambda: train_examples)

    optimizer = nlp.create_optimizer()
    optimizer.learn_rate = 0.001
    best_f1 = -1.0
    best_epoch = -1
    history = []

    # Only the NER component is updated.  ``tok2vec`` must still run its
    # forward pass so the NER's listener receives token vectors -- doing this
    # via ``Language.update`` with ``select_pipes(enable=["ner"])`` would skip
    # the encoder entirely and leave the head with no features.  Calling
    # ``tok2vec.update(..., sgd=None)`` computes the vectors and registers the
    # listener callback without touching the frozen encoder weights.
    tok2vec = nlp.get_pipe("tok2vec") if "tok2vec" in nlp.pipe_names else None
    print(f"training NER on {len(train_docs)} docs, dev={len(dev_docs)}, labels={len(NER_LABELS)}")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        examples = [Example(nlp.make_doc(doc.text), doc) for doc in train_docs]
        random.shuffle(examples)

        for batch in minibatch(examples, size=compounding(4.0, 32.0, 1.001)):
            if tok2vec is not None:
                tok2vec.update(batch, drop=0.0, sgd=None, losses={})
            ner.update(batch, sgd=optimizer, losses=losses, drop=args.dropout)

        dev_examples = [Example(nlp.make_doc(doc.text), doc) for doc in dev_docs]
        with nlp.select_pipes(enable=["tok2vec", "ner"]):
            metrics = evaluate(nlp, dev_examples)
        history.append({"epoch": epoch, **metrics, "loss": round(float(losses.get("ner", 0.0)), 2)})
        print(
            f"E{epoch:02d} loss={losses.get('ner', 0.0):8.2f} "
            f"P={metrics['ents_p']:.3f} R={metrics['ents_r']:.3f} F={metrics['ents_f']:.3f}"
        )

        if metrics["ents_f"] > best_f1:
            best_f1 = metrics["ents_f"]
            best_epoch = epoch
            OUTPUT.mkdir(parents=True, exist_ok=True)
            nlp.meta["name"] = "clinical_ner"
            nlp.meta["version"] = "1.0.0"
            nlp.meta["description"] = "en_core_web_sm with a physiotherapy NER head"
            nlp.to_disk(OUTPUT / "model-best")

    print(f"best epoch {best_epoch} F1={best_f1:.3f} -> {OUTPUT / 'model-best'}")
    (OUTPUT / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
