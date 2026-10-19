"""Evaluation harness: entity-level, field-level, and the rules-vs-NER ablation.

    python evaluation/evaluate.py                 # both splits, both regimes
    python evaluation/evaluate.py --split handwritten --ablation

Artefacts (under ``evaluation/``):

    results/ner_report.txt       human-readable summary
    results/per_label.csv        entity P/R/F1 per label and split
    results/field_accuracy.csv   exact-match accuracy per field
    results/ablation.csv         rules-only vs rules+NER headline numbers
    results/entity_f1_by_label.png
    errors/*.csv                 false positives / negatives and field misses

The generated test split shares templates with training, so its scores are
optimistic by construction.  The hand-written set is the honest test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp import lexicons  # noqa: E402
from nlp.normalize import canon_duration, canon_laterality, canon_pain  # noqa: E402
from nlp.pipeline import build_pipeline  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "evaluation" / "results"
ERRORS = ROOT / "evaluation" / "errors"

SPLITS = {
    "generated": RAW / "generated_test.jsonl",
    "handwritten": ROOT / "data" / "hand_written_test.jsonl",
}

FIELD_KEYS = [
    "body_part",
    "laterality",
    "pain_severity",
    "rom",
    "triggers",
    "relieving_factors",
    "muscles",
    "findings",
    "exercises",
    "duration",
    "frequency",
]

LIST_FIELDS = [
    "triggers",
    "relieving_factors",
    "muscles",
    "findings",
    "exercises",
]

# LATERALITY is a gold label but is consumed into a field by the reconciler,
# so it is not scored as a standalone entity.
SCORED_LABELS = [
    "BODY_PART",
    "PAIN_SEVERITY",
    "ROM",
    "MUSCLE",
    "FINDING",
    "EXERCISE",
    "TRIGGER",
    "RELIEVING_FACTOR",
    "DURATION",
    "FREQUENCY",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dedupe(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _canon(label: str, text: str) -> str:
    return lexicons.canonical(label, text)


def _movement_from_text(text: str) -> str | None:
    lowered = text.lower()
    for surface in sorted(lexicons.MOVEMENTS, key=len, reverse=True):
        if surface in lowered:
            return lexicons.canonical("MOVEMENT", surface)
    return None


def gold_fields(entities: list[dict], text: str) -> dict:
    """Derive the expected ``ClinicalRecord`` from gold entities."""
    positive = [e for e in entities if not e.get("negated")]

    body_parts = [e for e in positive if e["label"] == "BODY_PART"]
    laterality_ents = [e for e in positive if e["label"] == "LATERALITY"]
    pain_ents = [e for e in positive if e["label"] == "PAIN_SEVERITY"]
    rom_ents = [e for e in positive if e["label"] == "ROM"]

    body_part = _canon("BODY_PART", body_parts[0]["text"]) if body_parts else None
    if body_part is None:
        for rom in rom_ents:
            if rom.get("body_part"):
                body_part = _canon("BODY_PART", rom["body_part"])
                break

    laterality = None
    if laterality_ents:
        laterality = canon_laterality(laterality_ents[0]["text"])
    if laterality is None:
        for rom in rom_ents:
            if rom.get("laterality"):
                laterality = canon_laterality(rom["laterality"])
                break

    pain = None
    if pain_ents:
        parsed = canon_pain(pain_ents[0]["text"])
        if parsed:
            pain = parsed

    rom_rows = []
    for rom in rom_ents:
        movement = rom.get("movement") or _movement_from_text(rom["text"])
        value = rom.get("value")
        unit = rom.get("unit")
        quality = rom.get("quality")
        if value is None and quality is None:
            # Marker may omit metadata; recover it from the span text.
            from nlp.normalize import canon_rom_value

            value, quality = canon_rom_value(rom["text"])
        rom_rows.append(
            {
                "movement": _canon("MOVEMENT", movement) if movement else rom["text"].lower(),
                "value": value,
                "unit": unit,
                "quality": quality,
            }
        )

    def texts(label: str) -> list[str]:
        return dedupe(
            [_canon(label, e["text"]) for e in positive if e["label"] == label]
        )

    durations = [e["text"] for e in positive if e["label"] == "DURATION"]
    frequencies = [e["text"] for e in positive if e["label"] == "FREQUENCY"]

    return {
        "body_part": body_part,
        "laterality": laterality,
        "pain_severity": pain,
        "rom": rom_rows,
        "triggers": texts("TRIGGER"),
        "relieving_factors": texts("RELIEVING_FACTOR"),
        "muscles": texts("MUSCLE"),
        "findings": texts("FINDING"),
        "exercises": texts("EXERCISE"),
        "duration": canon_duration(durations[0]) if durations else None,
        "frequency": _canon("FREQUENCY", frequencies[0]) if frequencies else None,
    }


def entity_key(entity: dict) -> tuple[int, int, str]:
    return (entity["start"], entity["end"], entity["label"])


def score_entities(gold: list[dict], predicted: list[dict]) -> dict:
    """Micro + per-label exact-span P/R/F1, ignoring negated and LATERALITY."""
    gold_set = {
        entity_key(e)
        for e in gold
        if not e.get("negated") and e["label"] in SCORED_LABELS
    }
    pred_set = {
        entity_key(e)
        for e in predicted
        if not e.get("negated") and e["label"] in SCORED_LABELS
    }

    labels = sorted({label for _, _, label in gold_set | pred_set})
    per_label = {}
    for label in labels:
        gold_l = {key for key in gold_set if key[2] == label}
        pred_l = {key for key in pred_set if key[2] == label}
        tp = len(gold_l & pred_l)
        fp = len(pred_l - gold_l)
        fn = len(gold_l - pred_l)
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
            "support": len(gold_l),
        }

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    micro = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "support": len(gold_set),
    }
    macro = _macro_average(per_label)
    return {"micro": micro, "macro": macro, "per_label": per_label}


def _macro_average(per_label: dict) -> dict:
    """Unweighted mean of P/R/F1 over labels that occur in the gold data."""
    labels = [
        label for label, scores in per_label.items() if scores["support"] > 0
    ]
    if not labels:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "labels": 0,
            "support": 0,
        }
    return {
        "precision": sum(per_label[label]["precision"] for label in labels) / len(labels),
        "recall": sum(per_label[label]["recall"] for label in labels) / len(labels),
        "f1": sum(per_label[label]["f1"] for label in labels) / len(labels),
        "labels": len(labels),
        "support": sum(per_label[label]["support"] for label in labels),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _normalize_rom(rows: list[dict]) -> list[tuple]:
    out = [
        (
            (row.get("movement") or "").casefold(),
            row.get("value"),
            (row.get("quality") or "").casefold() or None,
        )
        for row in rows
    ]
    return sorted(out, key=lambda item: (item[0], str(item[1]), str(item[2])))


def _normalize_list(values: list[str]) -> list[str]:
    return sorted(value.casefold() for value in values)


def field_matches(key: str, gold: dict, predicted: dict) -> bool:
    if key == "pain_severity":
        gold_pain = gold.get(key)
        pred_pain = predicted.get(key)
        if gold_pain is None or pred_pain is None:
            return gold_pain is None and pred_pain is None
        return (gold_pain.get("value"), gold_pain.get("range")) == (
            pred_pain.get("value"),
            pred_pain.get("range"),
        )
    if key == "rom":
        return _normalize_rom(gold.get(key) or []) == _normalize_rom(predicted.get(key) or [])
    if key in LIST_FIELDS:
        return _normalize_list(gold.get(key) or []) == _normalize_list(predicted.get(key) or [])
    return (gold.get(key) or None) == (predicted.get(key) or None)


def score_fields(gold: dict, predicted: dict) -> dict[str, bool]:
    return {key: field_matches(key, gold, predicted) for key in FIELD_KEYS}


def run_split(pipeline, split: str, notes: list[dict]) -> dict:
    entity_totals = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    per_label_totals: dict[str, dict] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    )
    field_correct = {key: 0 for key in FIELD_KEYS}
    field_errors = []
    entity_errors = []

    for index, note in enumerate(notes):
        gold = gold_fields(note["entities"], note["text"])
        result = pipeline.analyze(note["text"])
        predicted = result["fields"]

        entity_scores = score_entities(note["entities"], result["entities"])
        for key in entity_totals:
            entity_totals[key] += entity_scores["micro"][key]
        for label, scores in entity_scores["per_label"].items():
            for key in per_label_totals[label]:
                per_label_totals[label][key] += scores[key]

        matches = score_fields(gold, predicted)
        for key, matched in matches.items():
            if matched:
                field_correct[key] += 1
            else:
                field_errors.append(
                    {
                        "split": split,
                        "note_index": index,
                        "field": key,
                        "gold": json.dumps(gold.get(key), ensure_ascii=False),
                        "predicted": json.dumps(predicted.get(key), ensure_ascii=False),
                        "note": note["text"],
                    }
                )

        gold_set = {
            entity_key(e)
            for e in note["entities"]
            if not e.get("negated") and e["label"] in SCORED_LABELS
        }
        pred_map = {
            entity_key(e): e
            for e in result["entities"]
            if not e.get("negated") and e["label"] in SCORED_LABELS
        }
        for key in sorted(gold_set - set(pred_map)):
            entity_errors.append(
                {
                    "split": split,
                    "error": "FN",
                    "label": key[2],
                    "start": key[0],
                    "end": key[1],
                    "text": note["text"][key[0]: key[1]],
                    "note": note["text"],
                }
            )
        for key in sorted(set(pred_map) - gold_set):
            entity = pred_map[key]
            entity_errors.append(
                {
                    "split": split,
                    "error": "FP",
                    "label": entity["label"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["text"],
                    "note": note["text"],
                }
            )

    per_label = {}
    for label, totals in per_label_totals.items():
        per_label[label] = {
            **totals,
            "precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
            "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
            "f1": _f1(totals["tp"], totals["fp"], totals["fn"]),
        }

    field_accuracy = {
        key: (field_correct[key], len(notes), field_correct[key] / len(notes) if notes else 0.0)
        for key in FIELD_KEYS
    }
    overall_field_accuracy = (
        sum(field_correct.values()) / (len(notes) * len(FIELD_KEYS)) if notes else 0.0
    )

    return {
        "split": split,
        "n_notes": len(notes),
        "entity_micro": {
            **entity_totals,
            "precision": _ratio(entity_totals["tp"], entity_totals["tp"] + entity_totals["fp"]),
            "recall": _ratio(entity_totals["tp"], entity_totals["tp"] + entity_totals["fn"]),
            "f1": _f1(entity_totals["tp"], entity_totals["fp"], entity_totals["fn"]),
        },
        "entity_macro": _macro_average(per_label),
        "per_label": per_label,
        "field_accuracy": field_accuracy,
        "overall_field_accuracy": overall_field_accuracy,
        "field_errors": field_errors,
        "entity_errors": entity_errors,
        "model_version": pipeline.model_version,
    }


def write_error_csvs(regime: str, results: dict[str, dict]) -> None:
    ERRORS.mkdir(parents=True, exist_ok=True)
    for split, result in results.items():
        with (ERRORS / f"{split}_{regime}_entities.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "error", "label", "start", "end", "text", "note"],
            )
            writer.writeheader()
            writer.writerows(result["entity_errors"])
        with (ERRORS / f"{split}_{regime}_fields.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "note_index", "field", "gold", "predicted", "note"],
            )
            writer.writeheader()
            writer.writerows(result["field_errors"])


def write_metric_csvs(runs: dict[str, dict], ablation: dict[str, dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "per_label.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["regime", "split", "label", "precision", "recall", "f1", "support", "tp", "fp", "fn"]
        )
        for regime, splits in runs.items():
            for split, result in splits.items():
                for label in SCORED_LABELS:
                    scores = result["per_label"].get(label)
                    if scores is None:
                        continue
                    writer.writerow(
                        [
                            regime,
                            split,
                            label,
                            round(scores["precision"], 4),
                            round(scores["recall"], 4),
                            round(scores["f1"], 4),
                            scores["support"],
                            scores["tp"],
                            scores["fp"],
                            scores["fn"],
                        ]
                    )

    with (RESULTS / "field_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["regime", "split", "field", "correct", "total", "accuracy"])
        for regime, splits in runs.items():
            for split, result in splits.items():
                for key in FIELD_KEYS:
                    correct, total, accuracy = result["field_accuracy"][key]
                    writer.writerow([regime, split, key, correct, total, round(accuracy, 4)])
                writer.writerow(
                    [
                        regime,
                        split,
                        "ALL (mean over fields)",
                        "",
                        result["n_notes"],
                        round(result["overall_field_accuracy"], 4),
                    ]
                )

    with (RESULTS / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "regime",
                "entity_micro_f1",
                "entity_macro_f1",
                "field_accuracy",
                "n_notes",
            ]
        )
        for split in SPLITS:
            for regime in ("rules_only", "rules_plus_ner"):
                if regime not in ablation:
                    continue
                result = ablation[regime].get(split) if isinstance(ablation[regime], dict) else None
                if result is None:
                    continue
                writer.writerow(
                    [
                        split,
                        regime,
                        round(result["entity_micro"]["f1"], 4),
                        round(result["entity_macro"]["f1"], 4),
                        round(result["overall_field_accuracy"], 4),
                        result["n_notes"],
                    ]
                )


def make_chart(runs: dict[str, dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        print("matplotlib not available; skipping chart")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    labels = SCORED_LABELS
    generated = [
        runs["rules_plus_ner"]["generated"]["per_label"].get(label, {}).get("f1", 0.0)
        for label in labels
    ]
    handwritten = [
        runs["rules_plus_ner"]["handwritten"]["per_label"].get(label, {}).get("f1", 0.0)
        for label in labels
    ]
    x = range(len(labels))
    width = 0.38
    figure, axes = plt.subplots(figsize=(11, 4.8))
    axes.bar([i - width / 2 for i in x], generated, width, label="generated test")
    axes.bar([i + width / 2 for i in x], handwritten, width, label="hand-written test")
    axes.set_xticks(list(x))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel("entity F1")
    axes.set_ylim(0, 1.05)
    axes.set_title("Entity F1 by label — rules + NER")
    axes.legend()
    figure.tight_layout()
    figure.savefig(RESULTS / "entity_f1_by_label.png", dpi=150)
    plt.close(figure)


def format_report(runs: dict[str, dict], ablation: dict[str, dict]) -> str:
    lines = []
    lines.append("Clinical Note Analyzer — evaluation report")
    lines.append("=" * 55)
    lines.append("")
    lines.append(
        "Generated test notes share templates with training; treat those scores "
        "as an upper bound. The hand-written split is the honest test."
    )
    lines.append("")
    for regime in ("rules_only", "rules_plus_ner"):
        if regime not in ablation:
            continue
        lines.append(f"Regime: {regime}")
        for split in SPLITS:
            result = runs[regime].get(split)
            if result is None:
                continue
            micro = result["entity_micro"]
            macro = result["entity_macro"]
            lines.append(
                f"  {split:<12} n={result['n_notes']:<4} "
                f"entity P={micro['precision']:.3f} R={micro['recall']:.3f} "
                f"F1={micro['f1']:.3f}   "
                f"macro P={macro['precision']:.3f} R={macro['recall']:.3f} "
                f"F1={macro['f1']:.3f}   "
                f"field accuracy={result['overall_field_accuracy']:.3f}"
            )
            for label in SCORED_LABELS:
                scores = result["per_label"].get(label)
                if scores is None:
                    continue
                lines.append(
                    f"      {label:<18} P={scores['precision']:.3f} "
                    f"R={scores['recall']:.3f} F1={scores['f1']:.3f} "
                    f"(support {scores['support']})"
                )
            lines.append("")
            lines.append("      field-level exact match:")
            for key in FIELD_KEYS:
                correct, total, accuracy = result["field_accuracy"][key]
                lines.append(f"        {key:<18} {correct}/{total} = {accuracy:.3f}")
            lines.append("")

    if "rules_plus_ner" in ablation and "rules_only" in ablation:
        lines.append("Ablation (why the model earns its place)")
        lines.append("-" * 55)
        for split in SPLITS:
            base = ablation["rules_only"].get(split)
            full = ablation["rules_plus_ner"].get(split)
            if base is None or full is None:
                continue
            lines.append(
                f"{split:<12} rules-only F1={base['entity_micro']['f1']:.3f} "
                f"field={base['overall_field_accuracy']:.3f}  ->  "
                f"rules+NER F1={full['entity_micro']['f1']:.3f} "
                f"field={full['overall_field_accuracy']:.3f}"
            )
        lines.append("")

    lines.append("Notes")
    lines.append("-" * 55)
    lines.append(
        "* Every entity is scored by exact character span and label; the "
        "reconciler guarantees non-overlap."
    )
    lines.append(
        "* Negated gold and negated predictions are excluded from entity scoring; "
        "they are reported separately by the app."
    )
    lines.append(
        "* ROM is compared as (movement, value, quality). Laterality is scored as "
        "a field, not as a standalone entity."
    )
    lines.append(
        "* No inter-annotator agreement claim: training data is synthetic and the "
        "hand-written set has a single annotator."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["all", "generated", "handwritten", "test"],
        default="all",
        help=(
            "which held-out split to evaluate ('test' is an alias for "
            "'generated')"
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="also evaluate the rules-only pipeline (default: on with --split all)",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="skip the rules-only regime",
    )
    args = parser.parse_args()

    split = "generated" if args.split == "test" else args.split
    splits = list(SPLITS) if split == "all" else [split]
    notes = {split: load_jsonl(SPLITS[split]) for split in splits}
    do_ablation = not args.no_ablation and (args.ablation or split == "all")

    runs: dict[str, dict] = {}
    ablation: dict[str, dict] = {}

    print("loading rules + NER pipeline...")
    full = build_pipeline(use_ner=True)
    runs["rules_plus_ner"] = {}
    for split in splits:
        print(f"evaluating rules+NER on {split} ({len(notes[split])} notes)...")
        runs["rules_plus_ner"][split] = run_split(full, split, notes[split])
    write_error_csvs("rules_plus_ner", runs["rules_plus_ner"])
    ablation["rules_plus_ner"] = runs["rules_plus_ner"]

    if do_ablation:
        print("loading rules-only pipeline...")
        rules_only = build_pipeline(use_ner=False)
        runs["rules_only"] = {}
        for split in splits:
            print(f"evaluating rules-only on {split} ({len(notes[split])} notes)...")
            runs["rules_only"][split] = run_split(rules_only, split, notes[split])
        write_error_csvs("rules_only", runs["rules_only"])
        ablation["rules_only"] = runs["rules_only"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_metric_csvs(runs, ablation)
    if "handwritten" in splits:
        make_chart(runs)

    report = format_report(runs, ablation)
    (RESULTS / "ner_report.txt").write_text(report, encoding="utf-8")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
