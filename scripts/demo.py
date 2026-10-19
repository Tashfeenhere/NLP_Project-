"""Print extraction results for three representative notes.

    python scripts/demo.py

Uses the same pipeline the API uses, so it is a quick way to inspect rule and
model behaviour without starting the server.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.pipeline import build_pipeline  # noqa: E402

NOTES = [
    (
        "Knee, canonical",
        "Right knee pain rated 6 out of 10, constant for three weeks. "
        "Worse with climbing stairs. Rest helps. "
        "Knee flexion is limited to 95 degrees. Quadriceps weakness. "
        "Prescribed straight leg raises and clamshells.",
    ),
    (
        "Shoulder, reversed ROM",
        "Bilateral rotator cuff pain 7/10. Overhead reaching aggravates it. "
        "ROM: 90 deg external rotation, shoulder. Rotator cuff muscles weakness. "
        "Advised wall slides and band pull aparts.",
    ),
    (
        "Negation",
        "Denies neck pain. No swelling or locking. Right knee pain 4/10, "
        "worse with kneeling.",
    ),
]


def main() -> int:
    pipeline = build_pipeline()
    print(f"model: {pipeline.model_version}\n")
    for title, note in NOTES:
        result = pipeline.analyze(note)
        print("=" * 78)
        print(title)
        print("-" * 78)
        print(note)
        print()
        print("fields:")
        print(json.dumps(result["fields"], indent=2))
        print("\nentities (span :: label :: source):")
        for entity in result["entities"]:
            print(
                f"  [{entity['start']:>3}:{entity['end']:<3}] "
                f"{entity['text']!r:28} {entity['label']:<18} {entity['source']}"
            )
        if result["negated_entities"]:
            print("\nnegated:")
            for entity in result["negated_entities"]:
                print(f"  {entity['text']!r} ({entity['label']})")
        if result["warnings"]:
            print("\nwarnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
