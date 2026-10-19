"""Pipeline assembly.

One ``spacy.Language`` object is built once at startup and shared.  It carries:

    tokenizer -> senter -> tagger -> attribute_ruler -> parser -> lemmatizer
    -> clinical_body_parts -> clinical_laterality -> clinical_severity
    -> clinical_rom -> clinical_context -> ner -> (reconciler, plain Python)

The reconciler is not a spaCy component: it needs the rule candidates *and* the
NER entities, and its output is a Pydantic-shaped dict rather than doc
annotations.  Keeping it out of the pipe also means the ablation in
``evaluation/evaluate.py`` can reuse the same rule layers with NER empty.

Model resolution order:

1. ``training/model/model-best`` when present,
2. otherwise ``en_core_web_sm`` with its default NER left in place (a
   rules-only regime tags NER output with ``source="ner"`` and the reconciler
   still prefers rules, but the ablation is meaningless -- the app reports the
   model version so this is visible rather than silent).
"""
from __future__ import annotations

import threading
from pathlib import Path

import spacy
from spacy.language import Language

from . import lexicons, readiness
from .candidates import Candidate
from .normalize import assert_same_length, normalize_text
from .reconciler import Reconciled, reconcile
from .rules import (  # noqa: F401  (import registers the factories)
    body_part,
    context,
    lexicon,
    rom,
    severity,
)
from .rules.negation import apply_negation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "training" / "model" / "model-best"

# Serialises inference: spaCy components are not documented as thread-safe and
# the dev server runs a single worker, so a lock is cheaper than a pool.
_LOCK = threading.Lock()

_CACHED = None


class ClinicalPipeline:
    """Wraps the spaCy pipeline plus the reconcile step."""

    def __init__(self, nlp: Language, model_version: str, use_ner: bool = True):
        self.nlp = nlp
        self.model_version = model_version
        self.use_ner = use_ner

    @property
    def labels(self) -> list[str]:
        return sorted(
            {
                "BODY_PART",
                "LATERALITY",
                "PAIN_SEVERITY",
                "ROM",
                "MUSCLE",
                "FINDING",
                "EXERCISE",
                "TRIGGER",
                "RELIEVING_FACTOR",
                "DURATION",
                "FREQUENCY",
            }
        )

    # -- analysis ---------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """Run the whole pipeline and return a JSON-ready analysis dict."""
        ready, reason = readiness.assess(text)
        if not ready:
            return {
                "ready": False,
                "warnings": [reason] if reason else [],
                "fields": empty_fields(),
                "entities": [],
                "negated_entities": [],
                "model_version": self.model_version,
            }

        normalized = normalize_text(text)
        assert_same_length(text, normalized)

        with _LOCK:
            doc = self.nlp(normalized)

        rule_candidates = list(doc._.rule_candidates)
        ner_candidates = self._ner_candidates(doc) if self.use_ner else []

        apply_negation(doc, rule_candidates + ner_candidates)

        result: Reconciled = reconcile(doc, rule_candidates, ner_candidates)

        return {
            "ready": True,
            "warnings": result.warnings,
            "fields": result.fields,
            "entities": [_entity_payload(c) for c in result.entities],
            "negated_entities": [_entity_payload(c) for c in result.negated],
            "model_version": self.model_version,
        }

    def _ner_candidates(self, doc) -> list[Candidate]:
        out = []
        for ent in doc.ents:
            label = ent.label_
            if label not in ALL_LABELS:
                continue
            if not _plausible_ner(ent):
                continue
            out.append(
                Candidate(
                    start=ent.start_char,
                    end=ent.end_char,
                    label=label,
                    text=ent.text,
                    source="ner",
                )
            )
        return out


ALL_LABELS = {
    "BODY_PART",
    "MUSCLE",
    "FINDING",
    "EXERCISE",
    "TRIGGER",
    "RELIEVING_FACTOR",
}

# Function words and cue verbs the small model occasionally tags as clinical
# entities ("Rest helps." -> "helps" as FINDING).  This is a precision guard,
# not a vocabulary filter: NER-only clinical terms still pass through.
_NER_NOISE = {
    "a",
    "an",
    "and",
    "advised",
    "bring",
    "brings",
    "daily",
    "difficult",
    "eases",
    "given",
    "help",
    "helps",
    "make",
    "makes",
    "measure",
    "measures",
    "noted",
    "pain",
    "patient",
    "prescribed",
    "relieved",
    "relieves",
    "reported",
    "reports",
    "the",
    "today",
}

# A NER exercise span must look like an activity.  This rejects the model's
# occasional "<verb> <direction>" fragments without limiting it to the rule
# lexicon (NER-only surfaces such as "sit to stand" remain valid).
_EXERCISE_HINTS = {
    "abduction",
    "apart",
    "balance",
    "band",
    "bridge",
    "bug",
    "camel",
    "clamshell",
    "curl",
    "dog",
    "exercise",
    "hinge",
    "lunge",
    "march",
    "plank",
    "press",
    "pull",
    "raise",
    "retraction",
    "row",
    "set",
    "setting",
    "shrug",
    "sit",
    "slide",
    "squat",
    "stand",
    "step",
    "stretch",
    "tilt",
    "tuck",
    "walk",
}


# A NER span whose words cannot be a clinical mention of that label.  Global
# entries are function words / cue verbs; label-specific entries catch the
# model's occasional label confusion ("programme" as FINDING) without
# suppressing a word that is legitimate under another label ("stand" is fine
# inside the exercise "sit to stand").
_NER_LABEL_NOISE = {
    "FINDING": {"programme", "program", "running", "tension"},
    "RELIEVING_FACTOR": {"to", "stand", "running"},
    "TRIGGER": {"tension"},
}


def _plausible_ner(ent) -> bool:
    words = [token.lower_ for token in ent if token.is_alpha]
    lemmas = [token.lemma_.lower() for token in ent if token.is_alpha]
    if not words:
        return False
    if any(word in _NER_NOISE for word in words + lemmas):
        return False
    if any(word in _NER_LABEL_NOISE.get(ent.label_, set()) for word in words + lemmas):
        return False
    # Body parts and muscles are short noun phrases.  Rejecting digits and
    # spans made only of verbs kills the model's "left knee with 5/10 pain"
    # (MUSCLE) and "causes" (BODY_PART) confusions.  Do not reject any span
    # that merely *contains* a verb: the tagger occasionally reads "glute" as
    # a verb in "glute max weakness".
    if ent.label_ in {"BODY_PART", "MUSCLE"}:
        if len(words) > 4:
            return False
        if any(token.like_num for token in ent):
            return False
        if all(token.pos_ in {"VERB", "AUX"} for token in ent):
            return False
    if ent.label_ == "EXERCISE":
        return any(word in _EXERCISE_HINTS for word in words + lemmas)
    return True


def _entity_payload(cand: Candidate) -> dict:
    return {
        "start": cand.start,
        "end": cand.end,
        "label": cand.label,
        "text": cand.text,
        "source": cand.provenance,
        "normalized": cand.normalized or cand.text.lower(),
        "negated": cand.negated,
        "laterality": cand.meta.get("laterality"),
    }


def empty_fields() -> dict:
    return {
        "body_part": None,
        "laterality": None,
        "pain_severity": None,
        "rom": [],
        "triggers": [],
        "relieving_factors": [],
        "muscles": [],
        "findings": [],
        "exercises": [],
        "duration": None,
        "frequency": None,
    }


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _load_base() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # pragma: no cover - env problem
        raise RuntimeError(
            "en_core_web_sm is not installed. Run: python -m spacy download en_core_web_sm"
        ) from exc


def _attach_rules(nlp: Language) -> None:
    """Add the rule components ahead of NER, in a fixed order."""
    order = [
        "clinical_body_parts",
        "clinical_laterality",
        "clinical_lexicon",
        "clinical_severity",
        "clinical_rom",
        "clinical_context",
    ]
    for name in order:
        if name in nlp.pipe_names:
            continue
        # Insert each before "ner" when NER exists, else append.  Inserting
        # before ner preserves the rule-before-model order the design calls
        # for; the reconciler enforces priority regardless.
        if "ner" in nlp.pipe_names:
            nlp.add_pipe(name, name=name, before="ner")
        else:
            nlp.add_pipe(name, name=name)


def build_pipeline(use_ner: bool = True, model_path: Path | None = None) -> ClinicalPipeline:
    """Load the model (trained if available) and attach the rule layers."""
    path = model_path or MODEL_PATH
    if path.exists() and use_ner:
        nlp = spacy.load(path)
        if "parser" in nlp.pipe_names:
            # Fine-tuning route: the artefact is a full en_core_web_sm clone
            # with a clinical NER head, so it is loaded as-is.
            _attach_rules(nlp)
            return ClinicalPipeline(nlp, _model_version(path), use_ner=True)

        # Fallback: a standalone ["tok2vec", "ner"] artefact from
        # `spacy train`.  Splice it onto the base model.  Only correct when the
        # tok2vec widths match, which training/configs/ner.cfg guarantees.
        nlp = _load_base()
        trained = spacy.load(path)
        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")
        nlp.add_pipe("ner", name="ner", source=trained, before="attribute_ruler")
        _attach_rules(nlp)
        return ClinicalPipeline(nlp, _model_version(path), use_ner=True)

    nlp = _load_base()
    if not use_ner:

        @nlp.component("noop_ner")
        def noop_ner(doc):
            return doc

        nlp.remove_pipe("ner")
        _attach_rules(nlp)
        return ClinicalPipeline(nlp, "en_core_web_sm (rules only)", use_ner=False)

    _attach_rules(nlp)
    return ClinicalPipeline(nlp, f"en_core_web_sm baseline ({spacy.__version__})", use_ner=True)


def _model_version(path: Path) -> str:
    meta = path / "meta.json"
    if meta.exists():
        import json

        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return f"{data.get('name', 'model')}-{data.get('version', '0')}"
        except Exception:
            pass
    return str(path)


def get_pipeline() -> ClinicalPipeline:
    """Process-wide singleton used by the FastAPI app."""
    global _CACHED
    if _CACHED is None:
        _CACHED = build_pipeline()
    return _CACHED
