# Clinical Note Analyzer

A physiotherapy note → structured record prototype. Paste a free-text session
note, review the extracted fields and highlighted source spans, correct
anything the extractor got wrong, and save the confirmed record to SQLite.

> **Assists documentation only — it does not diagnose.** Every extracted value
> must be checked by the treating clinician.

This is the Project 1 deliverable: an explainable information-extraction
pipeline (rules + a trained spaCy NER head), a FastAPI backend, a React/Vite
review UI, and an evaluation harness with a rules-only vs rules+NER ablation.
Everything runs on a laptop CPU.

---

## Quick start

Requirements: Python 3.11–3.13 and Node 18+.

```powershell
# 1. Environment, dependencies, spaCy model, corpus
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 2. Train the NER head (a few minutes on CPU)
.venv\Scripts\python.exe training\train_ner.py --epochs 30

# 3. Run backend (:8000) + frontend (:5173)
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```

Open <http://localhost:5173>, paste a note such as:

```
Right knee pain 6/10 for three weeks, worse with climbing stairs.
Knee flexion is limited to 95 degrees. Quadriceps weakness.
Prescribed straight leg raises and clamshells.
```

Press **Analyze note**, then edit and save.

Manual commands, if you prefer:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python data\generate_notes.py --seed 42 --n 260
python data\build_docbin.py
python data\build_handwritten.py
python training\train_ner.py --epochs 30
uvicorn app.main:app --reload
cd frontend; npm install; npm run dev
```

---

## What it extracts

| Field | Example | Main source |
|---|---|---|
| `body_part` | knee, lower back, cervical spine | gazetteer |
| `laterality` | right / left / bilateral (never guessed) | Matcher + binding |
| `pain_severity` | `{raw: "7/10", value: 7, range: null}` | Matcher + validation |
| `rom` | `[{movement, value, unit, quality}]` | ROM matcher + parser binding |
| `triggers` | climbing stairs, prolonged sitting | cue-anchored Matcher + NER |
| `relieving_factors` | rest, heat, walking | cue-anchored Matcher + NER |
| `muscles` | quadriceps, glute max | gazetteer + NER |
| `findings` | weakness, tightness, effusion | gazetteer + NER |
| `exercises` | straight leg raise, clamshell | lemma Matcher + NER |
| `duration` | three weeks, since March | Matcher |
| `frequency` | constant, intermittent | gazetteer |

`POST /api/analyze` returns the fields plus every entity's `start`/`end`
character offsets and `source` (`rule`, `gazetteer`, or `ner`). Offsets always
slice back to the entity text; entities never overlap. Negated mentions are
returned separately and are excluded from the fields.

ROM is modelled as one whole-phrase entity (“knee flexion is limited to 95
degrees”); the joint and movement are metadata, not nested spans. A body part
inside a ROM span is not emitted twice.

### Example

```powershell
curl -X POST http://localhost:8000/api/analyze `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Denies neck pain. Right knee pain 6/10. Knee flexion is limited to 95 degrees.\"}'
```

```json
{
  "ready": true,
  "fields": {
    "body_part": "knee",
    "laterality": "right",
    "pain_severity": {"raw": "6/10", "value": 6, "range": null},
    "rom": [{"movement": "flexion", "value": 95, "unit": "deg", "quality": null}]
  },
  "negated_entities": [{"label": "BODY_PART", "text": "neck"}],
  "model_version": "clinical_ner-1.0.0"
}
```

---

## Architecture

```
raw note
  │
  ├─ normalize (length-preserving: NFKC, dashes, whitespace)
  ├─ spaCy tokenize → senter → tagger → attribute_ruler → parser → lemmatizer
  ├─ rule components (before ner)
  │    clinical_body_parts   body-part gazetteer + laterality binding
  │    clinical_laterality   standalone laterality candidates
  │    clinical_lexicon      muscles, findings, exercises, durations
  │    clinical_severity     pain scores
  │    clinical_rom          forward / reversed / qualitative ROM
  │    clinical_context      cue-anchored triggers, relieving factors, frequency
  ├─ clinical NER            fine-tuned TransitionBasedParser head
  └─ reconciler (plain Python)
       precedence → non-overlap → laterality binding → negation → fields
```

Precedence (lower wins): `ROM > severity > relieving > trigger > context >
gazetteer > NER`. Same-label containment is special-cased so a longer NER span
(“single leg squat”) wins over a shorter rule match (“squat”). Every candidate
keeps its provenance for the UI and the report.

### Why this is explainable

- Rules are lexical and cue-anchored; the UI shows the matched span.
- The NER output is an ordinary spaCy span with character offsets.
- The response reports `source` for every entity.
- Unsupported/ambiguous input returns `ready: false` with a reason rather than
  a confident-looking empty record.

---

## Data and model

- **Generator** (`data/generate_notes.py`, seed 42): slot-grammar notes with
  offsets recorded as each slot is appended, so gold offsets are exact by
  construction. Splits are **by template group**, not by note:
  train 175 / dev 40 / generated test 45, plus 160 training-only syntax
  augmentation notes.
- **Hand-written set** (`data/build_handwritten.py` → `data/hand_written_test.jsonl`):
  45 marked-up notes with exact offsets, including six notes whose vocabulary
  exists only in the NER corpus (so the ablation has something honest to
  measure).
- **NER training** (`training/train_ner.py`): clone `en_core_web_sm`, replace
  its NER head with a fresh 6-label clinical head, freeze the other pipes, and
  train on CPU. The artefact is `training/model/model-best`.
- **Config-file route** (`training/configs/ner.cfg`): the proposal's
  `spacy init config --pipeline ner` equivalent. It trains a standalone
  `["tok2vec", "ner"]` artefact (useful for reproducing the plan's
  `spacy train` command); at runtime `nlp/pipeline.py` splices its NER head
  onto `en_core_web_sm`, which supplies the parser and lemmatizer the rules
  need:

  ```powershell
  python -m spacy train training\configs\ner.cfg `
      --paths.train data\processed\train.spacy `
      --paths.dev data\processed\dev.spacy `
      --output training\model-standalone --gpu-id -1
  ```

- **Model fallback**: if `model-best` is absent, the app still starts with the
  base spaCy NER and reports that model version; the ablation is then not
  meaningful, and the UI makes the loaded model visible.

---

## Evaluation

```powershell
.venv\Scripts\python.exe evaluation\evaluate.py
# one split only:
.venv\Scripts\python.exe evaluation\evaluate.py --split handwritten --ablation
# --split test is an alias for the generated held-out split
```

The harness scores both test sets in two regimes:

- **entity-level** exact-span P/R/F1 per label, reported as a support-weighted
  **micro** average and an unweighted **macro** average over labels present in
  gold (negated gold/predictions are excluded and reported separately;
  LATERALITY is scored as a field),
- **field-level** exact match, the number the app actually shows
  (ROM compared as `(movement, value, quality)`),
- **ablation**: rules-only vs rules+NER,
- error CSVs with false positives/negatives and field misses,
- a bar chart of entity F1 per label.

Artefacts: `evaluation/results/` and `evaluation/errors/`.

### Latest results (seed 42, CPU)

| Split | Regime | Entity F1 (micro) | Entity F1 (macro) | Field accuracy |
|---|---|---:|---:|---:|
| generated test (45) | rules only | 0.865 | 0.822 | 0.865 |
| generated test (45) | rules + NER | 0.978 | 0.971 | 0.984 |
| hand-written (45) | rules only | 0.943 | 0.945 | 0.954 |
| hand-written (45) | rules + NER | 0.986 | 0.988 | 0.986 |

The generated split shares its template families with training, so its scores
are optimistic. The hand-written split is the more meaningful number, but it
was authored after the rule lexicon and therefore shares a lot of standard
vocabulary with it; treat 0.98 as an upper bound rather than a claim about
unseen clinical prose. The six NER-only-vocabulary notes are the closest thing
to a genuine generalisation test, and the ablation on the generated split shows
where the trained model earns its place. There is **no inter-annotator
agreement claim**: training data is synthetic and the hand-written set has a
single annotator.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | analyze a note → fields, entities, warnings |
| POST | `/api/records` | save a confirmed record |
| GET | `/api/records?patient_id=` | list saved summaries |
| GET | `/api/records/{id}` | full saved record |
| DELETE | `/api/records/{id}` | delete |
| GET | `/api/records/export.csv` | flat CSV export |
| GET | `/api/health` | model version, labels, record count |
| GET | `/docs` | interactive OpenAPI docs |

Records are stored in SQLite (`data/app.db`; override with `CNA_DB_PATH`).
The database schema separates `patients` and `records` and stores fields as
JSON so a later dashboard/search can read old rows without a migration.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Coverage: body part/laterality/pain, exercises (plurals/aliases), ROM shapes
and sentence-boundary binding, trigger/relieving cue scoping, negation and
scope breakers, readiness guards, offset/no-overlap invariants, the full
API + SQLite CRUD/export round trip against a temporary database, and
acceptance fixtures for the three representative notes shown in the UI
(`tests/fixtures/acceptance_notes.json`). The acceptance test asserts the
complete structured record, determinism, and the offset/non-overlap contract.

---

## Proposal deviations

Three small choices differ from the proposal's literal wording; they are
deliberate and documented here so the implementation can be checked against
the plan:

- **LATERALITY is not emitted as a standalone entity.** It is bound into
  `fields.laterality` (and into the entity's metadata), because highlighting
  “right” separately adds noise and double-counts in evaluation. It is scored
  as a field.
- **Six NER labels** (`BODY_PART`, `MUSCLE`, `FINDING`, `EXERCISE`, `TRIGGER`,
  `RELIEVING_FACTOR`); measurements such as pain, ROM, duration, and frequency
  stay rule-based, where they are more reliable and explainable.
- **Negation is bounded-window NegEx** with comma/semicolon-aware scope
  handling, not full assertion semantics; hypotheticals, family history, and
  temporality are out of scope.

---

## Project layout

```
app/            FastAPI app, Pydantic schemas, SQLite layer, routers
nlp/            pipeline, lexicons, rule components, reconciler, normalization
data/           generator, DocBin builder, hand-written test builder + JSONL
training/       fine-tuning script, configs, model artefact
evaluation/     metrics, ablation, error CSVs, chart, report
frontend/       React + Vite review UI
tests/          pytest suite
scripts/        setup.ps1, dev.ps1
```

---

## Known limitations

- Training data is synthetic and closed-vocabulary; the NER is a small spaCy
  transition-based head, not a clinical language model. It generalises to
  phrasing it has seen, not to arbitrary clinical prose.
- The hand-written test set is small (45 notes) and single-annotator.
- Negation is bounded-window NegEx, not full clinical assertion semantics
  (no hypotheticals, family history, or temporality).
- The app stores free text and patient names unencrypted in local SQLite; it is
  a coursework prototype, not a deployed clinical system.
- Single uvicorn worker; inference is serialised by a lock. No authentication.
