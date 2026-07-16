# NVIT Vietnamese Command Assistant

A text-first NLU assistant for short Vietnamese commands. It accepts a transcript, normalizes
known spelling and regional lexical variants, predicts an intent, extracts slots, and returns a
mock action or a clarification. The default pipeline runs locally without an LLM or network call.

This submission focuses on the core of the challenge: Vietnamese command understanding, regional
variation in text, leakage-aware evaluation, and a complete path from input to action. It does not
claim to solve speech recognition or real-world accent recognition.

## Scope

The assistant supports five intents:

- `set_reminder`: create a reminder;
- `set_alarm`: set an alarm;
- `ask_weather`: ask for weather information;
- `play_music`: request a song or artist;
- `call_contact`: call a contact or phone number immediately.

It extracts seven slots: `datetime`, `location`, `song`, `artist`, `contact_name`, `phone_number`,
and `reminder_text`.

All actions use `status=mocked` by default. The program will not place calls, set device alarms, or
play media. A live Open-Meteo weather demo is available as an explicit opt-in.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m pip install -e .
```

Run the CLI:

```powershell
nvit-assistant "nhắc tôi uống thuốc lúc 8 giờ"
nvit-assistant --json "Bữa ni ở Huế trời răng rồi hỉ"
```

The command is registered through the `[project.scripts]` entry in `pyproject.toml`. The same CLI
can be run directly from source if the entry point is not installed:

```powershell
python scripts/run_assistant.py --json "mở nhạc Mỹ Tâm cho tui nghe"
```

Start the API:

```powershell
python -m uvicorn nvit_assistant.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. `POST /parse` accepts:

```json
{
  "text": "nhắc tôi uống thuốc lúc 8 giờ",
  "region_hint": "standard"
}
```

## Examples

These shortened outputs come from the current CLI.

```text
Input:  Bữa ni ở Huế trời răng rồi hỉ
Output: intent=ask_weather, region=central
        slots={datetime: "hôm nay", location: "huế"}
        response="Đã giả lập yêu cầu thời tiết tại huế vào hôm nay."
```

```text
Input:  mở nhạc Mỹ Tâm cho tui nghe
Output: intent=play_music
        slots={artist: "mỹ tâm"}
        response="Đã giả lập phát nhạc của mỹ tâm."
```

```text
Input:  nhắc tôi uống thuốc lúc 8 giờ
Output: intent=set_reminder
        slots={datetime: "8 giờ", reminder_text: "uống thuốc"}
        response="Đã giả lập tạo lời nhắc ‘uống thuốc’ vào 8 giờ."
```

## Pipeline

```text
text transcript
  -> Unicode, known STT/spelling, and regional lexical normalization
  -> word/character TF-IDF + Logistic Regression
  -> confidence gate
  -> intent-aware slot extraction
  -> action-safety gate
  -> required-slot check
  -> mock/live-weather action, clarification, or safe rejection
```

The CLI and FastAPI application use the same pipeline factory. The API loads the model once during
startup rather than rebuilding it for every request.

## Repository layout

```text
configs/                         model, slot, regional, and data-generation settings
data/raw_sources/                source data retained with provenance
data/samples/                    train, validation, and locked test JSONL files
models/                          intent artifact and train-only slot lexicon
scripts/                         build, validate, train, evaluate, and demo commands
src/nvit_assistant/
  actions/                       mock router and optional live-weather adapter
  eval/                          metrics and final evaluator
  nlu/                           normalization, intent, slots, and runtime pipeline
tests/                           unit, regression, and integration tests
```

The reasoning behind the scope and technical choices is in [DECISIONS.md](DECISIONS.md).

## Dataset

The checked-in snapshot contains 2,094 samples across all five intents:

| Split | Samples |
|---|---:|
| Train | 1,363 |
| Validation | 347 |
| Test | 384 |

The test set contains 174 standard Vietnamese samples and 70 samples for each Northern, Central,
and Southern lexical group. The data combines 969 mapped Vietnamese MASSIVE samples, 1,106
template-generated samples, and 19 manually reviewed hard cases. Provenance and licensing details
are documented in [data/SOURCES.md](data/SOURCES.md).

Template families are grouped before splitting. The validator checks exact duplicates,
diacritic-insensitive duplicates, near-similar samples, and group leakage across splits. The slot
lexicon is built from training data only.

The regional benchmark measures **textual lexical variation**, not speech accent. No sample has
been independently reviewed under a native-speaker protocol, and no audio is included. Regional
scores should therefore not be read as evidence of real-world accent recognition.

The locked test snapshot has SHA-256:

```text
47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a
```

To rebuild the dataset from MASSIVE 1.0 `vi-VN.jsonl`:

```powershell
python scripts/build_dataset.py --massive-jsonl path\to\vi-VN.jsonl
python scripts/validate_data.py --data-dir data/samples
python scripts/audit_normalization.py --data-dir data/samples
python scripts/build_slot_lexicon.py
```

## Model choice

The runtime model is word/character TF-IDF with Logistic Regression. Word n-grams preserve useful
command phrases, while character n-grams improve tolerance to missing diacritics and surface noise.
The classifier is small, fast on CPU, and provides probabilities for the confidence gate.

Three TF-IDF candidates were fitted on train and selected by validation macro-F1. The selected
candidate reached **98.17% validation macro-F1**. A separate `multilingual-e5-small` plus Logistic
Regression experiment reached **86.06%** on the same validation split, so E5 was not included in
runtime. The optional experiment remains reproducible without adding its large model to the repo:

```powershell
python -m pip install -e ".[semantic]"
```

Slots are extracted with intent-aware rules and regular expressions. This is easy to inspect and
fits the seven-slot scope, but it is less robust to unseen entities and phrasing than a trained
sequence tagger.

## Evaluation

The current 384-sample test snapshot gives:

| Metric | Result |
|---|---:|
| Raw intent accuracy / macro-F1 | **92.71% / 92.25%** |
| Runtime intent accuracy | **90.36%** |
| Runtime coverage / selective accuracy | **92.45% / 97.75%** |
| Oracle slot exact match / micro-F1 | **81.77% / 86.89%** |
| End-to-end slot exact match / micro-F1 | **74.48% / 83.15%** |
| Full-command success | **73.96%** |

Raw intent accuracy evaluates the classifier alone. Runtime intent includes confidence, boundary,
and safety decisions. Full-command success requires the intent, slots, and mock action to all be
correct, so **73.96%** is the most representative end-to-end number.

Runtime intent accuracy by lexical region is Northern **81.43%**, Central **88.57%**, and Southern
**94.29%**. The no-diacritics subset reaches **76.85%**. These figures come from a partly synthetic
text benchmark and must be interpreted with the limitations above.

The final test was inspected during late error analysis and influenced runtime rules. The current
figures are therefore regression results on the locked snapshot, not a fresh unbiased holdout. The
model and dataset were not fitted using test labels. A new independently reviewed holdout is the
next required step before making a generalization claim.

## Reproduce the checks

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m mypy src
python scripts/validate_data.py --data-dir data/samples
python scripts/audit_normalization.py --data-dir data/samples
```

The current snapshot passes **297 tests**, Ruff, mypy, dependency validation, dataset validation,
and the normalization/leakage audit.

Regenerate the detailed local evaluation report and confusion matrix with:

```powershell
python scripts/evaluate.py --post-audit --overwrite
```

Generated reports are intentionally ignored by Git because they are reproducible artifacts. The
headline results and their limitations are kept in this README and in `DECISIONS.md`.

## Optional live-weather demo and remaining scope

```powershell
nvit-assistant --live-weather "thời tiết ở Huế ngày mai"
```

This mode calls Open-Meteo only for weather. Contacts and the music catalog are fake demo data.
STT/TTS, authentication, privacy controls, and real device integrations are not implemented.
