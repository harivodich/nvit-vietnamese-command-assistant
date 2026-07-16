# Technical Decisions

This document follows the eight questions in the challenge brief. I have kept the claims limited to
what the repository and evaluation can support.

## 1. What I understood the problem to be

The core task is not to build a general chatbot. It is to turn a short Vietnamese command into an
intent and the details needed to carry it out, then produce an appropriate response or action.

What matters most is:

- clear intent and slot boundaries;
- an explicit approach to Vietnamese lexical and diacritic variation;
- a complete input-to-action path;
- an evaluation that separates model quality from pipeline quality;
- honest limits, especially around regional speech.

What matters less for this exercise is a polished interface, many external integrations, or a large
model chosen only because it looks modern. I therefore built a text-first assistant with five
intents and seven slots. STT/TTS and real device control are outside the completed scope.

## 2. How I analysed and scoped it

I split the work into four layers:

1. a shared schema and a dataset with provenance;
2. Vietnamese normalization and preprocessing;
3. intent classification and intent-aware slot extraction;
4. runtime gates, mock actions, CLI/API, and evaluation.

```text
transcript
  -> Vietnamese and regional lexical normalization
  -> intent classifier
  -> confidence gate
  -> intent-aware slot extraction
  -> action-safety gate
  -> required-slot check
  -> action, clarification, or rejection
```

The checked-in dataset has 2,094 samples: 1,363 train, 347 validation, and 384 test. MASSIVE keeps
its original partitions. Generated template families are grouped before splitting so close variants
do not appear on both sides of an evaluation. A validator checks exact, diacritic-insensitive, and
near-similar leakage.

Train data fits the model and builds the slot lexicon. Validation selects the model candidate and
confidence threshold. The selected runtime artifact is then fitted on train plus validation. The
test snapshot is checksum-locked.

All default actions are mocked. A live Open-Meteo adapter is available only as an opt-in demo;
contacts and music remain fake local data. This keeps the evaluated path deterministic and avoids
pretending that device permissions, privacy, or authentication have been solved.

## 3. Key choices and alternatives

### Intent classification

I chose word/character TF-IDF with Logistic Regression. Word n-grams capture phrases such as
`gọi cho` or `nhắc tôi`, while character n-grams help with missing diacritics and spelling noise.
Logistic Regression trains quickly on CPU, produces probabilities for the confidence gate, and
creates a small local artifact.

Three TF-IDF configurations were fitted on train and compared by validation macro-F1. Two reached
**98.17%** macro-F1; `word_1_2_char_3_5` won using a predefined tie-break, not test results.

I also considered or tested:

- **rules only**: useful as an explainable baseline, but too brittle for runtime;
- **Word2Vec/GloVe**: static embeddings offered no clear advantage for these short commands;
- **multilingual-e5-small + Logistic Regression**: tested and reproducible, but reached only
  **86.06% validation macro-F1** and required much more memory;
- **PhoBERT or a joint transformer**: promising with more natural, speaker-reviewed data, but likely
  to overfit this dataset and unnecessary for the current latency target;
- **an ensemble**: not used because there was no evidence that the models corrected complementary
  errors enough to justify extra complexity.

The project contribution is therefore not a new optimizer. It is the data mapping, leakage control,
Vietnamese normalization, intent-boundary policy, model comparison, slot extraction, safety gates,
and one runtime shared by CLI and API.

### Slots and runtime

Slots use intent-aware regex and lexicons rather than a sequence tagger. With seven slots and limited
labelled data, this is easy to inspect and debug. The trade-off is weaker handling of unseen entities
and sentence forms.

The normalizer is deterministic: Unicode NFC, whitespace cleanup, known transcription/spelling
variants, and selected regional terms. Longer phrases run before shorter ones, and ambiguous
particles are changed only when the context is strong enough. Training and runtime use the same
normalizer.

CLI and FastAPI call the same pipeline factory. FastAPI loads the model once at startup. Mock actions
allow end-to-end tests without network calls or side effects.

## 4. Vietnamese speech and regional variation

The submission accepts text, so it handles the **lexical surface effects** associated with regional
speech rather than audio accent itself. Examples include `bữa ni`, `răng`, `hông`, `tui`, and common
no-diacritics forms. The normalizer returns both normalized text and the rules that matched, which
makes its decisions inspectable.

MASSIVE has no Northern/Central/Southern speaker label, so all 969 mapped MASSIVE samples are marked
`standard`. The regional test groups come from project-authored text templates. They cover declared
vocabulary and particles, but they do not prove that the system understands real speakers from the
three regions.

This is the most important limitation: there are no audio recordings and no independently reviewed
native-speaker samples. A proper speech evaluation would require consented audio, balanced regional
and provincial coverage, speaker-disjoint splits, and separate reporting of STT WER/CER and NLU
intent/slot metrics.

## 5. Intent, slots, and evaluation

The supported intents are `set_reminder`, `set_alarm`, `ask_weather`, `play_music`, and
`call_contact`. The most difficult policy boundary is the word `gọi`:

- `gọi mẹ ngay đi` is `call_contact`;
- `6 giờ gọi mẹ` is `set_reminder`, because the future action should be remembered rather than run;
- `gọi tôi dậy lúc 6 giờ` is `set_alarm`, because the goal is to wake the user.

Intent evaluation includes accuracy, per-class precision/recall/F1, macro-F1, weighted-F1, and a
confusion matrix. Probability metrics include log loss, Brier score, ECE, and one-vs-rest ROC/PR
because runtime decisions use confidence.

Slots are reported in two ways. **Oracle slot** evaluation supplies the gold intent and isolates the
extractor. **End-to-end slot** evaluation uses the pipeline's predicted intent and therefore includes
upstream failures.

Current results on the 384-sample locked test snapshot are:

| Metric | Result |
|---|---:|
| Raw intent accuracy / macro-F1 | **92.71% / 92.25%** |
| Runtime intent accuracy | **90.36%** |
| Runtime coverage / selective accuracy | **92.45% / 97.75%** |
| Oracle slot exact match / micro-F1 | **81.77% / 86.89%** |
| End-to-end slot exact match / micro-F1 | **74.48% / 83.15%** |
| Full-command success | **73.96%** |

The raw classifier score is not the final product score. Full-command success is lower because the
pipeline must also extract slots, pass safety checks, and create a valid mock action.

Results by lexical region are Northern **81.43%**, Central **88.57%**, and Southern **94.29%** runtime
intent accuracy. No-diacritics commands reach **76.85%**. These groups are partly synthetic and do
not represent audio accents.

Late error analysis used failures from this test snapshot to improve runtime boundary and action
rules. The current numbers are consequently regression results, not an untouched estimate for new
data. The model and dataset were not fitted using test labels, but a fresh independently reviewed
holdout is still required for a strong generalization claim.

## 6. On-device constraints

The default path does not call an LLM or inference API. TF-IDF, normalization, and regex extraction
run locally on CPU. The intent artifact is about 514 KB. In one Windows laptop run, median end-to-end
parse latency was around 6 ms; this is a local diagnostic, not a mobile battery or cold-start
benchmark.

For a real device I would remove FastAPI from the inference path, package the model for the target
runtime, cache artifacts, and measure cold start, memory, battery, and latency on the actual device.
An offline STT engine would only be selected after measuring its WER/CER and its downstream effect
on intent and slots.

The runtime pins scikit-learn because the joblib artifact depends on its version. The artifact is
checksum-verified, but joblib must still only be loaded from a trusted source.

## 7. Hardest part and remaining limitations

The hardest part was defining and enforcing the boundary between reminder, alarm, and immediate
call commands while preserving names, song titles, and regional wording during normalization.
Leakage control was also important because generated variants can make validation results look much
better than real generalization.

The main remaining limitations are:

- regional data is mainly generated text, with no native-speaker or audio benchmark;
- the classifier is closed-set, so confidence is not a complete OOD detector;
- regex and catalog-based slots miss unseen entities or phrasing;
- full-command success is **73.96%**, leaving a clear gap after intent prediction;
- the safety set is a development regression set, not an independent production red-team test;
- actions do not cover device permissions, authentication, or privacy controls.

## 8. What I left out and how I would continue

The completed scope is the text-first pipeline, dataset validation, intent model, slot extraction,
CLI/API, mock actions, and evaluation. STT/TTS, real device actions, and production security are not
implemented.

My first next step would be a new development and test set written or reviewed by speakers from all
three regions. It should be kept independent from current error analysis. Only after that would I
change rules or models and evaluate again.

The audio phase would add consented recordings and speaker-disjoint splits, then report STT WER/CER
separately from NLU metrics. PhoBERT or joint intent-slot learning would be justified only if the new
data showed that the lightweight baseline had reached its limit. Real device adapters, permission
handling, and TTS would come later if the project moved from a challenge prototype toward a product.
