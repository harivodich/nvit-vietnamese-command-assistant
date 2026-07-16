# Data Sources

The current dataset contains 2,094 samples:

| Source | Samples | Role |
|---|---:|---|
| MASSIVE `vi-VN` | 969 | Mapped external NLU data |
| Project-authored templates | 1,106 | Intent coverage, regional vocabulary, no-diacritics forms |
| Manually reviewed hard cases | 19 | Boundary cases found during development |

The `source` field keeps these groups separate so evaluation can show whether a result comes from
more natural translated data or from easier generated templates.

## MASSIVE

The only external dataset included in train, validation, and test is the Vietnamese `vi-VN` portion
of [Amazon MASSIVE 1.0](https://github.com/alexa/massive). It is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Reference: FitzGerald et al., [*MASSIVE: A 1M-Example Multilingual Natural Language Understanding
Dataset with 51 Typologically-Diverse Languages*](https://arxiv.org/abs/2204.08582), 2022. MASSIVE
was built from SLURP; this attribution is retained because the repository contains transformed
external data rather than entirely project-authored text.

Only source intents with a clear mapping to this challenge were selected: alarms, weather, music,
and part of the reminder data. During the build:

- source intents and slots are mapped to the project schema;
- text is normalized to Unicode NFC;
- `source_ref` retains the original MASSIVE ID;
- original MASSIVE train/dev/test partitions are preserved.

MASSIVE is localized text NLU data. It has no speaker or Northern/Central/Southern accent labels, so
all 969 mapped samples use `region=standard`. It is not used as evidence of real-world regional
speech recognition.

## Project-authored data

The 1,106 synthetic samples are generated from templates for the five supported intents. They add
no-diacritics commands and selected Northern, Central, and Southern vocabulary that MASSIVE does not
label. Templates, slot values, and mappings are stored in `configs/`, making the generation process
readable and reproducible.

The 19 manual hard cases were written and labelled after development error analysis. They use
`annotation_quality=reviewed`, which means the project author checked the labels. It does **not**
mean independent review by native speakers from three regions.

No sample in this snapshot has audio or an independent native-speaker review. Regional metrics only
measure the declared lexical patterns in text.

`data/fake_contacts.json` and `data/music_catalog.json` support action demos only. Their names, phone
numbers, and metadata are fictional and are not used for training, validation, or testing.

## Provenance and leakage controls

Each sample stores `source`, `source_ref`, `group_id`, `region`, and annotation quality. Variants from
the same template family share a `group_id` and stay in one split. Validation also checks:

- exact duplicates;
- duplicates after removing diacritics;
- near-similar samples across splits;
- template-group leakage;
- slot values that do not appear in their source text;
- manifest and file checksums.

The locked 384-sample test snapshot has SHA-256:

```text
47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a
```

Changing this hash means changing the benchmark. Evaluation is broken down by source so strong
template results cannot hide weaker performance on MASSIVE.
