"""Preprocess, chọn intent classifier bằng validation và lưu artifact runtime."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.nlu.intent_classifier import (  # noqa: E402
    load_preprocessed_samples,
    load_training_config,
    save_classifier,
    train_with_validation,
)
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402
from nvit_assistant.nlu.preprocessing import preprocess_splits  # noqa: E402
from nvit_assistant.nlu.rule_intent_classifier import (  # noqa: E402
    evaluate_rule_classifier,
    load_rule_classifier,
)
from nvit_assistant.nlu.slot_lexicon import sha256_file  # noqa: E402


def main() -> None:
    """Không đọc test; validation chọn candidate, train+validation chỉ refit artifact cuối."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--preprocessed-dir", type=Path, default=ROOT / "data" / "preprocessed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "intent_training.yaml")
    parser.add_argument("--rules", type=Path, default=ROOT / "configs" / "intents.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "intent_classifier.joblib")
    parser.add_argument("--label-map", type=Path, default=ROOT / "models" / "intent_label_map.json")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "intent_training_report.json"
    )
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args()

    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    preprocessing = preprocess_splits(
        args.samples_dir.resolve(),
        args.preprocessed_dir.resolve(),
        normalizer,
        ("train.jsonl", "validation.jsonl"),
    )
    seed, candidates = load_training_config(args.config.resolve())
    train_samples = load_preprocessed_samples(args.preprocessed_dir / "train.jsonl")
    validation_samples = load_preprocessed_samples(args.preprocessed_dir / "validation.jsonl")
    rule_baseline = evaluate_rule_classifier(load_rule_classifier(args.rules.resolve()), validation_samples)
    classifier, _, training = train_with_validation(
        train_samples, validation_samples, candidates, seed, args.figures_dir.resolve()
    )
    model_path = args.model.resolve()
    label_map_path = args.label_map.resolve()
    save_classifier(classifier, model_path, label_map_path)
    files_sha256 = {
        "model": sha256_file(model_path),
        "label_map": sha256_file(label_map_path),
        "train": sha256_file(args.samples_dir.resolve() / "train.jsonl"),
        "validation": sha256_file(args.samples_dir.resolve() / "validation.jsonl"),
        "preprocessed_train": sha256_file(
            args.preprocessed_dir.resolve() / "train.jsonl"
        ),
        "preprocessed_validation": sha256_file(
            args.preprocessed_dir.resolve() / "validation.jsonl"
        ),
        "intent_training_config": sha256_file(args.config.resolve()),
        "regional_variants": sha256_file(
            ROOT / "configs" / "regional_variants.yaml"
        ),
    }
    metadata = {
        "schema_version": 1,
        "fit_split": "train_plus_validation",
        "selected_candidate": training["selected_config"]["name"],
        "sklearn_version": version("scikit-learn"),
        "files_sha256": files_sha256,
    }
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "artifacts_sha256": {
            **files_sha256,
            "metadata": sha256_file(metadata_path),
        },
        "artifact_metadata": "models/intent_classifier.metadata.json",
        "preprocessing": preprocessing,
        "rule_baseline_validation": rule_baseline,
        "training": training,
        "python_version": platform.python_version(),
        "sklearn_version": version("scikit-learn"),
        "test_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
