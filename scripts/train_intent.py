"""Preprocess, chọn intent classifier bằng validation và lưu artifact runtime."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import sklearn

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
from nvit_assistant.nlu.preprocessing import preprocess_dataset  # noqa: E402
from nvit_assistant.nlu.rule_intent_classifier import (  # noqa: E402
    evaluate_rule_classifier,
    load_rule_classifier,
)


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
    preprocessing = preprocess_dataset(
        args.samples_dir.resolve(), args.preprocessed_dir.resolve(), normalizer
    )
    seed, candidates = load_training_config(args.config.resolve())
    train_samples = load_preprocessed_samples(args.preprocessed_dir / "train.jsonl")
    validation_samples = load_preprocessed_samples(args.preprocessed_dir / "validation.jsonl")
    rule_baseline = evaluate_rule_classifier(load_rule_classifier(args.rules.resolve()), validation_samples)
    classifier, _, training = train_with_validation(
        train_samples, validation_samples, candidates, seed, args.figures_dir.resolve()
    )
    save_classifier(classifier, args.model.resolve(), args.label_map.resolve())
    report = {
        "preprocessing": preprocessing,
        "rule_baseline_validation": rule_baseline,
        "training": training,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "test_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
