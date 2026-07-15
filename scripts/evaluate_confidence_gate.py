"""Tái lập model train-only và đánh giá confidence gate trên validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.eval.confidence_evaluation import (  # noqa: E402
    collect_accepted_failures,
    evaluate_confidence_thresholds,
)
from nvit_assistant.nlu.intent_classifier import (  # noqa: E402
    build_pipeline,
    load_preprocessed_samples,
    load_training_config,
    samples_to_xy,
)
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402
from nvit_assistant.nlu.preprocessing import preprocess_splits  # noqa: E402
from nvit_assistant.nlu.slot_lexicon import sha256_file  # noqa: E402
from nvit_assistant.runtime import load_runtime_settings  # noqa: E402


THRESHOLDS = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> None:
    """Đo ngưỡng đã cấu hình cùng các mốc lân cận và ghi report tái lập được."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "confidence_gate_report.json"
    )
    args = parser.parse_args()

    runtime = load_runtime_settings(ROOT / "configs" / "app.yaml", ROOT)
    samples_dir = ROOT / "data" / "samples"
    preprocessed_dir = ROOT / "data" / "preprocessed"
    normalizer_config_path = ROOT / "configs" / "regional_variants.yaml"
    preprocessing = preprocess_splits(
        samples_dir,
        preprocessed_dir,
        VietnameseNormalizer(normalizer_config_path),
        ("train.jsonl", "validation.jsonl"),
    )
    seed, candidates = load_training_config(ROOT / "configs" / "intent_training.yaml")
    intent_report: dict[str, Any] = json.loads(
        (ROOT / "reports" / "intent_training_report.json").read_text(encoding="utf-8")
    )
    selected_name = intent_report["training"]["selected_config"]["name"]
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    train_path = preprocessed_dir / "train.jsonl"
    validation_path = preprocessed_dir / "validation.jsonl"
    train_samples = load_preprocessed_samples(train_path)
    validation_samples = load_preprocessed_samples(validation_path)
    train_texts, train_labels = samples_to_xy(train_samples)
    model = build_pipeline(selected, seed).fit(train_texts, train_labels)
    validation_texts, expected = samples_to_xy(validation_samples)
    probability_rows = model.predict_proba(validation_texts)
    labels = [str(label) for label in model.classes_]
    predicted = [labels[int(row.argmax())] for row in probability_rows]
    confidences = [float(row.max()) for row in probability_rows]
    thresholds = sorted({*THRESHOLDS, runtime.confidence_threshold})
    rows = evaluate_confidence_thresholds(expected, predicted, confidences, thresholds)
    configured_summary = next(
        row for row in rows if row["threshold"] == runtime.confidence_threshold
    )
    configured = {
        **configured_summary,
        "accepted_failures": collect_accepted_failures(
            expected,
            predicted,
            confidences,
            runtime.confidence_threshold,
        ),
    }
    minimum_per_intent_coverage = min(
        value["coverage"] for value in configured["per_expected_intent"].values()
    )
    report = {
        "selection_split": "validation",
        "test_used": False,
        "methodology": {
            "threshold_selection_model_fit": "train_only",
            "threshold_selection_evaluation_split": "validation",
            "runtime_artifact_fit": "train_plus_validation",
            "runtime_artifact_refit_after_selection": True,
            "runtime_artifact_evaluated_in_this_report": False,
        },
        "candidate": selected.name,
        "preprocessing": preprocessing,
        "sample_counts": {
            "train": len(train_samples),
            "validation": len(validation_samples),
        },
        "artifacts_sha256": {
            "preprocessed_train": sha256_file(train_path),
            "preprocessed_validation": sha256_file(validation_path),
            "train": sha256_file(samples_dir / "train.jsonl"),
            "validation": sha256_file(samples_dir / "validation.jsonl"),
            "regional_variants": sha256_file(normalizer_config_path),
            "intent_training_config": sha256_file(
                ROOT / "configs" / "intent_training.yaml"
            ),
            "app_config": sha256_file(ROOT / "configs" / "app.yaml"),
            "intent_training_report": sha256_file(
                ROOT / "reports" / "intent_training_report.json"
            ),
        },
        "configured_threshold": runtime.confidence_threshold,
        "configured_result": configured,
        "operating_point_policy": {
            "minimum_per_intent_coverage_target": 0.95,
            "minimum_per_intent_coverage_observed": minimum_per_intent_coverage,
            "ood_handled_by": "separate_action_gate",
        },
        "thresholds": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if minimum_per_intent_coverage < 0.95:
        raise SystemExit(
            "configured confidence threshold violates per-intent coverage target"
        )


if __name__ == "__main__":
    main()
