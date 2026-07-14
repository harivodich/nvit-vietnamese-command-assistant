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

from nvit_assistant.eval.confidence_evaluation import evaluate_confidence_thresholds  # noqa: E402
from nvit_assistant.nlu.intent_classifier import (  # noqa: E402
    build_pipeline,
    load_preprocessed_samples,
    load_training_config,
    samples_to_xy,
)
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
    seed, candidates = load_training_config(ROOT / "configs" / "intent_training.yaml")
    intent_report: dict[str, Any] = json.loads(
        (ROOT / "reports" / "intent_training_report.json").read_text(encoding="utf-8")
    )
    selected_name = intent_report["training"]["selected_config"]["name"]
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    train_samples = load_preprocessed_samples(ROOT / "data" / "preprocessed" / "train.jsonl")
    validation_samples = load_preprocessed_samples(
        ROOT / "data" / "preprocessed" / "validation.jsonl"
    )
    train_texts, train_labels = samples_to_xy(train_samples)
    model = build_pipeline(selected, seed).fit(train_texts, train_labels)
    validation_texts, expected = samples_to_xy(validation_samples)
    probability_rows = model.predict_proba(validation_texts)
    labels = [str(label) for label in model.classes_]
    predicted = [labels[int(row.argmax())] for row in probability_rows]
    confidences = [float(row.max()) for row in probability_rows]
    thresholds = sorted({*THRESHOLDS, runtime.confidence_threshold})
    rows = evaluate_confidence_thresholds(expected, predicted, confidences, thresholds)
    configured = next(
        row for row in rows if row["threshold"] == runtime.confidence_threshold
    )
    report = {
        "selection_split": "validation",
        "test_used": False,
        "candidate": selected.name,
        "configured_threshold": runtime.confidence_threshold,
        "configured_result": configured,
        "thresholds": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
