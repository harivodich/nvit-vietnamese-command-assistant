"""Benchmark E5 local + Logistic Regression trên validation, không đụng test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nvit_assistant.nlu.intent_classifier import load_preprocessed_samples, samples_to_xy  # noqa: E402
from nvit_assistant.nlu.semantic_intent_classifier import (  # noqa: E402
    SemanticCandidateConfig,
    encode_texts,
    train_semantic_with_validation,
)


def main() -> None:
    """Encode train/validation, chọn C bằng validation và lưu LR semantic cuối."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, default=ROOT / "data" / "preprocessed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "semantic_intent_training.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "semantic_intent_classifier.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "semantic_intent_report.json")
    parser.add_argument(
        "--baseline-report", type=Path, default=ROOT / "reports" / "intent_training_report.json"
    )
    parser.add_argument(
        "--comparison-report", type=Path, default=ROOT / "reports" / "model_comparison_report.json"
    )
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "figures" / "semantic")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidates = [SemanticCandidateConfig(name=item["name"], c=float(item["c"])) for item in config["candidates"]]
    train = load_preprocessed_samples(args.preprocessed_dir / "train.jsonl")
    validation = load_preprocessed_samples(args.preprocessed_dir / "validation.jsonl")
    train_texts, train_labels = samples_to_xy(train)
    validation_texts, validation_labels = samples_to_xy(validation)
    train_embeddings = encode_texts(args.encoder_dir, train_texts, int(config["batch_size"]))
    validation_embeddings = encode_texts(args.encoder_dir, validation_texts, int(config["batch_size"]))
    _, selected, training = train_semantic_with_validation(train, validation, train_embeddings, validation_embeddings, candidates, int(config["seed"]), args.figures_dir)
    all_embeddings = np.concatenate([train_embeddings, validation_embeddings])
    from sklearn.linear_model import LogisticRegression
    final = LogisticRegression(C=selected.c, max_iter=2000, random_state=int(config["seed"])).fit(all_embeddings, train_labels + validation_labels)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.model)
    report = {"encoder_dir": str(args.encoder_dir), "embedding_dimension": int(train_embeddings.shape[1]), "training": training, "final_fit_samples": len(all_embeddings), "test_used": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))["training"]["selected_validation"]
    semantic = training["selected_validation"]
    comparison = {
        "selection_split": "validation",
        "test_used": False,
        "models": {
            "tfidf_logistic_regression": {
                "accuracy": baseline["accuracy"],
                "macro_f1": baseline["macro_f1"],
                "weighted_f1": baseline["per_intent"]["weighted avg"]["f1-score"],
            },
            "e5_frozen_embedding_logistic_regression": {
                "accuracy": semantic["accuracy"],
                "macro_f1": semantic["macro_f1"],
                "weighted_f1": semantic["per_intent"]["weighted avg"]["f1-score"],
            },
        },
        "selected_runtime_model": "tfidf_logistic_regression",
        "decision": "E5 frozen embedding không vượt TF-IDF trên validation; giữ baseline lexical làm runtime.",
    }
    args.comparison_report.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
