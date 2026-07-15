"""Benchmark E5 local + Logistic Regression trên validation, không đụng test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nvit_assistant.nlu.intent_classifier import load_preprocessed_samples, samples_to_xy  # noqa: E402
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402
from nvit_assistant.nlu.preprocessing import preprocess_splits  # noqa: E402
from nvit_assistant.nlu.semantic_intent_classifier import (  # noqa: E402
    SemanticCandidateConfig,
    encode_texts,
    train_semantic_with_validation,
)
from nvit_assistant.nlu.slot_lexicon import sha256_file  # noqa: E402


ENCODER_ID = "intfloat/multilingual-e5-small"


def _package_version(name: str) -> str:
    """Đọc version dependency nhưng không làm hỏng benchmark nếu metadata bị thiếu."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _encoder_hashes(encoder_dir: Path) -> dict[str, str]:
    """Fingerprint các file quyết định trọng số/tokenizer thay vì ghi path riêng của máy."""
    filenames = (
        "config.json",
        "model.safetensors",
        "modules.json",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    return {
        filename: sha256_file(encoder_dir / filename)
        for filename in filenames
        if (encoder_dir / filename).is_file()
    }


def main() -> None:
    """Encode train/validation, chọn C bằng validation và lưu LR semantic cuối."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "data" / "samples")
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
    encoder_dir = args.encoder_dir.resolve()
    samples_dir = args.samples_dir.resolve()
    preprocessed_dir = args.preprocessed_dir.resolve()
    normalizer_config_path = ROOT / "configs" / "regional_variants.yaml"
    preprocessing = preprocess_splits(
        samples_dir,
        preprocessed_dir,
        VietnameseNormalizer(normalizer_config_path),
        ("train.jsonl", "validation.jsonl"),
    )
    train_path = preprocessed_dir / "train.jsonl"
    validation_path = preprocessed_dir / "validation.jsonl"
    baseline_report_path = args.baseline_report.resolve()
    train = load_preprocessed_samples(train_path)
    validation = load_preprocessed_samples(validation_path)
    input_hashes = {
        "preprocessed_train": sha256_file(train_path),
        "preprocessed_validation": sha256_file(validation_path),
        "train": sha256_file(samples_dir / "train.jsonl"),
        "validation": sha256_file(samples_dir / "validation.jsonl"),
        "regional_variants": sha256_file(normalizer_config_path),
        "semantic_training_config": sha256_file(args.config.resolve()),
        "baseline_report": sha256_file(baseline_report_path),
    }
    baseline_payload = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_hashes = baseline_payload.get("artifacts_sha256", {})
    for name in (
        "train",
        "validation",
        "preprocessed_train",
        "preprocessed_validation",
        "regional_variants",
    ):
        if baseline_hashes.get(name) != input_hashes[name]:
            raise ValueError(
                "intent_training_report.json không cùng snapshot dữ liệu; "
                "hãy chạy scripts/train_intent.py trước"
            )
    train_texts, train_labels = samples_to_xy(train)
    validation_texts, validation_labels = samples_to_xy(validation)
    batch_size = int(config["batch_size"])
    train_embeddings = encode_texts(encoder_dir, train_texts, batch_size)
    validation_embeddings = encode_texts(encoder_dir, validation_texts, batch_size)
    _, selected, training = train_semantic_with_validation(
        train,
        validation,
        train_embeddings,
        validation_embeddings,
        candidates,
        int(config["seed"]),
        args.figures_dir.resolve(),
    )
    all_embeddings = np.concatenate([train_embeddings, validation_embeddings])
    from sklearn.linear_model import LogisticRegression
    final = LogisticRegression(C=selected.c, max_iter=2000, random_state=int(config["seed"])).fit(all_embeddings, train_labels + validation_labels)
    model_path = args.model.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)
    report = {
        "encoder": {
            "id": ENCODER_ID,
            "local_directory_name": encoder_dir.name,
            "files_sha256": _encoder_hashes(encoder_dir),
            "encoding_contract": {
                "text_prefix": "query: ",
                "normalize_embeddings": True,
                "batch_size": batch_size,
            },
        },
        "embedding_dimension": int(train_embeddings.shape[1]),
        "preprocessing": preprocessing,
        "training": training,
        "final_fit_samples": len(all_embeddings),
        "artifact": {
            "contents": "logistic_regression_head_only",
            "sha256": sha256_file(model_path),
        },
        "inputs_sha256": input_hashes,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": _package_version("scikit-learn"),
            "sentence_transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "test_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    baseline = baseline_payload["training"]["selected_validation"]
    semantic = training["selected_validation"]
    comparison = {
        "selection_split": "validation",
        "test_used": False,
        "snapshot_sha256": {
            "preprocessed_train": input_hashes["preprocessed_train"],
            "preprocessed_validation": input_hashes["preprocessed_validation"],
        },
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
        "decision_policy": (
            "Ưu tiên macro-F1 validation; nếu không hơn rõ ràng thì giữ TF-IDF vì artifact "
            "nhỏ, suy luận CPU đơn giản và không phụ thuộc encoder 471 MB."
        ),
        "decision": (
            "E5 frozen embedding không vượt TF-IDF trên cùng validation; "
            "giữ TF-IDF làm runtime."
            if semantic["macro_f1"] <= baseline["macro_f1"]
            else "E5 cao hơn validation nhưng TF-IDF vẫn được giữ theo quyết định runtime đã chốt."
        ),
    }
    args.comparison_report.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_report.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
