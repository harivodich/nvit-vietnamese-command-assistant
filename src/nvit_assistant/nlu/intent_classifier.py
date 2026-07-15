"""Huấn luyện và suy luận intent bằng TF-IDF + Logistic Regression có thể giải thích."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nvit_matplotlib"))

import matplotlib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from nvit_assistant.nlu.runtime_intent_classifier import IntentClassifier
from nvit_assistant.nlu.runtime_intent_classifier import (
    IntentPrediction as IntentPrediction,
)
from nvit_assistant.nlu.runtime_intent_classifier import load_classifier as load_classifier
from nvit_assistant.schemas import Intent, PreprocessedSample

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CandidateConfig:
    """Một cấu hình feature/model được chọn hoàn toàn bằng validation."""

    name: str
    word_ngram_range: tuple[int, int]
    word_min_df: int
    use_char_features: bool
    char_ngram_range: tuple[int, int] = (3, 5)
    char_min_df: int = 2
    c: float = 1.0


def load_preprocessed_samples(path: Path) -> list[PreprocessedSample]:
    """Đọc artifact preprocess, không cho trainer vô tình dùng JSONL gốc."""
    samples: list[PreprocessedSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                samples.append(PreprocessedSample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"artifact preprocess trống: {path}")
    return samples


def load_training_config(path: Path) -> tuple[int, list[CandidateConfig]]:
    """Đọc candidate config từ YAML để thay đổi thí nghiệm không cần sửa code."""
    with path.open("r", encoding="utf-8") as file:
        raw_config: Any = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("seed"), int):
        raise ValueError(f"config train không hợp lệ: {path}")
    if raw_config.get("selection_metric") != "macro_f1":
        raise ValueError("selection_metric hiện chỉ hỗ trợ macro_f1")
    raw_candidates = raw_config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"config train cần candidates: {path}")

    candidates: list[CandidateConfig] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("mỗi candidate phải là mapping")
        word_range = raw.get("word_ngram_range")
        char_range = raw.get("char_ngram_range", [3, 5])
        if (
            not isinstance(word_range, list)
            or len(word_range) != 2
            or not all(isinstance(value, int) for value in word_range)
            or not isinstance(char_range, list)
            or len(char_range) != 2
            or not all(isinstance(value, int) for value in char_range)
        ):
            raise ValueError("ngram range phải là list gồm hai số nguyên")
        candidates.append(
            CandidateConfig(
                name=str(raw["name"]),
                word_ngram_range=(word_range[0], word_range[1]),
                word_min_df=int(raw["word_min_df"]),
                use_char_features=bool(raw["use_char_features"]),
                char_ngram_range=(char_range[0], char_range[1]),
                char_min_df=int(raw.get("char_min_df", 2)),
                c=float(raw["c"]),
            )
        )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate name bị trùng")
    return raw_config["seed"], candidates


def build_pipeline(config: CandidateConfig, seed: int) -> Pipeline[Any]:
    """Tạo pipeline feature và classifier; chỉ fit trên train ở bước gọi phía dưới."""
    features: list[tuple[str, Any]] = [
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=config.word_ngram_range,
                min_df=config.word_min_df,
                sublinear_tf=True,
            ),
        )
    ]
    if config.use_char_features:
        features.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=config.char_ngram_range,
                    min_df=config.char_min_df,
                    sublinear_tf=True,
                ),
            )
        )
    return Pipeline(
        [
            ("features", FeatureUnion(features)),
            ("classifier", LogisticRegression(C=config.c, max_iter=2000, random_state=seed)),
        ]
    )


def samples_to_xy(samples: list[PreprocessedSample]) -> tuple[list[str], list[str]]:
    """Tách feature `normalized_text` và nhãn intent ra khỏi artifact preprocess."""
    return (
        [sample.normalized_text for sample in samples],
        [sample.original.intent.value for sample in samples],
    )


def evaluate_probability_metrics(
    expected: list[str], predicted: list[str], probabilities: Any, labels: list[str]
) -> dict[str, Any]:
    """Tính metric xác suất đa lớp để confidence dùng sau này có cơ sở kiểm tra."""
    label_indexes = {label: index for index, label in enumerate(labels)}
    one_hot = np.zeros((len(expected), len(labels)))
    for row_index, label in enumerate(expected):
        one_hot[row_index, label_indexes[label]] = 1.0
    metric_labels = sorted(labels)
    metric_probabilities = probabilities[
        :, [label_indexes[label] for label in metric_labels]
    ]
    confidence = probabilities.max(axis=1)
    correctness = np.asarray([actual == guessed for actual, guessed in zip(expected, predicted)])
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return {
        "log_loss": float(log_loss(expected, metric_probabilities, labels=metric_labels)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bins": ece,
        "roc_auc_ovr_macro": float(roc_auc_score(one_hot, probabilities, average="macro")),
        "pr_auc_macro": float(average_precision_score(one_hot, probabilities, average="macro")),
        "per_intent": {
            label: {
                "roc_auc": float(roc_auc_score(one_hot[:, index], probabilities[:, index])),
                "average_precision": float(
                    average_precision_score(one_hot[:, index], probabilities[:, index])
                ),
            }
            for index, label in enumerate(labels)
        },
    }


def write_validation_figures(
    expected: list[str],
    predicted: list[str],
    probabilities: Any,
    labels: list[str],
    output_dir: Path,
) -> list[str]:
    """Vẽ confusion, ROC, PR và reliability diagram từ validation, không dùng test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    raw_confusion = confusion_matrix(expected, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(raw_confusion, display_labels=labels).plot(ax=axis, colorbar=False)
    axis.set_title("Validation confusion matrix")
    figure.tight_layout()
    confusion_path = output_dir / "intent_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)
    paths.append(confusion_path)

    label_indexes = {label: index for index, label in enumerate(labels)}
    one_hot = np.zeros((len(expected), len(labels)))
    for row_index, label in enumerate(expected):
        one_hot[row_index, label_indexes[label]] = 1.0

    figure, axis = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(labels):
        false_positive_rate, true_positive_rate, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        axis.plot(false_positive_rate, true_positive_rate, label=label)
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Validation ROC (one-vs-rest)",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    roc_path = output_dir / "intent_roc_curve.png"
    figure.savefig(roc_path, dpi=160)
    plt.close(figure)
    paths.append(roc_path)

    figure, axis = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(one_hot[:, index], probabilities[:, index])
        axis.plot(recall, precision, label=label)
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Validation precision-recall (one-vs-rest)",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    pr_path = output_dir / "intent_pr_curve.png"
    figure.savefig(pr_path, dpi=160)
    plt.close(figure)
    paths.append(pr_path)

    confidence = probabilities.max(axis=1)
    correctness = np.asarray([actual == guessed for actual, guessed in zip(expected, predicted)])
    mean_confidence: list[float] = []
    empirical_accuracy: list[float] = []
    for lower, upper in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            mean_confidence.append(float(confidence[mask].mean()))
            empirical_accuracy.append(float(correctness[mask].mean()))
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    axis.plot(mean_confidence, empirical_accuracy, marker="o", label="classifier")
    axis.set(
        xlabel="Mean confidence",
        ylabel="Empirical accuracy",
        title="Validation reliability curve",
    )
    axis.legend()
    figure.tight_layout()
    calibration_path = output_dir / "intent_reliability_curve.png"
    figure.savefig(calibration_path, dpi=160)
    plt.close(figure)
    paths.append(calibration_path)
    portable_paths: list[str] = []
    for path in paths:
        try:
            reports_index = path.parts.index("reports")
            portable_paths.append(Path(*path.parts[reports_index:]).as_posix())
        except ValueError:
            portable_paths.append(path.name)
    return portable_paths


def evaluate_pipeline(
    pipeline: Pipeline[Any], samples: list[PreprocessedSample], figures_dir: Path | None = None
) -> dict[str, Any]:
    """Tính metric validation; tuyệt đối không dùng test để chọn candidate."""
    texts, expected = samples_to_xy(samples)
    raw_probability_rows = pipeline.predict_proba(texts)
    model_labels = [str(label) for label in pipeline.classes_]
    labels = [intent.value for intent in Intent if intent is not Intent.UNKNOWN]
    probabilities = np.asarray(
        [[row[model_labels.index(label)] for label in labels] for row in raw_probability_rows]
    )
    predicted = [labels[int(row.argmax())] for row in probabilities]
    failures = []
    for sample, expected_label, predicted_label, row in zip(
        samples, expected, predicted, probabilities
    ):
        if expected_label != predicted_label:
            failures.append(
                {
                    "id": sample.original.id,
                    "text": sample.normalized_text,
                    "region": sample.original.region.value,
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "confidence": float(max(row)),
                }
            )
    report: dict[str, Any] = {
        "total": len(expected),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "per_intent": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "labels": labels,
        "failures": failures,
        "probability_metrics": evaluate_probability_metrics(expected, predicted, probabilities, labels),
    }
    if figures_dir is not None:
        report["figures"] = write_validation_figures(
            expected, predicted, probabilities, labels, figures_dir
        )
    return report


def train_with_validation(
    train_samples: list[PreprocessedSample],
    validation_samples: list[PreprocessedSample],
    candidates: list[CandidateConfig],
    seed: int,
    figures_dir: Path | None = None,
) -> tuple[IntentClassifier, CandidateConfig, dict[str, Any]]:
    """Chọn candidate theo macro-F1 validation, sau đó refit train+validation cho artifact cuối."""
    train_texts, train_labels = samples_to_xy(train_samples)
    candidate_reports: list[dict[str, Any]] = []
    best_config: CandidateConfig | None = None
    best_report: dict[str, Any] | None = None
    best_score = (-1.0, -1.0)
    for config in candidates:
        pipeline = build_pipeline(config, seed)
        pipeline.fit(train_texts, train_labels)
        report = evaluate_pipeline(pipeline, validation_samples)
        candidate_reports.append({"config": config.__dict__, "validation": report})
        score = (float(report["macro_f1"]), float(report["accuracy"]))
        if score > best_score:
            best_config = config
            best_report = report
            best_score = score

    if best_config is None or best_report is None:
        raise RuntimeError("không chọn được candidate")
    selected_report = evaluate_pipeline(
        build_pipeline(best_config, seed).fit(train_texts, train_labels), validation_samples, figures_dir
    )
    all_texts, all_labels = samples_to_xy(train_samples + validation_samples)
    final_pipeline = build_pipeline(best_config, seed)
    final_pipeline.fit(all_texts, all_labels)
    report = {
        "selection_policy": {
            "primary": "macro_f1",
            "secondary": "accuracy",
            "final_tie_breaker": "candidate_order_in_config",
            "selected_candidate_index": candidates.index(best_config),
        },
        "selected_config": best_config.__dict__,
        "selected_validation": selected_report,
        "candidates": candidate_reports,
        "final_fit_samples": len(all_labels),
    }
    return IntentClassifier(final_pipeline), best_config, report


def save_classifier(
    classifier: IntentClassifier,
    model_path: Path,
    label_map_path: Path,
) -> None:
    """Lưu pipeline và label map riêng để CLI/API load có kiểm tra rõ ràng."""
    for component in classifier.pipeline.get_params(deep=True).values():
        if isinstance(component, TfidfVectorizer):
            # sklearn cache id(...) của stop_words; địa chỉ này thay đổi theo process.
            component.__dict__.pop("_stop_words_id", None)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier.pipeline, model_path)
    labels = sorted(str(label) for label in classifier.pipeline.classes_)
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    label_map_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
