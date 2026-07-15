"""Huấn luyện intent classifier trên sentence embedding E5 đã tải cục bộ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from nvit_assistant.nlu.intent_classifier import evaluate_pipeline, samples_to_xy
from nvit_assistant.schemas import PreprocessedSample


@dataclass(frozen=True)
class SemanticCandidateConfig:
    """Một regularization candidate cho classifier đặt trên embedding cố định."""

    name: str
    c: float


def encode_texts(encoder_dir: Path, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Mã hóa câu bằng E5 local; cùng prefix `query:` cho mọi split classification."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("cần cài sentence-transformers để chạy semantic benchmark") from exc
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"không thấy E5 local tại: {encoder_dir}")
    encoder = SentenceTransformer(str(encoder_dir), device="cpu")
    return np.asarray(
        encoder.encode(
            [f"query: {text}" for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )


def train_semantic_with_validation(
    train_samples: list[PreprocessedSample],
    validation_samples: list[PreprocessedSample],
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    candidates: list[SemanticCandidateConfig],
    seed: int,
    figures_dir: Path | None = None,
) -> tuple[LogisticRegression, SemanticCandidateConfig, dict[str, Any]]:
    """Chọn LR trên validation, sau đó caller refit artifact cuối bằng train+validation."""
    _, train_labels = samples_to_xy(train_samples)
    _, validation_labels = samples_to_xy(validation_samples)
    if len(train_embeddings) != len(train_labels) or len(validation_embeddings) != len(validation_labels):
        raise ValueError("số embedding phải khớp số sample của từng split")
    reports: list[dict[str, Any]] = []
    selected: SemanticCandidateConfig | None = None
    selected_report: dict[str, Any] | None = None
    best_score = (-1.0, -1.0)
    for candidate in candidates:
        model = LogisticRegression(C=candidate.c, max_iter=2000, random_state=seed)
        model.fit(train_embeddings, train_labels)
        report = evaluate_embeddings(model, validation_samples, validation_embeddings)
        reports.append({"config": candidate.__dict__, "validation": report})
        score = (float(report["macro_f1"]), float(report["accuracy"]))
        if score > best_score:
            selected, selected_report, best_score = candidate, report, score
    if selected is None or selected_report is None:
        raise RuntimeError("không có semantic candidate")
    model = LogisticRegression(C=selected.c, max_iter=2000, random_state=seed)
    model.fit(train_embeddings, train_labels)
    selected_report = evaluate_embeddings(model, validation_samples, validation_embeddings, figures_dir)
    return model, selected, {"selected_config": selected.__dict__, "selected_validation": selected_report, "candidates": reports}


def evaluate_embeddings(
    model: LogisticRegression,
    samples: list[PreprocessedSample],
    embeddings: np.ndarray,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Dùng cùng evaluator với baseline, chỉ thay input từ text thành embedding."""
    class EmbeddingPipeline:
        def __init__(self, classifier: LogisticRegression, matrix: np.ndarray) -> None:
            self.classifier = classifier
            self.matrix = matrix
            self.classes_ = classifier.classes_

        def predict_proba(self, _: list[str]) -> np.ndarray:
            return cast(np.ndarray, self.classifier.predict_proba(self.matrix))

    return evaluate_pipeline(cast(Any, EmbeddingPipeline(model, embeddings)), samples, figures_dir)
