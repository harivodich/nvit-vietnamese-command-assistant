"""Suy luận intent gọn nhẹ, không import thư viện vẽ biểu đồ của bước train."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib

from nvit_assistant.schemas import Intent


@dataclass(frozen=True)
class IntentPrediction:
    """Intent, confidence lớn nhất và phân bố xác suất của một câu."""

    intent: Intent
    confidence: float
    probabilities: dict[str, float]


class IntentPredictor(Protocol):
    """Contract tối thiểu để pipeline nhận model sklearn hoặc test double."""

    def predict(self, text: str) -> IntentPrediction:
        """Dự đoán trên text đã normalize."""
        ...


class IntentClassifier:
    """Wrapper inference cho sklearn pipeline đã được train và lưu bằng joblib."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    @property
    def labels(self) -> tuple[str, ...]:
        """Đọc label nằm trong artifact để đối chiếu với contract runtime."""
        return tuple(str(label) for label in self.pipeline.classes_)

    def predict(self, text: str) -> IntentPrediction:
        """Chọn intent có xác suất lớn nhất và giữ toàn bộ xác suất để audit."""
        probabilities = self.pipeline.predict_proba([text])[0]
        probability_map = {
            label: float(score) for label, score in zip(self.labels, probabilities)
        }
        best_label = max(probability_map, key=probability_map.__getitem__)
        return IntentPrediction(
            intent=Intent(best_label),
            confidence=probability_map[best_label],
            probabilities=probability_map,
        )


def load_classifier(model_path: Path) -> IntentClassifier:
    """Nạp artifact cục bộ đáng tin cậy; joblib/pickle không dùng cho file không rõ nguồn."""
    return IntentClassifier(joblib.load(model_path))
