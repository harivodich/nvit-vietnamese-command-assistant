"""Đánh giá trade-off coverage/safety của confidence gate trên validation."""

from __future__ import annotations

from typing import Any, Sequence


def _validate_predictions(
    expected: Sequence[str],
    predicted: Sequence[str],
    confidences: Sequence[float],
) -> None:
    """Kiểm tra ba dãy dự đoán trước khi tính các chỉ số confidence."""
    if not expected or len(expected) != len(predicted) or len(expected) != len(confidences):
        raise ValueError("expected, predicted và confidences phải cùng độ dài khác 0")


def evaluate_confidence_thresholds(
    expected: Sequence[str],
    predicted: Sequence[str],
    confidences: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Tính coverage và lỗi được chấp nhận cho từng ngưỡng, không đụng tới test."""
    _validate_predictions(expected, predicted, confidences)
    rows: list[dict[str, Any]] = []
    total = len(expected)
    expected_intents = sorted(set(expected))
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold phải nằm trong [0, 1]")
        accepted_indexes = [
            index for index, confidence in enumerate(confidences) if confidence >= threshold
        ]
        accepted_set = set(accepted_indexes)
        rejected_indexes = [index for index in range(total) if index not in accepted_set]
        accepted_correct = sum(expected[index] == predicted[index] for index in accepted_indexes)
        accepted_errors = len(accepted_indexes) - accepted_correct
        correct_rejected = sum(expected[index] == predicted[index] for index in rejected_indexes)
        per_expected_intent: dict[str, dict[str, int | float]] = {}
        for intent in expected_intents:
            intent_indexes = [index for index, label in enumerate(expected) if label == intent]
            intent_accepted = [index for index in intent_indexes if index in accepted_set]
            intent_rejected = [index for index in intent_indexes if index not in accepted_set]
            intent_accepted_correct = sum(
                expected[index] == predicted[index] for index in intent_accepted
            )
            per_expected_intent[intent] = {
                "total": len(intent_indexes),
                "accepted": len(intent_accepted),
                "rejected": len(intent_rejected),
                "coverage": len(intent_accepted) / len(intent_indexes),
                "selective_accuracy": (
                    intent_accepted_correct / len(intent_accepted) if intent_accepted else 0.0
                ),
                "correct_rejected": sum(
                    expected[index] == predicted[index] for index in intent_rejected
                ),
            }
        rows.append(
            {
                "threshold": threshold,
                "accepted": len(accepted_indexes),
                "rejected": len(rejected_indexes),
                "coverage": len(accepted_indexes) / total,
                "accepted_errors": accepted_errors,
                "correct_rejected": correct_rejected,
                "selective_accuracy": (
                    accepted_correct / len(accepted_indexes) if accepted_indexes else 0.0
                ),
                "per_expected_intent": per_expected_intent,
            }
        )
    return rows


def collect_accepted_failures(
    expected: Sequence[str],
    predicted: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
) -> list[dict[str, str | int | float]]:
    """Liệt kê lỗi vẫn vượt confidence gate để tiện điều tra từng mẫu."""
    _validate_predictions(expected, predicted, confidences)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold phải nằm trong [0, 1]")
    return [
        {
            "index": index,
            "expected": expected[index],
            "predicted": predicted[index],
            "confidence": float(confidences[index]),
        }
        for index in range(len(expected))
        if confidences[index] >= threshold and expected[index] != predicted[index]
    ]
