"""Đánh giá trade-off coverage/safety của confidence gate trên validation."""

from __future__ import annotations

from typing import Any, Sequence


def evaluate_confidence_thresholds(
    expected: Sequence[str],
    predicted: Sequence[str],
    confidences: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Tính coverage và lỗi được chấp nhận cho từng ngưỡng, không đụng tới test."""
    if not expected or len(expected) != len(predicted) or len(expected) != len(confidences):
        raise ValueError("expected, predicted và confidences phải cùng độ dài khác 0")
    rows: list[dict[str, Any]] = []
    total = len(expected)
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
            }
        )
    return rows
