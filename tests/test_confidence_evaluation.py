import pytest

from nvit_assistant.eval.confidence_evaluation import (
    collect_accepted_failures,
    evaluate_confidence_thresholds,
)


def test_confidence_threshold_reports_coverage_and_selective_accuracy() -> None:
    rows = evaluate_confidence_thresholds(
        expected=["a", "b", "c"],
        predicted=["a", "x", "c"],
        confidences=[0.9, 0.4, 0.3],
        thresholds=[0.0, 0.5],
    )

    assert rows[0]["coverage"] == 1.0
    assert rows[0]["accepted_errors"] == 1
    assert rows[1]["coverage"] == pytest.approx(1 / 3)
    assert rows[1]["accepted_errors"] == 0
    assert rows[1]["correct_rejected"] == 1
    assert rows[1]["selective_accuracy"] == 1.0
    assert rows[1]["per_expected_intent"] == {
        "a": {
            "total": 1,
            "accepted": 1,
            "rejected": 0,
            "coverage": 1.0,
            "selective_accuracy": 1.0,
            "correct_rejected": 0,
        },
        "b": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "coverage": 0.0,
            "selective_accuracy": 0.0,
            "correct_rejected": 0,
        },
        "c": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "coverage": 0.0,
            "selective_accuracy": 0.0,
            "correct_rejected": 1,
        },
    }


def test_confidence_threshold_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="cùng độ dài"):
        evaluate_confidence_thresholds(["a"], [], [0.5], [0.5])


def test_collect_accepted_failures_returns_traceable_rows() -> None:
    failures = collect_accepted_failures(
        expected=["weather", "call", "music", "alarm"],
        predicted=["weather", "music", "call", "reminder"],
        confidences=[0.8, 0.7, 0.4, 0.5],
        threshold=0.5,
    )

    assert failures == [
        {
            "index": 1,
            "expected": "call",
            "predicted": "music",
            "confidence": 0.7,
        },
        {
            "index": 3,
            "expected": "alarm",
            "predicted": "reminder",
            "confidence": 0.5,
        },
    ]
