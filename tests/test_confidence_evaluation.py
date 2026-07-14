import pytest

from nvit_assistant.eval.confidence_evaluation import evaluate_confidence_thresholds


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


def test_confidence_threshold_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="cùng độ dài"):
        evaluate_confidence_thresholds(["a"], [], [0.5], [0.5])
