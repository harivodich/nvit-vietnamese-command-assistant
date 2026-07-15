from pathlib import Path

from nvit_assistant.eval.action_safety_evaluation import (
    evaluate_action_safety,
    load_action_safety_challenge,
)
from nvit_assistant.runtime import build_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_passes_reviewed_action_safety_challenge() -> None:
    report = evaluate_action_safety(
        build_pipeline(ROOT),
        load_action_safety_challenge(ROOT / "data" / "action_safety_challenge.jsonl"),
    )

    assert report["false_action_rate"] == 0.0
    assert report["positive_action_recall"] == 1.0
    assert report["failures"] == []
