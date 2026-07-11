"""Kiểm tra contract của các câu dễ nhầm intent, độc lập với test split."""

from pathlib import Path

import yaml

from nvit_assistant.nlu.intent_classifier import load_classifier
from nvit_assistant.schemas import Intent


ROOT = Path(__file__).resolve().parents[1]


def test_boundary_case_contract_has_unique_ids_and_known_intents() -> None:
    """Case ranh giới phải có nhãn hợp lệ để làm regression/audit ở các phase sau."""
    raw = yaml.safe_load((ROOT / "configs" / "intent_boundary_cases.yaml").read_text(encoding="utf-8"))
    cases = raw["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["intent"] for case in cases} <= {intent.value for intent in Intent}
    assert {case["id"] for case in cases} == {
        "immediate_contact",
        "scheduled_contact",
        "wake_up",
        "explicit_scheduled_contact",
    }


def test_trained_classifier_matches_reviewed_boundary_cases() -> None:
    """Artifact runtime phải qua toàn bộ acceptance case đã review."""
    cases = yaml.safe_load(
        (ROOT / "configs" / "intent_boundary_cases.yaml").read_text(encoding="utf-8")
    )["cases"]
    classifier = load_classifier(ROOT / "models" / "intent_classifier.joblib")
    failures = [
        f"{case['id']}: expected={case['intent']}, predicted={classifier.predict(case['text']).intent.value}"
        for case in cases
        if classifier.predict(case["text"]).intent.value != case["intent"]
    ]
    assert failures == []
