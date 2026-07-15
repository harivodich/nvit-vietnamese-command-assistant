"""Kiểm tra contract của các câu dễ nhầm intent, độc lập với test split."""

from pathlib import Path

import yaml

from nvit_assistant.runtime import build_pipeline
from nvit_assistant.schemas import Intent, ParseRequest


ROOT = Path(__file__).resolve().parents[1]


def test_boundary_case_contract_has_unique_ids_and_known_intents() -> None:
    """Case ranh giới phải có nhãn hợp lệ để làm regression/audit ở các phase sau."""
    raw = yaml.safe_load(
        (ROOT / "configs" / "intent_boundary_cases.yaml").read_text(encoding="utf-8")
    )
    cases = raw["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["intent"] for case in cases} <= {intent.value for intent in Intent}
    assert {case["id"] for case in cases} == {
        "immediate_contact",
        "immediate_contact_named_mai",
        "immediate_phone",
        "phone_before_verb",
        "scheduled_contact",
        "scheduled_contact_suffix",
        "scheduled_contact_prefix",
        "scheduled_phone",
        "wake_up",
        "explicit_scheduled_contact",
    }


def test_runtime_pipeline_matches_reviewed_boundary_cases() -> None:
    """Pipeline hoàn chỉnh phải qua các case vì boundary rule nằm sau classifier."""
    cases = yaml.safe_load(
        (ROOT / "configs" / "intent_boundary_cases.yaml").read_text(encoding="utf-8")
    )["cases"]
    pipeline = build_pipeline(ROOT)
    failures = [
        f"{case['id']}: expected={case['intent']}, "
        f"predicted={pipeline.parse(ParseRequest(text=case['text'])).intent.value}"
        for case in cases
        if pipeline.parse(ParseRequest(text=case["text"])).intent.value != case["intent"]
    ]
    assert failures == []
