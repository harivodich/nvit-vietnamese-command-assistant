"""Regression nhỏ cho cue vùng miền/no-diacritics, không dùng test split chính thức."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from nvit_assistant.nlu.action_gate import CommandActionGate
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.runtime import build_pipeline
from nvit_assistant.schemas import ActionType, Intent, ParseRequest, Region


ROOT = Path(__file__).resolve().parents[1]
CASES: dict[str, list[dict[str, Any]]] = yaml.safe_load(
    (ROOT / "configs" / "action_gate_cases.yaml").read_text(encoding="utf-8")
)
ACTION_BY_INTENT = {
    Intent.SET_REMINDER: ActionType.CREATE_REMINDER,
    Intent.SET_ALARM: ActionType.SET_ALARM,
    Intent.ASK_WEATHER: ActionType.QUERY_WEATHER,
    Intent.PLAY_MUSIC: ActionType.PLAY_MUSIC,
    Intent.CALL_CONTACT: ActionType.CALL,
}


@pytest.fixture(scope="module")
def nlu_parts() -> tuple[VietnameseNormalizer, RegexSlotExtractor, CommandActionGate]:
    return (
        VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml"),
        RegexSlotExtractor(ROOT / "configs" / "slot_values.yaml"),
        CommandActionGate(),
    )


@pytest.fixture(scope="module")
def runtime_pipeline() -> NLUPipeline:
    return build_pipeline(ROOT)


def test_positive_cases_cover_every_intent_for_each_text_variant() -> None:
    actual = {(case["variant"], case["intent"]) for case in CASES["positive_cases"]}
    expected = {
        (variant, intent.value)
        for variant in ("north", "central", "south", "no_diacritics")
        for intent in ACTION_BY_INTENT
    }

    assert actual == expected


@pytest.mark.parametrize(
    "case", CASES["positive_cases"], ids=lambda case: str(case["id"])
)
def test_reviewed_regional_and_plain_commands_complete_runtime(
    case: dict[str, Any],
    runtime_pipeline: NLUPipeline,
) -> None:
    intent = Intent(case["intent"])
    result = runtime_pipeline.parse(ParseRequest(text=case["text"]))

    assert result.region is Region(case["region"])
    assert result.intent is intent
    assert result.action is not None
    assert result.action.type is ACTION_BY_INTENT[intent]


@pytest.mark.parametrize(
    "case", CASES["negative_cases"], ids=lambda case: str(case["id"])
)
def test_reviewed_negative_counterparts_remain_blocked(
    case: dict[str, Any],
    nlu_parts: tuple[VietnameseNormalizer, RegexSlotExtractor, CommandActionGate],
) -> None:
    normalizer, extractor, gate = nlu_parts
    intent = Intent(case["intent"])
    normalized = normalizer.normalize(case["text"])
    slots = extractor.extract(normalized.normalized_text, intent).slots

    decision = gate.check(normalized.normalized_text, intent, slots)

    assert not decision.allowed
    assert decision.reason == case["reason"]
