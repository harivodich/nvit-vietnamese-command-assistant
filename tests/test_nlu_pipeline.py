from pathlib import Path

import pytest

from nvit_assistant.nlu.intent_classifier import load_classifier
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import Intent, ParseRequest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline() -> NLUPipeline:
    """Lắp pipeline bằng artifact thật của ngày 4 và config thật của dự án."""
    return NLUPipeline(
        VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml"),
        load_classifier(ROOT / "models" / "intent_classifier.joblib"),
        RegexSlotExtractor(ROOT / "configs" / "slot_values.yaml"),
    )


def test_pipeline_parses_scheduled_call_as_reminder(pipeline: NLUPipeline) -> None:
    result = pipeline.parse(ParseRequest(text="6h gọi mẹ"))
    assert result.intent is Intent.SET_REMINDER
    assert result.slots == {"datetime": "6h", "reminder_text": "gọi mẹ"}


def test_pipeline_normalizes_region_before_extracting_weather(pipeline: NLUPipeline) -> None:
    result = pipeline.parse(ParseRequest(text="Bữa ni ở Huế trời răng rồi hỉ"))
    assert result.intent is Intent.ASK_WEATHER
    assert result.slots["location"] == "huế"
    assert result.normalized_text == "hôm nay ở huế trời sao rồi nhỉ"


def test_pipeline_returns_clarification_when_required_slot_is_missing(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="đặt báo thức giúp tôi"))
    assert result.intent is Intent.SET_ALARM
    assert result.slots == {}
    assert result.response == "Cần bổ sung slot bắt buộc: datetime."
    assert "missing_required_slots:datetime" in result.matched_features
