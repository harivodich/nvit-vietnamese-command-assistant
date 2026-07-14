from pathlib import Path

import pytest

from nvit_assistant.actions import MockActionRouter
from nvit_assistant.nlu.intent_classifier import IntentClassifier
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


def test_pipeline_executes_mock_action_only_when_slots_are_complete(
    pipeline: NLUPipeline,
) -> None:
    pipeline.action_router = MockActionRouter()
    result = pipeline.parse(ParseRequest(text="gọi cho mẹ"))

    assert result.action is not None
    assert result.action.type.value == "call"
    assert result.action.status.value == "mocked"


class _LowConfidenceSklearnPipeline:
    """Giả lập đúng interface sklearn để kiểm tra confidence gate độc lập model thật."""

    classes_ = ["ask_weather", "call_contact", "play_music", "set_alarm", "set_reminder"]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        assert texts
        return [[0.30, 0.20, 0.20, 0.15, 0.15]]


def test_pipeline_rejects_low_confidence_before_slot_and_action() -> None:
    guarded_pipeline = NLUPipeline(
        VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml"),
        IntentClassifier(_LowConfidenceSklearnPipeline()),
        RegexSlotExtractor(ROOT / "configs" / "slot_values.yaml"),
        action_router=MockActionRouter(),
        confidence_threshold=0.45,
    )

    result = guarded_pipeline.parse(ParseRequest(text="làm cái đó đi"))

    assert result.intent is Intent.UNKNOWN
    assert result.slots == {}
    assert result.action is None
    assert "intent_rejected:ask_weather" in result.matched_features
