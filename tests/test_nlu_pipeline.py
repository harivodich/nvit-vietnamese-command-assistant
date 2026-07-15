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
    """Lắp pipeline bằng artifact intent và config thật của dự án."""
    return NLUPipeline(
        VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml"),
        load_classifier(ROOT / "models" / "intent_classifier.joblib"),
        RegexSlotExtractor(ROOT / "configs" / "slot_values.yaml"),
    )


def test_pipeline_parses_scheduled_call_as_reminder(pipeline: NLUPipeline) -> None:
    result = pipeline.parse(ParseRequest(text="6h gọi mẹ"))
    assert result.intent is Intent.SET_REMINDER
    assert result.slots == {"datetime": "6h", "reminder_text": "gọi mẹ"}


@pytest.mark.parametrize(
    ("text", "expected_reminder"),
    [
        ("gọi cho mẹ lúc 6 giờ", "gọi cho mẹ"),
        ("lúc 6 giờ gọi cho mẹ", "gọi cho mẹ"),
        ("lúc 6 giờ gọi cho hoàng", "gọi cho hoàng"),
        ("gọi số 0901234567 lúc 6 giờ", "gọi số 0901234567"),
    ],
)
def test_pipeline_routes_scheduled_calls_to_reminder_regardless_of_word_order(
    pipeline: NLUPipeline, text: str, expected_reminder: str
) -> None:
    result = pipeline.parse(ParseRequest(text=text))

    assert result.intent is Intent.SET_REMINDER
    assert result.slots == {"datetime": "6 giờ", "reminder_text": expected_reminder}


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
    assert result.response == "Bạn muốn đặt báo thức lúc nào?"
    assert "missing_required_slots:datetime" in result.matched_features


def test_pipeline_asks_for_reminder_content_instead_of_using_pronoun_as_slot(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="nhắc tôi"))

    assert result.intent is Intent.SET_REMINDER
    assert result.slots == {}
    assert result.action is None
    assert result.response == "Bạn muốn mình nhắc việc gì?"
    assert "missing_required_slots:reminder_text" in result.matched_features


def test_pipeline_routes_wake_up_phrase_to_alarm_and_asks_for_time(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="gọi tôi dậy"))

    assert result.intent is Intent.SET_ALARM
    assert result.slots == {}
    assert result.response == "Bạn muốn đặt báo thức lúc nào?"
    assert any(feature.endswith("->set_alarm") for feature in result.matched_features)


def test_pipeline_accepts_keu_as_alarm_boundary(pipeline: NLUPipeline) -> None:
    result = pipeline.parse(ParseRequest(text="kêu tôi dậy lúc 6 giờ"))

    assert result.intent is Intent.SET_ALARM
    assert result.slots == {"datetime": "6 giờ"}


@pytest.mark.parametrize("text", ["gọi dậy", "gọi em dậy", "kêu con dậy"])
def test_pipeline_wake_up_without_time_asks_for_datetime(pipeline: NLUPipeline, text: str) -> None:
    result = pipeline.parse(ParseRequest(text=text))

    assert result.intent is Intent.SET_ALARM
    assert result.action is None
    assert result.response == "Bạn muốn đặt báo thức lúc nào?"


def test_pipeline_does_not_call_contact_in_wake_up_phrase(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="gọi mẹ dậy"))

    assert result.intent is Intent.SET_ALARM
    assert result.action is None
    assert result.response == "Bạn muốn đặt báo thức lúc nào?"


def test_pipeline_routes_immediate_call_back_to_call_contact(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="gọi lại cho khách hàng"))

    assert result.intent is Intent.CALL_CONTACT
    assert result.slots == {"contact_name": "khách hàng"}


def test_pipeline_routes_spoken_phone_to_call_even_when_model_confuses_intent(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="gọi số không chín không một hai ba bốn năm sáu bảy"))

    assert result.intent is Intent.CALL_CONTACT
    assert result.slots == {"phone_number": "0901234567"}


@pytest.mark.parametrize(
    "text",
    [
        "gọi số 0901234567",
        "0901234567 gọi đi",
        "quay số 0901234567",
        "bấm số 0901234567 giúp tôi",
        "goi so 0901234567",
    ],
)
def test_pipeline_routes_immediate_numeric_calls_to_contact(
    pipeline: NLUPipeline, text: str
) -> None:
    result = pipeline.parse(ParseRequest(text=text))

    assert result.intent is Intent.CALL_CONTACT
    assert result.slots == {"phone_number": "0901234567"}


def test_pipeline_keeps_explicit_now_call_immediate(pipeline: NLUPipeline) -> None:
    result = pipeline.parse(ParseRequest(text="gọi mẹ bây giờ"))

    assert result.intent is Intent.CALL_CONTACT
    assert result.slots == {"contact_name": "mẹ"}


def test_pipeline_does_not_treat_mai_in_contact_name_as_datetime(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="nối máy với chị mai giúp tớ"))

    assert result.intent is Intent.CALL_CONTACT
    assert result.slots == {"contact_name": "chị mai"}


def test_pipeline_keeps_dont_forget_call_as_reminder(
    pipeline: NLUPipeline,
) -> None:
    result = pipeline.parse(ParseRequest(text="đừng quên gọi mẹ lúc 6 giờ"))

    assert result.intent is Intent.SET_REMINDER
    assert result.slots == {"datetime": "6 giờ", "reminder_text": "gọi mẹ"}


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
