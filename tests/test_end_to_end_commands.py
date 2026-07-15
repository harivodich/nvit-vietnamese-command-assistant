from pathlib import Path

import pytest

from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.runtime import build_pipeline
from nvit_assistant.schemas import ActionType, Intent, ParseRequest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime_pipeline() -> NLUPipeline:
    """Nạp runtime thật một lần để kiểm tra đủ năm nhánh end-to-end."""
    return build_pipeline(ROOT)


@pytest.mark.parametrize(
    ("text", "expected_intent", "expected_action", "expected_slots"),
    [
        (
            "nhắc tôi uống thuốc vào tối nay",
            Intent.SET_REMINDER,
            ActionType.CREATE_REMINDER,
            {"datetime": "tối nay", "reminder_text": "uống thuốc"},
        ),
        (
            "gọi tôi dậy lúc 7 giờ",
            Intent.SET_ALARM,
            ActionType.SET_ALARM,
            {"datetime": "7 giờ"},
        ),
        (
            "thời tiết hà nội ngày mai",
            Intent.ASK_WEATHER,
            ActionType.QUERY_WEATHER,
            {"datetime": "ngày mai", "location": "hà nội"},
        ),
        (
            "mở bài lạc trôi của sơn tùng",
            Intent.PLAY_MUSIC,
            ActionType.PLAY_MUSIC,
            {"song": "lạc trôi", "artist": "sơn tùng"},
        ),
        (
            "gọi cho mẹ",
            Intent.CALL_CONTACT,
            ActionType.CALL,
            {"contact_name": "mẹ"},
        ),
    ],
)
def test_complete_runtime_routes_all_intents(
    runtime_pipeline: NLUPipeline,
    text: str,
    expected_intent: Intent,
    expected_action: ActionType,
    expected_slots: dict[str, str],
) -> None:
    result = runtime_pipeline.parse(ParseRequest(text=text))

    assert result.intent is expected_intent
    assert result.confidence >= runtime_pipeline.confidence_threshold
    assert result.slots == expected_slots
    assert result.action is not None
    assert result.action.type is expected_action
    assert "giả lập" in result.response


@pytest.mark.parametrize(
    "text",
    ["thời tiết ở đâu", "thời tiết ở đây", "thời tiết ở chỗ nào", "thời tiết ở nơi nào"],
)
def test_runtime_asks_for_a_real_weather_location(
    runtime_pipeline: NLUPipeline, text: str
) -> None:
    result = runtime_pipeline.parse(ParseRequest(text=text))

    assert result.intent is Intent.ASK_WEATHER
    assert result.slots == {}
    assert result.action is None
    assert result.response == "Bạn muốn xem thời tiết ở tỉnh hoặc thành phố nào?"
    assert "location_clarification:placeholder" in result.matched_features
