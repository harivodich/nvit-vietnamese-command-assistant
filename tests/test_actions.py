import pytest

from nvit_assistant.actions import MockActionRouter
from nvit_assistant.schemas import ActionStatus, ActionType, Intent


@pytest.mark.parametrize(
    ("intent", "slots", "expected_type", "expected_payload"),
    [
        (
            Intent.SET_REMINDER,
            {"reminder_text": "uống thuốc", "datetime": "8 giờ"},
            ActionType.CREATE_REMINDER,
            {"reminder_text": "uống thuốc", "datetime": "8 giờ"},
        ),
        (
            Intent.SET_REMINDER,
            {"reminder_text": "gọi mẹ"},
            ActionType.CREATE_REMINDER,
            {"reminder_text": "gọi mẹ", "datetime": None},
        ),
        (
            Intent.SET_ALARM,
            {"datetime": "6 giờ sáng"},
            ActionType.SET_ALARM,
            {"datetime": "6 giờ sáng"},
        ),
        (
            Intent.ASK_WEATHER,
            {"location": "huế", "datetime": "ngày mai"},
            ActionType.QUERY_WEATHER,
            {"location": "huế", "datetime": "ngày mai"},
        ),
        (
            Intent.ASK_WEATHER,
            {},
            ActionType.QUERY_WEATHER,
            {"location": None, "datetime": None},
        ),
        (
            Intent.PLAY_MUSIC,
            {"song": "lạc trôi", "artist": "sơn tùng"},
            ActionType.PLAY_MUSIC,
            {"song": "lạc trôi", "artist": "sơn tùng"},
        ),
        (
            Intent.PLAY_MUSIC,
            {},
            ActionType.PLAY_MUSIC,
            {"song": None, "artist": None},
        ),
        (
            Intent.CALL_CONTACT,
            {"contact_name": "mẹ"},
            ActionType.CALL,
            {"contact_name": "mẹ", "phone_number": None, "target": "mẹ"},
        ),
        (
            Intent.CALL_CONTACT,
            {"phone_number": "090 000 0000"},
            ActionType.CALL,
            {"contact_name": None, "phone_number": "090 000 0000", "target": "090 000 0000"},
        ),
    ],
)
def test_mock_action_router_builds_deterministic_payload(
    intent: Intent,
    slots: dict[str, str],
    expected_type: ActionType,
    expected_payload: dict[str, object],
) -> None:
    execution = MockActionRouter().execute(intent, slots)

    assert execution.result.type is expected_type
    assert execution.result.status is ActionStatus.MOCKED
    assert execution.result.payload == expected_payload
    assert "giả lập" in execution.response


def test_mock_action_router_rejects_missing_required_slot() -> None:
    with pytest.raises(ValueError, match="reminder_text"):
        MockActionRouter().execute(Intent.SET_REMINDER, {})


def test_mock_action_router_rejects_unknown_intent() -> None:
    with pytest.raises(ValueError, match="không có action"):
        MockActionRouter().execute(Intent.UNKNOWN, {})
