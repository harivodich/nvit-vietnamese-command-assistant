import pytest

from nvit_assistant.nlu.action_gate import CommandActionGate
from nvit_assistant.schemas import Intent


def test_action_gate_rejects_negated_command() -> None:
    decision = CommandActionGate().check("đừng gọi cho mẹ", Intent.CALL_CONTACT, {})

    assert not decision.allowed
    assert decision.reason == "negated_command"


def test_action_gate_keeps_weather_question_with_khong() -> None:
    decision = CommandActionGate().check("hôm nay có mưa không", Intent.ASK_WEATHER, {})

    assert decision.allowed


def test_action_gate_rejects_out_of_domain_high_confidence_guess() -> None:
    decision = CommandActionGate().check("hôm nay tôi ăn cơm", Intent.ASK_WEATHER, {})

    assert not decision.allowed
    assert decision.reason == "missing_intent_cue"


def test_action_gate_keeps_positive_dont_forget_reminder() -> None:
    decision = CommandActionGate().check(
        "đừng quên nhắc tôi uống thuốc", Intent.SET_REMINDER, {"reminder_text": "uống thuốc"}
    )

    assert decision.allowed


def test_action_gate_supports_no_diacritics_command() -> None:
    decision = CommandActionGate().check(
        "goi dien cho me", Intent.CALL_CONTACT, {"contact_name": "mẹ"}
    )

    assert decision.allowed


def test_action_gate_does_not_confuse_dung_question_with_dung_command() -> None:
    decision = CommandActionGate().check("tuần này có tuyết đúng không", Intent.ASK_WEATHER, {})

    assert decision.allowed


def test_action_gate_rejects_non_action_statement() -> None:
    decision = CommandActionGate().check("tôi đang nghe nhạc", Intent.PLAY_MUSIC, {})

    assert not decision.allowed
    assert decision.reason == "non_action_statement"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("chưa cần nhắc tôi uống thuốc", Intent.SET_REMINDER),
        ("không cần báo thức lúc 7 giờ", Intent.SET_ALARM),
        ("dung hoi thoi tiet hom nay", Intent.ASK_WEATHER),
        ("khong can xem du bao thoi tiet", Intent.ASK_WEATHER),
        ("không cần thời tiết hôm nay", Intent.ASK_WEATHER),
        ("không cần lời nhắc nữa", Intent.SET_REMINDER),
        ("không muốn nhạc", Intent.PLAY_MUSIC),
        ("không muốn thời tiết", Intent.ASK_WEATHER),
    ],
)
def test_action_gate_rejects_extended_negation(text: str, intent: Intent) -> None:
    assert not CommandActionGate().check(text, intent, {}).allowed


def test_action_gate_rejects_unsupported_remove_alarm() -> None:
    decision = CommandActionGate().check("bỏ báo thức 7 giờ", Intent.SET_ALARM, {})

    assert not decision.allowed
    assert decision.reason == "unsupported_operation"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("hôm qua trời mưa", Intent.ASK_WEATHER),
        ("mẹ đang nghe nhạc", Intent.PLAY_MUSIC),
        ("cô ấy thích nhạc jazz", Intent.PLAY_MUSIC),
        ("trời đẹp", Intent.ASK_WEATHER),
        ("trời đang mưa", Intent.ASK_WEATHER),
        ("mẹ nghe nhạc", Intent.PLAY_MUSIC),
        ("tôi nghe nhạc mỗi ngày", Intent.PLAY_MUSIC),
        ("tôi uống thuốc", Intent.SET_REMINDER),
        ("tôi mua sữa", Intent.SET_REMINDER),
        ("tôi gọi mẹ", Intent.CALL_CONTACT),
    ],
)
def test_action_gate_rejects_third_person_or_historical_statement(
    text: str, intent: Intent
) -> None:
    decision = CommandActionGate().check(text, intent, {})

    assert not decision.allowed
    assert decision.reason == "non_action_statement"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("nhạc đang phát", Intent.PLAY_MUSIC),
        ("bài hát này hay", Intent.PLAY_MUSIC),
        ("nhiệt độ là 30 độ", Intent.ASK_WEATHER),
        ("hà nội đang mưa", Intent.ASK_WEATHER),
        ("ngoài trời đang mưa", Intent.ASK_WEATHER),
        ("mai có cuộc họp", Intent.SET_REMINDER),
    ],
)
def test_action_gate_rejects_clear_present_tense_statement(
    text: str, intent: Intent
) -> None:
    decision = CommandActionGate().check(text, intent, {})

    assert not decision.allowed
    assert decision.reason == "non_action_statement"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("gọi mẹ chưa", Intent.CALL_CONTACT),
        ("đã gọi mẹ chưa", Intent.CALL_CONTACT),
        ("gọi mẹ hay bố", Intent.CALL_CONTACT),
        ("mở nhạc chưa", Intent.PLAY_MUSIC),
        ("đã mở nhạc chưa", Intent.PLAY_MUSIC),
        ("có mở nhạc không", Intent.PLAY_MUSIC),
        ("nhạc đang phát à", Intent.PLAY_MUSIC),
        ("đang phát nhạc à", Intent.PLAY_MUSIC),
        ("có cần nhắc tôi uống thuốc không", Intent.SET_REMINDER),
        ("đã nhắc tôi uống thuốc chưa", Intent.SET_REMINDER),
        ("bài này tên gì", Intent.PLAY_MUSIC),
        ("mở nhạc phải không", Intent.PLAY_MUSIC),
        ("nhạc có đang phát không", Intent.PLAY_MUSIC),
        ("bài này của ai", Intent.PLAY_MUSIC),
        ("nhắc tôi uống thuốc phải không", Intent.SET_REMINDER),
    ],
)
def test_action_gate_rejects_status_or_choice_question(
    text: str, intent: Intent
) -> None:
    decision = CommandActionGate().check(text, intent, {})

    assert not decision.allowed
    assert decision.reason == "status_or_choice_question"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("hà nội có mưa không", Intent.ASK_WEATHER),
        ("gọi cho mẹ được không", Intent.CALL_CONTACT),
        ("mở nhạc giúp tôi được không", Intent.PLAY_MUSIC),
    ],
)
def test_action_gate_keeps_supported_question_shaped_command(
    text: str, intent: Intent
) -> None:
    assert CommandActionGate().check(text, intent, {}).allowed


def test_action_gate_keeps_dont_forget_scheduled_call() -> None:
    decision = CommandActionGate().check(
        "đừng quên gọi mẹ lúc 6 giờ",
        Intent.SET_REMINDER,
        {"datetime": "6 giờ", "reminder_text": "gọi mẹ"},
    )

    assert decision.allowed


@pytest.mark.parametrize(
    ("text", "intent", "slots"),
    [
        ("hẹn giờ tới 6 giờ nhé", Intent.SET_ALARM, {"datetime": "6 giờ"}),
        ("bấm số 0901234567 giúp tôi", Intent.CALL_CONTACT, {"phone_number": "0901234567"}),
        ("lien lac voi me giup toi", Intent.CALL_CONTACT, {"contact_name": "mẹ"}),
        ("bây giờ ngoài huế trời sao rồi", Intent.ASK_WEATHER, {"location": "huế"}),
        ("mở bài lạc trôi", Intent.PLAY_MUSIC, {"song": "lạc trôi"}),
    ],
)
def test_action_gate_keeps_legitimate_development_commands(
    text: str, intent: Intent, slots: dict[str, str]
) -> None:
    decision = CommandActionGate().check(text, intent, slots)

    assert decision.allowed
    assert decision.reason == "intent_cue"


@pytest.mark.parametrize(
    ("text", "intent", "reason"),
    [
        ("tôi hẹn giờ mỗi ngày", Intent.SET_ALARM, "non_action_statement"),
        ("tôi liên lạc với mẹ", Intent.CALL_CONTACT, "non_action_statement"),
        ("tôi bấm số 0901234567", Intent.CALL_CONTACT, "non_action_statement"),
        ("tôi sẽ gọi mẹ lúc 6 giờ", Intent.SET_REMINDER, "non_action_statement"),
        ("đã bấm số 0901234567 chưa", Intent.CALL_CONTACT, "status_or_choice_question"),
        ("mở bài lạc trôi chưa", Intent.PLAY_MUSIC, "status_or_choice_question"),
    ],
)
def test_new_action_cues_do_not_open_statements_or_status_questions(
    text: str, intent: Intent, reason: str
) -> None:
    decision = CommandActionGate().check(text, intent, {})

    assert not decision.allowed
    assert decision.reason == reason


def test_scheduled_third_person_statement_is_not_a_reminder_command() -> None:
    decision = CommandActionGate().check(
        "lúc 6 giờ mẹ gọi tôi",
        Intent.SET_REMINDER,
        {"datetime": "6 giờ", "reminder_text": "mẹ gọi tôi"},
    )

    assert not decision.allowed
    assert decision.reason == "non_action_statement"
