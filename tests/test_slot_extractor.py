from pathlib import Path

import pytest

from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import Intent


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def extractor() -> RegexSlotExtractor:
    """Dùng đúng config runtime để test không lệch từ điển dự án."""
    return RegexSlotExtractor(ROOT / "configs" / "slot_values.yaml")


@pytest.mark.parametrize(
    ("text", "intent", "expected"),
    [
        ("6h gọi mẹ", Intent.SET_REMINDER, {"datetime": "6h", "reminder_text": "gọi mẹ"}),
        ("nhắc tôi uống thuốc vào tối nay", Intent.SET_REMINDER, {"datetime": "tối nay", "reminder_text": "uống thuốc"}),
        ("gọi tôi dậy lúc 7 giờ", Intent.SET_ALARM, {"datetime": "7 giờ"}),
        ("thời tiết hà nội ngày mai", Intent.ASK_WEATHER, {"datetime": "ngày mai", "location": "hà nội"}),
        ("mở bài lạc trôi của sơn tùng", Intent.PLAY_MUSIC, {"song": "lạc trôi", "artist": "sơn tùng"}),
        ("gọi cho mẹ", Intent.CALL_CONTACT, {"contact_name": "mẹ"}),
        ("gọi số 090 000 0000", Intent.CALL_CONTACT, {"phone_number": "090 000 0000"}),
        ("báo thức lúc 7 giờ kém 15", Intent.SET_ALARM, {"datetime": "7 giờ kém 15"}),
        ("đặt báo thức 6 giờ rưỡi", Intent.SET_ALARM, {"datetime": "6 giờ rưỡi"}),
        (
            "gọi số không chín không một hai ba bốn năm sáu bảy",
            Intent.CALL_CONTACT,
            {"phone_number": "0901234567"},
        ),
        ("gọi cho hương", Intent.CALL_CONTACT, {"contact_name": "hương"}),
        (
            "phát bài con đường mưa của cao thái sơn",
            Intent.PLAY_MUSIC,
            {"song": "con đường mưa", "artist": "cao thái sơn"},
        ),
        (
            "nhắc tôi mang hồ sơ đi phỏng vấn vào ngày mai",
            Intent.SET_REMINDER,
            {"datetime": "ngày mai", "reminder_text": "mang hồ sơ đi phỏng vấn"},
        ),
        (
            "thời tiết ở nam định ngày mai",
            Intent.ASK_WEATHER,
            {"datetime": "ngày mai", "location": "nam định"},
        ),
        ("mở danh sách phát của tôi", Intent.PLAY_MUSIC, {}),
        (
            "hom nay luc 6 gio chieu tuoi cay",
            Intent.SET_REMINDER,
            {"datetime": "hôm nay lúc 6 giờ chiều", "reminder_text": "tưới cây"},
        ),
        (
            "o vinh long bay gio co mua khong",
            Intent.ASK_WEATHER,
            {"datetime": "bây giờ", "location": "vĩnh long"},
        ),
        ("goi dien cho chi linh", Intent.CALL_CONTACT, {"contact_name": "chị linh"}),
    ],
)
def test_extract_slots_by_intent(
    extractor: RegexSlotExtractor,
    text: str,
    intent: Intent,
    expected: dict[str, str],
) -> None:
    assert extractor.extract(text, intent).slots == expected


@pytest.mark.parametrize(
    "text",
    ["nhắc tôi", "nhắc giúp tôi", "tạo lời nhắc", "đặt lời nhắc", "nhắc"],
)
def test_reminder_trigger_without_content_is_not_a_slot(
    extractor: RegexSlotExtractor, text: str
) -> None:
    assert extractor.extract(text, Intent.SET_REMINDER).slots == {}


@pytest.mark.parametrize(
    "text",
    [
        "gọi giúp tôi",
        "gọi đi",
        "gọi ngay",
        "gọi cho",
        "gọi điện giúp tôi",
        "gọi điện cho tôi",
        "gọi số 123",
        "gọi 12345",
        "gọi đến số 123",
        "gọi +84",
        "gọi ai",
        "gọi cho ai",
        "gọi người nào",
        "gọi cho ai vậy",
    ],
)
def test_call_trigger_without_target_is_not_a_contact(
    extractor: RegexSlotExtractor, text: str
) -> None:
    assert extractor.extract(text, Intent.CALL_CONTACT).slots == {}


@pytest.mark.parametrize(
    "text",
    [
        "mở nhạc đi",
        "mở nhạc ngay",
        "phát nhạc lên",
        "mở nhạc chưa",
        "mở nhạc không",
        "nhạc đang phát à",
    ],
)
def test_music_discourse_particle_is_not_an_artist(
    extractor: RegexSlotExtractor, text: str
) -> None:
    assert extractor.extract(text, Intent.PLAY_MUSIC).slots == {}


def test_call_back_cue_is_not_part_of_unknown_contact(
    extractor: RegexSlotExtractor,
) -> None:
    assert extractor.extract("gọi lại cho khách hàng", Intent.CALL_CONTACT).slots == {
        "contact_name": "khách hàng"
    }


@pytest.mark.parametrize("text", ["nhắc tôi lúc 8 giờ", "nhắc lúc 8 giờ", "6h gọi", "gọi lúc 8 giờ", "gọi lại"])
def test_incomplete_reminder_action_is_not_content(
    extractor: RegexSlotExtractor, text: str
) -> None:
    assert "reminder_text" not in extractor.extract(text, Intent.SET_REMINDER).slots


@pytest.mark.parametrize(
    "text",
    [
        "nhắc tôi nhé",
        "nhắc tôi nha",
        "nhắc tôi với",
        "nhắc tôi đi",
        "nhắc tôi được không",
        "nhắc tôi lúc 8 giờ nhé",
        "nhắc tôi lúc 8 giờ ngay",
    ],
)
def test_polite_particle_alone_is_not_reminder_content(
    extractor: RegexSlotExtractor, text: str
) -> None:
    assert "reminder_text" not in extractor.extract(text, Intent.SET_REMINDER).slots


@pytest.mark.parametrize(
    "text",
    [
        "đặt báo thức -1 giờ",
        "đặt báo thức 7 giờ 70 phút",
        "đặt báo thức 7 giờ kém 99 phút",
        "đặt báo thức 7 giờ 75",
    ],
)
def test_invalid_numeric_time_is_not_extracted(
    extractor: RegexSlotExtractor, text: str
) -> None:
    assert "datetime" not in extractor.extract(text, Intent.SET_ALARM).slots


def test_overlong_separated_phone_is_not_truncated(
    extractor: RegexSlotExtractor,
) -> None:
    text = "gọi số 0 9 0 1 2 3 4 5 6 7 8 9"

    assert extractor.extract(text, Intent.CALL_CONTACT).slots == {}
