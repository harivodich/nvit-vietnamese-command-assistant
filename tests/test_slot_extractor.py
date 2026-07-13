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
    ],
)
def test_extract_slots_by_intent(
    extractor: RegexSlotExtractor,
    text: str,
    intent: Intent,
    expected: dict[str, str],
) -> None:
    assert extractor.extract(text, intent).slots == expected
