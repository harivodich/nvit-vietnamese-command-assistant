"""Trích xuất slot có thể giải thích bằng regex và từ điển cấu hình."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nvit_assistant.nlu.normalizer import normalize_surface
from nvit_assistant.nlu.slot_lexicon import load_slot_lexicon
from nvit_assistant.schemas import Intent, SlotName


PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?![ .-]?\d)"
)
OVERLONG_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){11,}(?![ .-]?\d)"
)
NUMERIC_CLOCK = (
    r"(?:[01]?\d|2[0-3])\s*(?:h|giờ)"
    r"(?:\s*rưỡi|\s+kém\s+(?:[0-5]?\d)(?:\s*phút)?|"
    r"\s*(?:[0-5]?\d)\s*(?:phút)?)?"
)
NUMERIC_TIME_PATTERN = re.compile(rf"(?<!\w){NUMERIC_CLOCK}(?!\w)")
INVALID_NUMERIC_TIME_PATTERN = re.compile(
    r"(?<!\w)[+-]\s*\d+\s*(?:h|giờ)(?!\w)|"
    r"(?<!\w)(?:[01]?\d|2[0-3])\s*(?:h|giờ)\s+"
    r"(?:kém\s+)?(?:[+-]\s*\d+|[6-9]\d|\d{3,})(?:\s*phút)?(?!\w)"
)
DURATION_PATTERN = re.compile(
    r"(?<!\w)(?:\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+"
    r"(?:phút|tiếng|giờ)\s+nữa(?!\w)"
)
RELATIVE_TIME_PATTERN = re.compile(
    r"(?<!\w)(?:sáng|trưa|chiều|tối|đêm)?\s*"
    r"(?:hôm nay|ngày mai|mai|thứ (?:hai|ba|tư|năm|sáu|bảy|chủ nhật)|cuối tuần này)"
    r"(?!\w)"
)
NUMBER_WORDS = (
    r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười|"
    r"mươi|trăm|nghìn|ngàn)(?:\s+(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|"
    r"sáu|bảy|tám|chín|mười|mươi|trăm|nghìn|ngàn))*"
)
CLOCK_WORD_PATTERN = re.compile(
    rf"(?<!\w){NUMBER_WORDS}(?:\s+giờ|\s+rưỡi)"
    rf"(?:\s+kém\s+{NUMBER_WORDS}(?:\s+phút)?|\s+{NUMBER_WORDS}\s+phút)?"
    r"(?:\s+(?:sáng|trưa|chiều|tối|đêm))?"
    r"(?:\s+(?:hôm nay|ngày mai|mai|mỗi buổi sáng))?(?!\w)"
)
DAY_WORDS = r"(?:hôm nay|ngày mai|mai|thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật)"
DAY_CLOCK_PATTERN = re.compile(
    rf"(?<!\w)(?:buổi\s+)?(?:sáng|trưa|chiều|tối|đêm)?\s*{DAY_WORDS}"
    rf"\s+(?:lúc\s+)?{NUMBER_WORDS}(?:\s+giờ|\s+rưỡi)"
    r"(?:\s+(?:sáng|trưa|chiều|tối|đêm))?(?!\w)"
)
DAY_NUMERIC_CLOCK_PATTERN = re.compile(
    rf"(?<!\w)(?:buổi\s+)?(?:sáng|trưa|chiều|tối|đêm)?\s*{DAY_WORDS}"
    rf"\s+(?:lúc\s+)?{NUMERIC_CLOCK}(?!\w)"
)
NUMERIC_CLOCK_DAY_PATTERN = re.compile(
    rf"(?<!\w){NUMERIC_CLOCK}\s+"
    rf"(?:buổi\s+)?(?:sáng|trưa|chiều|tối|đêm)?\s*{DAY_WORDS}(?!\w)"
)
AFTER_CLOCK_PATTERN = re.compile(
    rf"(?<!\w)sau\s+{NUMBER_WORDS}(?:\s+giờ|\s+rưỡi)"
    r"(?:\s+(?:sáng|trưa|chiều|tối|đêm))?(?!\w)"
)
CALENDAR_DATE_PATTERN = re.compile(
    rf"(?<!\w)(?:ngày\s+)?{NUMBER_WORDS}\s+tháng\s+{NUMBER_WORDS}"
    rf"(?:\s+(?:năm\s+)?{NUMBER_WORDS})?(?!\w)"
)
PERIOD_PATTERN = re.compile(
    rf"(?<!\w)(?:(?:trong|sau)\s+(?:{NUMBER_WORDS}|\d+)\s+"
    r"(?:phút|tiếng|giờ|ngày|tuần|tháng)"
    rf"|(?:{NUMBER_WORDS}|\d+)\s+(?:phút|tiếng|giờ|ngày|tuần|tháng)\s+"
    r"(?:nữa|tới|sau|kể từ bây giờ))(?!\w)"
)
WEEK_PATTERN = re.compile(
    r"(?<!\w)(?:buổi\s+)?(?:sáng|trưa|chiều|tối|đêm)?\s*"
    r"(?:thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật)"
    r"(?:\s+(?:tuần (?:này|sau|tới)|này))?(?!\w)"
)
NAMED_PERIOD_PATTERN = re.compile(
    r"(?<!\w)(?:sáng|trưa|chiều|tối|đêm) nay|"
    r"(?<!\w)(?:tuần (?:này|tới|sau)|tháng sau|cuối tuần(?: này)?)(?!\w)"
)
MONTHLY_DATE_PATTERN = re.compile(
    rf"(?<!\w)ngày\s+{NUMBER_WORDS}\s+hàng\s+tháng(?!\w)"
)
REMINDER_TRIGGER_PATTERN = re.compile(
    r"^(?:hãy\s+)?(?:nhắc|nhớ báo|đừng quên(?:\s+nhắc)?|tạo lời nhắc)"
    r"(?:\s+cho)?(?:\s+(?:tôi|tớ|tui|mình))?(?:\s+|$)"
)
POLITE_END_PATTERN = re.compile(
    r"(?:^|\s)(?:nhé|nhá|nha|nghen|hen|hỉ|với|giúp tôi|dùm|giùm|được không|đi|ngay)\s*$"
)
ENTITY_END_PATTERN = re.compile(
    r"\s+(?:bây giờ|ngay bây giờ|một lần nữa|cho (?:tôi|tớ|tui|mình)|"
    r"để (?:tôi|tớ|tui|mình)|nhé|nha|với|đi|ngay|được không|không)\s*$"
)
REMINDER_POLITE_PREFIX_PATTERN = re.compile(
    r"^(?:giúp|dùm|giùm)(?:\s+(?:tôi|tớ|tui|mình))?(?:\s+|$)"
)
REMINDER_EMPTY_CONTENT = frozenset(
    {
        "nhắc",
        "lời nhắc",
        "tạo lời nhắc",
        "đặt lời nhắc",
        "tôi",
        "tớ",
        "tui",
        "mình",
        "giúp",
        "giúp tôi",
        "dùm",
        "giùm",
    }
)
REMINDER_INCOMPLETE_ACTION = re.compile(
    r"^(?:gọi(?:\s+lại)?|mua|uống|gửi|nộp|tưới|đón|trả|thanh toán|kiểm tra)$"
)
CONTACT_EMPTY_CONTENT = frozenset(
    {
        "cho",
        "đến",
        "tới",
        "tôi",
        "tớ",
        "tui",
        "mình",
        "giúp",
        "giúp tôi",
        "dùm",
        "giùm",
        "đi",
        "ngay",
        "bây giờ",
        "ngay bây giờ",
        "điện",
        "số",
        "ai",
        "người nào",
        "nguoi nao",
    }
)
INVALID_MUSIC_ENTITY = re.compile(
    r"(?:^|\s)(?:tôi|mình|danh sách phát|yêu thích|tất cả|bất kỳ|bất cứ|"
    r"hàng đầu|thính phòng|đồng quê)(?:\s|$)"
)
MUSIC_ENTITY_STOP_ONLY = re.compile(
    r"^(?:đi|ngay|lên|thôi|nào|giúp tôi|chưa|không|à|hả|ư|vậy|thế|sao|phải|ai)$"
)
INVALID_LOCATION = re.compile(
    r"(?:^|\s)(?:tôi|mình|bao nhiêu|mưa|gió|độ|là|nên|khu vực của|"
    r"đâu|dau|đây|day|chỗ nào|cho nao|nơi nào|noi nao)(?:\s|$)"
)
SPOKEN_DIGITS = {
    "không": "0",
    "linh": "0",
    "lẻ": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "tư": "4",
    "năm": "5",
    "lăm": "5",
    "sáu": "6",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
}


@dataclass(frozen=True)
class SlotExtractionResult:
    """Các slot tìm được và dấu vết rule/lexicon đã match."""

    slots: dict[str, str]
    matched_features: tuple[str, ...] = ()


def _normalized_strings(values: Any, field_name: str) -> tuple[str, ...]:
    """Kiểm tra list chuỗi từ YAML và chuẩn hóa để lookup ổn định."""
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} phải là list chuỗi")
    unique = dict.fromkeys(normalize_surface(value) for value in values)
    return tuple(sorted(unique, key=lambda value: -len(value)))


def _find_longest(text: str, values: tuple[str, ...]) -> str | None:
    """Tìm cụm dài nhất; chỉ ưu tiên khớp có dấu khi hai ứng viên dài bằng nhau."""
    folded_text = _strip_diacritics(text)
    matches: list[tuple[int, bool, int, str]] = []
    for index, value in enumerate(values):
        exact = bool(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text))
        folded_value = _strip_diacritics(value)
        folded = bool(
            re.search(rf"(?<!\w){re.escape(folded_value)}(?!\w)", folded_text)
        )
        if exact or folded:
            matches.append((len(folded_value), exact, -index, value))
    return max(matches)[3] if matches else None


def _strip_diacritics(text: str) -> str:
    """Tạo dạng không dấu cục bộ để lookup chịu được transcript ASR thiếu dấu."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped).replace("đ", "d")


def _clean_entity(value: str) -> str | None:
    """Bỏ phần đuôi hội thoại khỏi entity được regex lấy theo ngữ cảnh."""
    cleaned = ENTITY_END_PATTERN.sub("", value.strip())
    cleaned = POLITE_END_PATTERN.sub("", cleaned).strip(" ,.!?")
    return cleaned or None


class RegexSlotExtractor:
    """Extractor theo intent; không tự suy diễn slot không xuất hiện trong câu."""

    def __init__(self, config_path: Path, lexicon_path: Path | None = None) -> None:
        """Nạp config thủ công và lexicon học từ train một lần cho runtime."""
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("slot_values YAML phải là mapping")
        learned = load_slot_lexicon(lexicon_path) if lexicon_path is not None else {}
        self.contacts = _normalized_strings(
            [
                *_normalized_strings(raw.get("contacts"), "contacts"),
                *learned.get("contact_name", ()),
            ],
            "contacts",
        )
        self.locations = _normalized_strings(
            [
                *_normalized_strings(raw.get("locations"), "locations"),
                *learned.get("location", ()),
            ],
            "locations",
        )
        self.datetimes = _normalized_strings(raw.get("datetimes"), "datetimes")
        self.weather_times = _normalized_strings(raw.get("weather_times"), "weather_times")
        self.reminder_texts = _normalized_strings(
            [
                *_normalized_strings(raw.get("reminder_texts"), "reminder_texts"),
                *learned.get("reminder_text", ()),
            ],
            "reminder_texts",
        )
        catalog = raw.get("music_catalog")
        if not isinstance(catalog, list) or not all(isinstance(item, dict) for item in catalog):
            raise ValueError("music_catalog phải là list mapping")
        extra_songs = _normalized_strings(raw.get("songs", []), "songs")
        extra_artists = _normalized_strings(raw.get("artists", []), "artists")
        self.songs = _normalized_strings(
            [item["song"] for item in catalog]
            + list(extra_songs)
            + list(learned.get("song", ())),
            "music songs",
        )
        self.artists = _normalized_strings(
            [item["artist"] for item in catalog]
            + list(extra_artists)
            + list(learned.get("artist", ())),
            "music artists",
        )

    def extract(self, text: str, intent: Intent) -> SlotExtractionResult:
        """Trích đúng nhóm slot hợp lệ của intent đã được classifier dự đoán."""
        normalized_text = normalize_surface(text)
        slots: dict[str, str] = {}
        matched: list[str] = []

        datetime = self._extract_datetime(normalized_text)
        if intent in {Intent.SET_REMINDER, Intent.SET_ALARM, Intent.ASK_WEATHER} and datetime:
            slots[SlotName.DATETIME.value] = datetime
            matched.append(f"datetime:{datetime}")

        if intent is Intent.SET_REMINDER:
            reminder_text = self._extract_reminder_text(normalized_text, datetime)
            if reminder_text:
                slots[SlotName.REMINDER_TEXT.value] = reminder_text
                matched.append(f"reminder_text:{reminder_text}")

        elif intent is Intent.ASK_WEATHER:
            location = self._extract_location(normalized_text)
            if location:
                slots[SlotName.LOCATION.value] = location
                matched.append(f"location:{location}")

        elif intent is Intent.PLAY_MUSIC:
            song, artist = self._extract_music_entities(normalized_text)
            if song:
                slots[SlotName.SONG.value] = song
                matched.append(f"song:{song}")
            if artist:
                slots[SlotName.ARTIST.value] = artist
                matched.append(f"artist:{artist}")

        elif intent is Intent.CALL_CONTACT:
            phone_match = (
                None
                if OVERLONG_PHONE_PATTERN.search(normalized_text)
                else PHONE_PATTERN.search(normalized_text)
            )
            contact = _find_longest(normalized_text, self.contacts)
            if phone_match:
                phone = re.sub(r"\s+", " ", phone_match.group()).strip()
                slots[SlotName.PHONE_NUMBER.value] = phone
                matched.append(f"phone_number:{phone}")
            elif spoken_phone := self._extract_spoken_phone(normalized_text):
                slots[SlotName.PHONE_NUMBER.value] = spoken_phone
                matched.append(f"phone_number:{spoken_phone}")
            elif contact:
                slots[SlotName.CONTACT_NAME.value] = contact
                matched.append(f"contact_name:{contact}")
            elif unknown_contact := self._extract_unknown_contact(normalized_text):
                slots[SlotName.CONTACT_NAME.value] = unknown_contact
                matched.append(f"contact_name:{unknown_contact}")

        return SlotExtractionResult(slots=slots, matched_features=tuple(matched))

    def _extract_datetime(self, text: str) -> str | None:
        """Sinh nhiều ứng viên rồi lấy cụm dài nhất để không cắt `ngày mai lúc sáu giờ`."""
        if INVALID_NUMERIC_TIME_PATTERN.search(text):
            return None
        configured = _find_longest(text, self.datetimes + self.weather_times)
        candidates = [configured] if configured else []
        patterns = (
            DAY_CLOCK_PATTERN,
            DAY_NUMERIC_CLOCK_PATTERN,
            NUMERIC_CLOCK_DAY_PATTERN,
            AFTER_CLOCK_PATTERN,
            CALENDAR_DATE_PATTERN,
            MONTHLY_DATE_PATTERN,
            PERIOD_PATTERN,
            DURATION_PATTERN,
            CLOCK_WORD_PATTERN,
            NUMERIC_TIME_PATTERN,
            WEEK_PATTERN,
            NAMED_PERIOD_PATTERN,
            RELATIVE_TIME_PATTERN,
        )
        candidates.extend(match.group().strip() for pattern in patterns for match in pattern.finditer(text))
        candidates = [candidate for candidate in candidates if candidate]
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: (len(candidate), candidate.count(" ")))

    def _extract_reminder_text(self, text: str, datetime: str | None) -> str | None:
        """Bỏ trigger, datetime và tiểu từ để giữ lại việc người dùng muốn được nhắc."""
        configured = _find_longest(text, self.reminder_texts)
        if configured:
            return configured
        reminder = re.sub(r"^vui lòng\s+", "", text)
        reminder = REMINDER_TRIGGER_PATTERN.sub("", reminder, count=1)
        reminder = re.sub(
            r"^(?:cài đặt một nhắc nhở|đặt lời nhắc|đặt lịch|tôi cần phải được nhắc rằng)"
            r"(?:\s+(?:cho|tôi))?(?:\s+|$)",
            "",
            reminder,
        )
        reminder = REMINDER_POLITE_PREFIX_PATTERN.sub("", reminder, count=1)
        if datetime:
            reminder = re.sub(rf"(?<!\w){re.escape(datetime)}(?!\w)", " ", reminder, count=1)
        reminder = re.sub(
            r"^(?:vào|lúc|đến|cho|về|rằng|sau|trước)(?:\s+|$)",
            "",
            reminder.strip(),
        )
        reminder = re.sub(r"\s+(?:vào|lúc|đến|cho|sau|trước)\s*$", "", reminder.strip())
        reminder = re.split(r"\s+(?:vì|tại|ở)\s+", reminder, maxsplit=1)[0]
        reminder = re.sub(r"\s+(?:của (?:tôi|mình)|tiếp theo)\s*$", "", reminder)
        reminder = POLITE_END_PATTERN.sub("", reminder).strip(" ,.!?")
        reminder = re.sub(r"\s+", " ", reminder)
        if reminder in REMINDER_EMPTY_CONTENT or REMINDER_INCOMPLETE_ACTION.fullmatch(reminder):
            return None
        return reminder or None

    def _extract_location(self, text: str) -> str | None:
        """Ưu tiên địa danh chuẩn; nếu ngoài từ điển thì lấy theo ngữ cảnh câu thời tiết."""
        configured = _find_longest(text, self.locations)
        if configured:
            return configured
        time_start = (
            r"bây giờ|hôm nay|ngày mai|tối nay|đêm nay|chiều nay|sáng mai|"
            r"tuần (?:này|tới|sau)|thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật"
        )
        patterns = (
            re.compile(
                rf"(?<!\w)(?:ở|tại|trên|của)\s+"
                rf"(?P<value>.+?)(?=\s+(?:{time_start}|vào)\b|$)"
            ),
        )
        for pattern in patterns:
            match = pattern.search(text)
            if (
                match
                and (value := _clean_entity(match.group("value")))
                and not INVALID_LOCATION.search(value)
            ):
                return value
        return None

    def _extract_music_entities(self, text: str) -> tuple[str | None, str | None]:
        """Kết hợp catalog và mẫu câu để nhận cả bài hát/nghệ sĩ chưa có trong từ điển."""
        song = _find_longest(text, self.songs)
        artist = _find_longest(text, self.artists)

        if artist is None:
            artist_patterns = (
                re.compile(
                    r"(?:của|bởi)\s+(?P<value>(?:(?!\s+(?:của|bởi)\s+).)+)$"
                ),
                re.compile(r"(?:mở|phát|chơi)\s+nhạc\s+(?P<value>.+)$"),
            )
            for pattern in artist_patterns:
                matches = list(pattern.finditer(text))
                if (
                    matches
                    and (value := _clean_entity(matches[-1].group("value")))
                    and not INVALID_MUSIC_ENTITY.search(value)
                    and not MUSIC_ENTITY_STOP_ONLY.fullmatch(value)
                ):
                    artist = value
                    break

        if song is None:
            song_patterns = (
                re.compile(
                    r"(?:phát|mở|nghe)(?:\s+(?:cho tôi|tiếp theo))?\s+"
                    r"(?:bài hát|bài)\s+(?!của\b|bởi\b)"
                    r"(?P<value>.+?)(?=\s+(?:của|bởi)\s+)"
                ),
            )
            for pattern in song_patterns:
                match = pattern.search(text)
                if match and (value := _clean_entity(match.group("value"))):
                    if (
                        not INVALID_MUSIC_ENTITY.search(value)
                        and not MUSIC_ENTITY_STOP_ONLY.fullmatch(value)
                        and not value.startswith("từ ")
                    ):
                        song = value
                        break
        return song, artist

    def _extract_spoken_phone(self, text: str) -> str | None:
        """Đổi chuỗi chữ số được đọc rời thành số điện thoại chuẩn liền nhau."""
        digit_words = "|".join(SPOKEN_DIGITS)
        match = re.search(
            rf"(?:số|gọi)\s+(?P<digits>(?:{digit_words})(?:\s+(?:{digit_words})){{8,10}})(?!\w)",
            text,
        )
        if not match:
            return None
        phone = "".join(SPOKEN_DIGITS[word] for word in match.group("digits").split())
        return phone if phone.startswith(("0", "84")) else None

    def _extract_unknown_contact(self, text: str) -> str | None:
        """Lấy tên ngoài danh bạ sau cue gọi; không buộc mọi tên riêng vào YAML."""
        candidate = re.sub(
            r"^(?:hãy\s+)?gọi(?:\s+(?:điện|lại)){0,2}(?=\s|$)",
            "",
            text,
            count=1,
        ).strip()
        candidate = re.sub(r"^(?:cho|đến|tới)(?=\s|$)", "", candidate, count=1).strip()
        value = _clean_entity(candidate)
        if value is None or value in CONTACT_EMPTY_CONTENT:
            return None
        if re.match(r"^(?:giúp|dùm|giùm|đi|ngay|bây giờ|điện|số)(?:\s|$)", value):
            return None
        if re.fullmatch(
            r"(?:ai|người nào|nguoi nao)(?:\s+(?:vậy|thế|nào|đó|vay|the|nao|do))?",
            value,
        ):
            return None
        if re.search(r"(?<!\w)dậy\s*$", value):
            return None
        if re.search(r"\d", value) or value.startswith("+"):
            return None
        return value
