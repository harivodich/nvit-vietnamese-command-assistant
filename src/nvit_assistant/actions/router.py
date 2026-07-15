"""Chuyển intent/slot thành action giả lập an toàn và câu phản hồi tiếng Việt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from nvit_assistant.schemas import ActionResult, ActionStatus, ActionType, Intent, SlotName


@dataclass(frozen=True)
class ActionExecution:
    """Gói kết quả máy đọc được và phản hồi ngắn cho người dùng."""

    result: ActionResult
    response: str


class ActionRouter(Protocol):
    """Interface nhỏ để sau này thay mock bằng adapter thiết bị thật."""

    mode: str

    def execute(self, intent: Intent, slots: dict[str, Any]) -> ActionExecution:
        """Thực thi hoặc giả lập một action từ intent và slot đã hợp lệ."""
        ...


def _required_string(slots: dict[str, Any], slot_name: SlotName) -> str:
    """Đọc slot chuỗi bắt buộc và chặn router bị gọi sai contract."""
    value = slots.get(slot_name.value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"thiếu slot bắt buộc: {slot_name.value}")
    return value


def _optional_string(slots: dict[str, Any], slot_name: SlotName) -> str | None:
    """Đọc slot tùy chọn; dữ liệu sai kiểu được coi là lỗi thay vì ép chuỗi."""
    value = slots.get(slot_name.value)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"slot {slot_name.value} phải là chuỗi không rỗng")
    return value


class MockActionRouter:
    """Tạo payload deterministic; không gọi API, thiết bị hoặc dữ liệu cá nhân thật."""

    mode = "mock-actions"

    def execute(self, intent: Intent, slots: dict[str, Any]) -> ActionExecution:
        """Điều phối đúng action handler theo intent đã qua confidence/slot gate."""
        handlers = {
            Intent.SET_REMINDER: self._create_reminder,
            Intent.SET_ALARM: self._set_alarm,
            Intent.ASK_WEATHER: self._query_weather,
            Intent.PLAY_MUSIC: self._play_music,
            Intent.CALL_CONTACT: self._call,
        }
        handler = handlers.get(intent)
        if handler is None:
            raise ValueError(f"không có action cho intent: {intent.value}")
        return handler(slots)

    def _create_reminder(self, slots: dict[str, Any]) -> ActionExecution:
        reminder_text = _required_string(slots, SlotName.REMINDER_TEXT)
        datetime = _optional_string(slots, SlotName.DATETIME)
        payload = {"reminder_text": reminder_text, "datetime": datetime}
        time_text = f" vào {datetime}" if datetime else ""
        return self._execution(
            ActionType.CREATE_REMINDER,
            payload,
            f"Đã giả lập tạo lời nhắc “{reminder_text}”{time_text}.",
        )

    def _set_alarm(self, slots: dict[str, Any]) -> ActionExecution:
        datetime = _required_string(slots, SlotName.DATETIME)
        return self._execution(
            ActionType.SET_ALARM,
            {"datetime": datetime},
            f"Đã giả lập đặt báo thức vào {datetime}.",
        )

    def _query_weather(self, slots: dict[str, Any]) -> ActionExecution:
        location = _optional_string(slots, SlotName.LOCATION)
        datetime = _optional_string(slots, SlotName.DATETIME)
        payload = {"location": location, "datetime": datetime}
        place_text = location or "vị trí hiện tại"
        time_text = datetime or "hiện tại"
        return self._execution(
            ActionType.QUERY_WEATHER,
            payload,
            f"Đã giả lập yêu cầu thời tiết tại {place_text} vào {time_text}.",
        )

    def _play_music(self, slots: dict[str, Any]) -> ActionExecution:
        song = _optional_string(slots, SlotName.SONG)
        artist = _optional_string(slots, SlotName.ARTIST)
        payload = {"song": song, "artist": artist}
        target = song or (f"nhạc của {artist}" if artist else "danh sách nhạc mặc định")
        artist_text = f" của {artist}" if song and artist else ""
        return self._execution(
            ActionType.PLAY_MUSIC,
            payload,
            f"Đã giả lập phát {target}{artist_text}.",
        )

    def _call(self, slots: dict[str, Any]) -> ActionExecution:
        contact = _optional_string(slots, SlotName.CONTACT_NAME)
        phone = _optional_string(slots, SlotName.PHONE_NUMBER)
        if not contact and not phone:
            raise ValueError("thiếu contact_name hoặc phone_number")
        target = contact or phone
        return self._execution(
            ActionType.CALL,
            {"contact_name": contact, "phone_number": phone, "target": target},
            f"Đã giả lập cuộc gọi tới {target}.",
        )

    @staticmethod
    def _execution(
        action_type: ActionType, payload: dict[str, Any], response: str
    ) -> ActionExecution:
        """Gắn trạng thái mocked thống nhất để không bị hiểu nhầm là action thật."""
        return ActionExecution(
            result=ActionResult(type=action_type, status=ActionStatus.MOCKED, payload=payload),
            response=response,
        )
