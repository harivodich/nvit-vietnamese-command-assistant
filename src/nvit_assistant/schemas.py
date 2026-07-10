"""Các contract dữ liệu dùng chung cho dataset, NLU, action, API và evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator


class StrEnum(str, Enum):
    """Enum dạng chuỗi, tương thích với Python 3.10."""

    def __str__(self) -> str:
        """Trả về giá trị chuỗi để serialize enum sang JSON dễ dàng."""
        return self.value


class Intent(StrEnum):
    SET_REMINDER = "set_reminder"
    SET_ALARM = "set_alarm"
    ASK_WEATHER = "ask_weather"
    PLAY_MUSIC = "play_music"
    CALL_CONTACT = "call_contact"
    UNKNOWN = "unknown"


class Region(StrEnum):
    NORTH = "north"
    CENTRAL = "central"
    SOUTH = "south"
    UNKNOWN = "unknown"


class SlotName(StrEnum):
    DATETIME = "datetime"
    LOCATION = "location"
    SONG = "song"
    ARTIST = "artist"
    CONTACT_NAME = "contact_name"
    PHONE_NUMBER = "phone_number"
    REMINDER_TEXT = "reminder_text"


class DataSource(StrEnum):
    SYNTHETIC = "synthetic"
    MANUAL = "manual"
    WEB_MINED = "web_mined"
    OLD_PROJECT = "old_project"
    ASR_NOISE = "asr_noise"


class VariantType(StrEnum):
    ACCENTED = "accented"
    NO_DIACRITICS = "no_diacritics"
    REGIONAL = "regional"
    ASR_NOISE = "asr_noise"
    FORMAL = "formal"
    CASUAL = "casual"


class ParseRequest(BaseModel):
    """Dữ liệu đầu vào cho pipeline parse từ CLI hoặc API."""

    text: str = Field(min_length=1)
    region_hint: Region | None = None

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        """Từ chối chuỗi chỉ gồm khoảng trắng."""
        if not text.strip():
            raise ValueError("text must not be blank")
        return text


class ActionResult(BaseModel):
    """Kết quả của action giả lập hoặc action thật trong tương lai."""

    type: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    """Kết quả end-to-end của một câu lệnh sau khi pipeline xử lý."""

    text: str
    normalized_text: str
    region: Region
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    slots: dict[str, Any] = Field(default_factory=dict)
    action: ActionResult | None = None
    response: str
    matched_features: list[str] = Field(default_factory=list)


class DatasetSample(BaseModel):
    """Một bản ghi JSONL có nhãn dùng để train, validation hoặc test."""

    allowed_slots: ClassVar[set[str]] = {slot.value for slot in SlotName}
    required_slot_groups: ClassVar[dict[Intent, tuple[frozenset[str], ...]]] = {
        Intent.SET_REMINDER: (frozenset({SlotName.REMINDER_TEXT.value}),),
        Intent.SET_ALARM: (frozenset({SlotName.DATETIME.value}),),
        Intent.ASK_WEATHER: (frozenset(),),
        Intent.PLAY_MUSIC: (frozenset({SlotName.SONG.value}),),
        Intent.CALL_CONTACT: (
            frozenset({SlotName.CONTACT_NAME.value}),
            frozenset({SlotName.PHONE_NUMBER.value}),
        ),
    }

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    region: Region
    intent: Intent
    slots: dict[str, Any] = Field(default_factory=dict)
    source: DataSource
    variant_type: VariantType

    @field_validator("intent")
    @classmethod
    def reject_unknown_intent(cls, intent: Intent) -> Intent:
        """Dataset chỉ được chứa nhãn intent đã biết, không chứa unknown."""
        if intent is Intent.UNKNOWN:
            raise ValueError("dataset samples cannot use unknown intent")
        return intent

    @field_validator("region")
    @classmethod
    def reject_unknown_region(cls, region: Region) -> Region:
        """Dataset phải có nhãn vùng cụ thể để đo metric theo vùng."""
        if region is Region.UNKNOWN:
            raise ValueError("dataset samples cannot use unknown region")
        return region

    @field_validator("slots")
    @classmethod
    def validate_slot_names(cls, slots: dict[str, Any]) -> dict[str, Any]:
        """Chặn slot ngoài whitelist để contract không bị lệch giữa các module."""
        unknown_slots = sorted(set(slots) - cls.allowed_slots)
        if unknown_slots:
            raise ValueError(f"unknown slot names: {unknown_slots}")
        return slots

    @model_validator(mode="after")
    def validate_required_slots(self) -> DatasetSample:
        """Yêu cầu ít nhất một nhóm slot hợp lệ của intent phải xuất hiện."""
        slot_names = set(self.slots)
        valid_groups = self.required_slot_groups[self.intent]
        if not any(group.issubset(slot_names) for group in valid_groups):
            expected_groups = [sorted(group) for group in valid_groups]
            raise ValueError(f"missing required slot group: one of {expected_groups}")
        return self


class EvaluationResult(BaseModel):
    """Contract cho báo cáo metric được tạo ở phase evaluation."""

    total: int = Field(ge=0)
    intent_accuracy: float = Field(ge=0.0, le=1.0)
    slot_exact_match: float = Field(ge=0.0, le=1.0)
    slot_precision: float = Field(ge=0.0, le=1.0)
    slot_recall: float = Field(ge=0.0, le=1.0)
    slot_f1: float = Field(ge=0.0, le=1.0)
    per_region_accuracy: dict[str, float] = Field(default_factory=dict)
    per_variant_accuracy: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
