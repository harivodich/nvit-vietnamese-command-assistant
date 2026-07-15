"""Các contract dữ liệu dùng chung cho dataset, NLU, action, API và evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrEnum(str, Enum):
    """Enum dạng chuỗi, tương thích với Python 3.10."""

    def __str__(self) -> str:
        """Trả về giá trị chuỗi để serialize enum sang JSON dễ dàng."""
        return str(self.value)


class Intent(StrEnum):
    SET_REMINDER = "set_reminder"
    SET_ALARM = "set_alarm"
    ASK_WEATHER = "ask_weather"
    PLAY_MUSIC = "play_music"
    CALL_CONTACT = "call_contact"
    UNKNOWN = "unknown"


class Region(StrEnum):
    STANDARD = "standard"
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
    MASSIVE = "massive"
    WEB_MINED = "web_mined"
    ASR_NOISE = "asr_noise"


class VariantType(StrEnum):
    ACCENTED = "accented"
    NO_DIACRITICS = "no_diacritics"
    REGIONAL = "regional"
    ASR_NOISE = "asr_noise"
    FORMAL = "formal"
    CASUAL = "casual"
    TRANSLATED = "translated"


class AnnotationQuality(StrEnum):
    """Mức độ tin cậy của nhãn để báo cáo chất lượng dataset minh bạch."""

    REVIEWED = "reviewed"
    AUTO_MAPPED = "auto_mapped"
    TEMPLATE_GENERATED = "template_generated"


class ActionType(StrEnum):
    """Loại hành động mà adapter thật có thể triển khai trong tương lai."""

    CREATE_REMINDER = "create_reminder"
    SET_ALARM = "set_alarm"
    QUERY_WEATHER = "query_weather"
    PLAY_MUSIC = "play_music"
    CALL = "call"


class ActionStatus(StrEnum):
    """Trạng thái action để phân biệt giả lập, hoàn tất và dịch vụ tạm lỗi."""

    MOCKED = "mocked"
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"


class ParseRequest(BaseModel):
    """Dữ liệu đầu vào cho pipeline parse từ CLI hoặc API."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
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

    type: ActionType
    status: ActionStatus
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
    allowed_slots_by_intent: ClassVar[dict[Intent, frozenset[str]]] = {
        Intent.SET_REMINDER: frozenset({SlotName.REMINDER_TEXT.value, SlotName.DATETIME.value}),
        Intent.SET_ALARM: frozenset({SlotName.DATETIME.value}),
        Intent.ASK_WEATHER: frozenset({SlotName.LOCATION.value, SlotName.DATETIME.value}),
        Intent.PLAY_MUSIC: frozenset({SlotName.SONG.value, SlotName.ARTIST.value}),
        Intent.CALL_CONTACT: frozenset({SlotName.CONTACT_NAME.value, SlotName.PHONE_NUMBER.value}),
    }
    required_slot_groups: ClassVar[dict[Intent, tuple[frozenset[str], ...]]] = {
        Intent.SET_REMINDER: (frozenset({SlotName.REMINDER_TEXT.value}),),
        Intent.SET_ALARM: (frozenset({SlotName.DATETIME.value}),),
        Intent.ASK_WEATHER: (frozenset(),),
        # Câu "mở nhạc" hợp lệ ngay cả khi người dùng không nói tên bài hát.
        Intent.PLAY_MUSIC: (frozenset(),),
        Intent.CALL_CONTACT: (
            frozenset({SlotName.CONTACT_NAME.value}),
            frozenset({SlotName.PHONE_NUMBER.value}),
        ),
    }

    id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    region: Region
    intent: Intent
    slots: dict[str, str | list[str]] = Field(default_factory=dict)
    source: DataSource
    source_ref: str | None = None
    variant_type: VariantType
    annotation_quality: AnnotationQuality

    @field_validator("id", "group_id", "text")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        """Từ chối ID, group ID hoặc câu lệnh chỉ gồm khoảng trắng."""
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

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
    def validate_slot_names(cls, slots: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
        """Chặn tên lạ và giá trị rỗng để nhãn slot luôn dùng được khi đánh giá."""
        unknown_slots = sorted(set(slots) - cls.allowed_slots)
        if unknown_slots:
            raise ValueError(f"unknown slot names: {unknown_slots}")
        for slot_name, value in slots.items():
            values = value if isinstance(value, list) else [value]
            if not values or any(not item.strip() for item in values):
                raise ValueError(f"slot {slot_name} must contain non-blank text")
        return slots

    @model_validator(mode="after")
    def validate_required_slots(self) -> DatasetSample:
        """Yêu cầu ít nhất một nhóm slot hợp lệ của intent phải xuất hiện."""
        slot_names = set(self.slots)
        invalid_for_intent = sorted(slot_names - self.allowed_slots_by_intent[self.intent])
        if invalid_for_intent:
            raise ValueError(
                f"slots not allowed for intent {self.intent.value}: {invalid_for_intent}"
            )
        valid_groups = self.required_slot_groups[self.intent]
        if not any(group.issubset(slot_names) for group in valid_groups):
            expected_groups = [sorted(group) for group in valid_groups]
            raise ValueError(f"missing required slot group: one of {expected_groups}")
        if self.source in {DataSource.MASSIVE, DataSource.WEB_MINED}:
            if not self.source_ref:
                raise ValueError("external samples must have source_ref")
        if self.source is DataSource.MASSIVE and self.region is not Region.STANDARD:
            raise ValueError(
                "MASSIVE samples must use standard region because accent is not labeled"
            )
        return self


class PreprocessedSample(BaseModel):
    """Một sample giữ nguyên nguồn gốc và bổ sung text/slot sau preprocess."""

    original: DatasetSample
    normalized_text: str = Field(min_length=1)
    normalized_slots: dict[str, str | list[str]] = Field(default_factory=dict)
    normalizer_region: Region
    matched_variants: list[str] = Field(default_factory=list)

    @field_validator("normalized_text")
    @classmethod
    def reject_blank_normalized_text(cls, text: str) -> str:
        """Không cho artifact preprocess chứa câu trống."""
        if not text.strip():
            raise ValueError("normalized_text must not be blank")
        return text


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
