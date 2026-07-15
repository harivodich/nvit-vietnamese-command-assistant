"""Ghép normalizer, intent, slot, safety gate và action thành core NLU."""

from __future__ import annotations

import re
from collections.abc import Mapping

from nvit_assistant.actions import ActionRouter
from nvit_assistant.nlu.action_gate import ActionGate
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.runtime_intent_classifier import IntentPredictor
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import DatasetSample, Intent, ParseRequest, ParseResult


def missing_required_slot_options(intent: Intent, slots: Mapping[str, object]) -> tuple[str, ...]:
    """Trả các phương án slot bắt buộc khi chưa nhóm nào thỏa contract của intent."""
    required_groups = DatasetSample.required_slot_groups.get(intent)
    if required_groups is None or any(group.issubset(slots) for group in required_groups):
        return ()
    return tuple("/".join(sorted(group)) for group in required_groups)


MISSING_SLOT_PROMPTS = {
    Intent.SET_REMINDER: "Bạn muốn mình nhắc việc gì?",
    Intent.SET_ALARM: "Bạn muốn đặt báo thức lúc nào?",
    Intent.ASK_WEATHER: "Bạn muốn xem thời tiết ở tỉnh hoặc thành phố nào?",
    Intent.CALL_CONTACT: "Bạn muốn gọi cho ai hoặc gọi tới số điện thoại nào?",
}
WAKE_UP_BOUNDARY_PATTERN = re.compile(
    r"(?<!\w)(?:(?:gọi|kêu)(?:\s+\w+){0,3}\s+dậy|"
    r"đánh thức(?:\s+(?:tôi|tớ|tui|mình|em|con|cháu|tao))?)(?!\w)"
)
CALL_COMMAND_PATTERN = re.compile(
    r"(?<!\w)(?:gọi|goi|liên lạc|lien lac|liên hệ|lien he|"
    r"quay (?:số|so)|bấm (?:số|so)|bam (?:so|may)|bấm máy|"
    r"nối máy|noi may|thực hiện cuộc gọi|thuc hien cuoc goi)(?!\w)"
)
IMMEDIATE_DATETIMES = frozenset({"bây giờ", "bay gio"})
WEATHER_LOCATION_PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\w)(?:đâu|dau|đây|day|chỗ nào|cho nao|nơi nào|noi nao)(?!\w)"
)


def missing_slot_prompt(intent: Intent, missing_options: tuple[str, ...]) -> str:
    """Đổi tên field kỹ thuật thành câu hỏi tự nhiên cho người dùng cuối."""
    return MISSING_SLOT_PROMPTS.get(
        intent,
        f"Bạn vui lòng bổ sung thông tin còn thiếu: {' hoặc '.join(missing_options)}.",
    )


def resolve_intent_boundary(
    text: str, predicted_intent: Intent, slot_extractor: RegexSlotExtractor
) -> tuple[Intent, bool]:
    """Sửa các ranh giới gọi ngay, gọi theo lịch và đánh thức bằng rule rõ nghĩa."""
    if WAKE_UP_BOUNDARY_PATTERN.search(text):
        return Intent.SET_ALARM, True

    reminder_slots = slot_extractor.extract(text, Intent.SET_REMINDER).slots
    call_slots = slot_extractor.extract(text, Intent.CALL_CONTACT).slots
    reminder_text = reminder_slots.get("reminder_text")
    if isinstance(reminder_text, str):
        call_slots = {
            **slot_extractor.extract(reminder_text, Intent.CALL_CONTACT).slots,
            **call_slots,
        }
    has_call_target = "contact_name" in call_slots or "phone_number" in call_slots
    has_call_command = CALL_COMMAND_PATTERN.search(text) is not None
    scheduled_call_command = (
        isinstance(reminder_text, str) and CALL_COMMAND_PATTERN.match(reminder_text) is not None
    )
    datetime = reminder_slots.get("datetime")
    contact_name = call_slots.get("contact_name")
    datetime_belongs_to_contact = (
        isinstance(datetime, str)
        and isinstance(contact_name, str)
        and re.search(rf"(?<!\w){re.escape(datetime)}(?!\w)", contact_name) is not None
    )

    if (
        scheduled_call_command
        and has_call_target
        and isinstance(datetime, str)
        and datetime not in IMMEDIATE_DATETIMES
        and not datetime_belongs_to_contact
    ):
        return Intent.SET_REMINDER, True
    if (
        has_call_command
        and has_call_target
        and (datetime is None or datetime in IMMEDIATE_DATETIMES or datetime_belongs_to_contact)
    ):
        return Intent.CALL_CONTACT, True
    return predicted_intent, False


class NLUPipeline:
    """Pipeline text-first dùng action router có thể inject; runtime hiện inject mock router."""

    def __init__(
        self,
        normalizer: VietnameseNormalizer,
        intent_classifier: IntentPredictor,
        slot_extractor: RegexSlotExtractor,
        action_router: ActionRouter | None = None,
        action_gate: ActionGate | None = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold phải nằm trong [0, 1]")
        self.normalizer = normalizer
        self.intent_classifier = intent_classifier
        self.slot_extractor = slot_extractor
        self.action_router = action_router
        self.action_gate = action_gate
        self.confidence_threshold = confidence_threshold

    def parse(self, request: ParseRequest) -> ParseResult:
        """Chuẩn hóa một lần, dự đoán intent rồi chỉ trích slot hợp lệ của intent đó."""
        normalized = self.normalizer.normalize(request.text, request.region_hint)
        prediction = self.intent_classifier.predict(normalized.normalized_text)
        resolved_intent, boundary_matched = resolve_intent_boundary(
            normalized.normalized_text, prediction.intent, self.slot_extractor
        )
        if resolved_intent is not prediction.intent:
            boundary_features = [
                f"intent_boundary:{prediction.intent.value}->{resolved_intent.value}"
            ]
        elif boundary_matched:
            boundary_features = [f"intent_boundary_confirmed:{resolved_intent.value}"]
        else:
            boundary_features = []
        if prediction.confidence < self.confidence_threshold and not boundary_matched:
            return ParseResult(
                text=request.text,
                normalized_text=normalized.normalized_text,
                region=normalized.region,
                intent=Intent.UNKNOWN,
                confidence=prediction.confidence,
                slots={},
                response="Mình chưa đủ chắc chắn để thực hiện yêu cầu. Bạn vui lòng nói rõ hơn.",
                matched_features=[
                    *normalized.matched_variants,
                    f"intent_rejected:{prediction.intent.value}",
                    f"confidence_threshold:{self.confidence_threshold:.2f}",
                ],
            )
        if prediction.confidence < self.confidence_threshold:
            boundary_features.append("confidence_override:high_precision_boundary")
        extraction = self.slot_extractor.extract(normalized.normalized_text, resolved_intent)
        if self.action_gate is not None:
            gate = self.action_gate.check(
                normalized.normalized_text, resolved_intent, extraction.slots
            )
            if not gate.allowed:
                return ParseResult(
                    text=request.text,
                    normalized_text=normalized.normalized_text,
                    region=normalized.region,
                    intent=Intent.UNKNOWN,
                    confidence=prediction.confidence,
                    slots={},
                    response=(
                        "Mình nhận ra đây chưa phải yêu cầu có thể thực hiện. "
                        "Bạn vui lòng nói lại bằng một lệnh được hỗ trợ."
                    ),
                    matched_features=[
                        *normalized.matched_variants,
                        *boundary_features,
                        f"intent_rejected:{resolved_intent.value}",
                        f"action_gate:{gate.reason}",
                    ],
                )
        weather_location_is_placeholder = (
            resolved_intent is Intent.ASK_WEATHER
            and "location" not in extraction.slots
            and WEATHER_LOCATION_PLACEHOLDER_PATTERN.search(normalized.normalized_text) is not None
        )
        missing_options = (
            ("location",)
            if weather_location_is_placeholder
            else missing_required_slot_options(resolved_intent, extraction.slots)
        )
        action = None
        if missing_options:
            response = missing_slot_prompt(resolved_intent, missing_options)
        elif self.action_router is not None:
            execution = self.action_router.execute(resolved_intent, extraction.slots)
            action = execution.result
            response = execution.response
        else:
            response = f"Đã nhận diện intent {resolved_intent.value}."
        return ParseResult(
            text=request.text,
            normalized_text=normalized.normalized_text,
            region=normalized.region,
            intent=resolved_intent,
            confidence=prediction.confidence,
            slots=extraction.slots,
            action=action,
            response=response,
            matched_features=[
                *normalized.matched_variants,
                f"intent_model:{prediction.intent.value}",
                *boundary_features,
                *extraction.matched_features,
                *(
                    ["location_clarification:placeholder"]
                    if weather_location_is_placeholder
                    else (
                        [f"missing_required_slots:{'|'.join(missing_options)}"]
                        if missing_options
                        else []
                    )
                ),
            ],
        )
