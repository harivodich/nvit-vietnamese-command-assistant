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
    Intent.CALL_CONTACT: "Bạn muốn gọi cho ai hoặc gọi tới số điện thoại nào?",
}
WAKE_UP_BOUNDARY_PATTERN = re.compile(
    r"(?<!\w)(?:(?:gọi|kêu)(?:\s+\w+){0,3}\s+dậy|"
    r"đánh thức(?:\s+(?:tôi|tớ|tui|mình|em|con|cháu|tao))?)(?!\w)"
)
IMMEDIATE_CALL_BOUNDARY_PATTERN = re.compile(
    r"^(?:hãy\s+)?gọi(?:\s+(?:điện|lại)){0,2}(?=\s|$)"
)


def missing_slot_prompt(intent: Intent, missing_options: tuple[str, ...]) -> str:
    """Đổi tên field kỹ thuật thành câu hỏi tự nhiên cho người dùng cuối."""
    return MISSING_SLOT_PROMPTS.get(
        intent,
        f"Bạn vui lòng bổ sung thông tin còn thiếu: {' hoặc '.join(missing_options)}.",
    )


def resolve_intent_boundary(
    text: str, predicted_intent: Intent, slot_extractor: RegexSlotExtractor
) -> Intent:
    """Áp dụng hai ranh giới high-precision mà classifier thường nhầm trong câu cực ngắn."""
    if WAKE_UP_BOUNDARY_PATTERN.search(text):
        return Intent.SET_ALARM
    if IMMEDIATE_CALL_BOUNDARY_PATTERN.search(text):
        reminder_slots = slot_extractor.extract(text, Intent.SET_REMINDER).slots
        call_slots = slot_extractor.extract(text, Intent.CALL_CONTACT).slots
        if "datetime" not in reminder_slots and (
            predicted_intent is Intent.SET_REMINDER
            or "contact_name" in call_slots
            or "phone_number" in call_slots
        ):
            return Intent.CALL_CONTACT
    return predicted_intent


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
        if prediction.confidence < self.confidence_threshold:
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
        resolved_intent = resolve_intent_boundary(
            normalized.normalized_text, prediction.intent, self.slot_extractor
        )
        boundary_features = (
            [f"intent_boundary:{prediction.intent.value}->{resolved_intent.value}"]
            if resolved_intent is not prediction.intent
            else []
        )
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
        missing_options = missing_required_slot_options(resolved_intent, extraction.slots)
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
                *([f"missing_required_slots:{'|'.join(missing_options)}"] if missing_options else []),
            ],
        )
