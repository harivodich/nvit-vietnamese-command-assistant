"""Ghép normalizer, intent classifier và slot extractor thành core NLU."""

from __future__ import annotations

from collections.abc import Mapping

from nvit_assistant.actions import ActionRouter
from nvit_assistant.nlu.intent_classifier import IntentClassifier
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import DatasetSample, Intent, ParseRequest, ParseResult


def missing_required_slot_options(intent: Intent, slots: Mapping[str, object]) -> tuple[str, ...]:
    """Trả các phương án slot bắt buộc khi chưa nhóm nào thỏa contract của intent."""
    required_groups = DatasetSample.required_slot_groups.get(intent)
    if required_groups is None or any(group.issubset(slots) for group in required_groups):
        return ()
    return tuple("/".join(sorted(group)) for group in required_groups)


class NLUPipeline:
    """Pipeline text-first; action thật và response tự nhiên được bổ sung ở phase sau."""

    def __init__(
        self,
        normalizer: VietnameseNormalizer,
        intent_classifier: IntentClassifier,
        slot_extractor: RegexSlotExtractor,
        action_router: ActionRouter | None = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold phải nằm trong [0, 1]")
        self.normalizer = normalizer
        self.intent_classifier = intent_classifier
        self.slot_extractor = slot_extractor
        self.action_router = action_router
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
        extraction = self.slot_extractor.extract(normalized.normalized_text, prediction.intent)
        missing_options = missing_required_slot_options(prediction.intent, extraction.slots)
        action = None
        if missing_options:
            response = f"Cần bổ sung slot bắt buộc: {' hoặc '.join(missing_options)}."
        elif self.action_router is not None:
            execution = self.action_router.execute(prediction.intent, extraction.slots)
            action = execution.result
            response = execution.response
        else:
            response = f"Đã nhận diện intent {prediction.intent.value}."
        return ParseResult(
            text=request.text,
            normalized_text=normalized.normalized_text,
            region=normalized.region,
            intent=prediction.intent,
            confidence=prediction.confidence,
            slots=extraction.slots,
            action=action,
            response=response,
            matched_features=[
                *normalized.matched_variants,
                f"intent_model:{prediction.intent.value}",
                *extraction.matched_features,
                *([f"missing_required_slots:{'|'.join(missing_options)}"] if missing_options else []),
            ],
        )
