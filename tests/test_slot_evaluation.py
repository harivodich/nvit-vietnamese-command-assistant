from pathlib import Path

from nvit_assistant.eval.slot_evaluation import evaluate_slot_extractor
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import (
    AnnotationQuality,
    DataSource,
    DatasetSample,
    Intent,
    PreprocessedSample,
    Region,
    VariantType,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample(identifier: str, text: str, intent: Intent, slots: dict[str, str]) -> PreprocessedSample:
    """Tạo sample tối thiểu để kiểm tra phép tính metric, không phụ thuộc JSONL thật."""
    original = DatasetSample(
        id=identifier,
        group_id=identifier,
        text=text,
        region=Region.STANDARD,
        intent=intent,
        slots=slots,
        source=DataSource.MANUAL,
        variant_type=VariantType.ACCENTED,
        annotation_quality=AnnotationQuality.REVIEWED,
    )
    return PreprocessedSample(
        original=original,
        normalized_text=text,
        normalized_slots=slots,
        normalizer_region=Region.STANDARD,
    )


def test_slot_evaluation_counts_exact_and_missing_values() -> None:
    extractor = RegexSlotExtractor(ROOT / "configs" / "slot_values.yaml")
    samples = [
        _sample("correct", "gọi cho mẹ", Intent.CALL_CONTACT, {"contact_name": "mẹ"}),
        _sample("missing", "đặt báo thức", Intent.SET_ALARM, {"datetime": "bảy giờ"}),
    ]

    report = evaluate_slot_extractor(extractor, samples)

    assert report["slot_exact_match"] == 0.5
    assert report["micro"]["precision"] == 1.0
    assert report["micro"]["recall"] == 0.5
    assert report["failure_count"] == 1
    assert report["breakdown"]["source"]["manual"]["total_samples"] == 2
    assert report["breakdown"]["intent"]["call_contact"]["slot_exact_match"] == 1.0
