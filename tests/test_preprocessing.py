from pathlib import Path

from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.preprocessing import preprocess_dataset, preprocess_sample
from nvit_assistant.schemas import (
    AnnotationQuality,
    DataSource,
    DatasetSample,
    Intent,
    Region,
    VariantType,
)


def make_sample(sample_id: str, text: str, contact_name: str) -> DatasetSample:
    """Tạo sample gọi điện tối thiểu cho test preprocess text và slot cùng lúc."""
    return DatasetSample(
        id=sample_id,
        group_id=sample_id,
        text=text,
        region=Region.SOUTH,
        intent=Intent.CALL_CONTACT,
        slots={"contact_name": contact_name},
        source=DataSource.MANUAL,
        variant_type=VariantType.REGIONAL,
        annotation_quality=AnnotationQuality.REVIEWED,
    )


def test_preprocess_sample_normalizes_slot_with_text() -> None:
    normalizer = VietnameseNormalizer(Path("configs/regional_variants.yaml"))
    prepared = preprocess_sample(make_sample("a", "gọi má dùm tui", "má"), normalizer)

    assert prepared.normalized_text == "gọi mẹ giúp tôi"
    assert prepared.normalized_slots == {"contact_name": "mẹ"}


def test_preprocess_dataset_deduplicates_train_only(tmp_path: Path) -> None:
    normalizer = VietnameseNormalizer(Path("configs/regional_variants.yaml"))
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    train_samples = [
        make_sample("a", "gọi má dùm tui", "má"),
        make_sample("b", "gọi mẹ giúp tôi", "mẹ"),
    ]
    test_samples = [
        make_sample("c", "gọi má dùm tui", "má"),
        make_sample("d", "gọi mẹ giúp tôi", "mẹ"),
    ]
    for filename, samples in {"train.jsonl": train_samples, "test_south.jsonl": test_samples}.items():
        with (input_dir / filename).open("w", encoding="utf-8") as file:
            for sample in samples:
                file.write(sample.model_dump_json() + "\n")

    report = preprocess_dataset(input_dir, tmp_path / "output", normalizer)

    assert report["files"]["train.jsonl"] == {"input": 2, "output": 1, "dropped": 1}
    assert report["files"]["test_south.jsonl"] == {"input": 2, "output": 2, "dropped": 0}
