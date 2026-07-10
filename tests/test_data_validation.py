import json
from pathlib import Path

from nvit_assistant.data_validation import validate_dataset_dir
from nvit_assistant.schemas import (
    AnnotationQuality,
    DataSource,
    DatasetSample,
    Intent,
    Region,
    VariantType,
)


def write_sample(path: Path, sample: DatasetSample) -> None:
    """Ghi một sample fixture để test validator trên file JSONL thật."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")


def make_call_sample(sample_id: str, group_id: str, text: str) -> DatasetSample:
    """Tạo fixture call_contact tối thiểu nhưng hợp lệ."""
    return DatasetSample(
        id=sample_id,
        group_id=group_id,
        text=text,
        region=Region.SOUTH,
        intent=Intent.CALL_CONTACT,
        slots={"contact_name": "mẹ"},
        source=DataSource.MANUAL,
        variant_type=VariantType.ACCENTED,
        annotation_quality=AnnotationQuality.REVIEWED,
    )


def test_validator_detects_group_leakage(tmp_path: Path) -> None:
    write_sample(tmp_path / "train.jsonl", make_call_sample("a", "same-group", "gọi mẹ"))
    write_sample(tmp_path / "test_south.jsonl", make_call_sample("b", "same-group", "gọi cho mẹ"))

    report = validate_dataset_dir(tmp_path, enforce_minimums=False)

    assert any("leakage group" in error for error in report["errors"])


def test_validator_allows_accent_variant_inside_same_group(tmp_path: Path) -> None:
    accented = make_call_sample("a", "same-group", "gọi mẹ")
    no_diacritics = accented.model_copy(
        update={"id": "b", "text": "goi me", "variant_type": VariantType.NO_DIACRITICS}
    )
    write_sample(tmp_path / "train.jsonl", accented)
    write_sample(tmp_path / "train.jsonl", no_diacritics)

    report = validate_dataset_dir(tmp_path, enforce_minimums=False)

    assert not any("tương đương không dấu" in error for error in report["errors"])


def test_validator_detects_slot_not_present_in_text(tmp_path: Path) -> None:
    sample = make_call_sample("a", "group-a", "gọi cho bố")
    write_sample(tmp_path / "train.jsonl", sample)

    report = validate_dataset_dir(tmp_path, enforce_minimums=False)

    assert any("không xuất hiện trong text" in error for error in report["errors"])


def test_validator_detects_near_similar_template_across_splits(tmp_path: Path) -> None:
    train_sample = make_call_sample("train-call", "train-group", "gọi cho mẹ giúp tôi")
    test_sample = DatasetSample(
        id="test-call",
        group_id="test-group",
        text="gọi cho bố giúp tôi",
        region=Region.SOUTH,
        intent=Intent.CALL_CONTACT,
        slots={"contact_name": "bố"},
        source=DataSource.MANUAL,
        variant_type=VariantType.ACCENTED,
        annotation_quality=AnnotationQuality.REVIEWED,
    )
    write_sample(tmp_path / "train.jsonl", train_sample)
    write_sample(tmp_path / "test_south.jsonl", test_sample)

    report = validate_dataset_dir(tmp_path, enforce_minimums=False)

    assert any("near-similar cross-split" in error for error in report["errors"])


def test_committed_dataset_passes_validation() -> None:
    report = validate_dataset_dir(Path("data/samples"))

    assert report["errors"] == []
    assert report["total"] >= 1200
