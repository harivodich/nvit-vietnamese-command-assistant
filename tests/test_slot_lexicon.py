import json
from pathlib import Path

import pytest

from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.slot_lexicon import (
    build_slot_lexicon,
    load_slot_lexicon,
    validate_slot_lexicon_provenance,
    write_slot_lexicon,
)
from nvit_assistant.schemas import (
    AnnotationQuality,
    DataSource,
    DatasetSample,
    Intent,
    Region,
    VariantType,
)


def write_train_sample(path: Path) -> None:
    sample = DatasetSample(
        id="train-music-1",
        group_id="train-music-1",
        text="mở bài kiểm thử của ca sĩ kiểm thử",
        region=Region.STANDARD,
        intent=Intent.PLAY_MUSIC,
        slots={"song": "bài kiểm thử", "artist": "ca sĩ kiểm thử"},
        source=DataSource.MANUAL,
        variant_type=VariantType.ACCENTED,
        annotation_quality=AnnotationQuality.REVIEWED,
    )
    path.write_text(
        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_slot_lexicon_records_train_only_provenance(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "slot_lexicon.json"
    write_train_sample(train_path)
    normalizer_config_path = Path("configs/regional_variants.yaml")
    normalizer = VietnameseNormalizer(normalizer_config_path)

    payload = build_slot_lexicon(train_path, normalizer, normalizer_config_path)
    write_slot_lexicon(payload, output_path)

    assert payload["source_split"] == "train"
    assert payload["slots"]["song"] == ["bài kiểm thử"]
    assert load_slot_lexicon(output_path)["artist"] == ("ca sĩ kiểm thử",)
    validate_slot_lexicon_provenance(output_path, train_path, normalizer_config_path)


def test_slot_lexicon_detects_stale_train_hash(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "slot_lexicon.json"
    write_train_sample(train_path)
    normalizer_config_path = Path("configs/regional_variants.yaml")
    normalizer = VietnameseNormalizer(normalizer_config_path)
    write_slot_lexicon(
        build_slot_lexicon(train_path, normalizer, normalizer_config_path), output_path
    )
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="đã cũ"):
        validate_slot_lexicon_provenance(output_path, train_path, normalizer_config_path)


def test_slot_lexicon_detects_stale_normalizer_config(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "slot_lexicon.json"
    normalizer_config_path = tmp_path / "regional_variants.yaml"
    write_train_sample(train_path)
    source_config = Path("configs/regional_variants.yaml")
    normalizer_config_path.write_bytes(source_config.read_bytes())
    normalizer = VietnameseNormalizer(normalizer_config_path)
    write_slot_lexicon(
        build_slot_lexicon(train_path, normalizer, normalizer_config_path), output_path
    )
    normalizer_config_path.write_text(
        normalizer_config_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="regional_variants"):
        validate_slot_lexicon_provenance(
            output_path, train_path, normalizer_config_path
        )
