"""Tạo lexicon entity chỉ từ train để slot extractor không học lén validation/test."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from nvit_assistant.data_validation import iter_slot_values, read_samples
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.preprocessing import preprocess_sample


LEXICON_SLOTS = frozenset(
    {"artist", "contact_name", "location", "reminder_text", "song"}
)


def sha256_file(path: Path) -> str:
    """Tính hash nội dung để runtime/report phát hiện lexicon đã cũ."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_slot_lexicon(
    train_path: Path,
    normalizer: VietnameseNormalizer,
    normalizer_config_path: Path,
) -> dict[str, Any]:
    """Thu thập slot từ train và khóa cả config đã dùng để normalize."""
    values: defaultdict[str, set[str]] = defaultdict(set)
    samples = read_samples(train_path)
    for sample in samples:
        prepared = preprocess_sample(sample, normalizer)
        for slot_name, raw_value in prepared.normalized_slots.items():
            if slot_name in LEXICON_SLOTS:
                values[slot_name].update(iter_slot_values(raw_value))
    return {
        "schema_version": 2,
        "source_split": "train",
        "source_file": "data/samples/train.jsonl",
        "source_sha256": sha256_file(train_path),
        "normalizer_config_sha256": sha256_file(normalizer_config_path),
        "sample_count": len(samples),
        "slots": {
            slot_name: sorted(values.get(slot_name, set()))
            for slot_name in sorted(LEXICON_SLOTS)
        },
    }


def write_slot_lexicon(payload: dict[str, Any], output_path: Path) -> None:
    """Ghi lexicon deterministic để có thể review diff và commit cùng model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_slot_lexicon(path: Path) -> dict[str, tuple[str, ...]]:
    """Validate cấu trúc artifact trước khi đưa giá trị vào extractor."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("slot lexicon phải có schema_version=2")
    if raw.get("source_split") != "train":
        raise ValueError("slot lexicon chỉ được phép sinh từ train")
    slots = raw.get("slots")
    if not isinstance(slots, dict):
        raise ValueError("slot lexicon.slots phải là mapping")
    unknown = sorted(set(slots) - LEXICON_SLOTS)
    if unknown:
        raise ValueError(f"slot lexicon có field không hỗ trợ: {unknown}")
    result: dict[str, tuple[str, ...]] = {}
    for slot_name in LEXICON_SLOTS:
        values = slots.get(slot_name, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"slot lexicon {slot_name} phải là list chuỗi không rỗng")
        result[slot_name] = tuple(values)
    return result


def validate_slot_lexicon_provenance(
    path: Path, train_path: Path, normalizer_config_path: Path
) -> None:
    """Fail-fast nếu train hoặc config normalizer đổi sau khi sinh lexicon."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 2
        or raw.get("source_split") != "train"
    ):
        raise ValueError("slot lexicon không khai báo schema_version=2 và source_split=train")
    if raw.get("source_sha256") != sha256_file(train_path):
        raise ValueError(
            "slot lexicon đã cũ so với data/samples/train.jsonl; "
            "hãy chạy scripts/build_slot_lexicon.py"
        )
    if raw.get("normalizer_config_sha256") != sha256_file(normalizer_config_path):
        raise ValueError(
            "slot lexicon đã cũ so với configs/regional_variants.yaml; "
            "hãy chạy scripts/build_slot_lexicon.py"
        )
