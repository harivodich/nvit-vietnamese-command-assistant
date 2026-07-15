"""Tạo artifact train/evaluation bằng cùng normalizer mà runtime áp dụng online."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from nvit_assistant.data_validation import iter_slot_values, slot_value_appears_in_text
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.schemas import DatasetSample, PreprocessedSample


def normalize_slot_value(
    slot_name: str,
    value: Any,
    normalized_text: str,
    normalizer: VietnameseNormalizer,
    sample: DatasetSample,
) -> Any:
    """Chuẩn hóa slot khi an toàn; no-diacritics mơ hồ thì giữ surface gốc để không lệch text."""
    if isinstance(value, str):
        candidate = normalizer.normalize(value, sample.region).normalized_text
        if slot_value_appears_in_text(slot_name, candidate, normalized_text):
            return candidate
        return value
    if isinstance(value, list):
        return [
            normalize_slot_value(slot_name, item, normalized_text, normalizer, sample)
            for item in value
        ]
    return value


def preprocess_sample(sample: DatasetSample, normalizer: VietnameseNormalizer) -> PreprocessedSample:
    """Chuẩn hóa đồng thời text/slot và kiểm tra slot vẫn truy ngược được từ text mới."""
    result = normalizer.normalize(sample.text, sample.region)
    normalized_slots = {
        slot_name: normalize_slot_value(
            slot_name, slot_value, result.normalized_text, normalizer, sample
        )
        for slot_name, slot_value in sample.slots.items()
    }
    for slot_name, slot_value in normalized_slots.items():
        for surface_value in iter_slot_values(slot_value):
            if not slot_value_appears_in_text(slot_name, surface_value, result.normalized_text):
                raise ValueError(
                    f"{sample.id}: slot {slot_name}={surface_value!r} mất khỏi normalized_text"
                )
    return PreprocessedSample(
        original=sample,
        normalized_text=result.normalized_text,
        normalized_slots=normalized_slots,
        normalizer_region=result.region,
        matched_variants=list(result.matched_variants),
    )


def preprocess_file(
    input_path: Path,
    output_path: Path,
    normalizer: VietnameseNormalizer,
    deduplicate: bool,
) -> dict[str, int]:
    """Preprocess một JSONL; chỉ train/validation dedupe để không đổi bộ test gốc."""
    from nvit_assistant.data_validation import read_samples

    prepared: list[PreprocessedSample] = []
    seen_intents: dict[str, str] = {}
    dropped = 0
    for sample in read_samples(input_path):
        item = preprocess_sample(sample, normalizer)
        previous_intent = seen_intents.get(item.normalized_text)
        if deduplicate and previous_intent is not None:
            if previous_intent != sample.intent.value:
                raise ValueError(
                    f"{input_path}: normalized_text {item.normalized_text!r} có hai intent khác nhau"
                )
            dropped += 1
            continue
        seen_intents[item.normalized_text] = sample.intent.value
        prepared.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for item in prepared:
            file.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
    return {"input": len(prepared) + dropped, "output": len(prepared), "dropped": dropped}


def preprocess_dataset(
    input_dir: Path, output_dir: Path, normalizer: VietnameseNormalizer
) -> dict[str, Any]:
    """Tạo artifact preprocess deterministic; train/validation dedupe, test giữ nguyên."""
    return preprocess_splits(
        input_dir,
        output_dir,
        normalizer,
        tuple(path.name for path in sorted(input_dir.glob("*.jsonl"))),
    )


def preprocess_splits(
    input_dir: Path,
    output_dir: Path,
    normalizer: VietnameseNormalizer,
    filenames: tuple[str, ...],
) -> dict[str, Any]:
    """Preprocess đúng các split được chỉ định để bước train không cần đọc test."""
    file_reports: dict[str, dict[str, int]] = {}
    for filename in filenames:
        input_path = input_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(f"không tìm thấy split cần preprocess: {input_path}")
        deduplicate = input_path.name in {"train.jsonl", "validation.jsonl"}
        file_reports[input_path.name] = preprocess_file(
            input_path,
            output_dir / input_path.name,
            normalizer,
            deduplicate=deduplicate,
        )
    totals: Counter[str] = Counter()
    for report in file_reports.values():
        totals.update(report)
    return {"files": file_reports, "totals": dict(totals)}
