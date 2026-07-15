"""Kiểm tra schema, provenance, duplicate, leakage và phân bố của dataset."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from nvit_assistant.schemas import DataSource, DatasetSample, Intent, Region

if TYPE_CHECKING:
    from nvit_assistant.nlu.normalizer import VietnameseNormalizer


EXPECTED_FILES = {
    "train.jsonl",
    "validation.jsonl",
    "test_standard.jsonl",
    "test_north.jsonl",
    "test_central.jsonl",
    "test_south.jsonl",
}
TEST_REGION_BY_FILE = {
    "test_standard.jsonl": Region.STANDARD,
    "test_north.jsonl": Region.NORTH,
    "test_central.jsonl": Region.CENTRAL,
    "test_south.jsonl": Region.SOUTH,
}
NEAR_SIMILARITY_THRESHOLD = 0.88


def _sha256_file(path: Path) -> str:
    """Hash file theo bytes; `.gitattributes` khóa LF để checksum ổn định đa nền tảng."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text(text: str) -> str:
    """Chuẩn hóa Unicode, chữ thường, dấu câu và khoảng trắng để tìm duplicate."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s:/]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def strip_diacritics(text: str) -> str:
    """Tạo fingerprint không dấu để bắt cùng một câu ở hai dạng chính tả."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped).replace("đ", "d").replace("Đ", "D")


def split_name(filename: str) -> str:
    """Quy mọi file test vùng miền về cùng logical split `test`."""
    if filename.startswith("test_"):
        return "test"
    return Path(filename).stem


def read_samples(path: Path) -> list[DatasetSample]:
    """Đọc UTF-8 JSONL và báo chính xác file/dòng nếu schema sai."""
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                samples.append(DatasetSample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return samples


def iter_slot_values(value: Any) -> Iterable[str]:
    """Trải phẳng slot dạng chuỗi hoặc danh sách để kiểm tra surface value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_slot_values(item)


def _masked_text(text: str, slots: dict[str, str | list[str]]) -> str:
    """Che các surface slot trên một biểu diễn text đã chọn."""
    text = strip_diacritics(canonical_text(text))
    replacements: list[tuple[str, str]] = []
    for slot_name, slot_value in slots.items():
        for surface_value in iter_slot_values(slot_value):
            normalized_value = strip_diacritics(canonical_text(surface_value))
            if normalized_value:
                replacements.append((normalized_value, f"<{slot_name}>"))
    for surface_value, placeholder in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(surface_value, placeholder)
    text = re.sub(r"\b\d[\d\s]*\d\b", "<number>", text)
    return re.sub(r"\s+", " ", text).strip()


def masked_sample_text(sample: DatasetSample) -> str:
    """Che surface slot trên câu gốc để bắt duplicate cấu trúc dễ thấy."""
    return _masked_text(sample.text, sample.slots)


def normalized_masked_sample_text(
    sample: DatasetSample, normalizer: VietnameseNormalizer
) -> str:
    """Tạo template trên đúng text sau normalize mà intent model sẽ nhìn thấy."""
    normalized_text = normalizer.normalize(sample.text, sample.region).normalized_text
    normalized_slots: dict[str, str | list[str]] = {}
    for slot_name, raw_value in sample.slots.items():
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        normalized_values = [
            normalizer.normalize(value, sample.region).normalized_text for value in values
        ]
        normalized_slots[slot_name] = (
            normalized_values if isinstance(raw_value, list) else normalized_values[0]
        )
    return _masked_text(normalized_text, normalized_slots)


def are_near_similar(left: str, right: str, threshold: float = NEAR_SIMILARITY_THRESHOLD) -> bool:
    """So sánh hai template đã mask bằng SequenceMatcher với các bước lọc nhanh."""
    if left == right:
        return True
    if not left or not right:
        return False
    length_ratio = min(len(left), len(right)) / max(len(left), len(right))
    if length_ratio < threshold:
        return False
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    if token_overlap < 0.5:
        return False
    return SequenceMatcher(None, left, right).ratio() >= threshold


def find_cross_split_near_duplicates(
    records: list[tuple[str, DatasetSample]],
    limit: int = 50,
    normalizer: VietnameseNormalizer | None = None,
) -> list[tuple[str, str, str, str, float]]:
    """Tìm template gần giống giữa split, tùy chọn trên text sau normalize."""
    templates_by_split: dict[str, dict[str, str]] = defaultdict(dict)
    for split, sample in records:
        template = (
            normalized_masked_sample_text(sample, normalizer)
            if normalizer is not None
            else masked_sample_text(sample)
        )
        templates_by_split[split].setdefault(template, sample.id)

    results: list[tuple[str, str, str, str, float]] = []
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left_split, right_split in split_pairs:
        for left_template, left_id in templates_by_split[left_split].items():
            for right_template, right_id in templates_by_split[right_split].items():
                if are_near_similar(left_template, right_template):
                    score = SequenceMatcher(None, left_template, right_template).ratio()
                    results.append((left_split, left_id, right_split, right_id, score))
                    if len(results) >= limit:
                        return results
    return results


def slot_value_appears_in_text(slot_name: str, value: str, text: str) -> bool:
    """Kiểm tra slot có thể truy ngược về câu, kể cả input không dấu."""
    if slot_name == "phone_number":
        value_digits = re.sub(r"\D", "", value)
        text_digits = re.sub(r"\D", "", text)
        return bool(value_digits) and value_digits in text_digits
    normalized_value = strip_diacritics(canonical_text(value))
    normalized_text = strip_diacritics(canonical_text(text))
    return bool(normalized_value) and normalized_value in normalized_text


def _distribution_ratio_errors(split_counts: Counter[str], total: int) -> list[str]:
    """Kiểm tra tỷ lệ split với tolerance vì còn phải giữ group và partition nguồn."""
    expected = {"train": 0.70, "validation": 0.15, "test": 0.15}
    errors: list[str] = []
    for split, expected_ratio in expected.items():
        actual = split_counts[split] / total if total else 0.0
        if abs(actual - expected_ratio) > 0.08:
            errors.append(
                f"split {split} ratio {actual:.3f} lệch quá 0.08 so với {expected_ratio:.2f}"
            )
    return errors


def validate_dataset_dir(
    data_dir: Path,
    enforce_minimums: bool = True,
    normalizer: VietnameseNormalizer | None = None,
) -> dict[str, Any]:
    """Chạy toàn bộ kiểm tra và trả report có errors/warnings thay vì dừng ở lỗi đầu."""
    files = sorted(path for path in data_dir.glob("*.jsonl") if path.is_file())
    errors: list[str] = []
    warnings: list[str] = []
    found_files = {path.name for path in files}
    missing_files = sorted(EXPECTED_FILES - found_files)
    if missing_files:
        errors.append(f"thiếu file dataset: {missing_files}")

    records: list[tuple[str, str, DatasetSample]] = []
    for path in files:
        try:
            samples = read_samples(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        logical_split = split_name(path.name)
        for sample in samples:
            records.append((path.name, logical_split, sample))

    manifest_verified = False
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        if enforce_minimums:
            errors.append("thiếu data/samples/manifest.json")
    else:
        try:
            manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"manifest.json không đọc được: {exc}")
        else:
            if not isinstance(manifest, dict):
                errors.append("manifest.json phải là mapping")
            else:
                recorded_hashes = manifest.get("files_sha256")
                if not isinstance(recorded_hashes, dict):
                    errors.append("manifest.files_sha256 phải là mapping")
                else:
                    for filename in sorted(EXPECTED_FILES):
                        path = data_dir / filename
                        if path.is_file() and recorded_hashes.get(filename) != _sha256_file(path):
                            errors.append(f"manifest checksum lệch: {filename}")
                recorded_counts = manifest.get("files")
                if not isinstance(recorded_counts, dict):
                    errors.append("manifest.files phải là mapping")
                else:
                    actual_counts = Counter(filename for filename, _, _ in records)
                    for filename in sorted(EXPECTED_FILES):
                        if recorded_counts.get(filename) != actual_counts[filename]:
                            errors.append(f"manifest sample count lệch: {filename}")
                if manifest.get("total") != len(records):
                    errors.append("manifest total không khớp số sample")
                test_digest = hashlib.sha256()
                for filename in sorted(
                    name for name in EXPECTED_FILES if name.startswith("test_")
                ):
                    path = data_dir / filename
                    if path.is_file():
                        test_digest.update(filename.encode("utf-8"))
                        test_digest.update(path.read_bytes())
                if manifest.get("test_set_sha256") != test_digest.hexdigest():
                    errors.append("manifest test_set_sha256 không khớp")
                manifest_verified = not any(
                    error.startswith("manifest") for error in errors
                )

    ids: dict[str, str] = {}
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    exact_fingerprints: dict[str, tuple[str, str]] = {}
    accentless_fingerprints: dict[str, tuple[str, str]] = {}
    source_refs: dict[str, str] = {}
    normalized_texts_by_split: dict[tuple[str, str], str] = {}

    for filename, logical_split, sample in records:
        if sample.id in ids:
            errors.append(f"ID trùng {sample.id}: {ids[sample.id]} và {filename}")
        ids[sample.id] = filename
        groups_by_split[sample.group_id].add(logical_split)

        exact = canonical_text(sample.text)
        previous_exact = exact_fingerprints.get(exact)
        if previous_exact and previous_exact[0] != sample.group_id:
            errors.append(
                f"câu trùng khác group: {sample.id} và {previous_exact[1]}"
            )
        exact_fingerprints[exact] = (sample.group_id, sample.id)

        accentless = strip_diacritics(exact)
        previous_accentless = accentless_fingerprints.get(accentless)
        if previous_accentless and previous_accentless[0] != sample.group_id:
            errors.append(
                f"câu tương đương không dấu khác group: {sample.id} và {previous_accentless[1]}"
            )
        accentless_fingerprints[accentless] = (sample.group_id, sample.id)

        if normalizer is not None:
            normalized_text = normalizer.normalize(sample.text, sample.region).normalized_text
            normalized_key = (logical_split, normalized_text)
            previous_normalized = normalized_texts_by_split.get(normalized_key)
            if previous_normalized is not None:
                errors.append(
                    "câu trùng trong cùng split sau normalize: "
                    f"{sample.id} và {previous_normalized} ({logical_split})"
                )
            normalized_texts_by_split[normalized_key] = sample.id

        if sample.source_ref and sample.source in {
            DataSource.MASSIVE,
            DataSource.OLD_PROJECT,
            DataSource.WEB_MINED,
        }:
            previous_ref = source_refs.get(sample.source_ref)
            if previous_ref and previous_ref != sample.group_id:
                errors.append(
                    f"source_ref {sample.source_ref} thuộc nhiều group: {previous_ref}, {sample.group_id}"
                )
            source_refs[sample.source_ref] = sample.group_id

        expected_region = TEST_REGION_BY_FILE.get(filename)
        if expected_region is not None and sample.region is not expected_region:
            errors.append(
                f"{sample.id} có region={sample.region.value} nhưng nằm trong {filename}"
            )

        for slot_name, slot_value in sample.slots.items():
            for surface_value in iter_slot_values(slot_value):
                if not slot_value_appears_in_text(slot_name, surface_value, sample.text):
                    errors.append(
                        f"{sample.id}: slot {slot_name}={surface_value!r} không xuất hiện trong text"
                    )

    for group_id, splits in groups_by_split.items():
        if len(splits) > 1:
            errors.append(f"leakage group {group_id} xuất hiện ở nhiều split: {sorted(splits)}")

    near_duplicates = find_cross_split_near_duplicates(
        [(logical_split, sample) for _, logical_split, sample in records],
        normalizer=normalizer,
    )
    for left_split, left_id, right_split, right_id, score in near_duplicates:
        errors.append(
            f"near-similar cross-split ({score:.3f}): "
            f"{left_split}/{left_id} và {right_split}/{right_id}"
        )

    samples = [sample for _, _, sample in records]
    split_counts = Counter(split for _, split, _ in records)
    intent_counts = Counter(sample.intent.value for sample in samples)
    region_counts = Counter(sample.region.value for sample in samples)
    source_counts = Counter(sample.source.value for sample in samples)
    quality_counts = Counter(sample.annotation_quality.value for sample in samples)
    variant_counts = Counter(sample.variant_type.value for sample in samples)
    split_intent_counts = Counter((split, sample.intent.value) for _, split, sample in records)
    split_region_counts = Counter((split, sample.region.value) for _, split, sample in records)
    split_slot_counts = Counter(
        (split, slot_name)
        for _, split, sample in records
        for slot_name in sample.slots
    )

    if enforce_minimums:
        if len(samples) < 1200:
            errors.append(f"dataset cần ít nhất 1200 sample, hiện có {len(samples)}")
        for intent in (intent for intent in Intent if intent is not Intent.UNKNOWN):
            if intent_counts[intent.value] < 200:
                errors.append(
                    f"intent {intent.value} cần ít nhất 200 sample, hiện có {intent_counts[intent.value]}"
                )
        for region in (Region.NORTH, Region.CENTRAL, Region.SOUTH):
            if region_counts[region.value] < 250:
                errors.append(
                    f"region {region.value} cần ít nhất 250 sample, hiện có {region_counts[region.value]}"
                )
        errors.extend(_distribution_ratio_errors(split_counts, len(samples)))

        test_records = [(filename, sample) for filename, split, sample in records if split == "test"]
        test_breakdown = Counter((sample.region.value, sample.intent.value) for _, sample in test_records)
        for region in (Region.NORTH, Region.CENTRAL, Region.SOUTH):
            for intent in (intent for intent in Intent if intent is not Intent.UNKNOWN):
                if test_breakdown[(region.value, intent.value)] < 8:
                    errors.append(
                        f"test {region.value}/{intent.value} cần >=8 sample, hiện có "
                        f"{test_breakdown[(region.value, intent.value)]}"
                    )

        for split in ("validation", "test"):
            for slot_name in sorted(DatasetSample.allowed_slots):
                if split_slot_counts[(split, slot_name)] < 8:
                    errors.append(
                        f"{split} slot {slot_name} cần >=8 sample, hiện có "
                        f"{split_slot_counts[(split, slot_name)]}"
                    )

    native_speaker_reviewed = sum(
        bool(sample.source_ref and sample.source_ref.startswith("native_review:"))
        for sample in samples
    )
    if native_speaker_reviewed == 0:
        warnings.append(
            "chưa có transcript/audio kèm provenance native_review; regional set hiện là template"
        )

    return {
        "manifest_verified": manifest_verified,
        "total": len(samples),
        "files": dict(sorted(Counter(filename for filename, _, _ in records).items())),
        "splits": dict(sorted(split_counts.items())),
        "intents": dict(sorted(intent_counts.items())),
        "regions": dict(sorted(region_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "annotation_quality": dict(sorted(quality_counts.items())),
        "variants": dict(sorted(variant_counts.items())),
        "by_split_intent": {
            split: {
                intent: split_intent_counts[(split, intent)]
                for intent in sorted(intent_counts)
            }
            for split in ("train", "validation", "test")
        },
        "by_split_region": {
            split: {
                region: split_region_counts[(split, region)]
                for region in sorted(region_counts)
            }
            for split in ("train", "validation", "test")
        },
        "by_split_slot": {
            split: {
                slot_name: split_slot_counts[(split, slot_name)]
                for slot_name in sorted(DatasetSample.allowed_slots)
            }
            for split in ("train", "validation", "test")
        },
        "native_speaker_reviewed": native_speaker_reviewed,
        "errors": errors,
        "warnings": warnings,
    }
