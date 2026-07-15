"""Xây dựng dataset có nguồn gốc rõ ràng và split chống leakage."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from nvit_assistant.data_validation import (
    are_near_similar,
    masked_sample_text,
    normalized_masked_sample_text,
)
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.schemas import (
    AnnotationQuality,
    DataSource,
    DatasetSample,
    Intent,
    Region,
    VariantType,
)


MASSIVE_INTENT_MAP = {
    "alarm_set": Intent.SET_ALARM,
    "weather_query": Intent.ASK_WEATHER,
    "play_music": Intent.PLAY_MUSIC,
    "calendar_set": Intent.SET_REMINDER,
}
MASSIVE_LIMITS = {
    Intent.SET_ALARM: 250,
    Intent.ASK_WEATHER: 400,
    Intent.PLAY_MUSIC: 400,
    Intent.SET_REMINDER: 150,
}
MASSIVE_PARTITIONS = {"train": "train", "dev": "validation", "test": "test"}
MASSIVE_TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REGIONAL_GROUP_TARGETS = {
    Intent.SET_REMINDER: 50,
    Intent.SET_ALARM: 52,
    Intent.ASK_WEATHER: 30,
    Intent.PLAY_MUSIC: 30,
    Intent.CALL_CONTACT: 74,
}
DATETIME_LABELS = {"date", "time", "timeofday", "general_frequency"}
ANNOTATED_SLOT_PATTERN = re.compile(r"\[([a-z_]+)\s*:\s*([^\]]+)\]")
REMINDER_MARKERS = re.compile(r"\b(nhắc|nhở|lời nhắc)\b", flags=re.IGNORECASE)
TEMPORAL_JOINER_PATTERN = re.compile(
    r"^\s*(?:(?:lúc|vào|khoảng|tầm|đến|tới)\s*)?$", flags=re.IGNORECASE
)
TEMPLATE_SLOT_PATTERN = re.compile(r"\{([a-z_]+)\}")


@dataclass(frozen=True)
class AnnotatedSpan:
    """Một slot span đã được chuyển từ MASSIVE markup về offset trên câu thuần."""

    label: str
    value: str
    start: int
    end: int


def normalize_text(text: str) -> str:
    """Chuẩn Unicode NFC và khoảng trắng nhưng không làm mất dấu tiếng Việt."""
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def strip_diacritics(text: str) -> str:
    """Tạo biến thể không dấu, bao gồm chuyển đ/Đ thành d/D."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped).replace("đ", "d").replace("Đ", "D")


def text_fingerprint(text: str) -> str:
    """Tạo fingerprint không dấu để deduplicate trước khi chia dataset."""
    text = strip_diacritics(normalize_text(text).casefold())
    text = re.sub(r"[^\w\s:/]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def parse_annotated_utterance(annotated_text: str) -> tuple[str, list[AnnotatedSpan]]:
    """Bỏ MASSIVE markup và trả lại câu thuần cùng offset của từng slot."""
    annotated_text = unicodedata.normalize("NFC", annotated_text)
    parts: list[str] = []
    spans: list[AnnotatedSpan] = []
    cursor = 0
    plain_length = 0

    for match in ANNOTATED_SLOT_PATTERN.finditer(annotated_text):
        prefix = annotated_text[cursor : match.start()]
        parts.append(prefix)
        plain_length += len(prefix)

        value = normalize_text(match.group(2))
        start = plain_length
        parts.append(value)
        plain_length += len(value)
        spans.append(AnnotatedSpan(match.group(1), value, start, plain_length))
        cursor = match.end()

    parts.append(annotated_text[cursor:])
    plain_text = "".join(parts)
    return plain_text, spans


def combine_temporal_spans(
    text: str, spans: list[AnnotatedSpan], labels: set[str]
) -> str | list[str] | None:
    """Chỉ ghép span thời gian liền nhau; không nuốt event nằm giữa hai span."""
    selected = sorted((span for span in spans if span.label in labels), key=lambda span: span.start)
    if not selected:
        return None

    components: list[str] = []
    component_start = selected[0].start
    component_end = selected[0].end
    for span in selected[1:]:
        separator = text[component_end : span.start]
        if TEMPORAL_JOINER_PATTERN.fullmatch(separator):
            component_end = span.end
        else:
            components.append(normalize_text(text[component_start:component_end]))
            component_start = span.start
            component_end = span.end
    components.append(normalize_text(text[component_start:component_end]))
    return components[0] if len(components) == 1 else components


def first_span_value(spans: list[AnnotatedSpan], label: str) -> str | None:
    """Lấy giá trị span đầu tiên của một nhãn MASSIVE."""
    for span in spans:
        if span.label == label:
            return span.value
    return None


def massive_row_has_good_judgments(row: dict[str, Any]) -> bool:
    """Giữ các bản dịch được đa số annotator xác nhận intent, ngôn ngữ và ngữ pháp."""
    judgments = row.get("judgments") or []
    if len(judgments) < 2:
        return False
    intent_votes = sum(int(item.get("intent_score") == 1) for item in judgments)
    language_votes = sum(item.get("language_identification") == "target" for item in judgments)
    grammar_scores = [float(item.get("grammar_score", 0)) for item in judgments]
    spelling_scores = [float(item.get("spelling_score", 0)) for item in judgments]
    slot_votes = sum(item.get("slots_score") in {1, 2} for item in judgments)
    return (
        intent_votes >= 2
        and language_votes >= 2
        and slot_votes >= 2
        and sum(grammar_scores) / len(grammar_scores) >= 3.5
        and sum(spelling_scores) / len(spelling_scores) >= 1.5
    )


def map_massive_row(row: dict[str, Any]) -> tuple[str, DatasetSample] | None:
    """Ánh xạ một dòng MASSIVE vào contract dự án; trả None nếu ánh xạ không chắc chắn."""
    source_intent = str(row.get("intent", ""))
    intent = MASSIVE_INTENT_MAP.get(source_intent)
    partition = MASSIVE_PARTITIONS.get(str(row.get("partition", "")))
    if intent is None or partition is None or not massive_row_has_good_judgments(row):
        return None

    marked_text, spans = parse_annotated_utterance(str(row.get("annot_utt", "")))
    text = normalize_text(
        re.sub(r"^(olly|alexa)\s*[,，]?\s*", "", marked_text, flags=re.IGNORECASE)
    )
    if len(text.split()) < 3:
        return None

    slots: dict[str, str | list[str]] = {}
    datetime_value = combine_temporal_spans(marked_text, spans, DATETIME_LABELS)
    # Năm intent hiện dùng một datetime duy nhất. Nhiều cụm thời gian tách rời
    # thường là quan hệ mơ hồ; loại thay vì tạo nhãn ghép mà extractor không thể học.
    if isinstance(datetime_value, list):
        return None

    if intent is Intent.SET_ALARM:
        if not datetime_value:
            return None
        slots["datetime"] = datetime_value
    elif intent is Intent.ASK_WEATHER:
        location = first_span_value(spans, "place_name")
        if location:
            slots["location"] = location
        if datetime_value:
            slots["datetime"] = datetime_value
    elif intent is Intent.PLAY_MUSIC:
        song = first_span_value(spans, "song_name")
        artist = first_span_value(spans, "artist_name")
        if song:
            slots["song"] = song
        if artist:
            slots["artist"] = artist
    elif intent is Intent.SET_REMINDER:
        reminder_text = first_span_value(spans, "event_name")
        if not reminder_text or not REMINDER_MARKERS.search(text):
            return None
        slots["reminder_text"] = reminder_text
        if datetime_value:
            slots["datetime"] = datetime_value

    source_id = str(row["id"])
    sample = DatasetSample(
        id=f"massive_vi_{source_intent}_{source_id}",
        group_id=f"massive_vi_{source_id}",
        text=text,
        region=Region.STANDARD,
        intent=intent,
        slots=slots,
        source=DataSource.MASSIVE,
        source_ref=f"massive:1.0:vi-VN:{source_id}",
        variant_type=VariantType.TRANSLATED,
        annotation_quality=AnnotationQuality.AUTO_MAPPED,
    )
    return partition, sample


def stable_order_key(seed: int, value: str) -> str:
    """Sinh khóa sắp xếp ổn định để sample không đổi khi chạy lại cùng seed."""
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def load_massive_samples(path: Path, seed: int) -> dict[str, list[DatasetSample]]:
    """Đọc, lọc chất lượng, cân bằng intent và giữ partition gốc của MASSIVE."""
    mapped_by_fingerprint: dict[str, list[tuple[str, DatasetSample]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                mapped = map_massive_row(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if mapped:
                partition, sample = mapped
                mapped_by_fingerprint[text_fingerprint(sample.text)].append((partition, sample))

    # Nếu cùng câu bị gán nhiều intent thì bỏ toàn bộ vì nhãn không còn chắc chắn.
    # Nếu chỉ trùng trong một intent, ưu tiên test rồi validation để tránh test leakage.
    candidates: dict[tuple[Intent, str], list[DatasetSample]] = defaultdict(list)
    partition_priority = {"test": 0, "validation": 1, "train": 2}
    for duplicate_rows in mapped_by_fingerprint.values():
        if len({sample.intent for _, sample in duplicate_rows}) != 1:
            continue
        partition, sample = min(
            duplicate_rows,
            key=lambda item: (partition_priority[item[0]], stable_order_key(seed, item[1].id)),
        )
        candidates[(sample.intent, partition)].append(sample)

    # Giữ test gốc trước, sau đó loại validation/train có cấu trúc quá giống split đã giữ.
    # Các câu cùng template trong một split vẫn được phép vì không làm rò rỉ sang split khác.
    filtered_candidates: dict[tuple[Intent, str], list[DatasetSample]] = defaultdict(list)
    protected_templates: list[str] = []
    for partition in ("test", "validation", "train"):
        current_templates: list[str] = []
        partition_samples = [
            sample
            for (intent_key, partition_key), values in candidates.items()
            if partition_key == partition
            for sample in values
        ]
        partition_samples.sort(key=lambda sample: stable_order_key(seed, sample.id))
        for sample in partition_samples:
            template = masked_sample_text(sample)
            if any(are_near_similar(template, protected) for protected in protected_templates):
                continue
            filtered_candidates[(sample.intent, partition)].append(sample)
            if template not in current_templates:
                current_templates.append(template)
        protected_templates.extend(current_templates)
    candidates = filtered_candidates

    selected: dict[str, list[DatasetSample]] = defaultdict(list)
    for intent, total_limit in MASSIVE_LIMITS.items():
        for partition, ratio in MASSIVE_TARGET_RATIOS.items():
            partition_candidates = candidates[(intent, partition)]
            partition_candidates.sort(key=lambda sample: stable_order_key(seed, sample.id))
            limit = round(total_limit * ratio)
            selected[partition].extend(partition_candidates[:limit])
    return selected


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc YAML và yêu cầu root là mapping để phát hiện config sai sớm."""
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def sample_values(rng: random.Random, slot_values: dict[str, Any]) -> dict[str, str]:
    """Chọn một bộ slot value tương thích cho template tổng hợp."""
    music_item = rng.choice(slot_values["music_catalog"])
    return {
        "contact_name": rng.choice(slot_values["contacts"]),
        "phone_number": rng.choice(slot_values["phone_numbers"]),
        "location": rng.choice(slot_values["locations"]),
        # Song và artist phải lấy cùng một record để không tạo cặp sai sự thật.
        "song": str(music_item["song"]),
        "artist": str(music_item["artist"]),
        "reminder_text": rng.choice(slot_values["reminder_texts"]),
        "datetime": rng.choice(slot_values["datetimes"]),
        "weather_time": rng.choice(slot_values["weather_times"]),
    }


def slots_for_template(
    intent: Intent, template: str, values: dict[str, str]
) -> dict[str, str | list[str]]:
    """Tạo nhãn slot từ placeholder thực sự xuất hiện trong template."""
    slots: dict[str, str | list[str]] = {}
    if "{datetime}" in template:
        slots["datetime"] = values["datetime"]
    if intent is Intent.SET_REMINDER:
        slots["reminder_text"] = values["reminder_text"]
    elif intent is Intent.ASK_WEATHER:
        slots["location"] = values["location"]
        if "{weather_time}" in template:
            slots["datetime"] = values["weather_time"]
    elif intent is Intent.PLAY_MUSIC:
        if "{song}" in template:
            slots["song"] = values["song"]
        if "{artist}" in template:
            slots["artist"] = values["artist"]
    elif intent is Intent.CALL_CONTACT:
        if "{contact_name}" in template:
            slots["contact_name"] = values["contact_name"]
        if "{phone_number}" in template:
            slots["phone_number"] = values["phone_number"]
    return slots


def split_group_ids(group_ids: list[str], seed: int) -> dict[str, str]:
    """Chia group theo tỷ lệ 70/15/15 để mọi variant cùng group ở một split."""
    ordered = sorted(group_ids, key=lambda value: stable_order_key(seed, value))
    train_end = round(len(ordered) * 0.70)
    validation_end = train_end + round(len(ordered) * 0.15)
    result: dict[str, str] = {}
    for index, group_id in enumerate(ordered):
        if index < train_end:
            result[group_id] = "train"
        elif index < validation_end:
            result[group_id] = "validation"
        else:
            result[group_id] = "test"
    return result


def template_slot_names(template: str) -> frozenset[str]:
    """Quy placeholder template về tên slot thật được ghi vào dataset."""
    placeholder_to_slot = {
        "weather_time": "datetime",
        "datetime": "datetime",
        "reminder_text": "reminder_text",
        "location": "location",
        "song": "song",
        "artist": "artist",
        "contact_name": "contact_name",
        "phone_number": "phone_number",
    }
    return frozenset(
        placeholder_to_slot[name]
        for name in TEMPLATE_SLOT_PATTERN.findall(template)
        if name in placeholder_to_slot
    )


def _covering_subsets(
    indexes: list[int], signatures: dict[int, frozenset[str]], required: frozenset[str]
) -> list[tuple[int, ...]]:
    """Liệt kê các tập family có đủ mọi slot, ưu tiên tập nhỏ để giữ nhiều family train."""
    candidates: list[tuple[int, ...]] = []
    for size in range(1, len(indexes) + 1):
        for subset in combinations(indexes, size):
            covered = frozenset().union(*(signatures[index] for index in subset))
            if required.issubset(covered):
                candidates.append(subset)
    return candidates


def split_template_indexes(templates: list[str], seed: int, scope: str) -> dict[int, str]:
    """Chia family theo slot coverage để validation/test không bị thiếu loại slot."""
    if len(templates) < 3:
        raise ValueError(f"Cần ít nhất 3 template cho {scope}, hiện có {len(templates)}")
    indexes = list(range(len(templates)))
    signatures = {index: template_slot_names(template) for index, template in enumerate(templates)}
    required = frozenset().union(*signatures.values())
    covers = _covering_subsets(indexes, signatures, required)
    covers.sort(
        key=lambda subset: (
            len(subset),
            stable_order_key(seed, f"{scope}:{','.join(map(str, subset))}"),
        )
    )

    choices: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    all_indexes = set(indexes)
    for validation_indexes in covers:
        validation_set = set(validation_indexes)
        for test_indexes in covers:
            test_set = set(test_indexes)
            if validation_set & test_set:
                continue
            train_indexes = all_indexes - validation_set - test_set
            if train_indexes:
                choices.append((validation_indexes, test_indexes))
    if not choices:
        raise ValueError(
            f"Template {scope} chưa đủ family để phủ mọi slot độc lập ở train/validation/test"
        )
    choices.sort(
        key=lambda pair: (
            len(pair[0]) + len(pair[1]),
            stable_order_key(seed, f"{scope}:validation:{pair[0]}:test:{pair[1]}"),
        )
    )
    validation_indexes, test_indexes = choices[0]
    assignment = {index: "train" for index in indexes}
    assignment.update({index: "validation" for index in validation_indexes})
    assignment.update({index: "test" for index in test_indexes})
    return assignment


def generate_regional_samples(
    templates_path: Path,
    slot_values_path: Path,
    seed: int,
) -> dict[str, list[DatasetSample]]:
    """Sinh sample thiếu theo intent; template family và variant không được cắt qua split."""
    templates = load_yaml(templates_path)["regional_templates"]
    slot_values = load_yaml(slot_values_path)
    rng = random.Random(seed)
    partitions: dict[str, list[DatasetSample]] = defaultdict(list)
    global_fingerprints: set[str] = set()

    for region in (Region.NORTH, Region.CENTRAL, Region.SOUTH):
        for intent in (
            Intent.SET_REMINDER,
            Intent.SET_ALARM,
            Intent.ASK_WEATHER,
            Intent.PLAY_MUSIC,
            Intent.CALL_CONTACT,
        ):
            intent_templates = templates[region.value][intent.value]
            # Template cùng vị trí ở ba vùng là một family song song; chúng phải ở cùng split.
            template_splits = split_template_indexes(intent_templates, seed, intent.value)
            group_target = REGIONAL_GROUP_TARGETS[intent]
            split_targets = {
                "train": round(group_target * 0.70),
                "validation": round(group_target * 0.15),
            }
            split_targets["test"] = group_target - sum(split_targets.values())
            group_index = 0

            for partition in ("train", "validation", "test"):
                allowed_indexes = [
                    index for index, split in template_splits.items() if split == partition
                ]
                generated = 0
                attempts = 0
                while generated < split_targets[partition]:
                    attempts += 1
                    if attempts > split_targets[partition] * 500:
                        raise RuntimeError(
                            f"Không đủ tổ hợp duy nhất cho "
                            f"{region.value}/{intent.value}/{partition}"
                        )
                    template_index = rng.choice(allowed_indexes)
                    template = intent_templates[template_index]
                    values = sample_values(rng, slot_values)
                    text = normalize_text(template.format(**values))
                    fingerprint = text_fingerprint(text)
                    if fingerprint in global_fingerprints:
                        continue
                    global_fingerprints.add(fingerprint)
                    slots: dict[str, str | list[str]] = slots_for_template(intent, template, values)
                    group_id = f"regional_{region.value}_{intent.value}_{group_index:03d}"
                    group_index += 1
                    generated += 1
                    source_ref = f"template:{region.value}:{intent.value}:{template_index:02d}"
                    partitions[partition].append(
                        DatasetSample(
                            id=f"{group_id}_regional",
                            group_id=group_id,
                            text=text,
                            region=region,
                            intent=intent,
                            slots=slots,
                            source=DataSource.SYNTHETIC,
                            source_ref=source_ref,
                            variant_type=VariantType.REGIONAL,
                            annotation_quality=AnnotationQuality.TEMPLATE_GENERATED,
                        )
                    )
                    partitions[partition].append(
                        DatasetSample(
                            id=f"{group_id}_no_diacritics",
                            group_id=group_id,
                            text=strip_diacritics(text),
                            region=region,
                            intent=intent,
                            slots=slots,
                            source=DataSource.SYNTHETIC,
                            source_ref=source_ref,
                            variant_type=VariantType.NO_DIACRITICS,
                            annotation_quality=AnnotationQuality.TEMPLATE_GENERATED,
                        )
                    )
    return partitions


def load_reviewed_seed(path: Path, _seed: int) -> dict[str, list[DatasetSample]]:
    """Đọc seed đã review; giữ toàn bộ ở train vì tập nhỏ không đủ split độc lập."""
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                samples.append(DatasetSample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return {"train": samples}


def merge_partitions(*sources: dict[str, list[DatasetSample]]) -> dict[str, list[DatasetSample]]:
    """Gộp các nguồn nhưng vẫn giữ split do từng nguồn quyết định."""
    merged: dict[str, list[DatasetSample]] = defaultdict(list)
    for source in sources:
        for partition, samples in source.items():
            merged[partition].extend(samples)
    return merged


def remove_cross_split_near_samples(
    partitions: dict[str, list[DatasetSample]],
    seed: int,
    normalizer: VietnameseNormalizer,
) -> dict[str, list[DatasetSample]]:
    """Loại group gần trùng trên đúng biểu diễn sau normalize mà model nhìn thấy."""
    result: dict[str, list[DatasetSample]] = defaultdict(list)
    protected_templates: list[str] = []
    for partition in ("test", "validation", "train"):
        groups: dict[str, list[DatasetSample]] = defaultdict(list)
        for sample in partitions.get(partition, []):
            groups[sample.group_id].append(sample)
        current_templates: list[str] = []
        for group_id in sorted(groups, key=lambda value: stable_order_key(seed, value)):
            group_samples = groups[group_id]
            representative = normalized_masked_sample_text(group_samples[0], normalizer)
            if any(are_near_similar(representative, template) for template in protected_templates):
                continue
            result[partition].extend(group_samples)
            if representative not in current_templates:
                current_templates.append(representative)
        protected_templates.extend(current_templates)
    return result


def remove_same_split_normalized_duplicates(
    partitions: dict[str, list[DatasetSample]],
    seed: int,
    normalizer: VietnameseNormalizer,
) -> dict[str, list[DatasetSample]]:
    """Giữ một sample cho mỗi câu sau normalize trong cùng split để metric không bị lặp."""
    quality_priority = {
        AnnotationQuality.REVIEWED: 0,
        AnnotationQuality.AUTO_MAPPED: 1,
        AnnotationQuality.TEMPLATE_GENERATED: 2,
    }
    result: dict[str, list[DatasetSample]] = defaultdict(list)
    for partition in ("train", "validation", "test"):
        groups: dict[str, list[DatasetSample]] = defaultdict(list)
        for sample in partitions.get(partition, []):
            normalized_text = normalizer.normalize(sample.text, sample.region).normalized_text
            groups[normalized_text].append(sample)

        # Các câu chỉ xuất hiện một lần được cố định trước. Với nhóm trùng, ưu tiên
        # annotation tốt rồi chọn ô region/intent đang ít mẫu để không làm méo test.
        retained_counts = Counter(
            (samples[0].region, samples[0].intent)
            for samples in groups.values()
            if len(samples) == 1
        )
        result[partition].extend(samples[0] for samples in groups.values() if len(samples) == 1)
        for normalized_text in sorted(
            (text for text, samples in groups.items() if len(samples) > 1),
            key=lambda value: stable_order_key(seed, value),
        ):
            candidates = groups[normalized_text]
            best_quality = min(quality_priority[sample.annotation_quality] for sample in candidates)
            candidates = [
                sample
                for sample in candidates
                if quality_priority[sample.annotation_quality] == best_quality
            ]
            selected = min(
                candidates,
                key=lambda sample: (
                    retained_counts[(sample.region, sample.intent)],
                    stable_order_key(seed, sample.id),
                    sample.id,
                ),
            )
            result[partition].append(selected)
            retained_counts[(selected.region, selected.intent)] += 1
    return result


def output_name(partition: str, sample: DatasetSample) -> str:
    """Tách test theo region; train và validation giữ chung để dễ sử dụng."""
    if partition == "test":
        return f"test_{sample.region.value}.jsonl"
    return f"{partition}.jsonl"


def sha256_file(path: Path) -> str:
    """Tính checksum nguồn để lần build sau biết đang dùng đúng phiên bản."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    partitions: dict[str, list[DatasetSample]],
    output_dir: Path,
    input_paths: dict[str, Path],
    seed: int,
    filtering_report: dict[str, Any],
) -> dict[str, Any]:
    """Ghi JSONL và manifest thống kê theo cách deterministic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    known_outputs = [
        "train.jsonl",
        "validation.jsonl",
        "test_standard.jsonl",
        "test_north.jsonl",
        "test_central.jsonl",
        "test_south.jsonl",
    ]
    buckets: dict[str, list[DatasetSample]] = defaultdict(list)
    for partition, samples in partitions.items():
        for sample in samples:
            buckets[output_name(partition, sample)].append(sample)

    for filename in known_outputs:
        path = output_dir / filename
        samples = sorted(buckets.get(filename, []), key=lambda sample: sample.id)
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for sample in samples:
                payload = sample.model_dump(mode="json")
                file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    output_hashes = {filename: sha256_file(output_dir / filename) for filename in known_outputs}
    test_digest = hashlib.sha256()
    for filename in sorted(name for name in known_outputs if name.startswith("test_")):
        test_digest.update(filename.encode("utf-8"))
        test_digest.update((output_dir / filename).read_bytes())

    all_samples = [sample for samples in buckets.values() for sample in samples]
    manifest = {
        "seed": seed,
        "split_policy": "slot_coverage_normalized_near_similarity_v3",
        "filtering": filtering_report,
        "inputs_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "files_sha256": output_hashes,
        "test_set_sha256": test_digest.hexdigest(),
        "total": len(all_samples),
        "files": {name: len(buckets.get(name, [])) for name in known_outputs},
        "intents": dict(sorted(Counter(sample.intent.value for sample in all_samples).items())),
        "regions": dict(sorted(Counter(sample.region.value for sample in all_samples).items())),
        "sources": dict(sorted(Counter(sample.source.value for sample in all_samples).items())),
        "annotation_quality": dict(
            sorted(Counter(sample.annotation_quality.value for sample in all_samples).items())
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def build_dataset(
    massive_path: Path,
    output_dir: Path,
    templates_path: Path,
    slot_values_path: Path,
    hard_cases_path: Path,
    regional_variants_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Điều phối build từ mọi nguồn; hard-case là input chính thức, không augment tay."""
    massive = load_massive_samples(massive_path, seed)
    regional = generate_regional_samples(templates_path, slot_values_path, seed)
    hard_cases = load_reviewed_seed(hard_cases_path, seed)
    normalizer = VietnameseNormalizer(regional_variants_path)
    unfiltered = merge_partitions(massive, regional, hard_cases)
    cross_split_filtered = remove_cross_split_near_samples(unfiltered, seed, normalizer)
    merged = remove_same_split_normalized_duplicates(cross_split_filtered, seed, normalizer)
    before_samples = [sample for values in unfiltered.values() for sample in values]
    cross_split_samples = [sample for values in cross_split_filtered.values() for sample in values]
    after_samples = [sample for values in merged.values() for sample in values]
    before_by_source = Counter(sample.source.value for sample in before_samples)
    after_by_source = Counter(sample.source.value for sample in after_samples)
    filtering_report = {
        "reason": "normalized_cross_split_near_and_same_split_exact_duplicates",
        "before": len(before_samples),
        "after": len(after_samples),
        "dropped": len(before_samples) - len(after_samples),
        "steps": {
            "cross_split_near_similarity": {
                "before": len(before_samples),
                "after": len(cross_split_samples),
                "dropped": len(before_samples) - len(cross_split_samples),
            },
            "same_split_exact_after_normalization": {
                "before": len(cross_split_samples),
                "after": len(after_samples),
                "dropped": len(cross_split_samples) - len(after_samples),
            },
        },
        "dropped_by_source": {
            source: before_by_source[source] - after_by_source[source]
            for source in sorted(before_by_source)
        },
    }
    input_paths = {
        "massive_vi_v1": massive_path,
        "regional_templates": templates_path,
        "slot_values": slot_values_path,
        "intent_hard_cases": hard_cases_path,
        "regional_variants": regional_variants_path,
    }
    return write_dataset(
        merged,
        output_dir,
        input_paths,
        seed,
        filtering_report,
    )
