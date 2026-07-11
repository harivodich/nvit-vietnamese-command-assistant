"""Chuẩn hóa câu lệnh tiếng Việt trước các bước NLU phía sau."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nvit_assistant.schemas import Region


QUESTION_CUE_PATTERN = re.compile(
    r"\?|\b(không|hông|hổng|hem|sao|răng|mấy|bao nhiêu|gì|đâu|chưa)\b"
)
REQUEST_CUE_PATTERN = re.compile(
    r"\b(gọi|liên lạc|mở|bật|phát|nghe|nhắc|đặt|cài|hẹn|xem|coi|giúp|dùm|giùm|hãy|đừng)\b"
)


@dataclass(frozen=True)
class NormalizationResult:
    """Kết quả chuẩn hóa kèm dấu vết rule đã áp dụng để dễ kiểm tra."""

    original_text: str
    normalized_text: str
    region: Region
    matched_variants: tuple[str, ...]


def normalize_surface(text: str) -> str:
    """Chuẩn Unicode, chữ thường và khoảng trắng nhưng giữ nguyên dấu câu cần thiết."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def load_variant_config(
    path: Path,
) -> tuple[dict[Region, dict[str, str]], dict[str, str], dict[Region, dict[str, dict[str, str]]]]:
    """Đọc config biến thể thường, lỗi STT và tiểu từ cần xét theo ngữ cảnh."""
    with path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML root phải là mapping: {path}")

    raw_regions = config.get("variants")
    raw_stt = config.get("stt_replacements")
    raw_particles = config.get("contextual_particles")
    if (
        not isinstance(raw_regions, dict)
        or not isinstance(raw_stt, dict)
        or not isinstance(raw_particles, dict)
    ):
        raise ValueError("config phải có variants, stt_replacements và contextual_particles dạng mapping")

    regional_variants: dict[Region, dict[str, str]] = {}
    for region in (Region.NORTH, Region.CENTRAL, Region.SOUTH):
        raw_values = raw_regions.get(region.value)
        if not isinstance(raw_values, dict):
            raise ValueError(f"thiếu mapping biến thể cho vùng {region.value}")
        regional_variants[region] = {
            normalize_surface(str(source)): normalize_surface(str(target))
            for source, target in raw_values.items()
        }
    stt_replacements = {
        normalize_surface(str(source)): normalize_surface(str(target))
        for source, target in raw_stt.items()
    }
    contextual_particles: dict[Region, dict[str, dict[str, str]]] = {}
    for region in (Region.NORTH, Region.CENTRAL, Region.SOUTH):
        raw_values = raw_particles.get(region.value, {})
        if not isinstance(raw_values, dict):
            raise ValueError(f"contextual_particles.{region.value} phải là mapping")
        contextual_particles[region] = {}
        for source, raw_targets in raw_values.items():
            if not isinstance(raw_targets, dict):
                raise ValueError(f"tiểu từ {source!r} phải có mapping theo ngữ cảnh")
            question = raw_targets.get("question")
            request = raw_targets.get("request")
            if not isinstance(question, str) or not isinstance(request, str):
                raise ValueError(f"tiểu từ {source!r} cần question và request dạng chuỗi")
            contextual_particles[region][normalize_surface(str(source))] = {
                "question": normalize_surface(question),
                "request": normalize_surface(request),
            }
    return regional_variants, stt_replacements, contextual_particles


def replace_phrases(text: str, replacements: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    """Thay cụm nguyên từ, ưu tiên cụm dài để `bữa ni` không bị thay `ni` trước."""
    matched: list[str] = []
    result = text
    for source, target in sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0])):
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)")
        result, count = pattern.subn(target, result)
        if count:
            matched.extend([f"{source} -> {target}"] * count)
    return result, tuple(matched)


def contextual_particle_target(text: str, source: str, targets: dict[str, str]) -> str | None:
    """Chọn dạng chuẩn cho tiểu từ ở cuối câu; không đủ ngữ cảnh thì giữ nguyên."""
    at_sentence_end = re.search(rf"(?<!\w){re.escape(source)}[?!.,]*\s*$", text)
    if at_sentence_end is None:
        return None
    if QUESTION_CUE_PATTERN.search(text):
        return targets["question"]
    if REQUEST_CUE_PATTERN.search(text[: at_sentence_end.start()]):
        return targets["request"]
    return None


class VietnameseNormalizer:
    """Normalizer dùng rule minh bạch, không tự suy đoán dấu cho từ không có trong config."""

    def __init__(self, config_path: Path) -> None:
        """Nạp config một lần để mỗi câu lệnh không phải đọc lại file YAML."""
        (
            self.regional_variants,
            self.stt_replacements,
            self.contextual_particles,
        ) = load_variant_config(config_path)

    def normalize(self, text: str, region_hint: Region | None = None) -> NormalizationResult:
        """Chuẩn hóa text và chỉ suy luận region khi các rule cho tín hiệu rõ ràng."""
        original_text = text
        normalized_text = normalize_surface(text)
        if not normalized_text:
            raise ValueError("text không được chỉ gồm khoảng trắng")

        normalized_text, stt_matches = replace_phrases(normalized_text, self.stt_replacements)
        active_regions = (
            (region_hint,)
            if region_hint in {Region.NORTH, Region.CENTRAL, Region.SOUTH}
            else (Region.NORTH, Region.CENTRAL, Region.SOUTH)
        )

        region_scores = {region: 0 for region in active_regions}
        for region in active_regions:
            _, matches = replace_phrases(normalized_text, self.regional_variants[region])
            region_scores[region] += len(matches)
            for source, targets in self.contextual_particles[region].items():
                if contextual_particle_target(normalized_text, source, targets) is not None:
                    region_scores[region] += 1

        combined_variants: dict[str, str] = {}
        for region in active_regions:
            for source, target in self.regional_variants[region].items():
                previous_target = combined_variants.get(source)
                if previous_target is not None and previous_target != target:
                    raise ValueError(f"biến thể {source!r} có hai dạng chuẩn khác nhau")
                combined_variants[source] = target
        normalized_text, regional_matches = replace_phrases(normalized_text, combined_variants)

        contextual_replacements: dict[str, str] = {}
        for region in active_regions:
            for source, targets in self.contextual_particles[region].items():
                contextual_target = contextual_particle_target(normalized_text, source, targets)
                if contextual_target is None:
                    continue
                previous_target = contextual_replacements.get(source)
                if previous_target is not None and previous_target != contextual_target:
                    raise ValueError(f"tiểu từ {source!r} có hai cách chuẩn hóa xung đột")
                contextual_replacements[source] = contextual_target
        normalized_text, contextual_matches = replace_phrases(
            normalized_text, contextual_replacements
        )

        detected_region: Region
        if region_hint in {Region.NORTH, Region.CENTRAL, Region.SOUTH}:
            detected_region = region_hint
        else:
            highest_score = max(region_scores.values(), default=0)
            leading_regions = [
                region for region, score in region_scores.items() if score == highest_score and score > 0
            ]
            if highest_score == 0:
                detected_region = Region.STANDARD
            elif len(leading_regions) == 1:
                detected_region = leading_regions[0]
            else:
                detected_region = Region.UNKNOWN

        return NormalizationResult(
            original_text=original_text,
            normalized_text=normalized_text,
            region=detected_region,
            matched_variants=stt_matches + regional_matches + contextual_matches,
        )
