"""Đánh giá normalizer bằng benchmark command-domain có expected text rõ ràng."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.schemas import Region


@dataclass(frozen=True)
class NormalizationChallenge:
    """Một câu benchmark độc lập với dataset intent để đo normalizer."""

    id: str
    region: Region
    category: str
    input_text: str
    expected_text: str


def load_normalization_challenge(path: Path) -> list[NormalizationChallenge]:
    """Đọc JSONL benchmark và báo file/dòng cụ thể nếu contract không hợp lệ."""
    challenges: list[NormalizationChallenge] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("mỗi dòng phải là object JSON")
                challenges.append(
                    NormalizationChallenge(
                        id=str(value["id"]),
                        region=Region(str(value["region"])),
                        category=str(value["category"]),
                        input_text=str(value["input"]),
                        expected_text=str(value["expected"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not challenges:
        raise ValueError(f"benchmark trống: {path}")
    if len({challenge.id for challenge in challenges}) != len(challenges):
        raise ValueError(f"benchmark có ID trùng: {path}")
    return challenges


def evaluate_normalizer(
    normalizer: VietnameseNormalizer, challenges: list[NormalizationChallenge]
) -> dict[str, Any]:
    """Trả text/region accuracy và failure đủ thông tin để bổ sung rule có kiểm soát."""
    text_correct = 0
    region_correct = 0
    failures: list[dict[str, str]] = []
    for challenge in challenges:
        result = normalizer.normalize(challenge.input_text)
        text_matches = result.normalized_text == challenge.expected_text
        region_matches = result.region is challenge.region
        text_correct += int(text_matches)
        region_correct += int(region_matches)
        if not text_matches or not region_matches:
            failures.append(
                {
                    "id": challenge.id,
                    "category": challenge.category,
                    "input": challenge.input_text,
                    "expected_text": challenge.expected_text,
                    "actual_text": result.normalized_text,
                    "expected_region": challenge.region.value,
                    "actual_region": result.region.value,
                }
            )
    total = len(challenges)
    return {
        "total": total,
        "text_accuracy": text_correct / total,
        "region_accuracy": region_correct / total,
        "failures": failures,
    }
