"""Đánh giá slot extractor độc lập với lỗi của intent classifier."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from nvit_assistant.nlu.normalizer import normalize_surface
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import PreprocessedSample


def _slot_pairs(slots: dict[str, Any]) -> set[tuple[str, str]]:
    """Trải dict slot thành tập cặp để hỗ trợ cả giá trị đơn và list."""
    pairs: set[tuple[str, str]] = set()
    for slot_name, raw_value in slots.items():
        values: Iterable[Any] = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if isinstance(value, str):
                pairs.add((slot_name, normalize_surface(value)))
    return pairs


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Tránh chia cho không khi một slot không xuất hiện trong tập đánh giá."""
    return numerator / denominator if denominator else 0.0


def _metrics(counts: Counter[str]) -> dict[str, float | int]:
    """Tính precision, recall và F1 từ tổng true/false positive/negative."""
    true_positive = counts["tp"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "support": true_positive + false_negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
    }


def load_preprocessed_samples(path: Path) -> list[PreprocessedSample]:
    """Đọc JSONL preprocess và kiểm tra từng dòng bằng contract Pydantic."""
    samples: list[PreprocessedSample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    samples.append(PreprocessedSample.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return samples


def evaluate_slot_extractor(
    extractor: RegexSlotExtractor, samples: list[PreprocessedSample]
) -> dict[str, Any]:
    """Dùng intent thật để metric chỉ phản ánh chất lượng trích xuất slot."""
    totals: Counter[str] = Counter()
    per_slot: defaultdict[str, Counter[str]] = defaultdict(Counter)
    exact_matches = 0
    failures: list[dict[str, Any]] = []
    breakdown_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def update_breakdown(
        dimension: str,
        value: str,
        true_positive_count: int,
        false_positive_count: int,
        false_negative_count: int,
        exact: bool,
    ) -> None:
        """Cộng metric cho một lát cắt mà không nhân bản logic tính slot."""
        counts = breakdown_counts[(dimension, value)]
        counts["samples"] += 1
        counts["exact"] += int(exact)
        counts["tp"] += true_positive_count
        counts["fp"] += false_positive_count
        counts["fn"] += false_negative_count

    for sample in samples:
        expected = _slot_pairs(sample.normalized_slots)
        result = extractor.extract(sample.normalized_text, sample.original.intent)
        predicted = _slot_pairs(result.slots)
        true_positive = expected & predicted
        false_positive = predicted - expected
        false_negative = expected - predicted

        for slot_name, _ in true_positive:
            totals["tp"] += 1
            per_slot[slot_name]["tp"] += 1
        for slot_name, _ in false_positive:
            totals["fp"] += 1
            per_slot[slot_name]["fp"] += 1
        for slot_name, _ in false_negative:
            totals["fn"] += 1
            per_slot[slot_name]["fn"] += 1

        exact = expected == predicted
        if exact:
            exact_matches += 1
        else:
            failures.append(
                {
                    "id": sample.original.id,
                    "intent": sample.original.intent.value,
                    "text": sample.normalized_text,
                    "expected": dict(sorted(sample.normalized_slots.items())),
                    "predicted": dict(sorted(result.slots.items())),
                }
            )

        dimensions = {
            "source": sample.original.source.value,
            "annotation_quality": sample.original.annotation_quality.value,
            "intent": sample.original.intent.value,
            "region": sample.original.region.value,
        }
        for dimension, value in dimensions.items():
            update_breakdown(
                dimension,
                value,
                len(true_positive),
                len(false_positive),
                len(false_negative),
                exact,
            )

    breakdown: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dimension, value), counts in sorted(breakdown_counts.items()):
        breakdown[dimension][value] = {
            "total_samples": counts["samples"],
            "slot_exact_match": _safe_ratio(counts["exact"], counts["samples"]),
            "micro": _metrics(counts),
        }

    return {
        "total_samples": len(samples),
        "slot_exact_match": _safe_ratio(exact_matches, len(samples)),
        "micro": _metrics(totals),
        "per_slot": {
            slot_name: _metrics(counts) for slot_name, counts in sorted(per_slot.items())
        },
        "breakdown": dict(breakdown),
        "failure_count": len(failures),
        "failures": failures,
    }
