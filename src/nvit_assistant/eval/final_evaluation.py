"""Đánh giá cuối toàn pipeline trên test holdout đã khóa."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from nvit_assistant.nlu.intent_classifier import evaluate_probability_metrics
from nvit_assistant.nlu.normalizer import normalize_surface
from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.nlu.preprocessing import preprocess_sample
from nvit_assistant.schemas import DatasetSample, Intent, ParseRequest, ParseResult, Region


INTENT_LABELS = [intent.value for intent in Intent if intent is not Intent.UNKNOWN]
REGION_LABELS = [region.value for region in Region if region is not Region.UNKNOWN]


@dataclass(frozen=True)
class SampleEvaluation:
    """Kết quả cần thiết của một test sample để tính mọi lát cắt sau cùng."""

    sample: DatasetSample
    raw_intent: str
    raw_confidence: float
    probabilities: dict[str, float]
    result: ParseResult
    expected_slots: dict[str, Any]
    oracle_slots: dict[str, Any]
    latency_ms: float


def _slot_pairs(slots: dict[str, Any]) -> set[tuple[str, str]]:
    """Đưa slot đơn/list về tập cặp đã chuẩn hóa để so sánh không phụ thuộc thứ tự."""
    pairs: set[tuple[str, str]] = set()
    for slot_name, raw_value in slots.items():
        values: Iterable[Any] = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if isinstance(value, str):
                pairs.add((slot_name, normalize_surface(value)))
    return pairs


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    expected: list[str],
    predicted: list[str],
    report_labels: list[str],
    matrix_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Tính cùng bộ accuracy/F1/report cho raw model, runtime intent và region."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("expected và predicted phải cùng độ dài khác 0")
    labels = matrix_labels or report_labels
    report: dict[str, Any] = classification_report(
        expected,
        predicted,
        labels=report_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=report_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                expected,
                predicted,
                labels=report_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "labels": labels,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "per_label": {label: report[label] for label in report_labels},
    }


def slot_metrics(
    expected_slots: list[dict[str, Any]], predicted_slots: list[dict[str, Any]]
) -> dict[str, Any]:
    """Đo exact match và micro/per-slot precision, recall, F1."""
    if not expected_slots or len(expected_slots) != len(predicted_slots):
        raise ValueError("hai danh sách slot phải cùng độ dài khác 0")
    totals: Counter[str] = Counter()
    per_slot: defaultdict[str, Counter[str]] = defaultdict(Counter)
    exact = 0
    for expected, predicted in zip(expected_slots, predicted_slots):
        expected_pairs = _slot_pairs(expected)
        predicted_pairs = _slot_pairs(predicted)
        exact += expected_pairs == predicted_pairs
        for slot_name, _ in expected_pairs & predicted_pairs:
            totals["tp"] += 1
            per_slot[slot_name]["tp"] += 1
        for slot_name, _ in predicted_pairs - expected_pairs:
            totals["fp"] += 1
            per_slot[slot_name]["fp"] += 1
        for slot_name, _ in expected_pairs - predicted_pairs:
            totals["fn"] += 1
            per_slot[slot_name]["fn"] += 1

    def scores(counts: Counter[str]) -> dict[str, int | float]:
        precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
        return {
            "support": counts["tp"] + counts["fn"],
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
        }

    return {
        "exact_match": exact / len(expected_slots),
        "micro": scores(totals),
        "per_slot": {
            slot_name: scores(counts) for slot_name, counts in sorted(per_slot.items())
        },
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    """Báo cả first request và steady-state để không che chi phí lazy initialization."""
    if not values:
        raise ValueError("không có latency sample")
    steady = values[1:] or values
    total_seconds = sum(values) / 1000.0
    return {
        "first_request_ms": values[0],
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
        "steady_state_median_ms": float(np.median(steady)),
        "sequential_throughput_commands_per_second": len(values) / total_seconds,
    }


def collect_sample_evaluations(
    pipeline: NLUPipeline, samples: list[DatasetSample]
) -> list[SampleEvaluation]:
    """Chạy mỗi sample đúng một lần qua full pipeline, không truyền region label thật."""
    records: list[SampleEvaluation] = []
    for sample in samples:
        started = perf_counter()
        result = pipeline.parse(ParseRequest(text=sample.text))
        latency_ms = (perf_counter() - started) * 1000.0
        raw_prediction = pipeline.intent_classifier.predict(result.normalized_text)
        prepared = preprocess_sample(sample, pipeline.normalizer)
        oracle_slots = pipeline.slot_extractor.extract(
            result.normalized_text, sample.intent
        ).slots
        records.append(
            SampleEvaluation(
                sample=sample,
                raw_intent=raw_prediction.intent.value,
                raw_confidence=raw_prediction.confidence,
                probabilities=raw_prediction.probabilities,
                result=result,
                expected_slots=prepared.normalized_slots,
                oracle_slots=oracle_slots,
                latency_ms=latency_ms,
            )
        )
    return records


def summarize_records(records: list[SampleEvaluation]) -> dict[str, Any]:
    """Tổng hợp intent, slot, frame, action, region, latency và failure analysis."""
    if not records:
        raise ValueError("final evaluation cần ít nhất một record")
    expected_intents = [record.sample.intent.value for record in records]
    raw_intents = [record.raw_intent for record in records]
    runtime_intents = [record.result.intent.value for record in records]
    expected_slots = [record.expected_slots for record in records]
    oracle_slots = [record.oracle_slots for record in records]
    runtime_slots = [record.result.slots for record in records]
    expected_regions = [record.sample.region.value for record in records]
    predicted_regions = [record.result.region.value for record in records]

    raw_metrics = classification_metrics(expected_intents, raw_intents, INTENT_LABELS)
    probabilities = np.asarray(
        [[record.probabilities[label] for label in INTENT_LABELS] for record in records]
    )
    raw_metrics["probability_metrics"] = evaluate_probability_metrics(
        expected_intents, raw_intents, probabilities, INTENT_LABELS
    )
    runtime_metrics = classification_metrics(
        expected_intents,
        runtime_intents,
        INTENT_LABELS,
        [*INTENT_LABELS, Intent.UNKNOWN.value],
    )
    accepted = [intent != Intent.UNKNOWN.value for intent in runtime_intents]
    accepted_correct = sum(
        is_accepted and expected == predicted
        for is_accepted, expected, predicted in zip(
            accepted, expected_intents, runtime_intents
        )
    )
    runtime_metrics["coverage"] = sum(accepted) / len(records)
    runtime_metrics["selective_accuracy"] = _ratio(accepted_correct, sum(accepted))

    oracle_slot_metrics = slot_metrics(expected_slots, oracle_slots)
    end_to_end_slot_metrics = slot_metrics(expected_slots, runtime_slots)
    frame_exact_flags: list[bool] = []
    action_flags: list[bool] = []
    full_success_flags: list[bool] = []
    failure_rows: list[dict[str, Any]] = []
    breakdown_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    action_statuses: Counter[str] = Counter()

    for record in records:
        intent_exact = record.result.intent is record.sample.intent
        slot_exact = _slot_pairs(record.result.slots) == _slot_pairs(record.expected_slots)
        oracle_slot_exact = _slot_pairs(record.oracle_slots) == _slot_pairs(
            record.expected_slots
        )
        frame_exact = intent_exact and slot_exact
        action_present = record.result.action is not None
        full_success = frame_exact and action_present
        frame_exact_flags.append(frame_exact)
        action_flags.append(action_present)
        full_success_flags.append(full_success)
        if record.result.action is not None:
            action_statuses[record.result.action.status.value] += 1

        dimensions = {
            "region": record.sample.region.value,
            "intent": record.sample.intent.value,
            "source": record.sample.source.value,
            "variant_type": record.sample.variant_type.value,
            "annotation_quality": record.sample.annotation_quality.value,
        }
        for dimension, value in dimensions.items():
            counts = breakdown_counts[(dimension, value)]
            counts["total"] += 1
            counts["intent_exact"] += intent_exact
            counts["slot_exact"] += slot_exact
            counts["oracle_slot_exact"] += oracle_slot_exact
            counts["frame_exact"] += frame_exact
            counts["action_present"] += action_present
            counts["full_success"] += full_success

        if not full_success:
            reasons = []
            if not intent_exact:
                reasons.append("intent")
            if not slot_exact:
                reasons.append("slots")
            if not action_present:
                reasons.append("action")
            failure_rows.append(
                {
                    "id": record.sample.id,
                    "text": record.sample.text,
                    "region": record.sample.region.value,
                    "source": record.sample.source.value,
                    "variant_type": record.sample.variant_type.value,
                    "reasons": reasons,
                    "expected_intent": record.sample.intent.value,
                    "raw_model_intent": record.raw_intent,
                    "runtime_intent": record.result.intent.value,
                    "confidence": record.result.confidence,
                    "expected_slots": record.expected_slots,
                    "predicted_slots": record.result.slots,
                    "action": (
                        record.result.action.model_dump(mode="json")
                        if record.result.action is not None
                        else None
                    ),
                    "matched_features": record.result.matched_features,
                }
            )

    breakdown: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dimension, value), counts in sorted(breakdown_counts.items()):
        total = counts["total"]
        breakdown[dimension][value] = {
            "total": total,
            "intent_accuracy": counts["intent_exact"] / total,
            "oracle_slot_exact_match": counts["oracle_slot_exact"] / total,
            "end_to_end_slot_exact_match": counts["slot_exact"] / total,
            "semantic_frame_exact_match": counts["frame_exact"] / total,
            "action_execution_rate": counts["action_present"] / total,
            "full_command_success": counts["full_success"] / total,
        }

    return {
        "total_samples": len(records),
        "raw_model_intent": raw_metrics,
        "runtime_intent": runtime_metrics,
        "region_inference": classification_metrics(
            expected_regions,
            predicted_regions,
            REGION_LABELS,
            [*REGION_LABELS, Region.UNKNOWN.value],
        ),
        "oracle_slots": oracle_slot_metrics,
        "end_to_end_slots": end_to_end_slot_metrics,
        "semantic_frame_exact_match": sum(frame_exact_flags) / len(records),
        "action_execution_rate": sum(action_flags) / len(records),
        "full_command_success": sum(full_success_flags) / len(records),
        "action_statuses": dict(sorted(action_statuses.items())),
        "breakdown": dict(breakdown),
        "latency": _latency_summary([record.latency_ms for record in records]),
        "failure_count": len(failure_rows),
        "failures": failure_rows,
    }


def evaluate_final_pipeline(
    pipeline: NLUPipeline, samples: list[DatasetSample]
) -> dict[str, Any]:
    """Public entrypoint: thu thập dự đoán rồi tổng hợp báo cáo final."""
    return summarize_records(collect_sample_evaluations(pipeline, samples))
