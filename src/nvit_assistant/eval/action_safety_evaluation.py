"""Đánh giá false-action rate trên bộ câu OOD/phủ định development riêng."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nvit_assistant.nlu.action_gate import ActionGate
from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.schemas import Intent, ParseRequest, PreprocessedSample


class ActionSafetySample(BaseModel):
    """Một câu safety được review, không thuộc train/validation/test intent."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: str = Field(min_length=1)
    should_execute: bool
    expected_intent: Intent | None = None


def load_action_safety_challenge(path: Path) -> list[ActionSafetySample]:
    """Đọc JSONL và bắt lỗi kèm số dòng."""
    samples: list[ActionSafetySample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    samples.append(ActionSafetySample.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not samples:
        raise ValueError("action safety challenge trống")
    return samples


def evaluate_action_safety(
    pipeline: NLUPipeline, samples: list[ActionSafetySample]
) -> dict[str, Any]:
    """Đo FAR trên negative và action recall/intent trên positive."""
    negative_total = 0
    false_actions = 0
    positive_total = 0
    positive_actions = 0
    failures: list[dict[str, Any]] = []
    for sample in samples:
        result = pipeline.parse(ParseRequest(text=sample.text))
        executed = result.action is not None
        intent_matches = sample.expected_intent is None or result.intent is sample.expected_intent
        passed = executed == sample.should_execute and (not sample.should_execute or intent_matches)
        if sample.should_execute:
            positive_total += 1
            positive_actions += executed and intent_matches
        else:
            negative_total += 1
            false_actions += executed
        if not passed:
            failures.append(
                {
                    "id": sample.id,
                    "category": sample.category,
                    "text": sample.text,
                    "should_execute": sample.should_execute,
                    "expected_intent": (
                        sample.expected_intent.value if sample.expected_intent else None
                    ),
                    "predicted_intent": result.intent.value,
                    "confidence": result.confidence,
                    "action": result.action.model_dump(mode="json") if result.action else None,
                    "matched_features": result.matched_features,
                }
            )
    return {
        "total": len(samples),
        "negative_total": negative_total,
        "false_actions": false_actions,
        "false_action_rate": false_actions / negative_total if negative_total else 0.0,
        "positive_total": positive_total,
        "positive_actions": positive_actions,
        "positive_action_recall": positive_actions / positive_total if positive_total else 0.0,
        "failures": failures,
    }


def evaluate_action_gate_coverage(
    gate: ActionGate,
    extractor: RegexSlotExtractor,
    samples: list[PreprocessedSample],
) -> dict[str, Any]:
    """Đo recall của gate trên command validation bằng intent thật, tách khỏi lỗi classifier."""
    totals: dict[str, int] = {}
    allowed_counts: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    for sample in samples:
        intent = sample.original.intent
        intent_name = intent.value
        totals[intent_name] = totals.get(intent_name, 0) + 1
        slots = extractor.extract(sample.normalized_text, intent).slots
        decision = gate.check(sample.normalized_text, intent, slots)
        if decision.allowed:
            allowed_counts[intent_name] = allowed_counts.get(intent_name, 0) + 1
        else:
            failures.append(
                {
                    "id": sample.original.id,
                    "intent": intent_name,
                    "text": sample.normalized_text,
                    "reason": decision.reason,
                }
            )
    total = len(samples)
    allowed = sum(allowed_counts.values())
    return {
        "total": total,
        "allowed": allowed,
        "coverage": allowed / total if total else 0.0,
        "per_intent": {
            intent: {
                "total": count,
                "allowed": allowed_counts.get(intent, 0),
                "coverage": allowed_counts.get(intent, 0) / count if count else 0.0,
            }
            for intent, count in sorted(totals.items())
        },
        "failure_count": len(failures),
        "failures": failures,
    }
