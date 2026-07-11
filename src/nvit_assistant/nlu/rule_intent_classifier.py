"""Baseline intent classifier dựa trên rule để giải thích và đối chiếu model trainable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nvit_assistant.nlu.intent_classifier import IntentPrediction
from nvit_assistant.schemas import Intent, PreprocessedSample


@dataclass(frozen=True)
class IntentRule:
    """Rule regex cho một intent, lấy từ config thay vì hard-code trong logic."""

    intent: Intent
    patterns: tuple[re.Pattern[str], ...]


class RuleIntentClassifier:
    """Chấm điểm intent theo số pattern match; tie hoặc không match trả unknown an toàn."""

    def __init__(self, rules: list[IntentRule]) -> None:
        self.rules = rules

    def predict(self, text: str) -> IntentPrediction:
        """Trả intent rule-based và confidence chỉ biểu thị tỷ trọng rule, không phải xác suất ML."""
        scores = {rule.intent: sum(bool(pattern.search(text)) for pattern in rule.patterns) for rule in self.rules}
        highest_score = max(scores.values(), default=0)
        winners = [intent for intent, score in scores.items() if score == highest_score and score > 0]
        if len(winners) != 1:
            return IntentPrediction(intent=Intent.UNKNOWN, confidence=0.0, probabilities={})
        total_score = sum(scores.values())
        return IntentPrediction(
            intent=winners[0],
            confidence=highest_score / total_score,
            probabilities={intent.value: score / total_score for intent, score in scores.items() if score},
        )


def load_rule_classifier(path: Path) -> RuleIntentClassifier:
    """Đọc pattern intent từ YAML và kiểm tra mỗi intent có ít nhất một regex hợp lệ."""
    with path.open("r", encoding="utf-8") as file:
        raw_config: Any = yaml.safe_load(file) or {}
    raw_intents = raw_config.get("intents") if isinstance(raw_config, dict) else None
    if not isinstance(raw_intents, dict):
        raise ValueError(f"thiếu intents mapping: {path}")
    rules: list[IntentRule] = []
    for intent in (item for item in Intent if item is not Intent.UNKNOWN):
        raw_rule = raw_intents.get(intent.value)
        if not isinstance(raw_rule, dict) or not isinstance(raw_rule.get("patterns"), list):
            raise ValueError(f"thiếu patterns cho intent {intent.value}")
        raw_patterns = raw_rule["patterns"]
        if not raw_patterns or not all(isinstance(pattern, str) for pattern in raw_patterns):
            raise ValueError(f"patterns không hợp lệ cho intent {intent.value}")
        rules.append(
            IntentRule(intent=intent, patterns=tuple(re.compile(pattern) for pattern in raw_patterns))
        )
    return RuleIntentClassifier(rules)


def evaluate_rule_classifier(
    classifier: RuleIntentClassifier, samples: list[PreprocessedSample]
) -> dict[str, Any]:
    """Đo accuracy rule baseline để report có điểm so sánh công bằng với ML baseline."""
    correct = 0
    unknown = 0
    failures: list[dict[str, str]] = []
    for sample in samples:
        prediction = classifier.predict(sample.normalized_text)
        expected = sample.original.intent
        if prediction.intent is expected:
            correct += 1
        else:
            unknown += int(prediction.intent is Intent.UNKNOWN)
            failures.append(
                {
                    "id": sample.original.id,
                    "text": sample.normalized_text,
                    "expected": expected.value,
                    "predicted": prediction.intent.value,
                }
            )
    return {
        "total": len(samples),
        "accuracy": correct / len(samples) if samples else 0.0,
        "unknown": unknown,
        "failures": failures,
    }
