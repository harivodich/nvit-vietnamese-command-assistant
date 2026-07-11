from pathlib import Path

from nvit_assistant.nlu.rule_intent_classifier import load_rule_classifier
from nvit_assistant.schemas import Intent


def test_rule_classifier_prioritizes_wake_up_pattern_over_call() -> None:
    classifier = load_rule_classifier(Path("configs/intents.yaml"))

    prediction = classifier.predict("hai tiếng nữa gọi tôi dậy nhé")

    assert prediction.intent is Intent.SET_ALARM


def test_rule_classifier_returns_unknown_when_no_rule_matches() -> None:
    classifier = load_rule_classifier(Path("configs/intents.yaml"))

    prediction = classifier.predict("xin chào trợ lý")

    assert prediction.intent is Intent.UNKNOWN
