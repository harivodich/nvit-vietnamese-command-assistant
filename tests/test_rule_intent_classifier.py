from pathlib import Path

import pytest

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


def test_rule_config_must_match_schema_slot_contract(tmp_path: Path) -> None:
    source = Path("configs/intents.yaml").read_text(encoding="utf-8")
    drifted = source.replace(
        "required_slots: [datetime]\n    optional_slots: []",
        "required_slots: [datetime, location]\n    optional_slots: []",
        1,
    )
    config = tmp_path / "intents.yaml"
    config.write_text(drifted, encoding="utf-8")

    with pytest.raises(ValueError, match="required slot contract"):
        load_rule_classifier(config)
