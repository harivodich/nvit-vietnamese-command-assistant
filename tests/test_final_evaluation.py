from nvit_assistant.eval.final_evaluation import (
    INTENT_LABELS,
    SampleEvaluation,
    classification_metrics,
    slot_metrics,
    summarize_records,
)
from nvit_assistant.schemas import (
    ActionResult,
    ActionStatus,
    ActionType,
    AnnotationQuality,
    DatasetSample,
    DataSource,
    Intent,
    ParseResult,
    Region,
    VariantType,
)


EXPECTED_SLOTS = {
    Intent.SET_REMINDER: {"reminder_text": "uống thuốc"},
    Intent.SET_ALARM: {"datetime": "7 giờ"},
    Intent.ASK_WEATHER: {},
    Intent.PLAY_MUSIC: {},
    Intent.CALL_CONTACT: {"contact_name": "mẹ"},
}
ACTION_TYPES = {
    Intent.SET_REMINDER: ActionType.CREATE_REMINDER,
    Intent.SET_ALARM: ActionType.SET_ALARM,
    Intent.ASK_WEATHER: ActionType.QUERY_WEATHER,
    Intent.PLAY_MUSIC: ActionType.PLAY_MUSIC,
    Intent.CALL_CONTACT: ActionType.CALL,
}


def make_record(index: int, intent: Intent, reject: bool = False) -> SampleEvaluation:
    slots = EXPECTED_SLOTS[intent]
    sample = DatasetSample(
        id=f"sample-{index}",
        group_id=f"group-{index}",
        text=f"câu lệnh {index}",
        region=Region.STANDARD,
        intent=intent,
        slots=slots,
        source=DataSource.SYNTHETIC,
        variant_type=VariantType.FORMAL,
        annotation_quality=AnnotationQuality.TEMPLATE_GENERATED,
    )
    probabilities = {label: 0.025 for label in INTENT_LABELS}
    probabilities[intent.value] = 0.9
    result_intent = Intent.UNKNOWN if reject else intent
    result_slots = {} if reject else slots
    action = (
        None
        if reject
        else ActionResult(
            type=ACTION_TYPES[intent],
            status=ActionStatus.MOCKED,
            payload={},
        )
    )
    result = ParseResult(
        text=sample.text,
        normalized_text=sample.text,
        region=Region.STANDARD,
        intent=result_intent,
        confidence=0.9,
        slots=result_slots,
        action=action,
        response="ok",
    )
    return SampleEvaluation(
        sample=sample,
        raw_intent=intent.value,
        raw_confidence=0.9,
        probabilities=probabilities,
        result=result,
        expected_slots=slots,
        oracle_slots=slots,
        latency_ms=float(index + 1),
    )


def test_classification_metrics_include_macro_and_weighted_scores() -> None:
    metrics = classification_metrics(
        INTENT_LABELS,
        INTENT_LABELS,
        INTENT_LABELS,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_slot_metrics_count_exact_and_pair_level_errors() -> None:
    metrics = slot_metrics(
        [{"contact_name": "mẹ"}, {"song": "lạc trôi"}],
        [{"contact_name": "mẹ"}, {"artist": "sơn tùng"}],
    )

    assert metrics["exact_match"] == 0.5
    assert metrics["micro"]["true_positive"] == 1
    assert metrics["micro"]["false_positive"] == 1
    assert metrics["micro"]["false_negative"] == 1
    assert metrics["micro"]["f1"] == 0.5


def test_summarize_records_separates_raw_model_and_runtime_rejection() -> None:
    records = [
        make_record(index, intent, reject=intent is Intent.CALL_CONTACT)
        for index, intent in enumerate(
            intent for intent in Intent if intent is not Intent.UNKNOWN
        )
    ]

    report = summarize_records(records)

    assert report["raw_model_intent"]["accuracy"] == 1.0
    assert report["runtime_intent"]["accuracy"] == 0.8
    assert report["runtime_intent"]["coverage"] == 0.8
    assert report["runtime_intent"]["selective_accuracy"] == 1.0
    assert report["oracle_slots"]["exact_match"] == 1.0
    assert report["end_to_end_slots"]["exact_match"] == 0.8
    assert report["full_command_success"] == 0.8
    assert report["failure_count"] == 1
    assert report["failures"][0]["reasons"] == ["intent", "slots", "action"]
