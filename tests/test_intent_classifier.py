from pathlib import Path

from nvit_assistant.nlu.intent_classifier import (
    CandidateConfig,
    load_classifier,
    save_classifier,
    train_with_validation,
)
from nvit_assistant.schemas import (
    AnnotationQuality,
    DataSource,
    DatasetSample,
    Intent,
    PreprocessedSample,
    Region,
    VariantType,
)


def make_sample(intent: Intent, text: str, sample_id: str) -> PreprocessedSample:
    """Tạo sample preprocess tối thiểu để test trainer với đủ năm intent."""
    slots_by_intent = {
        Intent.SET_REMINDER: {"reminder_text": "uống thuốc"},
        Intent.SET_ALARM: {"datetime": "6 giờ sáng"},
        Intent.ASK_WEATHER: {},
        Intent.PLAY_MUSIC: {},
        Intent.CALL_CONTACT: {"contact_name": "mẹ"},
    }
    original = DatasetSample(
        id=sample_id,
        group_id=sample_id,
        text=text,
        region=Region.STANDARD,
        intent=intent,
        slots=slots_by_intent[intent],
        source=DataSource.MANUAL,
        variant_type=VariantType.FORMAL,
        annotation_quality=AnnotationQuality.REVIEWED,
    )
    return PreprocessedSample(
        original=original,
        normalized_text=text,
        normalized_slots=slots_by_intent[intent],
        normalizer_region=Region.STANDARD,
    )


def test_trainer_selects_candidate_and_predicts_known_intent(tmp_path: Path) -> None:
    train = [
        make_sample(Intent.SET_REMINDER, "nhắc tôi uống thuốc tối nay", "reminder-train"),
        make_sample(Intent.SET_ALARM, "đặt báo thức lúc sáu giờ sáng", "alarm-train"),
        make_sample(Intent.ASK_WEATHER, "thời tiết ở hà nội hôm nay", "weather-train"),
        make_sample(Intent.PLAY_MUSIC, "mở nhạc giúp tôi", "music-train"),
        make_sample(Intent.CALL_CONTACT, "gọi cho mẹ", "call-train"),
    ]
    validation = [
        make_sample(Intent.SET_REMINDER, "nhắc tôi uống thuốc", "reminder-validation"),
        make_sample(Intent.SET_ALARM, "cài báo thức sáu giờ", "alarm-validation"),
        make_sample(Intent.ASK_WEATHER, "dự báo thời tiết hà nội", "weather-validation"),
        make_sample(Intent.PLAY_MUSIC, "bật nhạc lên", "music-validation"),
        make_sample(Intent.CALL_CONTACT, "gọi mẹ giúp tôi", "call-validation"),
    ]
    config = CandidateConfig(
        name="test-word",
        word_ngram_range=(1, 2),
        word_min_df=1,
        use_char_features=False,
        c=2.0,
    )

    figures_dir = tmp_path / "figures"
    classifier, selected, report = train_with_validation(
        train,
        validation,
        [config],
        seed=42,
        figures_dir=figures_dir,
    )
    prediction = classifier.predict("gọi cho mẹ")

    assert selected == config
    assert report["selected_validation"]["total"] == 5
    assert report["selected_validation"]["probability_metrics"]["log_loss"] >= 0.0
    assert len(report["selected_validation"]["figures"]) == 4
    assert (figures_dir / "intent_confusion_matrix.png").exists()
    assert (figures_dir / "intent_roc_curve.png").exists()
    assert (figures_dir / "intent_pr_curve.png").exists()
    assert (figures_dir / "intent_reliability_curve.png").exists()
    assert prediction.intent is Intent.CALL_CONTACT
    assert 0.0 <= prediction.confidence <= 1.0

    model_path = tmp_path / "intent.joblib"
    label_map_path = tmp_path / "labels.json"
    save_classifier(classifier, model_path, label_map_path)

    assert load_classifier(model_path).predict("gọi cho mẹ").intent is Intent.CALL_CONTACT
