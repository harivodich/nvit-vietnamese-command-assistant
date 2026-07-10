from nvit_assistant.dataset_builder import (
    combine_spans,
    map_massive_row,
    parse_annotated_utterance,
    split_group_ids,
    split_template_indexes,
)
from nvit_assistant.schemas import Intent, Region


def good_judgments() -> list[dict[str, object]]:
    """Tạo ba phiếu MASSIVE đạt ngưỡng chất lượng để dùng lại trong fixture."""
    return [
        {
            "intent_score": 1,
            "slots_score": 1,
            "grammar_score": 4,
            "spelling_score": 2,
            "language_identification": "target",
        }
        for _ in range(3)
    ]


def test_parse_annotated_utterance_preserves_slot_offsets() -> None:
    text, spans = parse_annotated_utterance(
        "cài báo thức [date : ngày mai] lúc [time : sáu giờ sáng]"
    )

    assert text == "cài báo thức ngày mai lúc sáu giờ sáng"
    assert combine_spans(text, spans, {"date", "time"}) == "ngày mai lúc sáu giờ sáng"


def test_map_massive_alarm_keeps_provenance_and_standard_region() -> None:
    mapped = map_massive_row(
        {
            "id": "42",
            "partition": "train",
            "intent": "alarm_set",
            "utt": "cài báo thức ngày mai lúc sáu giờ sáng",
            "annot_utt": "cài báo thức [date : ngày mai] lúc [time : sáu giờ sáng]",
            "judgments": good_judgments(),
        }
    )

    assert mapped is not None
    partition, sample = mapped
    assert partition == "train"
    assert sample.intent is Intent.SET_ALARM
    assert sample.region is Region.STANDARD
    assert sample.slots["datetime"] == "ngày mai lúc sáu giờ sáng"
    assert sample.source_ref == "massive:1.0:vi-VN:42"


def test_map_massive_does_not_treat_generic_notification_as_reminder() -> None:
    mapped = map_massive_row(
        {
            "id": "43",
            "partition": "test",
            "intent": "calendar_set",
            "utt": "đặt thông báo về thời tiết xấu",
            "annot_utt": "đặt thông báo về [event_name : thời tiết xấu]",
            "judgments": good_judgments(),
        }
    )

    assert mapped is None


def test_group_split_is_deterministic_and_exclusive() -> None:
    group_ids = [f"group-{index}" for index in range(20)]

    first = split_group_ids(group_ids, seed=42)
    second = split_group_ids(list(reversed(group_ids)), seed=42)

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}


def test_template_family_split_keeps_all_three_splits() -> None:
    assignments = split_template_indexes(4, seed=42, scope="set_alarm")

    assert set(assignments.values()) == {"train", "validation", "test"}
    assert split_template_indexes(4, seed=42, scope="set_alarm") == assignments
