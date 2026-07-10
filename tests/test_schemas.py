import pytest
from pydantic import ValidationError

from nvit_assistant.schemas import DataSource, DatasetSample, Intent, ParseRequest, Region, VariantType


def test_parse_request_accepts_text_and_region_hint() -> None:
    request = ParseRequest(text="mai nhắc tôi gọi mẹ", region_hint=Region.SOUTH)

    assert request.text == "mai nhắc tôi gọi mẹ"
    assert request.region_hint is Region.SOUTH


def test_parse_request_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        ParseRequest(text="   ")


def test_dataset_sample_rejects_unknown_slot() -> None:
    with pytest.raises(ValidationError, match="unknown slot names"):
        DatasetSample(
            id="sample-001",
            text="gọi cho mẹ",
            region=Region.SOUTH,
            intent=Intent.CALL_CONTACT,
            slots={"phone_number": "0123456789"},
            source=DataSource.MANUAL,
            variant_type=VariantType.ACCENTED,
        )


def test_dataset_sample_requires_intent_slot() -> None:
    with pytest.raises(ValidationError, match="missing required slot group"):
        DatasetSample(
            id="sample-002",
            text="đặt báo thức lúc 6 giờ",
            region=Region.NORTH,
            intent=Intent.SET_ALARM,
            source=DataSource.MANUAL,
            variant_type=VariantType.ACCENTED,
        )


def test_call_sample_accepts_phone_number_without_contact_name() -> None:
    sample = DatasetSample(
        id="sample-004",
        text="gọi 0901234567",
        region=Region.SOUTH,
        intent=Intent.CALL_CONTACT,
        slots={"phone_number": "0901234567"},
        source=DataSource.MANUAL,
        variant_type=VariantType.ACCENTED,
    )

    assert sample.slots["phone_number"] == "0901234567"


def test_dataset_sample_rejects_unknown_region() -> None:
    with pytest.raises(ValidationError, match="unknown region"):
        DatasetSample(
            id="sample-003",
            text="gọi cho mẹ",
            region=Region.UNKNOWN,
            intent=Intent.CALL_CONTACT,
            slots={"contact_name": "mẹ"},
            source=DataSource.MANUAL,
            variant_type=VariantType.ACCENTED,
        )
