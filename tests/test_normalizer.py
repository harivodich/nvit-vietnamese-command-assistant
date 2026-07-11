from pathlib import Path

import pytest

from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.schemas import Region


CONFIG_PATH = Path("configs/regional_variants.yaml")


@pytest.fixture
def normalizer() -> VietnameseNormalizer:
    """Tạo normalizer thật từ config version-controlled cho mọi test."""
    return VietnameseNormalizer(CONFIG_PATH)


def test_normalizes_north_variant_and_detects_region(normalizer: VietnameseNormalizer) -> None:
    result = normalizer.normalize("Ngoài Hà Nội giời thế nào nhá?")

    assert result.normalized_text == "ngoài hà nội trời thế nào nhé?"
    assert result.region is Region.NORTH
    assert "giời -> trời" in result.matched_variants


def test_normalizes_central_long_phrase_before_short_phrase(normalizer: VietnameseNormalizer) -> None:
    result = normalizer.normalize("Bữa ni ở Huế trời răng rồi hỉ")

    assert result.normalized_text == "hôm nay ở huế trời sao rồi nhỉ"
    assert result.region is Region.CENTRAL
    assert "bữa ni -> hôm nay" in result.matched_variants


def test_normalizes_south_variant_and_stt_without_guessing_other_words(
    normalizer: VietnameseNormalizer,
) -> None:
    result = normalizer.normalize("Hom nay mo nhac cho tui nghe nghen")

    assert result.normalized_text == "hôm nay mở nhạc cho tôi nghe nhé"
    assert result.region is Region.SOUTH
    assert "hom nay -> hôm nay" in result.matched_variants
    assert "nghen -> nhé" in result.matched_variants


def test_keeps_contextual_particle_when_sentence_type_is_unclear(
    normalizer: VietnameseNormalizer,
) -> None:
    result = normalizer.normalize("câu này hỉ")

    assert result.normalized_text == "câu này hỉ"
    assert result.region is Region.STANDARD


def test_contextual_particle_uses_ne_for_request(normalizer: VietnameseNormalizer) -> None:
    result = normalizer.normalize("Nhắc tui uống thuốc hỉ")

    assert result.normalized_text == "nhắc tôi uống thuốc nhé"
    assert result.region is Region.CENTRAL


def test_returns_unknown_when_regional_evidence_conflicts(normalizer: VietnameseNormalizer) -> None:
    result = normalizer.normalize("tớ cần gọi mẹ dùm")

    assert result.normalized_text == "tôi cần gọi mẹ giúp"
    assert result.region is Region.UNKNOWN


def test_region_hint_is_authoritative(normalizer: VietnameseNormalizer) -> None:
    result = normalizer.normalize("Bữa ni nhắc tui uống thuốc hỉ", Region.CENTRAL)

    assert result.normalized_text == "hôm nay nhắc tôi uống thuốc nhé"
    assert result.region is Region.CENTRAL


def test_rejects_blank_text(normalizer: VietnameseNormalizer) -> None:
    with pytest.raises(ValueError, match="không được"):
        normalizer.normalize("   ")


def test_preserves_phone_number_and_time_surface(normalizer: VietnameseNormalizer) -> None:
    result = normalizer.normalize("gọi số 090 000 0000 lúc 6 giờ", Region.SOUTH)

    assert result.normalized_text == "gọi số 090 000 0000 lúc 6 giờ"
