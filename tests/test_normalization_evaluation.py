from pathlib import Path

from nvit_assistant.nlu.normalization_evaluation import (
    evaluate_normalizer,
    load_normalization_challenge,
)
from nvit_assistant.nlu.normalizer import VietnameseNormalizer


def test_normalizer_passes_project_challenge_set() -> None:
    """Benchmark chứa câu command-domain độc lập với data train intent phải luôn xanh."""
    normalizer = VietnameseNormalizer(Path("configs/regional_variants.yaml"))
    challenges = load_normalization_challenge(Path("data/normalization_challenge.jsonl"))

    report = evaluate_normalizer(normalizer, challenges)

    assert report["total"] >= 30
    assert report["text_accuracy"] == 1.0
    assert report["region_accuracy"] == 1.0
    assert report["failures"] == []
