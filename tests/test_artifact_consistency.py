import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.preprocessing import preprocess_splits


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Tính checksum giống pipeline để CI phát hiện report/artifact bị cũ."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def assert_hashes(recorded: dict[str, str], expected_paths: dict[str, str]) -> None:
    for name, relative_path in expected_paths.items():
        assert recorded[name] == sha256_file(ROOT / relative_path), name


@pytest.fixture(scope="module")
def preprocessed_hashes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Tái tạo preprocess trong thư mục tạm để test không phụ thuộc file bị Git bỏ qua."""
    output_dir = tmp_path_factory.mktemp("preprocessed")
    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    report = preprocess_splits(
        ROOT / "data" / "samples",
        output_dir,
        normalizer,
        ("train.jsonl", "validation.jsonl"),
    )
    assert report["totals"]["dropped"] == 0
    return {
        "preprocessed_train": sha256_file(output_dir / "train.jsonl"),
        "preprocessed_validation": sha256_file(output_dir / "validation.jsonl"),
    }


def assert_preprocessed_hashes(
    recorded: dict[str, str], preprocessed_hashes: dict[str, str]
) -> None:
    """So checksum report với output vừa tái tạo từ JSONL nguồn."""
    for name, expected_hash in preprocessed_hashes.items():
        assert recorded[name] == expected_hash, name


def test_intent_training_artifacts_match_report(
    preprocessed_hashes: dict[str, str],
) -> None:
    report = load_json("reports/intent_training_report.json")

    assert_hashes(
        report["artifacts_sha256"],
        {
            "model": "models/intent_classifier.joblib",
            "label_map": "models/intent_label_map.json",
            "metadata": "models/intent_classifier.metadata.json",
            "train": "data/samples/train.jsonl",
            "validation": "data/samples/validation.jsonl",
            "intent_training_config": "configs/intent_training.yaml",
            "regional_variants": "configs/regional_variants.yaml",
        },
    )
    assert_preprocessed_hashes(report["artifacts_sha256"], preprocessed_hashes)
    assert report["test_used"] is False


def test_development_reports_match_current_inputs(
    preprocessed_hashes: dict[str, str],
) -> None:
    confidence = load_json("reports/confidence_gate_report.json")
    assert_hashes(
        confidence["artifacts_sha256"],
        {
            "app_config": "configs/app.yaml",
            "intent_training_config": "configs/intent_training.yaml",
            "intent_training_report": "reports/intent_training_report.json",
            "train": "data/samples/train.jsonl",
            "validation": "data/samples/validation.jsonl",
            "regional_variants": "configs/regional_variants.yaml",
        },
    )
    assert_preprocessed_hashes(confidence["artifacts_sha256"], preprocessed_hashes)

    slots = load_json("reports/slot_extraction_report.json")
    assert_hashes(
        slots["artifacts_sha256"],
        {
            "slot_lexicon": "models/slot_lexicon.json",
            "slot_values": "configs/slot_values.yaml",
            "regional_variants": "configs/regional_variants.yaml",
            "train": "data/samples/train.jsonl",
            "validation": "data/samples/validation.jsonl",
        },
    )

    action = load_json("reports/action_safety_report.json")
    assert_hashes(
        action["artifacts_sha256"],
        {
            "challenge": "data/action_safety_challenge.jsonl",
            "validation": "data/samples/validation.jsonl",
        },
    )
    assert confidence["test_used"] is False
    assert slots["methodology"]["test_used"] is False
    assert action["methodology"]["test_used"] is False


def test_semantic_comparison_uses_same_snapshot_as_tfidf(
    preprocessed_hashes: dict[str, str],
) -> None:
    semantic = load_json("reports/semantic_intent_report.json")
    comparison = load_json("reports/model_comparison_report.json")
    baseline = load_json("reports/intent_training_report.json")

    assert_hashes(
        semantic["inputs_sha256"],
        {
            "semantic_training_config": "configs/semantic_intent_training.yaml",
            "baseline_report": "reports/intent_training_report.json",
            "train": "data/samples/train.jsonl",
            "validation": "data/samples/validation.jsonl",
            "regional_variants": "configs/regional_variants.yaml",
        },
    )
    assert_preprocessed_hashes(semantic["inputs_sha256"], preprocessed_hashes)
    assert semantic["artifact"]["sha256"] == sha256_file(
        ROOT / "models/semantic_intent_classifier.joblib"
    )
    assert comparison["snapshot_sha256"] == {
        "preprocessed_train": semantic["inputs_sha256"]["preprocessed_train"],
        "preprocessed_validation": semantic["inputs_sha256"]["preprocessed_validation"],
    }
    assert comparison["models"]["tfidf_logistic_regression"]["macro_f1"] == (
        baseline["training"]["selected_validation"]["macro_f1"]
    )
    assert comparison["models"]["e5_frozen_embedding_logistic_regression"][
        "macro_f1"
    ] == semantic["training"]["selected_validation"]["macro_f1"]
    assert comparison["test_used"] is False
