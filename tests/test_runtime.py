import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pytest

from nvit_assistant.nlu.runtime_intent_classifier import IntentClassifier
from nvit_assistant.runtime import (
    _validate_classifier_labels,
    _validate_intent_artifact_provenance,
    build_pipeline,
    load_runtime_settings,
    resolve_project_root,
)


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_settings_resolve_paths_inside_project() -> None:
    settings = load_runtime_settings(ROOT / "configs" / "app.yaml", ROOT)

    assert settings.confidence_threshold == 0.35
    assert settings.intent_classifier_path == (ROOT / "models" / "intent_classifier.joblib")
    assert settings.label_map_path == (ROOT / "models" / "intent_label_map.json")
    assert settings.regional_variants_path == (ROOT / "configs" / "regional_variants.yaml")
    assert settings.slot_values_path == (ROOT / "configs" / "slot_values.yaml")
    assert settings.slot_lexicon_path == (ROOT / "models" / "slot_lexicon.json")


def test_runtime_settings_reject_invalid_threshold(tmp_path: Path) -> None:
    config = tmp_path / "app.yaml"
    config.write_text(
        """
confidence_threshold: 1.5
model:
  intent_classifier_path: models/model.joblib
nlu:
  regional_variants_path: configs/regions.yaml
  slot_values_path: configs/slots.yaml
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_runtime_settings(config, tmp_path)


def test_build_pipeline_has_action_router_and_confidence_gate() -> None:
    pipeline = build_pipeline(ROOT)

    assert pipeline.action_router is not None
    assert pipeline.confidence_threshold == 0.35


def test_resolve_project_root_uses_environment_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIT_PROJECT_ROOT", str(ROOT))

    assert resolve_project_root() == ROOT.resolve()


def test_classifier_label_map_mismatch_fails_fast(tmp_path: Path) -> None:
    class FakePipeline:
        classes_ = ["ask_weather", "call_contact", "play_music", "set_alarm", "set_reminder"]

    label_map = tmp_path / "labels.json"
    label_map.write_text(json.dumps(["ask_weather"]), encoding="utf-8")

    with pytest.raises(ValueError, match="contract Intent"):
        _validate_classifier_labels(IntentClassifier(FakePipeline()), label_map)


def test_build_pipeline_reports_missing_model_artifact(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text(
        """
confidence_threshold: 0.35
model:
  intent_classifier_path: models/missing.joblib
  label_map_path: models/labels.json
nlu:
  regional_variants_path: configs/regions.yaml
  slot_values_path: configs/slots.yaml
  slot_lexicon_path: models/slots.json
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="intent classifier artifact"):
        build_pipeline(tmp_path)


def test_intent_artifact_checksum_detects_tampering(tmp_path: Path) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"trusted-model")
    metadata = tmp_path / "model.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit_split": "train_plus_validation",
                "sklearn_version": version("scikit-learn"),
                "files_sha256": {"model": hashlib.sha256(model.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    _validate_intent_artifact_provenance(metadata, {"model": model})
    model.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum model"):
        _validate_intent_artifact_provenance(metadata, {"model": model})


def test_intent_artifact_rejects_different_sklearn_version(tmp_path: Path) -> None:
    metadata = tmp_path / "model.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit_split": "train_plus_validation",
                "sklearn_version": "0.0.0",
                "files_sha256": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scikit-learn runtime"):
        _validate_intent_artifact_provenance(metadata, {})
