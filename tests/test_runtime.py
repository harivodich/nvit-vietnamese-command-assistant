from pathlib import Path

import pytest

from nvit_assistant.runtime import build_pipeline, load_runtime_settings


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_settings_resolve_paths_inside_project() -> None:
    settings = load_runtime_settings(ROOT / "configs" / "app.yaml", ROOT)

    assert settings.confidence_threshold == 0.45
    assert settings.intent_classifier_path == (ROOT / "models" / "intent_classifier.joblib")
    assert settings.regional_variants_path == (ROOT / "configs" / "regional_variants.yaml")
    assert settings.slot_values_path == (ROOT / "configs" / "slot_values.yaml")


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
    assert pipeline.confidence_threshold == 0.45
