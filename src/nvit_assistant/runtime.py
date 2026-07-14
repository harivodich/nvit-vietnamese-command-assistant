"""Nạp config và lắp các dependency runtime dùng chung cho CLI/API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nvit_assistant.actions import MockActionRouter
from nvit_assistant.nlu.intent_classifier import load_classifier
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor


@dataclass(frozen=True)
class RuntimeSettings:
    """Các giá trị runtime đã kiểm tra và chuyển thành đường dẫn tuyệt đối."""

    confidence_threshold: float
    intent_classifier_path: Path
    regional_variants_path: Path
    slot_values_path: Path


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Kiểm tra một nhánh YAML là mapping trước khi đọc field con."""
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} phải là mapping")
    return value


def _relative_path(root: Path, value: Any, field_name: str) -> Path:
    """Chuyển đường dẫn trong config thành tuyệt đối và chặn giá trị sai kiểu."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} phải là chuỗi đường dẫn")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_runtime_settings(config_path: Path, project_root: Path) -> RuntimeSettings:
    """Đọc app.yaml; fail-fast nếu threshold hoặc đường dẫn cấu hình không hợp lệ."""
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = project_root.resolve()
    config = _mapping(raw, "app config")
    threshold = config.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("confidence_threshold phải là số")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("confidence_threshold phải nằm trong [0, 1]")
    model = _mapping(config.get("model"), "model")
    nlu = _mapping(config.get("nlu"), "nlu")
    return RuntimeSettings(
        confidence_threshold=float(threshold),
        intent_classifier_path=_relative_path(
            root, model.get("intent_classifier_path"), "model.intent_classifier_path"
        ),
        regional_variants_path=_relative_path(
            root, nlu.get("regional_variants_path"), "nlu.regional_variants_path"
        ),
        slot_values_path=_relative_path(root, nlu.get("slot_values_path"), "nlu.slot_values_path"),
    )


def build_pipeline(project_root: Path) -> NLUPipeline:
    """Lắp pipeline hoàn chỉnh một lần; CLI/API không tự tạo dependency khác nhau."""
    root = project_root.resolve()
    settings = load_runtime_settings(root / "configs" / "app.yaml", root)
    return NLUPipeline(
        normalizer=VietnameseNormalizer(settings.regional_variants_path),
        intent_classifier=load_classifier(settings.intent_classifier_path),
        slot_extractor=RegexSlotExtractor(settings.slot_values_path),
        action_router=MockActionRouter(),
        confidence_threshold=settings.confidence_threshold,
    )
