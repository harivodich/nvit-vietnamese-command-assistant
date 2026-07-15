"""Nạp config và lắp các dependency runtime dùng chung cho CLI/API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from nvit_assistant.actions import (
    ActionRouter,
    IntegratedActionRouter,
    MockActionRouter,
    OpenMeteoWeatherClient,
)
from nvit_assistant.nlu.action_gate import CommandActionGate
from nvit_assistant.nlu.normalizer import VietnameseNormalizer
from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.nlu.runtime_intent_classifier import IntentClassifier, load_classifier
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor
from nvit_assistant.nlu.slot_lexicon import sha256_file, validate_slot_lexicon_provenance
from nvit_assistant.schemas import Intent


@dataclass(frozen=True)
class RuntimeSettings:
    """Các giá trị runtime đã kiểm tra và chuyển thành đường dẫn tuyệt đối."""

    confidence_threshold: float
    intent_classifier_path: Path
    label_map_path: Path
    regional_variants_path: Path
    slot_values_path: Path
    slot_lexicon_path: Path
    action_mode: str
    contacts_path: Path
    music_catalog_path: Path
    weather_timeout_seconds: float


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


def _action_mode(value: Any, field_name: str) -> str:
    """Chỉ cho phép hai chế độ được mô tả rõ, tránh vô tình bật action thật khác."""
    if not isinstance(value, str) or value not in {"mock", "live-weather"}:
        raise ValueError(f"{field_name} phải là mock hoặc live-weather")
    return value


def resolve_project_root(explicit_root: Path | None = None) -> Path:
    """Tìm repo root từ tham số, biến môi trường, cwd rồi mới tới source tree."""
    if explicit_root is not None:
        resolved = explicit_root.resolve()
        if (resolved / "configs" / "app.yaml").is_file():
            return resolved
        raise FileNotFoundError(
            f"project root được chỉ định không có configs/app.yaml: {resolved}"
        )
    environment_root = os.environ.get("NVIT_PROJECT_ROOT")
    candidates = [
        Path(environment_root) if environment_root else None,
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.resolve()
            if (resolved / "configs" / "app.yaml").is_file():
                return resolved
    raise FileNotFoundError(
        "không tìm thấy configs/app.yaml; hãy chạy trong repo hoặc đặt NVIT_PROJECT_ROOT"
    )


def _require_file(path: Path, field_name: str) -> Path:
    """Báo lỗi cấu hình dễ hiểu trước khi thư viện bên dưới phát sinh traceback dài."""
    if not path.is_file():
        raise FileNotFoundError(f"không tìm thấy {field_name}: {path}")
    return path


def _validate_classifier_labels(classifier: IntentClassifier, label_map_path: Path) -> None:
    """Chặn model artifact và label map thuộc hai lần train khác nhau."""
    raw: Any = json.loads(label_map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(label, str) for label in raw):
        raise ValueError("intent label map phải là list chuỗi")
    configured_labels = sorted(raw)
    model_labels = sorted(classifier.labels)
    contract_labels = sorted(intent.value for intent in Intent if intent is not Intent.UNKNOWN)
    if configured_labels != contract_labels:
        raise ValueError("intent label map không khớp contract Intent")
    if model_labels != configured_labels:
        raise ValueError("intent classifier không khớp intent label map")


def _validate_intent_artifact_provenance(
    metadata_path: Path, expected_files: dict[str, Path]
) -> None:
    """Kiểm checksum trước khi joblib.load để bắt artifact cũ hoặc file bị đổi ngoài ý muốn."""
    raw: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("intent metadata phải có schema_version=1")
    if raw.get("fit_split") != "train_plus_validation":
        raise ValueError("intent metadata phải khai báo fit_split=train_plus_validation")
    trained_sklearn_version = raw.get("sklearn_version")
    runtime_sklearn_version = version("scikit-learn")
    if trained_sklearn_version != runtime_sklearn_version:
        raise ValueError(
            "scikit-learn runtime không khớp version đã train artifact: "
            f"{runtime_sklearn_version} != {trained_sklearn_version}"
        )
    recorded = raw.get("files_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("intent metadata.files_sha256 phải là mapping")
    for name, path in expected_files.items():
        expected_hash = recorded.get(name)
        if not isinstance(expected_hash, str) or expected_hash != sha256_file(path):
            raise ValueError(
                f"intent artifact không khớp checksum {name}; hãy chạy scripts/train_intent.py"
            )


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
    actions = _mapping(config.get("actions", {}), "actions")
    weather_timeout = actions.get("weather_timeout_seconds", 5.0)
    if (
        not isinstance(weather_timeout, (int, float))
        or isinstance(weather_timeout, bool)
        or not 0.0 < float(weather_timeout) <= 30.0
    ):
        raise ValueError("actions.weather_timeout_seconds phải nằm trong (0, 30]")
    return RuntimeSettings(
        confidence_threshold=float(threshold),
        intent_classifier_path=_relative_path(
            root, model.get("intent_classifier_path"), "model.intent_classifier_path"
        ),
        label_map_path=_relative_path(root, model.get("label_map_path"), "model.label_map_path"),
        regional_variants_path=_relative_path(
            root, nlu.get("regional_variants_path"), "nlu.regional_variants_path"
        ),
        slot_values_path=_relative_path(root, nlu.get("slot_values_path"), "nlu.slot_values_path"),
        slot_lexicon_path=_relative_path(
            root, nlu.get("slot_lexicon_path"), "nlu.slot_lexicon_path"
        ),
        action_mode=_action_mode(actions.get("mode", "mock"), "actions.mode"),
        contacts_path=_relative_path(
            root,
            actions.get("contacts_path", "data/fake_contacts.json"),
            "actions.contacts_path",
        ),
        music_catalog_path=_relative_path(
            root,
            actions.get("music_catalog_path", "data/music_catalog.json"),
            "actions.music_catalog_path",
        ),
        weather_timeout_seconds=float(weather_timeout),
    )


def build_pipeline(
    project_root: Path | None = None, action_mode: str | None = None
) -> NLUPipeline:
    """Lắp pipeline hoàn chỉnh một lần; CLI/API không tự tạo dependency khác nhau."""
    root = resolve_project_root(project_root)
    settings = load_runtime_settings(root / "configs" / "app.yaml", root)
    model_path = _require_file(
        settings.intent_classifier_path, "intent classifier artifact"
    )
    label_map_path = _require_file(settings.label_map_path, "intent label map")
    train_path = _require_file(root / "data" / "samples" / "train.jsonl", "train split")
    validation_path = _require_file(
        root / "data" / "samples" / "validation.jsonl", "validation split"
    )
    _validate_intent_artifact_provenance(
        _require_file(model_path.with_suffix(".metadata.json"), "intent artifact metadata"),
        {
            "model": model_path,
            "label_map": label_map_path,
            "train": train_path,
            "validation": validation_path,
            "intent_training_config": _require_file(
                root / "configs" / "intent_training.yaml", "intent training config"
            ),
            "regional_variants": _require_file(
                settings.regional_variants_path, "regional variants config"
            ),
        },
    )
    classifier = load_classifier(model_path)
    _validate_classifier_labels(classifier, label_map_path)
    slot_lexicon_path = _require_file(settings.slot_lexicon_path, "slot lexicon")
    validate_slot_lexicon_provenance(
        slot_lexicon_path, train_path, settings.regional_variants_path
    )
    selected_action_mode = _action_mode(
        action_mode or os.environ.get("NVIT_ACTION_MODE") or settings.action_mode,
        "action mode runtime",
    )
    action_router: ActionRouter
    if selected_action_mode == "live-weather":
        action_router = IntegratedActionRouter(
            _require_file(settings.contacts_path, "fake contacts data"),
            _require_file(settings.music_catalog_path, "music catalog data"),
            OpenMeteoWeatherClient(settings.weather_timeout_seconds),
        )
    else:
        action_router = MockActionRouter()
    return NLUPipeline(
        normalizer=VietnameseNormalizer(
            _require_file(settings.regional_variants_path, "regional variants config")
        ),
        intent_classifier=classifier,
        slot_extractor=RegexSlotExtractor(
            _require_file(settings.slot_values_path, "slot values config"),
            slot_lexicon_path,
        ),
        action_router=action_router,
        action_gate=CommandActionGate(),
        confidence_threshold=settings.confidence_threshold,
    )
