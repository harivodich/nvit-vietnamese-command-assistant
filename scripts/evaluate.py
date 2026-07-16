"""Đánh giá code hiện tại trên benchmark test đã khóa và sinh report nộp challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nvit_matplotlib"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from nvit_assistant.data_validation import read_samples  # noqa: E402
from nvit_assistant.eval.final_evaluation import evaluate_final_pipeline  # noqa: E402
from nvit_assistant.nlu.slot_lexicon import sha256_file  # noqa: E402
from nvit_assistant.runtime import build_pipeline  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TEST_FILES = (
    "test_central.jsonl",
    "test_north.jsonl",
    "test_south.jsonl",
    "test_standard.jsonl",
)


def load_locked_test_set(samples_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    """Xác minh từng checksum và combined hash trước khi đọc nhãn test."""
    manifest_path = samples_dir / "manifest.json"
    manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest test phải là JSON object")
    expected_hashes = manifest.get("files_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("manifest thiếu files_sha256")
    test_digest = hashlib.sha256()
    samples = []
    observed_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for filename in TEST_FILES:
        path = samples_dir / filename
        observed_hash = sha256_file(path)
        if expected_hashes.get(filename) != observed_hash:
            raise ValueError(f"test file không khớp manifest: {filename}")
        test_digest.update(filename.encode("utf-8"))
        test_digest.update(path.read_bytes())
        file_samples = read_samples(path)
        samples.extend(file_samples)
        observed_hashes[filename] = observed_hash
        counts[filename] = len(file_samples)
    combined_hash = test_digest.hexdigest()
    if manifest.get("test_set_sha256") != combined_hash:
        raise ValueError("combined test hash không khớp manifest")
    return samples, {
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "test_set_sha256": combined_hash,
        "files_sha256": observed_hashes,
        "sample_counts": counts,
    }


def snapshot_hashes() -> dict[str, str]:
    """Khóa các input có thể làm thay đổi kết quả final evaluation."""
    paths = {
        "app_config": ROOT / "configs" / "app.yaml",
        "regional_variants": ROOT / "configs" / "regional_variants.yaml",
        "slot_values": ROOT / "configs" / "slot_values.yaml",
        "intent_model": ROOT / "models" / "intent_classifier.joblib",
        "intent_metadata": ROOT / "models" / "intent_classifier.metadata.json",
        "intent_labels": ROOT / "models" / "intent_label_map.json",
        "slot_lexicon": ROOT / "models" / "slot_lexicon.json",
        "pipeline_source": ROOT / "src" / "nvit_assistant" / "nlu" / "pipeline.py",
        "normalizer_source": ROOT / "src" / "nvit_assistant" / "nlu" / "normalizer.py",
        "slot_extractor_source": ROOT / "src" / "nvit_assistant" / "nlu" / "slot_extractor.py",
        "action_gate_source": ROOT / "src" / "nvit_assistant" / "nlu" / "action_gate.py",
        "final_evaluator_source": (
            ROOT / "src" / "nvit_assistant" / "eval" / "final_evaluation.py"
        ),
        "evaluation_script": ROOT / "scripts" / "evaluate.py",
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def write_confusion_figure(report: dict[str, Any], output_path: Path) -> None:
    """Vẽ confusion matrix intent raw-model của snapshot đang được báo cáo."""
    intent = report["metrics"]["raw_model_intent"]
    matrix = np.asarray(intent["confusion_matrix"])
    labels = intent["labels"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axis, colorbar=False, xticks_rotation=30
    )
    axis.set_title("Current test intent confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a readable summary while keeping per-sample failures in JSON."""
    metrics = report["metrics"]
    raw = metrics["raw_model_intent"]
    runtime = metrics["runtime_intent"]
    oracle_slots = metrics["oracle_slots"]
    end_to_end_slots = metrics["end_to_end_slots"]
    provenance = report["evaluation_provenance"]
    post_audit = provenance["stage"] == "post_audit_diagnostic"
    lines = [
        (
            "# Post-audit Final Evaluation — NVIT Vietnamese Command Assistant"
            if post_audit
            else "# Final Evaluation — NVIT Vietnamese Command Assistant"
        ),
        "",
    ]
    if post_audit:
        lines.extend(
            [
                "> These results rerun the same 384 test samples inspected during late error analysis.",
                "> Some failures influenced runtime rules, so this is a post-fix regression run,",
                "> **not an independent holdout**. The model and dataset were not fitted using test labels.",
            ]
        )
    else:
        lines.extend(
            [
                "> This is the final evaluation on the locked holdout. The test set was not used to select",
                "> a model, threshold, rule, or dataset change after viewing the results.",
            ]
        )
    lines.extend(
        [
            "",
            "## Snapshot and method",
            "",
            f"- Test samples: **{metrics['total_samples']}**.",
            f"- Test SHA-256: `{report['test_snapshot']['test_set_sha256']}`.",
            f"- Evaluation stage: `{provenance['stage']}`; independent holdout: "
            f"**{'yes' if provenance['independent_holdout'] else 'no'}**.",
            "- Inputs are text transcripts; runtime receives no gold region label.",
            "- Runtime uses TF-IDF + Logistic Regression fitted on train + validation.",
            "- Evaluation uses mock actions and does not call a network or device.",
            "- Regional groups contain lexical/template variants, not an audio accent benchmark.",
            "",
        ]
    )
    lines.extend(
        [
            "## Current results",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Raw-model intent accuracy | {_percent(raw['accuracy'])} |",
            f"| Raw-model intent macro-F1 | {_percent(raw['macro_f1'])} |",
            f"| Runtime intent accuracy | {_percent(runtime['accuracy'])} |",
            f"| Runtime coverage | {_percent(runtime['coverage'])} |",
            f"| Runtime selective accuracy | {_percent(runtime['selective_accuracy'])} |",
            f"| Oracle slot exact match | {_percent(oracle_slots['exact_match'])} |",
            f"| Oracle slot micro-F1 | {_percent(oracle_slots['micro']['f1'])} |",
            f"| End-to-end slot exact match | {_percent(end_to_end_slots['exact_match'])} |",
            f"| End-to-end slot micro-F1 | {_percent(end_to_end_slots['micro']['f1'])} |",
            f"| Semantic frame exact match | {_percent(metrics['semantic_frame_exact_match'])} |",
            f"| Mock action generation rate | {_percent(metrics['action_execution_rate'])} |",
            f"| Full pipeline success (mock action) | {_percent(metrics['full_command_success'])} |",
            "",
            "Raw-model intent is the classifier output before confidence, boundary, and safety logic. Runtime",
            "intent is the user-facing decision and may be `unknown` when execution is rejected. In this run,",
            "action generation means that the mock router produced a payload, not that a real device acted.",
            "",
            "## Raw-model intent by class",
            "",
            "| Intent | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in raw["labels"]:
        row = raw["per_label"][label]
        lines.append(
            f"| `{label}` | {_percent(row['precision'])} | {_percent(row['recall'])} | "
            f"{_percent(row['f1-score'])} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            "## End-to-end slots by type",
            "",
            "| Slot | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for slot_name, row in end_to_end_slots["per_slot"].items():
        lines.append(
            f"| `{slot_name}` | {_percent(row['precision'])} | {_percent(row['recall'])} | "
            f"{_percent(row['f1'])} | {row['support']} |"
        )

    for dimension in (
        "region",
        "source",
        "variant_type",
        "annotation_quality",
    ):
        lines.extend(
            [
                "",
                f"## Breakdown: {dimension}",
                "",
                "| Group | N | Runtime intent acc. | E2E slot exact | Frame exact | Full success |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for value, row in metrics["breakdown"][dimension].items():
            lines.append(
                f"| `{value}` | {row['total']} | {_percent(row['intent_accuracy'])} | "
                f"{_percent(row['end_to_end_slot_exact_match'])} | "
                f"{_percent(row['semantic_frame_exact_match'])} | "
                f"{_percent(row['full_command_success'])} |"
            )

    latency = metrics["latency"]
    memory = report["performance"]["memory_mb"]
    failure_reasons = Counter("+".join(failure["reasons"]) for failure in metrics["failures"])
    gate_reasons = Counter(
        feature.removeprefix("action_gate:")
        for failure in metrics["failures"]
        for feature in failure["matched_features"]
        if feature.startswith("action_gate:")
    )
    lines.extend(
        [
            "",
            "## Local performance",
            "",
            f"- Pipeline build: **{report['performance']['pipeline_build_ms']:.2f} ms**.",
            f"- First request: **{latency['first_request_ms']:.2f} ms**.",
            f"- Median / p95 / p99: **{latency['median_ms']:.2f} / "
            f"{latency['p95_ms']:.2f} / {latency['p99_ms']:.2f} ms**.",
            f"- Sequential throughput: **{latency['sequential_throughput_commands_per_second']:.2f} "
            "commands/second**.",
            f"- RSS before load / after load / after evaluation: **{memory['before_load']:.2f} / "
            f"{memory['after_load']:.2f} / {memory['after_evaluation']:.2f} MB**.",
            "- Latency measures sequential `pipeline.parse` calls. It excludes process startup, networking,",
            "  and JSON API overhead.",
            "",
            "## Error analysis",
            "",
            f"- Failed samples: **{metrics['failure_count']} / {metrics['total_samples']}**.",
        ]
    )
    if failure_reasons:
        for reason, count in sorted(failure_reasons.items()):
            lines.append(f"- `{reason}`: {count}.")
    else:
        lines.append("- No sample failed the full-command-success criterion.")
    rejected = metrics["total_samples"] - sum(metrics["action_statuses"].values())
    lines.append(f"- The pipeline generated no mock action for **{rejected}** samples.")
    for reason, count in gate_reasons.most_common():
        lines.append(f"- Action gate `{reason}` appeared in **{count}** failed samples.")
    phone_oracle = oracle_slots["per_slot"].get("phone_number")
    phone_end_to_end = end_to_end_slots["per_slot"].get("phone_number")
    if phone_oracle is not None and phone_end_to_end is not None:
        lines.append(
            f"- `phone_number`: oracle F1 {_percent(phone_oracle['f1'])}, end-to-end F1 "
            f"{_percent(phone_end_to_end['f1'])}; remaining failures occur before or after the"
        )
        lines.append("  extractor, mainly in intent or action-gate decisions rather than phone parsing.")
    limitation_lines = [
        "- Per-sample failures, confidence, slots, and matched features are kept in",
        "  `reports/final_evaluation.json` to keep this Markdown report readable.",
        "",
        "## Limitations to read with these results",
        "",
        "- Regional groups contain lexical/template variants and do not prove audio accent recognition.",
        "- Part of the test set is synthetic and is generally easier than MASSIVE.",
        "- The intent classifier is closed-set. The safety set is a development regression set, not an",
        "  independent production red-team benchmark.",
        "- Live weather, STT/TTS, and real devices are outside this final evaluation.",
    ]
    if post_audit:
        limitation_lines.extend(
            [
                "- The test set was inspected and influenced runtime rules. These results are not an",
                "  unbiased estimate for completely new data.",
            ]
        )
    else:
        limitation_lines.append("- Final test results were not used to revise the model or rules.")
    lines.extend(
        limitation_lines
        + [
            "",
            "Confusion matrix: `reports/figures/final_test_confusion_matrix.png`.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "final_evaluation.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports" / "FINAL_EVALUATION.md")
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "reports" / "figures" / "final_test_confusion_matrix.png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè output; với report đã tồn tại phải đi cùng --post-audit",
    )
    parser.add_argument(
        "--post-audit",
        action="store_true",
        help="Bắt buộc với snapshot hiện tại vì lỗi test đã được xem",
    )
    args = parser.parse_args()
    if not args.post_audit:
        parser.error("snapshot hiện tại phải được đánh giá với --post-audit")
    output_paths = (args.report, args.markdown, args.figure)
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        existing_text = ", ".join(str(path) for path in existing_outputs)
        parser.error(
            f"output final đã tồn tại; từ chối ghi đè nếu không có --overwrite: {existing_text}"
        )

    samples, test_snapshot = load_locked_test_set(ROOT / "data" / "samples")
    process = psutil.Process()
    rss_before = process.memory_info().rss / (1024 * 1024)
    build_started = perf_counter()
    pipeline = build_pipeline(ROOT, action_mode="mock")
    build_ms = (perf_counter() - build_started) * 1000.0
    rss_after_load = process.memory_info().rss / (1024 * 1024)
    metrics = evaluate_final_pipeline(pipeline, samples)
    rss_after_evaluation = process.memory_info().rss / (1024 * 1024)

    provenance = {
        "stage": "post_audit_diagnostic" if args.post_audit else "independent_holdout",
        "test_seen_before_code_freeze": args.post_audit,
        "test_failures_informed_rule_changes": args.post_audit,
        "model_or_dataset_refit_on_test": False,
        "independent_holdout": not args.post_audit,
        "interpretation": (
            "current_code_regression_not_unbiased_generalization"
            if args.post_audit
            else "unbiased_generalization_snapshot"
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "final_test_used": True,
        "test_used_for_tuning": args.post_audit,
        "evaluation_provenance": provenance,
        "methodology": {
            "input": "text_transcript",
            "region_hint_passed_to_runtime": False,
            "model_selection_split": "validation",
            "runtime_artifact_fit": "train_plus_validation",
            "action_mode": "mock",
            "network_used": False,
            "audio_evaluated": False,
        },
        "test_snapshot": test_snapshot,
        "snapshot_sha256": snapshot_hashes(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "pydantic": version("pydantic"),
        },
        "performance": {
            "pipeline_build_ms": build_ms,
            "memory_mb": {
                "before_load": rss_before,
                "after_load": rss_after_load,
                "after_evaluation": rss_after_evaluation,
                "model_load_delta": rss_after_load - rss_before,
            },
            "artifact_sizes_bytes": {
                "intent_classifier": (ROOT / "models" / "intent_classifier.joblib").stat().st_size,
                "slot_lexicon": (ROOT / "models" / "slot_lexicon.json").stat().st_size,
            },
        },
        "metrics": metrics,
        "figures": ["reports/figures/final_test_confusion_matrix.png"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_confusion_figure(report, args.figure)
    write_markdown_report(report, args.markdown)
    summary = {
        "test_set_sha256": test_snapshot["test_set_sha256"],
        "total": metrics["total_samples"],
        "intent_accuracy": metrics["raw_model_intent"]["accuracy"],
        "intent_macro_f1": metrics["raw_model_intent"]["macro_f1"],
        "runtime_intent_accuracy": metrics["runtime_intent"]["accuracy"],
        "runtime_coverage": metrics["runtime_intent"]["coverage"],
        "oracle_slot_f1": metrics["oracle_slots"]["micro"]["f1"],
        "end_to_end_slot_f1": metrics["end_to_end_slots"]["micro"]["f1"],
        "full_command_success": metrics["full_command_success"],
        "failure_count": metrics["failure_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
