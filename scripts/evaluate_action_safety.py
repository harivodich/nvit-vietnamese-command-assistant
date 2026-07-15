"""Chạy development benchmark cho OOD, phủ định và thao tác chưa hỗ trợ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.eval.action_safety_evaluation import (  # noqa: E402
    evaluate_action_gate_coverage,
    evaluate_action_safety,
    load_action_safety_challenge,
)
from nvit_assistant.data_validation import read_samples  # noqa: E402
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402
from nvit_assistant.nlu.preprocessing import preprocess_sample  # noqa: E402
from nvit_assistant.nlu.slot_lexicon import sha256_file  # noqa: E402
from nvit_assistant.runtime import build_pipeline  # noqa: E402


def main() -> None:
    """Ghi report và trả exit code khác 0 nếu còn false action/failure."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge", type=Path, default=ROOT / "data" / "action_safety_challenge.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "action_safety_report.json"
    )
    args = parser.parse_args()
    challenge_path = args.challenge.resolve()
    pipeline = build_pipeline(ROOT)
    report = evaluate_action_safety(pipeline, load_action_safety_challenge(challenge_path))
    validation_path = ROOT / "data" / "samples" / "validation.jsonl"
    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    validation_samples = [
        preprocess_sample(sample, normalizer) for sample in read_samples(validation_path)
    ]
    if pipeline.action_gate is None:
        raise RuntimeError("runtime pipeline chưa cấu hình action gate")
    report["in_domain_validation_gate"] = evaluate_action_gate_coverage(
        pipeline.action_gate, pipeline.slot_extractor, validation_samples
    )
    report["methodology"] = {
        "challenge_role": "authored_development_regression",
        "validation_intent_mode": "oracle_intent",
        "test_used": False,
    }
    report["artifacts_sha256"] = {
        "challenge": sha256_file(challenge_path),
        "validation": sha256_file(validation_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["failures"] or report["in_domain_validation_gate"]["coverage"] < 0.95:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
