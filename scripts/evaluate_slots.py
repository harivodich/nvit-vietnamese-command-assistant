"""Đánh giá slot extractor trên validation đã preprocess và ghi báo cáo lỗi."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.eval.slot_evaluation import (  # noqa: E402
    evaluate_slot_extractor,
    load_preprocessed_samples,
)
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor  # noqa: E402


def main() -> None:
    """Chạy metric bằng intent thật để không trộn lỗi intent vào lỗi slot."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data" / "preprocessed" / "validation.jsonl",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "slot_values.yaml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "slot_extraction_report.json"
    )
    args = parser.parse_args()

    extractor = RegexSlotExtractor(args.config.resolve())
    samples = load_preprocessed_samples(args.validation.resolve())
    report = evaluate_slot_extractor(extractor, samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "failures"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
