"""Chạy benchmark normalizer và ghi report có thể kiểm tra trước khi train intent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.nlu.normalization_evaluation import (  # noqa: E402
    evaluate_normalizer,
    load_normalization_challenge,
)
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402


def main() -> None:
    """Đánh giá benchmark và trả mã lỗi nếu chưa đạt 100% benchmark đã review."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge", type=Path, default=ROOT / "data" / "normalization_challenge.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "normalization_evaluation.json"
    )
    args = parser.parse_args()

    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    report = evaluate_normalizer(normalizer, load_normalization_challenge(args.challenge.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
