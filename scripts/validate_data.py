"""CLI kiểm tra chất lượng toàn bộ JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.data_validation import validate_dataset_dir  # noqa: E402
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402


def main() -> None:
    """Chạy validator, in report UTF-8 và trả exit code 1 nếu có lỗi."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "data_validation_report.json",
    )
    args = parser.parse_args()

    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    report = validate_dataset_dir(args.data_dir.resolve(), normalizer=normalizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
