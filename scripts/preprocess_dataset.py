"""Tạo artifact preprocess cho train intent mà không ghi đè JSONL nguồn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402
from nvit_assistant.nlu.preprocessing import preprocess_dataset  # noqa: E402


def main() -> None:
    """Build artifact preprocess và ghi manifest để train có thể tái lập."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "preprocessed")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "preprocessing_report.json"
    )
    args = parser.parse_args()

    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    report = preprocess_dataset(args.input_dir.resolve(), args.output_dir.resolve(), normalizer)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
