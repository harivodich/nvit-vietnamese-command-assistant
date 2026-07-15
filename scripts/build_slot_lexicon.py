"""Sinh artifact lexicon slot từ train và ghi provenance chống leakage."""

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
from nvit_assistant.nlu.slot_lexicon import (  # noqa: E402
    build_slot_lexicon,
    write_slot_lexicon,
)


def main() -> None:
    """Đọc train nguồn, tuyệt đối không nhận validation/test làm input mặc định."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "samples" / "train.jsonl")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models" / "slot_lexicon.json"
    )
    args = parser.parse_args()

    normalizer_config_path = ROOT / "configs" / "regional_variants.yaml"
    normalizer = VietnameseNormalizer(normalizer_config_path)
    payload = build_slot_lexicon(
        args.train.resolve(), normalizer, normalizer_config_path
    )
    write_slot_lexicon(payload, args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
