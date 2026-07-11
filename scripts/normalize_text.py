"""CLI thử normalizer tiếng Việt với biến thể vùng miền và lỗi STT."""

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
from nvit_assistant.schemas import Region  # noqa: E402


def main() -> None:
    """Đọc câu lệnh, chuẩn hóa và in JSON để kiểm tra thủ công."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--region-hint", choices=[region.value for region in Region])
    args = parser.parse_args()

    hint = Region(args.region_hint) if args.region_hint else None
    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    result = normalizer.normalize(args.text, hint)
    print(
        json.dumps(
            {
                "original_text": result.original_text,
                "normalized_text": result.normalized_text,
                "region": result.region.value,
                "matched_variants": result.matched_variants,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
