"""Chạy trợ lý text-first từ command line bằng đúng runtime của API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.runtime import build_pipeline  # noqa: E402
from nvit_assistant.schemas import ParseRequest, Region  # noqa: E402


def main() -> None:
    """Nhận một câu lệnh, chạy end-to-end và in response hoặc JSON để debug."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Hiểu lệnh tiếng Việt và giả lập action")
    parser.add_argument("text", help="Câu lệnh tiếng Việt cần xử lý")
    parser.add_argument(
        "--region",
        choices=[region.value for region in Region if region is not Region.UNKNOWN],
        help="Gợi ý vùng miền nếu đã biết",
    )
    parser.add_argument("--json", action="store_true", help="In toàn bộ ParseResult dạng JSON")
    args = parser.parse_args()

    pipeline = build_pipeline(ROOT)
    region_hint = Region(args.region) if args.region else None
    result = pipeline.parse(ParseRequest(text=args.text, region_hint=region_hint))
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(result.response)


if __name__ == "__main__":
    main()
