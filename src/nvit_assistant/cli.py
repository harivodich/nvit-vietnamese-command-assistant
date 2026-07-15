"""Console entrypoint cho trợ lý text-first dùng chung runtime với FastAPI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from nvit_assistant.runtime import build_pipeline
from nvit_assistant.schemas import ParseRequest, Region


def build_argument_parser() -> argparse.ArgumentParser:
    """Khai báo CLI ở một nơi để console script và test dùng cùng contract."""
    parser = argparse.ArgumentParser(description="Hiểu lệnh tiếng Việt và giả lập action")
    parser.add_argument("text", help="Câu lệnh tiếng Việt cần xử lý")
    parser.add_argument(
        "--region",
        choices=[region.value for region in Region if region is not Region.UNKNOWN],
        help="Gợi ý vùng miền nếu đã biết",
    )
    parser.add_argument("--json", action="store_true", help="In toàn bộ ParseResult dạng JSON")
    parser.add_argument(
        "--live-weather",
        action="store_true",
        help="Gọi Open-Meteo thật; các action thiết bị khác vẫn chỉ giả lập",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Repo root; mặc định tìm từ cwd hoặc biến NVIT_PROJECT_ROOT",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Nhận một câu lệnh, chạy end-to-end và in response hoặc JSON để debug."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_argument_parser().parse_args(argv)
    pipeline = build_pipeline(
        args.project_root,
        action_mode="live-weather" if args.live_weather else None,
    )
    region_hint = Region(args.region) if args.region else None
    result = pipeline.parse(ParseRequest(text=args.text, region_hint=region_hint))
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(result.response)
