"""CLI build dataset từ MASSIVE, seed cũ và template vùng miền."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.dataset_builder import build_dataset  # noqa: E402


def main() -> None:
    """Đọc tham số CLI, build dataset và in manifest UTF-8."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--massive-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_dataset(
        massive_path=args.massive_jsonl.resolve(),
        output_dir=args.output_dir.resolve(),
        templates_path=ROOT / "configs" / "data_templates.yaml",
        slot_values_path=ROOT / "configs" / "slot_values.yaml",
        old_project_seed_path=ROOT / "data" / "raw_sources" / "old_project_seed.jsonl",
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
