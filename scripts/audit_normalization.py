"""Audit ảnh hưởng của normalizer trên dataset mà không làm thay đổi JSONL gốc."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.data_validation import (  # noqa: E402
    find_cross_split_near_duplicates,
    read_samples,
    split_name,
)
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402


def audit_normalization(data_dir: Path, normalizer: VietnameseNormalizer) -> dict[str, object]:
    """Đếm câu thay đổi và tìm normalized text xuất hiện ở nhiều group/split."""
    changed = 0
    total = 0
    region_counts: Counter[str] = Counter()
    locations_by_text: dict[str, set[tuple[str, str]]] = defaultdict(set)
    records = []
    for path in sorted(data_dir.glob("*.jsonl")):
        for sample in read_samples(path):
            total += 1
            result = normalizer.normalize(sample.text, sample.region)
            changed += result.normalized_text != sample.text.casefold()
            region_counts[result.region.value] += 1
            logical_split = split_name(path.name)
            records.append((logical_split, sample))
            locations_by_text[result.normalized_text].add((logical_split, sample.id))

    collisions = {
        text: sorted(f"{split}:{group_id}" for split, group_id in locations)
        for text, locations in locations_by_text.items()
        if len(locations) > 1
    }
    cross_split_collisions = {
        text: locations
        for text, locations in collisions.items()
        if len({location.split(":", maxsplit=1)[0] for location in locations}) > 1
    }
    same_split_collisions = {
        text: locations for text, locations in collisions.items() if text not in cross_split_collisions
    }
    near_duplicates = find_cross_split_near_duplicates(records, normalizer=normalizer)
    return {
        "total": total,
        "changed": changed,
        "unchanged": total - changed,
        "detected_regions": dict(sorted(region_counts.items())),
        "normalized_collisions": len(collisions),
        "same_split_collisions": same_split_collisions,
        "cross_split_collisions": cross_split_collisions,
        "near_similar_cross_split": [
            {
                "left_split": left_split,
                "left_id": left_id,
                "right_split": right_split,
                "right_id": right_id,
                "score": score,
            }
            for left_split, left_id, right_split, right_id, score in near_duplicates
        ],
    }


def main() -> None:
    """Chạy audit và ghi report JSON mà không thay đổi dataset gốc."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "normalization_audit.json"
    )
    args = parser.parse_args()

    normalizer = VietnameseNormalizer(ROOT / "configs" / "regional_variants.yaml")
    report = audit_normalization(args.data_dir.resolve(), normalizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if (
        report["same_split_collisions"]
        or report["cross_split_collisions"]
        or report["near_similar_cross_split"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
