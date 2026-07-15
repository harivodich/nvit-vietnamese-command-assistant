"""Gộp hard-case đã review vào train theo cách idempotent và cập nhật manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nvit_assistant.schemas import DatasetSample  # noqa: E402


def read_samples(path: Path) -> list[DatasetSample]:
    """Đọc JSONL và kiểm tra từng dòng bằng contract DatasetSample."""
    return [DatasetSample.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_samples(path: Path, samples: list[DatasetSample]) -> None:
    """Ghi ổn định theo ID để chạy lại không làm thay đổi thứ tự."""
    payload = "".join(
        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for sample in sorted(samples, key=lambda item: item.id)
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def main() -> None:
    """Chỉ thay train; validation và bốn test file được đọc để thống kê, không ghi lại."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw_sources" / "intent_hard_cases.jsonl")
    args = parser.parse_args()
    additions = read_samples(args.source)
    train_path = args.data_dir / "train.jsonl"
    train = read_samples(train_path)
    write_samples(
        train_path,
        [sample for sample in train if not sample.id.startswith("hard_")] + additions,
    )

    filenames = ["train.jsonl", "validation.jsonl", "test_standard.jsonl", "test_north.jsonl", "test_central.jsonl", "test_south.jsonl"]
    buckets = {name: read_samples(args.data_dir / name) for name in filenames}
    all_samples = [sample for samples in buckets.values() for sample in samples]
    old_manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        **old_manifest,
        "total": len(all_samples),
        "files": {name: len(samples) for name, samples in buckets.items()},
        "intents": dict(sorted(Counter(sample.intent.value for sample in all_samples).items())),
        "regions": dict(sorted(Counter(sample.region.value for sample in all_samples).items())),
        "sources": dict(sorted(Counter(sample.source.value for sample in all_samples).items())),
        "annotation_quality": dict(sorted(Counter(sample.annotation_quality.value for sample in all_samples).items())),
        "augmentations": {"intent_hard_cases": {"source": str(args.source.relative_to(ROOT)), "samples": len(additions)}},
    }
    (args.data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
