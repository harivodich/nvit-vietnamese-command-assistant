"""Đánh giá slot extractor trên validation đã preprocess và ghi báo cáo lỗi."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nvit_assistant.eval.slot_evaluation import (  # noqa: E402
    evaluate_slot_extractor,
)
from nvit_assistant.data_validation import read_samples  # noqa: E402
from nvit_assistant.nlu.normalizer import VietnameseNormalizer  # noqa: E402
from nvit_assistant.nlu.preprocessing import preprocess_sample  # noqa: E402
from nvit_assistant.nlu.slot_extractor import RegexSlotExtractor  # noqa: E402
from nvit_assistant.nlu.slot_lexicon import (  # noqa: E402
    sha256_file,
    validate_slot_lexicon_provenance,
)


def main() -> None:
    """Chạy metric bằng intent thật để không trộn lỗi intent vào lỗi slot."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data" / "samples" / "validation.jsonl",
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "samples" / "train.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "slot_values.yaml")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "models" / "slot_lexicon.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "slot_extraction_report.json"
    )
    args = parser.parse_args()

    validation_path = args.validation.resolve()
    train_path = args.train.resolve()
    config_path = args.config.resolve()
    lexicon_path = args.lexicon.resolve()
    normalizer_config_path = ROOT / "configs" / "regional_variants.yaml"
    validate_slot_lexicon_provenance(
        lexicon_path, train_path, normalizer_config_path
    )
    normalizer = VietnameseNormalizer(normalizer_config_path)
    extractor = RegexSlotExtractor(config_path, lexicon_path)
    samples = [preprocess_sample(sample, normalizer) for sample in read_samples(validation_path)]
    report = evaluate_slot_extractor(extractor, samples)
    report["methodology"] = {
        "split": "validation",
        "intent_mode": "oracle_intent",
        "lexicon_source": "train_only",
        "test_used": False,
    }
    report["artifacts_sha256"] = {
        "validation": sha256_file(validation_path),
        "train": sha256_file(train_path),
        "slot_values": sha256_file(config_path),
        "slot_lexicon": sha256_file(lexicon_path),
        "regional_variants": sha256_file(normalizer_config_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {key: value for key, value in report.items() if key != "failures"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
