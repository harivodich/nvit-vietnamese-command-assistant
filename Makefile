.PHONY: install test lint typecheck data augment-intent validate-data normalize audit-normalization evaluate-normalizer preprocess train-intent train-semantic-intent

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check .

typecheck:
	mypy src

data:
	python scripts/build_dataset.py --massive-jsonl "$(MASSIVE_JSONL)"
	python scripts/validate_data.py --data-dir data/samples

augment-intent:
	python scripts/augment_intent_hard_cases.py
	python scripts/validate_data.py --data-dir data/samples

validate-data:
	python scripts/validate_data.py --data-dir data/samples

normalize:
	python scripts/normalize_text.py "$(TEXT)"

audit-normalization:
	python scripts/audit_normalization.py --data-dir data/samples

evaluate-normalizer:
	python scripts/evaluate_normalizer.py

preprocess:
	python scripts/preprocess_dataset.py --input-dir data/samples

train-intent:
	python scripts/train_intent.py

train-semantic-intent:
	python scripts/train_semantic_intent.py --encoder-dir E:/models/multilingual-e5-small
