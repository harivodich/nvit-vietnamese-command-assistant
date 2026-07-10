.PHONY: install test lint typecheck data validate-data

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

validate-data:
	python scripts/validate_data.py --data-dir data/samples
