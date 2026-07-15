.PHONY: install test lint typecheck data validate-data normalize audit-normalization evaluate-normalizer preprocess build-slot-lexicon train-intent train-semantic-intent evaluate-slots evaluate-confidence evaluate-action-safety run serve

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
	python scripts/audit_normalization.py --data-dir data/samples
	python scripts/build_slot_lexicon.py

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

build-slot-lexicon:
	python scripts/build_slot_lexicon.py

train-intent:
	python scripts/train_intent.py

train-semantic-intent:
	python -c "import sys; sys.exit(0 if sys.argv[1] else 'Thiếu E5_MODEL_DIR')" "$(E5_MODEL_DIR)"
	python scripts/train_semantic_intent.py --encoder-dir "$(E5_MODEL_DIR)"

evaluate-slots:
	python scripts/evaluate_slots.py

evaluate-confidence:
	python scripts/evaluate_confidence_gate.py

evaluate-action-safety:
	python scripts/evaluate_action_safety.py

run:
	python scripts/run_assistant.py "$(TEXT)"

serve:
	python -m uvicorn nvit_assistant.api:app --app-dir src --host 127.0.0.1 --port 8000
