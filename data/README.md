# Dataset

Core dataset dùng schema `nvit_assistant.schemas.DatasetSample` và được lưu dưới dạng UTF-8 JSONL.

## Split

- `train.jsonl`: học trọng số/rule.
- `validation.jsonl`: chọn feature, threshold và hyperparameter.
- `test_standard.jsonl`: dữ liệu tiếng Việt chuẩn, chủ yếu từ MASSIVE test partition.
- `test_north.jsonl`, `test_central.jsonl`, `test_south.jsonl`: lexical/regional test theo từng vùng.

Mọi biến thể của cùng một câu có chung `group_id` và bắt buộc nằm trong cùng split để tránh leakage.

## Rebuild

Tải MASSIVE 1.0 từ URL chính thức được ghi trong `SOURCES.md`, giải nén file `1.0/data/vi-VN.jsonl`, sau đó chạy:

```powershell
python scripts/build_dataset.py --massive-jsonl path/to/vi-VN.jsonl
python scripts/validate_data.py --data-dir data/samples
python scripts/audit_normalization.py --data-dir data/samples
python scripts/build_slot_lexicon.py
```

Dataset đã build được commit để reviewer có thể chạy validation/evaluation mà không cần tải lại nguồn ngoài.
