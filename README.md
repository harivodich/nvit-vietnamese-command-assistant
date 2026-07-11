# NVIT Vietnamese Command Assistant

Repository triển khai hệ thống hiểu lệnh nói tiếng Việt theo hướng **text-first NLU**. Dự án được xây dựng theo từng phase, với kiểm thử và đánh giá rõ ràng ở mỗi giai đoạn.

## Mục tiêu cuối cùng

Input chính là transcript tiếng Việt; audio chỉ là phần mở rộng sau cùng. Hệ thống sẽ trả về intent, slots, confidence, mock action và câu phản hồi. Core không dùng LLM runtime hay external API.

Intent dự kiến: `set_reminder`, `set_alarm`, `ask_weather`, `play_music`, `call_contact`.

Slots dự kiến: `datetime`, `location`, `song`, `artist`, `contact_name`, `phone_number`, `reminder_text`.

```text
Text transcript (hoặc audio adapter)
  -> Vietnamese normalization + regional handling
  -> intent classification
  -> slot extraction
  -> confidence gate
  -> mock action + CLI/API response
```

## Kiến trúc

```text
configs/                         cấu hình intent, vùng miền và data templates
data/raw_sources/                nguồn tham khảo chưa gán nhãn
data/samples/                    JSONL train/validation/test đã được kiểm tra
models/                          model artifact cục bộ
reports/                         báo cáo evaluation
scripts/                         generate, validate, train, evaluate, demo
src/nvit_assistant/
  actions/                       mock action router
  asr/                           optional audio-to-text adapter
  eval/                          metric và evaluation runner
  nlu/                           normalizer, classifier, slots, pipeline
tests/                           unit và integration tests
```

## Kế hoạch triển khai

1. Contract: schema, kiểu dữ liệu và schema tests.
2. Config + dataset nhỏ + JSONL validator.
3. Vietnamese normalization và regional variants.
4. Rule-based intent classifier, sau đó classifier trainable nhẹ.
5. Regex slot extraction và end-to-end pipeline.
6. Mock actions, CLI, FastAPI và integration tests.
7. Metrics/evaluation; ASR chỉ thêm khi NLU core ổn định.

Mỗi phase xác định rõ input/output, implementation, test và tiêu chí hoàn thành trước khi chuyển sang phase tiếp theo.

## Dataset

Dataset hiện có 2.366 sample, được build từ MASSIVE tiếng Việt đã lọc chất lượng và template lexical Bắc/Trung/Nam có biến thể không dấu. Seed reminder đã review từ project cũ được giữ làm nguồn audit; nếu gần với test MASSIVE, validator sẽ loại khỏi final split để bảo toàn tính độc lập. Chi tiết license, provenance và các nguồn bị loại nằm trong `data/SOURCES.md`.

```text
train:       1.623
validation:    352
test:          391
```

Build lại dataset sau khi tải MASSIVE 1.0 và giải nén `1.0/data/vi-VN.jsonl`:

```powershell
python scripts/build_dataset.py --massive-jsonl path\to\vi-VN.jsonl
python scripts/validate_data.py --data-dir data\samples
```

Regional set hiện kiểm tra lexical variation trong transcript, chưa phải đánh giá audio accent của người nói thật. Giới hạn này được giữ nguyên trong validation report và `DECISIONS.md`.

## Trạng thái triển khai

### Ngày 1 — Contract và cấu hình nền tảng

Đã hoàn thành contract dùng chung cho toàn bộ hệ thống: 5 intent, 7 slot, vùng miền,
nguồn dữ liệu, kiểu biến thể và chất lượng gán nhãn. Đồng thời đã tạo các file cấu hình
giá trị slot/biến thể vùng miền, kiểm tra schema bằng test và ghi các quyết định kỹ thuật
ban đầu trong `DECISIONS.md`.

Commit liên quan: `Hoàn thiện contract, cấu hình nền tảng`.

### Ngày 2 — Xây dựng và kiểm định dataset

Đã tạo dataset JSONL có thể build lại với 2.366 mẫu cho đủ 5 intent. Nguồn dữ liệu gồm
MASSIVE `vi-VN` đã lọc chất lượng và template lexical Bắc/Trung/Nam do dự án kiểm soát.
Mỗi mẫu có intent, slot, region, nguồn gốc, loại biến thể và chất lượng annotation rõ ràng.

Dataset được chia thành train 1.623, validation 352 và test 391. Validator kiểm tra schema,
slot có xuất hiện trong câu, trùng nguyên văn, trùng khi bỏ dấu, leakage `group_id` và câu
gần giống giữa các split. Template cùng family và các biến thể cùng nhóm luôn nằm trong một
split; các group quá giống split ưu tiên hơn sẽ bị loại. Build lại từ cùng nguồn cho kết quả
checksum giống nhau.

Đã chạy `pytest` (18 test), Ruff, mypy và validator trước khi push. Báo cáo hiện tại không có
lỗi; giới hạn còn lại là dữ liệu vùng miền mới là text template, chưa phải transcript giọng nói
thật.

Commit liên quan: `Xây dựng và hoàn thiện dataset`.

### Ngày 3 — Chuẩn hóa tiếng Việt và biến thể vùng miền

Đã bổ sung normalizer rule-based đọc `configs/regional_variants.yaml`. Thành phần này chuẩn
hóa Unicode, khoảng trắng, lỗi STT có trong từ điển và biến thể từ vựng Bắc/Trung/Nam. Cụm dài
được thay trước cụm ngắn; kết quả kèm `matched_variants` để truy vết rule. Region chỉ được suy
luận khi tín hiệu rõ ràng, còn tín hiệu mâu thuẫn trả về `unknown` thay vì đoán bừa.

Tiểu từ cuối câu như `hỉ`, `hen`, `nghen` được xử lý theo ngữ cảnh: câu hỏi chuẩn về `nhỉ`,
câu yêu cầu/nhắc nhở chuẩn về `nhé`, còn không đủ tín hiệu thì giữ nguyên. Dataset JSONL gốc
không bị viết đè; script audit áp dụng normalizer lên toàn bộ sample để phát hiện collision sau
preprocess. Ngày train và runtime sau này đều phải gọi cùng normalizer trước classifier.

Normalizer được kiểm tra bằng 34 câu command-domain độc lập ở
`data/normalization_challenge.jsonl`, bao phủ ba vùng, lỗi STT, tiểu từ ngữ cảnh và cả năm
intent. Nguồn, license và giới hạn tái sử dụng nằm trong `data/NORMALIZATION_SOURCES.md`.

Preprocess tạo artifact tái lập trong `data/preprocessed/` nhưng không commit artifact này: JSONL
gốc vẫn là nguồn audit. Train/validation được deduplicate theo `normalized_text` để không tăng
trọng số vì câu vùng miền trở thành giống nhau; test giữ nguyên bề mặt để đo độ bền thật. Report
hiện tại có 2.335 input preprocess từ 2.366 sample gốc. Review native speaker và audio không thể
tự tạo bằng code; protocol, consent và tiêu chí acceptance ở `data/NATIVE_REVIEW_PROTOCOL.md`.

Thử thủ công:

```powershell
python scripts/normalize_text.py "Bữa ni ở Huế trời răng rồi hỉ"
python scripts/audit_normalization.py --data-dir data/samples
python scripts/evaluate_normalizer.py
python scripts/preprocess_dataset.py --input-dir data/samples
```

Ngày 4 trở đi mới bắt đầu intent classifier, sau đó slot extraction, pipeline, API, evaluation
và ASR. Chưa có model được huấn luyện hay kết quả đánh giá mô hình để tránh nhầm lẫn giữa việc
chuẩn bị dữ liệu/normalization và việc hoàn thành hệ thống NLU.
