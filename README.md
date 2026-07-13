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

Dataset hiện có 2.388 sample: MASSIVE tiếng Việt đã lọc, template lexical Bắc/Trung/Nam và 22 hard-case đã review từ error analysis. Validation/test giữ nguyên khi bổ sung hard-case vào train. Chi tiết license, provenance và các nguồn bị loại nằm trong `data/SOURCES.md`.

```text
train:       1.645
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

Đã tạo dataset JSONL với 2.388 mẫu cho đủ 5 intent. Nguồn dữ liệu gồm
MASSIVE `vi-VN` đã lọc chất lượng và template lexical Bắc/Trung/Nam do dự án kiểm soát.
Mỗi mẫu có intent, slot, region, nguồn gốc, loại biến thể và chất lượng annotation rõ ràng.

Dataset được chia thành train 1.645, validation 352 và test 391. Validator kiểm tra schema,
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
hiện tại có 2.357 output preprocess từ 2.388 sample gốc. Review native speaker và audio không thể
tự tạo bằng code; protocol, consent và tiêu chí acceptance ở `data/NATIVE_REVIEW_PROTOCOL.md`.

Thử thủ công:

```powershell
python scripts/normalize_text.py "Bữa ni ở Huế trời răng rồi hỉ"
python scripts/audit_normalization.py --data-dir data/samples
python scripts/evaluate_normalizer.py
python scripts/preprocess_dataset.py --input-dir data/samples
```

### Ngày 4 — Intent classifier baseline

Đã huấn luyện baseline TF-IDF word/character n-gram kết hợp Logistic Regression. Ba candidate
được fit trên train và chọn duy nhất bằng macro-F1 validation; test bị khóa hoàn toàn khi chọn
model. Candidate `word_1_2_char_3_5` đạt macro-F1 0.9654 và accuracy 0.9688 trên 352 sample
validation. Artifact, label map, candidate report và từng lỗi validation được lưu để phân tích.
Report có precision, recall, F1 theo từng intent; macro/weighted F1; confusion matrix; ROC/PR curve
one-vs-rest; log-loss, Brier score, ECE và reliability curve. Các metric và biểu đồ này đều là
validation; chưa công bố metric test ở phase này.

Một benchmark semantic độc lập dùng embedding local `multilingual-e5-small` (384 chiều) và Logistic
Regression cũng được chạy trên đúng validation. Kết quả macro-F1 0.8949, thấp hơn TF-IDF 0.9654;
đặc biệt recall `set_alarm` chỉ 0.5789. Vì vậy runtime giữ TF-IDF + Logistic Regression; E5 được
giữ như thí nghiệm tái lập và bằng chứng lựa chọn model, không được đưa vào runtime chỉ vì là
transformer. Xem `reports/model_comparison_report.json` và `reports/semantic_intent_report.json`.

```powershell
python scripts/train_intent.py
```

Fine-tune transformer chưa phải lựa chọn mặc định: chỉ thực hiện sau khi phân tích failure của
baseline và có benchmark native-speaker/audio phù hợp. Ngày 5 tiếp tục slot extraction, sau đó
pipeline, API, evaluation cuối cùng và ASR.

### Ngày 5 — Trích xuất slot và pipeline NLU

Đã xây dựng `RegexSlotExtractor` theo từng intent để không gán nhầm slot giữa các loại lệnh.
`datetime` được nhận diện bằng từ điển và regex thời gian; số điện thoại được nhận diện bằng
regex; địa điểm, liên hệ, bài hát và nghệ sĩ dùng phép dò biên từ với ưu tiên cụm dài nhất.
Riêng `reminder_text` được tạo bằng cách loại trigger nhắc việc, thời gian và tiểu từ lịch sự
khỏi câu đã chuẩn hóa. Mỗi slot kèm dấu vết match để có thể giải thích kết quả.

`NLUPipeline` nối một luồng thống nhất cho runtime:

```text
ParseRequest
  -> VietnameseNormalizer
  -> TF-IDF + Logistic Regression intent classifier
  -> RegexSlotExtractor theo intent vừa dự đoán
  -> ParseResult gồm intent, confidence, slots và matched_features
```

Các action thật, câu phản hồi tự nhiên, CLI và FastAPI chưa nằm trong core này; chúng thuộc Ngày 6.
Ngày 5 có hard-case cho thời gian dạng số/chữ, “rưỡi”, “kém”, số điện thoại viết/đọc, transcript
không dấu, entity ngoài catalog và trường hợp không được sinh entity giả từ playlist. Pipeline kiểm
tra nhóm slot bắt buộc; nếu báo thức, lời nhắc hoặc cuộc gọi thiếu đối tượng cần thiết thì trả về
yêu cầu bổ sung thay vì giả vờ đã hiểu đủ.

Extractor được đánh giá độc lập trên 352 mẫu validation bằng intent thật, để lỗi intent không làm
nhiễu phép đo slot. Kết quả hiện tại: exact-match 0.9460, micro precision 0.9575, recall 0.9721 và
F1 0.9647. Location/phone đạt F1 1.0; reminder 0.9851; artist 0.9655; datetime 0.9421; song 0.9.
19 failure còn lại được giữ trong report, chủ yếu là ranh giới annotation dịch từ MASSIVE như
nhãn `năm` trong câu `năm giờ`; không thêm rule riêng để học thuộc các sai khác surface này.

```powershell
python scripts/evaluate_slots.py
```

Báo cáo đầy đủ nằm ở `reports/slot_extraction_report.json`. Đây là metric validation để phát triển;
test vẫn được khóa đến đánh giá cuối cùng ở Ngày 7.
