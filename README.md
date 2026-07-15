# NVIT Vietnamese Command Assistant

Repository triển khai hệ thống hiểu lệnh nói tiếng Việt theo hướng **text-first NLU**. Dự án được xây
dựng theo từng phase, với kiểm thử và đánh giá rõ ràng ở mỗi giai đoạn. Trạng thái hiện tại nhận
transcript; STT/TTS và action trên thiết bị thật chưa thuộc đường chạy chính.

## Mục tiêu cuối cùng

Input chính là transcript tiếng Việt; audio là phần mở rộng sau cùng. Hệ thống trả về intent, slots,
confidence, mock action và câu phản hồi. Core không dùng LLM runtime hay external inference API.

Intent trong contract: `set_reminder`, `set_alarm`, `ask_weather`, `play_music`, `call_contact`.

Slots trong contract: `datetime`, `location`, `song`, `artist`, `contact_name`, `phone_number`,
`reminder_text`.

```text
Text transcript
  -> Vietnamese normalization + regional handling
  -> intent classification
  -> confidence gate
  -> slot extraction theo intent được chấp nhận
  -> action-safety gate
  -> required-slot check
  -> mock action, clarification hoặc safe rejection
```

## Kiến trúc

```text
configs/                         cấu hình intent, vùng miền và data templates
data/raw_sources/                nguồn đầu vào đã tuyển chọn/gán nhãn để build và audit
data/samples/                    JSONL train/validation/test đã được kiểm tra
models/                          model artifact cục bộ
reports/                         báo cáo evaluation
scripts/                         generate, validate, train, evaluate, demo
src/nvit_assistant/
  actions/                       mock action router
  asr/                           placeholder cho adapter audio-to-text, chưa có implementation
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

## Cài đặt và chạy demo

Project khai báo Python `>=3.11`; môi trường local đã được kiểm tra bằng Python 3.12, còn CI chạy cả
Python 3.11 và 3.12. scikit-learn được khóa ở 1.9.0 vì gắn trực tiếp với artifact; NumPy và joblib
dùng khoảng phiên bản tương thích.
Cài dependency runtime cho TF-IDF, CLI và API:

```powershell
python -m pip install -e .
```

Cài thêm công cụ test/lint/type check khi phát triển:

```powershell
python -m pip install -e ".[dev]"
```

Thí nghiệm E5 không nằm trong runtime mặc định. Chỉ cài extra nặng này khi cần tái tạo semantic
benchmark:

```powershell
python -m pip install -e ".[semantic]"
```

Chạy một câu lệnh end-to-end bằng CLI:

```powershell
nvit-assistant "gọi cho mẹ"
nvit-assistant --json "Bữa ni ở Huế trời răng rồi hỉ"

# Wrapper tương đương khi chạy trực tiếp từ source tree
python scripts/run_assistant.py "gọi cho mẹ"
python scripts/run_assistant.py --json "Bữa ni ở Huế trời răng rồi hỉ"
```

Trước khi gọi `joblib.load`, runtime kiểm metadata và SHA-256 của artifact trong tệp provenance do pipeline
huấn luyện tạo ra. Kiểm tra này phát hiện file bị thay thế hoặc metadata không khớp, nhưng không biến
joblib thành định dạng an toàn: tuyệt đối không nhận và load artifact do người dùng tải lên vì quá
trình deserialize có thể thực thi payload.

Khởi động API:

```powershell
python -m uvicorn nvit_assistant.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Swagger UI ở `http://127.0.0.1:8000/docs`; health check ở `/health`; `POST /parse` nhận:

```json
{
  "text": "nhắc tôi uống thuốc lúc 8 giờ",
  "region_hint": "standard"
}
```

Mọi action hiện đều có `status=mocked`: demo không gọi điện, đặt báo thức, truy vấn thời tiết
hay phát nhạc thật.

Kiểm tra project:

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```

## Dataset

Dataset sau audit hiện có 2.094 sample: MASSIVE tiếng Việt đã lọc, template lexical Bắc/Trung/Nam và
19 hard-case còn lại sau kiểm soát leakage. Nhãn `annotation_quality=reviewed` của 19 câu này chỉ có
nghĩa annotation đã được rà soát nội bộ, không đồng nghĩa native-speaker review. Chi tiết license,
provenance và các nguồn bị loại nằm trong `data/SOURCES.md`.

```text
train:       1.363
validation:    347
test:          384 = standard 174 + Bắc 70 + Trung 70 + Nam 70
```

Build lại dataset sau khi tải MASSIVE 1.0 và giải nén `1.0/data/vi-VN.jsonl`:

```powershell
python scripts/build_dataset.py --massive-jsonl path\to\vi-VN.jsonl
python scripts/validate_data.py --data-dir data\samples
python scripts/audit_normalization.py --data-dir data\samples
python scripts/build_slot_lexicon.py
```

Slot lexicon phải được build **sau** dataset để chỉ học catalog từ train hiện tại, không đọc
validation/test. Nếu data thay đổi, cần chạy lại preprocessing, huấn luyện và các report phụ thuộc.

Test chưa được dùng để tính metric hoặc chọn model/rule/threshold. Sau đợt audit dữ liệu, định nghĩa
test hiện tại được khóa bằng SHA-256
`47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a`; mọi thay đổi hash phải được
coi là thay đổi benchmark và ghi lại trước khi đánh giá cuối.

Regional set hiện chỉ kiểm tra lexical variation trong text, chưa phải đánh giá audio accent của người
nói thật. Đến trước Ngày 7, số sample/audio được native speaker review theo protocol chính thức là 0.
Giới hạn này được ghi trong `data/NORMALIZATION_SOURCES.md` và `DECISIONS.md`.

## Trạng thái triển khai

### Ngày 1 — Contract và cấu hình nền tảng

Đã hoàn thành contract dùng chung cho toàn bộ hệ thống: 5 intent, 7 slot, vùng miền,
nguồn dữ liệu, kiểu biến thể và chất lượng gán nhãn. Đồng thời đã tạo các file cấu hình
giá trị slot/biến thể vùng miền, kiểm tra schema bằng test và ghi các quyết định kỹ thuật
ban đầu trong `DECISIONS.md`.

Commit liên quan: `Hoàn thiện contract, cấu hình nền tảng`.

### Ngày 2 — Xây dựng và kiểm định dataset

Đã tạo dataset JSONL với 2.094 mẫu cho đủ 5 intent. Nguồn dữ liệu gồm
MASSIVE `vi-VN` đã lọc chất lượng và template lexical Bắc/Trung/Nam do dự án kiểm soát.
Mỗi mẫu có intent, slot, region, nguồn gốc, loại biến thể và chất lượng annotation rõ ràng.

Dataset được chia thành train 1.363, validation 347 và test 384. Test gồm 174 câu standard và 70 câu
cho mỗi vùng Bắc/Trung/Nam. Validator kiểm tra schema,
slot có xuất hiện trong câu, trùng nguyên văn, trùng khi bỏ dấu, leakage `group_id` và câu
gần giống giữa các split. Template cùng family và các biến thể cùng nhóm luôn nằm trong một
split; các group quá giống split ưu tiên hơn sẽ bị loại. Build lại từ cùng nguồn cho kết quả
checksum giống nhau.

Validator, preprocessing, test, Ruff và mypy là các cổng kiểm tra trước khi chuyển phase. Giới hạn
còn lại là dữ liệu vùng miền mới là text template, chưa phải transcript giọng nói thật. Test chưa
được dùng để chọn model; hash của định nghĩa test sau audit được khóa như phần Dataset phía trên.

Commit liên quan: `Xây dựng và hoàn thiện dataset`.

### Ngày 3 — Chuẩn hóa tiếng Việt và biến thể vùng miền

Đã bổ sung normalizer rule-based đọc `configs/regional_variants.yaml`. Thành phần này chuẩn
hóa Unicode, khoảng trắng, lỗi STT có trong từ điển và biến thể từ vựng Bắc/Trung/Nam. Cụm dài
được thay trước cụm ngắn; kết quả kèm `matched_variants` để truy vết rule. Region chỉ được suy
luận khi tín hiệu rõ ràng, còn tín hiệu mâu thuẫn trả về `unknown` thay vì đoán bừa.

Tiểu từ cuối câu như `hỉ`, `hen`, `nghen` được xử lý theo ngữ cảnh: câu hỏi chuẩn về `nhỉ`,
câu yêu cầu/nhắc nhở chuẩn về `nhé`, còn không đủ tín hiệu thì giữ nguyên. Dataset JSONL gốc
không bị viết đè; script audit áp dụng normalizer lên toàn bộ sample để phát hiện collision sau
preprocess. Khi train và khi chạy runtime đều phải gọi cùng normalizer trước classifier.

Normalizer được kiểm tra bằng 35 câu command-domain ở `data/normalization_challenge.jsonl`, bao phủ
ba vùng, lỗi STT, tiểu từ ngữ cảnh và cả năm intent. Đây là development regression set do dự án tự
biên soạn và tách khỏi train/validation/test intent, không phải benchmark độc lập từ người dùng thật.
Nguồn, license và giới hạn tái sử dụng nằm trong `data/NORMALIZATION_SOURCES.md`.

Preprocess tạo artifact tái lập trong `data/preprocessed/` nhưng không commit artifact này: JSONL
gốc vẫn là nguồn audit. Builder đã xử lý trùng và gần giống giữa split trước đó; preprocessing áp
dụng cùng normalizer cho cả sáu file và chỉ cho phép deduplicate train/validation, không deduplicate
test. Với snapshot hiện tại không còn collision mới: 2.094 input tạo 2.094 output, `dropped=0`.
Review native speaker và audio không thể tự tạo bằng code; protocol, consent và tiêu chí acceptance
ở `data/NATIVE_REVIEW_PROTOCOL.md`.

Thử thủ công:

```powershell
python scripts/normalize_text.py "Bữa ni ở Huế trời răng rồi hỉ"
python scripts/audit_normalization.py --data-dir data/samples
python scripts/evaluate_normalizer.py
python scripts/preprocess_dataset.py --input-dir data/samples
```

### Ngày 4 — Intent classifier baseline

Đã huấn luyện baseline TF-IDF word/character n-gram kết hợp Logistic Regression. Ba candidate
được fit trên train và chọn trên 347 mẫu validation theo thứ tự ưu tiên: macro-F1, accuracy, rồi thứ
tự khai báo trong config để tie-break có tính tái lập. Hai candidate có character n-gram cùng đạt
macro-F1 0.9816543297 và accuracy 0.9827089337; `word_1_2_char_3_5` thắng vì đứng trước trong config.
Weighted-F1 của candidate được chọn là 0.9825541936. Artifact, label map, candidate report và từng
lỗi validation được lưu để phân tích.
Report có precision, recall, F1 theo từng intent; macro/weighted F1; confusion matrix; ROC/PR curve
one-vs-rest; log-loss, Brier score, ECE và reliability curve. Các metric và biểu đồ này đều là
validation; chưa công bố metric test ở phase này.

Metric chọn model phía trên đến từ candidate fit **chỉ trên train**. Sau khi candidate và threshold
đã khóa, artifact dùng trong CLI/API được fit lại trên train + validation; không dùng artifact cuối này
để báo ngược validation metric. Test vẫn chưa được dùng.

Một thí nghiệm semantic tách biệt dùng embedding local `multilingual-e5-small` (384 chiều) và Logistic
Regression cũng được chạy trên đúng 347 mẫu validation. Kết quả accuracy 0.8731988473 và macro-F1
0.8605845611, thấp hơn TF-IDF trên cùng split. Vì vậy runtime giữ TF-IDF + Logistic Regression; E5
được giữ như thí nghiệm tái lập và bằng chứng lựa chọn model, không được ensemble chỉ vì là
transformer. Script nhận đường dẫn thư mục encoder E5 local qua `--encoder-dir`; report chỉ lưu model
ID chính thức và fingerprint file, không lưu đường dẫn tuyệt đối phụ thuộc máy. Xem
`reports/model_comparison_report.json` và
`reports/semantic_intent_report.json`.

```powershell
python scripts/train_intent.py
```

Fine-tune transformer chưa phải lựa chọn mặc định: chỉ thực hiện sau khi phân tích failure của
baseline và có benchmark dữ liệu thật phù hợp. Ngày 5 và Ngày 6 đã hoàn thành slot extraction,
pipeline, mock action, CLI/API và safety gate. ASR vẫn ngoài phạm vi core text-first của challenge.

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
  -> confidence gate
  -> RegexSlotExtractor theo intent được chấp nhận
  -> action-safety gate
  -> required-slot check
  -> MockActionRouter hoặc clarification/safe rejection
  -> ParseResult gồm intent, confidence, slots, matched_features, action và response
```

Ngày 5 hoàn thành lõi normalizer → intent → slot; Ngày 6 bổ sung confidence/safety gate, mock action,
câu phản hồi, CLI và FastAPI trên cùng pipeline. Action thật vẫn ngoài phạm vi.
Ngày 5 có hard-case cho thời gian dạng số/chữ, “rưỡi”, “kém”, số điện thoại viết/đọc, transcript
không dấu, entity ngoài catalog và trường hợp không được sinh entity giả từ playlist. Pipeline kiểm
tra nhóm slot bắt buộc; nếu báo thức, lời nhắc hoặc cuộc gọi thiếu đối tượng cần thiết thì trả về
yêu cầu bổ sung thay vì giả vờ đã hiểu đủ.

Extractor được đánh giá oracle trên 347 mẫu validation bằng intent thật, để lỗi intent không làm
nhiễu phép đo slot. Snapshot hiện tại đạt exact match 0.9221902017 và micro precision/recall/F1
0.9441747573. Theo nguồn, MASSIVE đạt micro-F1 0.8348623853, còn synthetic đạt 0.9834983498;
khoảng cách này cho thấy template dễ hơn dữ liệu bản địa hóa và phải
được trình bày như một giới hạn. Sau mỗi thay đổi data, slot config hoặc lexicon phải build lại lexicon
từ train rồi tái tạo `reports/slot_extraction_report.json`.
Một phần lỗi đến từ ranh giới annotation bản địa hóa MASSIVE như nhãn `năm` trong câu `năm giờ`;
không thêm rule riêng để học thuộc từng surface validation.

```powershell
python scripts/evaluate_slots.py
```

Báo cáo đầy đủ nằm ở `reports/slot_extraction_report.json`. Đây là metric validation để phát triển;
chưa tính slot metric trên test.

### Ngày 6 — Mock action, CLI và FastAPI

Đã thêm `MockActionRouter` cho đủ năm intent. Router tạo payload deterministic và phản hồi tiếng
Việt nhưng không chạm API/thiết bị/danh bạ thật; trạng thái `mocked` được ghi rõ trong contract để
không thể hiểu nhầm demo đã thực thi action thật. Pipeline chỉ gọi router sau confidence gate,
action-safety gate và kiểm tra nhóm slot bắt buộc. Nếu thiếu giờ báo thức hoặc đích gọi, hệ thống yêu
cầu bổ sung; nếu câu ngoài phạm vi, phủ định/hủy chưa được hỗ trợ hoặc confidence thấp, action không
chạy.

Ngưỡng confidence 0.35 được kiểm tra bằng model train-only trên 347 mẫu validation, không dùng test.
Ngưỡng này chấp nhận 345/347 câu: coverage 0.9942363112, selective accuracy 0.9855072464 và 5 lỗi
được chấp nhận. Coverage thấp nhất trong một intent là 0.9824561404. Chi tiết các ngưỡng lân cận nằm trong
`reports/confidence_gate_report.json` và có thể tái lập bằng:

```powershell
python scripts/evaluate_confidence_gate.py
```

Confidence gate trên validation in-domain **không phải OOD detector**: model closed-set vẫn có thể tự
tin với câu trò chuyện không thuộc năm intent. Vì vậy action-safety gate là policy riêng. Tập
`data/action_safety_challenge.jsonl` và report tương ứng là development regression do dự án tự xây
trong quá trình sửa rule; chúng không phải independent red-team/OOD benchmark và không bảo đảm an
toàn production.

Tập safety hiện có 99 case: 82 negative case cho false-action rate 0 và 17 positive case cho action
recall 1. Trên 347 câu oracle validation, action gate chấp nhận đúng 346 câu (0.9971181556). Đây là số
liệu development do rule và tập thử được xây cùng quá trình, không phải chứng nhận an toàn độc lập.

```powershell
python scripts/evaluate_action_safety.py
```

CLI và FastAPI đều gọi cùng `build_pipeline`, vì vậy model, normalizer, slot config, threshold và
action router không bị lệch giữa hai entrypoint. API nạp model đúng một lần trong lifespan, cung cấp
`GET /health` và `POST /parse`; request sai hoặc text rỗng được Pydantic/FastAPI trả HTTP 422.
Integration test chạy model artifact thật, API bằng ASGI test client và CLI bằng subprocess thật.

Các quyết định, phương án thay thế, giới hạn on-device và phần chưa làm được trình bày đầy đủ trong
`DECISIONS.md`. Đến hết Ngày 6, audit dữ liệu/model/report đã hoàn tất và test chưa được dùng cho metric
hoặc tuning. Ngày 7 chỉ chạy đánh giá cuối sau khi giữ nguyên code, data, threshold, cách tính metric
và test hash đã công bố.
