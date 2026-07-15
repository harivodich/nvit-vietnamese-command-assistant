# NVIT Vietnamese Command Assistant

Một trợ lý hiểu lệnh tiếng Việt theo hướng **text-first NLU**. Chương trình nhận transcript, chuẩn hóa
cách viết và từ vựng vùng miền, dự đoán intent, trích xuất slot rồi tạo action giả lập hoặc câu hỏi bổ
sung. Luồng mặc định chạy cục bộ, không cần LLM và không gọi mạng.

Project tập trung vào phần khó nhất của challenge: hiểu đúng câu lệnh ngắn, xử lý một số biến thể
Bắc/Trung/Nam, kiểm soát leakage và đánh giá cả model lẫn pipeline. STT/TTS và điều khiển thiết bị thật
không nằm trong phạm vi đã hoàn thành.

## Phạm vi

Năm intent được hỗ trợ:

- `set_reminder`: tạo lời nhắc;
- `set_alarm`: đặt báo thức;
- `ask_weather`: hỏi thời tiết;
- `play_music`: phát nhạc;
- `call_contact`: gọi một liên hệ hoặc số điện thoại ngay lúc đó.

Bảy slot: `datetime`, `location`, `song`, `artist`, `contact_name`, `phone_number` và
`reminder_text`.

Đầu ra có intent, confidence, slot, các dấu vết match, action và câu phản hồi. Action mặc định có
`status=mocked`: chương trình không tự gọi điện, đặt báo thức hay phát nhạc trên máy của người dùng.

## Chạy nhanh

Project yêu cầu Python 3.11 trở lên. Dependency chính gồm Pydantic, PyYAML, NumPy, scikit-learn,
joblib, FastAPI và Uvicorn; phiên bản cụ thể nằm trong `pyproject.toml`. Cài runtime:

```powershell
python -m pip install -e .
```

Chạy CLI:

```powershell
nvit-assistant "nhắc tôi uống thuốc lúc 8 giờ"
nvit-assistant --json "Bữa ni ở Huế trời răng rồi hỉ"
```

Nếu chưa cài entrypoint, có thể chạy thẳng từ source:

```powershell
python scripts/run_assistant.py --json "mở nhạc Mỹ Tâm cho tui nghe"
```

Khởi động API:

```powershell
python -m uvicorn nvit_assistant.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Swagger UI ở `http://127.0.0.1:8000/docs`. `POST /parse` nhận payload:

```json
{
  "text": "nhắc tôi uống thuốc lúc 8 giờ",
  "region_hint": "standard"
}
```

## Ba ví dụ thực tế

Các kết quả dưới đây được lấy từ CLI hiện tại và rút gọn còn những trường chính.

```text
Input:  Bữa ni ở Huế trời răng rồi hỉ
Output: intent=ask_weather, region=central
        slots={datetime: "hôm nay", location: "huế"}
        response="Đã giả lập yêu cầu thời tiết tại huế vào hôm nay."
```

```text
Input:  mở nhạc Mỹ Tâm cho tui nghe
Output: intent=play_music
        slots={artist: "mỹ tâm"}
        response="Đã giả lập phát nhạc của mỹ tâm."
```

```text
Input:  nhắc tôi uống thuốc lúc 8 giờ
Output: intent=set_reminder
        slots={datetime: "8 giờ", reminder_text: "uống thuốc"}
        response="Đã giả lập tạo lời nhắc “uống thuốc” vào 8 giờ."
```

## Luồng xử lý

```text
transcript
  -> chuẩn hóa Unicode, lỗi STT đã biết và từ vựng vùng miền
  -> TF-IDF word/character n-gram + Logistic Regression
  -> confidence gate
  -> trích xuất slot theo intent
  -> action-safety gate
  -> kiểm tra slot bắt buộc
  -> mock/live-weather action, hỏi bổ sung hoặc từ chối an toàn
```

CLI và FastAPI dùng cùng một pipeline factory nên không có hai bản logic riêng. Model được nạp một
lần khi API khởi động.

## Cấu trúc repository

```text
configs/                         cấu hình model, slot, vùng miền và data template
data/raw_sources/                nguồn đầu vào có provenance
data/samples/                    JSONL train, validation và test
models/                          artifact intent và slot lexicon
scripts/                         build, validate, train, evaluate và demo
src/nvit_assistant/
  actions/                       mock router và live-weather adapter tùy chọn
  eval/                          hàm metric và final evaluator
  nlu/                           normalizer, intent, slot và pipeline
tests/                           unit, regression và integration tests
```

Lý do chọn cách làm và các phương án đã loại được trình bày trong [DECISIONS.md](DECISIONS.md).

## Dataset

Snapshot hiện có 2.094 câu cho đủ năm intent:

| Split | Số câu |
|---|---:|
| Train | 1.363 |
| Validation | 347 |
| Test | 384 |

Test gồm 174 câu standard và 70 câu cho mỗi nhóm Bắc, Trung, Nam. Nguồn chính là MASSIVE `vi-VN`
đã lọc cùng dữ liệu template lexical do project tự biên soạn. Chi tiết provenance và license nằm ở
[data/SOURCES.md](data/SOURCES.md).

Các câu cùng template family được gom bằng `group_id` trước khi chia tập. Validator còn kiểm tra trùng
nguyên văn, trùng sau khi bỏ dấu, near-duplicate và leakage giữa split. Slot lexicon chỉ được build từ
train, không đọc validation/test.

Regional set hiện đo biến thể từ vựng trên **văn bản**, không đo accent âm thanh. Số transcript/audio
được native speaker review theo protocol chính thức vẫn là 0. Vì vậy project không tuyên bố hiểu đầy
đủ giọng Bắc/Trung/Nam ngoài đời.

Build và kiểm tra lại dataset sau khi tải MASSIVE 1.0 `vi-VN.jsonl`:

```powershell
python scripts/build_dataset.py --massive-jsonl path\to\vi-VN.jsonl
python scripts/validate_data.py --data-dir data/samples
python scripts/audit_normalization.py --data-dir data/samples
python scripts/build_slot_lexicon.py
```

Test snapshot được khóa bằng SHA-256:

```text
47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a
```

## Model và lựa chọn kỹ thuật

Model chính là TF-IDF word/character n-gram kết hợp Logistic Regression. Character n-gram giúp chịu
được câu không dấu và lỗi bề mặt; Logistic Regression nhẹ, có xác suất cho confidence gate và chạy
nhanh trên CPU.

Ba cấu hình TF-IDF được fit trên train rồi chọn bằng macro-F1 trên validation. Candidate thắng đạt
validation macro-F1 **98,17%**. Một thí nghiệm riêng dùng `multilingual-e5-small` + Logistic Regression
đạt **86,06%** trên cùng split, nên E5 không được đưa vào runtime. Extra `semantic` chỉ phục vụ tái tạo
thí nghiệm này:

```powershell
python -m pip install -e ".[semantic]"
```

Slot được trích bằng rule/regex theo intent. Cách này hợp với bảy slot và lượng dữ liệu hiện tại, dễ
audit hơn một sequence tagger nhưng sẽ yếu hơn với entity hoặc cách diễn đạt chưa từng gặp.

## Kết quả đánh giá

Tập test gồm 384 câu, bao phủ năm intent, bảy slot, tiếng Việt chuẩn, ba nhóm từ vựng vùng miền và câu
không dấu. Runtime không được truyền sẵn region label.

| Chỉ số | Kết quả |
|---|---:|
| Intent accuracy / macro-F1 | **92,71% / 92,25%** |
| Runtime intent accuracy | **90,36%** |
| Runtime coverage / selective accuracy | **92,45% / 97,75%** |
| Oracle slot exact match / micro-F1 | **81,77% / 86,89%** |
| End-to-end slot exact match / micro-F1 | **74,48% / 83,15%** |
| Full-command success | **73,96%** |

Intent accuracy đo riêng classifier. Runtime intent phản ánh kết quả sau confidence, boundary và safety
gate; full-command success chỉ đạt khi intent, slot và mock action đều đúng. Khoảng cách giữa **92,71%**
intent accuracy và **73,96%** full-command success cho thấy phần còn yếu nằm nhiều ở slot và policy,
không chỉ ở classifier.

Runtime intent accuracy hiện tại theo vùng là Bắc **81,43%**, Trung **88,57%**, Nam **94,29%**;
nhóm không dấu đạt **76,85%**. Đây vẫn chỉ là text benchmark có nhiều biến thể tự tạo, không đại diện
cho khả năng hiểu accent từ audio ngoài đời.

## Kiểm tra và tái tạo

Cài công cụ phát triển rồi chạy quality gate:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m mypy src
python scripts/validate_data.py --data-dir data/samples
python scripts/audit_normalization.py --data-dir data/samples
```

Snapshot hiện tại qua **297 test**, Ruff, mypy, dependency check, data validator và leakage audit.

Để tái tạo bảng đánh giá chi tiết:

```powershell
python scripts/evaluate.py --post-audit --overwrite
```

Evaluator kiểm tra checksum test rồi sinh JSON, Markdown và confusion matrix vào `reports/` ở máy
local. Thư mục này được Git bỏ qua vì đây là output có thể tái tạo, không phải source cần nộp. Các lỗi
trong tập test đã được dùng cho error analysis ở giai đoạn cuối, nên số hiện tại cần được đọc như kết
quả hồi quy; một holdout mới vẫn là bước tiếp theo để đo khả năng tổng quát hóa.

## Action và phần chưa làm

Mock action là mặc định để test và evaluation deterministic. Có thể bật demo Open-Meteo riêng cho
weather:

```powershell
nvit-assistant --live-weather "thời tiết ở Huế ngày mai"
```

Danh bạ và catalog nhạc trong `data/` chỉ là dữ liệu giả. Project chưa có STT/TTS, authentication,
privacy controls hay tích hợp thiết bị thật. Đây là các phần mở rộng production, không phải tính năng
đã hoàn thành của bản challenge.

Các giới hạn, trade-off và hướng tiếp tục được ghi ngắn gọn trong [DECISIONS.md](DECISIONS.md).
