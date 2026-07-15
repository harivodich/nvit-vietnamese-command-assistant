# Nguồn dữ liệu và quyết định sử dụng

## Nguồn được đưa vào core dataset

### Amazon MASSIVE 1.0 — tiếng Việt `vi-VN`

- Trang dự án và dữ liệu gốc: [alexa/massive](https://github.com/alexa/massive).
- License dữ liệu: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Citation: FitzGerald et al., [*MASSIVE: A 1M-Example Multilingual Natural Language Understanding
  Dataset with 51 Typologically-Diverse Languages*](https://arxiv.org/abs/2204.08582) (2022).
- Nguồn nền SLURP: Bastianelli et al., *SLURP: A Spoken Language Understanding Resource Package* (EMNLP 2020), cũng theo CC BY 4.0. MASSIVE được tạo bằng cách dịch/bản địa hóa dữ liệu text SLURP.
- Phần sử dụng: các intent `alarm_set`, `weather_query`, `play_music` và một tập con `calendar_set` có cấu trúc lời nhắc rõ ràng.
- Biến đổi: chuẩn Unicode NFC; đổi tên intent/slot về contract của dự án; giữ `source_ref` theo ID gốc;
  giữ partition gốc (`train`, `dev`, `test`). Khi phân phối bản biến đổi phải giữ attribution MASSIVE,
  chỉ rõ đã sửa đổi và kèm liên kết CC BY 4.0.
- Giới hạn: MASSIVE là dữ liệu bản địa hóa dạng văn bản, không có nhãn accent Bắc/Trung/Nam. Tất cả sample MASSIVE được gắn `region=standard`.

### Template vùng miền do dự án tự biên soạn

- Nguồn: template do dự án tự xây dựng, tham khảo cách phân chia ba vùng trong các nghiên cứu ViMD và ViDia2Std.
- Mục đích: kiểm tra lexical variation và dạng không dấu trong phạm vi năm intent.
- Trạng thái review: sample được đánh dấu `annotation_quality=template_generated`; đến đợt audit trước
  Ngày 7, số sample/audio được native speaker review theo protocol chính thức là 0.
- Giới hạn: đây không phải transcript của người nói thật và không được dùng để tuyên bố chất lượng
  ASR/accent ngoài đời. Các hard-case do tác giả rà soát trong error analysis vẫn là internal
  development data, không phải native-speaker benchmark.

### Hard-case nội bộ từ error analysis

- Snapshot final có 19 sample `source=manual`, được giữ lại sau bộ lọc trùng/gần giống giữa split.
- `annotation_quality=reviewed` chỉ có nghĩa intent/slot đã được người phát triển rà soát nội bộ.
  Nhãn này không chứng minh câu do native speaker vùng miền viết hoặc đã qua protocol native review.
- Đây là development data được tạo sau failure analysis, nên không được dùng như benchmark độc lập.

## Nguồn tham khảo hoặc seed audit không nhập core dataset

### Project cũ `data_R.csv`

- Quyền sử dụng nội bộ: dữ liệu do chủ repository cung cấp. Quyền **tái phân phối công khai** chưa có
  bằng chứng license riêng trong repository và phải được chủ dữ liệu xác nhận trước khi public source.
- Phần được chuẩn bị: một nhóm câu reminder được đọc và gán lại slot thủ công trong
  `raw_sources/old_project_seed.jsonl`. Nhãn `annotation_quality=reviewed` ở đây chỉ có nghĩa đã rà soát
  annotation nội bộ, không có nghĩa đã được native speaker review theo protocol vùng miền.
- Giới hạn: nhãn `calendar` cũ rộng hơn `set_reminder`, vì vậy không tự động nhập toàn bộ 408 câu calendar.
- Trạng thái hiện tại: không có sample `source=old_project` trong final train/validation/test. Seed chỉ
  được giữ ở `raw_sources/` để audit; nếu chưa xác nhận quyền tái phân phối thì phải loại file seed khỏi
  bản public trước khi nộp.

### VinAI JointIDSF / PhoATIS

Dataset có nhãn intent-slot tiếng Việt nhưng thuộc domain hàng không. Điều khoản yêu cầu chỉ dùng cho nghiên cứu/giáo dục và không phân phối lại dữ liệu hoặc bản sửa đổi, nên repository chỉ tham khảo format và paper, không copy sample.

### Mozilla Common Voice Vietnamese

License CC0 và có metadata accent tùy người nói, nhưng đây là ASR corpus không có intent/slot. Nguồn này phù hợp cho phase audio hoặc đánh giá lỗi transcription, không được gán intent tự động để đưa vào core NLU metric.

### ViMD và ViDia2Std

Hai nguồn có giá trị cho dialect/accent coverage nhưng không có nhãn năm intent của dự án. Dataset card
[ViDia2Std](https://huggingface.co/datasets/Biu3010/ViDia2Std) nêu CC BY-NC 4.0; dataset card
[ViMD](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) nêu CC BY-NC-ND 4.0. Điều khoản
phi thương mại, và với ViMD còn có hạn chế phái sinh, không phù hợp để nhập vào repository challenge
có thể dùng cho công ty. Dự án chỉ tham khảo phạm vi vùng miền, không copy câu/audio.

### VietSuperSpeech và các speech corpus tổng hợp khác

Không nhập do dataset card/license hoặc provenance chưa đủ rõ cho một public recruitment repository, đồng thời không có nhãn intent-slot phù hợp.

## Nguyên tắc

- Không gán nhãn vùng miền cho dữ liệu không có bằng chứng vùng miền.
- Không copy câu từ website có copyright không rõ.
- Không để các biến thể cùng `group_id` xuất hiện ở nhiều split.
- Test chưa được dùng để tính metric hoặc chọn rule/threshold. Sau audit, định nghĩa test được khóa bằng
  SHA-256 `47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a`; thay đổi hash là thay đổi
  benchmark và phải được ghi nhận trước khi đánh giá cuối.
- Ghi rõ tập nào do dự án tự xây trong lúc phát triển; normalization/action-safety/boundary challenge
  là regression set, không được trình bày như benchmark độc lập.
- Không public dữ liệu nội bộ khi chưa có bằng chứng cho phép tái phân phối, dù dữ liệu đó không nằm
  trong final split.
