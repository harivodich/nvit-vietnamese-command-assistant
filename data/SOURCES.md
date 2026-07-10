# Nguồn dữ liệu và quyết định sử dụng

## Nguồn được đưa vào core dataset

### Amazon MASSIVE 1.0 — tiếng Việt `vi-VN`

- Trang dự án: https://github.com/alexa/massive
- License dữ liệu: CC BY 4.0.
- Citation: FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages* (2022).
- Nguồn nền SLURP: Bastianelli et al., *SLURP: A Spoken Language Understanding Resource Package* (EMNLP 2020), cũng theo CC BY 4.0. MASSIVE được tạo bằng cách dịch/bản địa hóa dữ liệu text SLURP.
- Phần sử dụng: các intent `alarm_set`, `weather_query`, `play_music` và một tập con `calendar_set` có cấu trúc lời nhắc rõ ràng.
- Biến đổi: chuẩn Unicode NFC; đổi tên intent/slot về contract của dự án; giữ `source_ref` theo ID gốc; giữ partition gốc (`train`, `dev`, `test`).
- Giới hạn: MASSIVE là dữ liệu bản địa hóa dạng văn bản, không có nhãn accent Bắc/Trung/Nam. Tất cả sample MASSIVE được gắn `region=standard`.

### Project cũ `E:/DM/voice-assistant/data/data_R.csv`

- Quyền sử dụng: dữ liệu nội bộ do chủ repository cung cấp.
- Phần sử dụng: chỉ các câu reminder được đọc và gán lại slot thủ công trong `raw_sources/old_project_seed.jsonl`.
- Giới hạn: nhãn `calendar` cũ rộng hơn `set_reminder`, vì vậy không tự động nhập toàn bộ 408 câu calendar.
- Quy tắc leakage có thể loại seed này khỏi final dataset nếu cấu trúc câu gần test MASSIVE. Seed vẫn được giữ để audit và bổ sung sau khi có câu độc lập đã review.

### Template vùng miền được review

- Nguồn: template do dự án tự xây dựng, tham khảo cách phân chia ba vùng trong các nghiên cứu ViMD và ViDia2Std.
- Mục đích: kiểm tra lexical variation và dạng không dấu trong phạm vi năm intent.
- Giới hạn: đây không phải transcript của người nói thật; sample được đánh dấu `annotation_quality=template_generated` và không được dùng để tuyên bố chất lượng ASR/accent ngoài đời.

## Nguồn chỉ tham khảo, không phân phối trong repository

### VinAI JointIDSF / PhoATIS

Dataset có nhãn intent-slot tiếng Việt nhưng thuộc domain hàng không. Điều khoản yêu cầu chỉ dùng cho nghiên cứu/giáo dục và không phân phối lại dữ liệu hoặc bản sửa đổi, nên repository chỉ tham khảo format và paper, không copy sample.

### Mozilla Common Voice Vietnamese

License CC0 và có metadata accent tùy người nói, nhưng đây là ASR corpus không có intent/slot. Nguồn này phù hợp cho phase audio hoặc đánh giá lỗi transcription, không được gán intent tự động để đưa vào core NLU metric.

### ViMD và ViDia2Std

Hai nguồn có giá trị cho dialect/accent coverage. ViMD tập trung audio/dialect identification; ViDia2Std tập trung dialect-to-standard translation. Chưa đưa vào core dataset vì không có nhãn năm intent của dự án và cần kiểm tra điều khoản tải/phân phối cụ thể trước khi sử dụng dữ liệu.

### VietSuperSpeech và các speech corpus tổng hợp khác

Không nhập do dataset card/license hoặc provenance chưa đủ rõ cho một public recruitment repository, đồng thời không có nhãn intent-slot phù hợp.

## Nguyên tắc

- Không gán nhãn vùng miền cho dữ liệu không có bằng chứng vùng miền.
- Không copy câu từ website có copyright không rõ.
- Không để các biến thể cùng `group_id` xuất hiện ở nhiều split.
- Test set giữ nguyên đến lần đánh giá cuối; mọi rule và threshold chỉ được chọn bằng train/validation.
