# Quyết định kỹ thuật

## Phạm vi

Sản phẩm là hệ thống hiểu command tiếng Việt trong năm domain cố định, không phải chatbot tổng quát.

## Text-first

Transcript text là input cốt lõi. Audio/ASR sẽ được đặt sau core NLU để đánh giá lỗi ASR và lỗi NLU độc lập.

## Contract trước implementation

Schema Pydantic và YAML config là nguồn sự thật chung cho data, NLU, CLI/API và evaluation. Các quyết định classifier, slot extraction và ASR sẽ được bổ sung khi triển khai phase tương ứng.

## Chia dữ liệu và đánh giá

Mục tiêu split là 70% train, 15% validation, 15% test. MASSIVE giữ partition gốc và template được chia theo `group_id`, vì vậy tỷ lệ thực tế có thể lệch nhẹ để ưu tiên chống leakage. Validation dùng để chọn rule, threshold và hyperparameter; test được khóa đến lần đánh giá cuối. Báo cáo cuối ưu tiên intent accuracy, slot precision/recall/F1 và breakdown theo vùng miền, thay vì chỉ báo training loss.

## Nguồn và chất lượng dataset

Core dataset sử dụng MASSIVE `vi-VN` theo CC BY 4.0 và template lexical ba vùng do dự án kiểm soát. Seed reminder đã review từ project cũ được giữ làm nguồn audit, nhưng không được đưa vào final split nếu gần với test MASSIVE. MASSIVE phải vượt qua phiếu intent/ngôn ngữ, grammar trung bình từ 3.5 và spelling trung bình từ 1.5 trước khi ánh xạ. Câu trùng không dấu được loại trước split; mọi variant cùng `group_id` bắt buộc ở cùng split.

MASSIVE không có nhãn accent nên được gắn `region=standard`, không suy đoán Bắc/Trung/Nam. Regional set hiện là text template, không phải transcript người nói thật. Template family được gán nguyên vào một split; sau khi gộp các nguồn, validator tiếp tục loại cả group nếu template gần giống câu thuộc split ưu tiên hơn theo thứ tự test → validation → train. Vì vậy metric vùng miền sau này chỉ chứng minh độ bền với lexical/no-diacritics patterns đã định nghĩa, không phải chất lượng ASR trên accent tự nhiên.

## Đích gọi điện

Lệnh `call_contact` chấp nhận một trong hai slot: `contact_name` hoặc `phone_number`. Tên liên hệ phù hợp khi ứng dụng có danh bạ; số điện thoại là fallback cho lệnh đọc số trực tiếp. Việc tra cứu/sử dụng danh bạ thật nằm ngoài core NLU và sẽ chỉ được mock trong phạm vi challenge.

## Ranh giới gọi điện, nhắc việc và báo thức

`call_contact` chỉ dùng cho yêu cầu gọi ngay. Khi câu có thời điểm tương lai để thực hiện một việc bên ngoài hệ thống, intent là `set_reminder`, kể cả việc đó là gọi điện. `set_alarm` chỉ dùng khi mục đích là đánh thức/cảnh báo chính người dùng. Ví dụ: `gọi mẹ ngay đi` là `call_contact`; `6h gọi mẹ` và `nhắc tôi gọi mẹ lúc 6h` là `set_reminder`; `gọi tôi dậy lúc 6h` là `set_alarm`. Các câu ranh giới này được lưu ở `configs/intent_boundary_cases.yaml`; chúng không nằm trong test split và được dùng làm regression/audit riêng.

## Action và ranh giới an toàn

Challenge chỉ dùng action giả lập. `MockActionRouter` tạo payload cùng response để chứng minh luồng
end-to-end, nhưng không gọi điện, đặt lịch, truy vấn thời tiết hay phát nhạc thật. Mọi kết quả đều có
`status=mocked`. Pipeline không gọi action nếu thiếu slot bắt buộc hoặc confidence thấp. Interface
`ActionRouter` được giữ nhỏ để có thể thay bằng adapter thiết bị thật mà không sửa NLU.

## Confidence gate

Ngưỡng 0.45 được chọn trên validation bằng đúng candidate TF-IDF train-only; test không được đọc.
Trên 352 câu validation, gate chấp nhận 332 câu (coverage 0.9432), selective accuracy 0.9910 và
chấp nhận 3 lỗi, so với 11 lỗi khi không gate. Gate đồng thời từ chối nhầm 12 câu đúng, là trade-off
được chấp nhận để demo không thực thi action khi model thiếu chắc chắn. Báo cáo giữ toàn bộ sweep từ
0.0 đến 0.7 để quyết định có thể audit và thay đổi theo chi phí sai của sản phẩm thật.

## CLI và API

CLI và FastAPI dùng chung runtime factory thay vì tự lắp dependency. FastAPI lazy-load model một lần
trong lifespan; CLI load một lần cho mỗi process. API chỉ có health và parse, chưa thêm database,
authentication, queue hay external integration vì các phần đó không giúp chứng minh chất lượng NLU
trong phạm vi challenge.
