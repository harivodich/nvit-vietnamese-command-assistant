# Quyết định kỹ thuật

## Phạm vi

Sản phẩm là hệ thống hiểu command tiếng Việt trong năm domain cố định, không phải chatbot tổng quát.

## Text-first

Transcript text là input cốt lõi. Audio/ASR sẽ được đặt sau core NLU để đánh giá lỗi ASR và lỗi NLU độc lập.

## Contract trước implementation

Schema Pydantic và YAML config là nguồn sự thật chung cho data, NLU, CLI/API và evaluation. Các quyết định classifier, slot extraction và ASR sẽ được bổ sung khi triển khai phase tương ứng.

## Chia dữ liệu và đánh giá

Dataset sẽ được chia stratified theo intent, region và variant: 70% train, 15% validation, 15% test. Validation dùng để chọn rule, threshold và hyperparameter; test được khóa đến lần đánh giá cuối. Báo cáo cuối ưu tiên intent accuracy, slot precision/recall/F1 và breakdown theo vùng miền, thay vì chỉ báo training loss.

## Đích gọi điện

Lệnh `call_contact` chấp nhận một trong hai slot: `contact_name` hoặc `phone_number`. Tên liên hệ phù hợp khi ứng dụng có danh bạ; số điện thoại là fallback cho lệnh đọc số trực tiếp. Việc tra cứu/sử dụng danh bạ thật nằm ngoài core NLU và sẽ chỉ được mock trong phạm vi challenge.
