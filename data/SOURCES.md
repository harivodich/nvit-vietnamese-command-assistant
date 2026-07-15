# Nguồn dữ liệu

Dataset hiện tại có 2.094 câu. Trong đó, 969 câu được ánh xạ từ MASSIVE, 1.106 câu được sinh từ
template do tôi viết cho project và 19 câu hard-case được bổ sung thủ công sau khi phân tích lỗi. Tôi
giữ ba nhóm này tách biệt trong trường `source` để khi đọc metric có thể biết kết quả đến từ dữ liệu
tự nhiên hơn hay từ template dễ đoán hơn.

## Dữ liệu MASSIVE

Nguồn bên ngoài duy nhất được đưa vào tập train, validation và test là Amazon MASSIVE 1.0, phần tiếng
Việt `vi-VN`. Dataset gốc nằm tại [alexa/massive](https://github.com/alexa/massive) và được phát hành
theo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper tương ứng là FitzGerald và cộng
sự, [*MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51
Typologically-Diverse Languages*](https://arxiv.org/abs/2204.08582), 2022. MASSIVE được xây dựng từ
SLURP; thông tin attribution này được giữ lại vì bản trong repository là dữ liệu đã biến đổi, không
phải dữ liệu do project tự viết.

Tôi chỉ lấy các câu thuộc những intent có thể ánh xạ rõ sang phạm vi challenge: đặt báo thức, hỏi thời
tiết, phát nhạc và một phần lời nhắc. Trong lúc build, tên intent và slot được đổi về contract của
project, Unicode được chuẩn hóa về NFC, `source_ref` vẫn giữ ID gốc và partition train/dev/test của
MASSIVE không bị đảo lẫn.

MASSIVE là dữ liệu NLU dạng văn bản đã được bản địa hóa. Nó không có nhãn người nói hay accent
Bắc/Trung/Nam, nên toàn bộ 969 câu này được gắn `region=standard`. Tôi không dùng MASSIVE để tuyên bố
hệ thống đã hiểu giọng vùng miền ngoài đời.

## Dữ liệu do project tự xây dựng

Phần synthetic gồm 1.106 câu được tạo từ template cho năm intent. Mục tiêu của nhóm này là bổ sung cách
nói không dấu và một số từ vựng thường gặp ở miền Bắc, Trung và Nam mà MASSIVE không cung cấp nhãn.
Các template, slot value và mapping đều nằm trong `configs/`, vì vậy có thể đọc và tái tạo thay vì chỉ
tin vào file JSONL đã sinh.

Nhóm manual có 19 hard-case do tôi viết và gán lại intent/slot sau quá trình phát triển. Chúng được
đánh dấu `annotation_quality=reviewed`, nhưng “reviewed” ở đây chỉ có nghĩa tác giả đã kiểm tra nhãn.
Không có câu nào trong snapshot được native speaker ba vùng review theo một protocol độc lập, và cũng
không có audio. Đây là giới hạn cần đọc cùng mọi con số regional.

Hai file `fake_contacts.json` và `music_catalog.json` chỉ phục vụ demo action. Tất cả tên, số điện thoại
và metadata trong đó là dữ liệu giả; chúng không tham gia train, validation hay test.

## Cách giữ provenance và tránh leakage

Mỗi sample lưu `source`, `source_ref`, `group_id`, `region` và mức chất lượng annotation. Những biến thể
cùng template family được gom bằng `group_id` trước khi chia split. Validator còn kiểm tra trùng nguyên
văn, trùng sau khi bỏ dấu và near-similar giữa train, validation và test.

Snapshot test gồm 384 câu và được khóa bằng SHA-256
`47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a`. Thay đổi hash này đồng nghĩa
với thay đổi benchmark. Kết quả theo từng nguồn được báo riêng trong evaluator để tránh dùng điểm cao
của dữ liệu template che đi phần khó hơn của MASSIVE.
