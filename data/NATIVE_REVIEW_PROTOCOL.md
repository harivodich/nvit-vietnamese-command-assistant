# Protocol review native speaker và thu audio

Normalizing rule-based chỉ là baseline. Không được gắn nhãn `reviewed` cho câu/vocabulary nếu
chưa có người nói bản địa xác nhận. File này là quy trình bắt buộc trước khi khẳng định hệ thống
hỗ trợ phương ngữ thực tế.

## Review text

1. Mỗi mapping mới cần hai reviewer sống/lớn lên tại vùng hoặc tỉnh liên quan.
2. Reviewer đánh dấu `accept`, `reject` hoặc `context-dependent`; mapping context-dependent
   không được đưa vào `variants` một-từ-một.
3. Câu benchmark phải có đủ năm intent và ít nhất câu hỏi, câu yêu cầu, câu không dấu, lỗi STT.
4. Không ghi tên, số điện thoại hoặc địa chỉ thật vào Git. Dùng `reviewer_code` và `consent_ref`.
5. Chỉ khi hai reviewer đồng ý mới đổi trạng thái mapping/câu thành `reviewed`.

## Thu audio cho phase ASR

- Thu tối thiểu 10 command cho mỗi người nói, bao phủ năm intent; lưu transcript chuẩn và
  transcript ASR riêng để đo lỗi ASR/NLU độc lập.
- Mỗi bản ghi cần: `speaker_code`, tỉnh, vùng, điều kiện nhiễu, thiết bị, consent reference,
  audio path và transcript. Không commit audio hoặc PII khi chưa xác nhận license/consent.
- Mục tiêu coverage là có ít nhất một người nói native cho từng tỉnh cần hỗ trợ; số giờ audio và
  benchmark chính thức phải được quyết định sau khi có nguồn thu hợp lệ.

## Mẫu hàng review

```text
item_id,province,region,raw_text,expected_text,reviewer_code,decision,notes,consent_ref
```

Trước khi nộp challenge, report phải nêu rõ số mapping/câu đã review, số pending và tỉnh chưa có
reviewer. Không được suy diễn từ `north`/`central`/`south` rằng mọi tỉnh trong vùng đều giống nhau.
