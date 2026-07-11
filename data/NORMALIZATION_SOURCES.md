# Nguồn và phạm vi benchmark chuẩn hóa

`normalization_challenge.jsonl` là benchmark nội bộ gồm các câu lệnh do dự án tự biên soạn,
không sao chép câu từ corpus bên ngoài. Mỗi câu chỉ dùng vocabulary/biến thể phù hợp với năm
intent của challenge và được giữ riêng khỏi train/validation/test intent.

## Nguồn tham khảo

- Kondo, Mika (2013), *Vietnamese dialect maps on vocabulary*, DOI
  `10.5281/zenodo.6440811`, CC BY 4.0. Appendix khảo sát 67 mục từ tại nhiều điểm điều tra;
  dự án dùng làm một nguồn kiểm chứng cho biến thể từ vựng như `coi`/`xem`, không chép nguyên
  bảng dữ liệu vào repository. Trong code, mapping này chỉ áp dụng cho cụm `coi thời tiết` để
  không làm hỏng tên bài hát không dấu như `một cõi đi về`.
- Ta, Dinh, Nguyen (2026), *ViDia2Std*. Nghiên cứu cho thấy normalizer cần bao phủ cả ba vùng
  và biến thể theo tỉnh. Dataset không được nhập vào repository vì trang công bố không nêu
  license dữ liệu để tái phân phối.
- Dinh et al. (2024), *Multi-Dialect Vietnamese (ViMD)*. Dataset audio/transcript có phạm vi
  63 tỉnh nhưng chỉ cấp cho nghiên cứu, nên chỉ dùng làm tham khảo thiết kế evaluation, không
  sao chép câu/audio.
- Nguyen et al. (2024), *ViLexNorm*. Corpus normalization có license CC BY-NC-SA 4.0; không
  phù hợp để đưa dữ liệu vào challenge có thể dùng cho công ty, nên không nhập dữ liệu.

Benchmark hiện đo độ đúng của rule hiện có, không chứng minh coverage mọi phương ngữ. Trước
khi nộp challenge, các câu trong file này cần native speaker review theo từng vùng; protocol
không gán nhãn giả được mô tả trong `data/NATIVE_REVIEW_PROTOCOL.md`.
