# Nguồn và phạm vi benchmark chuẩn hóa

`normalization_challenge.jsonl` là **development regression set nội bộ** gồm các câu lệnh do dự án tự
biên soạn, không sao chép câu từ corpus bên ngoài. Mỗi câu chỉ dùng vocabulary/biến thể phù hợp với
năm intent của challenge và được giữ riêng khỏi train/validation/test intent. Việc tách file ngăn lẫn
với tập train, nhưng không biến nó thành benchmark độc lập vì câu và expected output vẫn được thiết kế
trong quá trình phát triển normalizer.

## Nguồn tham khảo

- Kondo, Mika (2013), *Vietnamese dialect maps on vocabulary*, DOI
  `10.5281/zenodo.6440811`, CC BY 4.0. Appendix khảo sát 67 mục từ tại nhiều điểm điều tra;
  dự án dùng làm một nguồn kiểm chứng cho biến thể từ vựng như `coi`/`xem`, không chép nguyên
  bảng dữ liệu vào repository. Trong code, mapping này chỉ áp dụng cho cụm `coi thời tiết` để
  không làm hỏng tên bài hát không dấu như `một cõi đi về`.
- Ta, Dinh, Nguyen (2026), [*ViDia2Std*](https://huggingface.co/datasets/Biu3010/ViDia2Std).
  Nghiên cứu cho thấy normalizer cần bao phủ cả ba vùng và biến thể theo tỉnh. Dataset card hiện nêu
  CC BY-NC 4.0; hạn chế phi thương mại không phù hợp với challenge có thể được công ty sử dụng, nên
  dự án không nhập hoặc phân phối lại dữ liệu.
- Dinh et al. (2024), [*Multi-Dialect Vietnamese (ViMD)*](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset).
  Dataset audio/transcript có phạm vi 63 tỉnh và dataset card hiện nêu CC BY-NC-ND 4.0. Hạn chế phi
  thương mại và không-phái-sinh khiến dữ liệu không phù hợp để nhập/biến đổi trong core dataset; dự án
  chỉ tham khảo thiết kế coverage, không sao chép câu/audio.
- Nguyen et al. (2024), *ViLexNorm*. Corpus normalization có license CC BY-NC-SA 4.0; không
  phù hợp để đưa dữ liệu vào challenge có thể dùng cho công ty, nên không nhập dữ liệu.

Tập regression hiện đo độ đúng của các rule/pattern đã định nghĩa, không chứng minh coverage mọi
phương ngữ. Tại thời điểm audit trước Ngày 7, số sample trong tập này được native speaker theo từng
vùng review bằng protocol chính thức là **0**; số audio/call recording được review cũng là **0**.
Nhãn `annotation_quality=reviewed` ở một số hard-case khác trong dự án chỉ là review annotation nội
bộ và không làm thay đổi hai con số native review này.
Protocol consent, reviewer independence và acceptance criteria được mô tả trong
`data/NATIVE_REVIEW_PROTOCOL.md`. Chỉ sau khi quy trình đó được thực hiện mới được đổi trạng thái review
hoặc đưa ra kết luận về người nói/accent tự nhiên.
