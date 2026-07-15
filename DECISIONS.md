# Quyết định kỹ thuật

Tài liệu này ghi lại trạng thái thiết kế đến hết Ngày 6 và trả lời trực tiếp các nội dung cần trình bày
trong coding challenge. Các con số đánh giá chi tiết chỉ được lấy từ report có thể tái tạo trong
`reports/`; trong đợt audit trước Ngày 7, report nào phụ thuộc config hoặc artifact vừa thay đổi phải
được chạy lại trước khi trích số vào báo cáo nộp.

## 1. Hiểu bài toán và giới hạn phạm vi

Đầu vào cốt lõi là một transcript tiếng Việt chứa yêu cầu ngắn của người dùng. Đầu ra gồm:

- một trong năm intent: `set_reminder`, `set_alarm`, `ask_weather`, `play_music`, `call_contact`;
- các slot liên quan: `datetime`, `location`, `song`, `artist`, `contact_name`, `phone_number`,
  `reminder_text`;
- confidence, dấu vết rule/feature phục vụ giải thích, action giả lập và câu phản hồi.

Đây là bài toán spoken-command understanding theo hướng **text-first NLU**, không phải chatbot tổng
quát. Đến hết Ngày 6, repository chưa nhận audio, chưa tích hợp speech-to-text (STT) và cũng chưa có
text-to-speech (TTS). Thư mục `asr/` chỉ là điểm mở rộng dự kiến. Việc giữ ranh giới này giúp đánh giá
lỗi hiểu ngôn ngữ độc lập với lỗi nhận dạng giọng nói; nó cũng tránh tuyên bố chất lượng audio khi dự
án chưa có corpus giọng nói phù hợp.

Hệ thống không thực thi tác vụ ngoài đời. Gọi điện, báo thức, lời nhắc, thời tiết và phát nhạc đều đi
qua `MockActionRouter` và trả `status=mocked`. Database, danh bạ thật, API thời tiết, media service,
authentication và điều khiển thiết bị nằm ngoài phạm vi hiện tại.

## 2. Phân tích bài toán, dữ liệu và kế hoạch đánh giá

Contract Pydantic là định nghĩa kiểu dữ liệu có thẩm quyền; YAML cung cấp giá trị/config có thể thay
đổi và được kiểm tra chéo với contract. Data builder, NLU, CLI, API và evaluator dùng cùng các contract
này. Luồng xử lý đến Ngày 6 là:

```text
transcript
  -> chuẩn hóa Unicode/từ vựng vùng miền và lỗi STT đã biết
  -> phân loại intent
  -> confidence gate
  -> trích xuất slot theo intent được chấp nhận
  -> action-safety gate
  -> kiểm tra slot bắt buộc
  -> mock action hoặc câu hỏi bổ sung/từ chối an toàn
```

MASSIVE giữ partition gốc. Dữ liệu do dự án tự biên soạn được chia theo `group_id`/template family,
không chia ngẫu nhiên từng câu, để các biến thể cùng gốc không xuất hiện ở nhiều split. Tỷ lệ mục tiêu
là khoảng 70% train, 15% validation và 15% test; tỷ lệ thực tế được phép lệch để ưu tiên chống leakage.

Snapshot sau audit gồm 2.094 câu: train 1.363, validation 347 và test 384 (standard 174, mỗi vùng
Bắc/Trung/Nam 70). Train dùng để fit tham số. Validation dùng để chọn rule, candidate TF-IDF,
hyperparameter và ngưỡng confidence. Test chưa được dùng để tính metric hoặc sửa model. Sau audit,
định nghĩa test được khóa bằng SHA-256
`47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a`. Các tập hard-case,
normalization challenge, intent boundary và action-safety challenge đều là **development regression
set do dự án tự xây dựng**; chúng hữu ích để ngăn lỗi quay lại nhưng không thay thế một test set độc
lập.

Cần phân biệt hai artifact intent:

1. Trong model selection, mỗi candidate chỉ fit trên train rồi được so sánh trên validation.
2. Sau khi candidate và threshold đã khóa, artifact phục vụ CLI/API được fit lại trên
   train + validation. Artifact này không phải artifact dùng để báo validation metric và vẫn không
   được nhìn test.

Vì vậy report chọn model/threshold phải được tạo bằng model train-only. Không được tải artifact
train + validation rồi đo ngược trên validation và gọi đó là kết quả tổng quát hóa.

## 3. Lựa chọn kỹ thuật và các phương án đã cân nhắc

### Intent classifier

Runtime chọn TF-IDF word/character n-gram + Logistic Regression. Rule-based classifier được giữ làm
baseline giải thích được, không phải model chính. Ba candidate TF-IDF được so sánh bằng macro-F1 trên
validation vì metric này cho năm intent trọng số ngang nhau, thay vì để intent nhiều mẫu chi phối.

Trên 347 câu validation, hai candidate có character n-gram cùng đạt macro-F1 0.9816543297 và accuracy
0.9827089337. Chính sách tie-break cố định là macro-F1 → accuracy → thứ tự khai báo trong config, nên
`word_1_2_char_3_5` được chọn; weighted-F1 của nó là 0.9825541936. Thứ tự config chỉ giải quyết tie,
không phải một metric mới. Mọi con số này đến từ artifact train-only; test chưa được dùng.

Character n-gram được thêm để chịu lỗi không dấu, lỗi gõ/STT và biến thể bề mặt; word n-gram giữ tín
hiệu cụm từ như `gọi tôi dậy` hay `nhắc tôi`. Logistic Regression cho xác suất phục vụ confidence gate,
fit nhanh trên CPU và có artifact nhỏ, phù hợp demo on-device.

Các phương án thay thế đã được cân nhắc:

- `multilingual-e5-small` + Logistic Regression đã được chạy như một thí nghiệm semantic cùng split.
  Trên cùng 347 câu, nó đạt accuracy 0.8731988473 và macro-F1 0.8605845611, thấp hơn baseline đã chọn
  và làm runtime nặng hơn, nên không được ensemble vào đường chạy chính. Dependency E5/PyTorch được
  để trong extra `semantic`, không bắt buộc với runtime. Script nhận đường dẫn thư mục encoder local
  qua `--encoder-dir`; report chỉ ghi model ID chính thức và fingerprint, không ghi đường dẫn tuyệt đối.
- Word2Vec/GloVe tạo embedding tĩnh nhưng mất nhiều thông tin theo ngữ cảnh và không có lợi thế rõ
  ràng cho dữ liệu command ngắn so với baseline n-gram đã đo.
- Fine-tune PhoBERT hoặc transformer đa ngôn ngữ có thể hữu ích khi có thêm dữ liệu thật và failure
  analysis chứng minh lỗi semantic. Làm ngay ở quy mô hiện tại tăng chi phí, độ phức tạp và nguy cơ
  overfit mà chưa có bằng chứng sẽ cải thiện test độc lập.
- Ensemble chỉ hợp lý nếu validation chứng minh hai model bổ sung lỗi cho nhau và lợi ích xứng đáng
  với latency/bộ nhớ. Không kết hợp model chỉ vì một model có kiến trúc mới hơn.

Đóng góp của dự án không nằm ở việc tự viết lại thuật toán tối ưu của scikit-learn. Phần được thiết kế
riêng gồm contract, ánh xạ dữ liệu, kiểm soát leakage, chuẩn hóa tiếng Việt, boundary policy, candidate
sweep, probability evaluation, slot extractor, safety gate và luồng CLI/API có thể tái tạo.

### Slot extraction và normalizer

Slot dùng rule/regex theo từng intent thay vì sequence tagger. Với bảy loại slot và dữ liệu gán nhãn
còn nhỏ, cách này dễ audit ranh giới, có dấu vết match và tránh thêm một model chưa đủ dữ liệu. Đổi
lại, entity ngoài catalog và cách diễn đạt mới vẫn là giới hạn. Nếu error analysis Ngày 7 cho thấy rule
không còn kiểm soát được, phương án tiếp theo là token classification hoặc joint intent-slot model.

Normalizer deterministic được chọn để cùng một input cho cùng một output và để train/runtime dùng
đúng một preprocessing. Rule dài được ưu tiên trước rule ngắn, và tiểu từ cuối câu được xử lý theo ngữ
cảnh. Không dùng dịch tự động hay LLM trong runtime vì khó tái lập và không phù hợp mục tiêu chạy cục bộ.

### Runtime và action

CLI và FastAPI gọi chung một runtime factory để không lệch model, normalizer, slot config, threshold
hay safety policy. Model được nạp một lần trong lifespan của API. Action dùng adapter nhỏ và mock
deterministic; cách này chứng minh end-to-end contract mà không tạo side effect hoặc yêu cầu credential.

## 4. Xử lý tiếng Việt và biến thể vùng miền

Preprocessing thực hiện Unicode NFC, chuẩn khoảng trắng, thay các lỗi STT đã liệt kê và ánh xạ một số
biến thể từ vựng Bắc/Trung/Nam. Cụm dài được thay trước cụm ngắn để giảm match một phần. Một số tiểu từ
như `hỉ`, `hen`, `nghen` chỉ chuẩn hóa thành `nhỉ` hoặc `nhé` khi ngữ cảnh cho phép; nếu tín hiệu không
đủ rõ thì giữ nguyên. Mỗi phép thay có `matched_variants` để truy vết.

MASSIVE không có nhãn accent nên toàn bộ sample nguồn này mang `region=standard`; dự án không suy đoán
vùng miền từ nội dung. Regional set là template text tự biên soạn để kiểm tra lexical variation và
dạng không dấu. Nó **không phải transcript người nói thật** và không đo accent âm thanh.

Tại thời điểm audit trước Ngày 7, số sample đã được native speaker theo từng vùng review bằng protocol
chính thức là **0**, và số audio/call recording đã review cũng là **0**. Các lượt rà soát nội bộ của tác
giả không được đổi tên thành native-speaker review. `data/normalization_challenge.jsonl` cũng là
development regression set do dự án biên soạn, không phải benchmark độc lập từ người dùng thật.

Do đó báo cáo chỉ được kết luận hệ thống chịu được các pattern lexical/no-diacritics đã định nghĩa.
Không được kết luận hệ thống hiểu mọi phương ngữ hoặc hoạt động tốt với accent tự nhiên. Nếu mở rộng
sang audio, cần tuyển người nói có consent, cân bằng vùng/tỉnh, tách speaker giữa các split và báo riêng
WER/CER của STT với intent/slot metric của NLU.

## 5. Intent, slot và phương pháp đánh giá

### Ranh giới intent quan trọng

- `call_contact` chỉ là yêu cầu gọi ngay và chấp nhận `contact_name` hoặc `phone_number`.
- Một việc cần làm trong tương lai là `set_reminder`, kể cả nội dung việc là gọi điện.
- `set_alarm` chỉ dùng khi mục tiêu là đánh thức/cảnh báo chính người dùng.

Ví dụ: `gọi mẹ ngay đi` là `call_contact`; `6 giờ gọi mẹ` là `set_reminder`; `gọi tôi dậy lúc 6 giờ`
là `set_alarm`. Các câu ranh giới nằm trong `configs/intent_boundary_cases.yaml` và chỉ đóng vai trò
development regression, không nằm trong test cuối.

### Metric

Intent được báo bằng accuracy, precision/recall/F1 từng lớp, macro-F1, weighted-F1 và confusion matrix.
Vì runtime dùng confidence, report còn có log-loss, multiclass Brier, ECE, ROC-AUC/PR-AUC one-vs-rest
và reliability curve. Slot được đánh giá độc lập bằng gold intent để tách lỗi slot khỏi lỗi intent,
với exact match của toàn bộ slot map và micro/per-slot precision, recall, F1. Ngày 7 cần bổ sung phép đo
end-to-end, trong đó intent dự đoán được đưa thật vào slot extractor và action policy.

Đánh giá slot oracle hiện tại trên 347 câu validation đạt exact match 0.9221902017 và micro
precision/recall/F1 0.9441747573. Breakdown nguồn cho thấy MASSIVE F1 0.8348623853 trong khi
synthetic F1 0.9834983498; vì vậy headline tổng phải đi cùng cảnh báo rằng
template synthetic dễ hơn và không đại diện đầy đủ cho câu tự nhiên.

Mọi headline validation phải chỉ ra source/split và artifact/config tương ứng. Sau mỗi thay đổi slot
config hoặc lexicon, `reports/slot_extraction_report.json` phải được tái tạo trước khi sao chép số liệu
vào tài liệu nộp. Test chưa được dùng cho metric/tuning; lần đánh giá cuối ở Ngày 7 phải dùng đúng test
hash đã công bố và không quay lại sửa model theo lỗi test.

### Confidence không phải OOD detector

Confidence gate chọn ngưỡng trên validation in-domain bằng model train-only. Nó kiểm soát trade-off
giữa coverage và selective accuracy, nhưng `max(predict_proba)` cao không chứng minh câu thuộc một
trong năm domain. Logistic Regression luôn phân phối xác suất vào các label đã biết, nên một câu trò
chuyện hoặc lệnh phủ định vẫn có thể nhận confidence cao.

Ngưỡng 0.35 hiện chấp nhận 345/347 câu validation: coverage 0.9942363112, selective accuracy
0.9855072464, 5 accepted errors và minimum per-intent coverage 0.9824561404. Đây chỉ là đo
selective-classification in-domain, không phải bằng chứng nhận diện OOD.

Vì vậy action-safety gate là lớp policy riêng: nó yêu cầu tín hiệu lệnh theo intent và từ chối một số
câu ngoài phạm vi, phủ định hoặc yêu cầu hủy chưa được hỗ trợ trước khi gọi mock action. File
`data/action_safety_challenge.jsonl` là tập regression do dự án tự viết trong lúc phát triển rule. Báo
cáo của nó đo false-action/positive-action trên chính tập development này, không được quảng bá như OOD
benchmark độc lập hay bảo đảm an toàn production.

Snapshot safety có 99 case: 82 negative case đạt false-action rate 0 và 17 positive case đạt action
recall 1. Với gold intent trên validation, gate chấp nhận đúng 346/347 câu (0.9971181556). Vì rule và
tập challenge được phát triển cùng nhau, các số này là regression evidence, không phải independent
red-team result.

## 6. Ràng buộc on-device và vận hành

Đường chạy chính không gọi LLM hoặc external inference API. TF-IDF + Logistic Regression, normalizer
và regex slot extractor đều chạy CPU; artifact được nạp cục bộ. Đây là lựa chọn ưu tiên khả năng tái
lập, thời gian khởi động và footprint thấp hơn transformer. Tuy nhiên, latency, peak RAM, kích thước
bundle và benchmark trên một thiết bị đích cụ thể chưa được đo đầy đủ; Ngày 7 phải ghi số thực thay vì
chỉ suy luận từ kiến trúc.

Project khai báo Python `>=3.11` vì artifact dùng scikit-learn 1.9.0, phiên bản này không hỗ trợ
Python 3.10. Môi trường local đã kiểm tra bằng Python **3.12**; CI chạy cả Python 3.11 và 3.12.
scikit-learn được khóa ở 1.9.0; NumPy và joblib dùng khoảng phiên bản tương thích để tránh ràng buộc
không cần thiết với một patch release cụ thể.
Dependency được tách theo mục đích:

- cài project mặc định cho runtime TF-IDF/CLI/API;
- extra `dev` cho test, lint và type checking;
- extra `semantic` chỉ để tái tạo thí nghiệm E5, không cần cho demo chính.

Artifact `.joblib` có thể thực thi payload khi deserialize. Runtime hiện kiểm metadata, label map và
SHA-256 theo metadata provenance **trước** khi gọi `joblib.load`, nhờ đó từ chối artifact bị thay đổi hoặc metadata
không khớp. Cơ chế này không sandbox payload; tuyệt đối không load file joblib người dùng tải lên.
Version dependency vẫn phải được quản lý cùng artifact để tránh lỗi tương thích scikit-learn.

## 7. Phần khó nhất và giới hạn còn lại

Những phần khó nhất không phải lệnh rõ như `mở nhạc`, mà là:

- ranh giới `set_alarm` / `set_reminder` / `call_contact` khi cùng có từ `gọi` và thời gian;
- giữ nghĩa khi chuẩn hóa phương ngữ nhưng không sửa nhầm tên riêng, tên bài hát;
- ranh giới slot trong bản địa hóa MASSIVE đôi khi không tự nhiên hoặc không khớp cách gán nhãn của
  dự án;
- tránh leakage giữa các template gần giống và tránh học thuộc lexicon validation/test;
- ngăn side effect cho câu ngoài domain, câu phủ định và yêu cầu hủy không được hỗ trợ.

Các giới hạn phải công khai khi nộp:

- chưa có transcript/audio native-speaker được review theo vùng và chưa đánh giá STT/TTS;
- regional data là synthetic template, nên region breakdown có thể phản ánh template style;
- intent classifier là closed-set; confidence không tự giải quyết open-set/OOD;
- slot extractor phụ thuộc regex/catalog và heuristic, nên entity/cách nói mới có thể bị bỏ sót hoặc
  cắt sai;
- action-safety regression được xây cùng rule, chưa có independent red-team set;
- mock action không chứng minh tích hợp thiết bị, quyền riêng tư, xác thực hay độ an toàn production;
- test hash đã khóa nhưng chưa công bố final test metric; chưa có benchmark latency/RAM trên thiết bị đích;
- quyền tái phân phối công khai của seed từ project cũ cần được xác nhận riêng; seed này hiện không
  thuộc final train/validation/test.

## 8. Phần chưa làm và hướng tiếp tục

Audit trước Ngày 7 đã chạy lại validator/preprocessor, kiểm tra leakage và độ phủ split, build slot
lexicon sau dataset, tái tạo artifact/report có provenance và smoke-test các đường chạy Ngày 6.
Preprocessing nhận 2.094 câu, xuất 2.094 câu và không drop thêm sample. Ngày 5 và Ngày 6 được xem là
hoàn thành trong phạm vi text-first; ASR hiện chỉ là placeholder và không phải điều kiện hoàn tất.

Ngày 7 chỉ chạy test sau khi code, rule, threshold, cách tính metric và hash test đã khóa. Test metric
được chạy một lần cho báo cáo, không dùng để quay lại tuning. Báo cáo cuối cần có:

- intent, slot và end-to-end metric; breakdown theo intent, slot, region, source và annotation quality;
- confusion/failure analysis, không chỉ một con số accuracy;
- action abstention/false-action trên tập an toàn tách biệt nếu có thể thu thập độc lập;
- thời gian khởi động, latency, RAM và kích thước artifact trên thiết bị/môi trường được nêu rõ;
- checksum/version của data, config, model và report để người chấm tái tạo.

Sau challenge, ưu tiên thu thập transcript người dùng thật có consent và native-speaker review trước
khi fine-tune transformer. STT chỉ nên được thêm như adapter và đánh giá riêng; TTS chỉ cần nếu sản phẩm
yêu cầu phản hồi bằng âm thanh. External action, authentication, privacy controls và monitoring là một
workstream production khác, không được ngầm coi là đã hoàn thành bởi mock demo.
