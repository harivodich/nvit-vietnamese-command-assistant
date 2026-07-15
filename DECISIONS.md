# Quyết định kỹ thuật

Tôi viết tài liệu này để giải thích vì sao project có hình dạng hiện tại, thay vì chỉ liệt kê những gì
đã làm. Các số liệu được trình bày theo snapshot code hiện tại; nơi nào dữ liệu hoặc cách đánh giá còn
hạn chế, tôi ghi rõ thay vì cố biến chúng thành một kết luận mạnh hơn.

## 1. Hiểu bài toán và giới hạn phạm vi

Tôi hiểu yêu cầu cốt lõi là biến một transcript tiếng Việt ngắn thành intent và slot đủ để trợ lý biết
người dùng muốn làm gì: spoken-command understanding theo hướng **text-first**, không phải chatbot tổng quát.

Đầu ra gồm một trong năm intent (`set_reminder`, `set_alarm`, `ask_weather`, `play_music`,
`call_contact`), bảy slot liên quan, confidence, dấu vết match, action và câu phản hồi.

Điều tôi ưu tiên nhất là ranh giới intent/slot đúng, cách xử lý tiếng Việt có thể giải thích, không làm
rò rỉ dữ liệu giữa các split và một phép đánh giá trung thực. Điều ít quan trọng hơn ở giai đoạn này là
giao diện đẹp, số lượng tích hợp thiết bị hay dùng model lớn chỉ để kiến trúc trông hiện đại.

Project chưa nhận audio và chưa có STT/TTS. Giữ phạm vi text-first giúp tách lỗi hiểu ngôn ngữ khỏi lỗi
nhận dạng giọng nói, đồng thời tránh tuyên bố chất lượng audio khi chưa có corpus phù hợp.

Mặc định mọi action đều là mock. Chế độ `live-weather` là demo opt-in duy nhất gọi dịch vụ thật;
danh bạ và catalog nhạc vẫn là dữ liệu giả. Project không thực hiện cuộc gọi, đặt báo thức hay phát
media trên thiết bị thật.

## 2. Phân tích bài toán, dữ liệu và kế hoạch đánh giá

Tôi chia bài toán thành bốn phần: contract/dữ liệu, chuẩn hóa tiếng Việt, intent-slot NLU và
runtime/evaluation. Project làm trọn luồng text đến mock action; STT/TTS và thiết bị thật được để lại vì
chúng cần corpus, quyền hệ điều hành và benchmark riêng. Pydantic, YAML, builder, CLI/API và evaluator
dùng chung contract để các phần không hiểu dữ liệu theo những cách khác nhau.

```text
transcript
  -> chuẩn hóa tiếng Việt và biến thể vùng miền đã biết
  -> phân loại intent
  -> confidence gate
  -> trích xuất slot theo intent
  -> action-safety gate
  -> kiểm tra slot bắt buộc
  -> mock/live-weather action, hỏi bổ sung hoặc từ chối an toàn
```

Snapshot dữ liệu có 2.094 câu: train 1.363, validation 347 và test 384. Test gồm 174 câu standard và
70 câu cho mỗi nhóm Bắc, Trung, Nam. MASSIVE giữ partition gốc; dữ liệu tự biên soạn được gom theo
`group_id`/template family trước khi chia. Validator kiểm tra trùng nguyên văn, trùng sau khi bỏ dấu,
near-duplicate và group leakage.

Train được dùng để fit model và tạo slot lexicon. Validation dành cho việc chọn candidate, rule và
confidence threshold. Test chỉ được mở sau khi những quyết định đó đã khóa. Cách tách này quan trọng
hơn việc cố lấy thêm vài điểm từ một tập dữ liệu vốn không lớn.

Test snapshot có SHA-256
`47cb9cf87cc53c5a210298453b4ae6ca75d045250c883bd5cca59709ddec9f2a`. Các tập normalization,
intent-boundary và action-safety là regression set do project tự viết; chúng ngăn lỗi quay lại nhưng
không thay thế test độc lập.

Trong model selection, mỗi candidate chỉ fit trên train rồi đo trên validation. Sau khi khóa candidate
và threshold, artifact dùng cho CLI/API mới được fit lại trên train + validation. Không đo artifact
này ngược trên validation rồi gọi đó là kết quả tổng quát hóa.

## 3. Lựa chọn kỹ thuật và phương án thay thế

### Intent classifier

Runtime dùng TF-IDF word/character n-gram + Logistic Regression. Word n-gram giữ tín hiệu cụm từ;
character n-gram hỗ trợ lỗi không dấu, lỗi gõ/STT và biến thể bề mặt. Logistic Regression fit nhanh
trên CPU, cho xác suất phục vụ confidence gate và tạo artifact nhỏ.

Ba candidate TF-IDF được chọn bằng macro-F1 trên validation vì năm intent cần được coi trọng ngang
nhau. Hai candidate có character n-gram cùng đạt validation macro-F1 **98,17%** và accuracy **98,27%**.
`word_1_2_char_3_5` thắng theo tie-break đã cố định trong config, không phải do xem test.

Tôi bắt đầu bằng classifier rule-based vì nó dễ giải thích, nhưng kết quả cho thấy rule đơn thuần không
đủ linh hoạt nên chỉ được giữ làm baseline. Thí nghiệm `multilingual-e5-small` kết hợp Logistic
Regression đạt validation macro-F1 **86,06%** trên cùng split, thấp hơn TF-IDF và nặng hơn đáng kể;
vì vậy E5 không đi vào runtime.

Word2Vec và GloVe cũng được cân nhắc, nhưng embedding tĩnh không cho lợi thế rõ với những command ngắn
này. Fine-tune PhoBERT hoặc một transformer có thể hợp lý khi có thêm câu thật và speaker thật; làm
ngay trên bộ dữ liệu hiện tại sẽ tăng chi phí và nguy cơ overfit. Tôi cũng không dùng ensemble vì chưa
có bằng chứng hai model sửa lỗi bổ sung cho nhau đủ để bù thêm latency và bộ nhớ.

Phần đóng góp không phải viết lại thuật toán tối ưu của scikit-learn. Công việc chính nằm ở contract,
ánh xạ dữ liệu, kiểm soát leakage, normalizer tiếng Việt, boundary policy, candidate sweep, slot
extractor, safety gate và một pipeline dùng chung cho CLI/API.

### Slot, normalizer và runtime

Slot dùng regex/rule theo intent thay vì sequence tagger. Với bảy slot và dữ liệu gán nhãn còn nhỏ,
cách này dễ audit và có dấu vết match. Đổi lại, entity ngoài catalog hoặc cách nói mới vẫn có thể bị bỏ
sót.

Normalizer deterministic thực hiện Unicode NFC, khoảng trắng, lỗi STT đã liệt kê và một số ánh xạ từ
vựng vùng miền. Rule dài chạy trước rule ngắn; tiểu từ như `hỉ`, `hen`, `nghen` chỉ được đổi khi ngữ
cảnh đủ rõ. Train và runtime dùng cùng một normalizer.

CLI và FastAPI gọi chung runtime factory. Model được nạp một lần trong API lifespan. Mock action giúp
test end-to-end mà không tạo side effect hay yêu cầu credential.

## 4. Xử lý tiếng Việt và biến thể vùng miền

MASSIVE không có nhãn accent, nên các câu nguồn này mang `region=standard`; project không tự đoán vùng
miền của chúng. Regional set là text template tự biên soạn để kiểm tra từ vựng địa phương, tiểu từ và
dạng không dấu.

Ví dụ, normalizer có thể đưa `Bữa ni ở Huế trời răng rồi hỉ` về `hôm nay ở huế trời sao rồi nhỉ` và
giữ lại danh sách rule đã match. Cách này có lợi cho những pattern đã định nghĩa, nhưng không chứng minh
hệ thống hiểu mọi phương ngữ.

Giới hạn quan trọng nhất là dữ liệu Bắc/Trung/Nam hiện chủ yếu đến từ synthetic template. Số sample
được native speaker review độc lập là 0 và số audio/call recording cũng là 0. Vì vậy region breakdown
chỉ đo lexical variation trong văn bản, không đo accent âm thanh.

Vì vậy tôi chỉ kết luận hệ thống xử lý được một tập pattern vùng miền đã khai báo. Muốn đánh giá ngoài
đời cần thu thập audio có consent, cân bằng vùng/tỉnh, tách speaker giữa split và báo riêng WER/CER của
STT với metric intent/slot của NLU.

## 5. Intent, slot và phương pháp đánh giá

### Ranh giới intent

Tôi định nghĩa `call_contact` là yêu cầu gọi ngay bằng tên liên hệ hoặc số điện thoại. Nếu việc gọi nằm
ở tương lai, hệ thống tạo `set_reminder`; `set_alarm` chỉ dùng khi mục tiêu là đánh thức hoặc cảnh báo
chính người dùng. Vì thế `gọi mẹ ngay đi` là `call_contact`, `6 giờ gọi mẹ` là `set_reminder`, còn
`gọi tôi dậy lúc 6 giờ` là `set_alarm`. Các câu ranh giới trong config là development regression,
không phải test cuối.

### Metric phát triển

Intent được báo bằng accuracy, precision/recall/F1 từng lớp, macro-F1, weighted-F1 và confusion matrix.
Report xác suất còn có log-loss, Brier, ECE và ROC/PR one-vs-rest vì runtime sử dụng confidence.

Slot được đo bằng oracle slot với gold intent để tách lỗi extractor khỏi classifier, và end-to-end slot
với intent thật sự do pipeline dự đoán.

Trên validation, oracle slot đạt exact match **92,22%** và micro-F1 **94,42%**. MASSIVE chỉ đạt
micro-F1 **83,49%**, trong khi synthetic đạt **98,35%**. Khoảng cách này cho thấy template dễ hơn câu
bản địa hóa tự nhiên và không nên chỉ trình bày con số tổng.

### Kết quả trên tập test

Tập test có 384 câu và runtime không được truyền region label. Kết quả của snapshot hiện tại là:

| Chỉ số | Kết quả |
|---|---:|
| Intent accuracy / macro-F1 | **92,71% / 92,25%** |
| Runtime intent accuracy | **90,36%** |
| Runtime coverage / selective accuracy | **92,45% / 97,75%** |
| Oracle slot exact / micro-F1 | **81,77% / 86,89%** |
| End-to-end slot exact / micro-F1 | **74,48% / 83,15%** |
| Full-command success | **73,96%** |

Con số intent cho biết classifier nhận đúng nhãn trong phần lớn câu. Khi chạy toàn pipeline, kết quả
thấp hơn vì hệ thống còn phải trích đúng slot, vượt qua safety gate và tạo được action hợp lệ. Oracle
slot micro-F1 **86,89%** nhưng end-to-end còn **83,15%**, cho thấy một phần lỗi slot bắt nguồn từ intent
hoặc quyết định runtime trước extractor.

Runtime intent accuracy là Bắc **81,43%**, Trung **88,57%**, Nam **94,29%** và nhóm không dấu
**76,85%**. Đây là kết quả trên text template và câu bản địa hóa, không phải phép đo accent từ audio.
Ngoài ra, các lỗi của test đã được dùng cho error analysis ở giai đoạn cuối; vì vậy bảng này phù hợp để
mô tả snapshot hiện tại nhưng không thay thế một holdout mới hoàn toàn độc lập.

Confidence gate không phải OOD detector. Logistic Regression closed-set luôn phân xác suất vào năm
label, nên action-safety gate vẫn cần tách riêng. Safety challenge 99 câu là tập regression được xây
cùng rule, không phải independent red-team benchmark hay bảo đảm an toàn production.

## 6. Ràng buộc on-device và vận hành

Luồng chính không gọi LLM hay inference API. TF-IDF, normalizer và regex extractor chạy CPU và được
nạp từ artifact cục bộ. Đây là lựa chọn thực dụng cho demo on-device, nhưng số đo hiện tại mới lấy trên
laptop Windows/Python 3.12.

Ở lần chạy hiện tại: pipeline build **58,37 ms**, median **6,72 ms**, p95 **10,15 ms** và throughput
tuần tự **137,87 lệnh/giây**. Nạp model làm RSS của tiến trình đã import thư viện tăng khoảng **2,28
MB**; intent artifact khoảng **514 KB**. Đây chỉ là một lần đo local, không phải benchmark pin, cold
start hoặc footprint native trên điện thoại.

Nếu đưa lên thiết bị thật, tôi sẽ bỏ FastAPI/Python server khỏi đường chạy, đóng gói model theo runtime
của nền tảng, cache các artifact và benchmark cold start, RAM, pin trên đúng máy đích. Normalizer và
regex có thể giữ lại; STT offline chỉ được chọn sau khi đo WER/CER và tác động của nó lên intent/slot.

Runtime khóa scikit-learn 1.9.0 vì artifact phụ thuộc phiên bản; E5 chỉ nằm trong extra `semantic`.
Joblib không phải định dạng an toàn: SHA-256 giúp phát hiện artifact bị thay, không sandbox file lạ.

## 7. Phần khó nhất và giới hạn còn lại

Phần khó nhất không phải gọi hàm train, mà là quyết định ranh giới giữa `set_alarm`, `set_reminder` và
`call_contact` khi cả ba có thể chứa từ “gọi” hoặc một mốc thời gian. Normalizer cũng phải đủ mạnh để
xử lý cách nói vùng miền nhưng không được sửa nhầm tên người hay tên bài hát. Ở phía dữ liệu, các
template gần giống phải nằm trong cùng split và slot lexicon không được học từ validation/test.

Hệ thống hiện vẫn là closed-set classifier với slot extractor dựa trên regex và catalog. Confidence
không thay thế được OOD detection, còn safety set được xây cùng rule nên chưa phải independent red-team
benchmark. Full-command success hiện tại là **73,96%**, nên vẫn còn khoảng cách đáng kể giữa việc nhận
đúng intent và hoàn thành trọn vẹn một command.

Giới hạn lớn nhất vẫn là chưa có audio hoặc native-speaker benchmark. Mock action cũng chưa kiểm tra
quyền thiết bị, authentication, privacy hay điều kiện vận hành production, và số đo hiệu năng chưa được
chạy trên thiết bị di động đích.

## 8. Phần chưa làm và hướng tiếp tục

Trong phạm vi challenge, pipeline text-first, kiểm tra dataset, intent classifier, slot extraction,
CLI/API, mock action và evaluator đã hoàn thành. STT/TTS, thiết bị thật và production security chưa
hoàn thành và không được mô tả như tính năng sẵn sàng sử dụng. Bảng số liệu ở mục 5 có thể tái tạo bằng
`scripts/evaluate.py`.

Nếu tiếp tục, việc đầu tiên tôi làm sẽ là thu một development/test set mới gồm câu tự nhiên và có người
nói ba vùng rà soát, thay vì tiếp tục chỉnh theo failure của test cũ. Sau đó mới cải thiện action gate
và đánh giá trên một holdout mới. Phase audio cần consent, speaker-disjoint split và phép đo WER/CER
riêng trước khi nối STT với NLU.

PhoBERT hoặc joint intent-slot chỉ đáng thử khi dữ liệu mới và error analysis cho thấy baseline nhẹ đã
chạm trần. Authentication, privacy control và adapter thiết bị là bước sau cùng nếu project chuyển từ
demo challenge sang một sản phẩm thực tế.

TTS chỉ cần khi sản phẩm yêu cầu phản hồi bằng âm thanh; weather adapter và mock action không phải bằng chứng hệ thống đã sẵn sàng production.
