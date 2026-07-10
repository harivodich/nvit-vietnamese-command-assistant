# NVIT Vietnamese Command Assistant

Repository triển khai hệ thống hiểu lệnh nói tiếng Việt theo hướng **text-first NLU**. Dự án được xây dựng theo từng phase, với kiểm thử và đánh giá rõ ràng ở mỗi giai đoạn.

## Mục tiêu cuối cùng

Input chính là transcript tiếng Việt; audio chỉ là phần mở rộng sau cùng. Hệ thống sẽ trả về intent, slots, confidence, mock action và câu phản hồi. Core không dùng LLM runtime hay external API.

Intent dự kiến: `set_reminder`, `set_alarm`, `ask_weather`, `play_music`, `call_contact`.

Slots dự kiến: `datetime`, `location`, `song`, `artist`, `contact_name`, `reminder_text`.

```text
Text transcript (hoặc audio adapter)
  -> Vietnamese normalization + regional handling
  -> intent classification
  -> slot extraction
  -> confidence gate
  -> mock action + CLI/API response
```

## Kiến trúc

```text
configs/                         cấu hình intent, vùng miền và data templates
data/raw_sources/                nguồn tham khảo chưa gán nhãn
data/samples/                    JSONL train/test đã được kiểm tra
models/                          model artifact cục bộ
reports/                         báo cáo evaluation
scripts/                         generate, validate, train, evaluate, demo
src/nvit_assistant/
  actions/                       mock action router
  asr/                           optional audio-to-text adapter
  eval/                          metric và evaluation runner
  nlu/                           normalizer, classifier, slots, pipeline
tests/                           unit và integration tests
```

## Kế hoạch triển khai

1. Contract: schema, kiểu dữ liệu và schema tests.
2. Config + dataset nhỏ + JSONL validator.
3. Vietnamese normalization và regional variants.
4. Rule-based intent classifier, sau đó classifier trainable nhẹ.
5. Regex slot extraction và end-to-end pipeline.
6. Mock actions, CLI, FastAPI và integration tests.
7. Metrics/evaluation; ASR chỉ thêm khi NLU core ổn định.

Mỗi phase xác định rõ input/output, implementation, test và tiêu chí hoàn thành trước khi chuyển sang phase tiếp theo.
