from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nvit_assistant.api import create_app
from nvit_assistant.runtime import build_pipeline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Khởi động API bằng pipeline thật nhưng mọi action vẫn chỉ là mock."""
    with TestClient(create_app(build_pipeline(ROOT))) as test_client:
        yield test_client


def test_health_reports_ready_mock_mode(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "mode": "mock-actions"}


def test_parse_returns_end_to_end_action(client: TestClient) -> None:
    response = client.post("/parse", json={"text": "gọi cho mẹ"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "call_contact"
    assert body["slots"] == {"contact_name": "mẹ"}
    assert body["action"]["type"] == "call"
    assert body["action"]["status"] == "mocked"
    assert body["action"]["payload"]["target"] == "mẹ"


def test_parse_does_not_execute_when_required_slot_is_missing(client: TestClient) -> None:
    response = client.post("/parse", json={"text": "đặt báo thức giúp tôi"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "set_alarm"
    assert body["action"] is None
    assert body["response"] == "Bạn muốn đặt báo thức lúc nào?"


@pytest.mark.parametrize(
    ("text", "expected_slots", "expected_response"),
    [
        ("nhắc tôi", {}, "Bạn muốn mình nhắc việc gì?"),
        ("gọi giúp tôi", {}, "Bạn muốn gọi cho ai hoặc gọi tới số điện thoại nào?"),
        ("gọi tôi dậy", {}, "Bạn muốn đặt báo thức lúc nào?"),
        ("nhắc tôi lúc 8 giờ", {"datetime": "8 giờ"}, "Bạn muốn mình nhắc việc gì?"),
        ("gọi cho ai", {}, "Bạn muốn gọi cho ai hoặc gọi tới số điện thoại nào?"),
    ],
)
def test_parse_clarifies_trigger_only_commands(
    client: TestClient,
    text: str,
    expected_slots: dict[str, str],
    expected_response: str,
) -> None:
    response = client.post("/parse", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] is None
    assert body["slots"] == expected_slots
    assert body["response"] == expected_response


def test_parse_blocks_negated_command(client: TestClient) -> None:
    response = client.post("/parse", json={"text": "đừng gọi cho mẹ"})

    assert response.status_code == 200
    assert response.json()["intent"] == "unknown"
    assert response.json()["action"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "   "},
        {"text": "a" * 501},
        {"text": "gọi cho mẹ", "region_hint": "west"},
        {"text": "gọi cho mẹ", "region": "south"},
    ],
)
def test_parse_rejects_invalid_request(client: TestClient, payload: dict[str, str]) -> None:
    response = client.post("/parse", json=payload)

    assert response.status_code == 422


def test_production_lifespan_builds_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    pipeline = build_pipeline(ROOT)

    def fake_build_pipeline() -> object:
        nonlocal calls
        calls += 1
        return pipeline

    monkeypatch.setattr("nvit_assistant.api.build_pipeline", fake_build_pipeline)
    with TestClient(create_app()) as production_client:
        assert production_client.get("/health").json()["status"] == "ready"
        assert production_client.post("/parse", json={"text": "gọi cho mẹ"}).status_code == 200

    assert calls == 1


def test_production_lifespan_fails_fast_when_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build_pipeline() -> object:
        raise FileNotFoundError("missing model artifact")

    monkeypatch.setattr("nvit_assistant.api.build_pipeline", fail_build_pipeline)
    with pytest.raises(FileNotFoundError, match="missing model artifact"):
        with TestClient(create_app()):
            pass
