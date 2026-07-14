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
    assert body["response"] == "Cần bổ sung slot bắt buộc: datetime."


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "   "},
        {"text": "a" * 501},
        {"text": "gọi cho mẹ", "region_hint": "west"},
    ],
)
def test_parse_rejects_invalid_request(client: TestClient, payload: dict[str, str]) -> None:
    response = client.post("/parse", json=payload)

    assert response.status_code == 422
