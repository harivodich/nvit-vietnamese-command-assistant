from pathlib import Path
from typing import Any

import pytest

from nvit_assistant.actions import IntegratedActionRouter, OpenMeteoWeatherClient
from nvit_assistant.schemas import ActionStatus, Intent


ROOT = Path(__file__).resolve().parents[1]


def fake_open_meteo(url: str, timeout_seconds: float) -> dict[str, Any]:
    assert timeout_seconds == 1.0
    if "geocoding-api" in url:
        return {
            "results": [
                {
                    "name": "Huế",
                    "latitude": 16.4619,
                    "longitude": 107.5955,
                }
            ]
        }
    return {
        "current": {
            "temperature_2m": 31.0,
            "apparent_temperature": 35.5,
            "relative_humidity_2m": 70,
            "precipitation": 0.0,
            "weather_code": 2,
        },
        "daily": {
            "time": ["2026-07-15", "2026-07-16", "2026-07-17"],
            "weather_code": [2, 61, 3],
            "temperature_2m_max": [34.0, 33.0, 32.0],
            "temperature_2m_min": [26.0, 25.0, 25.0],
            "precipitation_probability_max": [20, 70, 40],
        },
    }


def build_integrated_router() -> IntegratedActionRouter:
    client = OpenMeteoWeatherClient(timeout_seconds=1.0, fetch_json=fake_open_meteo)
    return IntegratedActionRouter(
        ROOT / "data" / "fake_contacts.json",
        ROOT / "data" / "music_catalog.json",
        client,
    )


def test_live_weather_returns_completed_result_with_attribution() -> None:
    execution = build_integrated_router().execute(
        Intent.ASK_WEATHER,
        {"location": "huế", "datetime": "ngày mai"},
    )

    assert execution.result.status is ActionStatus.COMPLETED
    assert execution.result.payload["provider"] == "Open-Meteo"
    assert execution.result.payload["target"] == "ngày mai"
    assert execution.result.payload["condition"] == "có mưa"
    assert execution.result.payload["attribution_url"] == "https://open-meteo.com/"
    assert "25–33°C" in execution.response


def test_live_weather_asks_for_location_instead_of_guessing() -> None:
    execution = build_integrated_router().execute(Intent.ASK_WEATHER, {})

    assert execution.result.status is ActionStatus.UNAVAILABLE
    assert execution.result.payload == {"reason": "missing_location"}
    assert execution.response == "Bạn muốn xem thời tiết ở địa điểm nào?"


@pytest.mark.parametrize("location", ["đâu", "đây", "chỗ nào", "nơi nào"])
def test_live_weather_does_not_send_placeholder_to_provider(location: str) -> None:
    execution = build_integrated_router().execute(
        Intent.ASK_WEATHER, {"location": location}
    )

    assert execution.result.status is ActionStatus.UNAVAILABLE
    assert execution.result.payload == {"reason": "missing_location"}


def test_fake_contact_is_resolved_but_never_called() -> None:
    execution = build_integrated_router().execute(
        Intent.CALL_CONTACT,
        {"contact_name": "má"},
    )

    assert execution.result.status is ActionStatus.MOCKED
    assert execution.result.payload["resolved_name"] == "mẹ"
    assert execution.result.payload["resolved_phone_number"] == "000 000 0001"
    assert "chưa thực hiện cuộc gọi thật" in execution.response


def test_music_catalog_is_looked_up_without_playing_audio() -> None:
    execution = build_integrated_router().execute(
        Intent.PLAY_MUSIC,
        {"song": "lạc trôi", "artist": "sơn tùng"},
    )

    assert execution.result.status is ActionStatus.MOCKED
    assert execution.result.payload["catalog_matches"] == [
        {"song": "lạc trôi", "artist": "sơn tùng"}
    ]
    assert "chưa phát trên thiết bị thật" in execution.response


def test_weather_api_failure_becomes_unavailable_action() -> None:
    def no_location(url: str, timeout_seconds: float) -> dict[str, Any]:
        return {"results": []}

    router = IntegratedActionRouter(
        ROOT / "data" / "fake_contacts.json",
        ROOT / "data" / "music_catalog.json",
        OpenMeteoWeatherClient(timeout_seconds=1.0, fetch_json=no_location),
    )

    execution = router.execute(Intent.ASK_WEATHER, {"location": "không tồn tại"})

    assert execution.result.status is ActionStatus.UNAVAILABLE
    assert "không tìm thấy địa điểm" in execution.result.payload["reason"]
