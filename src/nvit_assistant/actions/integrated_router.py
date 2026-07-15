"""Action demo có weather thật và dữ liệu danh bạ/âm nhạc hoàn toàn giả lập."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nvit_assistant.actions.router import ActionExecution, MockActionRouter
from nvit_assistant.schemas import ActionResult, ActionStatus, ActionType, Intent, SlotName


OPEN_METEO_ATTRIBUTION_URL = "https://open-meteo.com/"
GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
LOCATION_PLACEHOLDERS = frozenset(
    {"đâu", "dau", "đây", "day", "chỗ nào", "cho nao", "nơi nào", "noi nao"}
)


class WeatherServiceError(RuntimeError):
    """Lỗi có thể hiển thị gọn khi API thời tiết hoặc dữ liệu trả về không dùng được."""


JsonFetcher = Callable[[str, float], dict[str, Any]]


def _fetch_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    """Gọi HTTP GET bằng standard library để không thêm dependency runtime chỉ vì demo."""
    request = Request(url, headers={"User-Agent": "nvit-command-assistant/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload: Any = json.load(response)
    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        raise WeatherServiceError("không kết nối được dịch vụ thời tiết") from error
    if not isinstance(payload, dict):
        raise WeatherServiceError("dịch vụ thời tiết trả dữ liệu không hợp lệ")
    return payload


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Đọc một JSON object và báo rõ field nào sai cấu trúc."""
    if not isinstance(value, dict):
        raise WeatherServiceError(f"weather response thiếu object {field_name}")
    return value


def _number(value: Any, field_name: str) -> float:
    """Đọc số từ response và chặn bool vì bool là lớp con của int trong Python."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WeatherServiceError(f"weather response thiếu số {field_name}")
    return float(value)


def _integer(value: Any, field_name: str) -> int:
    """Đọc mã thời tiết nguyên từ response."""
    number = _number(value, field_name)
    if not number.is_integer():
        raise WeatherServiceError(f"weather response có {field_name} không phải số nguyên")
    return int(number)


def _daily_value(daily: dict[str, Any], field_name: str, index: int) -> Any:
    """Đọc một phần tử daily forecast theo ngày và kiểm tra giới hạn mảng."""
    values = daily.get(field_name)
    if not isinstance(values, list) or index >= len(values):
        raise WeatherServiceError(f"weather response thiếu daily.{field_name}[{index}]")
    return values[index]


def _format_number(value: float) -> str:
    """Hiển thị số gọn, ví dụ 28 thay vì 28.0."""
    return f"{value:g}"


def weather_condition(code: int) -> str:
    """Ánh xạ nhóm WMO weather code sang mô tả tiếng Việt ngắn."""
    if code == 0:
        return "trời quang"
    if code in {1, 2, 3}:
        return "có mây"
    if code in {45, 48}:
        return "có sương mù"
    if code in {51, 53, 55, 56, 57}:
        return "có mưa phùn"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "có mưa"
    if code in {71, 73, 75, 77, 85, 86}:
        return "có tuyết"
    if code in {95, 96, 99}:
        return "có dông"
    return "thời tiết chưa xác định"


@dataclass(frozen=True)
class WeatherReport:
    """Dữ liệu máy đọc được và câu trả lời đã ghi nguồn Open-Meteo."""

    payload: dict[str, Any]
    response: str


class OpenMeteoWeatherClient:
    """Tra tọa độ rồi lấy current/daily forecast từ hai endpoint Open-Meteo."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        fetch_json: JsonFetcher = _fetch_json,
    ) -> None:
        if not 0.0 < timeout_seconds <= 30.0:
            raise ValueError("weather timeout phải nằm trong (0, 30]")
        self.timeout_seconds = timeout_seconds
        self.fetch_json = fetch_json

    def query(self, location: str, datetime: str | None) -> WeatherReport:
        """Trả thời tiết hiện tại, hôm nay hoặc ngày mai cho một địa danh Việt Nam."""
        place_name, latitude, longitude = self._geocode(location)
        forecast = self._forecast(latitude, longitude)
        if datetime is None or any(word in datetime for word in ("bây giờ", "hiện tại")):
            return self._current_report(place_name, latitude, longitude, forecast)
        if "mai" in datetime:
            return self._daily_report(
                place_name, latitude, longitude, forecast, index=1, label="ngày mai"
            )
        if any(word in datetime for word in ("hôm nay", "nay")):
            return self._daily_report(
                place_name, latitude, longitude, forecast, index=0, label="hôm nay"
            )
        raise WeatherServiceError(
            "demo weather thật hiện chỉ hỗ trợ hiện tại, hôm nay hoặc ngày mai"
        )

    def _geocode(self, location: str) -> tuple[str, float, float]:
        query = urlencode(
            {
                "name": location,
                "count": 1,
                "language": "vi",
                "countryCode": "VN",
                "format": "json",
            }
        )
        payload = self.fetch_json(f"{GEOCODING_ENDPOINT}?{query}", self.timeout_seconds)
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise WeatherServiceError(f"không tìm thấy địa điểm “{location}” tại Việt Nam")
        first = _mapping(results[0], "geocoding.results[0]")
        name = first.get("name")
        if not isinstance(name, str) or not name.strip():
            raise WeatherServiceError("geocoding response thiếu tên địa điểm")
        return (
            name,
            _number(first.get("latitude"), "latitude"),
            _number(first.get("longitude"), "longitude"),
        )

    def _forecast(self, latitude: float, longitude: float) -> dict[str, Any]:
        query = urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": 3,
                "timezone": "auto",
            }
        )
        return self.fetch_json(f"{FORECAST_ENDPOINT}?{query}", self.timeout_seconds)

    def _current_report(
        self,
        place_name: str,
        latitude: float,
        longitude: float,
        forecast: dict[str, Any],
    ) -> WeatherReport:
        current = _mapping(forecast.get("current"), "current")
        temperature = _number(current.get("temperature_2m"), "current.temperature_2m")
        apparent = _number(
            current.get("apparent_temperature"), "current.apparent_temperature"
        )
        humidity = _number(
            current.get("relative_humidity_2m"), "current.relative_humidity_2m"
        )
        precipitation = _number(current.get("precipitation"), "current.precipitation")
        code = _integer(current.get("weather_code"), "current.weather_code")
        payload = {
            "provider": "Open-Meteo",
            "attribution_url": OPEN_METEO_ATTRIBUTION_URL,
            "location": place_name,
            "latitude": latitude,
            "longitude": longitude,
            "target": "hiện tại",
            "temperature_c": temperature,
            "apparent_temperature_c": apparent,
            "relative_humidity_percent": humidity,
            "precipitation_mm": precipitation,
            "weather_code": code,
            "condition": weather_condition(code),
        }
        response = (
            f"Hiện tại ở {place_name} {weather_condition(code)}, "
            f"{_format_number(temperature)}°C, cảm giác như {_format_number(apparent)}°C. "
            f"Dữ liệu: Open-Meteo ({OPEN_METEO_ATTRIBUTION_URL})."
        )
        return WeatherReport(payload=payload, response=response)

    def _daily_report(
        self,
        place_name: str,
        latitude: float,
        longitude: float,
        forecast: dict[str, Any],
        index: int,
        label: str,
    ) -> WeatherReport:
        daily = _mapping(forecast.get("daily"), "daily")
        date = _daily_value(daily, "time", index)
        if not isinstance(date, str):
            raise WeatherServiceError(f"weather response có daily.time[{index}] không phải chuỗi")
        minimum = _number(
            _daily_value(daily, "temperature_2m_min", index),
            f"daily.temperature_2m_min[{index}]",
        )
        maximum = _number(
            _daily_value(daily, "temperature_2m_max", index),
            f"daily.temperature_2m_max[{index}]",
        )
        rain_probability = _number(
            _daily_value(daily, "precipitation_probability_max", index),
            f"daily.precipitation_probability_max[{index}]",
        )
        code = _integer(
            _daily_value(daily, "weather_code", index),
            f"daily.weather_code[{index}]",
        )
        payload = {
            "provider": "Open-Meteo",
            "attribution_url": OPEN_METEO_ATTRIBUTION_URL,
            "location": place_name,
            "latitude": latitude,
            "longitude": longitude,
            "target": label,
            "date": date,
            "temperature_min_c": minimum,
            "temperature_max_c": maximum,
            "precipitation_probability_max_percent": rain_probability,
            "weather_code": code,
            "condition": weather_condition(code),
        }
        response = (
            f"{label.capitalize()} ở {place_name} {weather_condition(code)}, "
            f"nhiệt độ {_format_number(minimum)}–{_format_number(maximum)}°C, "
            f"khả năng mưa cao nhất {_format_number(rain_probability)}%. "
            f"Dữ liệu: Open-Meteo ({OPEN_METEO_ATTRIBUTION_URL})."
        )
        return WeatherReport(payload=payload, response=response)


def _load_records(path: Path, field_name: str) -> list[dict[str, Any]]:
    """Nạp list record từ data JSON và fail-fast nếu file demo sai schema."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"{path} phải có schema_version=1")
    records = raw.get(field_name)
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"{path} phải chứa list {field_name}")
    return records


def _record_string(record: dict[str, Any], field_name: str, source: Path) -> str:
    """Đọc chuỗi bắt buộc trong record catalog."""
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: field {field_name} phải là chuỗi không rỗng")
    return value.strip()


def _key(value: str) -> str:
    """Chuẩn hóa khóa lookup vì slot runtime đã được lower-case và chuẩn khoảng trắng."""
    return " ".join(value.casefold().split())


class IntegratedActionRouter:
    """Weather thật; danh bạ và nhạc chỉ tra data demo, không chạm thiết bị."""

    mode = "live-weather"

    def __init__(
        self,
        contacts_path: Path,
        music_catalog_path: Path,
        weather_client: OpenMeteoWeatherClient | None = None,
    ) -> None:
        self.fallback = MockActionRouter()
        self.weather_client = weather_client or OpenMeteoWeatherClient()
        self.contacts = self._load_contacts(contacts_path)
        self.music_catalog = self._load_music(music_catalog_path)

    def execute(self, intent: Intent, slots: dict[str, Any]) -> ActionExecution:
        """Điều phối weather thật; các action còn lại vẫn giữ trạng thái mocked rõ ràng."""
        if intent is Intent.ASK_WEATHER:
            return self._query_weather(slots)
        if intent is Intent.CALL_CONTACT:
            return self._resolve_fake_contact(slots)
        if intent is Intent.PLAY_MUSIC:
            return self._select_demo_music(slots)
        return self.fallback.execute(intent, slots)

    def _query_weather(self, slots: dict[str, Any]) -> ActionExecution:
        location = slots.get(SlotName.LOCATION.value)
        datetime = slots.get(SlotName.DATETIME.value)
        if (
            not isinstance(location, str)
            or not location.strip()
            or _key(location) in LOCATION_PLACEHOLDERS
        ):
            return ActionExecution(
                result=ActionResult(
                    type=ActionType.QUERY_WEATHER,
                    status=ActionStatus.UNAVAILABLE,
                    payload={"reason": "missing_location"},
                ),
                response="Bạn muốn xem thời tiết ở địa điểm nào?",
            )
        time_text = datetime if isinstance(datetime, str) else None
        try:
            report = self.weather_client.query(location, time_text)
        except WeatherServiceError as error:
            return ActionExecution(
                result=ActionResult(
                    type=ActionType.QUERY_WEATHER,
                    status=ActionStatus.UNAVAILABLE,
                    payload={
                        "location": location,
                        "datetime": time_text,
                        "provider": "Open-Meteo",
                        "attribution_url": OPEN_METEO_ATTRIBUTION_URL,
                        "reason": str(error),
                    },
                ),
                response=f"Chưa lấy được thời tiết: {error}.",
            )
        return ActionExecution(
            result=ActionResult(
                type=ActionType.QUERY_WEATHER,
                status=ActionStatus.COMPLETED,
                payload=report.payload,
            ),
            response=report.response,
        )

    def _resolve_fake_contact(self, slots: dict[str, Any]) -> ActionExecution:
        execution = self.fallback.execute(Intent.CALL_CONTACT, slots)
        contact = slots.get(SlotName.CONTACT_NAME.value)
        if not isinstance(contact, str) or _key(contact) not in self.contacts:
            return execution
        record = self.contacts[_key(contact)]
        payload = {
            **execution.result.payload,
            "resolved_name": record["name"],
            "resolved_phone_number": record["phone_number"],
            "data_source": "fake_contacts",
        }
        return ActionExecution(
            result=ActionResult(
                type=ActionType.CALL,
                status=ActionStatus.MOCKED,
                payload=payload,
            ),
            response=(
                f"Đã tìm thấy {record['name']} trong danh bạ giả lập "
                f"({record['phone_number']}); chưa thực hiện cuộc gọi thật."
            ),
        )

    def _select_demo_music(self, slots: dict[str, Any]) -> ActionExecution:
        execution = self.fallback.execute(Intent.PLAY_MUSIC, slots)
        song = slots.get(SlotName.SONG.value)
        artist = slots.get(SlotName.ARTIST.value)
        song_key = _key(song) if isinstance(song, str) else None
        artist_key = _key(artist) if isinstance(artist, str) else None
        matches = [
            item
            for item in self.music_catalog
            if (song_key is None or _key(item["song"]) == song_key)
            and (artist_key is None or _key(item["artist"]) == artist_key)
        ]
        if not matches or (song_key is None and artist_key is None):
            return execution
        payload = {
            **execution.result.payload,
            "catalog_matches": matches,
            "data_source": "music_catalog",
        }
        if song_key is not None:
            response = (
                f"Đã chọn “{matches[0]['song']}” của {matches[0]['artist']} "
                "trong catalog demo; chưa phát trên thiết bị thật."
            )
        else:
            response = (
                f"Catalog demo có {len(matches)} bài của {matches[0]['artist']}; "
                "chưa phát trên thiết bị thật."
            )
        return ActionExecution(
            result=ActionResult(
                type=ActionType.PLAY_MUSIC,
                status=ActionStatus.MOCKED,
                payload=payload,
            ),
            response=response,
        )

    @staticmethod
    def _load_contacts(path: Path) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        for record in _load_records(path, "contacts"):
            name = _record_string(record, "name", path)
            phone = _record_string(record, "phone_number", path)
            aliases = record.get("aliases", [])
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) and alias.strip() for alias in aliases
            ):
                raise ValueError(f"{path}: aliases phải là list chuỗi không rỗng")
            normalized = {"name": name, "phone_number": phone}
            for alias in [name, *aliases]:
                lookup = _key(alias)
                if lookup in index:
                    raise ValueError(f"{path}: alias danh bạ bị trùng: {alias}")
                index[lookup] = normalized
        return index

    @staticmethod
    def _load_music(path: Path) -> tuple[dict[str, str], ...]:
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for record in _load_records(path, "tracks"):
            item = {
                "song": _record_string(record, "song", path),
                "artist": _record_string(record, "artist", path),
            }
            identity = (_key(item["song"]), _key(item["artist"]))
            if identity in seen:
                raise ValueError(f"{path}: bài hát bị trùng: {item}")
            seen.add(identity)
            result.append(item)
        return tuple(result)
