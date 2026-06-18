import requests
from .config import settings

class APIError(RuntimeError):
    pass

def api_football_get(path: str, params: dict | None = None) -> dict:
    if not settings.api_football_key:
        raise APIError("Falta API_FOOTBALL_KEY en .env")

    url = f"{settings.api_football_base_url}{path}"
    headers = {"x-apisports-key": settings.api_football_key}
    response = requests.get(url, headers=headers, params=params or {}, timeout=25)

    if response.status_code >= 400:
        raise APIError(f"API-Football error {response.status_code}: {response.text[:300]}")

    return response.json()

def football_data_get(path: str, params: dict | None = None) -> dict:
    if not settings.football_data_token:
        raise APIError("Falta FOOTBALL_DATA_TOKEN en .env")

    url = f"{settings.football_data_base_url}{path}"
    headers = {"X-Auth-Token": settings.football_data_token}
    response = requests.get(url, headers=headers, params=params or {}, timeout=25)

    if response.status_code >= 400:
        raise APIError(f"football-data.org error {response.status_code}: {response.text[:300]}")

    return response.json()

def open_meteo_forecast(lat: float, lon: float, date_iso: str | None = None) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m",
        "timezone": "America/Guatemala",
    }

    if date_iso:
        params["start_date"] = date_iso
        params["end_date"] = date_iso

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=25,
    )
    response.raise_for_status()
    return response.json()
