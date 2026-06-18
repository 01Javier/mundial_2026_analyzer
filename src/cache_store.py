from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache")
METADATA_FILE = "cache_metadata"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(name: str, ext: str = "json") -> Path:
    ensure_cache_dir()
    safe_name = name.replace("/", "_").replace("\\", "_")
    safe_ext = ext.lstrip(".")
    return CACHE_DIR / f"{safe_name}.{safe_ext}"


def get_cache_age_minutes(path: Path) -> float | None:
    if not path.exists():
        return None
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (utc_now() - modified_at).total_seconds() / 60


def cache_is_fresh(path: Path, max_age_minutes: int) -> bool:
    age = get_cache_age_minutes(path)
    return age is not None and age <= max_age_minutes


def load_json_cache(name: str, max_age_minutes: int | None = None):
    path = cache_path(name, "json")
    if not path.exists():
        return None
    if max_age_minutes is not None and not cache_is_fresh(path, max_age_minutes):
        return None

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json_cache(name: str, data: dict):
    path = cache_path(name, "json")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    return path


def load_csv_cache(name: str, max_age_minutes: int | None = None) -> pd.DataFrame | None:
    path = cache_path(name, "csv")
    if not path.exists():
        return None
    if max_age_minutes is not None and not cache_is_fresh(path, max_age_minutes):
        return None
    return pd.read_csv(path)


def save_csv_cache(name: str, df: pd.DataFrame):
    path = cache_path(name, "csv")
    df.to_csv(path, index=False)
    return path


def read_cache_metadata() -> dict:
    path = cache_path(METADATA_FILE, "json")
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def update_cache_metadata(key: str, metadata: dict):
    current = read_cache_metadata()
    current[key] = {
        **current.get(key, {}),
        **metadata,
        "updated_at": utc_now().isoformat(),
    }

    path = cache_path(METADATA_FILE, "json")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(current, fh, ensure_ascii=False, indent=2, default=str)
    return path


def metadata_for_cache(
    provider: str,
    endpoint: str,
    params: dict,
    source_file: Path,
    ttl_minutes: int,
    last_error: str | None = None,
) -> dict:
    cached_at = utc_now()
    return {
        "provider": provider,
        "endpoint": endpoint,
        "params": params,
        "cached_at": cached_at.isoformat(),
        "expires_at": (cached_at + timedelta(minutes=ttl_minutes)).isoformat(),
        "source_file": str(source_file),
        "requests_saved": 0,
        "last_error": last_error,
    }


def increment_requests_saved(key: str):
    metadata = read_cache_metadata().get(key, {})
    requests_saved = int(metadata.get("requests_saved", 0) or 0) + 1
    update_cache_metadata(key, {"requests_saved": requests_saved, "last_error": None})
