from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cache_store import cache_path, get_cache_age_minutes, load_json_cache, save_json_cache, update_cache_metadata
from .clients import api_football_get
from .config import settings
from .team_registry import backup_csv, canonicalize_team_name, load_worldcup_teams

DATA_DIR = Path("data")
TEAM_FORM_PATH = DATA_DIR / "team_recent_form.csv"
H2H_PATH = DATA_DIR / "h2h.csv"
INJURIES_PATH = DATA_DIR / "injuries.csv"

FORM_EXTRA_COLUMNS = [
    "clean_sheets",
    "failed_to_score",
    "over_2_5_rate",
    "both_teams_score_rate",
    "source",
    "last_updated",
    "is_estimated",
    "confidence",
]


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm(value):
    return str(value or "").strip().lower()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _registry_team_id(team: str) -> int | None:
    registry = load_worldcup_teams()
    if registry.empty or "api_football_team_id" not in registry.columns:
        return None
    canonical = canonicalize_team_name(team)
    row = registry[registry["canonical_name"].map(lambda value: _norm(value) == _norm(canonical))]
    if row.empty:
        return None
    try:
        value = row.iloc[0].get("api_football_team_id")
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _save_api_cache(name: str, payload: dict, endpoint: str, params: dict, requests_used: int = 1):
    path = save_json_cache(name, payload)
    update_cache_metadata(
        name,
        {
            "provider": "API-Football",
            "endpoint": endpoint,
            "params": params,
            "mode": "api_live",
            "requests_used_now": requests_used,
            "cached_at": _now_iso(),
            "cache_age_minutes": 0,
            "source_file": str(path),
            "last_error": None,
        },
    )
    return path


def _cache_minutes() -> int:
    return int(settings.team_form_cache_hours * 60)


def _load_cached_api_payload(name: str):
    payload = load_json_cache(name, max_age_minutes=_cache_minutes())
    if payload:
        update_cache_metadata(
            name,
            {
                "mode": "cache_fresh",
                "requests_used_now": 0,
                "cache_age_minutes": get_cache_age_minutes(cache_path(name, "json")),
                "last_error": None,
            },
        )
    return payload


def _team_id_from_search_payload(raw: dict) -> int | None:
    response = raw.get("response", []) if raw else []
    if not response:
        return None
    try:
        return int(response[0]["team"]["id"])
    except Exception:
        return None


def _search_api_football_team_id_with_count(team: str) -> tuple[int | None, int]:
    existing = _registry_team_id(team)
    if existing:
        return existing, 0

    endpoint = "/teams"
    params = {"search": canonicalize_team_name(team)}
    cache_name = f"api_football_team_search_{canonicalize_team_name(team)}"
    cached = _load_cached_api_payload(cache_name)
    cached_id = _team_id_from_search_payload(cached)
    if cached_id:
        return cached_id, 0

    raw = api_football_get(endpoint, params=params)
    _save_api_cache(cache_name, raw, endpoint, params)
    return _team_id_from_search_payload(raw), 1


def search_api_football_team_id(team: str) -> int | None:
    team_id, _ = _search_api_football_team_id_with_count(team)
    return team_id


def _fetch_recent_fixtures_for_team_with_count(team_id: int, last: int = 10) -> tuple[list[dict], int]:
    endpoint = "/fixtures"
    params = {"team": team_id, "last": last}
    cache_name = f"api_football_team_{team_id}_last_{last}"
    cached = _load_cached_api_payload(cache_name)
    if cached:
        return cached.get("response", []), 0

    raw = api_football_get(endpoint, params=params)
    _save_api_cache(cache_name, raw, endpoint, params)
    return raw.get("response", []), 1


def fetch_recent_fixtures_for_team(team_id: int, last: int = 10):
    fixtures, _ = _fetch_recent_fixtures_for_team_with_count(team_id, last=last)
    return fixtures


def calculate_team_form_from_api_football(team: str, team_id: int, fixtures: list[dict]) -> dict:
    rows = []
    canonical = canonicalize_team_name(team)
    for item in fixtures:
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home = teams.get("home", {}) or {}
        away = teams.get("away", {}) or {}
        is_home = int(home.get("id") or 0) == int(team_id)
        is_away = int(away.get("id") or 0) == int(team_id)
        if not (is_home or is_away):
            continue
        gf = goals.get("home") if is_home else goals.get("away")
        ga = goals.get("away") if is_home else goals.get("home")
        if gf is None or ga is None:
            continue
        rows.append({"gf": int(gf), "ga": int(ga), "is_home": is_home})

    matches = len(rows)
    if matches == 0:
        return {
            "team": canonical,
            "matches": 0,
            "source": "API-Football",
            "last_updated": _now_iso(),
            "is_estimated": False,
            "confidence": "low",
        }

    gf_total = sum(row["gf"] for row in rows)
    ga_total = sum(row["ga"] for row in rows)
    wins = sum(row["gf"] > row["ga"] for row in rows)
    draws = sum(row["gf"] == row["ga"] for row in rows)
    losses = sum(row["gf"] < row["ga"] for row in rows)
    home_rows = [row for row in rows if row["is_home"]]
    away_rows = [row for row in rows if not row["is_home"]]

    def avg(items, key, default):
        return sum(row[key] for row in items) / len(items) if items else default

    return {
        "team": canonical,
        "matches": matches,
        "goals_for": gf_total,
        "goals_against": ga_total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "xg_for": None,
        "xg_against": None,
        "home_gf": round(avg(home_rows, "gf", gf_total / matches), 3),
        "home_ga": round(avg(home_rows, "ga", ga_total / matches), 3),
        "away_gf": round(avg(away_rows, "gf", gf_total / matches), 3),
        "away_ga": round(avg(away_rows, "ga", ga_total / matches), 3),
        "clean_sheets": round(sum(row["ga"] == 0 for row in rows) / matches, 3),
        "failed_to_score": round(sum(row["gf"] == 0 for row in rows) / matches, 3),
        "over_2_5_rate": round(sum((row["gf"] + row["ga"]) > 2.5 for row in rows) / matches, 3),
        "both_teams_score_rate": round(sum(row["gf"] > 0 and row["ga"] > 0 for row in rows) / matches, 3),
        "source": "API-Football",
        "last_updated": _now_iso(),
        "is_estimated": False,
        "confidence": "high" if matches >= 8 else "medium" if matches >= 5 else "low",
    }


def upsert_team_recent_form(row: dict):
    df = _read_csv(TEAM_FORM_PATH)
    if df.empty:
        df = pd.DataFrame()
    for col in [
        "team",
        "matches",
        "goals_for",
        "goals_against",
        "wins",
        "draws",
        "losses",
        "xg_for",
        "xg_against",
        "home_gf",
        "home_ga",
        "away_gf",
        "away_ga",
        *FORM_EXTRA_COLUMNS,
    ]:
        if col not in df.columns:
            df[col] = None
    backup_csv(TEAM_FORM_PATH)
    canonical = canonicalize_team_name(row["team"])
    df = df[df["team"].map(lambda value: _norm(canonicalize_team_name(value)) != _norm(canonical))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TEAM_FORM_PATH, index=False)


def update_team_forms_from_api_football(teams: list[str], last: int = 10, max_requests: int | None = None):
    max_requests = max_requests or settings.max_api_requests_per_run
    requests_used = 0
    updated = []
    errors = []

    for team in teams:
        if requests_used >= max_requests:
            break
        try:
            team_id, search_requests = _search_api_football_team_id_with_count(team)
            requests_used += search_requests
            if not team_id or requests_used >= max_requests:
                continue
            fixtures, fixture_requests = _fetch_recent_fixtures_for_team_with_count(team_id, last=last)
            requests_used += fixture_requests
            row = calculate_team_form_from_api_football(team, team_id, fixtures)
            if row.get("matches", 0) > 0:
                upsert_team_recent_form(row)
                updated.append(canonicalize_team_name(team))
        except Exception as exc:
            errors.append(f"{team}: {exc}")

    return {"updated": updated, "errors": errors, "requests_used": requests_used}


def fetch_h2h_api_football(home_team_id: int, away_team_id: int):
    endpoint = "/fixtures/headtohead"
    params = {"h2h": f"{home_team_id}-{away_team_id}"}
    cache_name = f"api_football_h2h_{home_team_id}_{away_team_id}"
    raw = _load_cached_api_payload(cache_name)
    requests_used = 0
    if not raw:
        raw = api_football_get(endpoint, params=params)
        _save_api_cache(cache_name, raw, endpoint, params)
        requests_used = 1
    rows = []
    for item in raw.get("response", []):
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        rows.append(
            {
                "date": fixture.get("date"),
                "home": teams.get("home", {}).get("name"),
                "away": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "competition": item.get("league", {}).get("name", "API-Football"),
            }
        )
    if rows:
        existing = _read_csv(H2H_PATH)
        backup_csv(H2H_PATH)
        df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates()
        df.to_csv(H2H_PATH, index=False)
    return {"rows": len(rows), "requests_used": requests_used}


def fetch_injuries_api_football(fixture_id: int):
    endpoint = "/injuries"
    params = {"fixture": fixture_id}
    cache_name = f"api_football_injuries_{fixture_id}"
    raw = _load_cached_api_payload(cache_name)
    requests_used = 0
    if not raw:
        raw = api_football_get(endpoint, params=params)
        _save_api_cache(cache_name, raw, endpoint, params)
        requests_used = 1
    rows = []
    for item in raw.get("response", []):
        player = item.get("player", {})
        team = item.get("team", {})
        rows.append(
            {
                "player": player.get("name"),
                "team": team.get("name"),
                "status": item.get("type") or item.get("reason"),
                "impact": "media",
                "source_url": "API-Football",
            }
        )
    if rows:
        existing = _read_csv(INJURIES_PATH)
        backup_csv(INJURIES_PATH)
        df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates()
        df.to_csv(INJURIES_PATH, index=False)
    return {"rows": len(rows), "requests_used": requests_used}


def fetch_lineups_api_football(fixture_id: int):
    endpoint = "/fixtures/lineups"
    params = {"fixture": fixture_id}
    cache_name = f"api_football_lineups_{fixture_id}"
    raw = _load_cached_api_payload(cache_name)
    requests_used = 0
    if not raw:
        raw = api_football_get(endpoint, params=params)
        _save_api_cache(cache_name, raw, endpoint, params)
        requests_used = 1
    return {"lineups": len(raw.get("response", [])), "requests_used": requests_used}
