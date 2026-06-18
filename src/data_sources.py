from pathlib import Path
import pandas as pd

from .clients import api_football_get, football_data_get, open_meteo_forecast
from .cache_store import (
    increment_requests_saved,
    load_json_cache,
    metadata_for_cache,
    save_json_cache,
    update_cache_metadata,
)
from .config import settings
from .mock_data import mock_worldcup_matches
from .team_registry import get_team_fallback_recent_form
from .utils import normalize_team_name, gt_time_from_utc

DATA_DIR = Path("data")
API_CACHE_TTL_MINUTES = 12 * 60
API_FOOTBALL_FIXTURES_CACHE = f"api_football_fixtures_{settings.api_football_season}"
FOOTBALL_DATA_MATCHES_CACHE = f"football_data_matches_{settings.football_data_season}"

def load_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def normalize_api_football_fixtures(raw_json: dict) -> pd.DataFrame:
    rows = []

    for item in raw_json.get("response", []):
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})

        status_short = fixture.get("status", {}).get("short")
        status_long = fixture.get("status", {}).get("long")

        if status_short in ["FT", "AET", "PEN"]:
            estado = "Jugado"
        elif status_short in ["NS", "TBD"]:
            estado = "Próximo"
        elif status_short in ["1H", "2H", "HT", "ET", "P", "BT"]:
            estado = "En curso"
        else:
            estado = status_long or "Desconocido"

        round_text = str(league.get("round", "") or "")
        group = (
            round_text
            .replace("Group Stage - ", "")
            .replace("Group ", "")
            .replace("First Stage - ", "")
            .strip()
        )

        rows.append({
            "match_id": str(fixture.get("id")),
            "home": teams.get("home", {}).get("name"),
            "away": teams.get("away", {}).get("name"),
            "home_id": teams.get("home", {}).get("id"),
            "away_id": teams.get("away", {}).get("id"),
            "group": group or "N/D",
            "date_utc": fixture.get("date"),
            "venue": fixture.get("venue", {}).get("name") or "N/D",
            "city": fixture.get("venue", {}).get("city") or "N/D",
            "status": estado,
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "source": "API-Football",
        })

    return pd.DataFrame(rows)

def normalize_football_data_matches(raw_json: dict) -> pd.DataFrame:
    rows = []

    for item in raw_json.get("matches", []):
        score = item.get("score", {})
        full_time = score.get("fullTime", {}) or {}

        status = item.get("status", "")
        if status == "FINISHED":
            estado = "Jugado"
        elif status in ["SCHEDULED", "TIMED"]:
            estado = "Próximo"
        elif status in ["IN_PLAY", "PAUSED"]:
            estado = "En curso"
        else:
            estado = status or "Desconocido"

        group = item.get("group") or item.get("stage") or "N/D"
        group = str(group).replace("GROUP_", "").replace("Group ", "")

        rows.append({
            "match_id": str(item.get("id")),
            "home": item.get("homeTeam", {}).get("name"),
            "away": item.get("awayTeam", {}).get("name"),
            "home_id": item.get("homeTeam", {}).get("id"),
            "away_id": item.get("awayTeam", {}).get("id"),
            "group": group,
            "date_utc": item.get("utcDate"),
            "venue": item.get("venue") or "N/D",
            "city": "N/D",
            "status": estado,
            "home_goals": full_time.get("home"),
            "away_goals": full_time.get("away"),
            "source": "football-data.org",
        })

    return pd.DataFrame(rows)


def _with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    view = df.copy()
    if not view.empty:
        view["source"] = source
    return view


def _load_cached_fixtures() -> tuple[pd.DataFrame, list[str]] | None:
    raw = load_json_cache(API_FOOTBALL_FIXTURES_CACHE)
    if raw:
        df = normalize_api_football_fixtures(raw)
        if not df.empty:
            increment_requests_saved(API_FOOTBALL_FIXTURES_CACHE)
            return _with_source(df, "Cache local: API-Football"), [
                f"Cache local: data/cache/{API_FOOTBALL_FIXTURES_CACHE}.json"
            ]

    raw = load_json_cache(FOOTBALL_DATA_MATCHES_CACHE)
    if raw:
        df = normalize_football_data_matches(raw)
        if not df.empty:
            increment_requests_saved(FOOTBALL_DATA_MATCHES_CACHE)
            return _with_source(df, "Cache local: football-data.org"), [
                f"Cache local: data/cache/{FOOTBALL_DATA_MATCHES_CACHE}.json"
            ]

    return None


def fetch_sports_data(dataset: str = "fixtures", refresh: bool = False) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Función central estilo fetch_sports_data.
    Devuelve: dataframe, fuentes_usadas, errores.
    """
    if dataset != "fixtures":
        raise ValueError("Por ahora solo se soporta dataset='fixtures'")

    errors = []

    if not refresh:
        cached = _load_cached_fixtures()
        if cached is not None:
            return cached[0], cached[1], errors

        local = load_csv_if_exists(DATA_DIR / "worldcup_matches.csv")
        if not local.empty:
            return _with_source(local, "CSV local: data/worldcup_matches.csv"), [
                "CSV local: data/worldcup_matches.csv"
            ], errors

        if settings.use_mock_data:
            return _with_source(mock_worldcup_matches(), "Mock data"), ["Mock data"], errors

        return pd.DataFrame(), [], errors

    try:
        if settings.api_football_key:
            endpoint = "/fixtures"
            params = {
                "league": settings.api_football_world_cup_league_id,
                "season": settings.api_football_season,
                "timezone": "America/Guatemala",
            }
            raw = api_football_get(endpoint, params=params)
            cache_file = save_json_cache(API_FOOTBALL_FIXTURES_CACHE, raw)
            update_cache_metadata(
                API_FOOTBALL_FIXTURES_CACHE,
                metadata_for_cache(
                    provider="API-Football",
                    endpoint=endpoint,
                    params=params,
                    source_file=cache_file,
                    ttl_minutes=API_CACHE_TTL_MINUTES,
                ),
            )
            df = normalize_api_football_fixtures(raw)
            if not df.empty:
                return _with_source(df, "API en vivo: API-Football"), ["API en vivo: API-Football"], errors
    except Exception as exc:
        errors.append(f"API-Football: {exc}")
        update_cache_metadata(
            API_FOOTBALL_FIXTURES_CACHE,
            {
                "provider": "API-Football",
                "endpoint": "/fixtures",
                "params": {
                    "league": settings.api_football_world_cup_league_id,
                    "season": settings.api_football_season,
                    "timezone": "America/Guatemala",
                },
                "cached_at": None,
                "expires_at": None,
                "source_file": f"data/cache/{API_FOOTBALL_FIXTURES_CACHE}.json",
                "requests_saved": 0,
                "last_error": str(exc),
            },
        )

    try:
        if settings.football_data_token:
            endpoint = f"/competitions/{settings.football_data_competition}/matches"
            params = {"season": settings.football_data_season}
            raw = football_data_get(endpoint, params=params)
            cache_file = save_json_cache(FOOTBALL_DATA_MATCHES_CACHE, raw)
            update_cache_metadata(
                FOOTBALL_DATA_MATCHES_CACHE,
                metadata_for_cache(
                    provider="football-data.org",
                    endpoint=endpoint,
                    params=params,
                    source_file=cache_file,
                    ttl_minutes=API_CACHE_TTL_MINUTES,
                ),
            )
            df = normalize_football_data_matches(raw)
            if not df.empty:
                return _with_source(df, "API en vivo: football-data.org"), ["API en vivo: football-data.org"], errors
    except Exception as exc:
        errors.append(f"football-data.org: {exc}")
        update_cache_metadata(
            FOOTBALL_DATA_MATCHES_CACHE,
            {
                "provider": "football-data.org",
                "endpoint": f"/competitions/{settings.football_data_competition}/matches",
                "params": {"season": settings.football_data_season},
                "cached_at": None,
                "expires_at": None,
                "source_file": f"data/cache/{FOOTBALL_DATA_MATCHES_CACHE}.json",
                "requests_saved": 0,
                "last_error": str(exc),
            },
        )

    cached = _load_cached_fixtures()
    if cached is not None:
        return cached[0], cached[1], errors

    local = load_csv_if_exists(DATA_DIR / "worldcup_matches.csv")
    if not local.empty:
        return _with_source(local, "CSV local: data/worldcup_matches.csv"), ["CSV local: data/worldcup_matches.csv"], errors

    if settings.use_mock_data:
        return _with_source(mock_worldcup_matches(), "Mock data"), ["Mock data"], errors

    return pd.DataFrame(), [], errors

def get_team_recent_form(team: str) -> dict | None:
    weighted = calculate_weighted_recent_form(team)
    if weighted is not None:
        return weighted

    df = load_csv_if_exists(DATA_DIR / "team_recent_form.csv")
    if df.empty:
        fallback = get_team_fallback_recent_form(team)
        return _recent_form_record(team, fallback) if fallback else None

    row = df[df["team"].map(normalize_team_name) == normalize_team_name(team)]
    if row.empty:
        fallback = get_team_fallback_recent_form(team)
        return _recent_form_record(team, fallback) if fallback else None

    r = row.iloc[0].to_dict()
    return _recent_form_record(team, r)


def _recent_form_record(team: str, r: dict) -> dict | None:
    matches = float(r.get("matches", 0) or 0)
    if matches <= 0:
        return None

    return {
        "team": team,
        "matches": matches,
        "gf_per_match": float(r.get("goals_for", 0) or 0) / matches,
        "ga_per_match": float(r.get("goals_against", 0) or 0) / matches,
        "wins": int(float(r.get("wins", 0) or 0)),
        "draws": int(float(r.get("draws", 0) or 0)),
        "losses": int(float(r.get("losses", 0) or 0)),
        "xg_for": None if pd.isna(r.get("xg_for", None)) else float(r.get("xg_for")),
        "xg_against": None if pd.isna(r.get("xg_against", None)) else float(r.get("xg_against")),
        "home_gf": None if pd.isna(r.get("home_gf", None)) else float(r.get("home_gf")),
        "home_ga": None if pd.isna(r.get("home_ga", None)) else float(r.get("home_ga")),
        "away_gf": None if pd.isna(r.get("away_gf", None)) else float(r.get("away_gf")),
        "away_ga": None if pd.isna(r.get("away_ga", None)) else float(r.get("away_ga")),
        "source": r.get("source", "team_recent_form.csv"),
        "data_quality": r.get("data_quality", "normal"),
    }


def calculate_weighted_recent_form(team, half_life_days=180):
    """
    Usa data/team_match_history.csv con decaimiento exponencial si existe.
    """
    df = load_csv_if_exists(DATA_DIR / "team_match_history.csv")
    if df.empty:
        return None

    required = {"date", "team", "goals_for", "goals_against"}
    if not required.issubset(df.columns):
        return None

    team_df = df[df["team"].map(normalize_team_name) == normalize_team_name(team)].copy()
    if team_df.empty:
        return None

    team_df["date"] = pd.to_datetime(team_df["date"], errors="coerce", utc=True)
    team_df = team_df.dropna(subset=["date"])
    if team_df.empty:
        return None

    now = pd.Timestamp.utcnow()
    team_df["days_ago"] = (now - team_df["date"]).dt.total_seconds() / 86400
    team_df["weight"] = 0.5 ** (team_df["days_ago"].clip(lower=0) / half_life_days)
    total_weight = team_df["weight"].sum()
    if total_weight <= 0:
        return None

    def weighted_avg(col):
        if col not in team_df.columns:
            return None
        values = pd.to_numeric(team_df[col], errors="coerce")
        mask = values.notna()
        if not mask.any():
            return None
        weights = team_df.loc[mask, "weight"]
        return float((values[mask] * weights).sum() / weights.sum())

    goals_for = pd.to_numeric(team_df["goals_for"], errors="coerce")
    goals_against = pd.to_numeric(team_df["goals_against"], errors="coerce")

    return {
        "team": team,
        "matches": float(len(team_df)),
        "gf_per_match": weighted_avg("goals_for"),
        "ga_per_match": weighted_avg("goals_against"),
        "wins": int((goals_for > goals_against).sum()),
        "draws": int((goals_for == goals_against).sum()),
        "losses": int((goals_for < goals_against).sum()),
        "xg_for": weighted_avg("xg_for"),
        "xg_against": weighted_avg("xg_against"),
        "home_gf": None,
        "home_ga": None,
        "away_gf": None,
        "away_ga": None,
        "form_source": "team_match_history.csv",
    }


def get_team_strength(team):
    df = load_csv_if_exists(DATA_DIR / "team_strength.csv")
    if df.empty or "team" not in df.columns or "strength_score" not in df.columns:
        return None

    row = df[df["team"].map(normalize_team_name) == normalize_team_name(team)]
    if row.empty:
        return None

    record = row.iloc[0].to_dict()
    try:
        record["strength_score"] = float(record.get("strength_score"))
    except Exception:
        return None
    return record

def get_h2h(home: str, away: str, limit: int = 5) -> pd.DataFrame:
    df = load_csv_if_exists(DATA_DIR / "h2h.csv")
    if df.empty:
        return pd.DataFrame()

    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)

    mask = (
        ((df["home"].map(normalize_team_name) == home_n) & (df["away"].map(normalize_team_name) == away_n))
        |
        ((df["home"].map(normalize_team_name) == away_n) & (df["away"].map(normalize_team_name) == home_n))
    )

    h2h = df[mask].copy()

    if "date" in h2h.columns:
        h2h["date"] = pd.to_datetime(h2h["date"], errors="coerce")
        h2h = h2h.sort_values("date", ascending=False)

    return h2h.head(limit)

def get_player_form(home: str, away: str) -> pd.DataFrame:
    df = load_csv_if_exists(DATA_DIR / "player_form.csv")
    if df.empty:
        return pd.DataFrame()

    teams = {normalize_team_name(home), normalize_team_name(away)}
    return df[df["team"].map(normalize_team_name).isin(teams)].copy()

def get_injuries(home: str, away: str) -> pd.DataFrame:
    df = load_csv_if_exists(DATA_DIR / "injuries.csv")
    if df.empty:
        return pd.DataFrame()

    teams = {normalize_team_name(home), normalize_team_name(away)}
    return df[df["team"].map(normalize_team_name).isin(teams)].copy()

def get_group_table(group: str) -> pd.DataFrame:
    df = load_csv_if_exists(DATA_DIR / "group_tables.csv")
    if df.empty:
        return pd.DataFrame()

    return df[df["group"].astype(str).str.upper() == str(group).upper()].copy()

def get_stadium_info(venue: str) -> dict | None:
    df = load_csv_if_exists(DATA_DIR / "stadiums.csv")
    if df.empty:
        return None

    row = df[df["venue"].map(normalize_team_name) == normalize_team_name(venue)]
    if row.empty:
        return None

    return row.iloc[0].to_dict()

def get_weather_for_match(match: dict) -> dict | None:
    stadium = get_stadium_info(match.get("venue", ""))
    if not stadium:
        return None

    try:
        lat = float(stadium["lat"])
        lon = float(stadium["lon"])
    except Exception:
        return None

    dt_gt = gt_time_from_utc(match.get("date_utc", ""), settings.gt_tz)
    date_iso = dt_gt.date().isoformat() if dt_gt else None

    try:
        weather = open_meteo_forecast(lat, lon, date_iso=date_iso)
        return {
            "stadium": stadium,
            "forecast": weather,
        }
    except Exception:
        return {
            "stadium": stadium,
            "forecast": None,
        }
