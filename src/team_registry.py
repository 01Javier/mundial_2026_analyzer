from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
BACKUP_DIR = DATA_DIR / "backups"
TEAMS_PATH = DATA_DIR / "worldcup_teams.csv"
RECENT_FORM_PATH = DATA_DIR / "team_recent_form.csv"
TEAM_STRENGTH_PATH = DATA_DIR / "team_strength.csv"
LATEST_MATCHES_CACHE = CACHE_DIR / "worldcup_matches_latest.csv"

TEAM_COLUMNS = [
    "team_id",
    "team",
    "canonical_name",
    "group",
    "qualified_status",
    "source",
    "api_football_team_id",
    "football_data_team_id",
    "confederation",
    "fifa_rank",
    "elo",
    "strength_score",
    "is_host",
    "last_updated",
]

RECENT_FORM_COLUMNS = [
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
]

ALIASES = {
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina",
    "usa": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "congo dr": "Congo DR",
    "dr congo": "Congo DR",
}

HOSTS = {"United States", "Mexico", "Canada"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value) -> str:
    return str(value or "").strip().lower()


def canonicalize_team_name(team: str) -> str:
    if pd.isna(team):
        return ""
    text = str(team or "").strip()
    if not text or text.lower() == "nan":
        return ""
    return ALIASES.get(_norm(text), text)


def backup_csv(path: Path):
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


def _empty_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=TEAM_COLUMNS)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _strength_lookup() -> dict:
    df = _read_csv(TEAM_STRENGTH_PATH)
    if df.empty or "team" not in df.columns:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        canonical = canonicalize_team_name(row.get("team"))
        lookup[_norm(canonical)] = row.to_dict()
    return lookup


def build_worldcup_team_registry(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae todos los equipos unicos desde home/away/group/source.
    """
    if matches_df is None or matches_df.empty:
        return _empty_registry()

    strength = _strength_lookup()
    rows = {}
    for _, match in matches_df.iterrows():
        for side, id_col in [("home", "home_id"), ("away", "away_id")]:
            team = canonicalize_team_name(match.get(side))
            if not team:
                continue
            key = _norm(team)
            existing = rows.get(key, {})
            strength_row = strength.get(key, {})
            source = str(match.get("source") or "fixtures")
            provider_id = match.get(id_col)
            api_id = provider_id if "API-Football" in source else None
            football_data_id = provider_id if "football-data" in source else None
            rows[key] = {
                "team_id": existing.get("team_id") or team,
                "team": existing.get("team") or team,
                "canonical_name": team,
                "group": existing.get("group") or match.get("group") or "N/D",
                "qualified_status": existing.get("qualified_status") or "qualified",
                "source": existing.get("source") or source,
                "api_football_team_id": existing.get("api_football_team_id") or api_id,
                "football_data_team_id": existing.get("football_data_team_id") or football_data_id,
                "confederation": existing.get("confederation") or strength_row.get("confederation"),
                "fifa_rank": existing.get("fifa_rank") or strength_row.get("fifa_rank"),
                "elo": existing.get("elo") or strength_row.get("elo"),
                "strength_score": existing.get("strength_score") or strength_row.get("strength_score"),
                "is_host": team in HOSTS,
                "last_updated": _now_iso(),
            }

    return pd.DataFrame(rows.values(), columns=TEAM_COLUMNS).sort_values(["group", "team"])


def load_worldcup_teams() -> pd.DataFrame:
    if not TEAMS_PATH.exists():
        return _empty_registry()
    df = pd.read_csv(TEAMS_PATH)
    for col in TEAM_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[TEAM_COLUMNS]


def save_worldcup_teams(df: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TEAMS_PATH.exists():
        backup_csv(TEAMS_PATH)
    cleaned = df.copy()
    for col in TEAM_COLUMNS:
        if col not in cleaned.columns:
            cleaned[col] = None
    cleaned = cleaned[TEAM_COLUMNS].drop_duplicates(subset=["canonical_name"], keep="last")
    cleaned = cleaned[cleaned["canonical_name"].map(lambda value: bool(canonicalize_team_name(value)))]
    cleaned = cleaned.sort_values(["group", "team"], na_position="last")
    cleaned.to_csv(TEAMS_PATH, index=False)
    return cleaned


def update_worldcup_teams_from_matches(matches_df: pd.DataFrame):
    """
    Agrega equipos nuevos sin borrar existentes; completa grupo/source/last_updated si falta.
    """
    new_df = build_worldcup_team_registry(matches_df)
    current = load_worldcup_teams()
    if current.empty:
        return save_worldcup_teams(new_df)
    if new_df.empty:
        return current

    current_by_key = {_norm(row["canonical_name"]): row.to_dict() for _, row in current.iterrows()}
    changed = False
    for _, row in new_df.iterrows():
        key = _norm(row["canonical_name"])
        if key not in current_by_key:
            current_by_key[key] = row.to_dict()
            changed = True
            continue
        existing = current_by_key[key]
        row_changed = False
        for col in TEAM_COLUMNS:
            value = row.get(col)
            if (pd.isna(existing.get(col)) or existing.get(col) in ["", "N/D", None]) and not pd.isna(value):
                existing[col] = value
                row_changed = True
        if row_changed:
            existing["last_updated"] = _now_iso()
            changed = True

    merged = pd.DataFrame(current_by_key.values(), columns=TEAM_COLUMNS)
    if not changed:
        return current
    return save_worldcup_teams(merged)


def get_all_worldcup_teams() -> list[str]:
    df = load_worldcup_teams()
    if df.empty:
        return []
    return sorted(df["canonical_name"].dropna().astype(str).unique().tolist())


def get_missing_teams_in_recent_form() -> list[str]:
    teams = set(_norm(team) for team in get_all_worldcup_teams())
    form = _read_csv(RECENT_FORM_PATH)
    if form.empty or "team" not in form.columns:
        return sorted(get_all_worldcup_teams())
    existing = set(_norm(canonicalize_team_name(team)) for team in form["team"].dropna())
    registry = load_worldcup_teams()
    missing = registry[~registry["canonical_name"].map(lambda team: _norm(team) in existing)]
    return sorted(missing["canonical_name"].dropna().astype(str).tolist())


def save_latest_matches_cache(matches_df: pd.DataFrame):
    if matches_df is None or matches_df.empty:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(LATEST_MATCHES_CACHE, index=False)
    return LATEST_MATCHES_CACHE


def load_matches_for_registry() -> pd.DataFrame:
    frames = []
    for path in [LATEST_MATCHES_CACHE, DATA_DIR / "worldcup_matches.csv"]:
        df = _read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _recent_form_global_defaults(form_df: pd.DataFrame) -> dict:
    if form_df.empty or "matches" not in form_df.columns:
        return {"gf": 1.25, "ga": 1.25, "xgf": 1.25, "xga": 1.25}
    df = form_df.copy()
    df["matches"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)
    valid = df[df["matches"] > 0]
    if valid.empty:
        return {"gf": 1.25, "ga": 1.25, "xgf": 1.25, "xga": 1.25}
    gf = (pd.to_numeric(valid["goals_for"], errors="coerce") / valid["matches"]).mean()
    ga = (pd.to_numeric(valid["goals_against"], errors="coerce") / valid["matches"]).mean()
    xgf = pd.to_numeric(valid.get("xg_for"), errors="coerce").mean()
    xga = pd.to_numeric(valid.get("xg_against"), errors="coerce").mean()
    return {
        "gf": float(gf) if pd.notna(gf) else 1.25,
        "ga": float(ga) if pd.notna(ga) else 1.25,
        "xgf": float(xgf) if pd.notna(xgf) else 1.25,
        "xga": float(xga) if pd.notna(xga) else 1.25,
    }


def fallback_recent_form_row(team: str, registry_row: dict | None, defaults: dict) -> dict:
    strength_score = None
    if registry_row:
        try:
            strength_score = float(registry_row.get("strength_score"))
        except Exception:
            strength_score = None
    strength_delta = 0.0 if strength_score is None else max(-0.10, min(0.10, (strength_score - 1700) / 3000))
    gf = max(0.75, min(2.20, defaults["gf"] * (1 + strength_delta)))
    ga = max(0.75, min(2.20, defaults["ga"] * (1 - strength_delta)))
    matches = 3
    return {
        "team": team,
        "matches": matches,
        "goals_for": round(gf * matches, 2),
        "goals_against": round(ga * matches, 2),
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "xg_for": round(max(0.75, min(2.20, defaults["xgf"] * (1 + strength_delta))), 2),
        "xg_against": round(max(0.75, min(2.20, defaults["xga"] * (1 - strength_delta))), 2),
        "home_gf": round(gf, 2),
        "home_ga": round(ga, 2),
        "away_gf": round(gf * 0.95, 2),
        "away_ga": round(ga * 1.05, 2),
        "source": "registry_fallback",
        "data_quality": "low",
        "last_updated": _now_iso(),
    }


def get_team_fallback_recent_form(team: str) -> dict | None:
    registry = load_worldcup_teams()
    registry_row = None
    if not registry.empty:
        rows = registry[registry["canonical_name"].map(lambda value: _norm(value) == _norm(canonicalize_team_name(team)))]
        if not rows.empty:
            registry_row = rows.iloc[0].to_dict()
    if registry_row is None:
        return None
    form = _read_csv(RECENT_FORM_PATH)
    defaults = _recent_form_global_defaults(form)
    return fallback_recent_form_row(canonicalize_team_name(team), registry_row, defaults)


def ensure_recent_form_for_all_teams() -> list[str]:
    missing = get_missing_teams_in_recent_form()
    if not missing:
        return []

    form = _read_csv(RECENT_FORM_PATH)
    if form.empty:
        form = pd.DataFrame(columns=RECENT_FORM_COLUMNS)
    defaults = _recent_form_global_defaults(form)
    registry = load_worldcup_teams()
    registry_by_key = {_norm(row["canonical_name"]): row.to_dict() for _, row in registry.iterrows()}
    rows = [
        fallback_recent_form_row(team, registry_by_key.get(_norm(team)), defaults)
        for team in missing
    ]

    backup_csv(RECENT_FORM_PATH)
    updated = pd.concat([form, pd.DataFrame(rows)], ignore_index=True)
    updated = updated.drop_duplicates(subset=["team"], keep="last")
    updated.to_csv(RECENT_FORM_PATH, index=False)
    return missing


def refresh_local_team_catalog(matches_df: pd.DataFrame | None = None):
    if matches_df is None or matches_df.empty:
        matches_df = load_matches_for_registry()
    if matches_df is None or matches_df.empty:
        return {
            "teams_count": len(get_all_worldcup_teams()),
            "missing_recent_form": get_missing_teams_in_recent_form(),
            "fallback_added": [],
        }
    save_latest_matches_cache(matches_df)
    teams_df = update_worldcup_teams_from_matches(matches_df)
    fallback_added = ensure_recent_form_for_all_teams()
    return {
        "teams_count": int(len(teams_df)),
        "missing_recent_form": get_missing_teams_in_recent_form(),
        "fallback_added": fallback_added,
    }
