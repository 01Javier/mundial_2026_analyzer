from pathlib import Path

import pandas as pd

from .team_registry import canonicalize_team_name

TEAM_STRENGTH_PATH = Path("data/team_strength.csv")


def _norm(value) -> str:
    return str(value or "").strip().lower()


def load_team_strength() -> pd.DataFrame:
    if not TEAM_STRENGTH_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(TEAM_STRENGTH_PATH)
    expected = [
        "team",
        "canonical_name",
        "elo",
        "fifa_rank",
        "confederation",
        "strength_score",
        "market_value_eur",
        "last_updated",
        "source",
        "confidence",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    if "canonical_name" in df.columns:
        df["canonical_name"] = df["canonical_name"].fillna(df["team"]).map(canonicalize_team_name)
    return df[expected]


def get_team_strength(team: str) -> dict | None:
    df = load_team_strength()
    if df.empty:
        return None

    canonical = canonicalize_team_name(team)
    mask = df["canonical_name"].map(lambda value: _norm(value) == _norm(canonical))
    row = df[mask]
    if row.empty:
        mask = df["team"].map(lambda value: _norm(canonicalize_team_name(value)) == _norm(canonical))
        row = df[mask]
    if row.empty:
        return None

    record = row.iloc[0].to_dict()
    for col in ["elo", "fifa_rank", "strength_score", "market_value_eur"]:
        try:
            record[col] = float(record[col])
        except Exception:
            record[col] = None
    return record


def strength_adjustment(team: str, opponent: str) -> float:
    """
    Devuelve ajuste entre -0.15 y +0.15.
    """
    team_strength = get_team_strength(team)
    opp_strength = get_team_strength(opponent)
    if not team_strength or not opp_strength:
        return 0.0

    team_score = team_strength.get("strength_score") or team_strength.get("elo")
    opp_score = opp_strength.get("strength_score") or opp_strength.get("elo")

    if team_score is None or opp_score is None:
        team_rank = team_strength.get("fifa_rank")
        opp_rank = opp_strength.get("fifa_rank")
        if team_rank is None or opp_rank is None:
            return 0.0
        # Lower FIFA rank is stronger. Convert rank gap to a conservative score gap.
        return max(-0.15, min(0.15, (opp_rank - team_rank) / 600))

    return max(-0.15, min(0.15, (team_score - opp_score) / 500))
