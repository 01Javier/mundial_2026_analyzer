from datetime import datetime, timezone
from pathlib import Path
import math

import pandas as pd

PREDICTIONS_PATH = Path("data/model_predictions.csv")
RESULTS_PATH = Path("data/model_results.csv")

PREDICTION_COLUMNS = [
    "prediction_id",
    "match_id",
    "date_utc",
    "home",
    "away",
    "group",
    "predicted_home_win",
    "predicted_draw",
    "predicted_away_win",
    "predicted_winner",
    "top1_score",
    "top1_score_prob",
    "top2_score",
    "top2_score_prob",
    "top3_score",
    "top3_score_prob",
    "top4_score",
    "top4_score_prob",
    "top5_score",
    "top5_score_prob",
    "lambda_home",
    "lambda_away",
    "confidence",
    "data_score",
    "source_mode",
    "created_at",
]

RESULT_COLUMNS = [
    "match_id",
    "home",
    "away",
    "real_home_goals",
    "real_away_goals",
    "real_score",
    "real_outcome",
    "predicted_winner",
    "winner_hit",
    "top1_score_hit",
    "top3_score_hit",
    "top5_score_hit",
    "brier_score_1x2",
    "log_loss_1x2",
    "exact_score_rank",
    "exact_score_probability",
    "evaluated_at",
    "predicted_home_win",
    "predicted_draw",
    "predicted_away_win",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def _write_upsert(path: Path, columns: list[str], row: dict, key: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _read_csv(path, columns)
    df = df[df[key].astype(str) != str(row[key])]
    df = pd.concat([df, pd.DataFrame([row], columns=columns)], ignore_index=True)
    df.to_csv(path, index=False)


def _outcome_keys(match: dict):
    return str(match.get("home")), "Empate", str(match.get("away"))


def calculate_brier_score_1x2(probs: dict, real_outcome: str) -> float:
    score = 0.0
    for label, prob in probs.items():
        observed = 1.0 if label == real_outcome else 0.0
        score += (float(prob or 0) - observed) ** 2
    return float(score)


def calculate_log_loss_1x2(probs: dict, real_outcome: str, eps: float = 1e-15) -> float:
    prob = float(probs.get(real_outcome, 0) or 0)
    prob = max(eps, min(1 - eps, prob))
    return float(-math.log(prob))


def save_prediction(match: dict, analysis: dict, source_mode: str):
    if not analysis.get("ok"):
        return None

    home, draw, away = _outcome_keys(match)
    winner_probs = analysis.get("winner_probs", {})
    top = analysis.get("top_results", [])
    match_id = str(match.get("match_id") or f"{home}-{away}-{match.get('date_utc', '')}")
    row = {
        "prediction_id": match_id,
        "match_id": match_id,
        "date_utc": match.get("date_utc"),
        "home": home,
        "away": away,
        "group": match.get("group"),
        "predicted_home_win": winner_probs.get(home),
        "predicted_draw": winner_probs.get(draw),
        "predicted_away_win": winner_probs.get(away),
        "predicted_winner": analysis.get("predicted_winner"),
        "lambda_home": analysis.get("lambda_home"),
        "lambda_away": analysis.get("lambda_away"),
        "confidence": analysis.get("confidence"),
        "data_score": analysis.get("data_score"),
        "source_mode": source_mode,
        "created_at": _utc_now(),
    }

    for idx in range(5):
        item = top[idx] if idx < len(top) else {}
        row[f"top{idx + 1}_score"] = item.get("Resultado")
        row[f"top{idx + 1}_score_prob"] = item.get("Probabilidad")

    _write_upsert(PREDICTIONS_PATH, PREDICTION_COLUMNS, row, "prediction_id")
    return row


def evaluate_prediction(match: dict, analysis: dict):
    if not analysis.get("ok") or str(match.get("status")) != "Jugado":
        return None

    try:
        real_h = int(float(match.get("home_goals")))
        real_a = int(float(match.get("away_goals")))
    except Exception:
        return None

    home, draw, away = _outcome_keys(match)
    real_score = f"{real_h} - {real_a}"
    if real_h > real_a:
        real_outcome = home
    elif real_a > real_h:
        real_outcome = away
    else:
        real_outcome = draw

    top = analysis.get("top_results", [])
    exact_rank = None
    exact_prob = None
    for item in top:
        if item.get("Resultado") == real_score:
            exact_rank = int(item.get("#"))
            exact_prob = float(item.get("Probabilidad"))
            break

    probs = analysis.get("winner_probs", {})
    match_id = str(match.get("match_id") or f"{home}-{away}-{match.get('date_utc', '')}")
    row = {
        "match_id": match_id,
        "home": home,
        "away": away,
        "real_home_goals": real_h,
        "real_away_goals": real_a,
        "real_score": real_score,
        "real_outcome": real_outcome,
        "predicted_winner": analysis.get("predicted_winner"),
        "winner_hit": analysis.get("predicted_winner") == real_outcome,
        "top1_score_hit": exact_rank == 1,
        "top3_score_hit": exact_rank is not None and exact_rank <= 3,
        "top5_score_hit": exact_rank is not None and exact_rank <= 5,
        "brier_score_1x2": calculate_brier_score_1x2(probs, real_outcome),
        "log_loss_1x2": calculate_log_loss_1x2(probs, real_outcome),
        "exact_score_rank": exact_rank,
        "exact_score_probability": exact_prob,
        "evaluated_at": _utc_now(),
        "predicted_home_win": probs.get(home),
        "predicted_draw": probs.get(draw),
        "predicted_away_win": probs.get(away),
    }
    _write_upsert(RESULTS_PATH, RESULT_COLUMNS, row, "match_id")
    return row


def load_backtest_summary() -> dict:
    df = _read_csv(RESULTS_PATH, RESULT_COLUMNS)
    if df.empty:
        return {
            "evaluated_matches": 0,
            "winner_accuracy": None,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "avg_brier": None,
            "avg_log_loss": None,
            "results_df": df,
        }

    for col in ["winner_hit", "top1_score_hit", "top3_score_hit", "top5_score_hit"]:
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes", "si", "sí"])

    return {
        "evaluated_matches": int(len(df)),
        "winner_accuracy": float(df["winner_hit"].mean()),
        "top1_accuracy": float(df["top1_score_hit"].mean()),
        "top3_accuracy": float(df["top3_score_hit"].mean()),
        "top5_accuracy": float(df["top5_score_hit"].mean()),
        "avg_brier": float(pd.to_numeric(df["brier_score_1x2"], errors="coerce").mean()),
        "avg_log_loss": float(pd.to_numeric(df["log_loss_1x2"], errors="coerce").mean()),
        "results_df": df,
    }
