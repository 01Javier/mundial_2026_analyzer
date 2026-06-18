from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .config import settings

CALIBRATION_PATH = Path("data/model_calibration.json")
RESULTS_PATH = Path("data/model_results.csv")
MIN_CALIBRATION_MATCHES = 20
VALID_RESULT_SOURCES = {"api_real", "csv_real", "manual"}
DEFAULT_CALIBRATION = {
    "home_win_factor": 1.0,
    "draw_factor": 1.0,
    "away_win_factor": 1.0,
    "low_score_draw_factor": 1.0,
    "active": False,
    "reason": "Calibracion desactivada por muestra pequena",
    "matches_used": 0,
    "last_updated": None,
}


def load_calibration():
    if not CALIBRATION_PATH.exists():
        return DEFAULT_CALIBRATION.copy()

    try:
        with CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CALIBRATION.copy()

    return {**DEFAULT_CALIBRATION, **saved}


def save_calibration(params: dict):
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {**DEFAULT_CALIBRATION, **params}
    payload["last_updated"] = datetime.now(timezone.utc).isoformat()
    with CALIBRATION_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def _clip_factor(value: float) -> float:
    return max(0.80, min(1.20, float(value)))


def fit_simple_calibration(results_df):
    """
    Ajusta sesgo 1X2 acumulado. No calibra si hay menos de 20 partidos.
    """
    original_count = 0 if results_df is None else int(len(results_df))
    if settings.ignore_mock_results_for_calibration and results_df is not None and not results_df.empty:
        if "result_source" not in results_df.columns:
            return save_calibration(
                {
                    "active": False,
                    "reason": "Calibracion desactivada: resultados sin fuente valida para calibrar",
                    "matches_used": 0,
                }
            )
        valid_df = results_df[results_df["result_source"].isin(VALID_RESULT_SOURCES)].copy()
        mock_count = int((results_df["result_source"] == "mock").sum())
        if mock_count > len(valid_df):
            return save_calibration(
                {
                    "active": False,
                    "reason": "Calibracion desactivada: resultados mock no validos para calibrar",
                    "matches_used": int(len(valid_df)),
                }
            )
        results_df = valid_df

    if results_df is None or results_df.empty or len(results_df) < MIN_CALIBRATION_MATCHES:
        return save_calibration(
            {
                "active": False,
                "reason": "Calibracion desactivada por muestra pequena",
                "matches_used": 0 if results_df is None else int(len(results_df)),
                "total_results_seen": original_count,
            }
        )

    required = {"real_outcome", "predicted_home_win", "predicted_draw", "predicted_away_win"}
    if not required.issubset(results_df.columns):
        return save_calibration(
            {
                "active": False,
                "reason": "Faltan columnas historicas para calibrar",
                "matches_used": int(len(results_df)),
            }
        )

    df = results_df.dropna(subset=["real_outcome"]).copy()
    if len(df) < MIN_CALIBRATION_MATCHES:
        return save_calibration(
            {
                "active": False,
                "reason": "Calibracion desactivada por muestra pequena",
                "matches_used": int(len(df)),
            }
        )

    actual_home = (df["real_outcome"] == df["home"]).mean()
    actual_draw = (df["real_outcome"] == "Empate").mean()
    actual_away = (df["real_outcome"] == df["away"]).mean()

    pred_home = df["predicted_home_win"].astype(float).mean()
    pred_draw = df["predicted_draw"].astype(float).mean()
    pred_away = df["predicted_away_win"].astype(float).mean()

    params = {
        "home_win_factor": _clip_factor(actual_home / pred_home) if pred_home > 0 else 1.0,
        "draw_factor": _clip_factor(actual_draw / pred_draw) if pred_draw > 0 else 1.0,
        "away_win_factor": _clip_factor(actual_away / pred_away) if pred_away > 0 else 1.0,
        "low_score_draw_factor": 0.95 if pred_draw > actual_draw + 0.05 else 1.0,
        "active": True,
        "reason": "Calibracion activa con muestra suficiente",
        "matches_used": int(len(df)),
    }
    return save_calibration(params)


def apply_1x2_calibration(probs: dict, calibration: dict) -> dict:
    """
    Ajusta probabilidades 1X2 y normaliza para sumar 1.
    """
    if not probs:
        return probs

    keys = list(probs.keys())
    draw_key = "Empate"
    home_key = keys[0]
    away_key = keys[-1]

    if not calibration.get("active"):
        total = sum(float(v or 0) for v in probs.values())
        return {k: (float(v or 0) / total if total > 0 else 0) for k, v in probs.items()}

    adjusted = {
        home_key: float(probs.get(home_key, 0) or 0) * float(calibration.get("home_win_factor", 1.0)),
        draw_key: (
            float(probs.get(draw_key, 0) or 0)
            * float(calibration.get("draw_factor", 1.0))
            * float(calibration.get("low_score_draw_factor", 1.0))
        ),
        away_key: float(probs.get(away_key, 0) or 0) * float(calibration.get("away_win_factor", 1.0)),
    }
    total = sum(adjusted.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in adjusted.items()}


def apply_draw_bias_correction(probs: dict, bias_report: dict) -> dict:
    if not settings.use_draw_bias_correction or not probs:
        return probs
    if bias_report.get("sample_size", 0) < MIN_CALIBRATION_MATCHES:
        return probs
    draw_level = bias_report.get("draw_bias_level")
    if draw_level not in {"Medio", "Alto"}:
        return probs

    keys = list(probs.keys())
    home_key = keys[0]
    draw_key = "Empate"
    away_key = keys[-1]
    factor = 0.85 if draw_level == "Alto" else 0.95
    old_draw = float(probs.get(draw_key, 0) or 0)
    new_draw = old_draw * factor
    released = old_draw - new_draw
    home_prob = float(probs.get(home_key, 0) or 0)
    away_prob = float(probs.get(away_key, 0) or 0)
    side_total = home_prob + away_prob
    corrected = dict(probs)
    corrected[draw_key] = new_draw
    if side_total > 0:
        corrected[home_key] = home_prob + released * (home_prob / side_total)
        corrected[away_key] = away_prob + released * (away_prob / side_total)
    total = sum(float(value or 0) for value in corrected.values())
    return {key: float(value or 0) / total for key, value in corrected.items()} if total > 0 else probs


def detect_draw_bias(results_df):
    """
    Compara probabilidad predicha media de empate contra frecuencia real.
    """
    if results_df is not None and not results_df.empty and settings.ignore_mock_results_for_calibration:
        if "result_source" not in results_df.columns:
            results_df = pd.DataFrame()
        else:
            results_df = results_df[results_df["result_source"].isin(VALID_RESULT_SOURCES)].copy()

    if results_df is None or results_df.empty or len(results_df) < MIN_CALIBRATION_MATCHES:
        return {
            "status": "Sin muestra suficiente",
            "draw_bias": "Sin muestra suficiente",
            "draw_bias_level": "Sin muestra suficiente",
            "home_bias": "Sin muestra suficiente",
            "away_bias": "Sin muestra suficiente",
            "predicted_draw_rate": None,
            "actual_draw_rate": None,
            "sample_size": 0 if results_df is None else int(len(results_df)),
        }

    pred_draw = results_df["predicted_draw"].astype(float).mean()
    actual_draw = (results_df["real_outcome"] == "Empate").mean()
    pred_home = results_df["predicted_home_win"].astype(float).mean()
    actual_home = (results_df["real_outcome"] == results_df["home"]).mean()
    pred_away = results_df["predicted_away_win"].astype(float).mean()
    actual_away = (results_df["real_outcome"] == results_df["away"]).mean()

    draw_gap = pred_draw - actual_draw
    away_gap = actual_away - pred_away
    home_gap = pred_home - actual_home

    def level(gap):
        gap = abs(float(gap))
        if gap >= 0.12:
            return "Alto"
        if gap >= 0.06:
            return "Medio"
        return "Bajo"

    draw_level = level(draw_gap)

    return {
        "status": "Evaluado",
        "draw_bias": "Empate sobreestimado" if draw_gap > 0.06 else draw_level,
        "draw_bias_level": draw_level,
        "home_bias": "Local sobreestimado" if home_gap > 0.06 else level(home_gap),
        "away_bias": "Visitante subestimado" if away_gap > 0.06 else level(away_gap),
        "predicted_draw_rate": float(pred_draw),
        "actual_draw_rate": float(actual_draw),
        "sample_size": int(len(results_df)),
    }


def load_results_df() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH)
