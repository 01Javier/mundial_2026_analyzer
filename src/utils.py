from datetime import datetime
from zoneinfo import ZoneInfo
import math
import pandas as pd

def safe_float(value, default=None):
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default=0):
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default

def pct(x):
    if x is None:
        return "N/D"
    try:
        if pd.isna(x):
            return "N/D"
    except Exception:
        pass
    return f"{x * 100:.1f}%"

def gt_time_from_utc(dt_text: str, tz: ZoneInfo):
    if not dt_text:
        return None
    try:
        dt = datetime.fromisoformat(str(dt_text).replace("Z", "+00:00"))
        return dt.astimezone(tz)
    except Exception:
        return None

def probability_bar(prob, width=18):
    if prob is None:
        return ""
    filled = int(round(prob * width))
    return "█" * filled + "░" * max(0, width - filled)

def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "ALTA"
    if score >= 0.55:
        return "MEDIA"
    return "BAJA"

def clamp(value, low, high):
    return max(low, min(high, value))

def normalize_team_name(name: str) -> str:
    return str(name or "").strip().lower()

def poisson_first_goal_probs(lambda_a, lambda_b):
    total_rate = lambda_a + lambda_b
    if total_rate <= 0:
        return None, None, None
    p_no_goal = math.exp(-total_rate)
    p_any_goal = 1 - p_no_goal
    return (
        p_any_goal * (lambda_a / total_rate),
        p_any_goal * (lambda_b / total_rate),
        p_no_goal,
    )
