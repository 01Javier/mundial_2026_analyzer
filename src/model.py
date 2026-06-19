import math
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .backtesting import calculate_brier_score_1x2, calculate_log_loss_1x2, load_backtest_summary
from .calibration import apply_1x2_calibration, apply_draw_bias_correction, detect_draw_bias, load_calibration
from .config import settings
from .dixon_coles import poisson_score_matrix_dc
from .data_sources import (
    get_team_recent_form,
    get_h2h,
    get_player_form,
    get_injuries,
    get_group_table,
    get_stadium_info,
)
from .team_strength import get_team_strength, strength_adjustment as team_strength_adjustment
from .utils import clamp, confidence_label, poisson_first_goal_probs, safe_int
from .web_enrichment import summarize_web_enrichment

HOST_TEAMS = {
    "united states", "usa", "estados unidos",
    "mexico", "méxico",
    "canada", "canadá",
}

def estimate_team_lambda(team: str, opponent: str, is_home_side: bool = False) -> tuple[float | None, list[str], dict]:
    """
    Estima goles esperados para un equipo.
    Usa xG/xGA cuando existe. Si no existe, usa goles reales.
    """
    missing = []
    info = {}

    team_form = get_team_recent_form(team)
    opp_form = get_team_recent_form(opponent)

    if team_form is None:
        missing.append(f"forma reciente de {team}")
    if opp_form is None:
        missing.append(f"forma reciente de {opponent}")

    if team_form is None or opp_form is None:
        return None, missing, info

    team_xg = team_form.get("xg_for")
    opp_xga = opp_form.get("xg_against")

    has_xg = team_xg is not None and opp_xga is not None

    # Capa 1: xG si existe.
    if has_xg:
        expected = (
            0.50 * team_xg +
            0.30 * opp_xga +
            0.20 * team_form["gf_per_match"]
        )
        info["lambda_method"] = "xG/xGA + goles recientes"
    else:
        expected = (
            0.55 * team_form["gf_per_match"] +
            0.45 * opp_form["ga_per_match"]
        )
        missing.append("xG/xGA")
        info["lambda_method"] = "goles recientes sin xG"

    # Capa 2: rendimiento local/visitante si el CSV lo trae.
    if is_home_side and team_form.get("home_gf") is not None and opp_form.get("away_ga") is not None:
        local_component = 0.50 * team_form["home_gf"] + 0.50 * opp_form["away_ga"]
        expected = 0.85 * expected + 0.15 * local_component
        info["home_away_adjustment"] = True
    elif (not is_home_side) and team_form.get("away_gf") is not None and opp_form.get("home_ga") is not None:
        away_component = 0.50 * team_form["away_gf"] + 0.50 * opp_form["home_ga"]
        expected = 0.85 * expected + 0.15 * away_component
        info["home_away_adjustment"] = True
    else:
        info["home_away_adjustment"] = False

    expected = clamp(expected, 0.15, 4.50)

    info["team_form"] = team_form
    info["opp_form"] = opp_form
    info["has_xg"] = has_xg

    return expected, missing, info


def sample_weight_for_matches(matches: float) -> float:
    if matches >= 10:
        return 0.85
    if matches >= 5:
        return 0.65
    return 0.45


def global_average_goals() -> float:
    teams = []
    try:
        df = pd.read_csv("data/team_recent_form.csv")
        if "matches" not in df.columns:
            return 1.35
        for _, row in df.iterrows():
            matches = float(row.get("matches", 0) or 0)
            if matches > 0:
                teams.append(float(row.get("goals_for", 0) or 0) / matches)
    except Exception:
        return 1.35

    return float(np.mean(teams)) if teams else 1.35


def strength_adjustment(team: str, opponent: str) -> tuple[float, str | None]:
    adjustment = team_strength_adjustment(team, opponent)
    if adjustment == 0:
        return 1.0, None
    return 1 + adjustment, f"Fuerza relativa {team}: ajuste {adjustment:+.1%} vs {opponent}."


def build_fallback_form(team: str, opponent: str, reason: str):
    avg = global_average_goals()
    adjustment = team_strength_adjustment(team, opponent)
    if adjustment == 0:
        team_hash = sum((idx + 1) * ord(char) for idx, char in enumerate(team.lower()))
        opp_hash = sum((idx + 1) * ord(char) for idx, char in enumerate(opponent.lower()))
        if team_hash != opp_hash:
            adjustment = clamp((team_hash - opp_hash) / max(team_hash + opp_hash, 1), -0.03, 0.03)
    gf = clamp(avg * (1 + adjustment), 0.80, 2.10)
    ga = clamp(avg * (1 - adjustment), 0.80, 2.10)
    strength = get_team_strength(team)
    return {
        "team": team,
        "matches": 3,
        "gf_per_match": gf,
        "ga_per_match": ga,
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "xg_for": None,
        "xg_against": None,
        "home_gf": gf,
        "home_ga": ga,
        "away_gf": gf * 0.95,
        "away_ga": ga * 1.05,
        "clean_sheets": None,
        "failed_to_score": None,
        "over_2_5_rate": None,
        "both_teams_score_rate": None,
        "source": "model_fallback",
        "data_quality": "low",
        "confidence": "low",
        "is_estimated": True,
        "reason": reason,
        "strength_available": strength is not None,
    }


def estimate_team_lambda_advanced(team, opponent, match, is_home_side, web_summary: dict | None = None):
    """
    Estima lambda con xG, goles, shrinkage, fuerza rival, sede y robustez ante muestras pequenas.
    """
    missing = []
    factors = []
    info = {}

    team_form = get_team_recent_form(team)
    opp_form = get_team_recent_form(opponent)

    if team_form is None:
        missing.append(f"forma reciente de {team}")
        team_form = build_fallback_form(team, opponent, "sin forma reciente API/CSV")
    if opp_form is None:
        missing.append(f"forma reciente de {opponent}")
        opp_form = build_fallback_form(opponent, team, "sin forma reciente API/CSV")

    if team_form.get("data_quality") == "low":
        missing.append(f"forma estimada de {team}")
        factors.append(f"{team}: forma reciente fallback de baja confianza.")
    if opp_form.get("data_quality") == "low":
        missing.append(f"forma estimada de {opponent}")
        factors.append(f"{opponent}: forma rival fallback de baja confianza.")

    team_gf = float(team_form.get("gf_per_match") or 0)
    opp_ga = float(opp_form.get("ga_per_match") or 0)
    web_summary = web_summary or {}
    web_xg = web_summary.get("xg_home") if is_home_side else web_summary.get("xg_away")

    team_xg = team_form.get("xg_for")
    opp_xga = opp_form.get("xg_against")
    has_xg = team_xg is not None and opp_xga is not None
    has_web_xg = team_xg is None and web_xg is not None

    if has_xg:
        base_lambda = (
            0.35 * float(team_xg) +
            0.25 * float(opp_xga) +
            0.20 * team_gf +
            0.20 * opp_ga
        )
        method = "xG/xGA + goles + shrinkage"
    elif has_web_xg:
        web_source_count = int(web_summary.get("source_count", 0) or 0)
        opp_xga_component = float(opp_xga) if opp_xga is not None else global_average_goals()
        if web_source_count >= settings.web_confidence_min_sources:
            base_lambda = (
                0.35 * float(web_xg) +
                0.25 * opp_xga_component +
                0.20 * team_gf +
                0.20 * opp_ga
            )
            method = "xG web + goles + shrinkage"
        else:
            base_lambda = (
                0.20 * float(web_xg) +
                0.35 * team_gf +
                0.30 * opp_ga +
                0.15 * global_average_goals()
            )
            method = "xG web fuente unica + goles + shrinkage"
        factors.append(f"{team}: xG externo web usado con fuente trazable.")
    else:
        over_values = [
            value
            for value in [team_form.get("over_2_5_rate"), opp_form.get("over_2_5_rate")]
            if value is not None
        ]
        btts_values = [
            value
            for value in [team_form.get("both_teams_score_rate"), opp_form.get("both_teams_score_rate")]
            if value is not None
        ]
        over_component = float(np.mean(over_values)) if over_values else 0.45
        btts_component = float(np.mean(btts_values)) if btts_values else 0.45
        base_lambda = (
            0.35 * team_gf +
            0.30 * opp_ga +
            0.15 * global_average_goals() +
            0.10 * (0.8 + over_component) +
            0.10 * (0.8 + btts_component)
        )
        method = "goles recientes + shrinkage"
        missing.append("xG/xGA")

    matches = float(team_form.get("matches", 0) or 0)
    sample_weight = sample_weight_for_matches(matches)
    avg_goals = global_average_goals()
    expected = sample_weight * base_lambda + (1 - sample_weight) * avg_goals
    factors.append(f"Shrinkage {team}: peso muestra {sample_weight:.0%}, promedio global {avg_goals:.2f}.")

    strength_factor, strength_note = strength_adjustment(team, opponent)
    expected *= strength_factor
    if strength_note:
        factors.append(strength_note)
    else:
        missing.append("fuerza relativa")

    if is_home_side and team_form.get("home_gf") is not None and opp_form.get("away_ga") is not None:
        local_component = 0.50 * float(team_form["home_gf"]) + 0.50 * float(opp_form["away_ga"])
        expected = 0.90 * expected + 0.10 * local_component
        factors.append(f"Rendimiento local/visitante aplicado para {team}.")
    elif (not is_home_side) and team_form.get("away_gf") is not None and opp_form.get("home_ga") is not None:
        away_component = 0.50 * float(team_form["away_gf"]) + 0.50 * float(opp_form["home_ga"])
        expected = 0.90 * expected + 0.10 * away_component
        factors.append(f"Rendimiento visitante/local aplicado para {team}.")

    venue = str(match.get("venue", ""))
    stadium = get_stadium_info(venue)
    if stadium and stadium.get("altitude_m") is not None:
        try:
            altitude = float(stadium.get("altitude_m"))
            if altitude >= 1200:
                expected *= 0.98
                factors.append(f"Sede en altitud: {altitude:.0f} msnm, ajuste conservador.")
        except Exception:
            pass

    opp_clean_sheets = opp_form.get("clean_sheets")
    team_failed_to_score = team_form.get("failed_to_score")
    if opp_clean_sheets is not None and opp_clean_sheets > 0.45:
        expected *= 0.92
        factors.append(f"{opponent}: alta tasa de porterias a cero, reduce lambda de {team}.")
    if team_failed_to_score is not None and team_failed_to_score > 0.35:
        expected *= 0.90
        factors.append(f"{team}: alta tasa sin anotar, reduce lambda propia.")
    team_over = team_form.get("over_2_5_rate")
    opp_over = opp_form.get("over_2_5_rate")
    if team_over is not None and opp_over is not None and team_over > 0.55 and opp_over > 0.55:
        expected *= 1.06
        factors.append("Ambos equipos muestran tendencia over 2.5; sube total de goles.")
    team_btts = team_form.get("both_teams_score_rate")
    opp_btts = opp_form.get("both_teams_score_rate")
    if team_btts is not None and opp_btts is not None and team_btts > 0.60 and opp_btts > 0.60:
        factors.append("BTTS alto: se informa como correlacion, sin forzar 1-1.")

    if team.lower() in HOST_TEAMS:
        expected *= 1.04
        factors.append(f"{team} recibe ajuste ligero de anfitrion/localia.")

    info.update(
        {
            "lambda_method": method,
            "team_form": team_form,
            "opp_form": opp_form,
            "has_xg": has_xg,
            "has_web_xg": has_web_xg,
            "xg_source": "web" if has_web_xg else "csv" if has_xg else None,
            "sample_weight": sample_weight,
            "base_lambda": base_lambda,
            "global_avg_goals": avg_goals,
            "advanced_factors": factors,
            "is_fallback": bool(team_form.get("is_estimated") or team_form.get("data_quality") == "low"),
            "form_source": team_form.get("source", "N/D"),
            "form_confidence": team_form.get("confidence", "medium"),
            "strength_available": get_team_strength(team) is not None,
        }
    )
    return clamp(expected, 0.15, 4.50), missing, info

def h2h_adjustment(lambda_home, lambda_away, home, away):
    """
    Ajuste pequeño basado en últimos 5 H2H.
    Evita que el H2H domine el modelo.
    """
    h2h = get_h2h(home, away)
    if h2h.empty:
        return lambda_home, lambda_away, "No hay H2H cargado."

    home_points = 0
    away_points = 0
    count = 0

    for _, r in h2h.iterrows():
        hg = safe_int(r.get("home_goals"))
        ag = safe_int(r.get("away_goals"))
        h = str(r.get("home"))
        a = str(r.get("away"))

        if hg == ag:
            home_points += 1
            away_points += 1
        else:
            winner = h if hg > ag else a
            if winner.lower() == home.lower():
                home_points += 3
            elif winner.lower() == away.lower():
                away_points += 3
        count += 1

    if count == 0:
        return lambda_home, lambda_away, "H2H sin marcadores válidos."

    diff = home_points - away_points
    # Máximo ajuste aproximado ±6%.
    factor = clamp(diff / (count * 3), -1, 1) * 0.06

    lambda_home *= (1 + factor)
    lambda_away *= (1 - factor)

    return (
        clamp(lambda_home, 0.15, 4.50),
        clamp(lambda_away, 0.15, 4.50),
        f"H2H últimos {count}: ajuste {factor:+.2%} hacia {home if factor > 0 else away if factor < 0 else 'neutral'}."
    )

def apply_context_adjustments(lambda_home, lambda_away, match, injuries):
    factors = []

    home = match["home"]
    away = match["away"]

    if injuries is not None and not injuries.empty:
        for _, row in injuries.iterrows():
            team = str(row.get("team"))
            impact = str(row.get("impact", "")).lower()
            player = str(row.get("player", "Jugador"))

            if impact == "alta":
                penalty = 0.08
            elif impact == "media":
                penalty = 0.04
            else:
                penalty = 0.02

            if team.lower() == home.lower():
                lambda_home *= (1 - penalty)
                factors.append(f"Ausencia/duda en {home}: {player} ({impact}).")
            elif team.lower() == away.lower():
                lambda_away *= (1 - penalty)
                factors.append(f"Ausencia/duda en {away}: {player} ({impact}).")

    stadium = get_stadium_info(match.get("venue", ""))
    if stadium:
        altitude = stadium.get("altitude_m")
        if altitude is not None:
            try:
                altitude = float(altitude)
                if altitude >= 1200:
                    factors.append(f"Sede con altitud relevante desde CSV: {altitude:.0f} msnm.")
            except Exception:
                pass

    return clamp(lambda_home, 0.15, 4.50), clamp(lambda_away, 0.15, 4.50), factors


def apply_web_fact_adjustments(lambda_home, lambda_away, match, web_summary: dict):
    facts = web_summary.get("facts_df")
    if facts is None or facts.empty:
        return lambda_home, lambda_away, []

    home = match["home"]
    away = match["away"]
    factors = []
    home_adjust = 0.0
    away_adjust = 0.0

    for _, row in facts.sort_values("confidence", ascending=False).head(12).iterrows():
        fact_type = str(row.get("fact_type", ""))
        team = str(row.get("team", "") or "")
        value = str(row.get("value", "") or "")
        domain = str(row.get("source_domain", "web") or "web")
        lower = value.lower()

        if fact_type not in {"injury", "doubt", "lineup", "key_player"} or not team:
            continue

        if fact_type == "injury":
            delta = -0.06
            if any(word in lower for word in ["goalkeeper", "keeper", "defender", "centre-back", "portero", "defensa"]):
                if team.lower() == home.lower():
                    away_adjust += 0.04
                elif team.lower() == away.lower():
                    home_adjust += 0.04
            if team.lower() == home.lower():
                home_adjust += delta
            elif team.lower() == away.lower():
                away_adjust += delta
            factors.append(f"Web: baja/lesion en {team} segun {domain}; ajuste ofensivo moderado.")
        elif fact_type == "doubt":
            delta = -0.03
            if team.lower() == home.lower():
                home_adjust += delta
            elif team.lower() == away.lower():
                away_adjust += delta
            factors.append(f"Web: duda en {team} segun {domain}; ajuste ofensivo suave.")
        elif fact_type in {"lineup", "key_player"} and any(word in lower for word in ["starts", "starting", "confirmed", "titular"]):
            delta = 0.03
            if team.lower() == home.lower():
                home_adjust += delta
            elif team.lower() == away.lower():
                away_adjust += delta
            factors.append(f"Web: nota de alineacion/jugador clave en {team} segun {domain}; ajuste +3%.")

    home_adjust = clamp(home_adjust, -0.10, 0.10)
    away_adjust = clamp(away_adjust, -0.10, 0.10)
    return (
        clamp(lambda_home * (1 + home_adjust), 0.15, 4.50),
        clamp(lambda_away * (1 + away_adjust), 0.15, 4.50),
        factors[:4],
    )


def blend_model_with_market(model_probs: dict, market_probs: dict, data_score: float) -> dict:
    """
    Mezcla probabilidades del modelo con consenso de mercado sin tratar odds como verdad absoluta.
    """
    if not model_probs or not market_probs:
        return model_probs
    if set(model_probs.keys()) != set(market_probs.keys()):
        return model_probs

    if data_score < 0.65:
        market_weight = 0.35
    elif data_score <= 0.85:
        market_weight = 0.20
    else:
        market_weight = 0.10

    blended = {
        key: (1 - market_weight) * float(model_probs.get(key, 0) or 0) + market_weight * float(market_probs.get(key, 0) or 0)
        for key in model_probs
    }
    total = sum(blended.values())
    return {key: value / total for key, value in blended.items()} if total > 0 else model_probs


def calculate_data_quality_score(match, analysis, web_summary):
    home_info = analysis.get("home_info", {})
    away_info = analysis.get("away_info", {})
    h2h = analysis.get("h2h")
    injuries = analysis.get("injuries")
    players = analysis.get("players")
    coverage = float(analysis.get("coverage", 1.0) or 1.0)
    source_count = int(web_summary.get("source_count", 0) or 0)
    web_confidence = float(web_summary.get("confidence", 0) or 0)
    enough_web = source_count >= settings.web_confidence_min_sources and web_confidence >= 0.55

    positives = []
    negatives = []
    score = 0.40
    both_fallback = bool(home_info.get("is_fallback") and away_info.get("is_fallback"))

    if not home_info.get("is_fallback") and not away_info.get("is_fallback"):
        score += 0.25
        positives.append("forma real para ambos equipos")
    elif not home_info.get("is_fallback") or not away_info.get("is_fallback"):
        score += 0.12
        positives.append("forma real parcial")
    else:
        negatives.append("fallback en ambos equipos")

    has_xg = home_info.get("has_xg") or away_info.get("has_xg") or web_summary.get("xg_home") is not None or web_summary.get("xg_away") is not None
    if has_xg:
        score += 0.15 if enough_web or home_info.get("has_xg") or away_info.get("has_xg") else 0.05
        positives.append("xG disponible en CSV/API o web")
    else:
        negatives.append("sin xG/xGA")

    if h2h is not None and not h2h.empty:
        score += 0.10
        positives.append("H2H local disponible")
    elif web_summary.get("h2h_summary") and enough_web:
        score += 0.10
        positives.append("H2H externo con fuentes web")
    else:
        negatives.append("sin H2H")

    has_injuries = injuries is not None and not injuries.empty
    has_web_news = bool(web_summary.get("injuries_home") or web_summary.get("injuries_away"))
    if has_injuries or (has_web_news and enough_web):
        score += 0.10
        positives.append("lesiones/dudas disponibles")
    else:
        negatives.append("sin lesiones/dudas")

    has_players = players is not None and not players.empty
    has_web_lineups = bool(web_summary.get("lineups_notes"))
    if has_players or (has_web_lineups and enough_web):
        score += 0.10
        positives.append("jugadores o lineups disponibles")
    else:
        negatives.append("sin jugadores/lineups")

    if web_summary.get("market_probs") and enough_web:
        score += 0.10
        positives.append("consenso de mercado disponible")
    elif web_summary.get("market_probs") and source_count == 1:
        score += 0.05
        positives.append("mercado web con una fuente")
    else:
        negatives.append("sin odds/mercado")

    if home_info.get("strength_available") and away_info.get("strength_available"):
        score += 0.10
        positives.append("strength score disponible")
    else:
        negatives.append("strength score incompleto")

    if coverage < 0.97:
        score -= 0.05
        negatives.append("cobertura de matriz incompleta")

    if source_count == 1:
        score = min(score, 0.05 + (0.50 if both_fallback else score))
    if both_fallback and source_count < settings.web_confidence_min_sources:
        score = min(score, 0.55)

    score = clamp(score, 0.25, 1.0)
    if score >= 0.80:
        level = "alta"
    elif score >= 0.60:
        level = "media"
    else:
        level = "baja"
    return {
        "score": score,
        "razones_positivas": positives,
        "razones_negativas": negatives,
        "nivel": level,
    }

def fit_or_select_dixon_coles_rho(results_df: pd.DataFrame) -> tuple[float, str]:
    if not settings.use_dixon_coles:
        return 0.0, "Dixon-Coles desactivado"
    valid_results = results_df
    if (
        settings.ignore_mock_results_for_calibration
        and results_df is not None
        and not results_df.empty
        and "result_source" in results_df.columns
    ):
        valid_results = results_df[
            results_df["result_source"].fillna("mock").astype(str).isin(["api_real", "csv_real", "manual"])
        ].copy()
    if valid_results is None or valid_results.empty or len(valid_results) < 30:
        return 0.0, "Muestra insuficiente para usar rho agresivo"
    bias = detect_draw_bias(valid_results)
    if settings.disable_dc_if_draw_bias_high and bias.get("draw_bias_level") == "Alto":
        return max(settings.dixon_coles_rho, 0.0), "Rho no negativo porque se detecto sesgo alto de empate"
    return settings.dixon_coles_rho, "Rho configurado"


def poisson_score_matrix(lambda_home, lambda_away, max_goals=8, rho: float | None = None):
    if settings.use_dixon_coles:
        return poisson_score_matrix_dc(
            lambda_home,
            lambda_away,
            rho=settings.dixon_coles_rho if rho is None else rho,
            max_goals=max_goals,
        )

    matrix = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            matrix[(hg, ag)] = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)
    coverage = sum(matrix.values())
    return matrix, coverage

def scoreline_type(hg, ag, lambda_home, lambda_away, match_context):
    if match_context.get("low_confidence"):
        return "Baja confianza"
    total_lambda = lambda_home + lambda_away
    diff = abs(lambda_home - lambda_away)
    if hg == ag and hg <= 1:
        return "Empate bajo"
    if total_lambda >= 3.2 and max(hg, ag) >= 3:
        return "Partido abierto"
    if diff >= 0.45 and abs(hg - ag) >= 2:
        return "Favorito fuerte"
    if abs(hg - ag) == 1:
        return "Victoria minima"
    return "Partido abierto" if total_lambda >= 2.8 else "Balanceado"


def rank_scorelines_with_context(matrix, lambda_home, lambda_away, match_context):
    ordered = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    return [
        (score, prob, scoreline_type(score[0], score[1], lambda_home, lambda_away, match_context))
        for score, prob in ordered
    ]


def top_scorelines(lambda_home, lambda_away, home, away, top_n=5, rho: float | None = None, match_context: dict | None = None):
    matrix, coverage = poisson_score_matrix(lambda_home, lambda_away, rho=rho)

    match_context = match_context or {}
    ordered = rank_scorelines_with_context(matrix, lambda_home, lambda_away, match_context)
    results = []

    for rank, ((hg, ag), p, kind) in enumerate(ordered[:top_n], start=1):
        if hg > ag:
            note = f"ventaja de {home}"
        elif ag > hg:
            note = f"ventaja de {away}"
        else:
            note = "empate estadísticamente fuerte"

        results.append({
            "#": rank,
            "Resultado": f"{hg} - {ag}",
            "Probabilidad": p,
            "Tipo": kind,
            "Justificación": f"λ {home}={lambda_home:.2f}, λ {away}={lambda_away:.2f}; {note}.",
        })

    return results, coverage

def outcome_probabilities_from_matrix(matrix: dict):
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for (hg, ag), p in matrix.items():
        if hg > ag:
            home_win += p
        elif hg == ag:
            draw += p
        else:
            away_win += p

    total = home_win + draw + away_win
    if total > 0:
        return home_win / total, draw / total, away_win / total

    return None, None, None


def outcome_probabilities(lambda_home, lambda_away, max_goals=12, rho: float | None = None):
    matrix, _ = poisson_score_matrix(lambda_home, lambda_away, max_goals=max_goals, rho=rho)
    return outcome_probabilities_from_matrix(matrix)


def simulate_match_monte_carlo(lambda_home, lambda_away, n=20000, use_dc=False, rho: float | None = None):
    n = max(1000, int(n or 20000))
    seed = int((lambda_home * 1000) + (lambda_away * 10000)) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    if use_dc:
        matrix, _ = poisson_score_matrix(lambda_home, lambda_away, max_goals=8, rho=rho)
        score_items = list(matrix.items())
        probabilities = np.array([max(0.0, prob) for _, prob in score_items], dtype=float)
        probabilities = probabilities / probabilities.sum()
        picks = rng.choice(len(score_items), size=n, p=probabilities)
        home_goals = np.array([score_items[i][0][0] for i in picks])
        away_goals = np.array([score_items[i][0][1] for i in picks])
    else:
        home_goals = rng.poisson(lambda_home, n)
        away_goals = rng.poisson(lambda_away, n)

    home_win = home_goals > away_goals
    draw = home_goals == away_goals
    away_win = home_goals < away_goals

    scores = {}
    for hg, ag in zip(home_goals, away_goals):
        if hg <= 8 and ag <= 8:
            key = f"{int(hg)} - {int(ag)}"
            scores[key] = scores.get(key, 0) + 1

    top_5 = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    probs = {
        "prob_home_win": float(home_win.mean()),
        "prob_draw": float(draw.mean()),
        "prob_away_win": float(away_win.mean()),
    }

    ci = {}
    for key, prob in probs.items():
        se = math.sqrt(max(prob * (1 - prob), 0) / n)
        ci[key] = (max(0.0, prob - 1.96 * se), min(1.0, prob + 1.96 * se))

    return {
        **probs,
        "score_counts": scores,
        "top_5_scores": [(score, count / n) for score, count in top_5],
        "confidence_interval_1x2": ci,
        "simulations": n,
        "use_dc": use_dc,
    }

def player_first_goal_candidates(player_df, home, away, p_home_first, p_away_first):
    if player_df is None or player_df.empty:
        return []

    candidates = []

    for team, team_first_prob in [(home, p_home_first), (away, p_away_first)]:
        team_players = player_df[player_df["team"].str.lower() == team.lower()].copy()
        if team_players.empty or team_first_prob is None:
            continue

        for col in ["goals_min_1_30", "goals_last10", "shots_last10", "starts_last10"]:
            if col not in team_players.columns:
                team_players[col] = 0

        if "penalty_taker" not in team_players.columns:
            team_players["penalty_taker"] = False

        team_players["penalty_bonus"] = team_players["penalty_taker"].astype(str).str.lower().isin(["true", "1", "yes", "si", "sí"]).astype(float)

        team_players["score"] = (
            team_players["goals_min_1_30"].fillna(0).astype(float) * 2.0 +
            team_players["goals_last10"].fillna(0).astype(float) * 1.3 +
            team_players["shots_last10"].fillna(0).astype(float) * 0.20 +
            team_players["starts_last10"].fillna(0).astype(float) * 0.08 +
            team_players["penalty_bonus"] * 1.5
        )

        total_score = team_players["score"].sum()
        if total_score <= 0:
            continue

        for _, row in team_players.sort_values("score", ascending=False).head(3).iterrows():
            p = team_first_prob * (row["score"] / total_score)
            candidates.append({
                "Jugador": row["player"],
                "Equipo": team,
                "Probabilidad": p,
                "Razón": (
                    f"{safe_int(row.get('goals_last10'))} goles últimos 10, "
                    f"{safe_int(row.get('shots_last10'))} tiros, "
                    f"{safe_int(row.get('goals_min_1_30'))} goles min 1-30."
                )
            })

    return sorted(candidates, key=lambda x: x["Probabilidad"], reverse=True)[:3]

def pressure_factor(match):
    group = match.get("group", "N/D")
    group_table = get_group_table(group)
    if group_table.empty:
        return "No hay tabla de grupo cargada para medir presión clasificatoria."

    home = str(match.get("home", ""))
    away = str(match.get("away", ""))

    rows = group_table[group_table["team"].str.lower().isin([home.lower(), away.lower()])]
    if rows.empty:
        return "Tabla de grupo cargada, pero no coincide con los equipos."

    details = []
    for _, row in rows.iterrows():
        details.append(f"{row['team']}: {safe_int(row.get('points'))} pts, posición {safe_int(row.get('position'))}")

    return "Presión de grupo: " + "; ".join(details)

def analyze_match(match: dict) -> dict:
    home = match["home"]
    away = match["away"]
    web_summary = summarize_web_enrichment(match)

    lambda_home, missing_home, home_info = estimate_team_lambda_advanced(
        home,
        away,
        match,
        is_home_side=True,
        web_summary=web_summary,
    )
    lambda_away, missing_away, away_info = estimate_team_lambda_advanced(
        away,
        home,
        match,
        is_home_side=False,
        web_summary=web_summary,
    )

    missing = sorted(set(missing_home + missing_away))

    if lambda_home is None or lambda_away is None:
        return {
            "ok": False,
            "reason": "No hay datos suficientes para calcular probabilidades reales.",
            "missing": missing,
        }

    lambda_home, lambda_away, h2h_factor = h2h_adjustment(lambda_home, lambda_away, home, away)

    injuries = get_injuries(home, away)
    lambda_home, lambda_away, context_factors = apply_context_adjustments(lambda_home, lambda_away, match, injuries)
    lambda_home, lambda_away, web_factors = apply_web_fact_adjustments(lambda_home, lambda_away, match, web_summary)

    backtest_summary = load_backtest_summary()
    bias_report = detect_draw_bias(backtest_summary["results_df"])
    effective_rho, rho_reason = fit_or_select_dixon_coles_rho(backtest_summary["results_df"])
    both_fallback = bool(home_info.get("is_fallback") and away_info.get("is_fallback"))
    match_context = {
        "fallback": both_fallback,
        "low_confidence": both_fallback,
        "bias_report": bias_report,
    }
    top_results, coverage = top_scorelines(
        lambda_home,
        lambda_away,
        home,
        away,
        rho=effective_rho,
        match_context=match_context,
    )
    p_home, p_draw, p_away = outcome_probabilities(lambda_home, lambda_away, rho=effective_rho)
    direct_winner_probs = {
        home: p_home,
        "Empate": p_draw,
        away: p_away,
    }

    p_home_first, p_away_first, p_no_goal = poisson_first_goal_probs(lambda_home, lambda_away)

    player_df = get_player_form(home, away)
    player_candidates = player_first_goal_candidates(player_df, home, away, p_home_first, p_away_first)

    h2h = get_h2h(home, away)
    quality = calculate_data_quality_score(
        match,
        {
            "home_info": home_info,
            "away_info": away_info,
            "h2h": h2h,
            "injuries": injuries,
            "players": player_df,
            "coverage": coverage,
        },
        web_summary,
    )
    data_score = quality["score"]

    calibration = load_calibration()
    winner_probs = apply_1x2_calibration(direct_winner_probs, calibration)
    winner_probs = apply_draw_bias_correction(winner_probs, bias_report)
    pre_market_winner_probs = dict(winner_probs)
    market_probs = web_summary.get("market_probs", {})
    market_adjusted = bool(
        market_probs
        and web_summary.get("source_count", 0) >= settings.web_confidence_min_sources
        and web_summary.get("confidence", 0) >= 0.55
    )
    if market_adjusted:
        winner_probs = blend_model_with_market(winner_probs, market_probs, data_score)

    predicted_winner = max(winner_probs, key=winner_probs.get)
    predicted_prob = winner_probs[predicted_winner]
    confidence = (
        "BAJA"
        if both_fallback and web_summary.get("source_count", 0) < settings.web_confidence_min_sources
        else confidence_label(data_score)
    )
    warnings = []
    if both_fallback:
        warnings.append("Predicción basada en fallback. No usar como pronóstico fuerte.")
    if abs(lambda_home - lambda_away) < 0.05 and data_score < 0.65:
        warnings.append(
            "El modelo igualó demasiado a los equipos por falta de datos; considera actualizar forma real desde API-Football."
        )

    if web_summary.get("source_count", 0) == 1:
        warnings.append("Enriquecimiento web con una sola fuente: impacto limitado por baja confirmacion.")

    factors = [
        f"Expectativa de goles: {home} {lambda_home:.2f} vs {away} {lambda_away:.2f}.",
        f"Método base: {home_info.get('lambda_method')} / {away_info.get('lambda_method')}.",
        h2h_factor,
        pressure_factor(match),
    ]

    factors.extend(home_info.get("advanced_factors", [])[:2])
    factors.extend(away_info.get("advanced_factors", [])[:2])
    factors.extend(context_factors)
    factors.extend(web_factors)
    factors.extend(warnings)

    monte_carlo = None
    if settings.use_monte_carlo:
        monte_carlo = simulate_match_monte_carlo(
            lambda_home,
            lambda_away,
            n=settings.monte_carlo_sims,
            use_dc=settings.use_dixon_coles,
            rho=effective_rho,
        )

    return {
        "ok": True,
        "home": home,
        "away": away,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "top_results": top_results,
        "winner_probs": winner_probs,
        "direct_winner_probs": direct_winner_probs,
        "pre_market_winner_probs": pre_market_winner_probs,
        "market_probs": market_probs,
        "market_adjusted": market_adjusted,
        "first_goal": {
            home: p_home_first,
            away: p_away_first,
            "Sin gol": p_no_goal,
        },
        "player_first_goal": player_candidates,
        "predicted_winner": predicted_winner,
        "predicted_prob": predicted_prob,
        "confidence": confidence,
        "data_score": data_score,
        "data_quality": quality,
        "factors": factors[:8],
        "missing": missing,
        "h2h": h2h,
        "injuries": injuries,
        "players": player_df,
        "coverage": coverage,
        "model_name": "Poisson + Dixon-Coles" if settings.use_dixon_coles else "Poisson simple",
        "configured_rho": settings.dixon_coles_rho,
        "effective_rho": effective_rho,
        "rho_reason": rho_reason,
        "calibration": calibration,
        "calibration_active": bool(calibration.get("active")),
        "monte_carlo": monte_carlo,
        "warnings": warnings,
        "bias_report": bias_report,
        "web_summary": web_summary,
        "team_data": {
            home: {
                "source": home_info.get("form_source", "N/D"),
                "confidence": home_info.get("form_confidence", "N/D"),
                "is_fallback": home_info.get("is_fallback", False),
                "strength_available": home_info.get("strength_available", False),
                "xg_source": home_info.get("xg_source"),
            },
            away: {
                "source": away_info.get("form_source", "N/D"),
                "confidence": away_info.get("form_confidence", "N/D"),
                "is_fallback": away_info.get("is_fallback", False),
                "strength_available": away_info.get("strength_available", False),
                "xg_source": away_info.get("xg_source"),
            },
        },
    }

def evaluate_played_match(match: dict, analysis: dict) -> dict | None:
    if not analysis.get("ok"):
        return None

    if str(match.get("status")) != "Jugado":
        return None

    try:
        real_h = int(match.get("home_goals"))
        real_a = int(match.get("away_goals"))
    except Exception:
        return None

    real_score = f"{real_h} - {real_a}"
    top = analysis["top_results"]

    rank = None
    prob = None
    for item in top:
        if item["Resultado"] == real_score:
            rank = item["#"]
            prob = item["Probabilidad"]
            break

    winner_probs = analysis["winner_probs"]
    home = match["home"]
    away = match["away"]

    if real_h > real_a:
        real_outcome = home
    elif real_a > real_h:
        real_outcome = away
    else:
        real_outcome = "Empate"

    predicted = analysis["predicted_winner"]
    top1_score = top[0]["Resultado"] if top else None
    comment = "Evaluacion consistente con el resultado observado."
    if predicted == real_outcome and top1_score != real_score:
        if top1_score and top1_score.endswith(" - " + top1_score.split(" - ")[0]):
            comment = (
                "El modelo acerto ganador, pero el marcador top fue empate. "
                "Posible sesgo hacia empate o lambda rival algo alta."
            )
        else:
            comment = "El modelo acerto ganador, pero no el marcador mas probable."
    elif predicted != real_outcome:
        comment = "El modelo fallo el resultado 1X2; revisar calibracion, fuerza rival y datos recientes."

    return {
        "real_score": real_score,
        "real_outcome": real_outcome,
        "predicted_outcome": predicted,
        "outcome_hit": real_outcome == predicted,
        "score_rank": rank,
        "score_prob": prob,
        "top1_score_hit": rank == 1,
        "top3_score_hit": rank is not None and rank <= 3,
        "top5_score_hit": rank is not None and rank <= 5,
        "brier_score": calculate_brier_score_1x2(winner_probs, real_outcome),
        "log_loss": calculate_log_loss_1x2(winner_probs, real_outcome),
        "comment": comment,
    }
