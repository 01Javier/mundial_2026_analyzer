import math
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .backtesting import calculate_brier_score_1x2, calculate_log_loss_1x2
from .calibration import apply_1x2_calibration, load_calibration
from .config import settings
from .dixon_coles import poisson_score_matrix_dc
from .data_sources import (
    get_team_recent_form,
    get_team_strength,
    get_h2h,
    get_player_form,
    get_injuries,
    get_group_table,
    get_stadium_info,
)
from .utils import clamp, confidence_label, poisson_first_goal_probs, safe_int

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
    team_strength = get_team_strength(team)
    opp_strength = get_team_strength(opponent)
    if not team_strength or not opp_strength:
        return 1.0, None

    diff = team_strength["strength_score"] - opp_strength["strength_score"]
    adjustment = clamp(diff / 400, -0.12, 0.12)
    return 1 + adjustment, f"Fuerza relativa {team}: ajuste {adjustment:+.1%} vs {opponent}."


def estimate_team_lambda_advanced(team, opponent, match, is_home_side):
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
    if opp_form is None:
        missing.append(f"forma reciente de {opponent}")
    if team_form is None or opp_form is None:
        return None, missing, info

    if team_form.get("data_quality") == "low":
        missing.append(f"forma estimada de {team}")
        factors.append(f"{team}: forma reciente fallback de baja confianza.")
    if opp_form.get("data_quality") == "low":
        missing.append(f"forma estimada de {opponent}")
        factors.append(f"{opponent}: forma rival fallback de baja confianza.")

    team_gf = float(team_form.get("gf_per_match") or 0)
    opp_ga = float(opp_form.get("ga_per_match") or 0)
    team_xg = team_form.get("xg_for")
    opp_xga = opp_form.get("xg_against")
    has_xg = team_xg is not None and opp_xga is not None

    if has_xg:
        base_lambda = (
            0.40 * float(team_xg) +
            0.25 * float(opp_xga) +
            0.20 * team_gf +
            0.15 * opp_ga
        )
        method = "xG/xGA + goles + shrinkage"
    else:
        base_lambda = 0.55 * team_gf + 0.45 * opp_ga
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

    if team.lower() in HOST_TEAMS:
        expected *= 1.04
        factors.append(f"{team} recibe ajuste ligero de anfitrion/localia.")

    info.update(
        {
            "lambda_method": method,
            "team_form": team_form,
            "opp_form": opp_form,
            "has_xg": has_xg,
            "sample_weight": sample_weight,
            "base_lambda": base_lambda,
            "global_avg_goals": avg_goals,
            "advanced_factors": factors,
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

def poisson_score_matrix(lambda_home, lambda_away, max_goals=8):
    if settings.use_dixon_coles:
        return poisson_score_matrix_dc(
            lambda_home,
            lambda_away,
            rho=settings.dixon_coles_rho,
            max_goals=max_goals,
        )

    matrix = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            matrix[(hg, ag)] = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)
    coverage = sum(matrix.values())
    return matrix, coverage

def top_scorelines(lambda_home, lambda_away, home, away, top_n=5):
    matrix, coverage = poisson_score_matrix(lambda_home, lambda_away)

    ordered = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    results = []

    for rank, ((hg, ag), p) in enumerate(ordered[:top_n], start=1):
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


def outcome_probabilities(lambda_home, lambda_away, max_goals=12):
    matrix, _ = poisson_score_matrix(lambda_home, lambda_away, max_goals=max_goals)
    return outcome_probabilities_from_matrix(matrix)


def simulate_match_monte_carlo(lambda_home, lambda_away, n=20000, use_dc=False):
    n = max(1000, int(n or 20000))
    seed = int((lambda_home * 1000) + (lambda_away * 10000)) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    if use_dc:
        matrix, _ = poisson_score_matrix(lambda_home, lambda_away, max_goals=8)
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

    lambda_home, missing_home, home_info = estimate_team_lambda_advanced(home, away, match, is_home_side=True)
    lambda_away, missing_away, away_info = estimate_team_lambda_advanced(away, home, match, is_home_side=False)

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

    top_results, coverage = top_scorelines(lambda_home, lambda_away, home, away)
    p_home, p_draw, p_away = outcome_probabilities(lambda_home, lambda_away)
    direct_winner_probs = {
        home: p_home,
        "Empate": p_draw,
        away: p_away,
    }
    calibration = load_calibration()
    winner_probs = apply_1x2_calibration(direct_winner_probs, calibration)

    p_home_first, p_away_first, p_no_goal = poisson_first_goal_probs(lambda_home, lambda_away)

    player_df = get_player_form(home, away)
    player_candidates = player_first_goal_candidates(player_df, home, away, p_home_first, p_away_first)

    predicted_winner = max(winner_probs, key=winner_probs.get)
    predicted_prob = winner_probs[predicted_winner]

    h2h = get_h2h(home, away)

    data_score = 1.0
    if missing:
        data_score -= 0.25
    if h2h.empty:
        data_score -= 0.10
    if injuries.empty:
        data_score -= 0.05
    if player_df.empty:
        data_score -= 0.10
    if coverage < 0.97:
        data_score -= 0.05

    data_score = clamp(data_score, 0.25, 1.0)

    factors = [
        f"Expectativa de goles: {home} {lambda_home:.2f} vs {away} {lambda_away:.2f}.",
        f"Método base: {home_info.get('lambda_method')} / {away_info.get('lambda_method')}.",
        h2h_factor,
        pressure_factor(match),
    ]

    factors.extend(home_info.get("advanced_factors", [])[:2])
    factors.extend(away_info.get("advanced_factors", [])[:2])
    factors.extend(context_factors)

    monte_carlo = None
    if settings.use_monte_carlo:
        monte_carlo = simulate_match_monte_carlo(
            lambda_home,
            lambda_away,
            n=settings.monte_carlo_sims,
            use_dc=settings.use_dixon_coles,
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
        "first_goal": {
            home: p_home_first,
            away: p_away_first,
            "Sin gol": p_no_goal,
        },
        "player_first_goal": player_candidates,
        "predicted_winner": predicted_winner,
        "predicted_prob": predicted_prob,
        "confidence": confidence_label(data_score),
        "data_score": data_score,
        "factors": factors[:8],
        "missing": missing,
        "h2h": h2h,
        "injuries": injuries,
        "players": player_df,
        "coverage": coverage,
        "model_name": "Poisson + Dixon-Coles" if settings.use_dixon_coles else "Poisson simple",
        "calibration": calibration,
        "calibration_active": bool(calibration.get("active")),
        "monte_carlo": monte_carlo,
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
