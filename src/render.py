import pandas as pd
import streamlit as st

from .config import settings
from .utils import gt_time_from_utc, pct, probability_bar, safe_int
from .model import evaluate_played_match

def format_match_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()

    def fmt_date(x):
        dt = gt_time_from_utc(str(x), settings.gt_tz)
        if not dt:
            return "N/D"
        return dt.strftime("%d %b %Y %H:%M")

    view["Fecha Guatemala"] = view["date_utc"].apply(fmt_date)
    view["Partido"] = view["home"].astype(str) + " vs " + view["away"].astype(str)

    view = view.reset_index(drop=True)
    view.insert(0, "#", view.index + 1)

    return view[["#", "Partido", "group", "Fecha Guatemala", "status"]].rename(
        columns={"group": "Grupo", "status": "Estado"}
    )

def render_h2h(h2h: pd.DataFrame):
    if h2h is None or h2h.empty:
        st.caption("H2H: no cargado.")
        return

    st.markdown("#### Últimos enfrentamientos directos")
    display = h2h.copy()
    st.dataframe(display, use_container_width=True, hide_index=True)

def render_injuries(injuries: pd.DataFrame):
    if injuries is None or injuries.empty:
        st.caption("Lesiones: no hay registros cargados.")
        return

    st.markdown("#### Ausencias o dudas")
    st.dataframe(injuries, use_container_width=True, hide_index=True)

def render_match_analysis(match: dict, analysis: dict):
    home = match["home"]
    away = match["away"]
    group = match.get("group", "N/D")
    venue = match.get("venue", "N/D")
    status = match.get("status", "N/D")

    dt_gt = gt_time_from_utc(str(match.get("date_utc", "")), settings.gt_tz)
    dt_text = dt_gt.strftime("%d %b %Y") if dt_gt else "N/D"
    hour_text = dt_gt.strftime("%H:%M Guatemala CT") if dt_gt else "N/D"

    st.markdown("══════════════════════════════════════════════════════")
    st.markdown(f"## MUNDIAL 2026 — GRUPO {group}")
    st.markdown(f"### {home} 🆚 {away}")
    st.markdown(f"📅 **{dt_text}** | 🏟️ **{venue}** | ⏰ **{hour_text}**")

    if status == "Jugado":
        st.info(
            f"Partido jugado. Resultado real: "
            f"**{safe_int(match.get('home_goals'))} - {safe_int(match.get('away_goals'))}**."
        )

    if not analysis["ok"]:
        st.error(analysis["reason"])
        if analysis.get("missing"):
            st.warning("Datos faltantes: " + ", ".join(analysis["missing"]))
        return

    st.markdown("### Modelo y calidad de datos")
    model_cols = st.columns(4)
    model_cols[0].metric("Modelo actual", analysis.get("model_name", "Poisson simple"))
    model_cols[1].metric("Calibracion", "Activa" if analysis.get("calibration_active") else "Inactiva")
    model_cols[2].metric("Calidad de datos", f"{analysis.get('data_score', 0):.2f}")
    model_cols[3].metric("Cobertura matriz", pct(analysis.get("coverage")))
    if not analysis.get("calibration_active"):
        st.caption("Calibracion desactivada por muestra pequena o datos insuficientes.")

    evaluation = evaluate_played_match(match, analysis)
    if evaluation:
        if evaluation["score_rank"]:
            st.success(
                f"Evaluación del modelo: el marcador real {evaluation['real_score']} "
                f"estaba en el Top {evaluation['score_rank']} "
                f"con probabilidad {pct(evaluation['score_prob'])}."
            )
        else:
            st.warning(
                f"Evaluación del modelo: el marcador real {evaluation['real_score']} "
                f"no estaba dentro del Top 5 de marcadores exactos."
            )

        st.write(
            f"Resultado predicho: **{evaluation['predicted_outcome']}** | "
            f"Resultado real: **{evaluation['real_outcome']}** | "
            f"Acierto ganador/empate: **{'Sí' if evaluation['outcome_hit'] else 'No'}**"
        )
        st.write(
            f"Top 1: **{'Si' if evaluation['top1_score_hit'] else 'No'}** | "
            f"Top 3: **{'Si' if evaluation['top3_score_hit'] else 'No'}** | "
            f"Top 5: **{'Si' if evaluation['top5_score_hit'] else 'No'}** | "
            f"Brier: **{evaluation['brier_score']:.3f}** | "
            f"Log Loss: **{evaluation['log_loss']:.3f}**"
        )
        st.caption(evaluation["comment"])

    st.markdown("### 📊 TOP 5 RESULTADOS MÁS PROBABLES")

    top_df = pd.DataFrame(analysis["top_results"])
    top_df["Probabilidad"] = top_df["Probabilidad"].apply(lambda x: f"{x * 100:.1f}%")
    st.table(top_df)

    st.caption("Las probabilidades se calculan con Poisson usando λ derivados de xG, forma reciente y ajustes contextuales.")

    st.markdown("### ⚽ PRIMER GOL — ¿Quién anota primero?")

    if analysis["player_first_goal"]:
        for item in analysis["player_first_goal"]:
            st.markdown(
                f"• **{item['Jugador']}** ({item['Equipo']}) ........... "
                f"**{pct(item['Probabilidad'])}**  \n"
                f"  Razón: {item['Razón']}"
            )
    else:
        st.caption("No hay datos suficientes de jugadores para estimar primer goleador individual.")

    first_goal = analysis["first_goal"]
    st.markdown(f"• **{home} completo** anota 1ro .. **{pct(first_goal[home])}**")
    st.markdown(f"• **{away} completo** anota 1ro .. **{pct(first_goal[away])}**")
    st.markdown(f"• **Sin gol** .................... **{pct(first_goal['Sin gol'])}**")

    st.markdown("### 🏆 PROBABILIDAD DE GANADOR")

    for label, prob in analysis["winner_probs"].items():
        st.markdown(f"`{probability_bar(prob)}` **{label}** .......... **{pct(prob)}**")

    direct_probs = analysis.get("direct_winner_probs", {})
    monte_carlo = analysis.get("monte_carlo")
    if direct_probs or monte_carlo:
        with st.expander("Probabilidad directa Poisson vs Monte Carlo"):
            comparison_rows = []
            labels = list(analysis["winner_probs"].keys())
            for idx, label in enumerate(labels):
                direct = direct_probs.get(label)
                mc = None
                if monte_carlo:
                    mc_key = ["prob_home_win", "prob_draw", "prob_away_win"][idx]
                    mc = monte_carlo.get(mc_key)
                comparison_rows.append(
                    {
                        "Resultado": label,
                        "Poisson directo": pct(direct),
                        "Monte Carlo": pct(mc),
                        "Diferencia": "N/D" if direct is None or mc is None else f"{(mc - direct) * 100:+.2f} pp",
                    }
                )
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
            if monte_carlo:
                st.caption(f"Simulaciones: {monte_carlo.get('simulations'):,}")

    st.success(
        f"→ Ganador predicho: **{analysis['predicted_winner']}** "
        f"(confianza: **{analysis['confidence']}**, calidad de datos: {analysis['data_score']:.2f})"
    )

    st.markdown("### 📌 FACTORES DETERMINANTES")
    for i, factor in enumerate(analysis["factors"], start=1):
        st.markdown(f"{i}. {factor}")

    if analysis.get("missing"):
        st.warning("Datos faltantes o incompletos: " + ", ".join(analysis["missing"]))

    with st.expander("Ver H2H, lesiones y jugadores cargados"):
        render_h2h(analysis.get("h2h"))
        render_injuries(analysis.get("injuries"))

        players = analysis.get("players")
        if players is not None and not players.empty:
            st.markdown("#### Jugadores en forma")
            st.dataframe(players, use_container_width=True, hide_index=True)
        else:
            st.caption("Jugadores en forma: no cargado.")

    st.markdown("### 📎 Fuentes consultadas")
    sources = [
        match.get("source", "N/D"),
        "CSV local: data/team_recent_form.csv",
        "CSV local: data/h2h.csv",
        "CSV local: data/player_form.csv",
        "CSV local: data/injuries.csv",
        "CSV local: data/group_tables.csv",
        "CSV local: data/stadiums.csv para sede/altitud",
    ]

    for s in sources:
        st.markdown(f"- {s}")
