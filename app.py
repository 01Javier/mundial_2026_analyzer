import pandas as pd
import streamlit as st

from src.backtesting import evaluate_prediction, load_backtest_summary, save_prediction
from src.cache_store import read_cache_metadata
from src.calibration import detect_draw_bias, fit_simple_calibration, load_calibration
from src.config import settings
from src.data_sources import fetch_sports_data
from src.model import analyze_match
from src.render import format_match_table, render_match_analysis
from src.team_registry import get_missing_teams_in_recent_form, load_worldcup_teams, refresh_local_team_catalog

st.set_page_config(
    page_title="Analizador Mundial 2026",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Analizador estadístico — Mundial FIFA 2026")
st.caption(
    "48 equipos, 12 grupos, sede USA/México/Canadá. "
    "Modelo Poisson con forma reciente, xG, H2H, lesiones, jugadores en forma y contexto."
)

with st.sidebar:
    st.header("Configuración")
    st.write("Zona horaria:", settings.guatemala_timezone)
    st.write("Mock activo:", settings.use_mock_data)
    st.write("API-Football:", "✅ configurado" if settings.api_football_key else "❌ sin key")
    st.write("football-data.org:", "✅ configurado" if settings.football_data_token else "❌ sin token")
    st.caption("Las credenciales no se muestran ni se guardan en cache.")
    refresh_from_api = st.button(
        "Actualizar datos desde API",
        help="Usa cuota de API-Football/football-data.org solo cuando presionas este boton.",
        type="primary",
    )
    cache_metadata = read_cache_metadata()
    if cache_metadata:
        with st.expander("Cache local"):
            for key, meta in cache_metadata.items():
                st.write(f"**{key}**")
                st.caption(
                    f"Proveedor: {meta.get('provider', 'N/D')} | "
                    f"Actualizado: {meta.get('cached_at', meta.get('updated_at', 'N/D'))} | "
                    f"Requests ahorrados: {meta.get('requests_saved', 0)}"
                )
    st.divider()
    st.markdown("### CSV esperados")
    st.code(
        "data/worldcup_matches.csv\n"
        "data/worldcup_teams.csv\n"
        "data/team_recent_form.csv\n"
        "data/team_strength.csv\n"
        "data/h2h.csv\n"
        "data/player_form.csv\n"
        "data/injuries.csv\n"
        "data/group_tables.csv\n"
        "data/stadiums.csv"
    )

st.markdown("## PASO 1 — MOSTRAR PARTIDOS DISPONIBLES")

matches_df, sources_used, errors = fetch_sports_data("fixtures", refresh=refresh_from_api)

if sources_used:
    st.success("Fuente de partidos: " + ", ".join(sources_used))

backtest_summary = load_backtest_summary()
calibration = fit_simple_calibration(backtest_summary["results_df"])
bias_report = detect_draw_bias(backtest_summary["results_df"])

st.markdown("## Rendimiento del modelo")
metric_cols = st.columns(7)
metric_cols[0].metric("Partidos evaluados", backtest_summary["evaluated_matches"])
metric_cols[1].metric(
    "Acierto ganador/empate",
    "N/D" if backtest_summary["winner_accuracy"] is None else f"{backtest_summary['winner_accuracy'] * 100:.1f}%",
)
metric_cols[2].metric(
    "Top 1 exacto",
    "N/D" if backtest_summary["top1_accuracy"] is None else f"{backtest_summary['top1_accuracy'] * 100:.1f}%",
)
metric_cols[3].metric(
    "Top 3 exacto",
    "N/D" if backtest_summary["top3_accuracy"] is None else f"{backtest_summary['top3_accuracy'] * 100:.1f}%",
)
metric_cols[4].metric(
    "Top 5 exacto",
    "N/D" if backtest_summary["top5_accuracy"] is None else f"{backtest_summary['top5_accuracy'] * 100:.1f}%",
)
metric_cols[5].metric(
    "Brier promedio",
    "N/D" if backtest_summary["avg_brier"] is None else f"{backtest_summary['avg_brier']:.3f}",
)
metric_cols[6].metric(
    "Log Loss promedio",
    "N/D" if backtest_summary["avg_log_loss"] is None else f"{backtest_summary['avg_log_loss']:.3f}",
)

panel_cols = st.columns(4)
with panel_cols[0]:
    st.markdown("### Calidad de datos")
    st.write("Modelo actual:", "Poisson + Dixon-Coles" if settings.use_dixon_coles else "Poisson simple")
    st.write("Monte Carlo:", "Activo" if settings.use_monte_carlo else "Inactivo")
with panel_cols[1]:
    st.markdown("### Calibración")
    st.write("Estado:", "Activa" if calibration.get("active") else "Inactiva")
    st.caption(calibration.get("reason", "N/D"))
with panel_cols[2]:
    st.markdown("### Backtesting")
    st.write("Archivo:", "data/model_results.csv")
    st.write("Predicciones:", "data/model_predictions.csv")
with panel_cols[3]:
    st.markdown("### Sesgos detectados")
    st.write("Empate:", bias_report.get("draw_bias", "N/D"))
    st.write("Local:", bias_report.get("home_bias", "N/D"))
    st.write("Visitante:", bias_report.get("away_bias", "N/D"))

with st.expander("Cómo mejorar la precisión del modelo"):
    st.markdown(
        """
        - **Backtesting:** guarda predicciones y evalua contra resultados reales.
        - **Brier Score:** mide calidad de probabilidades 1X2; menor es mejor.
        - **Log Loss:** penaliza con fuerza cuando el modelo asigna baja probabilidad al resultado real.
        - **Dixon-Coles:** corrige marcadores bajos como 0-0, 1-0, 0-1 y 1-1.
        - **Calibración:** ajusta sesgos acumulados solo con muestra suficiente.
        - **Recency weighting:** da mas peso a partidos recientes cuando existe historial partido a partido.
        - **Shrinkage:** evita sobreajustar lambdas cuando hay poca muestra.
        - **Fuerza del rival:** limita ajustes por diferencia de calidad a +/-12%.
        - **Monte Carlo:** verifica que las probabilidades directas sean estables por simulación.
        """
    )

with st.expander("Por qué no se debe calibrar con un solo partido"):
    st.write(
        "Ghana vs Panama sirve como ejemplo de diagnóstico, no como base para cambiar pesos. "
        "Un partido puede mostrar que el marcador real estuvo bien rankeado aunque el top fuera 1-1. "
        "Para detectar sesgos reales, como empates inflados, se necesita una muestra acumulada; "
        "por eso la calibración fuerte queda desactivada con menos de 20 partidos evaluados."
    )

if errors:
    with st.expander("Errores de proveedores"):
        for err in errors:
            st.warning(err)

if matches_df.empty:
    st.error("No se encontraron partidos disponibles.")
    st.stop()

catalog_status = refresh_local_team_catalog(matches_df)
teams_df = load_worldcup_teams()

with st.expander("Catalogo local de equipos y estadisticas"):
    st.write(f"Equipos en catalogo: **{catalog_status['teams_count']}**")
    if catalog_status["fallback_added"]:
        st.warning(
            "Se agregaron estadisticas fallback de baja confianza para: "
            + ", ".join(catalog_status["fallback_added"][:20])
            + ("..." if len(catalog_status["fallback_added"]) > 20 else "")
        )
    missing_recent_form = get_missing_teams_in_recent_form()
    if missing_recent_form:
        st.warning("Equipos aun sin forma reciente: " + ", ".join(missing_recent_form[:20]))
    else:
        st.success("Todos los equipos del catalogo tienen forma reciente o fallback local.")
    if not teams_df.empty:
        st.dataframe(
            teams_df[["team", "canonical_name", "group", "qualified_status", "source", "is_host"]],
            use_container_width=True,
            hide_index=True,
        )

display_df = format_match_table(matches_df)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.info("¿Qué partido(s) deseas analizar? Escribe los números o selecciónalos abajo.")

available_numbers = display_df["#"].tolist()

selected_numbers = st.multiselect(
    "Partidos a analizar",
    options=available_numbers,
    format_func=lambda n: f"{n} — {display_df.loc[display_df['#'] == n, 'Partido'].iloc[0]}",
)

manual_text = st.text_input("También puedes escribir números separados por coma. Ejemplo: 1, 4, 5")

if manual_text.strip():
    parsed = []
    for part in manual_text.split(","):
        part = part.strip()
        if part.isdigit():
            num = int(part)
            if num in available_numbers:
                parsed.append(num)
    if parsed:
        selected_numbers = sorted(set(selected_numbers + parsed))

if not selected_numbers:
    st.stop()

st.markdown("## PASO 2 — RECOLECCIÓN DE DATOS")
st.write(
    "Para cada partido elegido se combinan APIs disponibles y CSV locales. "
    "Si faltan datos, la app lo indica y baja la confianza del pronóstico."
)

st.markdown("## PASO 3 — ANÁLISIS Y SALIDA")

for n in selected_numbers:
    match = matches_df.reset_index(drop=True).iloc[n - 1].to_dict()
    analysis = analyze_match(match)
    source_mode = ", ".join(sources_used) if sources_used else str(match.get("source", "N/D"))
    save_prediction(match, analysis, source_mode=source_mode)
    evaluate_prediction(match, analysis)
    render_match_analysis(match, analysis)
