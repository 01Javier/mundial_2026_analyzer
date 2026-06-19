# Analizador estadistico - Mundial FIFA 2026

App en Python + Streamlit para analizar partidos del Mundial 2026 con datos de APIs, CSV locales y un modelo Poisson para probabilidades de marcador, ganador y primer gol.

## Requisitos

- Windows con PowerShell.
- Python 3.9 o superior. El proyecto usa `zoneinfo`, incluido en Python desde la version 3.9.
- No necesitas Visual Studio Build Tools si instalas con las dependencias flexibles de este proyecto.

## Instalacion en Windows PowerShell

Ejecuta estos comandos desde PowerShell:

```powershell
cd C:\Users\Javier\mundial_2026_analyzer
deactivate
Remove-Item -Recurse -Force .venv

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

copy .env.example .env
python -m streamlit run app.py
```

Si `deactivate` muestra que no se reconoce el comando, significa que no habia un entorno virtual activo. Puedes continuar con el siguiente comando.

## Validacion opcional

Antes de abrir la app, puedes confirmar que Python, pip, Streamlit y pandas quedaron instalados:

```powershell
python --version
python -m pip --version
python -m pip show streamlit
python -m pip show pandas
```

## Solucion de errores comunes

### Si `streamlit` no se reconoce

Usa Streamlit como modulo de Python:

```powershell
python -m streamlit run app.py
```

Esto evita depender de que el ejecutable `streamlit` este en el `PATH`.

### Si pip intenta compilar pandas desde fuente

Actualiza las herramientas de instalacion y usa este `requirements.txt` flexible:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

El proyecto no fija `pandas==2.2.3` de forma rigida, para que Python 3.13 o versiones nuevas puedan resolver una version con wheel compatible en Windows.

### Si PowerShell bloquea la activacion del entorno virtual

Puedes permitir scripts solo para la sesion actual:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

## Instalacion en Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
cp .env.example .env
python -m streamlit run app.py
```

## Datos en vivo y cache local

Por defecto, la app no consulta APIs deportivas en cada recarga de `localhost:8501`.
Primero intenta reutilizar cache local en `data/cache/`, luego CSV locales y finalmente mock data.

Para gastar una llamada real a API-Football o football-data.org, usa los botones manuales de la barra lateral de Streamlit. Si la llamada funciona, la respuesta se guarda en cache y las siguientes recargas leen ese archivo local.

La app usa estas fuentes, en este orden:

1. Cache fresco de API-Football.
2. API-Football en vivo, solo si presionas **Actualizar fixtures desde API-Football**.
3. Cache fresco de football-data.org.
4. football-data.org en vivo, solo como respaldo si API-Football falla y pediste actualizar.
5. CSV local `data/worldcup_matches.csv`.
6. Mock data si `USE_MOCK_DATA=true`.

Botones disponibles en la app:

- **Actualizar fixtures desde API-Football**: refresca partidos del Mundial y guarda `data/cache/api_football_fixtures_2026.json` y `data/cache/worldcup_matches_latest.csv`.
- **Actualizar forma de equipos faltantes desde API-Football**: busca forma reciente real para equipos con fallback.
- **Actualizar datos reales de estos equipos desde API-Football**: actualiza solo los dos equipos del partido seleccionado.
- **Actualizar H2H del partido seleccionado**: consulta enfrentamientos directos solo bajo demanda.
- **Actualizar lesiones del partido seleccionado**: consulta lesiones solo bajo demanda.
- **Actualizar lineups del partido seleccionado**: solo se permite si el partido esta en curso, ya jugado o dentro de 24 horas.
- **Buscar datos web de este partido**: usa Tavily o SerpAPI solo bajo demanda para enriquecer xG, odds, H2H, lesiones, lineups y notas de previa.

Archivos de cache:

```text
data/cache/api_football_fixtures_2026.json
data/cache/football_data_matches_2026.json
data/cache/worldcup_matches_latest.csv
data/cache/cache_metadata.json
```

`cache_metadata.json` guarda proveedor, endpoint, parametros sin credenciales, fecha de cache, vencimiento, archivo fuente, requests ahorrados y ultimo error.

Variables recomendadas en `.env`:

```text
API_PROVIDER_PRIORITY=api_football,football_data,csv,mock
ALLOW_API_ON_PAGE_LOAD=false
MAX_API_REQUESTS_PER_RUN=20
TEAM_FORM_CACHE_HOURS=72
USE_DIXON_COLES=true
DIXON_COLES_RHO=0.0
AUTO_FIT_DIXON_COLES=true
DISABLE_DC_IF_DRAW_BIAS_HIGH=true
USE_DRAW_BIAS_CORRECTION=true
IGNORE_MOCK_RESULTS_FOR_CALIBRATION=true
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=
SERPAPI_API_KEY=
ENABLE_WEB_ENRICHMENT=true
ALLOW_WEB_SEARCH_ON_PAGE_LOAD=false
WEB_SEARCH_CACHE_HOURS=24
MAX_WEB_SEARCHES_PER_RUN=5
WEB_CONFIDENCE_MIN_SOURCES=2
WEB_ALLOWED_DOMAINS=fifa.com,espn.com,skysports.com,theanalyst.com,lineups.com,fotmob.com,sofascore.com,flashscore.com,sportsmole.co.uk,wincomparator.com,oddschecker.com,squawka.com
WEB_BLOCKED_DOMAINS=reddit.com,facebook.com,twitter.com,x.com,tiktok.com
```

## Enriquecimiento web manual

La capa web complementa a API-Football, no la reemplaza. Sirve para buscar previews, xG publicado, lesiones, alineaciones probables, cuotas, H2H y noticias. No se ejecuta al abrir ni recargar Streamlit: solo corre si presionas **Buscar datos web de este partido**.

Resultados guardados:

```text
data/cache/web_search/
data/web_facts.csv
```

Cada hecho web guarda `source_url`, dominio, confianza y fecha. La app no usa redes sociales, Reddit, foros ni dominios bloqueados. Si una fuente solo contiene una prediccion sin dato concreto, se guarda como `source_note` o `market_consensus`, no como lesion/xG real.

## Calidad de datos

El `data_score` sube por capas reales:

- Forma real de ambos equipos desde API/CSV.
- xG desde CSV/API o xG externo con fuente clara.
- H2H local o web.
- Lesiones/dudas.
- Jugadores/lineups.
- Odds o consenso de mercado.
- Strength score.

Si ambos equipos siguen en fallback y no hay al menos 2 fuentes web confiables, la calidad queda limitada y la confianza se muestra baja.

## Por que la calidad queda en 0.50

Normalmente ocurre porque:

- Ambos equipos usan `registry_fallback`.
- No hay H2H local.
- No hay lesiones/dudas.
- No hay jugadores o lineups.
- Falta xG/xGA.
- No existen `web_facts` guardados para el partido.

## Como subir la calidad de datos

1. Configura `API_FOOTBALL_KEY`.
2. Pon `USE_MOCK_DATA=false` si quieres evitar datos simulados.
3. Presiona **Actualizar fixtures desde API-Football**.
4. Revisa o reconstruye el catalogo de equipos.
5. Presiona **Actualizar forma de equipos faltantes desde API-Football**.
6. Entra a un partido concreto.
7. Presiona **Buscar datos web de este partido**.
8. Revisa el panel **Enriquecimiento web**.
9. Analiza nuevamente el partido.

## Control de costos

API-Football Free suele tener limite diario y Tavily/SerPAPI pueden tener creditos limitados. Por eso:

- No hay llamadas API/web al recargar.
- Cada consulta requiere boton manual.
- Las respuestas se guardan en `data/cache/`.
- Las busquedas web frescas se reutilizan durante `WEB_SEARCH_CACHE_HOURS`.
- `MAX_API_REQUESTS_PER_RUN` y `MAX_WEB_SEARCHES_PER_RUN` limitan gasto por ejecucion.

## Seguridad

- No imprimas API keys en pantalla ni logs.
- No guardes API keys en archivos de cache.
- `.env`, `.tmp/`, `.venv/`, `__pycache__/` y caches generados estan ignorados por Git.

## Catalogo maestro de equipos

La app mantiene un catalogo local en:

```text
data/worldcup_teams.csv
```

Se actualiza desde fixtures ya cargados por API/cache, `data/worldcup_matches.csv` y `data/cache/worldcup_matches_latest.csv`. No hace llamadas API para mantener el catalogo; solo usa datos locales salvo que presiones **Actualizar datos desde API**.

Si un equipo del catalogo no existe en `data/team_recent_form.csv`, la app agrega una fila fallback conservadora de baja confianza y antes crea backup en:

```text
data/backups/
```

Esto permite analizar partidos como Switzerland vs Bosnia-Herzegovina aunque falten datos reales de forma reciente. El modelo marca esos datos como estimados, baja la confianza, limita la calibracion y muestra advertencias cuando las lambdas quedan demasiado parecidas por falta de datos.

## Backtesting y calibracion

Cuando analizas un partido, la app guarda la prediccion en:

```text
data/model_predictions.csv
```

Si el partido ya esta jugado, tambien guarda la evaluacion en:

```text
data/model_results.csv
```

La evaluacion incluye acierto de ganador/empate, acierto de marcador exacto Top 1/3/5, Brier Score, Log Loss, ranking del marcador real y probabilidad asignada al marcador real.

La calibracion 1X2 se guarda en:

```text
data/model_calibration.json
```

Si hay menos de 20 partidos evaluados reales, la calibracion fuerte queda desactivada por muestra pequena. Las probabilidades siempre se normalizan para sumar 1. Los resultados `mock` no se usan para calibrar.

## Por que salia siempre 1-1

El patron repetido `1-1`, `2-1`, `1-2`, `1-0`, `0-1`, `2-2` aparecia sobre todo cuando:

- Los dos equipos usaban fallback y sus lambdas quedaban casi iguales.
- Faltaba forma reciente real en `data/team_recent_form.csv`.
- Un `DIXON_COLES_RHO` negativo podia aumentar demasiado marcadores bajos como 1-1.
- La calibracion podia activarse con resultados mock o de baja calidad.
- Faltaban `strength_score`, H2H, lesiones y datos de jugadores.

Ahora el modelo penaliza esos casos con baja confianza, usa `team_strength.csv` para separar equipos cuando hay Elo/ranking, desactiva rho agresivo con poca muestra y no calibra con resultados mock.

## Como mejorar datos reales

1. Configura `API_FOOTBALL_KEY` en `.env`.
2. Usa `USE_MOCK_DATA=false` si quieres evitar datos simulados.
3. Abre la app con `python -m streamlit run app.py`.
4. Presiona **Actualizar fixtures desde API-Football**.
5. Revisa o reconstruye el catalogo local de equipos.
6. Presiona **Actualizar datos reales de equipos faltantes** o **Actualizar datos reales de estos equipos desde API-Football**.
7. Revisa `data/team_recent_form.csv`.
8. Analiza partidos y confirma en la UI la fuente de forma, H2H, lesiones, jugadores, strength score y calidad de datos.

## Como mejorar la precision del modelo

- **Backtesting:** acumula predicciones y resultados para medir si el modelo mejora.
- **Brier Score:** evalua la calidad de probabilidades 1X2; menor es mejor.
- **Log Loss:** penaliza predicciones confiadas que fallan.
- **Dixon-Coles:** corrige marcadores bajos como 0-0, 1-0, 0-1 y 1-1.
- **Calibracion:** ajusta sesgos acumulados de local, empate y visitante solo con muestra suficiente.
- **Recency weighting:** usa `data/team_match_history.csv` si existe y da mas peso a partidos recientes.
- **Shrinkage:** mezcla lambdas hacia el promedio global para evitar sobreajustar muestras pequenas.
- **Fuerza del rival:** usa `data/team_strength.csv` con ajuste limitado a +/-15%.
- **Monte Carlo:** compara probabilidades directas contra simulacion.

## Por que no se debe calibrar con un solo partido

Ghana vs Panama sirve como ejemplo de diagnostico: si Ghana gana 1-0 y el marcador real aparece en Top 5, el modelo puede estar razonablemente cerca aunque el marcador Top 1 haya sido 1-1. Eso no basta para concluir que hay sesgo estructural de empate. Para ajustar factores de calibracion se necesita una muestra acumulada; por eso la app no aplica calibracion fuerte con menos de 20 partidos evaluados.

## CSV importantes

### `data/team_recent_form.csv`

Debe tener:

```csv
team,matches,goals_for,goals_against,wins,draws,losses,xg_for,xg_against,home_gf,home_ga,away_gf,away_ga,clean_sheets,failed_to_score,over_2_5_rate,both_teams_score_rate,source,data_quality,last_updated,is_estimated,confidence
```

### `data/player_form.csv`

```csv
player,team,goals_last10,shots_last10,goals_min_1_30,starts_last10,penalty_taker
```

### `data/injuries.csv`

```csv
player,team,status,impact,source_url
```

### `data/h2h.csv`

```csv
date,home,away,home_goals,away_goals,competition
```

### `data/web_facts.csv`

```csv
match_id,home,away,fact_type,value,numeric_value,team,player,source_title,source_url,source_domain,confidence,extracted_at
```

### `data/stadiums.csv`

```csv
venue,city,country,lat,lon,altitude_m
```

## Importante

El modelo no garantiza marcadores exactos. Entrega probabilidades estimadas con base en los datos cargados. Si faltan datos, baja la confianza o muestra N/D.
