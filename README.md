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

Para gastar una llamada real a API-Football o football-data.org, usa el boton **Actualizar datos desde API** en la barra lateral de Streamlit. Si la llamada funciona, la respuesta se guarda en cache y las siguientes recargas leen ese archivo local.

La app usa estas fuentes, en este orden:

1. API en vivo, solo si presionas **Actualizar datos desde API**.
2. Cache local en `data/cache/`.
3. CSV local `data/worldcup_matches.csv`.
4. Mock data si `USE_MOCK_DATA=true`.

Archivos de cache:

```text
data/cache/api_football_fixtures_2026.json
data/cache/football_data_matches_2026.json
data/cache/worldcup_matches_latest.csv
data/cache/cache_metadata.json
```

`cache_metadata.json` guarda proveedor, endpoint, parametros sin credenciales, fecha de cache, vencimiento, archivo fuente, requests ahorrados y ultimo error.

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

Esto permite analizar partidos como Switzerland vs Bosnia-Herzegovina aunque falten datos reales de forma reciente. El modelo marca esos datos como estimados y baja la confianza.

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

Si hay menos de 20 partidos evaluados, la calibracion fuerte queda desactivada por muestra pequena. Las probabilidades siempre se normalizan para sumar 1.

## Como mejorar la precision del modelo

- **Backtesting:** acumula predicciones y resultados para medir si el modelo mejora.
- **Brier Score:** evalua la calidad de probabilidades 1X2; menor es mejor.
- **Log Loss:** penaliza predicciones confiadas que fallan.
- **Dixon-Coles:** corrige marcadores bajos como 0-0, 1-0, 0-1 y 1-1.
- **Calibracion:** ajusta sesgos acumulados de local, empate y visitante solo con muestra suficiente.
- **Recency weighting:** usa `data/team_match_history.csv` si existe y da mas peso a partidos recientes.
- **Shrinkage:** mezcla lambdas hacia el promedio global para evitar sobreajustar muestras pequenas.
- **Fuerza del rival:** usa `data/team_strength.csv` con ajuste limitado a +/-12%.
- **Monte Carlo:** compara probabilidades directas contra simulacion.

## Por que no se debe calibrar con un solo partido

Ghana vs Panama sirve como ejemplo de diagnostico: si Ghana gana 1-0 y el marcador real aparece en Top 5, el modelo puede estar razonablemente cerca aunque el marcador Top 1 haya sido 1-1. Eso no basta para concluir que hay sesgo estructural de empate. Para ajustar factores de calibracion se necesita una muestra acumulada; por eso la app no aplica calibracion fuerte con menos de 20 partidos evaluados.

## CSV importantes

### `data/team_recent_form.csv`

Debe tener:

```csv
team,matches,goals_for,goals_against,wins,draws,losses,xg_for,xg_against,home_gf,home_ga,away_gf,away_ga
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

### `data/stadiums.csv`

```csv
venue,city,country,lat,lon,altitude_m
```

## Importante

El modelo no garantiza marcadores exactos. Entrega probabilidades estimadas con base en los datos cargados. Si faltan datos, baja la confianza o muestra N/D.
