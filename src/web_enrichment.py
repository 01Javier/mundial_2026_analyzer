from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from .config import settings

WEB_CACHE_DIR = Path("data/cache/web_search")
WEB_FACTS_PATH = Path("data/web_facts.csv")
WEB_FACT_COLUMNS = [
    "match_id",
    "home",
    "away",
    "fact_type",
    "value",
    "numeric_value",
    "team",
    "player",
    "source_title",
    "source_url",
    "source_domain",
    "confidence",
    "extracted_at",
]
HIGH_TRUST_DOMAINS = {
    "fifa.com",
    "espn.com",
    "skysports.com",
    "theanalyst.com",
    "fotmob.com",
    "sofascore.com",
    "flashscore.com",
}
MEDIUM_TRUST_DOMAINS = {
    "lineups.com",
    "sportsmole.co.uk",
    "squawka.com",
    "wincomparator.com",
    "oddschecker.com",
}
ALLOWED_FACT_TYPES = {
    "xg_home",
    "xg_away",
    "odds_home",
    "odds_draw",
    "odds_away",
    "implied_prob_home",
    "implied_prob_draw",
    "implied_prob_away",
    "injury",
    "doubt",
    "lineup",
    "h2h",
    "recent_form",
    "key_player",
    "tactical_note",
    "weather_note",
    "market_consensus",
    "source_note",
}


@dataclass
class WebFact:
    match_id: str
    home: str
    away: str
    fact_type: str
    value: str
    numeric_value: float | None
    team: str | None
    player: str | None
    source_title: str
    source_url: str
    source_domain: str
    confidence: float
    extracted_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_domains(value: str) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def _domain_matches(domain: str, candidates: set[str]) -> bool:
    domain = str(domain or "").lower()
    return any(domain == item or domain.endswith(f".{item}") for item in candidates)


def source_domain(url: str) -> str:
    parsed = urlparse(str(url or ""))
    domain = parsed.netloc.lower().replace("www.", "")
    return domain.split(":")[0]


def source_reliability(domain: str) -> float:
    domain = str(domain or "").lower().replace("www.", "")
    blocked = _split_domains(settings.web_blocked_domains)
    allowed = _split_domains(settings.web_allowed_domains)
    if not domain or _domain_matches(domain, blocked):
        return 0.0
    if _domain_matches(domain, HIGH_TRUST_DOMAINS):
        return 0.90
    if _domain_matches(domain, MEDIUM_TRUST_DOMAINS):
        return 0.70
    if allowed and _domain_matches(domain, allowed):
        return 0.60
    return 0.35


def ensure_web_storage():
    WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WEB_FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not WEB_FACTS_PATH.exists():
        pd.DataFrame(columns=WEB_FACT_COLUMNS).to_csv(WEB_FACTS_PATH, index=False)


def _safe_match_id(match: dict) -> str:
    match_id = str(match.get("match_id") or f"{match.get('home', '')}_{match.get('away', '')}_{match.get('date_utc', '')}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", match_id).strip("_") or "match"


def _query_hash(query: str) -> str:
    return sha256(query.encode("utf-8")).hexdigest()[:16]


def _cache_file(match: dict, query: str) -> Path:
    ensure_web_storage()
    return WEB_CACHE_DIR / f"{_safe_match_id(match)}_{_query_hash(query)}.json"


def web_cache_is_fresh(path: Path, hours: int) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
    return age_hours <= hours


def build_match_search_queries(match: dict) -> list[str]:
    home = str(match.get("home", "")).strip()
    away = str(match.get("away", "")).strip()
    date_text = str(match.get("date_utc", "")).split("T")[0]
    base = f"{home} vs {away} World Cup 2026"
    compact = f"{home} {away}"
    return [
        f"{base} preview xG injuries predicted lineups odds",
        f"{compact} H2H recent form {date_text} World Cup 2026",
        f"{compact} predicted lineups injuries World Cup 2026",
        f"{compact} betting odds implied probability World Cup 2026",
        f"{compact} expected goals xG preview World Cup 2026",
    ][: max(1, settings.max_web_searches_per_run)]


def _normalize_result(item: dict, query: str, provider: str, from_cache: bool) -> dict:
    url = item.get("url") or item.get("link") or item.get("source_url") or ""
    title = item.get("title") or item.get("name") or ""
    content = item.get("content") or item.get("snippet") or item.get("description") or item.get("body") or ""
    domain = source_domain(url)
    return {
        "query": query,
        "provider": provider,
        "title": str(title or "").strip(),
        "url": str(url or "").strip(),
        "content": str(content or "").strip(),
        "source_domain": domain,
        "reliability": source_reliability(domain),
        "from_cache": from_cache,
    }


def search_web_tavily(query: str, max_results: int = 5) -> list[dict]:
    if not settings.tavily_api_key:
        raise RuntimeError("Falta TAVILY_API_KEY en .env")
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError("Instala tavily-python para usar Tavily") from exc

    client = TavilyClient(api_key=settings.tavily_api_key)
    payload = client.search(
        query=query,
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )
    return payload.get("results", [])


def search_web_serpapi(query: str, max_results: int = 5) -> list[dict]:
    if not settings.serpapi_api_key:
        raise RuntimeError("Falta SERPAPI_API_KEY en .env")
    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "api_key": settings.serpapi_api_key,
            "num": max_results,
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("organic_results", [])[:max_results]


def _read_query_cache(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_query_cache(path: Path, query: str, provider: str, results: list[dict]):
    payload = {
        "query": query,
        "provider": provider,
        "fetched_at": _now_iso(),
        "results": results,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def search_match_web(match: dict, force_refresh: bool = False) -> list[dict]:
    """
    Usa Tavily o SerpAPI segun WEB_SEARCH_PROVIDER.
    Usa cache fresco primero y guarda raw results en data/cache/web_search/.
    """
    if not settings.enable_web_enrichment:
        return []

    provider = settings.web_search_provider.lower().strip()
    queries = build_match_search_queries(match)
    all_results = []
    searches_used = 0

    for query in queries:
        path = _cache_file(match, query)
        cached = _read_query_cache(path) if web_cache_is_fresh(path, settings.web_search_cache_hours) else None
        if cached:
            provider_name = cached.get("provider", provider)
            all_results.extend(
                _normalize_result(item, query, provider_name, from_cache=True)
                for item in cached.get("results", [])
            )
            continue

        if not force_refresh and not settings.allow_web_search_on_page_load:
            continue

        if searches_used >= settings.max_web_searches_per_run:
            break

        if provider == "serpapi":
            raw_results = search_web_serpapi(query, max_results=5)
        else:
            raw_results = search_web_tavily(query, max_results=5)
            provider = "tavily"

        searches_used += 1
        _write_query_cache(path, query, provider, raw_results)
        all_results.extend(_normalize_result(item, query, provider, from_cache=False) for item in raw_results)

    return [
        result
        for result in all_results
        if result.get("url") and result.get("reliability", 0) > 0
    ]


def _text_for_result(result: dict) -> str:
    return " ".join(
        str(result.get(key, "") or "")
        for key in ["title", "content"]
    ).strip()


def _contains_team(text: str, team: str) -> bool:
    if not team:
        return False
    team_norm = re.escape(team.lower())
    return re.search(team_norm, text.lower()) is not None


def _short_value(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def _make_fact(match: dict, result: dict, fact_type: str, value: str, numeric_value=None, team=None, player=None) -> WebFact | None:
    if fact_type not in ALLOWED_FACT_TYPES:
        return None
    url = result.get("url")
    domain = result.get("source_domain") or source_domain(url)
    confidence = min(1.0, max(0.0, float(result.get("reliability", source_reliability(domain)) or 0)))
    if not url or confidence <= 0:
        return None
    return WebFact(
        match_id=_safe_match_id(match),
        home=str(match.get("home", "")),
        away=str(match.get("away", "")),
        fact_type=fact_type,
        value=_short_value(value),
        numeric_value=None if numeric_value is None else float(numeric_value),
        team=team,
        player=player,
        source_title=str(result.get("title", ""))[:180],
        source_url=url,
        source_domain=domain,
        confidence=confidence,
        extracted_at=_now_iso(),
    )


def _extract_xg_facts(match: dict, result: dict, text: str) -> list[WebFact]:
    facts = []
    for team, fact_type in [(match.get("home"), "xg_home"), (match.get("away"), "xg_away")]:
        if not team:
            continue
        patterns = [
            rf"{re.escape(str(team))}\s*(?:xG|expected goals)\s*[:=]?\s*([0-4](?:\.\d{{1,2}})?)",
            rf"(?:xG|expected goals)\s*[:=]?\s*([0-4](?:\.\d{{1,2}})?)\s*(?:for\s+)?{re.escape(str(team))}",
        ]
        for pattern in patterns:
            match_obj = re.search(pattern, text, flags=re.IGNORECASE)
            if match_obj:
                value = float(match_obj.group(1))
                if 0 <= value <= 4.5:
                    fact = _make_fact(match, result, fact_type, f"{team} xG {value}", value, team=team)
                    if fact:
                        facts.append(fact)
                    break
    return facts


def _extract_market_facts(match: dict, result: dict, text: str) -> list[WebFact]:
    facts = []
    lower = text.lower()
    if not any(word in lower for word in ["odds", "betting", "implied probability", "probability", "cuota"]):
        return facts

    decimal_numbers = [float(item) for item in re.findall(r"(?<!\d)([1-9]\d?\.\d{1,2})(?!\d)", text)]
    if len(decimal_numbers) >= 3 and source_reliability(result.get("source_domain")) >= 0.60:
        odds = decimal_numbers[:3]
        for fact_type, team, value in [
            ("odds_home", match.get("home"), odds[0]),
            ("odds_draw", "Empate", odds[1]),
            ("odds_away", match.get("away"), odds[2]),
        ]:
            fact = _make_fact(match, result, fact_type, f"{team} decimal odds {value}", value, team=None if team == "Empate" else team)
            if fact:
                facts.append(fact)
        inverse = [1 / value for value in odds if value > 0]
        total = sum(inverse)
        if total > 0:
            for fact_type, team, prob in [
                ("implied_prob_home", match.get("home"), inverse[0] / total),
                ("implied_prob_draw", "Empate", inverse[1] / total),
                ("implied_prob_away", match.get("away"), inverse[2] / total),
            ]:
                fact = _make_fact(match, result, fact_type, f"{team} implied probability {prob:.3f}", prob, team=None if team == "Empate" else team)
                if fact:
                    facts.append(fact)
    else:
        fact = _make_fact(match, result, "market_consensus", text)
        if fact:
            facts.append(fact)
    return facts


def extract_web_facts(match: dict, search_results: list[dict]) -> list[WebFact]:
    """
    Extrae hechos utiles desde titulos/snippets/contenido disponible.
    No inventa datos: si no hay texto claro, no crea fact.
    """
    facts = []
    for result in search_results:
        url = result.get("url")
        domain = result.get("source_domain") or source_domain(url)
        reliability = source_reliability(domain)
        if not url or reliability <= 0:
            continue

        text = _text_for_result(result)
        if not text:
            continue
        lower = text.lower()
        home = str(match.get("home", ""))
        away = str(match.get("away", ""))

        facts.extend(_extract_xg_facts(match, result, text))
        facts.extend(_extract_market_facts(match, result, text))

        team = home if _contains_team(text, home) else away if _contains_team(text, away) else None
        if any(word in lower for word in ["injury", "injured", "out injured", "lesion", "lesión", "ruled out"]):
            fact = _make_fact(match, result, "injury", text, team=team)
            if fact:
                facts.append(fact)
        elif any(word in lower for word in ["doubt", "questionable", "doubtful", "fitness test"]):
            fact = _make_fact(match, result, "doubt", text, team=team)
            if fact:
                facts.append(fact)

        if any(word in lower for word in ["predicted lineup", "lineups", "starting xi", "probable lineup", "alineacion", "alineación"]):
            fact = _make_fact(match, result, "lineup", text, team=team)
            if fact:
                facts.append(fact)
        if any(word in lower for word in ["h2h", "head-to-head", "head to head"]):
            fact = _make_fact(match, result, "h2h", text)
            if fact:
                facts.append(fact)
        if any(word in lower for word in ["recent form", "form guide", "last five", "last 5"]):
            fact = _make_fact(match, result, "recent_form", text, team=team)
            if fact:
                facts.append(fact)
        if any(word in lower for word in ["key player", "ones to watch", "player to watch"]):
            fact = _make_fact(match, result, "key_player", text, team=team)
            if fact:
                facts.append(fact)
        if any(word in lower for word in ["tactical", "formation", "pressing", "low block"]):
            fact = _make_fact(match, result, "tactical_note", text, team=team)
            if fact:
                facts.append(fact)
        if any(word in lower for word in ["weather", "temperature", "rain", "humidity"]):
            fact = _make_fact(match, result, "weather_note", text)
            if fact:
                facts.append(fact)
        if any(word in lower for word in ["preview", "prediction"]) and not any(f.fact_type == "source_note" and f.source_url == url for f in facts):
            fact = _make_fact(match, result, "source_note", text)
            if fact:
                facts.append(fact)

    deduped = {}
    for fact in facts:
        key = (fact.match_id, fact.fact_type, fact.value, fact.source_url)
        deduped[key] = fact
    return list(deduped.values())


def save_web_facts(facts: list[WebFact]):
    ensure_web_storage()
    if not facts:
        return WEB_FACTS_PATH

    existing = pd.read_csv(WEB_FACTS_PATH) if WEB_FACTS_PATH.exists() else pd.DataFrame(columns=WEB_FACT_COLUMNS)
    for col in WEB_FACT_COLUMNS:
        if col not in existing.columns:
            existing[col] = None
    new_df = pd.DataFrame([asdict(fact) for fact in facts])
    combined = pd.concat([existing[WEB_FACT_COLUMNS], new_df[WEB_FACT_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["match_id", "fact_type", "value", "source_url"], keep="last")
    combined.to_csv(WEB_FACTS_PATH, index=False)
    return WEB_FACTS_PATH


def load_web_facts_for_match(match_id: str) -> pd.DataFrame:
    ensure_web_storage()
    df = pd.read_csv(WEB_FACTS_PATH)
    if df.empty or "match_id" not in df.columns:
        return pd.DataFrame(columns=WEB_FACT_COLUMNS)
    result = df[df["match_id"].astype(str) == str(match_id)].copy()
    for col in WEB_FACT_COLUMNS:
        if col not in result.columns:
            result[col] = None
    return result[WEB_FACT_COLUMNS]


def _probability_from_facts(df: pd.DataFrame, home: str, away: str) -> dict:
    probs = {}
    prob_map = {
        "implied_prob_home": home,
        "implied_prob_draw": "Empate",
        "implied_prob_away": away,
    }
    for fact_type, key in prob_map.items():
        values = pd.to_numeric(df.loc[df["fact_type"] == fact_type, "numeric_value"], errors="coerce").dropna()
        if not values.empty:
            probs[key] = float(values.mean())
    if len(probs) == 3:
        total = sum(probs.values())
        return {key: value / total for key, value in probs.items()} if total > 0 else {}

    odds_map = {
        "odds_home": home,
        "odds_draw": "Empate",
        "odds_away": away,
    }
    inverses = {}
    for fact_type, key in odds_map.items():
        values = pd.to_numeric(df.loc[df["fact_type"] == fact_type, "numeric_value"], errors="coerce").dropna()
        if not values.empty:
            odds = float(values.mean())
            if odds > 1:
                inverses[key] = 1 / odds
    if len(inverses) == 3:
        total = sum(inverses.values())
        return {key: value / total for key, value in inverses.items()} if total > 0 else {}
    return {}


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "alta"
    if value >= 0.55:
        return "media"
    return "baja"


def summarize_web_enrichment(match: dict) -> dict:
    """
    Devuelve resumen usable por el modelo.
    """
    match_id = _safe_match_id(match)
    home = str(match.get("home", ""))
    away = str(match.get("away", ""))
    df = load_web_facts_for_match(match_id)
    if df.empty:
        return {
            "xg_home": None,
            "xg_away": None,
            "odds_home": None,
            "odds_draw": None,
            "odds_away": None,
            "market_probs": {},
            "h2h_summary": None,
            "injuries_home": [],
            "injuries_away": [],
            "lineups_notes": [],
            "recent_form_notes": [],
            "source_count": 0,
            "confidence": 0.0,
            "confidence_label": "baja",
            "facts_df": df,
            "last_search": None,
        }

    df = df[df["source_url"].notna() & (df["source_url"].astype(str).str.len() > 0)].copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df = df[df["confidence"] > 0]
    source_count = int(df["source_url"].nunique())
    avg_confidence = float(df["confidence"].mean()) if not df.empty else 0.0

    def numeric_mean(fact_type):
        values = pd.to_numeric(df.loc[df["fact_type"] == fact_type, "numeric_value"], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def notes(fact_type, team=None):
        subset = df[df["fact_type"].isin(fact_type if isinstance(fact_type, list) else [fact_type])]
        if team:
            subset = subset[subset["team"].fillna("").astype(str).str.lower() == team.lower()]
        return subset.sort_values("confidence", ascending=False).head(5)["value"].astype(str).tolist()

    last_search = None
    extracted = pd.to_datetime(df["extracted_at"], errors="coerce", utc=True)
    if extracted.notna().any():
        last_search = extracted.max().isoformat()

    return {
        "xg_home": numeric_mean("xg_home"),
        "xg_away": numeric_mean("xg_away"),
        "odds_home": numeric_mean("odds_home"),
        "odds_draw": numeric_mean("odds_draw"),
        "odds_away": numeric_mean("odds_away"),
        "market_probs": _probability_from_facts(df, home, away),
        "h2h_summary": notes("h2h")[:1],
        "injuries_home": notes(["injury", "doubt"], team=home),
        "injuries_away": notes(["injury", "doubt"], team=away),
        "lineups_notes": notes("lineup"),
        "recent_form_notes": notes("recent_form"),
        "source_count": source_count,
        "confidence": avg_confidence,
        "confidence_label": _confidence_label(avg_confidence),
        "facts_df": df,
        "last_search": last_search,
    }
