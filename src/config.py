import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    api_football_key: str = os.getenv("API_FOOTBALL_KEY", "")
    api_football_base_url: str = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
    api_football_world_cup_league_id: int = int(os.getenv("API_FOOTBALL_WORLD_CUP_LEAGUE_ID", "1"))
    api_football_season: int = int(os.getenv("API_FOOTBALL_SEASON", "2026"))

    football_data_token: str = os.getenv("FOOTBALL_DATA_TOKEN", "")
    football_data_base_url: str = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    football_data_competition: str = os.getenv("FOOTBALL_DATA_COMPETITION", "WC")
    football_data_season: int = int(os.getenv("FOOTBALL_DATA_SEASON", "2026"))

    use_mock_data: bool = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
    guatemala_timezone: str = os.getenv("GUATEMALA_TIMEZONE", "America/Guatemala")
    use_dixon_coles: bool = os.getenv("USE_DIXON_COLES", "true").lower() == "true"
    dixon_coles_rho: float = float(os.getenv("DIXON_COLES_RHO", "0.0"))
    use_monte_carlo: bool = os.getenv("USE_MONTE_CARLO", "true").lower() == "true"
    monte_carlo_sims: int = int(os.getenv("MONTE_CARLO_SIMS", "20000"))
    api_provider_priority: str = os.getenv("API_PROVIDER_PRIORITY", "api_football,football_data,csv,mock")
    allow_api_on_page_load: bool = os.getenv("ALLOW_API_ON_PAGE_LOAD", "false").lower() == "true"
    max_api_requests_per_run: int = int(os.getenv("MAX_API_REQUESTS_PER_RUN", "20"))
    team_form_cache_hours: int = int(os.getenv("TEAM_FORM_CACHE_HOURS", "72"))
    auto_fit_dixon_coles: bool = os.getenv("AUTO_FIT_DIXON_COLES", "true").lower() == "true"
    disable_dc_if_draw_bias_high: bool = os.getenv("DISABLE_DC_IF_DRAW_BIAS_HIGH", "true").lower() == "true"
    use_draw_bias_correction: bool = os.getenv("USE_DRAW_BIAS_CORRECTION", "true").lower() == "true"
    ignore_mock_results_for_calibration: bool = os.getenv("IGNORE_MOCK_RESULTS_FOR_CALIBRATION", "true").lower() == "true"
    web_search_provider: str = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")
    enable_web_enrichment: bool = os.getenv("ENABLE_WEB_ENRICHMENT", "true").lower() == "true"
    allow_web_search_on_page_load: bool = os.getenv("ALLOW_WEB_SEARCH_ON_PAGE_LOAD", "false").lower() == "true"
    web_search_cache_hours: int = int(os.getenv("WEB_SEARCH_CACHE_HOURS", "24"))
    max_web_searches_per_run: int = int(os.getenv("MAX_WEB_SEARCHES_PER_RUN", "5"))
    web_confidence_min_sources: int = int(os.getenv("WEB_CONFIDENCE_MIN_SOURCES", "2"))
    web_allowed_domains: str = os.getenv(
        "WEB_ALLOWED_DOMAINS",
        "fifa.com,espn.com,skysports.com,theanalyst.com,lineups.com,fotmob.com,"
        "sofascore.com,flashscore.com,sportsmole.co.uk,wincomparator.com,oddschecker.com,squawka.com",
    )
    web_blocked_domains: str = os.getenv(
        "WEB_BLOCKED_DOMAINS",
        "reddit.com,facebook.com,twitter.com,x.com,tiktok.com",
    )

    @property
    def gt_tz(self) -> ZoneInfo:
        return ZoneInfo(self.guatemala_timezone)

settings = Settings()
