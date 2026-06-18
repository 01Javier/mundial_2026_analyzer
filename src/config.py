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
    dixon_coles_rho: float = float(os.getenv("DIXON_COLES_RHO", "-0.05"))
    use_monte_carlo: bool = os.getenv("USE_MONTE_CARLO", "true").lower() == "true"
    monte_carlo_sims: int = int(os.getenv("MONTE_CARLO_SIMS", "20000"))

    @property
    def gt_tz(self) -> ZoneInfo:
        return ZoneInfo(self.guatemala_timezone)

settings = Settings()
