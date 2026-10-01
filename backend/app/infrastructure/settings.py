from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml import config as ml_config

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    flood_model_path: Path = ml_config.MODEL_PATH
    flood_model_card_path: Path = ml_config.MODEL_CARD_PATH

    # Browser origins allowed to call the API (the Vite dev server by default).
    # In .env as JSON: CORS_ORIGINS=["http://localhost:5173"]
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # OpenAI interpretation is optional: without a key and model, /explain returns the
    # ML prediction with interpretation_status "disabled".
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_language: str = "English"
    openai_timeout_seconds: float = 30.0

    @property
    def interpretation_enabled(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip() and self.openai_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
