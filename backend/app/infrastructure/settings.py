"""Application settings, read from environment variables or backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ml import config as ml_config

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    flood_model_path: Path = ml_config.MODEL_PATH
    flood_model_card_path: Path = ml_config.MODEL_CARD_PATH


@lru_cache
def get_settings() -> Settings:
    return Settings()
