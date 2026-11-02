from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables / ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ChurnSense API"
    environment: str = "development"
    log_level: str = "INFO"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    # Comma-separated list of allowed browser origins, e.g. "https://churnsense.vercel.app".
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Optional regex, handy for Vercel preview URLs: r"https://churnsense-.*\.vercel\.app"
    cors_origin_regex: Optional[str] = None
    # When set, scoring endpoints require an ``X-API-Key`` header with this value.
    api_key: Optional[str] = None
    max_batch_rows: int = 10_000
    max_upload_mb: float = 5.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
