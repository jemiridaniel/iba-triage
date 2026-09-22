"""Environment settings. Everything configurable (keys, model IDs, prices) comes from here."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    """USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Nebius Token Factory
    nebius_api_key: SecretStr | None = None
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"

    # Model IDs: never hard-coded, always from env (verify against GET /v1/models).
    model_fast: str = ""
    model_mid: str = ""
    model_reason: str = ""
    model_embed: str = ""

    # Optional per-model prices for cost accounting, e.g. MODEL_PRICES='{"id": {"input": 0.1}}'
    model_prices: dict[str, ModelPrice] = {}

    tavily_api_key: SecretStr | None = None

    app_env: Literal["dev", "prod"] = "dev"
    cache_dir: Path = Path(".cache")
    cache_enabled: bool | None = None  # None -> on in dev, off in prod
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    log_level: str = "INFO"

    @property
    def use_cache(self) -> bool:
        if self.cache_enabled is not None:
            return self.cache_enabled
        return self.app_env == "dev"

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
