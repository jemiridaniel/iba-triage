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
    # Every live call sends a max_tokens so its worst-case cost is bounded.
    llm_default_max_tokens: int = 2048
    # Per-step reasoning. Off = chat_template_kwargs {"enable_thinking": false}, the switch
    # Token Factory honours for Nemotron (a "/no_think" system prompt is ignored).
    reasoning_intake: bool = False
    reasoning_outbreak: bool = False
    reasoning_reason: bool = True
    reasoning_compose: bool = False
    # Completion budgets. Reasoning tokens count toward max_tokens, and a truncated reply
    # is rejected, so reasoning steps need a much larger budget.
    max_tokens_reasoning_off: int = 1024
    max_tokens_reasoning_on: int = 8192
    # Hard cap on cumulative live spend, tracked in CACHE_DIR/spend.json. Raise it deliberately.
    max_spend_usd: float = 0.50
    log_level: str = "INFO"

    # Guideline index built by rag/ingest.py (or scripts/build_fake_index.py for dev).
    index_dir: Path = Path("data/index")
    retrieve_top_k: int = 6
    # Tavily. Without a key the outbreak step reports "outbreak data unavailable".
    # OUTBREAK_MOCK_FILE replays canned search results instead (tests, demos before credits).
    outbreak_mock_file: Path | None = None

    @property
    def spend_file(self) -> Path:
        return self.cache_dir / "spend.json"

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
