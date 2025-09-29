"""Configuration utilities for the Agent orchestration service."""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Settings definition with inline documentation for future maintainers."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "plasma-engine-agent"
    cors_origins: List[str] = ["http://localhost:3000"]
    rube_mcp_server: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_llm_model: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> AgentSettings:
    """Return cached AgentSettings to avoid repeated environment parsing."""

    return AgentSettings()

