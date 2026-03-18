"""Centralized configuration using Pydantic settings and YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from the config/ directory."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Database
    database_url: str = Field(default="sqlite:///./btc_alpha.db")

    # Coinalyze
    coinalyze_api_key: str = Field(default="")

    # Hyperliquid
    hyperliquid_api_key: str = Field(default="")
    hyperliquid_api_secret: str = Field(default="")
    hyperliquid_wallet_address: str = Field(default="")
    hyperliquid_base_url: str = Field(default="https://api.hyperliquid-testnet.xyz")

    # Targon (CPU compute + inference)
    targon_api_key: str = Field(default="")
    targon_base_url: str = Field(default="https://api.targon.com/v1")
    targon_model: str = Field(default="meta-llama/Meta-Llama-3.1-70B-Instruct")

    # Lium (GPU rental for training)
    lium_api_key: str = Field(default="")

    # Chutes.ai (serverless agent inference)
    chutes_api_key: str = Field(default="")
    chutes_base_url: str = Field(default="https://llm.chutes.ai/v1")
    chutes_model: str = Field(default="deepseek-ai/DeepSeek-V3-0324")

    # Feature flags
    live_trading_enabled: bool = Field(default=False)
    paper_mode: bool = Field(default=True)
    flatten_on_breach: bool = Field(default=True)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Risk
    daily_loss_limit_pct: float = Field(default=0.05)
    eod_trailing_loss_limit_pct: float = Field(default=0.05)

    # Agent
    agent_delay_seconds: int = Field(default=3600)
    agent_equity: float = Field(default=100000.0)

    # Monitoring
    prometheus_port: int = Field(default=9090)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton settings accessor."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
