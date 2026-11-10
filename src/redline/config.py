"""Runtime settings, read from environment variables with offline-safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("REDLINE_DATA_DIR", ROOT / "data"))
PLAYBOOK_PATH = Path(os.getenv("REDLINE_PLAYBOOK", ROOT / "playbook" / "playbook.yaml"))
TRAIN_PATH = DATA_DIR / "cuad" / "train_clauses.jsonl"
GOLD_PATH = DATA_DIR / "gold" / "gold.jsonl"


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    backend: str = field(default_factory=lambda: os.getenv("REDLINE_BACKEND", "offline"))
    # Below this classifier probability the clause type is not trusted: escalate to a human.
    min_type_confidence: float = field(
        default_factory=lambda: _float("REDLINE_MIN_TYPE_CONFIDENCE", 0.45)
    )
    # An uncertain clause is escalated only if some playbook type scores at least this much;
    # otherwise it is treated as boilerplate outside the playbook.
    min_governed_score: float = field(
        default_factory=lambda: _float("REDLINE_MIN_GOVERNED_SCORE", 0.25)
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("REDLINE_LLM_PROVIDER", "openai"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("REDLINE_LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("REDLINE_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("REDLINE_LLM_MODEL", "gpt-4o-mini"))
    azure_api_version: str = field(
        default_factory=lambda: os.getenv("REDLINE_AZURE_API_VERSION", "2024-06-01")
    )


def get_settings() -> Settings:
    return Settings()
