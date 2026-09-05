"""
EduPilot AI Faculty Assistant
Module: config.py
Version: 4.3.0
Author: EduPilot Team
Purpose: Central typed configuration loader. Reads settings from environment
         variables / .env file via python-dotenv with sensible defaults and
         validation. Safe to import when GROQ_API_KEY is absent; raises
         ConfigurationError only when the key is explicitly required at runtime.

         v4.3 additions:
           - runs_dir: Path for pipeline run history JSON files.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (directory containing this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when a required configuration value is missing or invalid."""


# ---------------------------------------------------------------------------
# Typed configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EduPilotConfig:
    """Immutable runtime configuration for EduPilot.

    All values are loaded once at module import time from environment
    variables (or .env). Sensitive fields (e.g. groq_api_key) are
    Optional[str]; callers must invoke ``require_groq_api_key()`` before
    making LLM requests so that the error surface is clear.
    """

    # LLM settings
    groq_api_key: Optional[str]
    groq_model_name: str
    groq_max_tokens: int

    # Embedding / retrieval
    embedding_model_name: str
    vectorstore_path: Path
    knowledge_dir: Path
    diagrams_dir: Path

    # Observability
    log_level: str

    # Run history
    runs_dir: Path

    # Batching knobs — control when and how generation is split into batches
    # to avoid hitting the Groq free-tier 6000-token output budget limit.
    # batch_threshold: hard cap — always batch above this question count
    #   regardless of per-type token estimate.  Set low (e.g. 5) to force
    #   batching in tests; keep at 30 to rely purely on the estimate.
    batch_threshold: int
    # batch_size: max questions per LLM call in a batched generation.  Actual
    #   batch sizes are balanced so no tiny tail batch is produced
    #   (e.g. 25 questions → 9, 8, 8 rather than 10, 10, 5).
    batch_size: int
    # batch_inter_call_delay_s: length (seconds) of the TPM cooldown window
    #   measured from the START of the previous batch's LLM call.  Groq's
    #   free-tier rolling 60 s window counts the requested max_tokens up
    #   front, so once 62 s have passed since the previous call began, its
    #   budget has expired.  The time the call itself takes (often 20-40 s)
    #   counts toward the window, so the actual sleep is only the remainder.
    batch_inter_call_delay_s: float

    def require_groq_api_key(self) -> str:
        """Return the Groq API key or raise ConfigurationError if absent.

        Call this lazily — only when an LLM request is about to be made —
        so that the app starts cleanly even without the key set.

        Returns:
            str: The validated GROQ_API_KEY.

        Raises:
            ConfigurationError: When GROQ_API_KEY is missing or empty.
        """
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. "
                "Add it as a Replit secret or to your .env file."
            )
        return self.groq_api_key


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a value from the environment, returning *default* if absent.

    Args:
        key: Environment variable name.
        default: Fallback value when the variable is not set.

    Returns:
        The string value or *default*.
    """
    value = os.environ.get(key, default)
    return value if value != "" else default


def _resolve_path(raw: str) -> Path:
    """Resolve *raw* relative to the project root.

    Args:
        raw: A path string (absolute or relative).

    Returns:
        Resolved absolute Path.
    """
    p = Path(raw)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _build_config() -> EduPilotConfig:
    """Construct and return an :class:`EduPilotConfig` from the environment.

    Returns:
        A populated, frozen EduPilotConfig instance.
    """
    return EduPilotConfig(
        groq_api_key=_get_env("GROQ_API_KEY"),
        groq_model_name=_get_env("GROQ_MODEL_NAME", "openai/gpt-oss-120b"),
        # Output-token budget per LLM call. Groq's free tier counts the
        # REQUESTED max_tokens against an 8000 tokens-per-minute limit, so
        # the default must leave headroom for the prompt (~1-2k tokens).
        groq_max_tokens=int(_get_env("GROQ_MAX_TOKENS", "6000") or 6000),
        embedding_model_name=_get_env(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
        ),
        vectorstore_path=_resolve_path(
            _get_env("VECTORSTORE_PATH", "vectorstore") or "vectorstore"
        ),
        knowledge_dir=_resolve_path(
            _get_env("KNOWLEDGE_DIR", "knowledge") or "knowledge"
        ),
        # Extracted content diagrams (figures, circuits, network diagrams —
        # decorative template graphics are filtered out during extraction,
        # see rag.py's extract_content_diagrams()). Defaults alongside
        # vectorstore/knowledge so it lives on the same persistent disk
        # when VECTORSTORE_PATH/KNOWLEDGE_DIR are pointed at one (e.g.
        # /var/data/...) — set DIAGRAMS_DIR explicitly to override.
        diagrams_dir=_resolve_path(
            _get_env("DIAGRAMS_DIR", "diagrams") or "diagrams"
        ),
        log_level=(_get_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        runs_dir=_resolve_path(
            _get_env("RUNS_DIR", "runs") or "runs"
        ),
        batch_threshold=int(_get_env("BATCH_THRESHOLD", "30") or 30),
        batch_size=int(_get_env("BATCH_SIZE", "10") or 10),
        batch_inter_call_delay_s=float(
            _get_env("BATCH_INTER_CALL_DELAY_S", "62") or 62
        ),
    )


# Singleton — imported by all modules
config: EduPilotConfig = _build_config()
