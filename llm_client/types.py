from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["ok", "unavailable", "error"]


@dataclass
class LLMResult:
    """Structured result of a call to the model.

    We never propagate a raw exception to the workflow: every failure materializes
    here as `status` + `reason`, so downstream can decide what to do.
    """

    text: str
    model: str
    status: Status
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    used_fallback: bool = False
    reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "LLMResult":
        """Retries (primary + fallback) exhausted: service unavailable."""
        return cls(text="", model="", status="unavailable", reason=reason)

    @classmethod
    def error(cls, reason: str) -> "LLMResult":
        """Permanent error (non-retryable): the request must be fixed."""
        return cls(text="", model="", status="error", reason=reason)
