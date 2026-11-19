from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

# ok        -> complete answer (stop_reason end_turn / stop_sequence)
# truncated -> the model hit max_tokens: `text` is a PARTIAL answer
# refused   -> the model declined (stop_reason "refusal"): `text` may be empty
# error     -> permanent request error, nothing was generated
# unavailable -> primary retries and fallback exhausted, nothing was generated
Status = Literal["ok", "truncated", "refused", "error", "unavailable"]


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
    stop_reason: str | None = None  # raw API stop_reason, for traceability
    reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "LLMResult":
        """Retries (primary + fallback) exhausted: service unavailable."""
        return cls(text="", model="", status="unavailable", reason=reason)

    @classmethod
    def error(cls, reason: str) -> "LLMResult":
        """Permanent error (non-retryable): the request must be fixed."""
        return cls(text="", model="", status="error", reason=reason)


class LLMClient(Protocol):
    """What callers depend on: anything with this shape is a client.

    Structural (duck) typing checked by the type checker: ProductionLLMClient
    and the offline/test fakes satisfy it without inheriting from anything.
    """

    model: str
    fallback_model: str

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        output_schema: dict | None = None,
    ) -> LLMResult: ...
