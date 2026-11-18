import dataclasses
import logging
import time

import anthropic

from .errors import PERMANENT, TRANSIENT
from .pricing import cost_usd
from .types import LLMResult, Status

log = logging.getLogger("llm_client")


class ProductionLLMClient:
    """LLM client with streaming, retries, fallback and cost observability.

    Contract: `complete()` never propagates a network/SDK exception upward.
    It always returns an `LLMResult` whose `status` says what happened:
    ok / truncated / refused (the model answered, maybe not fully) or
    error / unavailable (nothing was generated). `status="ok"` only means a
    COMPLETE answer: a reply cut at max_tokens or a refusal is never "ok".
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        fallback_model: str = "claude-haiku-4-5-20251001",
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        # The SDK handles the exponential backoff of retries internally.
        self._c = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.model = model
        self.fallback_model = fallback_model

    def complete(self, system, messages, max_tokens: int = 1024) -> LLMResult:
        try:
            return self._call(self.model, system, messages, max_tokens)
        except PERMANENT as e:
            # Retrying won't help: the request is invalid. Fail clearly.
            log.error("permanent error (no retry): %s", e)
            return LLMResult.error(reason=f"{type(e).__name__}: {e}")
        except TRANSIENT as e:
            # The SDK already retried and exhausted its retries -> we try the fallback.
            log.warning("primary failed after retries: %s -> fallback", e)
            try:
                return self._call(
                    self.fallback_model, system, messages, max_tokens, used_fallback=True
                )
            except Exception as e2:
                # Fallback failed too -> graceful degradation, not a raw exception.
                log.error("fallback failed: %s -> graceful degradation", e2)
                return LLMResult.unavailable(reason=f"{type(e2).__name__}: {e2}")

    def _call(self, model, system, messages, max_tokens, used_fallback=False) -> LLMResult:
        t0 = time.perf_counter()
        chunks: list[str] = []
        with self._c.messages.stream(
            model=model, system=system, messages=messages, max_tokens=max_tokens
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
            final = stream.get_final_message()

        status, reason = _status_from_stop(final, max_tokens)
        u = final.usage
        result = LLMResult(
            text="".join(chunks),  # kept even when truncated: the caller decides
            model=model,
            status=status,
            stop_reason=final.stop_reason,
            reason=reason,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            latency_s=time.perf_counter() - t0,
            used_fallback=used_fallback,
        )
        result.cost_usd = cost_usd(result)
        log.info("trace %s", dataclasses.asdict(result))  # seed for C-E1 (tracing)
        return result


def _status_from_stop(final, max_tokens: int) -> tuple[Status, str | None]:
    """Map the API stop_reason to our status, so "ok" never hides a partial answer.

    Neither case is retried or sent to the fallback here: a truncation is fixed
    by the caller (raise max_tokens or tighten the prompt) and a refusal is a
    policy decision; both are reported so the caller can act.
    """
    if final.stop_reason == "max_tokens":
        return "truncated", f"stop_reason=max_tokens: output cut at max_tokens={max_tokens}"
    if final.stop_reason == "refusal":
        details = final.stop_details
        category = details.category if details else None
        return "refused", f"stop_reason=refusal (category={category})"
    return "ok", None
