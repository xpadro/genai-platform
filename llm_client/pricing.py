"""Claude API (first-party) prices, in USD per million tokens (MTok).

Source: https://platform.claude.com/docs/en/about-claude/pricing
Verified: 2026-06-30. Prices change — check the source before relying on these
numbers for real cost decisions.
"""
from __future__ import annotations

PRICING_DATE = "2026-06-30"

# USD per million tokens, by token type.
PRICES = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write_5m": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cache_write_5m": 1.25, "cache_read": 0.10},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_write_5m": 6.25, "cache_read": 0.50},
}

_MTOK = 1_000_000


def cost_usd(result) -> float:
    """Cost of a call from its `usage`.

    In the API, `input_tokens` excludes cache tokens (write/read are reported
    separately), so summing the four categories does not double-count.
    Unknown model -> 0.0 (we don't invent prices).
    """
    p = PRICES.get(result.model)
    if p is None:
        return 0.0
    return (
        result.input_tokens / _MTOK * p["input"]
        + result.output_tokens / _MTOK * p["output"]
        + result.cache_read / _MTOK * p["cache_read"]
        + result.cache_write / _MTOK * p["cache_write_5m"]
    )
