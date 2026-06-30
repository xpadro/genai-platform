"""Precios de la Claude API (first-party), en USD por millón de tokens (MTok).

Fuente: https://platform.claude.com/docs/en/about-claude/pricing
Verificado: 2026-06-30. Los precios cambian — revisa la fuente antes de confiar
en estos números para decisiones de coste reales.
"""
from __future__ import annotations

PRICING_DATE = "2026-06-30"

# USD por millón de tokens, por tipo de token.
PRICES = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write_5m": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cache_write_5m": 1.25, "cache_read": 0.10},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_write_5m": 6.25, "cache_read": 0.50},
}

_MTOK = 1_000_000


def cost_usd(result) -> float:
    """Coste de una llamada a partir de su `usage`.

    En la API, `input_tokens` excluye los tokens de cache (write/read se reportan
    aparte), así que sumar las cuatro categorías no produce doble conteo.
    Modelo desconocido -> 0.0 (no inventamos precios).
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
