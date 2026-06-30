from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["ok", "unavailable", "error"]


@dataclass
class LLMResult:
    """Resultado estructurado de una llamada al modelo.

    Nunca propagamos una excepción cruda al workflow: todo fallo se materializa
    aquí como `status` + `reason`, para que aguas abajo se pueda decidir qué hacer.
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
        """Retries (primario + fallback) agotados: servicio no disponible."""
        return cls(text="", model="", status="unavailable", reason=reason)

    @classmethod
    def error(cls, reason: str) -> "LLMResult":
        """Error permanente (no reintentable): la request hay que corregirla."""
        return cls(text="", model="", status="error", reason=reason)
