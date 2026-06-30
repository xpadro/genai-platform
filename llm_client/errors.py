"""Clasificación de errores del SDK de Anthropic.

Transitorio = el reintento puede tener éxito (estado del servidor/red).
Permanente  = el reintento daría el mismo error (la request es inválida).
"""
import anthropic

# Reintentables. El SDK ya reintenta internamente con backoff; aquí, tras agotar
# esos retries, decidimos el fallback a otro modelo.
TRANSIENT = (
    anthropic.RateLimitError,        # 429 rate limit
    anthropic.InternalServerError,   # 5xx / overloaded (529)
    anthropic.APIConnectionError,    # fallo de red (incluye APITimeoutError)
)

# No reintentables. Fallar claro y rápido; reintentar solo quema latencia y dinero
# y enmascara el bug real.
PERMANENT = (
    anthropic.BadRequestError,           # 400 request mal formada
    anthropic.AuthenticationError,       # 401 key inválida
    anthropic.PermissionDeniedError,     # 403 sin permiso
    anthropic.NotFoundError,             # 404 recurso/modelo inexistente
    anthropic.UnprocessableEntityError,  # 422 semánticamente inválida
)
