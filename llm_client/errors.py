"""Classification of Anthropic SDK errors.

Transient = a retry might succeed (server/network state).
Permanent = a retry would give the same error (the request is invalid).
"""
import anthropic

# Retryable. The SDK already retries internally with backoff; here, once those
# retries are exhausted, we decide to fall back to another model.
TRANSIENT = (
    anthropic.RateLimitError,        # 429 rate limit
    anthropic.InternalServerError,   # 5xx / overloaded (529)
    anthropic.APIConnectionError,    # network failure (includes APITimeoutError)
)

# Non-retryable. Fail clearly and fast; retrying only burns latency and money
# and masks the real bug.
PERMANENT = (
    anthropic.BadRequestError,           # 400 malformed request
    anthropic.AuthenticationError,       # 401 invalid key
    anthropic.PermissionDeniedError,     # 403 not permitted
    anthropic.NotFoundError,             # 404 nonexistent resource/model
    anthropic.UnprocessableEntityError,  # 422 semantically invalid
)
