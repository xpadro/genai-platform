import anthropic

from llm_client import ProductionLLMClient
from _fakes import make_connection_error, make_status_error, patch_stream


def test_both_fail_returns_unavailable_not_exception(monkeypatch):
    """Primary and fallback both go down: status=unavailable, no raw exception."""
    client = ProductionLLMClient(model="primary", fallback_model="fallback")

    def handler(model):
        if model == "primary":
            raise make_status_error(anthropic.InternalServerError, 529)
        raise make_connection_error()

    patch_stream(monkeypatch, client, handler)
    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])

    assert res.status == "unavailable"
    assert res.text == ""
    assert res.reason is not None


def test_permanent_error_does_not_try_fallback(monkeypatch):
    """A 400 is neither retried nor triggers fallback: status=error."""
    client = ProductionLLMClient(model="primary", fallback_model="fallback")
    seen = []

    def handler(model):
        seen.append(model)
        raise make_status_error(anthropic.BadRequestError, 400)

    patch_stream(monkeypatch, client, handler)
    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])

    assert res.status == "error"
    assert seen == ["primary"]  # never reaches the fallback
