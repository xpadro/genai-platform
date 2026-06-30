import anthropic

from llm_client import ProductionLLMClient
from _fakes import FakeStream, FakeUsage, make_status_error, patch_stream


def test_transient_on_primary_falls_back_to_ok(monkeypatch):
    """Un error transitorio en el primario debe degradar al fallback y devolver ok."""
    client = ProductionLLMClient(model="primary", fallback_model="fallback")
    seen = []

    def handler(model):
        seen.append(model)
        if model == "primary":
            raise make_status_error(anthropic.RateLimitError, 429)
        return FakeStream(["resp ", "uesta"], FakeUsage(100, 20))

    patch_stream(monkeypatch, client, handler)
    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])

    assert res.status == "ok"
    assert res.model == "fallback"
    assert res.text == "resp uesta"
    assert res.used_fallback is True
    assert seen == ["primary", "fallback"]


def test_primary_ok_marks_used_fallback_false(monkeypatch):
    """Si el primario responde, used_fallback debe ser False."""
    client = ProductionLLMClient(model="primary", fallback_model="fallback")

    def handler(model):
        return FakeStream(["ok"], FakeUsage(10, 2))

    patch_stream(monkeypatch, client, handler)
    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])

    assert res.status == "ok"
    assert res.model == "primary"
    assert res.used_fallback is False
