"""status must never claim "ok" for an incomplete answer.

The API can return HTTP 200 with a reply that was cut at max_tokens or declined
(stop_reason "refusal"). Before this, the client mapped both to status="ok", so
a truncated judge verdict was silently graded as "incorrect" downstream.
"""

from _fakes import FakeStopDetails, FakeStream, FakeUsage, patch_stream

from llm_client import ProductionLLMClient


def _client_answering(monkeypatch, stream: FakeStream) -> tuple[ProductionLLMClient, list[str]]:
    client = ProductionLLMClient(model="primary", fallback_model="fallback")
    seen: list[str] = []

    def handler(model):
        seen.append(model)
        return stream

    patch_stream(monkeypatch, client, handler)
    return client, seen


def test_max_tokens_is_truncated_keeps_partial_text_and_no_fallback(monkeypatch):
    stream = FakeStream(['{"type": "fe'], FakeUsage(100, 1024), stop_reason="max_tokens")
    client, seen = _client_answering(monkeypatch, stream)

    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=1024)

    assert res.status == "truncated"
    assert res.stop_reason == "max_tokens"
    assert res.text == '{"type": "fe'  # partial answer kept: the caller decides
    assert "max_tokens=1024" in (res.reason or "")
    assert seen == ["primary"]  # neither retried nor sent to the fallback
    assert res.cost_usd >= 0  # the tokens were billed, so cost is still reported


def test_refusal_is_refused_with_category(monkeypatch):
    stream = FakeStream(
        [], FakeUsage(100, 5), stop_reason="refusal", stop_details=FakeStopDetails("cyber")
    )
    client, seen = _client_answering(monkeypatch, stream)

    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])

    assert res.status == "refused"
    assert res.stop_reason == "refusal"
    assert "category=cyber" in (res.reason or "")
    assert seen == ["primary"]


def test_end_turn_is_ok_and_records_stop_reason(monkeypatch):
    stream = FakeStream(["done"], FakeUsage(10, 2))
    client, _ = _client_answering(monkeypatch, stream)

    res = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])

    assert res.status == "ok"
    assert res.stop_reason == "end_turn"
    assert res.reason is None
