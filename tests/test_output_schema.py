"""output_schema reaches the SDK as output_config, only when given, fallback included."""

import anthropic
from _fakes import FakeStream, FakeUsage, make_status_error

from llm_client import ProductionLLMClient

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "hi"}]


def record_stream_calls(monkeypatch, client, fail_models=()):
    """Patches messages.stream to record each call's kwargs; models in fail_models raise 529."""
    calls: list[dict] = []

    def fake_stream(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] in fail_models:
            raise make_status_error(anthropic.InternalServerError, 529)
        return FakeStream(['{"a": 1}'], FakeUsage())

    monkeypatch.setattr(client._c.messages, "stream", fake_stream)
    return calls


def test_schema_is_sent_as_output_config(monkeypatch):
    client = ProductionLLMClient(model="primary", fallback_model="fallback")
    calls = record_stream_calls(monkeypatch, client)

    client.complete(system="s", messages=MESSAGES, output_schema=SCHEMA)

    assert calls[0]["output_config"] == {
        "format": {"type": "json_schema", "schema": SCHEMA}
    }


def test_no_schema_sends_no_output_config(monkeypatch):
    """The judge path: free text, and output_config must not even be present (None is a 400)."""
    client = ProductionLLMClient(model="primary", fallback_model="fallback")
    calls = record_stream_calls(monkeypatch, client)

    client.complete(system="s", messages=MESSAGES)

    assert "output_config" not in calls[0]


def test_fallback_keeps_the_same_schema(monkeypatch):
    """The contract belongs to complete(), not to the model that happens to answer."""
    client = ProductionLLMClient(model="primary", fallback_model="fallback")
    calls = record_stream_calls(monkeypatch, client, fail_models={"primary"})

    res = client.complete(system="s", messages=MESSAGES, output_schema=SCHEMA)

    assert res.used_fallback
    assert [c["model"] for c in calls] == ["primary", "fallback"]
    assert calls[1]["output_config"] == calls[0]["output_config"]
