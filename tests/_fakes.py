"""Test doubles so we don't spend API in CI."""
import anthropic
import httpx


class FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class _FakeFinal:
    def __init__(self, usage):
        self.usage = usage


class FakeStream:
    """Mimics the context manager of messages.stream()."""

    def __init__(self, chunks, usage):
        self._chunks = chunks
        self._usage = usage

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return _FakeFinal(self._usage)


def make_status_error(cls, status_code):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status_code, request=req)
    return cls("simulated", response=resp, body=None)


def make_connection_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="simulated", request=req)


def patch_stream(monkeypatch, client, handler):
    """Replaces client._c.messages.stream with a handler(model) -> FakeStream | raise."""

    def fake_stream(*, model, system, messages, max_tokens):
        return handler(model)

    monkeypatch.setattr(client._c.messages, "stream", fake_stream)
