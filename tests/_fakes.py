"""Test doubles so we don't spend API in CI."""
import anthropic
import httpx


class FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class FakeStopDetails:
    def __init__(self, category=None):
        self.category = category


class _FakeFinal:
    def __init__(self, usage, stop_reason="end_turn", stop_details=None):
        self.usage = usage
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class FakeStream:
    """Mimics the context manager of messages.stream()."""

    def __init__(self, chunks, usage, stop_reason="end_turn", stop_details=None):
        self._chunks = chunks
        self._usage = usage
        self._stop_reason = stop_reason
        self._stop_details = stop_details

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return _FakeFinal(self._usage, self._stop_reason, self._stop_details)


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
