import pytest

from llm_client import LLMResult
from llm_client.pricing import cost_usd


def test_cost_basic_sonnet():
    r = LLMResult(
        text="", model="claude-sonnet-4-6", status="ok",
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    # 1 MTok input ($3) + 1 MTok output ($15)
    assert cost_usd(r) == pytest.approx(18.0)


def test_cache_read_is_tenth_of_base_input():
    model = "claude-sonnet-4-6"
    base = LLMResult(text="", model=model, status="ok", input_tokens=1_000_000)
    cached = LLMResult(text="", model=model, status="ok", cache_read=1_000_000)
    assert cost_usd(cached) == pytest.approx(cost_usd(base) * 0.1)


def test_unknown_model_costs_zero():
    r = LLMResult(text="", model="nonexistent-model", status="ok", input_tokens=1_000_000)
    assert cost_usd(r) == 0.0
