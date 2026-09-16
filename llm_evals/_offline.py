"""Offline doubles to smoke-test the whole pipeline without spending API.

Used ONLY by `python -m llm_evals run --fake`. They mimic the surface the
runner and the judge grader actually touch (`.complete()`, `.model`,
`.fallback_model`), so the pipeline runs end-to-end, offline and
deterministically. Real runs use llm_client.ProductionLLMClient instead.

These are NOT the SUT and NOT a real judge: they exist to exercise the WIRING,
not to measure a model. The numbers a fake run reports are meaningless by design.
"""
from __future__ import annotations

import json

from llm_client import LLMResult


class FakeSUT:
    """Stands in for the system under test (SUT) inside the runner.

    Returns one fixed, plausible extraction for EVERY case. It never sees the
    golden, so across the dataset some fields land right and some wrong -> the
    report shows a realistic mix per field. All outputs are valid JSON, so every
    case ends up `graded` (to see the parse_error/sut_error paths flow through
    the report, make this return garbage or a non-ok status for one case).
    """

    model = "fake-sut"
    fallback_model = "fake-sut-fallback"

    def complete(self, system, messages, max_tokens: int = 1024) -> LLMResult:
        canned = json.dumps(
            {"type": "feat", "scope": "auth", "breaking_change": False, "issues": []}
        )
        return LLMResult(
            text=canned,
            model=self.model,
            status="ok",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )


class FakeJudge:
    """Stands in for the judge_client in grade_scope.

    Always returns 'VERDICT: correct' so the semantic-scope path runs offline
    without hitting the API. (grade_scope only reads .status and .text.)
    """

    model = "fake-judge"
    fallback_model = "fake-judge-fallback"

    def complete(self, system, messages, max_tokens: int = 1024) -> LLMResult:
        return LLMResult(
            text="reasoning goes here.\nVERDICT: correct",
            model=self.model,
            status="ok",
        )
