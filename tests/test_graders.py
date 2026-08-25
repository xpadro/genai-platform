"""Tests for Piece 3 (graders) with a mocked judge — no API spend.

They cover the three outcomes of grade_case (graded / parse_error / sut_error)
and the LB2 boundary: semantic scope ("auth" ≈ "authentication") via the judge.
"""

import json

import pytest
from llm_client import LLMResult
from llm_evals.graders import (
    grade_case,
    grade_issues,
    grade_type,
)

# --- Test double: a fake judge that doesn't touch the network ---


class FakeJudge:
    """Mimics ProductionLLMClient for grade_scope.

    grade_scope only calls .complete() and reads .status and .text from the result.
    We tell it up front which verdict it should return.
    """

    def __init__(self, verdict: str = "correct", status: str = "ok"):
        self._verdict = verdict
        self._status = status

    def complete(self, system, messages, max_tokens=1024) -> LLMResult:
        text = f"The reasoning goes here.\nVERDICT: {self._verdict}"
        return LLMResult(text=text, model="fake-judge", status=self._status)


# --- Helper: builds a record like the one the runner writes ---


def make_record(response: str, golden: dict, status: str = "ok") -> dict:
    """A record from <run_id>.jsonl. `response` is the SUT's RAW text."""
    return {
        "case_id": "test-case",
        "input": "irrelevant to the grader",
        "golden": golden,
        "response": response,
        "status": status,
        "model": "claude-sonnet-4-6",
        "used_fallback": False,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


GOLDEN = {
    "type": "feat",
    "scope": "auth",
    "breaking_change": False,
    "issues": ["PROJ-101"],
}


# --- Tests for the code-based graders (pure, no judge) ---


def test_grade_issues_ignores_order():
    # Decision 3: compare as sets, not as lists.
    assert grade_issues(["A", "B"], ["B", "A"]) is True



def test_grade_type_none_is_false():
    # A missing field (None) must not blow up → False.
    assert grade_type(None, "feat") is False


# --- grade_case tests: the three outcomes ---


def test_grade_case_all_correct():
    response = json.dumps(GOLDEN)  # the SUT got everything right, valid JSON
    record = make_record(response, GOLDEN)
    result = grade_case(record, FakeJudge(verdict="correct"))

    assert result["outcome"] == "graded"
    assert result["fields"]["type"] is True
    assert result["fields"]["scope"] is True
    assert result["fields"]["breaking_change"] is True
    assert result["fields"]["issues"] is True


def test_grade_case_malformed_json():
    response = "this is not json {"  # the SUT returned garbage
    record = make_record(response, GOLDEN)
    result = grade_case(record, FakeJudge())

    assert result["outcome"] == "parse_error"


def test_grade_case_sut_down():
    record = make_record(response="", golden=GOLDEN, status="unavailable")
    result = grade_case(record, FakeJudge())

    assert result["outcome"] == "sut_error"


def test_grade_case_scope_synonym_demonstrates_LB2():
    # The SUT extracts "authentication"; the golden says "auth".
    # Exact-match (code) WOULD FAIL; the judge (mocked to correct) approves it.
    pred = {
        "type": "feat",
        "scope": "authentication",
        "breaking_change": False,
        "issues": ["PROJ-101"],
    }
    record = make_record(json.dumps(pred), GOLDEN)
    result = grade_case(record, FakeJudge(verdict="correct"))

    assert result["fields"]["scope"] is True
    assert result["fields"]["type"] is True

def test_grade_case_scope_judge_rejects():
    # golden "auth", pred "billing": the judge (mocked) says incorrect → scope False
    pred = {"type": "feat", "scope": "billing", "breaking_change": False, "issues": ["PROJ-101"]}
    record = make_record(json.dumps(pred), GOLDEN)
    result = grade_case(record, FakeJudge(verdict="incorrect"))

    assert result["fields"]["scope"] is False

def test_grade_scope_judge_down_raises():
    pred = {"type": "feat", "scope": "authentication", "breaking_change": False, "issues": ["PROJ-101"]}
    record = make_record(json.dumps(pred), GOLDEN)
    with pytest.raises(RuntimeError):
        grade_case(record, FakeJudge(status="unavailable"))
