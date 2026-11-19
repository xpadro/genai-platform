"""Tests for Piece 4 (report): build_report aggregation.

These pin the *honest denominator*: field accuracy is computed ONLY over graded
cases, and when there are no graded cases the accuracy is None (undefined), never
0.0 (a measured-and-terrible claim). Getting this wrong reintroduces survivorship
bias — a run that fails to parse everything would look "merely bad" instead of
"unmeasured".
"""

import pytest

from llm_evals.report import build_report

# --- Helpers: build grade dicts exactly as grade_case emits them ---


def graded(
    case_id: str, *, type_=True, scope=True, breaking_change=True, issues=True
) -> dict:
    """A `graded` outcome with the four field verdicts (bools)."""
    return {
        "case_id": case_id,
        "outcome": "graded",
        "fields": {
            "type": type_,
            "scope": scope,
            "breaking_change": breaking_change,
            "issues": issues,
        },
    }


def parse_error(case_id: str) -> dict:
    return {"case_id": case_id, "outcome": "parse_error"}


def sut_error(case_id: str, status: str = "unavailable") -> dict:
    return {"case_id": case_id, "outcome": "sut_error", "status": status}


# --- Test 1: all graded, mixed field correctness ---


def test_all_graded_field_accuracy():
    grades = [
        graded("a", type_=True, scope=True, breaking_change=True, issues=True),
        graded("b", type_=True, scope=False, breaking_change=True, issues=False),
        graded("c", type_=False, scope=True, breaking_change=True, issues=False),
        graded("d", type_=True, scope=True, breaking_change=False, issues=True),
    ]
    s = build_report(grades)["summary"]

    # assert n_total, n_graded, outcomes, and field_accuracy.
    # (type: 3/4, scope: 3/4, breaking_change: 3/4, issues: 2/4)
    assert s["n_total"] == 4
    assert s["n_graded"] == 4
    assert s["outcomes"] == {"graded": 4}
    assert s["field_accuracy"] == pytest.approx(
        {
            "type": 3 / 4,
            "scope": 3 / 4,
            "breaking_change": 3 / 4,
            "issues": 2 / 4,
        }
    )


# --- Test 2: mixed outcomes — the honest denominator ---


def test_mixed_outcomes_denominator():
    grades = [
        graded("a", type_=True, scope=True, breaking_change=True, issues=True),
        graded("b", type_=False, scope=True, breaking_change=True, issues=True),
        parse_error("c"),
        sut_error("d"),
    ]
    s = build_report(grades)["summary"]

    # YOUR CODE: n_total is 4 but n_graded is 2; outcomes counts all three kinds;
    # field_accuracy is computed ONLY over the 2 graded (e.g. type == 1/2), NOT
    # diluted by the 2 failures. Assert exactly that.
    assert s["n_total"] == 4
    assert s["n_graded"] == 2
    assert s["outcomes"] == {"graded": 2, "parse_error": 1, "sut_error": 1}
    assert s["field_accuracy"] == pytest.approx(
        {
            "type": 1 / 2,
            "scope": 2 / 2,
            "breaking_change": 2 / 2,
            "issues": 2 / 2,
        }
    )


# --- Test 3: zero graded -> None, NOT 0.0 (the decision we debated) ---


def test_zero_graded_is_none_not_zero():
    grades = [parse_error("a"), sut_error("b")]
    s = build_report(grades)["summary"]

    # YOUR CODE: n_graded == 0, and EVERY field in field_accuracy is None
    # (undefined), never 0.0. This is the test that must never silently flip.
    assert s["n_total"] == 2
    assert s["n_graded"] == 0
    assert s["outcomes"] == {"parse_error": 1, "sut_error": 1}
    assert s["field_accuracy"] == {
        "type": None,
        "scope": None,
        "breaking_change": None,
        "issues": None,
    }
