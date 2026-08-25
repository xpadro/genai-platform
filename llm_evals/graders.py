"""Piece 3 of the harness: the graders.

They read the frozen RAW response, parse it, and emit a verdict per field.
The json.loads lives here: if the raw doesn't parse, that's a GRADE (parse_error), not a crash.
LB2 boundary: code-based for the enumerable fields (type/breaking_change/issues),
LLM-judge for the open-vocabulary one (scope).
"""
from __future__ import annotations

import json

from llm_client import ProductionLLMClient

# --- Code-based graders (deterministic, 0 LLM calls) ---


def grade_type(predicted: str|None, golden: str) -> bool:
    if predicted is None:
        return False

    return predicted.strip().lower() == golden.strip().lower()


def grade_breaking_change(predicted: bool|None, golden: bool) -> bool:
    return predicted == golden


def grade_issues(predicted: list[str]|None, golden: list[str]) -> bool:
    if predicted is None:
        return False

    return set(predicted) == set(golden)


# --- LLM-judge grader (for scope; open vocabulary) ---

JUDGE_MODEL = "claude-opus-4-8"  # decoupled; ideally different from/superior to the SUT


def grade_scope(predicted: str|None, golden: str, judge_client: ProductionLLMClient) -> bool:
    # Code short-circuits before paying for the llm judge
    if predicted == golden:  # null==null and exact-match
        return True
    if predicted is None or golden is None:
        return False  # One is null, the other is not (no equivalence)

    judge_prompt = (
        "You grade whether two code-scope labels refer to the same component, "
        "even with different wording (e.g. 'auth' and 'authentication' match; "
        "'auth' and 'billing' do not).\n\n"
        f"<predicted>{predicted}</predicted>\n"
        f"<golden>{golden}</golden>\n\n"
        "First reason briefly. Then, on the final line, output exactly "
        "'VERDICT: correct' or 'VERDICT: incorrect'."
    )

    result = judge_client.complete(
        system="", messages=[{"role": "user", "content": judge_prompt}]
    )
    
    if result.status != "ok":
        raise RuntimeError(f"judge unavailable: {result.reason}")
    
    last_line = result.text.strip().splitlines()[-1]
    verdict = last_line.split("VERDICT:")[-1].strip().lower()
    return verdict == "correct"


# --- Orchestration: grade a full case ---


def grade_case(record: dict, judge_client: ProductionLLMClient) -> dict:
    # Decision 1: distinct outcomes, without blowing up.
    if record["status"] != "ok":
        return {
            "case_id": record["case_id"],
            "outcome": "sut_error",
            "status": record["status"],
        }
    try:
        pred = json.loads(record["response"])
    except json.JSONDecodeError:
        return {"case_id": record["case_id"], "outcome": "parse_error"}

    golden = record["golden"]
    return {
        "case_id": record["case_id"],
        "outcome": "graded",
        "fields": {
            "type": grade_type(pred.get("type"), golden["type"]),
            "breaking_change": grade_breaking_change(pred.get("breaking_change"), golden["breaking_change"]),
            "issues": grade_issues(pred.get("issues"), golden["issues"]),
            "scope": grade_scope(pred.get("scope"), golden["scope"], judge_client)
        },
    }
