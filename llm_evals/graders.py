"""Pieza 3 del harness: los graders.

Leen el response CRUDO congelado, lo parsean, y emiten un veredicto por campo.
Aquí vive el json.loads: si el crudo no parsea, es un GRADE (parse_error), no un crash.
Frontera LB2: code-based para lo enumerable (type/breaking_change/issues),
LLM-judge para lo de vocabulario abierto (scope).
"""
from __future__ import annotations

import json

from llm_client import ProductionLLMClient

# --- Graders code-based (deterministas, 0 llamadas a LLM) ---


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


# --- Grader LLM-judge (para scope; vocabulario abierto) ---

JUDGE_MODEL = "claude-opus-4-8"  # desacoplado; idealmente distinto/superior al SUT


def grade_scope(predicted: str|None, golden: str, judge_client: ProductionLLMClient) -> bool:
    # Code short-circuits before paying for the llm judge
    if predicted == golden:  # null==null y exact-match
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


# --- Orquestación: gradar un caso completo ---


def grade_case(record: dict, judge_client: ProductionLLMClient) -> dict:
    # Decisión 1: outcomes distintos, sin reventar.
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
