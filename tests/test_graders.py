"""Tests de la Pieza 3 (graders) con judge mockeado — sin gastar API.

Cubren los tres outcomes de grade_case (graded / parse_error / sut_error)
y la frontera LB2: scope semántico ("auth" ≈ "authentication") vía judge.
"""

import json

import pytest
from llm_client import LLMResult
from llm_evals.graders import (
    grade_case,
    grade_issues,
    grade_type,
)

# --- Doble de prueba: un judge falso que no toca la red ---


class FakeJudge:
    """Imita ProductionLLMClient para grade_scope.

    grade_scope solo llama a .complete() y lee .status y .text del resultado.
    Le decimos de antemano qué veredicto debe devolver.
    """

    def __init__(self, verdict: str = "correct", status: str = "ok"):
        self._verdict = verdict
        self._status = status

    def complete(self, system, messages, max_tokens=1024) -> LLMResult:
        text = f"El razonamiento va aquí.\nVERDICT: {self._verdict}"
        return LLMResult(text=text, model="fake-judge", status=self._status)


# --- Helper: construye un registro como el que escribe el runner ---


def make_record(response: str, golden: dict, status: str = "ok") -> dict:
    """Un registro del <run_id>.jsonl. `response` es el texto CRUDO del SUT."""
    return {
        "case_id": "test-case",
        "input": "irrelevante para el grader",
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


# --- Tests de los graders code-based (puros, sin judge) ---


def test_grade_issues_ignora_el_orden():
    # Decisión 3: comparar como conjuntos, no como listas.
    # TU CÓDIGO: assert que ["A","B"] y ["B","A"] son iguales para grade_issues
    assert grade_issues(["A", "B"], ["B", "A"]) is True



def test_grade_type_none_es_false():
    # Campo ausente (None) no debe reventar → False.
    assert grade_type(None, "feat") is False


# --- Tests de grade_case: los tres outcomes ---


def test_grade_case_todo_correcto():
    response = json.dumps(GOLDEN)  # el SUT acertó todo, JSON válido
    record = make_record(response, GOLDEN)
    result = grade_case(record, FakeJudge(verdict="correct"))

    assert result["outcome"] == "graded"
    assert result["fields"]["type"] is True
    assert result["fields"]["scope"] is True
    assert result["fields"]["breaking_change"] is True
    assert result["fields"]["issues"] is True


def test_grade_case_json_malformado():
    response = "esto no es json {"  # el SUT devolvió basura
    record = make_record(response, GOLDEN)
    result = grade_case(record, FakeJudge())

    assert result["outcome"] == "parse_error"


def test_grade_case_sut_caido():
    record = make_record(response="", golden=GOLDEN, status="unavailable")
    result = grade_case(record, FakeJudge())

    assert result["outcome"] == "sut_error"


def test_grade_case_scope_sinonimo_demuestra_LB2():
    # El SUT extrae "authentication"; el golden dice "auth".
    # Exact-match (code) FALLARÍA; el judge (mockeado a correct) lo aprueba.
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

def test_grade_case_scope_judge_rechaza():
    # golden "auth", pred "billing": el judge (mockeado) dice incorrect → scope False
    pred = {"type": "feat", "scope": "billing", "breaking_change": False, "issues": ["PROJ-101"]}
    record = make_record(json.dumps(pred), GOLDEN)
    result = grade_case(record, FakeJudge(verdict="incorrect"))

    assert result["fields"]["scope"] is False

def test_grade_scope_judge_caido_revienta():
    pred = {"type": "feat", "scope": "authentication", "breaking_change": False, "issues": ["PROJ-101"]}
    record = make_record(json.dumps(pred), GOLDEN)
    with pytest.raises(RuntimeError):
        grade_case(record, FakeJudge(status="unavailable"))
