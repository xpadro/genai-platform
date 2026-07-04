"""Pieza 1 del harness: el dataset.

Casos versionados en JSONL (git). Cada caso = input + golden esperado.
El dataset es dato, no código: se lee, no se genera al vuelo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    golden: dict


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                id=obj.get("id", f"case-{i}"),
                input=obj["input"],
                golden=obj["golden"],
            )
        )
    return cases
