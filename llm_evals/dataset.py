"""Piece 1 of the harness: the dataset.

Cases versioned in JSONL (git). Each case = input + expected golden.
The dataset is data, not code: it's read, not generated on the fly.
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
