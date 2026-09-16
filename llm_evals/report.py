"""Piece 4 of the harness: the report.

Aggregates per-case grades into metrics and writes them to disk (diffable
across runs). Honest denominator: field accuracy is computed ONLY over graded
cases; ungraded cases (parse_error / sut_error) are reported separately as
counts — never hidden, never mixed into the field metric.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

FIELDS = ["type", "scope", "breaking_change", "issues"]


def build_report(grades: list[dict]) -> dict:
    """Pure aggregation: grades (from grade_case) -> report dict. No I/O."""
    n_total = len(grades)

    # count how many grades have each outcome
    # (graded / parse_error / sut_error). Counter over g["outcome"] is idiomatic.
    outcomes = Counter(g["outcome"] for g in grades)

    graded = [g for g in grades if g["outcome"] == "graded"]
    n_graded = len(graded)

    # Field accuracy ONLY over graded cases (Decision 1). Guard division by zero.
    field_accuracy: dict[str, float | None] = {}
    for field in FIELDS:
        # proportion of graded cases whose fields[field] is True.
        # If n_graded == 0 -> None (undefined), not 0.0.
        if n_graded == 0:
            field_accuracy[field] = None
        else:
            correct_count = sum(1 for g in graded if g["fields"].get(field) is True)
            field_accuracy[field] = correct_count / n_graded

    return {
        "summary": {
            "n_total": n_total,
            "n_graded": n_graded,
            "outcomes": dict(outcomes),
            "field_accuracy": field_accuracy,
        },
        # Per-case table: enables error analysis (which cases failed → go read
        # their raw response) and per-case diffing across runs (regressions).
        "cases": grades,
    }


def write_report(report: dict, run_id: str, out_dir: str | Path = "data/reports") -> Path:
    """Write the report to disk as JSON (machine-readable, diffable)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_report(report: dict) -> None:
    """Human-readable summary to stdout (the headline the engineer/VP reads)."""
    s = report["summary"]
    # YOUR CODE: print n_total, n_graded, the outcomes breakdown, and each
    # field's accuracy in a readable way. Show None as "n/a" (no graded cases).
    print(f"Total cases: {s['n_total']}")
    print(f"Graded cases: {s['n_graded']}")
    print("Outcomes breakdown:")
    for outcome, count in s["outcomes"].items():
        print(f"  {outcome}: {count}")
    print("Field accuracy:")
    for field, accuracy in s["field_accuracy"].items():
        accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "n/a"
        print(f"  {field}: {accuracy_str}")
