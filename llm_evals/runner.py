"""Piece 2 of the harness: the runner.

Generates and FREEZES the SUT's raw outputs (once), separating generation
from grading. Writes line by line + flush so we don't lose already-paid-for
calls if the run is interrupted. It doesn't parse: the json.loads is the grader's job.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from llm_client import LLMClient

from .dataset import EvalCase

SYSTEM = (
    "Extract metadata from a PR/commit message and respond ONLY with a JSON object "
    "with these fields:\n"
    '- "type": one of ["feat","fix","refactor","docs","chore","test"]\n'
    '- "scope": the affected area of the system, decided in this order:\n'
    "  1. if the title has an explicit scope, as in \"fix(parser): ...\", use it;\n"
    "  2. otherwise infer it from the title or body ONLY if the text names a concrete\n"
    '     part of the system (e.g. "the billing service" -> "billing", "the CLI parser" -> "cli");\n'
    "  3. otherwise null. A scope that only repeats the type (e.g. \"docs\" for a docs\n"
    "     change) or names no concrete area is null.\n"
    '- "breaking_change": true or false\n'
    '- "issues": list of issue keys (e.g. ["PROJ-123"]), or [] if none'
)

# Output contract of the SUT, enforced by the API (structured outputs): the reply
# is always parseable JSON of this shape. It guarantees FORM, not content: a
# wrong scope or a made-up issue still gets through, and catching it is the eval's job.
# - scope: required but nullable -> "no scope" is an explicit null, never a missing key.
# - issues: plain strings, no pattern -> issue-key conventions vary per team/tool, and
#   a pattern would force the model to bend a "#123" into shape, hiding the error.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["feat", "fix", "refactor", "docs", "chore", "test"],
        },
        "scope": {"type": ["string", "null"]},
        "breaking_change": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["type", "scope", "breaking_change", "issues"],
    "additionalProperties": False,
}


def run(
    cases: list[EvalCase],
    client: LLMClient,
    out_dir: str | Path = "data",
    max_tokens: int = 1024,
    dataset: str | None = None,
) -> str:

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    out = Path(out_dir) / "generations"
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"{run_id}.jsonl"
    meta_path = out / f"{run_id}.meta.json"

    with open(
        jsonl_path, "w", buffering=1, encoding="utf-8"
    ) as f:  # Decision 2: line-buffered
        for case in cases:
            result = client.complete(
                system=SYSTEM,
                messages=[{"role": "user", "content": case.input}],
                max_tokens=max_tokens,
                output_schema=EXTRACTION_SCHEMA,
            )
            record = {
                "case_id": case.id,
                "input": case.input,
                "golden": case.golden,
                "response": result.text,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "model": result.model,
                "used_fallback": result.used_fallback,
                "cost_usd": result.cost_usd,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "run_id": run_id,
        "date": datetime.now().isoformat(),
        "dataset": dataset,  # which eval set produced this run (traceability)
        # Identity of the dataset = hash of its content. The path alone can't tell
        # two runs apart if the golden was edited in place (committed or not).
        "dataset_sha256": dataset_sha256(dataset) if dataset else None,
        "model": client.model,
        "fallback_model": client.fallback_model,
        "max_tokens": max_tokens,
        # The whole SUT config (model + prompt + params), so two runs can be
        # told apart from their meta alone.
        "system": SYSTEM,
        "output_schema": EXTRACTION_SCHEMA,
        "n_cases": len(cases),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_id


def dataset_sha256(path: str | Path) -> str:
    """Fingerprint of the dataset's CONTENT, the same on every platform.

    Line endings are normalized first: git checks the file out with CRLF on
    Windows (core.autocrlf) and LF on Linux/CI, and the same commit must not
    produce two different fingerprints.
    """
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_generations(run_id: str, out_dir: str | Path = "data") -> list[dict]:
    """Reload a frozen run from disk into records — the inverse of `run`.

    This is the seam between generation and grading, and it goes through DISK on
    purpose: the grader consumes what was frozen, so you can re-grade a run
    (iterate the rubric) without re-generating (0 API calls). It's the mirror of
    dataset.load_cases: one line == one record, exactly as `run` wrote it.
    """
    jsonl_path = Path(out_dir) / "generations" / f"{run_id}.jsonl"
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records
