"""Piece 2 of the harness: the runner.

  Generates and FREEZES the SUT's raw outputs (once), separating generation
  from grading. Writes line by line + flush so we don't lose already-paid-for
  calls if the run is interrupted. It doesn't parse: the json.loads is the grader's job.
  """
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llm_client import ProductionLLMClient
from .dataset import EvalCase

SYSTEM = (
      "Extract metadata from a PR/commit message and respond ONLY with a JSON object "
      "with these fields:\n"
      '- "type": one of ["feat","fix","refactor","docs","chore","test"]\n'
      '- "scope": the affected area of the code as a string, or null if not applicable\n'
      '- "breaking_change": true or false\n'
      '- "issues": list of issue keys (e.g. ["PROJ-123"]), or [] if none'
  )
   
def run(cases: list[EvalCase], 
        client: ProductionLLMClient,
        out_dir: str | Path = "data/generations", 
        max_tokens: int = 1024) -> str:

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"{run_id}.jsonl"
    meta_path  = out / f"{run_id}.meta.json"

    with open(jsonl_path, "w", buffering=1, encoding="utf-8") as f:  # Decision 2: line-buffered
        for case in cases:
            result = client.complete(system=SYSTEM,
                                    messages=[{"role": "user", "content": case.input}],
                                    max_tokens=max_tokens)
            record = {
                "case_id": case.id,
                "input": case.input,
                "golden": case.golden,
                "response": result.text,
                "status": result.status,
                "model": result.model,
                "used_fallback": result.used_fallback,
                "cost_usd": result.cost_usd,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # is a flush needed here, or does buffering=1 already cover it?  ← think about it

    meta = {
        "run_id": run_id,
        "date": datetime.now().isoformat(),
        "model": client.model,
        "fallback_model": client.fallback_model,
        "max_tokens": max_tokens,
        "n_cases": len(cases)
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_id