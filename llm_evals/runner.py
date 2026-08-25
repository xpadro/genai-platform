"""Pieza 2 del harness: el runner.

  Genera y CONGELA los outputs crudos del SUT (una vez), separando generación
  de grading. Escribe línea a línea + flush para no perder llamadas ya pagadas
  si el run se corta. No parsea: el json.loads es del grader.
  """
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llm_client import ProductionLLMClient
from .dataset import EvalCase

SYSTEM = (
      "Extrae metadata de un mensaje de PR/commit y responde SOLO con un objeto JSON "
      "con estos campos:\n"
      '- "type": uno de ["feat","fix","refactor","docs","chore","test"]\n'
      '- "scope": el área del código afectada como string, o null si no aplica\n'
      '- "breaking_change": true o false\n'
      '- "issues": lista de issue keys (ej. ["PROJ-123"]), o [] si no hay'
  )
   
def run(cases: list[EvalCase], 
        client: ProductionLLMClient,
        out_dir: str | Path = "data/generations", 
        max_tokens: int = 1024) -> str:

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"{run_id}.jsonl"
    meta_path  = out / f"{run_id}.meta.json"

    with open(jsonl_path, "w", buffering=1, encoding="utf-8") as f:  # Decisión 2: line-buffered
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
            # ¿hace falta flush aquí, o buffering=1 ya lo cubre?  ← piénsalo

    meta = {
        # TU CÓDIGO: run_id, date, dataset?, model, fallback_model, max_tokens, n_cases
        "run_id": run_id,
        "date": datetime.now().isoformat(),
        "model": client.model,
        "fallback_model": client.fallback_model,
        "max_tokens": max_tokens,
        "n_cases": len(cases)
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_id