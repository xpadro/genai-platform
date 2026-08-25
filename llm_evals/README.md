# llm_evals — eval harness reproducible

Mide, de forma **reproducible y diffable**, si el SUT (el cliente LLM de `llm_client`)
extrae bien metadata structured-output desde texto de PR/commit.

Principio rector: **separar generación de grading**. El runner genera y *congela* los
outputs crudos una vez (caro, no-determinista); el grader los puntúa después (barato,
determinista, re-ejecutable a coste 0 de API). Congelar el crudo es lo que hace honesto
el *error analysis*: el fallo que analizas no desaparece al re-generar.

## Flujo

```text
                         data/eval_set.jsonl          ← INPUT, versionado en git
                         {id, input, golden}            (una línea por caso)
                                 │
                                 │ dataset.load_cases()
                                 ▼
                           list[EvalCase]
                                 │
                                 │ runner: por cada caso → client.complete(input)
                                 ▼
                            LLMResult                   (.text = texto CRUDO, sin parsear
                          (text, status, model,          + status/model/coste/tokens)
                           cost_usd, tokens, …)
                                 │
                                 │ runner escribe DOS ficheros por ejecución:
                                 ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ OUTPUT de 1 ejecución = generaciones CONGELADAS                               │
   │                                                                               │
   │  <run_id>.jsonl              ← 1 LÍNEA POR CASO (8 casos → 8 líneas)          │
   │    {case_id, input, golden, response(crudo), status,                          │
   │     model, used_fallback, cost_usd, tokens}                                   │
   │                                                                               │
   │  <run_id>.meta.json          ← 1 SOLA VEZ por ejecución (lo común al run)     │
   │    {run_id, date, dataset, model, fallback_model, max_tokens, n_cases}        │
   └─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ grader lee el JSONL congelado:
                                 │   parsed = json.loads(response)   ← el parseo vive AQUÍ, no en el runner
                                 │   code-based: type / breaking_change / issues
                                 │   llm-judge:  scope  ("auth" ≈ "authentication")
                                 ▼
                            list[Grade]  (por caso, por campo)
                                 │
                                 │ report: agrega (accuracy / P·R·F1) + tabla por caso
                                 ▼
                    reporte a disco (JSON/CSV), diffable entre runs
```

## Cómo se relacionan las piezas

- **`case_id` es la clave que une todo.** Nace en `eval_set.jsonl` (`id`), viaja al registro de
  generación (`case_id`) y es por donde el grader une output ↔ golden y el report une grade ↔ caso.
  Sin él no hay join ni re-grading.
- **El `golden` se *copia* dentro de `<run_id>.jsonl`** (no se referencia). Así el fichero de
  generaciones es **autocontenido**: el grader puntúa sin volver a abrir el dataset, y el run queda
  fotografiado tal cual estaba el golden ese día (si mañana editas `eval_set.jsonl`, los runs viejos
  no cambian).
- **`response` se congela CRUDO.** El `json.loads()` es responsabilidad del *grader*, no del runner.
  Motivo: si el modelo emite JSON malformado, `json.loads` peta → si el runner parseara, perderías el
  crudo y quedarías ciego justo ante ese failure mode. Con el crudo congelado, "JSON inválido" pasa a
  ser un *grade medible*, no un crash; y si mejoras el parser, re-parseas a coste 0.
- **`status` distingue "mal output" de "sin output".** `ok | error | unavailable` (de `LLMResult`).
  Si `status != ok`, `response` viene vacío → el grader trata el caso como fallo de disponibilidad,
  no como extracción incorrecta.
- **`used_fallback` explica outputs raros.** Si el primario cayó y respondió el fallback (haiku), un
  caso "malo" puede serlo por el modelo degradado, no por el prompt. `model` guarda el que *realmente*
  respondió.
- **Metadata de run vs. de caso.** Lo que es igual para toda la ejecución (`run_id`, `date`, config
  del modelo, dataset) va al `.meta.json`; lo que varía por caso va al `.jsonl`.

## Piezas

| Fichero | Rol | Estado |
|---|---|---|
| `dataset.py` | carga `eval_set.jsonl` → `list[EvalCase]` | ✅ |
| `runner.py` | genera y **congela** outputs crudos (`<run_id>.jsonl` + `.meta.json`) | 🚧 |
| `graders.py` | code-based (`type`/`breaking_change`/`issues`) + llm-judge (`scope`) | ⬜ |
| `report.py` | agrega métricas + tabla por caso, a disco | ⬜ |
| `__main__.py` | CLI: `python -m llm_evals run --set data/eval_set.jsonl` | ⬜ |
