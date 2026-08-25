# llm_evals — reproducible eval harness

Measures, in a **reproducible and diffable** way, whether the SUT (the LLM client from `llm_client`)
correctly extracts structured-output metadata from PR/commit text.

Guiding principle: **separate generation from grading**. The runner generates and *freezes* the raw
outputs once (expensive, non-deterministic); the grader scores them afterwards (cheap, deterministic,
re-runnable at 0 API cost). Freezing the raw output is what makes *error analysis* honest: the failure
you analyze doesn't disappear when you re-generate.

## Flow

```text
                         data/eval_set.jsonl          ← INPUT, versioned in git
                         {id, input, golden}            (one line per case)
                                 │
                                 │ dataset.load_cases()
                                 ▼
                           list[EvalCase]
                                 │
                                 │ runner: for each case → client.complete(input)
                                 ▼
                            LLMResult                   (.text = RAW text, unparsed
                          (text, status, model,          + status/model/cost/tokens)
                           cost_usd, tokens, …)
                                 │
                                 │ runner writes TWO files per run:
                                 ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ OUTPUT of 1 run = FROZEN generations                                          │
   │                                                                               │
   │  <run_id>.jsonl              ← 1 LINE PER CASE (8 cases → 8 lines)            │
   │    {case_id, input, golden, response(raw), status,                            │
   │     model, used_fallback, cost_usd, tokens}                                   │
   │                                                                               │
   │  <run_id>.meta.json          ← ONCE per run (what's common to the run)        │
   │    {run_id, date, dataset, model, fallback_model, max_tokens, n_cases}        │
   └─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ grader reads the frozen JSONL:
                                 │   parsed = json.loads(response)   ← parsing lives HERE, not in the runner
                                 │   code-based: type / breaking_change / issues
                                 │   llm-judge:  scope  ("auth" ≈ "authentication")
                                 ▼
                            list[Grade]  (per case, per field)
                                 │
                                 │ report: aggregates (accuracy / P·R·F1) + per-case table
                                 ▼
                    report to disk (JSON/CSV), diffable across runs
```

## How the pieces relate

- **`case_id` is the key that ties everything together.** It's born in `eval_set.jsonl` (`id`), travels to the
  generation record (`case_id`), and is how the grader joins output ↔ golden and the report joins grade ↔ case.
  Without it there's no join and no re-grading.
- **The `golden` is *copied* into `<run_id>.jsonl`** (not referenced). This makes the generations file
  **self-contained**: the grader scores without reopening the dataset, and the run is snapshotted exactly as the
  golden was that day (if tomorrow you edit `eval_set.jsonl`, old runs don't change).
- **`response` is frozen RAW.** The `json.loads()` is the *grader's* responsibility, not the runner's.
  Reason: if the model emits malformed JSON, `json.loads` blows up → if the runner parsed, you'd lose the
  raw output and be blind to exactly that failure mode. With the raw output frozen, "invalid JSON" becomes a
  *measurable grade*, not a crash; and if you improve the parser, you re-parse at 0 cost.
- **`status` distinguishes "bad output" from "no output".** `ok | error | unavailable` (from `LLMResult`).
  If `status != ok`, `response` comes back empty → the grader treats the case as an availability failure,
  not as incorrect extraction.
- **`used_fallback` explains weird outputs.** If the primary went down and the fallback (haiku) responded, a
  "bad" case may be bad because of the degraded model, not the prompt. `model` records the one that *actually*
  responded.
- **Run metadata vs. case metadata.** What's the same for the whole run (`run_id`, `date`, model config,
  dataset) goes into the `.meta.json`; what varies per case goes into the `.jsonl`.

## Pieces

| File | Role | Status |
|---|---|---|
| `dataset.py` | loads `eval_set.jsonl` → `list[EvalCase]` | ✅ |
| `runner.py` | generates and **freezes** raw outputs (`<run_id>.jsonl` + `.meta.json`) | ✅ |
| `graders.py` | code-based (`type`/`breaking_change`/`issues`) + llm-judge (`scope`) | ✅ |
| `report.py` | aggregates metrics + per-case table, to disk | ⬜ |
| `__main__.py` | CLI: `python -m llm_evals run --set data/eval_set.jsonl` | ⬜ |
