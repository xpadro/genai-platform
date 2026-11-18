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
   │    {case_id, input, golden, response(raw), status, stop_reason,               │
   │     model, used_fallback, cost_usd, tokens}                                   │
   │                                                                               │
   │  <run_id>.meta.json          ← ONCE per run (what's common to the run)        │
   │    {run_id, date, dataset, dataset_sha256, model, fallback_model,             │
   │     max_tokens, system, output_schema, n_cases}                               │
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
- **`status` distinguishes "bad output" from "no output".** `ok | truncated | refused | error | unavailable`
  (from `LLMResult`). Only `ok` is graded; any other status → `sut_error` (with the status), not incorrect
  extraction. `truncated` keeps the partial raw `response` frozen for inspection. The judge follows the same
  rule the other way round: a judge reply that isn't `ok` raises (broken instrument ≠ data).
- **`used_fallback` explains weird outputs.** If the primary went down and the fallback (haiku) responded, a
  "bad" case may be bad because of the degraded model, not the prompt. `model` records the one that *actually*
  responded.
- **Run metadata vs. case metadata.** What's the same for the whole run (`run_id`, `date`, model config,
  dataset) goes into the `.meta.json`; what varies per case goes into the `.jsonl`.
- **The `.meta.json` records the whole SUT config**, so two runs can be told apart from their meta alone:
  model + prompt (`system`) + params (`max_tokens`, `output_schema`) + the dataset's identity.
- **The dataset's identity is its content, not its path.** `dataset_sha256` hashes the file with line
  endings normalized (git checks out CRLF on Windows and LF on CI; the same commit must give one hash).
  The path alone can't tell two runs apart if the golden was edited in place; a git commit hash would
  lie when the run is made with uncommitted changes.
- **The SUT's output contract is enforced by the API** (structured outputs, `EXTRACTION_SCHEMA` in
  `runner.py`): the reply is always parseable JSON of that shape. That guarantees *form*, not content:
  a wrong scope still gets through, which is what the eval is for. `parse_error` stays in the grader as
  defense in depth, but is no longer informative.

## Pieces

| File | Role | Status |
|---|---|---|
| `dataset.py` | loads `eval_set.jsonl` → `list[EvalCase]` | ✅ |
| `runner.py` | generates and **freezes** raw outputs (`<run_id>.jsonl` + `.meta.json`) | ✅ |
| `graders.py` | code-based (`type`/`breaking_change`/`issues`) + llm-judge (`scope`) | ✅ |
| `report.py` | aggregates metrics + per-case table, to disk | ✅ |
| `__main__.py` | CLI: `python -m llm_evals run --set data/eval_set.jsonl [--out DIR] [--fake]` | ✅ |

## Running

**Run from the repo root (`genai-platform/`).** That's where the package and `data/` live; running
repo tooling from the repo root is the convention (like `pytest` or `make`).

```bash
python -m llm_evals run --set data/eval_set.jsonl          # real API (needs a key)
python -m llm_evals run --set data/eval_set.jsonl --fake   # offline smoke, no API spend
python -m llm_evals run --set data/eval_set.jsonl --out /tmp/evals  # custom output base dir
```

All outputs go under one **base dir**, `--out` (default `data`): frozen generations in
`<out>/generations/` and reports in `<out>/reports/`. Re-grading an old run only needs that base
dir plus its `run_id`. Relative paths (`--set`, and `--out` incl. its default) resolve against the
**current working directory**; running from the repo root keeps them all consistent. (The package itself is an editable install, so it imports from any
CWD — only these paths depend on where you run from.)

## Baseline (real runs, 2026-11-18)

SUT `claude-sonnet-4-6` (fallback `claude-haiku-4-5-20251001`, not used), judge `claude-opus-4-8`,
8 cases, 1 sample per case. Frozen generations + meta + report of each run are kept in
[`data/baselines/`](../data/baselines/) (the regular `data/generations|reports/` are git-ignored).

| Run | Change | `parse_error` | `type` | `scope` (judge) | `breaking_change` | `issues` | Cost |
|---|---|---|---|---|---|---|---|
| [0](../data/baselines/run-0-no-format/) | — (prompt asks for "ONLY a JSON object") | **8/8** | n/a | n/a | n/a | n/a | $0.009 |
| [1](../data/baselines/run-1-structured/) | SUT: structured outputs (`EXTRACTION_SCHEMA`) | 0/8 | 87.5% | 87.5% | 100% | 100% | $0.013 |
| [2](../data/baselines/run-2-scope-rule/) | SUT prompt: explicit `scope` rule · golden: 2 cases fixed | 0/8 | 100% | 87.5% | 100% | 100% | $0.016 |

Dataset `dataset_sha256`: runs 0–1 `4410f00d…`, run 2 `21f9cfc8…` (computed after the fact: these
runs predate the field, and their frozen meta is left untouched).

What each run taught:

- **Run 0 — the harness caught a real SUT defect.** The model wrapped every reply in a markdown
  ```` ```json ```` fence despite the prompt; `json.loads` fails, so all 8 cases are `parse_error`.
  A production consumer would break the same way, so the fix belongs in the SUT, not in a more lenient
  grader (that would make the instrument lie).
- **Run 1 — form guaranteed, content measured.** With the schema enforced, every case is graded. The two
  misses turned out to be defects of the **spec and the golden**, not of the model: the golden inferred
  `scope` from the body in one case and refused to in two others, and one case's `type` was ambiguous
  (no prefix; the text reads as a fix, the golden said `feat`). The `scope` judge accepted
  `authentication` ≈ `auth`, which exact-match would have failed.
- **Run 2 — rule written down, golden made consistent.** `scope` = explicit title scope → else inferred
  only if the text names a concrete part of the system → else `null` (a scope that only repeats the
  `type` is `null`). The remaining miss, `fix-inferred-scope` ("config file is missing" → golden
  `config`, model `null`), is a borderline case where the model said `null` in 2 of 3 runs. With one
  sample per case that can't be told apart from noise, and tuning the prompt until it passes would be
  overfitting to 8 cases.

Known limits of this baseline: 8 hand-written cases, 1 sample per case (no variance reported), judge not
validated against human labels (C-D3). Prompt examples are deliberately **not** taken from the eval set.
