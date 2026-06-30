# llm-client

A production-ready LLM client layer on top of the Anthropic SDK. Not a bare
`messages.create()` call: it's everything around it —streaming, retries, fallback,
timeouts and cost/token observability— that keeps a workflow from falling over
when the provider returns a 429/529 under load.

## Why it exists

Every production GenAI system rests on a reliable, observable client layer.
Without one, RAG, agents and evals are built on sand. This repo is the first
brick of a future *model gateway*.

## Running locally

### 1. Set up the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux  (Windows: .venv\Scripts\activate)
pip install -e ".[dev]"
```

> If `python` resolves to Python 2 on your machine (you'll see a
> `Non-ASCII character` SyntaxError), use `python3` or call the venv binary
> directly: `.venv/bin/python ...`.

### 2. Run the unit tests (no API cost)

```bash
python -m pytest -q                # 7 tests, SDK is mocked — no API calls, no spend
```

### 3. Real run (hits the API, costs a few cents)

Provide your key, either as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

…or in a `.env` file (already gitignored) that you load before running:

```bash
set -a && source .env && set +a
```

Then run the example. **If your shell sets `ANTHROPIC_BASE_URL`** (e.g. a corporate
proxy that intercepts the Anthropic SDK), override it to reach the public API
directly — otherwise your request is routed to that proxy and may be blocked:

```bash
ANTHROPIC_BASE_URL=https://api.anthropic.com python examples/real_run.py
```

If `ANTHROPIC_BASE_URL` is not set in your environment, you can drop that prefix.

## Contract

`ProductionLLMClient.complete()` **never** propagates a network/SDK exception
upward. It always returns an `LLMResult` with a `status`:

| status | meaning |
|---|---|
| `ok` | valid response (primary or fallback) |
| `error` | **permanent** error (400/401/403/404/422): the request must be fixed |
| `unavailable` | primary retries **and** fallback exhausted: graceful degradation |

## Design decisions & trade-offs

- **Only transient errors are retried** (429, 529, connection errors). Permanent
  ones fail fast: retrying them returns the same error and burns latency/money.
- **Fallback fires only on transient errors**, not permanent ones (another model
  won't fix a malformed prompt).
- **`max_retries` is configurable (default 3).** More retries = more resilience but
  worse tail latency (p99): each retry adds wait + a new call.
- **Explicit timeout (default 30s).** Short protects p99 but kills legitimately slow
  calls; long avoids false negatives but leaves the workflow hanging.
- **Cost uses dated prices** (`pricing.py`, verified 2026-06-30). Unknown model →
  cost 0 (we don't invent prices). Prices change: re-check them.

## Known risks (what it does NOT solve yet)

- **Falling back to a cheaper model can silently degrade quality.** For a *quality
  gate* (e.g. code review) this is dangerous: better `unavailable` + human review
  than a worse review treated as trustworthy. Per-task fallback policy is future work.
- **The trace logs the full `LLMResult`**: it must not include PII in `text` in a
  real environment (pending, C-E6).
- Cost is per *request*, not per *workflow*: aggregation by business unit is built
  on top (C-F1).

## How it's evaluated

- `tests/test_retries.py` — transient on primary → fallback → `ok`; primary ok marks
  `used_fallback=False`.
- `tests/test_degradation.py` — primary + fallback both fail → `unavailable` (no
  exception); permanent → `error` without trying the fallback.
- `tests/test_cost.py` — correct cost; cache read = 0.1× of base input; unknown
  model costs 0.
- CI (`.github/workflows/ci.yml`) runs the suite on every push/PR.

## Real run (end-to-end trace)

Output of `python examples/real_run.py` against `api.anthropic.com` (2026-06-30):

```text
INFO HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO trace {'text': 'Hola', 'model': 'claude-sonnet-4-6', 'status': 'ok',
            'input_tokens': 26, 'output_tokens': 6, 'cache_read': 0, 'cache_write': 0,
            'latency_s': 1.455, 'cost_usd': 0.000168, 'used_fallback': False, 'reason': None}
----------------------------------------
status     : ok
model      : claude-sonnet-4-6
text       : Hola
in/out tok : 26 / 6
latency_s  : 1.455
cost_usd   : 0.000168
```

The structured trace (tokens, per-call cost, latency, `used_fallback`) is the seed
of tracing (C-E1) and cost budgeting (C-F1).

## Next version

Per-task fallback policy, async for concurrency, real prompt-caching
instrumentation (C-A4), and PII redaction in the trace (C-E6).
