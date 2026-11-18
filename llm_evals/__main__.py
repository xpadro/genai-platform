"""CLI entry point: orchestrates the four pieces end-to-end.

    python -m llm_evals run --set data/eval_set.jsonl          # real API
    python -m llm_evals run --set data/eval_set.jsonl --fake   # offline smoke
    python -m llm_evals run --set data/eval_set.jsonl --out /tmp/evals  # custom output dir

This is the ONLY place the four pieces are wired together. Read run_eval() top
to bottom to see the whole pipeline:

    DATASET -> RUNNER -> [reload] -> GRADERS -> REPORT

Outputs go under a single base dir (`--out`, default `data`): frozen generations
in <out>/generations/ and reports in <out>/reports/. Relative paths resolve
against the current working directory, so run it from the repo root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_client import LLMClient

from .dataset import load_cases
from .graders import JUDGE_MODEL, grade_case
from .report import build_report, print_report, write_report
from .runner import load_generations, run


def _load_dotenv() -> None:
    """Load .env if python-dotenv is installed (dev extra); it's optional for the package."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()  # never overrides variables already set in the environment


def _clients(fake: bool) -> tuple[LLMClient, LLMClient]:
    """Return (sut_client, judge_client).

    Fake doubles offline; decoupled real clients otherwise (the judge is a
    different/superior model to the SUT — see graders.JUDGE_MODEL).
    """
    if fake:
        from ._offline import FakeJudge, FakeSUT

        return FakeSUT(), FakeJudge()

    _load_dotenv()
    from llm_client import ProductionLLMClient

    sut = ProductionLLMClient()  # default model = the SUT under evaluation
    judge = ProductionLLMClient(model=JUDGE_MODEL)
    return sut, judge


def run_eval(cases_path: str | Path, out_path: str | Path, fake: bool = False) -> dict:
    out_path = Path(out_path)

    # 1. DATASET — versioned cases (input + golden), read from disk.
    cases = load_cases(cases_path)
    print(f"loaded {len(cases)} cases from {cases_path}")

    sut, judge = _clients(fake)

    # 2. RUNNER — generate once and FREEZE the raw outputs to disk; return run_id.
    run_id = run(cases, sut, out_path)
    print(
        f"run_id = {run_id}  (frozen to {out_path / 'generations' / f'{run_id}.jsonl'})"
    )

    # 3. RELOAD — read the frozen run back into records. The runner->grader seam
    #    goes through DISK, which is what lets you re-grade without re-generating.
    #    (load_generations is YOUR CODE in runner.py.)
    records = load_generations(run_id, out_path)
    assert len(records) == len(cases), "N->N invariant broken on reload"

    # 4. GRADERS — score each record (code-based + judge); a bad SUT is a GRADE,
    #    not a crash (parse_error / sut_error).
    grades = [grade_case(rec, judge) for rec in records]

    # 5. REPORT — aggregate, persist (diffable), and print the headline.
    report = build_report(grades)
    path = write_report(report, run_id, out_path)
    print_report(report)
    print(f"report written to {path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm_evals")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the eval pipeline end-to-end")
    p_run.add_argument(
        "--set", dest="set_path", required=True, help="path to the eval set JSONL"
    )
    p_run.add_argument(
        "--fake", action="store_true", help="offline smoke test (no API spend)"
    )

    p_run.add_argument(
        "--out",
        dest="out_path",
        required=False,
        default=Path("data"),
        help="output directory",
        type=Path,
    )

    args = parser.parse_args()
    if args.command == "run":
        run_eval(args.set_path, args.out_path, fake=args.fake)


if __name__ == "__main__":
    main()
