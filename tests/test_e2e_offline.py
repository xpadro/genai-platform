"""End-to-end offline test: dataset -> runner -> reload -> graders -> report.

Exercises the WIRING with the offline doubles (FakeSUT / FakeJudge), not model
quality. The fakes are deterministic, so the report numbers are exact and
derivable by hand from the cases below.

What unit tests can't see and this test pins:
  - N->N through DISK: N cases in -> N frozen lines -> N grades -> n_total == N.
  - The reload seam: grading reads the frozen jsonl, not in-memory outputs.
  - Output isolation: everything lands under the given out dir (tmp_path).

Known limit (deliberate): run_eval builds its own clients and FakeSUT always
returns valid JSON, so parse_error / sut_error paths are NOT exercised here —
they're covered by test_graders.py and test_report.py.
"""

import json

import pytest

from llm_evals.__main__ import run_eval
from llm_evals.runner import dataset_sha256

# FakeSUT returns this SAME extraction for every case:
#   {"type": "feat", "scope": "auth", "breaking_change": False, "issues": []}
# FakeJudge ALWAYS answers "VERDICT: correct" — but grade_scope only calls it
# when neither short-circuit applies. Each case below targets one scope path.
CASES = [
    {
        # scope: exact-match short-circuit (judge NOT called) -> True
        "id": "scope-exact",
        "input": "feat(auth): add OAuth2 login flow",
        "golden": {
            "type": "feat",  # == feat -> True
            "scope": "auth",  # == auth -> True (short-circuit)
            "breaking_change": False,  # == False -> True
            "issues": [],  # set() == set() -> True
        },
    },
    {
        # scope: judge path ('auth' vs 'ui') -> FakeJudge says correct -> True
        # (a false positive by construction: it's the fake, not a real verdict)
        "id": "scope-judge",
        "input": "fix(ui): correct button alignment\n\nCloses PROJ-456",
        "golden": {
            "type": "fix",  # != feat -> False
            "scope": "ui",  # judge called -> True
            "breaking_change": False,  # == False -> True
            "issues": ["PROJ-456"],  # {} != {PROJ-456} -> False
        },
    },
    {
        # scope: null short-circuit (golden None, predicted 'auth') -> False
        "id": "scope-null",
        "input": "chore!: drop Python 3.9 support\n\nBREAKING CHANGE: 3.10+. Closes PROJ-9",
        "golden": {
            "type": "chore",  # != feat -> False
            "scope": None,  # one null -> False (short-circuit)
            "breaking_change": True,  # != False -> False
            "issues": ["PROJ-9"],  # -> False
        },
    },
]

# Computed BY HAND from the per-field comments above (3 graded cases).
EXPECTED_FIELD_ACCURACY = {
    "type": 1 / 3,
    "scope": 2 / 3,
    "breaking_change": 2 / 3,
    "issues": 1 / 3,
}


def test_offline_pipeline_reports_exact_accuracy_and_freezes_one_line_per_case(tmp_path):
    # ARRANGE — the dataset also goes through disk, like a real run.
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(json.dumps(case) for case in CASES), encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    # ACT
    report = run_eval(cases_path, out_dir, fake=True)

    # ASSERT 1 — returned report: exact numbers (approx for floats).
    summary = report["summary"]
    assert summary["n_total"] == len(CASES)
    assert summary["n_graded"] == len(CASES)
    assert summary["outcomes"] == {"graded": len(CASES)}
    assert summary["field_accuracy"] == pytest.approx(EXPECTED_FIELD_ACCURACY)

    # ASSERT 2 — disk: exactly one frozen run, with N lines (N->N invariant),
    # one record per case id, in order.
    jsonls = list((out_dir / "generations").glob("*.jsonl"))
    assert len(jsonls) == 1
    lines = jsonls[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(CASES)
    assert [json.loads(line)["case_id"] for line in lines] == [c["id"] for c in CASES]

    # ...plus its meta, and exactly one report sharing the same run_id.
    run_id = jsonls[0].stem
    meta_path = out_dir / "generations" / f"{run_id}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dataset"] == str(cases_path)
    # The dataset's identity is its CONTENT, not its path: edit the golden in
    # place and the hash changes, committed or not.
    assert meta["dataset_sha256"] == dataset_sha256(cases_path)
    report_path = out_dir / "reports" / f"{run_id}.json"
    assert report_path.is_file()
    assert len(list((out_dir / "reports").glob("*.json"))) == 1

    # ASSERT 3 — the report on disk IS the returned report (persisted, diffable).
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
