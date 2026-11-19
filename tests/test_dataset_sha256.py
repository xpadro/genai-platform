"""The dataset fingerprint identifies CONTENT: stable across platforms, sensitive to edits."""

from llm_evals.runner import dataset_sha256

LINES = ['{"id": "a", "golden": {"scope": null}}', '{"id": "b", "golden": {"scope": "db"}}']


def test_same_content_same_hash_regardless_of_line_endings(tmp_path):
    """git checks out CRLF on Windows and LF on CI: same commit, same fingerprint."""
    lf, crlf = tmp_path / "lf.jsonl", tmp_path / "crlf.jsonl"
    lf.write_bytes("\n".join(LINES).encode())
    crlf.write_bytes("\r\n".join(LINES).encode())

    assert dataset_sha256(lf) == dataset_sha256(crlf)


def test_editing_the_golden_changes_the_hash(tmp_path):
    """The run 1 -> run 2 case: golden edited in place, same path."""
    path = tmp_path / "eval_set.jsonl"
    path.write_text("\n".join(LINES), encoding="utf-8")
    before = dataset_sha256(path)

    path.write_text("\n".join(LINES).replace('"scope": null', '"scope": "config"'), encoding="utf-8")

    assert dataset_sha256(path) != before
