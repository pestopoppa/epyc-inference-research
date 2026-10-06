from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.kb_rag import eval_k7
from src.retrieval import colbert_encoder, kb_catalog_dependency, kb_rag


def _report_inputs(tmp_path: Path) -> tuple[Path, Path]:
    evidence = tmp_path / "evidence.md"
    evidence.write_text("# evidence\n")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "case_a",
                        "protocol": "offline_fixture",
                        "query": "where is the evidence?",
                        "evidence_files": [str(evidence)],
                    }
                ],
            }
        )
    )
    return cases_path, evidence


def _make_catalog(index_dir: Path, *, with_dependency: bool) -> tuple[Path, dict | None]:
    index_dir.mkdir(parents=True, exist_ok=True)
    catalog = index_dir / "catalog.sqlite"
    conn = sqlite3.connect(catalog)
    try:
        conn.executescript(kb_rag._CATALOG_SCHEMA)
        conn.execute(
            "INSERT INTO chunk (chunk_id, file_path, heading_path, line_start, line_end, "
            "content_hash, mtime, emb_path, text_preview, token_count) "
            "VALUES (1, ?, '[]', 1, 1, 'fixture-hash', 1, '', 'fixture evidence', 1)",
            (str(index_dir / "evidence.md"),),
        )
        conn.commit()
    finally:
        conn.close()
    record = None
    if with_dependency:
        conn = sqlite3.connect(catalog)
        with kb_catalog_dependency.writer_connection(conn):
            record = kb_catalog_dependency.commit_completed_writer(
                conn, "build_index", loaded=None
            )
    return catalog, record


def _run_offline_report(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cases_path: Path,
    evidence: Path,
    index_dir: Path,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    real_evaluate = eval_k7.evaluate
    query_calls: list[dict] = []

    def fake_query(text: str, **kwargs):
        query_calls.append({"text": text, **kwargs})
        return [{"file": str(evidence), "score": 1.0}]

    def offline_evaluate(*, cases, configs, index_dir, top_k, cutoffs):
        return real_evaluate(
            cases=cases,
            configs=configs,
            index_dir=index_dir,
            top_k=top_k,
            cutoffs=cutoffs,
            query_fn=fake_query,
        )

    monkeypatch.setattr(eval_k7, "evaluate", offline_evaluate)
    monkeypatch.setattr(
        colbert_encoder,
        "ensure_loaded",
        lambda: pytest.fail("report fixture reached the real encoder"),
    )
    summary = eval_k7.run_eval(
        cases_path=cases_path,
        index_dir=index_dir,
        output_dir=output_dir,
        configs="maxsim",
        cutoffs=(1, 2),
    )
    return summary, query_calls


def test_score_case_counts_each_file_once(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    c = tmp_path / "c.md"
    for path in (a, b, c):
        path.write_text("# doc\n")

    results = [
        {"file": str(c), "score": 0.9},
        {"file": str(a), "score": 0.8},
        {"file": str(a), "score": 0.7},
        {"file": str(b), "score": 0.6},
    ]

    scored = eval_k7.score_case([str(a), str(b)], results, cutoffs=(1, 3, 4))

    assert scored["found_count"] == 2
    assert scored["recall@1"] == 0.0
    assert scored["recall@3"] == 0.5
    assert scored["recall@4"] == 1.0
    assert scored["perfect@3"] is False
    assert scored["perfect@4"] is True
    assert scored["first_evidence_rank"] == 2
    assert scored["all_evidence_rank"] == 4


def test_evaluate_summarizes_config_and_protocol(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# a\n")
    b.write_text("# b\n")

    cases = [
        {
            "id": "c1",
            "protocol": "hotpotqa_template",
            "query": "alpha",
            "resolved_evidence_files": [str(a)],
        },
        {
            "id": "c2",
            "protocol": "locomo_template",
            "query": "beta",
            "resolved_evidence_files": [str(a), str(b)],
        },
    ]
    configs = [eval_k7.EvalConfig("maxsim"), eval_k7.EvalConfig("rerank_w0.3", rerank=True)]

    def fake_query(text: str, **kwargs):
        if text == "alpha":
            return [{"file": str(a), "score": 1.0}]
        return [{"file": str(a), "score": 1.0}, {"file": str(b), "score": 0.9}]

    rows = eval_k7.evaluate(
        cases=cases,
        configs=configs,
        index_dir=tmp_path / "idx",
        top_k=2,
        cutoffs=(1, 2),
        query_fn=fake_query,
    )
    summary = eval_k7.summarize_rows(rows, cases, configs, cutoffs=(1, 2))

    assert len(rows) == 4
    assert summary["case_count"] == 2
    assert summary["protocol_counts"] == {
        "hotpotqa_template": 1,
        "locomo_template": 1,
    }
    assert summary["configs"]["maxsim"]["overall"]["mean_recall@1"] == 0.75
    assert summary["configs"]["maxsim"]["overall"]["mean_recall@2"] == 1.0
    assert summary["configs"]["maxsim"]["overall"]["perfect@2"] == "2/2"


def test_load_cases_normalizes_seed_schema(tmp_path: Path) -> None:
    evidence = tmp_path / "e.md"
    evidence.write_text("# evidence\n")
    case_file = tmp_path / "cases.json"
    case_file.write_text(
        """
        {
          "version": 1,
          "cases": [
            {
              "id": "case_a",
              "protocol": "hotpotqa_template",
              "query": "Where is evidence?",
              "evidence_files": ["%s"]
            }
          ]
        }
        """
        % evidence
    )

    cases, metadata = eval_k7.load_cases(case_file)

    assert metadata["version"] == 1
    assert cases[0]["id"] == "case_a"
    assert cases[0]["resolved_evidence_files"] == [str(evidence.resolve())]


def test_run_eval_attaches_native_dependency_once_at_report_time(tmp_path, monkeypatch):
    cases_path, evidence = _report_inputs(tmp_path)
    index_dir = tmp_path / "index"
    catalog, record = _make_catalog(index_dir, with_dependency=True)
    before = catalog.read_bytes()
    read_dependency = eval_k7.read_catalog_dependency
    dependency_reads: list[Path] = []

    def tracked_dependency_read(path):
        dependency_reads.append(Path(path))
        return read_dependency(path)

    monkeypatch.setattr(eval_k7, "read_catalog_dependency", tracked_dependency_read)

    summary, query_calls = _run_offline_report(
        monkeypatch,
        cases_path=cases_path,
        evidence=evidence,
        index_dir=index_dir,
        output_dir=tmp_path / "report",
    )

    assert summary["catalog_dependency_at_report"] == record
    assert dependency_reads == [catalog]
    assert summary["case_count"] == 1
    assert summary["configs"]["maxsim"]["overall"]["mean_recall@1"] == 1.0
    assert len(query_calls) == 1
    assert catalog.read_bytes() == before
    report = tmp_path / "report"
    saved_summary = json.loads((report / "summary.json").read_text())
    rows = [json.loads(line) for line in (report / "rows.jsonl").read_text().splitlines()]
    cases_out = json.loads((report / "cases.json").read_text())
    assert saved_summary["catalog_dependency_at_report"] == record
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case_a"
    assert rows[0]["recall@1"] == 1.0
    assert rows[0]["perfect@1"] is True
    assert "catalog_dependency_at_report" not in rows[0]
    assert "catalog_dependency_at_report" not in cases_out
    assert cases_out["cases"][0]["id"] == "case_a"


def test_run_eval_missing_catalog_is_unknown_and_does_not_create_it(tmp_path, monkeypatch):
    cases_path, evidence = _report_inputs(tmp_path)
    index_dir = tmp_path / "missing-index"

    summary, query_calls = _run_offline_report(
        monkeypatch,
        cases_path=cases_path,
        evidence=evidence,
        index_dir=index_dir,
        output_dir=tmp_path / "report",
    )

    assert summary["catalog_dependency_at_report"] is None
    assert len(query_calls) == 1
    assert not index_dir.exists()


def test_run_eval_legacy_catalog_is_unknown_without_migration_or_ddl(tmp_path, monkeypatch):
    cases_path, evidence = _report_inputs(tmp_path)
    index_dir = tmp_path / "index"
    catalog, _ = _make_catalog(index_dir, with_dependency=False)
    before = catalog.read_bytes()

    summary, _ = _run_offline_report(
        monkeypatch,
        cases_path=cases_path,
        evidence=evidence,
        index_dir=index_dir,
        output_dir=tmp_path / "report",
    )

    assert summary["catalog_dependency_at_report"] is None
    assert catalog.read_bytes() == before
    with sqlite3.connect(catalog) as conn:
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='catalog_dependency'"
        ).fetchone()


@pytest.mark.parametrize("tamper", ["seal", "malformed_json", "contract", "singleton"])
def test_run_eval_refuses_invalid_native_dependency_before_writing_reports(
    tmp_path, monkeypatch, tamper
):
    cases_path, evidence = _report_inputs(tmp_path)
    index_dir = tmp_path / "index"
    catalog, _ = _make_catalog(index_dir, with_dependency=True)
    with sqlite3.connect(catalog) as conn:
        if tamper == "seal":
            conn.execute("UPDATE catalog_dependency SET record_json='{}'")
        elif tamper == "malformed_json":
            body = "{"
            conn.execute(
                "UPDATE catalog_dependency SET record_json=?, record_sha256=?",
                (body, hashlib.sha256(body.encode()).hexdigest()),
            )
        elif tamper == "contract":
            record = json.loads(
                conn.execute("SELECT record_json FROM catalog_dependency").fetchone()[0]
            )
            record["unexpected"] = True
            body = json.dumps(record, sort_keys=True, separators=(",", ":"))
            conn.execute(
                "UPDATE catalog_dependency SET record_json=?, record_sha256=?",
                (body, hashlib.sha256(body.encode()).hexdigest()),
            )
        else:
            conn.execute("PRAGMA ignore_check_constraints=ON")
            conn.execute("UPDATE catalog_dependency SET singleton=2")

    output_dir = tmp_path / "report"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _run_offline_report(
            monkeypatch,
            cases_path=cases_path,
            evidence=evidence,
            index_dir=index_dir,
            output_dir=output_dir,
        )
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "rows.jsonl").exists()
    assert not (output_dir / "cases.json").exists()


def test_run_eval_refuses_stale_logical_catalog_without_rewriting_it(tmp_path, monkeypatch):
    cases_path, evidence = _report_inputs(tmp_path)
    index_dir = tmp_path / "index"
    catalog, _ = _make_catalog(index_dir, with_dependency=True)
    with sqlite3.connect(catalog) as conn:
        conn.execute("UPDATE chunk SET text_preview='changed after capture'")
    before = catalog.read_bytes()
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="content no longer matches"):
        _run_offline_report(
            monkeypatch,
            cases_path=cases_path,
            evidence=evidence,
            index_dir=index_dir,
            output_dir=output_dir,
        )

    assert catalog.read_bytes() == before
    assert not (output_dir / "summary.json").exists()
