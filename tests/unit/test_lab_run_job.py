from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path
import sys

import pytest
import yaml

from scripts.lab import run_job


def _write_jobs_file(path: Path, *, enabled: bool = False, gated: bool = False) -> None:
    job = {
        "job_id": "sample_shadow",
        "title": "Sample shadow job",
        "stage": "shadow",
        "enabled": enabled,
        "risk": "write_reviewed",
        "model_role": "worker_explore",
        "input_spec": {
            "sources": [{"repo": "epyc-orchestrator", "path": "docs/source.md"}],
            "context_budget_tokens": 100,
            "forbidden_actions": ["edit_directly"],
        },
        "output_contract": {
            "format": "json",
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id", "run_id", "generated_at", "summary", "items"],
                "properties": {
                    "job_id": {"const": "sample_shadow"},
                    "run_id": {"type": "string"},
                    "generated_at": {"type": "string"},
                    "summary": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
    if gated:
        job["gates"] = ["frontier-f5-intake-injection-hardening"]
    doc = {
        "version": 1,
        "schema_version": "lab_jobs.v1",
        "policy": {
            "review_queue": "orchestration/lab_review_queue/",
            "direct_repo_writes_allowed": False,
            "default_stage": "shadow",
        },
        "jobs": [job],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _write_dcp_jobs_file(path: Path) -> None:
    _write_jobs_file(path, enabled=True)
    doc = yaml.safe_load(path.read_text())
    doc["jobs"][0]["input_spec"]["context_modes"] = ["dcp_pack", "source_excerpt"]
    doc["jobs"][0]["input_spec"]["max_bundle_files"] = 4
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _write_kb_jobs_file(path: Path, *, index_dir: Path | None = None) -> None:
    _write_jobs_file(path, enabled=True)
    doc = yaml.safe_load(path.read_text())
    doc["jobs"][0]["input_spec"]["context_modes"] = ["kb_rag", "source_excerpt"]
    doc["jobs"][0]["input_spec"]["kb_queries"] = ["freshness lint handoff"]
    doc["jobs"][0]["input_spec"]["kb_top_k"] = 2
    doc["jobs"][0]["input_spec"]["kb_index_dir"] = str(index_dir or path.parent / "kb-index")
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _write_catalog(index_dir: Path, *, with_dependency: bool) -> tuple[Path, dict | None]:
    index_dir.mkdir(parents=True, exist_ok=True)
    catalog = index_dir / "catalog.sqlite"
    conn = sqlite3.connect(catalog)
    try:
        conn.executescript(run_job.kb_rag._CATALOG_SCHEMA)
        conn.execute(
            "INSERT INTO chunk (chunk_id, file_path, heading_path, line_start, line_end, "
            "content_hash, mtime, emb_path, text_preview, token_count) "
            "VALUES (1, ?, '[]', 1, 1, 'fixture-hash', 1, '', 'fixture evidence', 1)",
            (str(index_dir / "evidence.md"),),
        )
        conn.commit()
    finally:
        conn.close()
    if not with_dependency:
        return catalog, None
    conn = sqlite3.connect(catalog)
    with run_job.kb_catalog_dependency.writer_connection(conn):
        record = run_job.kb_catalog_dependency.commit_completed_writer(
            conn, "build_index", loaded=None
        )
    return catalog, record


def _write_native_catalog(index_dir: Path) -> tuple[Path, dict]:
    catalog, record = _write_catalog(index_dir, with_dependency=True)
    assert record is not None
    return catalog, record


def _write_legacy_catalog(index_dir: Path) -> Path:
    catalog, record = _write_catalog(index_dir, with_dependency=False)
    assert record is None
    return catalog


def _write_command_jobs_file(path: Path, *, risk: str = "read_only") -> None:
    job = {
        "job_id": "command_shadow",
        "title": "Command shadow job",
        "stage": "shadow",
        "enabled": True,
        "risk": risk,
        "runtime_class": "active_safe_deterministic",
        "active_safe": True,
        "execution": {
            "mode": "deterministic_command",
            "command": [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'job_id': 'command_shadow', 'status': 'ok'}))"
                ),
            ],
        },
        "input_spec": {"sources": []},
        "output_contract": {
            "format": "json",
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id", "status"],
                "properties": {
                    "job_id": {"const": "command_shadow"},
                    "status": {"const": "ok"},
                },
            },
        },
    }
    doc = {
        "version": 1,
        "schema_version": "lab_jobs.v1",
        "policy": {
            "review_queue": "orchestration/lab_review_queue/",
            "direct_repo_writes_allowed": False,
            "default_stage": "shadow",
        },
        "jobs": [job],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _args(tmp_path: Path, jobs_file: Path, **overrides) -> Namespace:
    values = {
        "job_id": "sample_shadow",
        "jobs_file": str(jobs_file),
        "repo_root": str(tmp_path),
        "queue_dir": str(tmp_path / "queue"),
        "repo_map": [f"epyc-orchestrator={tmp_path}"],
        "allow_disabled": True,
        "allow_gated": False,
        "run_id": "sample-run",
        "max_context_chars": 1000,
        "dry_run_stub": True,
        "response_fixture": None,
        "execute_chat": False,
        "api_url": "http://127.0.0.1:8000",
        "timeout_s": 1.0,
        "print_output": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_dry_run_stub_writes_review_artifacts(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("# source\nEvidence.\n")
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)

    result = run_job.run_from_args(_args(tmp_path, jobs_file))

    output = json.loads(result.output_path.read_text())
    assert output["job_id"] == "sample_shadow"
    assert output["run_id"] == "sample-run"
    task_record = json.loads(result.task_record_path.read_text())
    assert task_record["record_type"] == "task_record"
    assert task_record["invocation_mode"] == "dry_run_contract_stub"
    assert task_record["validation"]["output_contract"] == "passed"
    assert task_record["context"]["schema_version"] == "lab_context_summary.v1"
    assert task_record["context"]["source_count"] == 1
    assert task_record["context"]["missing_source_count"] == 0
    assert task_record["context"]["repos"] == ["epyc-orchestrator"]
    assert task_record["context"]["kinds"] == {"source_excerpt": 1}
    context_manifest = json.loads((result.output_path.parent / "context_manifest.json").read_text())
    assert context_manifest["summary"] == task_record["context"]
    assert context_manifest["kb_catalog_dependency_at_context_collection"] is None
    assert "source.md" in (result.output_path.parent / "prompt.txt").read_text()
    rows = [json.loads(line) for line in result.task_record_log.read_text().splitlines()]
    assert rows[-1]["run_id"] == "sample-run"


def test_disabled_job_requires_explicit_allow_disabled(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("source")
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file, enabled=False)

    with pytest.raises(run_job.LabRunnerError, match="disabled"):
        run_job.run_from_args(_args(tmp_path, jobs_file, allow_disabled=False))


def test_gated_job_requires_explicit_allow_gated(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("source")
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file, enabled=True, gated=True)

    with pytest.raises(run_job.LabRunnerError, match="gated"):
        run_job.run_from_args(_args(tmp_path, jobs_file, allow_disabled=False))


def test_response_fixture_must_satisfy_contract(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("source")
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file, enabled=True)

    bad_fixture = tmp_path / "bad.json"
    bad_fixture.write_text(json.dumps({"job_id": "sample_shadow"}))
    args = _args(
        tmp_path,
        jobs_file,
        allow_disabled=False,
        dry_run_stub=False,
        response_fixture=str(bad_fixture),
    )

    with pytest.raises(run_job.LabRunnerError, match="output contract failed"):
        run_job.run_from_args(args)


def test_dcp_context_mode_packs_declared_sources_with_fallback_available(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("# source\nEvidence.\n")
    (tmp_path / "docs" / "helper.py").write_text(
        "def useful_helper(value: str) -> str:\n"
        "    \"\"\"Small helper.\"\"\"\n"
        "    return value.upper()\n"
    )
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_dcp_jobs_file(jobs_file)

    result = run_job.run_from_args(
        _args(
            tmp_path,
            jobs_file,
            allow_disabled=False,
            max_context_chars=200,
        )
    )

    task_record = json.loads(result.task_record_path.read_text())
    assert task_record["context"]["source_count"] >= 1
    assert any(kind.startswith("dcp_pack:") for kind in task_record["context"]["kinds"])
    manifest = json.loads((result.output_path.parent / "context_manifest.json").read_text())
    assert manifest["summary"] == task_record["context"]


def test_kb_rag_context_mode_adds_retrieved_snippets(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("# source\nEvidence.\n")
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_kb_jobs_file(jobs_file)

    def fake_query(text: str, *, top_k: int, index_dir) -> list[dict]:
        assert text == "freshness lint handoff"
        assert top_k == 2
        return [
            {
                "file": "/workspace/wiki/autonomous-research.md",
                "heading_path": ["AutoPilot"],
                "line_range": (22, 24),
                "snippet": "AutoPilot may suggest closure candidates but not archive handoffs.",
                "score": 0.91,
                "content_hash": "abc123",
            }
        ]

    monkeypatch.setattr(run_job.kb_rag, "query", fake_query)

    result = run_job.run_from_args(
        _args(
            tmp_path,
            jobs_file,
            allow_disabled=False,
            max_context_chars=1000,
        )
    )

    task_record = json.loads(result.task_record_path.read_text())
    assert task_record["context"]["kinds"] == {"kb_rag": 1}
    assert task_record["context"]["repos"] == ["kb-rag"]
    manifest = json.loads((result.output_path.parent / "context_manifest.json").read_text())
    assert manifest["sources"][0]["path"].endswith("autonomous-research.md:22-24")
    assert manifest["kb_catalog_dependency_at_context_collection"] == {
        "catalog_path": str((tmp_path / "kb-index" / "catalog.sqlite").resolve()),
        "record": None,
    }
    prompt = (result.output_path.parent / "prompt.txt").read_text()
    assert "AutoPilot may suggest closure candidates" in prompt


@pytest.mark.parametrize("query_outcome", ["empty", "exception"])
def test_kb_rag_context_mode_falls_back_to_sources_on_query_miss(
    tmp_path: Path,
    monkeypatch,
    query_outcome: str,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text("# source\nEvidence.\n")
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_kb_jobs_file(jobs_file)
    if query_outcome == "empty":
        monkeypatch.setattr(run_job.kb_rag, "query", lambda *args, **kwargs: [])
    else:
        def fail_query(*args, **kwargs):
            raise RuntimeError("synthetic query failure")

        monkeypatch.setattr(run_job.kb_rag, "query", fail_query)

    result = run_job.run_from_args(
        _args(
            tmp_path,
            jobs_file,
            allow_disabled=False,
            max_context_chars=1000,
        )
    )

    task_record = json.loads(result.task_record_path.read_text())
    assert task_record["context"]["kinds"] == {"source_excerpt": 1}
    manifest = json.loads((result.output_path.parent / "context_manifest.json").read_text())
    expected_reason = (
        "kb_rag_no_results"
        if query_outcome == "empty"
        else "kb_rag_error:synthetic query failure"
    )
    assert manifest["missing_sources"] == [
        {"repo": "kb-rag", "path": "freshness lint handoff", "reason": expected_reason}
    ]
    assert manifest["kb_catalog_dependency_at_context_collection"]["record"] is None


@pytest.mark.parametrize("corruption", ["record_seal", "logical_catalog"])
@pytest.mark.parametrize("backend", ["chat", "command"])
def test_invalid_kb_catalog_dependency_refuses_before_query_backend_or_publication(
    tmp_path: Path,
    monkeypatch,
    corruption: str,
    backend: str,
) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_kb_jobs_file(jobs_file)
    if backend == "command":
        jobs_doc = yaml.safe_load(jobs_file.read_text())
        jobs_doc["jobs"][0]["risk"] = "read_only"
        jobs_doc["jobs"][0]["execution"] = {
            "mode": run_job.DETERMINISTIC_COMMAND_MODE,
            "command": [sys.executable, "-c", "raise SystemExit(0)"],
        }
        jobs_file.write_text(yaml.safe_dump(jobs_doc, sort_keys=False))
    catalog, _ = _write_native_catalog(tmp_path / "kb-index")
    with sqlite3.connect(catalog) as conn:
        if corruption == "record_seal":
            conn.execute(
                "UPDATE catalog_dependency SET record_json = record_json || ' ' WHERE singleton=1"
            )
        else:
            conn.execute(
                "UPDATE chunk SET content_hash = 'changed-after-record' WHERE chunk_id=1"
            )

    events: list[str] = []
    read_dependency = run_job.kb_catalog_dependency.read_dependency

    def observe_dependency(path):
        events.append("read_dependency")
        return read_dependency(path)

    def forbidden_query(*args, **kwargs):
        events.append("query")
        raise AssertionError("KB query must not run after dependency refusal")

    def forbidden_chat(**kwargs):
        events.append("chat")
        raise AssertionError("chat backend must not run after dependency refusal")

    def forbidden_command(**kwargs):
        events.append("command")
        raise AssertionError("command backend must not run after dependency refusal")

    monkeypatch.setattr(run_job.kb_catalog_dependency, "read_dependency", observe_dependency)
    monkeypatch.setattr(run_job.kb_rag, "query", forbidden_query)
    monkeypatch.setattr(run_job, "call_chat_api", forbidden_chat)
    monkeypatch.setattr(run_job, "run_deterministic_command", forbidden_command)

    with pytest.raises(run_job.LabRunnerError, match="KB catalog dependency refused"):
        run_job.run_from_args(
            _args(
                tmp_path,
                jobs_file,
                allow_disabled=False,
                dry_run_stub=False,
                execute_chat=backend == "chat",
                execute_command=backend == "command",
            )
        )

    assert not (tmp_path / "queue").exists()
    assert events == ["read_dependency"]


def test_valid_kb_catalog_dependency_is_attached_at_collection_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_kb_jobs_file(jobs_file)
    _, expected_record = _write_native_catalog(tmp_path / "kb-index")
    events: list[str] = []
    read_dependency = run_job.kb_catalog_dependency.read_dependency

    def observe_dependency(path):
        events.append("read_dependency")
        return read_dependency(path)

    def fake_query(*args, **kwargs):
        events.append("query")
        return [
            {
                "file": "/fixture/kb.md",
                "line_range": (1, 2),
                "snippet": "synthetic KB context",
                "content_hash": "fixture-hash",
            }
        ]

    monkeypatch.setattr(run_job.kb_catalog_dependency, "read_dependency", observe_dependency)
    monkeypatch.setattr(run_job.kb_rag, "query", fake_query)

    result = run_job.run_from_args(
        _args(tmp_path, jobs_file, allow_disabled=False, max_context_chars=1000)
    )

    manifest = json.loads((result.output_path.parent / "context_manifest.json").read_text())
    assert manifest["kb_catalog_dependency_at_context_collection"] == {
        "catalog_path": str((tmp_path / "kb-index" / "catalog.sqlite").resolve()),
        "record": expected_record,
    }
    assert events == ["read_dependency", "query"]


def test_legacy_kb_catalog_has_unknown_dependency_at_collection_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_kb_jobs_file(jobs_file)
    _write_legacy_catalog(tmp_path / "kb-index")
    monkeypatch.setattr(run_job.kb_rag, "query", lambda *args, **kwargs: [])

    result = run_job.run_from_args(
        _args(tmp_path, jobs_file, allow_disabled=False, max_context_chars=1000)
    )

    manifest = json.loads((result.output_path.parent / "context_manifest.json").read_text())
    assert manifest["kb_catalog_dependency_at_context_collection"] == {
        "catalog_path": str((tmp_path / "kb-index" / "catalog.sqlite").resolve()),
        "record": None,
    }


def test_collect_context_keeps_two_value_return_shape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_job.kb_rag, "query", lambda *args, **kwargs: [])

    excerpts, missing = run_job.collect_context(
        {
            "context_modes": ["kb_rag"],
            "kb_queries": ["fixture query"],
            "kb_index_dir": str(tmp_path / "missing-index"),
        },
        {},
        max_context_chars=100,
    )

    assert excerpts == []
    assert missing == [
        {"repo": "kb-rag", "path": "fixture query", "reason": "kb_rag_no_results"}
    ]


def test_deterministic_command_writes_review_artifacts(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_command_jobs_file(jobs_file)

    result = run_job.run_from_args(
        _args(
            tmp_path,
            jobs_file,
            job_id="command_shadow",
            allow_disabled=False,
            dry_run_stub=False,
            execute_command=True,
        )
    )

    output = json.loads(result.output_path.read_text())
    assert output == {"job_id": "command_shadow", "status": "ok"}
    task_record = json.loads(result.task_record_path.read_text())
    assert task_record["invocation_mode"] == run_job.DETERMINISTIC_COMMAND_MODE
    assert task_record["risk"] == "read_only"
    assert task_record["chat_meta"]["returncode"] == 0
    assert task_record["chat_meta"]["command"][0] == sys.executable


def test_deterministic_command_requires_read_only_job(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_command_jobs_file(jobs_file, risk="write_reviewed")

    with pytest.raises(run_job.LabRunnerError, match="risk=read_only"):
        run_job.run_from_args(
            _args(
                tmp_path,
                jobs_file,
                job_id="command_shadow",
                allow_disabled=False,
                dry_run_stub=False,
                execute_command=True,
            )
        )


def test_call_chat_api_resolves_lab_role_and_requests_real_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "answer": json.dumps({"job_id": "sample_shadow"}),
                "mock_mode": False,
                "real_mode": True,
                "routed_to": "architect_general",
            }

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, json: dict):
            captured["url"] = url
            captured["payload"] = json
            return _Response()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)

    output, meta = run_job.call_chat_api(
        api_url="http://unit",
        role="verifier",
        prompt="return json",
        run_id="run-1",
        timeout_s=12.0,
    )

    assert output == {"job_id": "sample_shadow"}
    assert captured["url"] == "http://unit/chat"
    payload = captured["payload"]
    assert payload["force_role"] == "architect_general"
    assert payload["role"] == "architect_general"
    assert payload["mock_mode"] is False
    assert payload["real_mode"] is True
    assert meta["requested_role"] == "verifier"
    assert meta["resolved_role"] == "architect_general"


def test_call_chat_api_rejects_mock_response(monkeypatch) -> None:
    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "answer": "[MOCK] Processed prompt",
                "mock_mode": True,
                "real_mode": False,
            }

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, json: dict):
            return _Response()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)

    with pytest.raises(run_job.LabRunnerError, match="mock mode"):
        run_job.call_chat_api(
            api_url="http://unit",
            role="worker_explore",
            prompt="return json",
            run_id="run-1",
            timeout_s=12.0,
        )
