"""BEP (J8) wiring tests: the flag-gated batched-edit divergence in
src/graph/helpers.py (_maybe_batch_edit_turn). The core BEP-1/4/5 modules are
tested separately (test_batch_edit*.py); these cover the _execute_turn hook:

  * flag OFF  → returns None (provably zero behavior change), even with a valid
    patchset present;
  * flag ON, no/ malformed patchset → returns None (fall through to REPL);
  * flag ON, valid patchset → sandbox-apply → py_compile verify → promote →
    terminal 4-tuple (is_final=True), file actually written;
  * flag ON, patchset that fails verify → nudge 4-tuple, live tree UNTOUCHED.

_batch_edit_repo_root is monkeypatched to a tmp repo in every test so the apply
can never touch the real project tree.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.batch_edit import sha256_text
from src.graph import helpers as H


def _ctx() -> SimpleNamespace:
    """Minimal Ctx: _maybe_batch_edit_turn only touches ctx.state.turns."""
    return SimpleNamespace(state=SimpleNamespace(turns=1))


def _wrap(json_body: str) -> str:
    return f"Reasoning done; here is the patch.\n\n```patchset\n{json_body}\n```\n"


@pytest.fixture(autouse=True)
def _no_session_record(monkeypatch):
    # _record_session_turn needs a full TaskState; stub it for these unit tests.
    monkeypatch.setattr(H, "_record_session_turn", lambda *a, **k: None)


def _set_flag(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(
        "src.features.features", lambda: SimpleNamespace(batch_edit_mode=on)
    )


def _point_repo(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(H, "_batch_edit_repo_root", lambda: Path(root))


async def _run(raw: str) -> tuple | None:
    return await H._maybe_batch_edit_turn(_ctx(), "coder_escalation", raw)


# ─── flag OFF: provably zero behavior change ─────────────────────────────────────

@pytest.mark.asyncio
async def test_flag_off_returns_none_even_with_valid_patchset(monkeypatch, tmp_path):
    _set_flag(monkeypatch, False)
    _point_repo(monkeypatch, tmp_path)
    body = '{"files": [{"path": "x.py", "operation": "create", "new_content": "y = 1\\n"}]}'
    assert await _run(_wrap(body)) is None
    assert not (tmp_path / "x.py").exists()  # nothing applied


# ─── flag ON but nothing to do → fall through ────────────────────────────────────

@pytest.mark.asyncio
async def test_flag_on_no_patchset_block_returns_none(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    assert await _run('FINAL("the answer is 42")') is None


@pytest.mark.asyncio
async def test_flag_on_malformed_block_returns_none(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    # modify without base_content_sha256 → parse raises ValueError → fall through
    body = '{"files": [{"path": "m.py", "operation": "modify", "hunks": [{"start_line": 1, "end_line": 1, "replacement": "x"}]}]}'
    assert await _run(_wrap(body)) is None


# ─── flag ON, valid patchset → apply + verify + promote ──────────────────────────

@pytest.mark.asyncio
async def test_valid_patchset_applies_verifies_promotes(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    body = '{"files": [{"path": "added.py", "operation": "create", "new_content": "VALUE = 7\\n"}]}'

    result = await _run(_wrap(body))

    assert result is not None
    output, error, is_final, artifacts = result
    assert error is None
    assert is_final is True
    assert "added.py" in output
    assert artifacts["_batch_edit"]["files"] == ["added.py"]
    assert artifacts["_batch_edit"]["verified"] is True
    # promoted to the (tmp) live tree with the exact content
    assert (tmp_path / "added.py").read_text() == "VALUE = 7\n"


@pytest.mark.asyncio
async def test_modify_patchset_applies(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    original = "a = 1\nb = 2\nc = 3\n"
    (tmp_path / "f.py").write_text(original)
    body = (
        '{"files": [{"path": "f.py", "operation": "modify", '
        f'"base_content_sha256": "{sha256_text(original)}", '
        '"hunks": [{"start_line": 2, "end_line": 2, "replacement": "b = 22\\n"}]}]}'
    )
    result = await _run(_wrap(body))
    assert result is not None and result[2] is True
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 22\nc = 3\n"


# ─── flag ON, verify failure → nudge, live tree UNTOUCHED ─────────────────────────

@pytest.mark.asyncio
async def test_verify_failure_does_not_promote(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    # syntactically invalid Python → py_compile verify fails → no promotion
    body = '{"files": [{"path": "broken.py", "operation": "create", "new_content": "def (:\\n"}]}'

    result = await _run(_wrap(body))

    assert result is not None
    output, error, is_final, meta = result
    assert is_final is False          # not terminal — model should retry
    assert "_nudge" in meta           # nudged to re-emit
    assert not (tmp_path / "broken.py").exists()  # live tree untouched


@pytest.mark.asyncio
async def test_verify_command_uses_full_tree_sandbox(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    (tmp_path / "untouched.py").write_text("KEEP = 1\n")
    monkeypatch.setenv(
        "ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD",
        "test -f added.py && test -f untouched.py",
    )
    body = '{"files": [{"path": "added.py", "operation": "create", "new_content": "VALUE = 7\\n"}]}'

    result = await _run(_wrap(body))

    assert result is not None
    assert result[2] is True
    assert (tmp_path / "added.py").read_text() == "VALUE = 7\n"
    assert "py_compile + owner command passed" in result[0]
    assert result[3]["_batch_edit"]["verifier"] == "py_compile + owner command passed"


@pytest.mark.asyncio
async def test_verifier_label_uses_the_command_snapshot(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", "true")
    from src import batch_edit_runner

    apply_real = batch_edit_runner.apply_patchset_sandboxed

    def change_env_then_apply(*args, **kwargs):
        # The invocation already captured `true`; changing ambient env now must
        # not change which command runs or the label attached to its result.
        monkeypatch.setenv("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", "exit 7")
        return apply_real(*args, **kwargs)

    monkeypatch.setattr(batch_edit_runner, "apply_patchset_sandboxed", change_env_then_apply)
    body = '{"files": [{"path": "snapshot.py", "operation": "create", "new_content": "OK = 1\\n"}]}'

    result = await _run(_wrap(body))

    assert result is not None and result[2] is True
    assert result[3]["_batch_edit"]["verifier"] == "py_compile + owner command passed"


@pytest.mark.asyncio
async def test_syntax_only_success_is_not_described_as_acceptance_verified(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    monkeypatch.delenv("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", raising=False)
    body = '{"files": [{"path": "syntax.py", "operation": "create", "new_content": "OK = 1\\n"}]}'

    result = await _run(_wrap(body))

    assert result is not None and result[2] is True
    assert "syntax-only (py_compile)" in result[0]
    assert "acceptance verified" not in result[0].lower()
    assert result[3]["_batch_edit"]["verifier"] == "syntax-only (py_compile)"


@pytest.mark.asyncio
async def test_verify_command_failure_does_not_promote(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", "exit 7")
    body = '{"files": [{"path": "added.py", "operation": "create", "new_content": "VALUE = 7\\n"}]}'

    result = await _run(_wrap(body))

    assert result is not None
    assert result[2] is False
    assert not (tmp_path / "added.py").exists()


@pytest.mark.asyncio
async def test_stale_base_does_not_promote(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    (tmp_path / "f.py").write_text("real = 1\n")
    stale_sha = sha256_text("STALE\n")
    # base_content_sha256 references different content → stale-base rejection
    body = (
        '{"files": [{"path": "f.py", "operation": "modify", '
        f'"base_content_sha256": "{stale_sha}", '
        '"hunks": [{"start_line": 1, "end_line": 1, "replacement": "hacked = 1\\n"}]}]}'
    )
    result = await _run(_wrap(body))
    assert result is not None and result[2] is False  # nudge, not terminal
    assert (tmp_path / "f.py").read_text() == "real = 1\n"  # unchanged


# ─── BEP-1c telemetry: malformed != absent (hard gate) ────────────────────────────

@pytest.mark.asyncio
async def test_telemetry_distinguishes_absent_vs_malformed(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    H._BATCH_EDIT_STATE_COUNTS.clear()
    await _run('FINAL("done")')  # no patchset block → absent
    bad = '{"files": [{"path": "m.py", "operation": "modify", "hunks": [{"start_line": 1, "end_line": 1, "replacement": "x"}]}]}'
    await _run(_wrap(bad))       # block present, invalid (modify w/o base sha) → malformed
    assert H._BATCH_EDIT_STATE_COUNTS.get("absent") == 1
    assert H._BATCH_EDIT_STATE_COUNTS.get("malformed") == 1   # DISTINCT signal, not collapsed


@pytest.mark.asyncio
async def test_telemetry_records_applied_and_verify_failed(monkeypatch, tmp_path):
    _set_flag(monkeypatch, True)
    _point_repo(monkeypatch, tmp_path)
    H._BATCH_EDIT_STATE_COUNTS.clear()
    good = '{"files": [{"path": "ok.py", "operation": "create", "new_content": "V = 7\\n"}]}'
    await _run(_wrap(good))
    bad_py = '{"files": [{"path": "broken.py", "operation": "create", "new_content": "def (:\\n"}]}'
    await _run(_wrap(bad_py))    # py_compile verify fails
    assert H._BATCH_EDIT_STATE_COUNTS.get("applied") == 1
    assert H._BATCH_EDIT_STATE_COUNTS.get("verify_failed") == 1
