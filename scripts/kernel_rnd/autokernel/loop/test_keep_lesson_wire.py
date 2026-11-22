"""AK-PM-12: run.py's keep path calls keep_lesson; idempotent, never raises."""
from autokernel.journal import Journal
from autokernel.loop import run

ROW = {"status": "kept", "mechanism_id": "akm-fuse-x", "statement": "fuse the epilogue",
       "falsifier": "decode tok/s does not rise", "target_surface": "ggml-cpu.c",
       "champion_head": "a" * 40}


def _receipts(store):
    j = Journal(str(store / run.KEEP_LESSON_JOURNAL))
    return [e for e in j.read_all()
            if (e.payload or {}).get("schema") == "autokernel.lesson_admission.v1"]


def test_keep_records_once_and_is_idempotent(tmp_path):
    first = run.submit_keep_lesson(tmp_path, dict(ROW))
    assert first["recorded"] is True
    again = run.submit_keep_lesson(tmp_path, dict(ROW))
    assert again == {"recorded": False, "reason": "already_recorded", "decision": None}
    assert len(_receipts(tmp_path)) == 1


def test_journal_or_producer_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    out = run.submit_keep_lesson(blocker, dict(ROW))
    assert out["recorded"] is False and out["reason"].startswith("journal_unavailable")

    def boom(*_a, **_k):
        raise RuntimeError("nope")
    out = run.submit_keep_lesson(tmp_path, dict(ROW), producer=boom)
    assert out["recorded"] is False and "producer_refused" in out["reason"]


def test_keep_path_wires_the_call_before_archive_record():
    import inspect
    src = inspect.getsource(run)
    call = src.index('attempt["keep_lesson"] = submit_keep_lesson(args.store, attempt)')
    assert call < src.index("archive.record(args.store, attempt, epoch=epoch,", call)
