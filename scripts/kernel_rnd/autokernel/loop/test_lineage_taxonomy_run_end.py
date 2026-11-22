"""AK-lineage: run.py's run-end publish_lineage also retains the taxonomy receipt."""
import inspect
import sqlite3

from autokernel.loop import run


def _db(store):
    store.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(store / "experiments.db")
    con.execute("CREATE TABLE experiments(attempt_id,recorded_at,campaign_id,target_symbol,"
                "statement,status,effect_fraction)")
    con.execute("INSERT INTO experiments VALUES('a1','2026-11-22T00:00:00Z','ak','f',"
                "'fuse the dequant into the matmul tile loop','kept',0.03)")
    con.commit()
    con.close()


def test_run_end_writes_receipt(tmp_path):
    _db(tmp_path)
    path = run.publish_lineage_taxonomy(tmp_path, "a" * 40)
    assert path is not None and path.parent == tmp_path / "lineage-taxonomy" and path.is_file()


def test_missing_db_or_bad_commit_never_raises(tmp_path):
    assert run.publish_lineage_taxonomy(tmp_path, "a" * 40) is None
    _db(tmp_path)
    assert run.publish_lineage_taxonomy(tmp_path, "short") is None


def test_publish_lineage_calls_taxonomy():
    src = inspect.getsource(run)
    assert "publish_lineage_taxonomy(args.store, source_revision)" in src
