"""kpf64 adapter reads opencode from the operator's READ-ONLY snapshots; live stores are refused (2026-10-22)."""
import json
import re
import sqlite3
from pathlib import Path

import pytest

import adapter
import opencode_ro

HERE = Path(__file__).resolve().parents[1]
LIVE = ["~/.local/share/opencode/opencode.db", "/home/node/.local/share/opencode/opencode.db",
        "/mnt/raid0/llm/opencode/epyc/opencode.db"]


def mkdb(path, sid):
    con = sqlite3.connect(path)
    con.executescript("""create table session(id text, parent_id text, time_created int, time_updated int);
      create table message(id text, session_id text, time_created int, data text);
      create table part(id text, message_id text, session_id text, time_created int, data text);
      create table event(id text, data text);""")
    con.execute("insert into session values(?,NULL,1,2)", (sid,))
    con.execute("insert into message values('m1',?,1,?)", (sid, json.dumps({"role": "user"})))
    con.execute("insert into part values('p1','m1',?,1,?)", (sid, json.dumps({"type": "text", "text": "hi " * 20})))
    con.commit()
    con.close()
    return str(path)


def test_default_dbs_are_the_two_ro_snapshots():
    assert adapter.OPENCODE_RO_DBS == {"epyc": "/mnt/raid0/llm/opencode-db/epyc.db",
                                       "host": "/mnt/raid0/llm/opencode-db/host-daniele.db"}
    assert all(not opencode_ro.is_live_opencode_path(p) for p in adapter.OPENCODE_RO_DBS.values())
    a = [x for x in re.findall(r'add_argument\("--opencode-db"', (HERE / "adapter.py").read_text())]
    assert a


def test_home_no_longer_locates_opencode(tmp_path):
    h = tmp_path / "home"
    (h / ".local/share/opencode").mkdir(parents=True)
    mkdb(h / ".local/share/opencode/opencode.db", "s_home")
    cands = adapter.candidates("opencode", str(h), opencode_dbs=[str(tmp_path / "nonexistent.db")])
    assert cands == []                                 # the db under --home is ignored


def test_candidates_cover_both_dbs_and_carry_source(tmp_path):
    e, o = mkdb(tmp_path / "e.db", "s_e"), mkdb(tmp_path / "h.db", "s_h")
    c = adapter.candidates("opencode", str(tmp_path), opencode_dbs=[e, o])
    assert sorted(x[1] for x in c) == ["s_e", "s_h"] and {x[2] for x in c} == {e, o}
    assert opencode_ro.source_for("/mnt/raid0/llm/opencode-db/host-daniele.db") == "host"
    assert opencode_ro.source_for(e) == "unknown"


@pytest.mark.parametrize("live", LIVE)
def test_live_paths_refused(live):
    assert opencode_ro.is_live_opencode_path(live)
    with pytest.raises(opencode_ro.LiveOpencodeStoreError, match="LIVE opencode store"):
        adapter.candidates("opencode", "/nonexistent", opencode_dbs=[live])
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        adapter.store_fanout("/nonexistent", ("opencode",), opencode_dbs=[live])
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        adapter._ro(live)


def test_symlink_into_live_refused(tmp_path):
    link = tmp_path / "x.db"
    link.symlink_to("/mnt/raid0/llm/opencode/epyc/opencode.db")
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        opencode_ro.open_ro(link)


def test_open_ro_query_only_retry_once(tmp_path):
    db = mkdb(tmp_path / "a.db", "s")
    conn = opencode_ro.open_ro(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("delete from session")
    conn.close()
    calls = []

    def flaky(target, **kw):
        calls.append(target)
        if len(calls) == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return sqlite3.connect(target, **kw)

    opencode_ro.open_ro(db, retry_delay=0, _connect=flaky).close()
    assert len(calls) == 2 and calls[0].endswith("?mode=ro")
    calls.clear()

    def dead(target, **kw):
        calls.append(target)
        raise sqlite3.OperationalError("unable to open database file")

    with pytest.raises(sqlite3.OperationalError):
        opencode_ro.open_ro(db, retry_delay=0, _connect=dead)
    assert len(calls) == 2


def test_no_event_table_reads():
    for f in ("adapter.py", "opencode_ro.py"):
        assert not re.search(r"(?i)\b(from|join)\s+event\b", (HERE / f).read_text()), f


@pytest.mark.parametrize("tag", ["epyc", "host"])
def test_smoke_real_snapshot(tag):
    db = Path(opencode_ro.OPENCODE_RO_DBS[tag])
    if not db.exists():
        pytest.skip(f"{db} absent")
    conn = opencode_ro.open_ro(db)
    try:
        n = conn.execute("select count(*) from session").fetchone()[0]
    finally:
        conn.close()
    print(f"SMOKE {tag}: {n} sessions")
    assert n > 0
