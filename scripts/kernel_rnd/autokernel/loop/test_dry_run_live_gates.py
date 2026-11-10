"""A green --dry-run must predict that the live start clears its startup refusals."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from pathlib import Path

import pytest

from . import accumulate, actor_wire_gate, champion, claim, kernel_coverage, lane_watchdog, run


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def args_for(store, **extra):
    base = dict(store=store, bundle_surface=None, new_store=False, worktree=store,
                rebaseline_stale=False, new_anchor_epoch=False)
    return SimpleNamespace(**{**base, **extra})


def test_scaffold_is_not_history(tmp_path):
    store = tmp_path / "s"
    assert run._store_holds_only_startup_scaffold(store)  # absent
    (store / "inbox").mkdir(parents=True)
    (store / kernel_coverage.CAPTURE_DIR).mkdir()
    (store / "actor-wire-request.json").write_text("{}")
    assert run._store_holds_only_startup_scaffold(store)
    (store / "loop-run.json").write_text(json.dumps({"state": "actor_admitted"}))
    assert run._store_holds_only_startup_scaffold(store)
    (store / "loop-run.json").write_text(json.dumps({"state": "complete"}))
    assert not run._store_holds_only_startup_scaffold(store)
    (store / "loop-run.json").unlink()
    (store / "inbox" / "note").write_text("x")
    assert not run._store_holds_only_startup_scaffold(store)


def test_refusal_3_new_store_with_wire_request_initializes(tmp_path):
    store = tmp_path / "s"
    store.mkdir()
    (store / "actor-wire-request.json").write_text("{}")
    assert run._accumulator_startup_refusal(
        args_for(store, new_store=True), anchor_commit="a" * 40, experimental=False) is None
    # Without the opt-in the same bytes are a populated store that live refuses.
    refusal = run._accumulator_startup_refusal(
        args_for(store), anchor_commit="a" * 40, experimental=False)
    assert refusal.startswith("accumulator: bundle recovery required")


def test_refusal_3_real_content_still_refused_and_read_only(tmp_path):
    store = tmp_path / "s"
    store.mkdir()
    (store / "stray").write_text("x")
    before = tree(store)
    assert "bundle recovery required" in run._accumulator_startup_refusal(
        args_for(store, new_store=True), anchor_commit="a" * 40, experimental=False)
    assert tree(store) == before


def test_surface_gate_matches_live_on_history_without_authority(tmp_path, monkeypatch):
    monkeypatch.setattr(accumulate, "_has_experiment_history", lambda _s: True)
    refused, reason, _ = accumulate.would_refuse_anchor(
        tmp_path, anchor_commit="a", is_ancestor=lambda *_: True, surface="cpu")
    assert refused and "experiment history" in reason


def test_wire_evaluate_writes_nothing_and_admit_still_writes(tmp_path):
    with pytest.raises(actor_wire_gate.WireGateError, match="no passing actor wire record"):
        actor_wire_gate.evaluate([{"role": "x"}])
    assert actor_wire_gate.evaluate([{"role": "x"}], override="why")["status"] == "OVERRIDDEN"
    assert tree(tmp_path) == []
    with pytest.raises(actor_wire_gate.WireGateError):
        actor_wire_gate.admit([{"role": "x"}], store=tmp_path, out=None)
    assert tree(tmp_path) == ["actor-wire-request.json"]


def test_device_probe_absent_lock_not_created(tmp_path):
    lock = tmp_path / "gpu.lock"
    assert claim.device_claim_refusal(lock) is None
    assert not lock.exists()


def test_device_probe_refuses_own_tree_and_keeps_holder(tmp_path):
    lock = tmp_path / "gpu.lock"
    with lock.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert "held by this process tree" in claim.device_claim_refusal(lock)
        other = lock.open("a")  # the holder's exclusive lock survived the probe
        with pytest.raises(OSError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        other.close()
    assert claim.device_claim_refusal(lock) is None


def test_device_probe_unrelated_holder_is_not_a_refusal(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    flag = tmp_path / "held"
    code = ("import fcntl,sys,pathlib;f=open(sys.argv[1]);fcntl.flock(f,fcntl.LOCK_EX);"
            "pathlib.Path(sys.argv[2]).touch();sys.stdin.read()")
    child = subprocess.Popen([sys.executable, "-c", code, str(lock), str(flag)],
                             stdin=subprocess.PIPE)
    try:
        deadline = time.time() + 20
        while not flag.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert flag.exists()
        with lock.open() as busy, pytest.raises(OSError):
            fcntl.flock(busy, fcntl.LOCK_SH | fcntl.LOCK_NB)  # precondition: really held
        assert claim.device_claim_refusal(lock) is None
    finally:
        child.stdin.close()
        child.wait(timeout=20)


def test_device_probe_inherited_hold_passes(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    with lock.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        st = os.stat(lock)
        monkeypatch.setenv(claim.INHERITED_DEVICE_LOCK_ENV,
                           f"{holder.fileno()}:{st.st_dev}:{st.st_ino}:{os.getppid()}")
        assert claim.device_claim_refusal(lock) is None


KEEPER_HOLDER = (
    "import fcntl,os,subprocess,sys;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);"
    "os.set_inheritable(f.fileno(),False);"
    "sys.exit(subprocess.call([sys.executable,'-c',sys.argv[2],sys.argv[1]],"
    "env=dict(os.environ,**dict(a.split('=',1) for a in sys.argv[3:]))))")
KEEPER_CHILD = (
    "import sys\nfrom pathlib import Path\n"
    "from scripts.kernel_rnd.autokernel.loop import claim\n"
    "lock=Path(sys.argv[1])\n"
    "assert claim.device_claim_refusal(lock) is None\n"
    "with claim.hold(lock) as r:\n"
    "    assert r['window_hold_pid']\n"
    "    assert r.observe()['status']=='held'\n"
    "    assert claim.device_claim_refusal(lock) is not None\n")
REFUSED_CHILD = (
    "import sys\nfrom pathlib import Path\n"
    "from scripts.kernel_rnd.autokernel.loop import claim\n"
    "lock=Path(sys.argv[1])\n"
    "assert 'process tree' in claim.device_claim_refusal(lock)\n"
    "try:\n    claim.hold(lock).__enter__()\n"
    "except claim.ClaimRefused:\n    sys.exit(0)\nsys.exit(9)\n")


def _keeper(tmp_path, code, *env):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    repo = Path(__file__).resolve().parents[4]
    scrub = {k: v for k, v in os.environ.items() if not k.startswith("FILLER_")}
    return subprocess.run([sys.executable, "-c", KEEPER_HOLDER, str(lock), code, *env],
                          env={**scrub, "PYTHONPATH": str(repo)}, cwd=repo, timeout=120,
                          capture_output=True, text=True)


def test_keeper_ancestor_hold_is_accepted(tmp_path):
    result = _keeper(tmp_path, KEEPER_CHILD, "FILLER_NEEDGPU=1", "FILLER_ID=row")
    assert result.returncode == 0, result.stderr


def test_ancestor_hold_needs_keeper_env(tmp_path):
    result = _keeper(tmp_path, REFUSED_CHILD)
    assert result.returncode == 0, result.stderr


def test_keeper_env_does_not_admit_unrelated_holder(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    flag = tmp_path / "held"
    code = ("import fcntl,sys,pathlib;f=open(sys.argv[1]);fcntl.flock(f,fcntl.LOCK_EX);"
            "pathlib.Path(sys.argv[2]).touch();sys.stdin.read()")
    child = subprocess.Popen([sys.executable, "-c", code, str(lock), str(flag)],
                             stdin=subprocess.PIPE)
    try:
        deadline = time.time() + 20
        while not flag.exists() and time.time() < deadline:
            time.sleep(0.02)
        monkeypatch.setenv("FILLER_NEEDGPU", "1")
        monkeypatch.setenv("FILLER_ID", "row")
        assert claim._ancestor_device_holder(lock) is None
        with pytest.raises(claim.ClaimRefused), claim.hold(lock):
            pass
    finally:
        child.stdin.close()
        child.wait(timeout=20)


def test_watchdog_state_file_refusal_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(lane_watchdog.tempfile, "gettempdir", lambda: str(tmp_path))
    store = tmp_path / "store"
    assert lane_watchdog.startup_refusal(store) is None
    assert tree(tmp_path) == []
    root = lane_watchdog._state_root()
    root.mkdir()
    digest = hashlib.sha256(str(store.absolute().resolve()).encode()).hexdigest()
    (root / f"{digest}.json").write_text("{}")
    assert lane_watchdog.startup_refusal(store) == lane_watchdog._UNRESOLVED_STATE


def test_main_names_every_refusal_and_exits_nonzero(monkeypatch, capsys):
    def fake(_argv):
        run._DRY_RUN_REFUSALS.extend(["first", "second"])
        return 0
    monkeypatch.setattr(run, "_main", fake)
    assert run.main(["--dry-run", "--store", "x"]) == 1
    out = capsys.readouterr().out
    assert "DRY RUN REFUSED: first" in out and "DRY RUN REFUSED: second" in out


def test_main_names_refusals_found_before_a_startup_refusal(monkeypatch, capsys):
    def fake(_argv):
        run._DRY_RUN_REFUSALS.append("store missing")
        raise champion.StartupRefused("late")
    monkeypatch.setattr(run, "_main", fake)
    with pytest.raises(champion.StartupRefused):
        run.main(["--dry-run", "--store", "x"])
    out = capsys.readouterr().out
    assert "DRY RUN REFUSED: store missing" in out and "DRY RUN REFUSED: late" in out


def _cor_world(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from . import test_floor_registry as tfr
    repo = tmp_path / "llama"
    repo.mkdir()
    tfr.git(repo, "init", "-q", "-b", tfr.BRANCH)
    anchor_old = tfr.commit(repo, {"a.c": "1"}, "old")
    anchor = tfr.commit(repo, {"a.c": "2"}, "new")
    monkeypatch.setattr(run, "_FLOOR_REGISTRY", object())
    def image(name, commit):
        path = tmp_path / name
        path.mkdir()
        (path / "VERSION.txt").write_text(f"version: 1 ({commit[:9]})\nbuilt with GNU 15.2.0 for Linux x86_64\n")
        (path / "SHA256SUMS").write_text("")
        (path / "PROVENANCE.md").write_text("store\n")
        return path
    store = tmp_path / "store"
    store.mkdir()
    args = SimpleNamespace(store=store, worktree=repo, cor_build=None, anchor_build=tmp_path / "tip",
                           allow_unverified_anchor=False)
    return args, image, anchor_old, anchor


def _check_no_bundle(args, anchor):
    run._dry_run_cor_for_restore(args, anchor_commit=anchor, experimental=False,
                                 surface=None, has_bundle=False)


def test_cor_build_of_another_commit_refused_on_store_with_no_accumulator(tmp_path, monkeypatch):
    args, image, old, anchor = _cor_world(tmp_path, monkeypatch)
    args.cor_build = image("cor", old)
    with pytest.raises(champion.StartupRefused, match="exact arm required"):
        _check_no_bundle(args, anchor)


def test_cor_build_of_the_anchor_passes_on_store_with_no_accumulator(tmp_path, monkeypatch, capsys):
    args, image, _old, anchor = _cor_world(tmp_path, monkeypatch)
    args.cor_build = image("cor", anchor)
    _check_no_bundle(args, anchor)
    assert "exact build identity verified" in capsys.readouterr().out


def test_no_cor_build_on_store_with_no_accumulator_needs_nothing(tmp_path, monkeypatch, capsys):
    args, _image, _old, anchor = _cor_world(tmp_path, monkeypatch)
    _check_no_bundle(args, anchor)
    assert "current anchor" in capsys.readouterr().out


def test_dry_run_reaches_the_cor_check_when_the_restore_is_acceptable():
    source = Path(run.__file__).read_text()
    site = source[source.index("has_bundle = ((args.store / bundle_journal)"):]
    site = site[:site.index("actor_wire_gate.request")]
    assert "_dry_run_cor_for_restore(" in site and "_DRY_RUN_REFUSALS.append(str(exc))" in site
