"""VB-RUNNER-PATHS: GPU-runner belief capture resolves its root from EPYC_ROOT and never fails quietly."""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))

import belief_capture as bc  # noqa: E402

CAPTURE_SRC = '''
def write_belief_measurements(path, **kw):
    return str(path) + ".beliefs.jsonl"
'''


def fake_root(tmp_path: Path, name: str = "tale_budget_capture", body: str = CAPTURE_SRC) -> Path:
    adapters = tmp_path / "root" / bc.ADAPTERS_REL
    adapters.mkdir(parents=True)
    (adapters / f"{name}.py").write_text(body)
    return tmp_path / "root"


def test_unset_root_is_refused_not_guessed():
    with pytest.raises(bc.CaptureUnavailable, match="EPYC_ROOT is not set"):
        bc.root_checkout({})


def test_a_root_without_adapters_is_refused(tmp_path):
    with pytest.raises(bc.CaptureUnavailable, match="has no scripts/vidya/adapters"):
        bc.root_checkout({"EPYC_ROOT": str(tmp_path)})


def test_capture_is_loaded_from_the_named_root_only(tmp_path):
    root = fake_root(tmp_path)
    mod = bc.load_capture("tale_budget_capture", {"EPYC_ROOT": str(root)})
    assert Path(mod.__file__).resolve().is_relative_to(root.resolve())
    assert mod.write_belief_measurements("x") == "x.beliefs.jsonl"


def test_missing_module_and_missing_writer_are_refused(tmp_path):
    root = fake_root(tmp_path, body="X = 1\n")
    env = {"EPYC_ROOT": str(root)}
    with pytest.raises(bc.CaptureUnavailable, match="capture module missing"):
        bc.preflight("review_f1_capture", env=env)
    with pytest.raises(bc.CaptureUnavailable, match="no write_belief_measurements"):
        bc.preflight("tale_budget_capture", env=env)


def test_a_failed_capture_is_loud_recorded_and_sets_the_exit_code():
    err = io.StringIO()
    log = bc.CaptureLog(stream=err)
    assert log.run("ok", lambda: "a.beliefs.jsonl") == "a.beliefs.jsonl"
    assert log.exit_code() == 0

    def boom():
        raise ValueError("row refused")

    assert log.run("bad", boom) == "REFUSED ValueError: row refused"
    assert "BELIEF CAPTURE FAILED [bad]" in err.getvalue()
    assert log.as_record()["failed"] == {"bad": "REFUSED ValueError: row refused"}
    assert log.as_record()["written"] == {"ok": "a.beliefs.jsonl"}
    assert log.exit_code() == bc.EXIT_CAPTURE_FAILED
    assert log.exit_code(rc=7) == 7, "a driver failure keeps its own code"


# --- the drivers ------------------------------------------------------------------------------

DRIVERS = [HERE / "prb_t4_tale_gpu.py", HERE / "review_f1" / "ev13b_run.py"]


@pytest.mark.parametrize("path", DRIVERS, ids=lambda p: p.name)
def test_drivers_carry_no_hardcoded_root_or_tmp_worktree(path):
    src = path.read_text()
    assert "/workspace/scripts/vidya" not in src
    assert "/mnt/raid0/llm/worktrees" not in src
    assert "belief_capture.preflight(" in src and "captures.exit_code(" in src


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"driver_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", DRIVERS, ids=lambda p: p.name)
def test_drivers_refuse_before_claiming_the_gpu(path, tmp_path, monkeypatch, capsys):
    """An unset EPYC_ROOT must cost no GPU time: the refusal precedes the claim and any server."""
    mod = _load(path)

    def no_claim():
        raise AssertionError("GPU claim taken before the capture preflight")

    monkeypatch.setattr(mod.claim, "hold", no_claim)
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", [path.name, "--out", str(tmp_path / "out")])
    assert mod.main() == 2
    assert "EPYC_ROOT is not set" in capsys.readouterr().err


def test_prb_t4_refuses_a_missing_pool_before_claiming(tmp_path, monkeypatch, capsys):
    mod = _load(HERE / "prb_t4_tale_gpu.py")
    monkeypatch.setattr(mod.claim, "hold", lambda: (_ for _ in ()).throw(AssertionError("claimed")))
    monkeypatch.setenv("EPYC_ROOT", str(fake_root(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["prb", "--out", str(tmp_path / "out"),
                                      "--pool", str(tmp_path / "nope.jsonl"),
                                      "--python", sys.executable])
    assert mod.main() == 2
    assert "--pool" in capsys.readouterr().err
