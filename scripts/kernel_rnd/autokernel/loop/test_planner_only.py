"""`--planner-only`: refusals, recorded report and no-build stop, with a stub actor only."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from ..controller import workload_contract
from . import actors, belief_context, loop, planner_only as po


def _tree(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    (tree / po.GPU_SOURCE_DIR).mkdir(parents=True)
    (tree / po.GPU_SOURCE_DIR / "a.cu").write_text("x\n")
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True)
    return tree


def _args(tmp_path: Path, **kw) -> argparse.Namespace:
    model = tmp_path / "m-Q8_0.gguf"
    model.write_text("")
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    base = dict(out=tmp_path / "out", model=model, worktree=tmp_path / "tree", store=store,
                profile=None, surface="pp512", planner_model="m", planner_effort="high",
                actor_timeout_s=1, actor_belief_context="required", belief_root_repo=tmp_path,
                belief_ledger=None, belief_known_good=None, escalate=None,
                belief_escalate_after_s=1.0, belief_reader_timeout_s=1.0,
                belief_focus_symbol=None, preseeded_hypotheses=None,
                cpu_serving_launch=None, planner_surface=None)
    return argparse.Namespace(**{**base, **kw})


class _Stub:
    """Stands in for the actor seat: writes the receipt `propose` would have written."""

    def __init__(self, result, **kw):
        self.result, self.workspace, self.authored = result, Path(kw["workspace"]), 0

    def propose(self, context):
        self.context = context
        log = self.workspace.parent / actors.ACTOR_REPLY_DIR
        log.mkdir(parents=True, exist_ok=True)
        (log / belief_context.RECEIPT_LOG).write_text(json.dumps({
            "receipt_id": "sha256:r", "ledger_state_digest": "sha256:l",
            "reliance": {"declared": ["clm_a"], "accepted": ["clm_a"], "rejected": []}}) + "\n")
        return self.result

    def author(self, hypothesis, context):
        self.authored += 1
        return ("ggml/src/ggml-cuda/a.cu",)


@pytest.fixture(autouse=True)
def _census(monkeypatch):
    class C:
        quant_label = "q8_0"
    monkeypatch.setattr(workload_contract, "read_split_census", lambda _: C)
    monkeypatch.setattr(actors, "backend_for", lambda *_: type("B", (), {"describe": lambda s: {}})())
    monkeypatch.setattr(actors, "ActorSeat", lambda **_: None)


def _run(tmp_path, result, **kw):
    made = []

    def make(**fields):
        made.append(_Stub(result, **fields))
        return made[0]

    code = po.run(_args(tmp_path, **kw), seat_common={}, make_planner=make)
    return code, made


def test_refuses_without_profile(tmp_path, capsys):
    _tree(tmp_path)
    code, made = _run(tmp_path, None)
    assert code == 2 and not made and "REFUSED" in capsys.readouterr().err


def test_refuses_empty_profile(tmp_path):
    _tree(tmp_path)
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({"hotspots": []}))
    code, made = _run(tmp_path, None, profile=prof)
    assert code == 2 and not made


def test_refuses_without_source_tree(tmp_path):
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps([{"signature": "k"}]))
    code, made = _run(tmp_path, None, profile=prof)
    assert code == 2 and not made


def test_records_hypothesis_and_stops_before_build(tmp_path):
    tree = _tree(tmp_path)
    store = tmp_path / "store"
    store.mkdir()
    (store / "loop-status.json").write_text(json.dumps({"hotspots": [{"signature": "k"}]}))
    hypo = loop.Hypothesis("m", "s", "f", "ggml/src/ggml-cuda/a.cu", "sym")
    code, made = _run(tmp_path, hypo, store=store)
    assert code == 0 and made[0].authored == 1
    assert made[0].context["kernel_hotspots"] == [{"signature": "k"}]
    report = json.loads((tmp_path / "out" / po.REPORT_NAME).read_text())
    assert report["status"] == "authored" and report["hypothesis"] == "s"
    assert report["target_symbol"] == "sym" and report["relies_on_claims"] == ["clm_a"]
    assert report["receipt_id"] == "sha256:r" and report["read_record_id"] == "sha256:l"
    assert subprocess.run(["git", "-C", str(tree), "status", "--porcelain"],
                          capture_output=True, text=True).stdout == ""


def test_abstain_records_without_authoring(tmp_path):
    _tree(tmp_path)
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps([{"signature": "k"}]))
    code, made = _run(tmp_path, loop.Abstain("no route"), profile=prof)
    assert code == 0 and made[0].authored == 0
    report = json.loads((tmp_path / "out" / po.REPORT_NAME).read_text())
    assert report["status"] == "abstained" and report["abstain_reason"] == "no route"


def test_claim_texts_keep_what_the_planner_saw():
    section = "## Beliefs\n- clm_a [t] here\n  value 1\n  mechanism: m\n- clm_b [t] b\n\nend"
    assert po.claim_texts(section, ["clm_a", "clm_c"]) == {
        "clm_a": "clm_a [t] here\nvalue 1\nmechanism: m", "clm_c": ""}


def _cpu_setup(tmp_path, *, status="observed"):
    tree = tmp_path / "tree"
    (tree / po.CPU_SOURCE_DIR).mkdir(parents=True)
    (tree / po.CPU_SOURCE_DIR / "a.cpp").write_text("x\n")
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True)
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"backend": "cpu", "model": {"path": str(tmp_path / "m-Q8_0.gguf")}}))
    prof = tmp_path / "node.json"
    prof.write_text(json.dumps({"schema": po.NODE_PROFILE_SCHEMA,
                                "section": {"status": status, "mechanism_shares": []}}))
    (tmp_path / "store" / "inbox").mkdir(parents=True)
    (tmp_path / "store" / "inbox" / "73-s.md").write_text("seed 73\n")
    return {"cpu_serving_launch": launch, "profile": prof, "planner_surface": "serving:r-cpu"}


def test_cpu_target_builds_cpu_regime_with_profile_and_inbox(tmp_path):
    hypo = loop.Hypothesis("m", "s", "f", "ggml/src/ggml-cpu/a.cpp", "sym")
    code, made = _run(tmp_path, hypo, **_cpu_setup(tmp_path))
    assert code == 0
    context = made[0].context
    assert actors._cpu_target(context) and context["current_regime"]["backend"] == "cpu"
    assert context["current_regime"]["measurement_surface"] == "serving:r-cpu"
    assert context["node_profile"]["status"] == "observed"
    assert any("seed 73" in item for item in context["inbox"])


@pytest.mark.parametrize("change", ["unobserved", "gpu_rows", "no_surface", "gpu_launch"])
def test_cpu_target_refuses_without_real_inputs(tmp_path, change, capsys):
    kw = _cpu_setup(tmp_path, status="not_collected" if change == "unobserved" else "observed")
    if change == "gpu_rows":
        kw["profile"].write_text(json.dumps({"hotspots": [{"signature": "k"}]}))
    if change == "no_surface":
        kw["planner_surface"] = None
    if change == "gpu_launch":
        kw["cpu_serving_launch"].write_text(json.dumps({"backend": "gpu"}))
    code, made = _run(tmp_path, None, **kw)
    assert code == 2 and not made and "REFUSED" in capsys.readouterr().err


def test_cpu_target_refuses_gpu_only_tree(tmp_path):
    kw = _cpu_setup(tmp_path)
    (tmp_path / "tree" / po.CPU_SOURCE_DIR / "a.cpp").unlink()
    code, made = _run(tmp_path, None, **kw)
    assert code == 2 and not made


def test_cpu_card_and_report_name_profile_anchor_mismatch():
    profile = {"node_profile": {"status": "observed", "anchor_commit": "a" * 40}}
    assert po.anchor_mismatch(profile, "a" * 40) is None
    assert po.anchor_mismatch({"node_profile": {"status": "observed"}}, "b" * 40) is None
    named = po.anchor_mismatch(profile, "b" * 40)
    assert "aaaaaaaaaaaa" in named and "bbbbbbbbbbbb" in named


def test_cpu_profile_record_carries_anchor(tmp_path):
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({
        "schema": po.NODE_PROFILE_SCHEMA, "cache_key": {"anchor_commit": "c" * 40},
        "section": {"status": "observed"}}))
    assert po.load_profile(prof, cpu=True)["node_profile"]["anchor_commit"] == "c" * 40
