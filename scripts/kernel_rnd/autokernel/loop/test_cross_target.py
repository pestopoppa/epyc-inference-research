"""ONE-champion cross-target lineage (cross_target.py): temp git repos, injected measurement.

The scenario is the DS41 / Q38FN pair. Each lane keeps on its own WORKING branch.
A DS41 keep that does not regress Q38FN folds into THE champion immediately, and Q38FN
picks it up from the champion at its next keep. A DS41 keep that regresses Q38FN
(b3e0b0902's -29..-41% prefill class) is held on DS41's working branch as
target_only_pending_gate, and a gating hypothesis is queued in DS41's inbox. Once a
gating keep lands (keyed on GGUF metadata, never an env var), Q38FN fold-checks the
held series on its own target. On approval the series folds into the champion. There
is never a second champion or trunk.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from . import cross_target as ct, lane_targets

CHAMPION = "ak/champion/llama-cpp-test"
DS41_WB, Q38_WB = "experimental/ds41-wb", "experimental/q38-wb"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _write(path: Path, body) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def g0_pass(**_kw):
    return {"passed": True, "reason": "no kernel path lost"}


def g0_fail(**_kw):
    return {"passed": False, "reason": "source : omp_pragma:ggml/ops.cpp 1->0 (undeclared)"}


@pytest.fixture
def world(tmp_path):
    """One repo. THE champion is checked out in its own tree, as pool.CHAMPION_TREE
    is. Each lane's working branch is checked out in its lane tree."""
    repo = tmp_path / "llama"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    (repo / "ggml").mkdir()
    (repo / "ggml" / "ops.cpp").write_text("a\nb\nc\nd\ne\nf\ng\nh\n")
    (repo / "src").mkdir()
    (repo / "src" / "deepseek.cpp").write_text("ds\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "champion")
    for branch in (CHAMPION, DS41_WB, Q38_WB):
        _git(repo, "branch", branch)
    champ_tree = tmp_path / "ak-loop-tree"
    _git(repo, "worktree", "add", "-q", str(champ_tree), CHAMPION)
    trees = {}
    for name, branch in (("ds41", DS41_WB), ("q38", Q38_WB)):
        trees[name] = tmp_path / f"lane-{name}"
        _git(repo, "worktree", "add", "-q", str(trees[name]), branch)
    owned0 = _write(tmp_path / "ds41" / "owned.json", {"ds41-t": {
        "launch": str(tmp_path / "l0.json"), "frozen_prompts": str(tmp_path / "p0.json")}})
    owned1 = _write(tmp_path / "q38" / "owned.json", {"q38-t": {
        "launch": str(tmp_path / "l1.json"), "frozen_prompts": str(tmp_path / "p1.json")}})
    binding = _write(tmp_path / "bind" / "lane-targets.json", {
        "schema": lane_targets.SCHEMA, "champion": {"branch": CHAMPION}, "lanes": {
            "lane0": {"target_id": "ds41-t", "owned_targets": str(owned0),
                      "exclusive_paths": ["src/deepseek*.cpp"], "as_peer_floor_file": None,
                      "as_peer_max_regression_pct": 2.0, "working_branch": DS41_WB},
            "lane1": {"target_id": "q38-t", "owned_targets": str(owned1),
                      "exclusive_paths": ["src/qwen*.cpp"], "as_peer_floor_file": None,
                      "as_peer_max_regression_pct": 2.0, "working_branch": Q38_WB}}})
    ds41 = lane_targets.resolve(binding, "lane0", target_id="ds41-t", workers=1, lock=False,
                                working_branch=DS41_WB, canonical_champion=CHAMPION)
    q38 = lane_targets.resolve(binding, "lane1", target_id="q38-t", workers=1, lock=False,
                               working_branch=Q38_WB, canonical_champion=CHAMPION)
    worker = tmp_path / "worker-q38"
    _git(repo, "worktree", "add", "-q", "--detach", str(worker), Q38_WB)
    return {"repo": repo, "trees": trees, "champ_tree": champ_tree, "worker": worker,
            "ds41": ds41, "q38": q38, "store": tmp_path / "ds41-store"}


def _keep(tree: Path, rel: str, text: str, message: str) -> str:
    (tree / rel).write_text(text)
    _git(tree, "commit", "-q", "-am", message)
    return _git(tree, "rev-parse", "HEAD")


def _cross(passed=True, effect=0.0, decisive=False, required=True):
    return {"required": required, "passed": passed, "peers": [] if not required else [
        {"lane": "lane1", "target_id": "q38-t", "passed": passed, "effect": effect,
         "decisive": decisive, "reason": "peer regressed" if not passed else "ok"}]}


UNCHANGED = {"q38-t": {"observed": True, "unchanged": True, "losses": [], "changed": []}}
REGRESSED = _cross(passed=False, effect=-0.35, decisive=True)


def _hold(world, text="a\nb\nc\nd\ne\nf\ng\nH\n", mechanism="m-ds41-only"):
    """A DS41 keep that regresses Q38FN: held, gate queued."""
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", text, "ds41 keep")
    row = ct.record_keep(world["ds41"], repo=world["repo"], keep_commit=keep,
                         decision=ct.decide(REGRESSED, UNCHANGED), cross=REGRESSED,
                         coverage=UNCHANGED, mechanism_id=mechanism, store=world["store"],
                         fold_check=g0_pass)
    return keep, row


def _gate(world, pid, line="if (model.arch == LLM_ARCH_DEEPSEEK2) { fast(); }\n", n=1):
    tree = world["trees"]["ds41"]
    path = tree / "src" / "deepseek.cpp"
    path.write_text(path.read_text() + line)
    _git(tree, "commit", "-q", "-am", f"gate {n}")
    gate = _git(tree, "rev-parse", "HEAD")
    statement = f"Gate the held keep to DeepSeek only. GATES-KEEP: {pid[:12]}"
    row = ct.record_keep(world["ds41"], repo=world["repo"], keep_commit=gate,
                         decision=ct.decide(_cross(effect=0.3, decisive=True), UNCHANGED),
                         cross=_cross(effect=0.3, decisive=True), coverage=UNCHANGED,
                         mechanism_id=f"m-gate-{n}", store=world["store"],
                         declaration_texts=(statement,), fold_check=g0_pass)
    return gate, row


def _inbox(world):
    inbox = world["store"] / "inbox"
    return sorted(p.name for p in inbox.glob("*.md")) if inbox.is_dir() else []


# ------------------------------------------------------------------ pure decisions

def test_decide():
    assert ct.decide(_cross(required=False), {})["decision"] == ct.FOLD
    assert ct.decide(_cross(), UNCHANGED)["decision"] == ct.FOLD
    regressed = ct.decide(REGRESSED, UNCHANGED)
    assert regressed["decision"] == ct.PENDING_GATE and "regressed" in regressed["reason"]
    assert ct.PENDING_GATE == "target_only_pending_gate"
    lost = {"q38-t": {"observed": True, "losses": [{"key": "iqk.gemm:Q8_0:act=Q8_0"}],
                      "changed": ["s:iqk.gemm:Q8_0:act=Q8_0"]}}
    assert ct.decide(_cross(effect=0.2, decisive=True), lost)["decision"] == ct.PENDING_GATE
    changed = {"q38-t": {"observed": True, "losses": [], "changed": ["s:iqk.gemm:Q4_K"]}}
    assert ct.decide(_cross(effect=0.001), changed)["decision"] == ct.PENDING_GATE
    assert ct.decide(_cross(effect=0.05, decisive=True), changed)["decision"] == ct.FOLD
    assert ct.decide(_cross(), {})["decision"] == ct.PENDING_GATE    # coverage unobserved


def test_gate_declarations_and_attestation_visible_mechanism():
    assert ct.gate_declarations("x GATES-KEEP: 0123456789ab y", None) == ["0123456789ab"]
    assert ct.gate_declarations("no gate here") == []
    arch = "+++ b/src/x.cpp\n+    if (model.arch == LLM_ARCH_DEEPSEEK2) {\n"
    assert ct.gate_mechanism(arch)[0]
    assert ct.gate_mechanism("+ if (t->type == GGML_TYPE_Q4_K && t->ne[0] % 256 == 0)\n")[0]
    env = "+ if (getenv(\"GGML_DS41_FAST\")) {\n+   (void) model.arch;\n"
    ok, why = ct.gate_mechanism(env)
    assert not ok and "runtime_attestation" in why
    assert not ct.gate_mechanism("+ static bool fast = std::getenv(\"X\") != nullptr;\n")[0]
    assert not ct.gate_mechanism("+ auto rd = &::getenv; if (rd(\"X\") && hparams.n_embd) {}\n")[0]
    assert not ct.gate_mechanism("+ for (char **e = environ; *e; ++e) { (void) model.arch; }\n")[0]
    assert not ct.gate_mechanism("+ if (src0->type == dst->type) { go(); }\n")[0]   # no model key
    ok, why = ct.gate_mechanism("+ if (fast) { go(); }\n")
    assert not ok and "GGUF" in why


# ------------------------------------------------------------------ fold now

def test_non_regressing_keep_folds_into_the_champion_and_syncs_into_the_peer(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k1")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m-fold", store=world["store"],
                         fold_check=g0_pass)
    assert row["decision"] == ct.FOLD and row["fold"]["result"] == "folded", row["fold"]
    assert row["fold"]["branch"] == CHAMPION
    assert _git(repo, "show", f"{CHAMPION}:ggml/ops.cpp").startswith("A\n")
    # The clean champion checkout (pool.CHAMPION_TREE) was fast-forwarded with it.
    assert (world["champ_tree"] / "ggml/ops.cpp").read_text().startswith("A\n")
    assert _git(world["champ_tree"], "status", "--porcelain", "--untracked-files=no") == ""
    assert ct.held(repo, CHAMPION, DS41_WB) == []
    assert _inbox(world) == []
    synced = ct.sync_from_champion(q38, repo=repo, champion_tree=world["trees"]["q38"],
                                   branch=Q38_WB)
    assert [r["applied"] for r in synced] == [True]
    assert (world["trees"]["q38"] / "ggml/ops.cpp").read_text().startswith("A\n")
    assert ct.sync_from_champion(q38, repo=repo, champion_tree=world["trees"]["q38"],
                                 branch=Q38_WB) == []
    assert ct.sync_from_champion(ds41, repo=repo, champion_tree=world["trees"]["ds41"],
                                 branch=DS41_WB) == []
    # No second lineage anywhere: the only ak/champion ref is THE champion.
    refs = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/ak/")
    assert refs.split() == [CHAMPION]


def test_manual_champion_work_reaches_every_lane(world):
    repo = world["repo"]
    _keep(world["champ_tree"], "ggml/ops.cpp", "a\nb\nc\nd\nE\nf\ng\nh\n", "manual fold")
    for lane, branch in (("ds41", DS41_WB), ("q38", Q38_WB)):
        rows = ct.sync_from_champion(world[lane], repo=repo,
                                     champion_tree=world["trees"][lane], branch=branch)
        assert [r["applied"] for r in rows] == [True]
        assert "E\n" in (world["trees"][lane] / "ggml/ops.cpp").read_text()


def test_g0_refusal_leaves_the_champion_untouched(world):
    repo, ds41 = world["repo"], world["ds41"]
    before = _git(repo, "rev-parse", CHAMPION)
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m", fold_check=g0_fail)
    assert row["fold"]["result"] == "g0_refused" and "omp_pragma" in row["fold"]["g0"]
    assert _git(repo, "rev-parse", CHAMPION) == before
    assert ct.held(repo, CHAMPION, DS41_WB) == [keep]


def test_dirty_champion_checkout_defers_the_fold_then_the_retry_folds(world):
    repo, ds41 = world["repo"], world["ds41"]
    before = _git(repo, "rev-parse", CHAMPION)
    (world["champ_tree"] / "ggml/ops.cpp").write_text("dirty\n")
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m", fold_check=g0_pass)
    assert row["fold"]["result"] == "checkout_dirty"
    assert _git(repo, "rev-parse", CHAMPION) == before
    _git(world["champ_tree"], "checkout", "--", "ggml/ops.cpp")
    out = ct.refresh_gates(ds41, repo=repo, store=world["store"], branch=DS41_WB,
                           fold_check=g0_pass)
    assert [r["fold"]["result"] for r in out if r["event"] == "fold_retry"] == ["folded"]
    assert ct.held(repo, CHAMPION, DS41_WB) == []


def test_fold_stands_when_the_checkout_refresh_is_refused(world):
    """The ref moves first; a checkout the fast-forward cannot refresh (an untracked
    file in the way of `read-tree -u`) is reported stale, never as a failed fold."""
    repo, ds41 = world["repo"], world["ds41"]
    (world["champ_tree"] / "ggml" / "new.cpp").write_text("untracked\n")   # not in index
    tree = world["trees"]["ds41"]
    (tree / "ggml" / "new.cpp").write_text("kept\n")
    _git(tree, "add", "ggml/new.cpp")
    _git(tree, "commit", "-q", "-m", "adds a file")
    keep = _git(tree, "rev-parse", "HEAD")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m", fold_check=g0_pass)
    assert row["fold"]["result"] == "folded"
    assert row["fold"]["checkouts_stale"][0]["checkout"] == str(world["champ_tree"])
    assert _git(repo, "show", f"{CHAMPION}:ggml/new.cpp") == "kept"
    assert ct.held(repo, CHAMPION, DS41_WB) == []
    assert (world["champ_tree"] / "ggml" / "new.cpp").read_text() == "untracked\n"
    # The stale checkout now defers every later fold until a human reconciles it.
    keep2 = _keep(tree, "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k2")
    row2 = ct.record_keep(ds41, repo=repo, keep_commit=keep2,
                          decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                          coverage=UNCHANGED, mechanism_id="m2", fold_check=g0_pass)
    assert row2["fold"]["result"] == "checkout_dirty"


def test_conflicting_fold_is_held_not_forced(world):
    repo, ds41 = world["repo"], world["ds41"]
    _keep(world["champ_tree"], "ggml/ops.cpp", "Z\nb\nc\nd\ne\nf\ng\nh\n", "champion edit")
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k4")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m4", fold_check=g0_pass)
    assert row["fold"]["result"] == "conflict"
    assert _git(repo, "show", f"{CHAMPION}:ggml/ops.cpp").startswith("Z\n")


# ------------------------------------------------------------------ hold, gate, fold

def test_regressing_keep_is_held_and_a_gate_hypothesis_is_queued(world):
    repo, q38 = world["repo"], world["q38"]
    before = _git(repo, "rev-parse", CHAMPION)
    keep, row = _hold(world)
    assert row["decision"] == ct.PENDING_GATE and "fold" not in row
    assert _git(repo, "rev-parse", CHAMPION) == before
    assert ct.held(repo, CHAMPION, DS41_WB) == [keep]
    names = _inbox(world)
    assert names == [f"00-gate-{row['patch_id'][:12]}-a1.md"]
    text = (world["store"] / "inbox" / names[0]).read_text()
    assert f"GATES-KEEP: {row['patch_id'][:12]}" in text
    assert "general.architecture" in text and "getenv" in text and "REFUSED" in text
    events = [r["event"] for r in ct.read(q38)]
    assert events == ["keep_decision", "gate_enqueued"]
    # Not eligible for a fold-check until a gating keep exists.
    assert ct.eligible(q38, repo=repo) == []


def test_gated_series_is_fold_checked_by_the_peer_then_folds(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    keep, held_row = _hold(world)
    gate, gate_row = _gate(world, held_row["patch_id"])
    assert gate_row["decision"] == ct.GATE_COMMIT
    assert gate_row["gates"] == [held_row["patch_id"]] and "fold" not in gate_row
    queue = ct.eligible(q38, repo=repo)
    assert [s["commits"] for s in queue] == [[keep, gate]]
    assert ct.eligible(ds41, repo=repo) == []        # a lane never checks its own series
    seen = []

    def approve(series):
        seen.append(series["commits"])
        _git(world["worker"], "checkout", "-q", "--detach", Q38_WB)
        for commit in series["commits"]:
            assert ct.apply_in_worktree(world["worker"], commit)
        return {"row": {"effect": 0.001, "decisive": False}, "passed": True,
                "reason": "within", "coverage": UNCHANGED["q38-t"]}

    event = ct.check_one(q38, repo=repo, measure=approve, fold_check=g0_pass)
    assert seen == [[keep, gate]]
    assert event["result"] == "approve" and event["fold"]["result"] == "folded"
    assert _git(repo, "show", f"{CHAMPION}:ggml/ops.cpp").endswith("H")
    assert "LLM_ARCH_DEEPSEEK2" in _git(repo, "show", f"{CHAMPION}:src/deepseek.cpp")
    assert _git(repo, "rev-parse", Q38_WB) == _git(repo, "rev-parse", "base")  # never moved
    assert ct.held(repo, CHAMPION, DS41_WB) == []
    # DS41's next boundary retires the queued gate: the champion carries the keep.
    out = ct.refresh_gates(ds41, repo=repo, store=world["store"], branch=DS41_WB)
    assert [r["event"] for r in out] == ["gate_resolved"] and _inbox(world) == []
    # Once per origin head: nothing more to check.
    assert ct.check_one(q38, repo=repo, measure=approve) is None


def test_rejected_series_requeues_the_gate_then_exhausts(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    keep, held_row = _hold(world)
    pid = held_row["patch_id"]

    def reject(_series):
        return {"row": {"effect": -0.2, "decisive": True}, "passed": False,
                "reason": "peer regressed -20.000% past the declared -2.000% point bar",
                "coverage": UNCHANGED["q38-t"]}

    for attempt in range(1, ct.MAX_GATE_ATTEMPTS + 1):
        _gate(world, pid, line=f"if (hparams.n_expert == {attempt}) {{ f(); }}\n", n=attempt)
        event = ct.check_one(q38, repo=repo, measure=reject, fold_check=g0_pass)
        assert event["result"] == "reject" and "fold" not in event
        out = ct.refresh_gates(ds41, repo=repo, store=world["store"], branch=DS41_WB)
        if attempt < ct.MAX_GATE_ATTEMPTS:
            assert [r["event"] for r in out] == ["gate_enqueued"]
            assert _inbox(world) == [f"00-gate-{pid[:12]}-a{attempt + 1}.md"]
            assert "past the declared" in out[0]["rejection"]
        else:
            assert [r["event"] for r in out] == ["gate_exhausted"]
            assert "NOT in production" in out[0]["why"] and _inbox(world) == []
    assert keep in ct.held(repo, CHAMPION, DS41_WB)       # still held, never folded
    assert ct.refresh_gates(ds41, repo=repo, store=world["store"], branch=DS41_WB) == []


def test_env_var_gate_is_refused_and_cannot_fold(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    _keep_commit, held_row = _hold(world)
    _gate_commit, row = _gate(world, held_row["patch_id"],
                              line='if (getenv("GGML_DS41_FAST")) { fast(); }\n')
    assert row["decision"] == ct.GATE_REFUSED and "runtime_attestation" in row["reason"]
    assert row["gates"] == [] and row["refused_gates"] == [held_row["patch_id"]]
    assert ct.eligible(q38, repo=repo) == []
    out = ct.refresh_gates(ds41, repo=repo, store=world["store"], branch=DS41_WB)
    assert [r["event"] for r in out] == ["gate_enqueued"]
    assert "runtime_attestation" in out[0]["rejection"]


def test_fold_check_failure_is_recorded_as_a_rejection(world):
    repo, q38 = world["repo"], world["q38"]
    _keep_commit, held_row = _hold(world)
    _gate(world, held_row["patch_id"])

    def boom(_series):
        raise RuntimeError("build failed")

    event = ct.check_one(q38, repo=repo, measure=boom)
    assert event["result"] == "reject" and "build failed" in event["reason"]


def test_real_g0_runs_on_a_fold(world):
    """The default G0 is kernel_coverage.fold_check on the champion tip vs the fold."""
    repo, ds41 = world["repo"], world["ds41"]
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m")
    assert row["fold"]["result"] == "folded", row["fold"]


def test_patch_id(world):
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "a\n", "shrink")
    assert len(ct.patch_id(world["repo"], keep)) == 40


# ------------------------------------------------------------------ binding

def test_binding_refuses_a_second_lineage(tmp_path, world):
    path = world["ds41"].path
    body = json.loads(path.read_text())
    trunk = {**body, "trunk": {"branch": "ak-trunk"}}
    with pytest.raises(lane_targets.LaneBindingError, match="second lineage"):
        lane_targets.load_binding(_write(tmp_path / "trunk.json", trunk))
    per_lane = json.loads(path.read_text())
    per_lane["lanes"]["lane0"]["champion_branch"] = "ds41-champ"
    with pytest.raises(lane_targets.LaneBindingError, match="second champion"):
        lane_targets.load_binding(_write(tmp_path / "perlane.json", per_lane))
    named = json.loads(path.read_text())
    named["lanes"]["lane0"]["working_branch"] = "ak/champion/ds41"
    with pytest.raises(lane_targets.LaneBindingError, match="neither a champion"):
        lane_targets.load_binding(_write(tmp_path / "named.json", named))
    same = json.loads(path.read_text())
    same["lanes"]["lane1"]["working_branch"] = DS41_WB
    with pytest.raises(lane_targets.LaneBindingError, match="distinct"):
        lane_targets.load_binding(_write(tmp_path / "same.json", same))
    with pytest.raises(lane_targets.LaneBindingError, match="not THE champion"):
        lane_targets.resolve(path, "lane0", target_id="ds41-t", workers=1, lock=False,
                             working_branch=DS41_WB,
                             canonical_champion="ak/champion/llama-cpp-ffc1bac82eec")
    with pytest.raises(lane_targets.LaneBindingError, match="not --experimental-branch"):
        lane_targets.resolve(path, "lane0", target_id="ds41-t", workers=1, lock=False,
                             working_branch=CHAMPION, canonical_champion=CHAMPION)
    assert lane_targets.load_binding(path)[1] == CHAMPION
