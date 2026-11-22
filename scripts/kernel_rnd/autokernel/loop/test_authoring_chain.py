"""AKU-06d4: durable advice->patch->commit->build chain, replay and planner refresh."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from ..execution import sandbox, worktree as W
from . import authoring_chain as chain, contained_authoring as ca
from . import source_build_preparation as bridge
from .test_contained_authoring import _reserve, actor_state  # noqa: F401 (fixture)
from .test_source_build_preparation import tiny_source  # noqa: F401 (fixture)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(snapshot: str, *, exit_code=0) -> W.BuildIdentity:
    receipts = tuple({"phase": p, "activation": {"sandbox_id": sandbox.SANDBOX_ID,
                                                 "writable_root": "/b"},
                      "teardown": {"verified_empty": True, "removed": True}}
                     for p in ("configure", "build"))
    facts = replace(W.parse_build_log(""), succeeded_by_log=True)
    return W.BuildIdentity(
        "akc-source-bridge", "ak-source-bridge", {}, snapshot, 1, 1, "/s", "/b", True,
        _h("pre"), "/s", ("/p",), "gcc", "gcc 13", "cmake", "make", (), {}, "/b/log",
        _h("log"), "/b/bin", _h("bin"), (), None, receipts, facts, exit_code, 1.0,
        "2026-11-22T00:00:00Z")


def _authored(owner, instance, selected, fixture):
    res = _reserve(owner, instance, selected, fixture)
    out = ca.encode_output(res.attempt_id, fixture.patch.encode())
    return owner.accept(res.attempt_id, out, ca.sign_output(res.worker_key, res.attempt_id, out))


def test_full_chain_replays_on_restart_and_refreshes_planner(actor_state, tmp_path):  # noqa: F811
    instance, selected, fixture = actor_state
    path = str(tmp_path / "chain.jsonl")
    owner = ca.ContainedAuthoringOwner(journal=chain.AuthoringChainJournal(path))
    authored = _authored(owner, instance, selected, fixture)
    capability = bridge.materialize_source(authored.bound, campaign_driver=instance,
                                           actor_worktree=fixture.actor)
    journal = owner.journal
    with pytest.raises(chain.ChainRefused, match="no committed source"):
        journal.build_verified(authored.advice_digest, _identity(capability["source_tree_digest"]))
    journal.committed(authored.advice_digest, capability)
    with pytest.raises(chain.ChainRefused, match="runner return"):
        journal.build_verified(authored.advice_digest, {"exit_code": 0, "binary": "x"})
    with pytest.raises(chain.ChainRefused, match="committed source tree"):
        journal.build_verified(authored.advice_digest, _identity(_h("other-tree")))
    with pytest.raises(chain.ChainRefused, match="committed source tree"):
        journal.build_verified(authored.advice_digest,
                               _identity(capability["source_tree_digest"], exit_code=2))
    target = selected.actor_request.proposal.target_revision_digest
    prereqs = {target: (chain.BUILD_PREREQUISITE, "other"), "f" * 64: (chain.BUILD_PREREQUISITE,)}
    assert chain.refresh_prerequisites(prereqs, journal.state) == prereqs
    journal.build_verified(authored.advice_digest, _identity(capability["source_tree_digest"]))

    # Restart: a fresh owner replays the durable chain from disk only.
    restarted = chain.replay(path)
    replayed = restarted.chains[authored.advice_digest]
    assert replayed.stage == "BUILD_VERIFIED"
    assert replayed.records["COMMITTED"]["source_commit"] == capability["source_commit"]
    assert replayed.records["AUTHORED"]["preparation_digest"] == authored.bound.digest
    assert dict(chain.refresh_prerequisites(prereqs, restarted)) == {
        target: ("other",), "f" * 64: (chain.BUILD_PREREQUISITE,)}
    owner2 = ca.ContainedAuthoringOwner(journal=chain.AuthoringChainJournal(path))
    with pytest.raises(ca.AuthoringRefused, match="finished advice"):
        _reserve(owner2, instance, selected, fixture)


def test_open_attempt_at_crash_is_never_regranted_and_refusals_are_durable(actor_state,  # noqa: F811
                                                                          tmp_path):
    instance, selected, fixture = actor_state
    path = str(tmp_path / "chain.jsonl")
    owner = ca.ContainedAuthoringOwner(journal=chain.AuthoringChainJournal(path))
    res = _reserve(owner, instance, selected, fixture)
    del owner  # crash with the attempt open
    state = chain.replay(path)
    assert state.chains[res.advice_digest].stage == "RESERVED"
    assert chain.refresh_prerequisites({"t": (chain.BUILD_PREREQUISITE,)}, state)["t"]
    owner = ca.ContainedAuthoringOwner(journal=chain.AuthoringChainJournal(path))
    with pytest.raises(ca.AuthoringRefused, match="finished advice"):
        _reserve(owner, instance, selected, fixture)


def test_refused_output_records_failed(actor_state, tmp_path):  # noqa: F811
    instance, selected, fixture = actor_state
    path = str(tmp_path / "chain.jsonl")
    owner = ca.ContainedAuthoringOwner(journal=chain.AuthoringChainJournal(path))
    res = _reserve(owner, instance, selected, fixture)
    with pytest.raises(ca.AuthoringRefused):
        owner.accept(res.attempt_id, b"prose", "0" * 64)
    assert chain.replay(path).chains[res.advice_digest].stage == "FAILED"


def test_tampered_or_reordered_journal_refuses_replay(actor_state, tmp_path):  # noqa: F811
    instance, selected, fixture = actor_state
    path = tmp_path / "chain.jsonl"
    owner = ca.ContainedAuthoringOwner(journal=chain.AuthoringChainJournal(str(path)))
    _authored(owner, instance, selected, fixture)
    lines = path.read_bytes().splitlines()
    row = json.loads(lines[1])
    row["patch_bundle_sha256"] = "e" * 64
    (tmp_path / "t.jsonl").write_bytes(lines[0] + b"\n" + chain._canon(row) + b"\n")
    with pytest.raises(chain.ChainRefused):
        chain.replay(str(tmp_path / "t.jsonl"))
    # Hash-valid but stage-skipping journal (COMMITTED before AUTHORED) is refused.
    first = json.loads(lines[0])
    bad = {"schema": chain.SCHEMA, "seq": 1, "kind": "COMMITTED",
           "prev": chain.hashlib.sha256(chain._canon(first)).hexdigest(),
           "advice_digest": first["advice_digest"], "preparation_digest": "a" * 64,
           "source_commit": "b" * 40, "source_tree_digest": "c" * 64,
           "patch_bundle_sha256": "d" * 64}
    bad["sha256"] = chain._self_digest(bad)
    (tmp_path / "s.jsonl").write_bytes(lines[0] + b"\n" + chain._canon(bad) + b"\n")
    with pytest.raises(chain.ChainRefused, match="out of order"):
        chain.replay(str(tmp_path / "s.jsonl"))
