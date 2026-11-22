"""AKU-06d3: patch bytes only from authenticated bounded worker output."""
from __future__ import annotations

from dataclasses import replace

import pytest

from . import contained_authoring as ca, source_build_preparation as bridge
from .test_source_build_preparation import (  # noqa: F401 (fixture)
    advice, selected_actor, source_inputs, tiny_source)


def _reserve(owner, instance, selected, fixture, **changes):
    assignment, _manifest, proposal = source_inputs(fixture, selected)
    inputs = dict(selected_actor_work=selected, preparation_result=advice(selected),
                  resolved_campaign=instance.resolved, assignment=assignment,
                  legacy_proposal=proposal)
    inputs.update(changes)
    return owner.reserve(**inputs)


@pytest.fixture
def actor_state(tmp_path, tiny_source):  # noqa: F811
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        yield instance, instance.materialize_actor(outcome), tiny_source
    finally:
        controller.close()


def test_authenticated_output_constructs_bound_native_manifest(actor_state):
    instance, selected, fixture = actor_state
    owner = ca.ContainedAuthoringOwner()
    res = _reserve(owner, instance, selected, fixture)
    brief = res.worker_brief()
    assert brief["execution_authorized"] is False and "worker_key" not in brief
    assert brief["source_context"]["instrument_commit"] == fixture.base
    out = ca.encode_output(res.attempt_id, fixture.patch.encode())
    authored = owner.accept(res.attempt_id, out, ca.sign_output(res.worker_key, res.attempt_id, out))
    assert authored.manifest.patch_bytes == fixture.patch.encode()
    assert authored.manifest.mechanism_id == "threads"
    assert isinstance(authored.bound, bridge.BoundSourcePreparation)
    assert authored.bound.to_dict()["advice"] == advice(selected).proposed_output
    # The owner-assigned scope, not the worker, fixed files/symbols.
    assert authored.manifest.declared_files == tuple(res.scope["declared_files"])
    assert res.advice_digest in owner.finished_advice
    with pytest.raises(ca.AuthoringRefused, match="finished advice"):
        _reserve(owner, instance, selected, fixture)


def test_fresh_attempt_per_reservation_and_no_parallel_reuse(actor_state):
    instance, selected, fixture = actor_state
    owner = ca.ContainedAuthoringOwner()
    res = _reserve(owner, instance, selected, fixture)
    with pytest.raises(ca.AuthoringRefused, match="open authoring attempt"):
        _reserve(owner, instance, selected, fixture)
    owner.fail(res.attempt_id)
    with pytest.raises(ca.AuthoringRefused, match="finished advice"):
        _reserve(owner, instance, selected, fixture)
    other = ca.ContainedAuthoringOwner()
    a, b = (_reserve(o, instance, selected, fixture) for o in (other, ca.ContainedAuthoringOwner()))
    assert a.attempt_id != b.attempt_id and a.worker_key != b.worker_key


@pytest.mark.parametrize("case", ["bad_mac", "other_key", "oversize", "wrong_attempt",
                                  "prose", "out_of_scope"])
def test_unauthenticated_unbounded_or_invented_output_is_refused(actor_state, case):
    instance, selected, fixture = actor_state
    owner = ca.ContainedAuthoringOwner(max_output_bytes=4096)
    res = _reserve(owner, instance, selected, fixture)
    patch = fixture.patch.encode()
    out = ca.encode_output(res.attempt_id, patch)
    key = res.worker_key
    if case == "bad_mac":
        mac = "0" * 64
    else:
        if case == "other_key":
            key = b"k" * 32
        elif case == "oversize":
            out = ca.encode_output(res.attempt_id, patch + b"x" * 8192)
        elif case == "wrong_attempt":
            out = ca.encode_output("aka-other", patch)
        elif case == "prose":
            out = b"please change the fixture addition"
        elif case == "out_of_scope":
            out = ca.encode_output(res.attempt_id, patch.replace(b"a/", b"a/x").replace(b"b/", b"b/x"))
        mac = ca.sign_output(key, res.attempt_id, out)
    with pytest.raises(ca.AuthoringRefused):
        owner.accept(res.attempt_id, out, mac)
    # A refused attempt still finishes its advice; it is never re-granted.
    assert res.advice_digest in owner.finished_advice
    with pytest.raises(ca.AuthoringRefused):
        owner.accept(res.attempt_id, out, mac)


def test_non_advice_and_mismatched_proposal_cannot_reserve(actor_state):
    instance, selected, fixture = actor_state
    owner = ca.ContainedAuthoringOwner()
    bad = replace(advice(selected), request_digest="c" * 64)
    with pytest.raises(ca.AuthoringRefused, match="not accepted source advice"):
        _reserve(owner, instance, selected, fixture, preparation_result=bad)
    _a, _m, proposal = source_inputs(fixture, selected)
    with pytest.raises(ca.AuthoringRefused, match="assignment"):
        _reserve(owner, instance, selected, fixture,
                 legacy_proposal=proposal | {"proposal_id": "akp-other"})
