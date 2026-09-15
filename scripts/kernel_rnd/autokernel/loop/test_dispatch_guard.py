from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

from autokernel.loop import dispatch_guard as D
from autokernel.loop.loop import Hypothesis


def identity(**overrides):
    args = dict(diff="+ int  x = 1;\n", champion="c0", cmake_defines=("-DX=1",),
                bench_recipe={"pairs": 5, "pp": 512}, model="m.gguf", surface="pp512")
    args.update(overrides)
    return D.attempt_identity(**args)


def test_identity_normalizes_whitespace_and_reopens_on_recipe_or_champion():
    assert identity(diff="+ int x = 1;\n") == identity(diff="+   int   x = 1; \n")
    assert identity(champion="c1") != identity()
    assert identity(bench_recipe={"pairs": 6, "pp": 512}) != identity()


def test_answered_identity_refuses_after_fresh_process_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        code = (
            "from pathlib import Path; from autokernel.loop.dispatch_guard import Registry; "
            f"r=Registry(Path({tmp!r})); r.reserve({identity()!r}); "
            f"r.finish({identity()!r},status='measured_null',effect=-.01,epoch='e0'); r.close()")
        package_root = str(Path(__file__).resolve().parents[2])
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = package_root + (
            os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
        subprocess.run([sys.executable, "-c", code], check=True, env=child_env)
        with pytest.raises(D.DispatchRefused) as caught:
            D.Registry(Path(tmp)).reserve(identity())
        assert caught.value.duplicate_of == identity()
        assert caught.value.attempt_identity == identity()
        assert caught.value.prior_effect == -.01
        assert caught.value.prior_epoch == "e0"


def test_non_answer_gets_one_identical_retry_then_closes_infeasible():
    with tempfile.TemporaryDirectory() as tmp:
        registry = D.Registry(Path(tmp))
        assert registry.reserve(identity()).dispatch_count == 1
        registry.finish(identity(), status="bench_failed", effect=None, epoch="e0")
        assert registry.reserve(identity()).dispatch_count == 2
        registry.finish(identity(), status="planner_transient", effect=None, epoch="e0")
        with pytest.raises(D.DispatchRefused, match="closed infeasible") as caught:
            registry.reserve(identity())
        assert caught.value.attempt_identity == identity()


def test_changed_recipe_or_champion_reopens_in_the_persistent_registry():
    """Acceptance is registry behaviour, not merely unequal hash strings."""
    with tempfile.TemporaryDirectory() as tmp:
        registry = D.Registry(Path(tmp))
        original = identity()
        registry.reserve(original)
        registry.finish(original, status="measured_null", effect=0.0, epoch="e0")

        changed_champion = identity(champion="c1")
        changed_recipe = identity(bench_recipe={"pairs": 6, "pp": 512})
        assert registry.reserve(changed_champion).dispatch_count == 1
        assert registry.reserve(changed_recipe).dispatch_count == 1


def test_corrupt_registry_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "dispatch-identity.sqlite3").write_bytes(b"not sqlite")
        with pytest.raises(D.DispatchRefused, match="corrupt"):
            D.Registry(Path(tmp))


def test_characterised_gate_uses_facets_and_epoch_not_statement():
    h = Hypothesis("akm-x", "new prose", "f", "a.cu", "sym")
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "statement": f"different {n}",
             "epoch_sha256": "e0", "status": "measured_null"} for n in range(3)]
    assert "characterised" in D.characterised_reason(
        h, {"epoch_sha256": "e0", "prior_experiments": rows})
    assert D.characterised_reason(h, {"epoch_sha256": "e1", "prior_experiments": rows}) is None


def test_characterised_gate_requires_same_recorded_regime():
    current = {"model": {"path": "/models/m.gguf"}, "quant": "Q4_K",
               "backend": "gpu", "recipe": {"build_recipe": {"id": "r"}},
               "measurement_surface": "tg128"}
    # Exact shape emitted by archive.original_research_scope, including fields which
    # are deliberately not regime identity (model hash, serving arms, request).
    archived = {"model": {"path": "/models/m.gguf", "sha256": "a" * 64},
                "quant": "Q4_K", "backend": "gpu", "measurement_surface": "tg128",
                "recipe": {"build_recipe": {"id": "r"},
                           "original_serving_arms": {"anchor": "old"}},
                "request_digest": "b" * 64}
    h = Hypothesis("akm-x", "p", "f", "a.cu", "sym")
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "epoch_sha256": "e0",
             "status": "measured_null", "research_scope": archived} for _ in range(3)]
    context = {"epoch_sha256": "e0", "current_regime": current,
               "prior_experiments": rows}
    assert "characterised" in D.characterised_reason(h, context)
    context["current_regime"] = {**current, "quant": "IQ2_XXS"}
    assert D.characterised_reason(h, context) is None


def test_characterised_gate_reopens_only_for_a_host_digested_changed_diff():
    old = "1" * 64
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "candidate_diff_sha256": old,
             "epoch_sha256": "e0", "status": "measured_null"} for _ in range(3)]
    h = Hypothesis("akm-x", "same prose", "f", "a.cu", "sym")
    context = {"epoch_sha256": "e0", "prior_experiments": rows}
    assert "characterised" in D.characterised_reason(h, context)
    context["prior_experiments"].append({
        "mechanism_id": "akm-x", "target_surface": "a.cu", "target_symbol": "sym",
        "candidate_diff_sha256": "2" * 64, "epoch_sha256": "e0",
        "status": "superseded"})
    assert D.characterised_reason(h, context) is None


def test_characterised_changed_diff_is_fail_closed_when_nonanswers_disagree():
    old = "1" * 64
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "candidate_diff_sha256": old,
             "epoch_sha256": "e0", "status": "measured_null"} for _ in range(3)]
    for digest in ("2" * 64, "3" * 64):
        rows.append({"mechanism_id": "akm-x", "target_surface": "a.cu",
                     "target_symbol": "sym", "candidate_diff_sha256": digest,
                     "epoch_sha256": "e0", "status": "superseded"})
    h = Hypothesis("akm-x", "p", "f", "a.cu", "sym")
    assert "characterised" in D.characterised_reason(
        h, {"epoch_sha256": "e0", "prior_experiments": rows})


def test_characterised_gate_reopens_for_digest_bound_operator_artifact():
    diff = "1" * 64
    h = Hypothesis("akm-x", "p", "f", "a.cu", "sym")
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "candidate_diff_sha256": diff,
             "epoch_sha256": "e0", "status": "measured_null"} for _ in range(3)]
    body = {"schema": "epyc.autokernel.operator_unblock.v1",
            "gate": "do_not_repeat", "epoch_sha256": "e0",
            "mechanism_id": "akm-x", "target_surface": "a.cu",
            "target_symbol": "sym", "candidate_diff_sha256": diff}
    artifact = {**body, "sha256": hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    context = {"epoch_sha256": "e0", "prior_experiments": rows,
               "operator_unblock_artifacts": [artifact]}
    context["prior_experiments"].append({
        "mechanism_id": "akm-x", "target_surface": "a.cu", "target_symbol": "sym",
        "candidate_diff_sha256": diff, "epoch_sha256": "e0", "status": "superseded"})
    assert D.characterised_reason(h, context) is None
    context["operator_unblock_artifacts"][0]["target_symbol"] = "other"
    assert "characterised" in D.characterised_reason(h, context)


def test_operator_unblock_loader_is_digest_bound_and_fails_closed(tmp_path):
    body = {"schema": "epyc.autokernel.operator_unblock.v1",
            "gate": "do_not_repeat", "epoch_sha256": "e0",
            "mechanism_id": "akm-x", "target_surface": "a.cu",
            "target_symbol": "sym", "candidate_diff_sha256": None}
    valid = {**body, "sha256": hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    path = tmp_path / "unblock.json"
    path.write_text(json.dumps(valid))
    assert D.load_operator_unblocks([path]) == (valid,)
    path.write_text(json.dumps({**valid, "target_symbol": "tampered"}))
    with pytest.raises(D.DispatchRefused, match="schema/digest"):
        D.load_operator_unblocks([path])


def test_characterised_reopen_context_round_trips_through_a_fresh_process(tmp_path):
    """Epoch, changed-diff and operator amendment survive process-local state."""
    old, changed = "1" * 64, "2" * 64
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "candidate_diff_sha256": old,
             "epoch_sha256": "e0", "status": "measured_null"} for _ in range(3)]
    body = {"schema": "epyc.autokernel.operator_unblock.v1",
            "gate": "do_not_repeat", "epoch_sha256": "e0",
            "mechanism_id": "akm-x", "target_surface": "a.cu",
            "target_symbol": "sym", "candidate_diff_sha256": old}
    artifact = {**body, "sha256": hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    cases = [
        {"epoch_sha256": "e1", "prior_experiments": rows},
        {"epoch_sha256": "e0", "prior_experiments": rows + [{
            "mechanism_id": "akm-x", "target_surface": "a.cu", "target_symbol": "sym",
            "candidate_diff_sha256": changed, "epoch_sha256": "e0", "status": "superseded"}]},
        {"epoch_sha256": "e0", "prior_experiments": rows + [{
            "mechanism_id": "akm-x", "target_surface": "a.cu", "target_symbol": "sym",
            "candidate_diff_sha256": old, "epoch_sha256": "e0", "status": "superseded"}],
         "operator_unblock_artifacts": [artifact]},
    ]
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps(cases))
    code = """
import json, sys
from pathlib import Path
from autokernel.loop.dispatch_guard import characterised_reason
from autokernel.loop.loop import Hypothesis
out = []
for context in json.loads(Path(sys.argv[1]).read_text()):
    h = Hypothesis('akm-x', 'p', 'f', 'a.cu', 'sym')
    out.append(characterised_reason(h, context))
print(json.dumps(out))
"""
    package_root = str(Path(__file__).resolve().parents[2])
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = package_root + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
    completed = subprocess.run([sys.executable, "-c", code, str(path)],
                               check=True, capture_output=True, text=True, env=child_env)
    assert json.loads(completed.stdout) == [None, None, None]
