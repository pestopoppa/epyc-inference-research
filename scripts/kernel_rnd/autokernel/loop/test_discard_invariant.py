"""Review fence for custody control flow; behavioral custody tests remain mandatory.

A changed AST in any producer, transfer, or release path requires reviewing every
artifact disposition and adding a failure-injection test before updating the pinned
fingerprint. This conservative fence deliberately includes enclosing conditions.
"""
import ast
import hashlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REVIEWED = {
    "actor_metrics.py": "daf18fa28acc973213351333c897ae20b6c26c68c10be863fa11001965c37f0f",  # custody review 2026-11-19 OAB-34: _summary gains output_capped_empty_rate and output_capped_steps_total report fields only; no disposition change
    "../custody.py": "187c9cc71c35c0c94e44430202667505f45e616f7c70790d64a59d159ade1b55",
    "actors.py": "fecb6230a8da21aae5b11ee43c4834ec998b50cf6f01953fa05b30497d2a84b0",  # custody review 2026-11-09: claude binary path only; custody review 2026-11-19 OAB-32/RVP-CTR-1/AK-AMDAHL-PRUNE: default wall budgets (default off; unbounded local backend refused before any call), counter-citation rule (a proposal retried as ProviderTransient, no artifact yet), orchestrator-bundle refusal degrades to inline, abstention gains accepted relies_on_claims, metrics lane field, prompt text; behavioral tests test_an_abstention_survives_its_belief_receipt_either_way + test_an_unbounded_local_critic_is_refused_before_any_agent_call
    "archive.py": "45e0c852450561850335be709f0d39b1a10d673f2f07a7f8957c221b46d4e4d0",  # custody review 2026-11-19 AK-HR-3: bind_history_root binds history capture to --belief-root-repo and refuses a conflicting AK_HISTORY_ROOT before compute; record() order unchanged (experiment commits first, history failure only warns)
    "bestof.py": "a42ade5e6075eaff43d5cf065c93015ff2445e80546bed3eb7d2a7f279012959",
    "cross_target.py": "d1cf23b3378ab31972487d42c778397e33ecb35e70f515a09cbec693e997f6e8",  # custody review 2026-11-19 AKX-AUTO-3/PARETO-RETAIN: ppl_contract hold now lifts only on admissible originating-lane evidence (no lane/evidence = hold unchanged, series stays held); eligible() reorders and tags, never drops; behavioral tests test_cross_lane_promotion_wiring.py + test_gated_series_*cross_lane_evidence* in test_cross_target.py
    "loop.py": "4ed1bd9456a7d9e2c824706878a4fb4789bef0e98197c1f815b5f6d684c8a2b7",  # custody review 2026-11-07 AutoKernel: PlannerSameStateSkip -> typed planner_same_state_skip outcome (no hypothesis, no artifact; recorded by the normal path) + planner_same_state row field; custody review 2026-11-19 INF06-EXACT-LANE/C5-ISA-1/AK-PM-17: gate wrapped by exact_lane.lane_gate (exact fail/no judge/judge raising = recorded rejection, never a pass) and isa_claims.wrap_gate (harness fault keeps the gate result); reflexion field on the attempt row is observe-only and cannot raise (test_reflexion_is_observe_only_and_never_costs_the_attempt_its_record); Hypothesis.lane / Abstain.relies_on_claims row fields
    "pipeline.py": "8f2bbc485a2ceb5286a005755a6e2b61c295a1e850dc39da5bd94f5530c14afd",  # custody review 2026-11-09: expected_head at validation, no disposition change; custody review 2026-11-19 AK-AMDAHL-PRUNE: SerializedTail demote reorders admission only (demoted candidates still admitted, a raising demote is not a demotion, a superseded one releases admission); make_exact_judge/make_isa_check plumbing; tests in TheTailTimesTheHeaviestKernelFirst
    "pool.py": "2c4c72f98c1ab7a5ecadedf04cc936ff1ee0418f665e311f3581679fa9c9154f",  # custody review CUSTODY-REVIEW.md 2026-11-07; custody review 2026-11-19 AK-AMDAHL-PRUNE/INF06-EXACT-LANE/C5-ISA-1: tail_demote, make_exact_judge, make_isa_check plumbing passed through to SerializedTail/run_pool only; no disposition change
    "resume.py": "f166268ed0546172869976df0eefcceabb1f1d05f74afc7f891d336e27c43d7b",  # custody review CUSTODY-REVIEW.md 2026-11-07 (fix)
    "run.py": "608bfbefce0669702861682b96affb8f3da03f72a35271a0069eae7bd1e662e4",  # custody review 2026-11-18 AutoKernel: AK-FUSE-GRAPH-KLD-1 appends graph_kld_check to the gate chain for graph-fusion patches (a refusal only, never a keep; no disposition change); custody review 2026-11-19 AKX-AUTO-9/AKX-ALL-14/AKX-AUTO-3: a lone default-route CPU file is admitted or refused through the recorded op_scope verdict path (test_default_route_admission.py); keep_dimensions always recorded (undeclared = all skipped, passed, never a veto; test_keep_dimensions.py) plus gpu_build_receipt evidence; check_one gets store so the cross-lane gate can lift a hold; blast_radius champion_commit and import only
    "scratch.py": "22cc3536d195f17b39617982df3116c83b5c9f9c213233d59c8775d06badceb8",
}


def normalized(node):
    if isinstance(node, ast.AST):
        return [type(node).__name__, [[name, normalized(value)]
                for name, value in ast.iter_fields(node) if name != "type_params"]]
    if isinstance(node, list):
        return [normalized(value) for value in node]
    if node is Ellipsis:
        return ["ellipsis"]
    if isinstance(node, bytes):
        return ["bytes", node.hex()]
    if isinstance(node, complex):
        return ["complex", repr(node)]
    return node


def fingerprint(tree):
    data = json.dumps(normalized(tree), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def assert_reviewed(name, tree):
    assert fingerprint(tree) == REVIEWED[name], (
        f"{name}: custody control flow changed; review every produced artifact "
        "and add a behavioral failure-injection test before updating the inventory")


@pytest.mark.parametrize("name", sorted(REVIEWED))
def test_custody_control_flow_requires_review(name):
    assert_reviewed(name, ast.parse((HERE / name).read_text()))


def test_an_unrecorded_new_drop_is_caught():
    tree = ast.parse((HERE / "loop.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_iterate")
    function.body.insert(0, ast.If(test=ast.Constant(True),
                                   body=[ast.Return(value=ast.Constant(None))], orelse=[]))
    with pytest.raises(AssertionError, match="custody control flow changed"):
        assert_reviewed("loop.py", tree)


def test_a_drop_hidden_in_an_existing_condition_is_caught():
    tree = ast.parse((HERE / "loop.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_iterate")
    condition = next(node for node in ast.walk(function) if isinstance(node, ast.If))
    condition.body.insert(0, ast.Assign(
        targets=[ast.Subscript(value=ast.Name(id="progress", ctx=ast.Load()),
                               slice=ast.Constant("critic2"), ctx=ast.Store())],
        value=ast.Constant(None)))
    with pytest.raises(AssertionError, match="custody control flow changed"):
        assert_reviewed("loop.py", tree)



def test_a_new_pipeline_drop_is_caught():
    tree = ast.parse((HERE / "pipeline.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_pool")
    function.body.insert(0, ast.If(test=ast.Constant(True),
                                   body=[ast.Return(value=ast.List(elts=[], ctx=ast.Load()))],
                                   orelse=[]))
    with pytest.raises(AssertionError, match="custody control flow changed"):
        assert_reviewed("pipeline.py", tree)


def test_removing_disposition_record_is_caught():
    tree = ast.parse((HERE / "run.py").read_text())
    recorder = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "record_pooled")
    call = next(node for node in ast.walk(recorder)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "archive" and node.func.attr == "record")
    call.func.attr = "discard"
    with pytest.raises(AssertionError, match="custody control flow changed"):
        assert_reviewed("run.py", tree)


def test_dropping_durable_transfer_from_shared_owner_is_caught():
    tree = ast.parse((HERE / "../custody.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "transfer_tail")
    function.body.insert(0, ast.Return(value=ast.Constant(None)))
    with pytest.raises(AssertionError, match="custody control flow changed"):
        assert_reviewed("../custody.py", tree)


def test_shared_append_rollback_cannot_be_removed_without_review():
    tree = ast.parse((HERE / "../custody.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "append_rows")
    function.body.insert(0, ast.Return(value=ast.Constant(None)))
    with pytest.raises(AssertionError, match="custody control flow changed"):
        assert_reviewed("../custody.py", tree)
