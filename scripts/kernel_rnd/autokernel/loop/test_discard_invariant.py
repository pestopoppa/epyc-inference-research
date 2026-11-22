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
    "actors.py": "73b59d057aaa143fd333ed8572504a6c639c819c710bf093d185e044410db4bc",  # custody review 2026-11-09: claude binary path only; custody review 2026-11-19 OAB-32/RVP-CTR-1/AK-AMDAHL-PRUNE: default wall budgets (default off; unbounded local backend refused before any call), counter-citation rule (a proposal retried as ProviderTransient, no artifact yet), orchestrator-bundle refusal degrades to inline, abstention gains accepted relies_on_claims, metrics lane field, prompt text; behavioral tests test_an_abstention_survives_its_belief_receipt_either_way + test_an_unbounded_local_critic_is_refused_before_any_agent_call; custody review 2026-11-22 WP3/AK-PM-17-S1/S2 (826571ff1, a2a247f15): Hypothesis carries the planner's optional expected_outcome verbatim; reflexion broadcast appended to the worker prompt (observe-only, prompt text); no artifact or disposition change (test_reflexion.py) | code-budget split 2026-11-22 (OP-AK-CODEBUDGET-43 trim-or-split): verbatim move of _budgets_of to actor_budgets.py, re-exported; no control-flow or disposition change; moved code stays fenced under actor_budgets.py | code-budget split 2026-11-22 (OP-AK-CODEBUDGET-43 trim-or-split): code moved verbatim to actor_budgets.py and re-exported, no behaviour or disposition change; moved code stays fenced under actor_budgets.py
    "archive.py": "45e0c852450561850335be709f0d39b1a10d673f2f07a7f8957c221b46d4e4d0",  # custody review 2026-11-19 AK-HR-3: bind_history_root binds history capture to --belief-root-repo and refuses a conflicting AK_HISTORY_ROOT before compute; record() order unchanged (experiment commits first, history failure only warns)
    "bestof.py": "a42ade5e6075eaff43d5cf065c93015ff2445e80546bed3eb7d2a7f279012959",
    "cross_target.py": "d1cf23b3378ab31972487d42c778397e33ecb35e70f515a09cbec693e997f6e8",  # custody review 2026-11-19 AKX-AUTO-3/PARETO-RETAIN: ppl_contract hold now lifts only on admissible originating-lane evidence (no lane/evidence = hold unchanged, series stays held); eligible() reorders and tags, never drops; behavioral tests test_cross_lane_promotion_wiring.py + test_gated_series_*cross_lane_evidence* in test_cross_target.py
    "loop.py": "a87ca790b67035f518b2d5be7334be4af086a683d5de66ec53bd5a0d1c4316e0",  # custody review 2026-11-07 AutoKernel: PlannerSameStateSkip -> typed planner_same_state_skip outcome (no hypothesis, no artifact; recorded by the normal path) + planner_same_state row field; custody review 2026-11-19 INF06-EXACT-LANE/C5-ISA-1/AK-PM-17: gate wrapped by exact_lane.lane_gate (exact fail/no judge/judge raising = recorded rejection, never a pass) and isa_claims.wrap_gate (harness fault keeps the gate result); reflexion field on the attempt row is observe-only and cannot raise (test_reflexion_is_observe_only_and_never_costs_the_attempt_its_record); Hypothesis.lane / Abstain.relies_on_claims row fields; custody review 2026-11-21 AKPL-P3: HandedOff (a TailRefused) -> typed handed_off outcome carrying the candidate record (no in-process artifact; the snapshot commit + spool record are the hand-off), tail_session may take paths; custody review 2026-11-22 census/WP3/AK-BLOCKED-VS-FAILED (32dc13639, 826571ff1, 918a0a7d2): program_text appends the census journal (missing journal = program.md unchanged); planned expected_outcome merged into the reflexion row (observe-only); a host-caused gate refusal returns typed host_blocked carrying the build checkpoint (a refusal, never a keep; no artifact; test_host_blocked.py) | code-budget split 2026-11-22 (OP-AK-CODEBUDGET-43 trim-or-split): verbatim move of _critic_provenance helpers to critic_provenance.py, re-exported; no control-flow or disposition change; moved code stays fenced under critic_provenance.py | code-budget split 2026-11-22 (OP-AK-CODEBUDGET-43 trim-or-split): code moved verbatim to critic_provenance.py and re-exported, no behaviour or disposition change; moved code stays fenced under critic_provenance.py
    "pipeline.py": "d2323ac6cad89ab8deb966272c51a26593dcf8b9679ce5904bef4722f598869a",  # custody review 2026-11-09: expected_head at validation, no disposition change; custody review 2026-11-19 AK-AMDAHL-PRUNE: SerializedTail demote reorders admission only (demoted candidates still admitted, a raising demote is not a demotion, a superseded one releases admission); make_exact_judge/make_isa_check plumbing; tests in TheTailTimesTheHeaviestKernelFirst; custody review 2026-11-21 AKPL-P3: make_emitting_tail replaces the SerializedTail session per draw (snapshot raising / hand-off raising = contained lane_error, breaker unchanged; tests in test_tailsplit.py failure-injection block); handed_off scratch not marked failed | code-budget split 2026-11-22 (OP-AK-CODEBUDGET-43 trim-or-split): code moved verbatim to pipeline_budget.py and re-exported, no behaviour or disposition change; moved code stays fenced under pipeline_budget.py
    "pool.py": "bfb34a1d82e88ad97b7f543dd84dad581ce487ee17f017b4c67df87a021b2706",  # custody review CUSTODY-REVIEW.md 2026-11-07; custody review 2026-11-19 AK-AMDAHL-PRUNE/INF06-EXACT-LANE/C5-ISA-1: tail_demote, make_exact_judge, make_isa_check plumbing passed through to SerializedTail/run_pool only; no disposition change; custody review 2026-11-21 AKPL-P3: make_emitting_tail plumbing passed through to run_pool only; no disposition change | code-budget split 2026-11-22 (OP-AK-CODEBUDGET-43 trim-or-split): code moved verbatim to pool_commit_message.py and re-exported, no behaviour or disposition change; moved code stays fenced under pool_commit_message.py
    "resume.py": "f166268ed0546172869976df0eefcceabb1f1d05f74afc7f891d336e27c43d7b",  # custody review CUSTODY-REVIEW.md 2026-11-07 (fix)
    "run.py": "6c125d2c3a73b0988f6ecf983ecb63effbb556499198a0b5dd824f421b379014",  # custody review 2026-11-18 AutoKernel: AK-FUSE-GRAPH-KLD-1 appends graph_kld_check to the gate chain for graph-fusion patches (a refusal only, never a keep; no disposition change); custody review 2026-11-19 AKX-AUTO-9/AKX-ALL-14/AKX-AUTO-3: a lone default-route CPU file is admitted or refused through the recorded op_scope verdict path (test_default_route_admission.py); keep_dimensions always recorded (undeclared = all skipped, passed, never a veto; test_keep_dimensions.py) plus gpu_build_receipt evidence; check_one gets store so the cross-lane gate can lift a hold; blast_radius champion_commit and import only; custody review 2026-11-21 AKPL-P3: --legacy-tail / --chain-spool; pooled runs (--workers > 1) hand candidates to tailsplit.EmittingTail (ChainHandoff spools the record, scaffold failure recorded never raised); --workers 1 / --legacy-tail keep the serialized tail; custody review 2026-11-22 census/AK-PM-12/AK-lineage (32dc13639, d7cdc4151, bf2875a3a): planner program via loop.program_text; kept attempts get an idempotent never-raising keep_lesson receipt (a refusal never costs the keep; test_keep_lesson_wire.py); run end publishes the lineage taxonomy receipt, never raises (test_lineage_taxonomy_run_end.py); custody review 2026-11-22 AK-PORT-LB1.1 (f71dc7ab5): gate_for extends both chains with gates.noise_relative_chain_checks (empty unless <store>/noise_relative/<mechanism>.json declares non-bit-exact and rows>1; unreadable record or missing logits = oracle_unavailable refusal, never a pass or a keep; run_all still short-circuits; no artifact or disposition change; test_noise_relative_tolerance.py)
    "actor_budgets.py": "d8cd71030c9f38004dcf4c5cc01f95fac16b739b8515d7e1d6e903fc5ebd5d8c",  # code-budget split 2026-11-22: verbatim from actors.py; fence carried with the moved code
    "critic_provenance.py": "7ec01f0eb9363198cd55623c79df373a36bce45987fe588d152e2dc66a8af214",  # code-budget split 2026-11-22: verbatim from loop.py; fence carried with the moved code
    "pipeline_budget.py": "881889bdaef08a064f98c633ebdf27153b6a6e3467e2940f4fdefc4413275a4e",  # code-budget split 2026-11-22: verbatim from pipeline.py; fence carried with the moved code
    "pool_commit_message.py": "58f955ac3d59a08cc9bc7bc883901c7ebc3f3ea70bfec4d25886de3918402072",  # code-budget split 2026-11-22: verbatim from pool.py; fence carried with the moved code
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
