"""Fixture-only checks for the source-selected GPU reference-metric gate."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from autokernel.loop import gates


HELP = ("Usage: test-backend-ops test -o <op> -b <backend> --output <format> "
        "--suite-seed <u64> --autokernel-properties\n")
REFERENCE = (
    "Testing 1 devices\n\nBackend 1/1: ROCm0\n"
    "  MUL_MAT(type_a=q4_K,m=16,n=1,k=256): "
    "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
    "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 OK\n"
    "  1/1 tests passed\n  Backend ROCm0: OK\n1/1 backends passed\nOK\n")


def _run(help_text=HELP, suite=REFERENCE, *, rc=0):
    calls = []

    def invoke(argv, **_kwargs):
        calls.append(tuple(argv))
        if "--help" in argv:
            return SimpleNamespace(returncode=1, stdout=help_text, stderr="")
        return SimpleNamespace(returncode=rc, stdout=suite, stderr="")

    return calls, invoke


def test_selected_gpu_op_emits_existing_per_case_metric_receipt():
    calls, invoke = _run()
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        verdict = gates.op_correctness(Path("/candidate"), op="MUL_MAT",
                                       require_reference=True)
    assert verdict.passed
    assert len(calls) == 2
    assert calls[1][-3:] == ("--suite-seed", "71", "--autokernel-properties")
    receipt = json.loads(verdict.detail)
    assert receipt["worst_metric"] == "test_backend_ops_error/v1"
    assert receipt["metrics"] == ["test_backend_ops_error/v1"]
    assert receipt["cases"] == 1
    assert receipt["observed"] == 2.5e-09
    assert receipt["tolerance"] == 1e-07
    assert receipt["worst_fraction_of_tolerance"] == 0.025


def test_old_instrument_does_not_run_or_misclassify_suite():
    calls, invoke = _run(help_text=HELP.replace("--suite-seed <u64> ", ""))
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        verdict = gates.op_correctness(Path("/candidate"), require_reference=True)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    assert len(calls) == 1


def test_missing_reference_on_passing_suite_is_unavailable():
    missing = REFERENCE.replace(
        "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
        "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 ", "")
    calls, invoke = _run(suite=missing)
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        verdict = gates.op_correctness(Path("/candidate"), require_reference=True)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    assert len(calls) == 2


def test_cpu_candidate_reference_cannot_be_accepted_as_independent():
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.subprocess, "run") as invoke:
        verdict = gates.op_correctness(Path("/candidate"), backend="CPU",
                                       require_reference=True)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    invoke.assert_not_called()


def _is_source_gate(call):
    return isinstance(call, ast.Call) and (
        isinstance(call.func, ast.Name) and call.func.id == "local_op_correctness"
        or isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "gates" and call.func.attr == "op_correctness")


def _assert_reference_forwarder(source):
    wrappers = [node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "local_op_correctness"]
    assert len(wrappers) == 1
    wrapper = wrappers[0]
    assert wrapper.args.vararg.arg == "positional"
    assert wrapper.args.kwarg.arg == "keyword"
    assert len(wrapper.body) == 1 and isinstance(wrapper.body[0], ast.With)
    returns = wrapper.body[0].body
    assert len(returns) == 1 and isinstance(returns[0], ast.Return)
    call = returns[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name) and call.func.value.id == "gates"
    assert call.func.attr == "op_correctness"
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Starred)
    assert isinstance(call.args[0].value, ast.Name) and call.args[0].value.id == "positional"
    assert len(call.keywords) == 1 and call.keywords[0].arg is None
    assert isinstance(call.keywords[0].value, ast.Name) and call.keywords[0].value.id == "keyword"


def _source_reference_branch(source):
    branches = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name) \
                or node.test.id != "cpu_launch":
            continue
        if len(node.body) != 1 or len(node.orelse) != 1:
            continue
        calls = [item for item in ast.walk(node) if _is_source_gate(item)]
        if len(calls) == 2:
            branches.append(node)
    assert len(branches) == 1, "expected the original CPU/GPU source candidate gate"
    return branches[0]


def _assert_source_reference_branch(branch):
    for rows, expected in ((branch.body, False), (branch.orelse, True)):
        calls = [item for row in rows for item in ast.walk(row)
                 if _is_source_gate(item)]
        assert len(calls) == 1
        flags = [kw.value for kw in calls[0].keywords if kw.arg == "require_reference"]
        assert len(flags) == 1 and isinstance(flags[0], ast.Constant)
        assert flags[0].value is expected


def test_live_source_gate_requests_metric_only_for_gpu():
    source = Path(gates.__file__).with_name("run.py").read_text()
    _assert_reference_forwarder(source)
    _assert_source_reference_branch(_source_reference_branch(source))


@pytest.mark.parametrize("arm", ["cpu", "gpu"])
def test_source_reference_guard_rejects_wrong_backend_requirement(arm):
    source = Path(gates.__file__).with_name("run.py").read_text()
    branch = copy.deepcopy(_source_reference_branch(source))
    rows = branch.body if arm == "cpu" else branch.orelse
    flags = [kw for row in rows for item in ast.walk(row)
             if _is_source_gate(item) for kw in item.keywords
             if kw.arg == "require_reference"]
    assert len(flags) == 1
    flags[0].value = ast.Constant(value=arm == "cpu")
    with pytest.raises(AssertionError):
        _assert_source_reference_branch(branch)


@pytest.mark.parametrize("mutation", ["drop_keywords", "override_reference", "wrong_gate"])
def test_source_reference_guard_rejects_forwarder_mutation(mutation):
    tree = ast.parse(Path(gates.__file__).with_name("run.py").read_text())
    wrapper = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "local_op_correctness")
    call = wrapper.body[0].body[0].value
    if mutation == "drop_keywords":
        call.keywords = []
    elif mutation == "override_reference":
        call.keywords.insert(0, ast.keyword(arg="require_reference", value=ast.Constant(False)))
    else:
        call.func.attr = "compiles"
    with pytest.raises(AssertionError):
        _assert_reference_forwarder(ast.unparse(tree))
