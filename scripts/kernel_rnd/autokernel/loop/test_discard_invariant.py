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
    "actor_metrics.py": "48f0e9495d7d4ca375cacfa6fd70cd007136ffbbd1205a5ffea3b80871aa0614",
    "../custody.py": "187c9cc71c35c0c94e44430202667505f45e616f7c70790d64a59d159ade1b55",
    "actors.py": "f27344d84427e0f4b1e5567781f585ba3dd8600cf750748627de3c74de644c0f",
    "archive.py": "4a8cdaaf7469ae915358e2f06cfdd7adbb8bbdb2a6186a5641f3ae95aecad6c1",
    "bestof.py": "a42ade5e6075eaff43d5cf065c93015ff2445e80546bed3eb7d2a7f279012959",
    "cross_target.py": "8a4afa59ce62645b23f9f1e02e4bd17e0b7ba1cc997ad25cdcc1aa91e9900409",
    "loop.py": "117f9512f5f22f51b9418db6ad0983a1715166c145853337e6d1887b93878b2c",
    "pipeline.py": "5bb05e3847f54e6582911a88528e627fbc2ac0c799aef6246c461104b68213f1",
    "pool.py": "986925addcd51b1f6ca2942a84c050044dc175ab37b9be2aa8103507ec1565b4",
    "resume.py": "1fb7d04d47ec88fbccf3b5a42a273d84f08ad5e7fe7dc00bd25646adf12b0b4c",
    "run.py": "7c97f4a051504b75f773e6db11f07de4efefd9569eace3f5d785dac591ce84cb",
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
