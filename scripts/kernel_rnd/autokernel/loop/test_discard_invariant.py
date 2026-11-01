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
    "loop.py": "add46ea62614dc0f1dffa32fec2d3d9c81753aac6a6ff62e3529fe1d22a723a0",
    "pipeline.py": "642b5ade0bd6dc972fa22ca93a5249889b1f1ba564af248a31e69604bdcadfcf",
    "pool.py": "807fe1b7e6e9f26b9d036b29489fe4a5a5e975b4e63f22129e5e72e77467a23c",
    "resume.py": "ed9c0bbc0bdd584e76fb5abbcfc9ccb80ae4cc0379a5d5531cebb47659a82c3f",
    "run.py": "8b37c71c5a63b60bec97695e675c0ef6aaf2a08856e90dbb9c8b89c83d16bdca",
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
