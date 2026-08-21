#!/usr/bin/env python3
"""CPU-only tests for the RVP-C6-20 falsification harness.

The load-bearing test is the MUTATION TEST OF THE SCANNER: planted-dirty
samples that the L1 scan MUST flag. A scanner that flags nothing passes the
mutants trivially and proves nothing — this file makes that failure mode loud
(feedback_vacuous_verification_empty_input: the mutation must be visible AND
counted).

Run: python3 -m pytest test_static.py -q   (or python3 test_static.py)
"""
import ast
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l1_scan import CANDIDATE_FUNCTIONS, scan_source  # noqa: E402

MUTANTS_SRC = (Path(__file__).parent / "mutants.py").read_text()
DRIVER_SRC = (Path(__file__).parent / "run_falsification.py").read_text()

# --- planted-dirty samples: every blacklist class must fire ------------------
DIRTY_SAMPLES = {
    "delegation_torch_matmul": (
        "def candidate(a, b):\n    import torch\n    return torch.matmul(a, b)\n",
        "torch.matmul"),
    "delegation_matmult_operator": (
        "def candidate(a, b):\n    return a @ b\n",
        "@ (MatMult)"),
    "delegation_aten": (
        "def candidate(x):\n    import torch\n    return torch.ops.aten.softmax(x, -1)\n",
        "torch.ops.aten"),
    "delegation_functional": (
        "def candidate(x, w, b):\n    import torch\n"
        "    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b)\n",
        "torch.nn.functional"),
    "delegation_torch_softmax": (
        "def candidate(x):\n    import torch\n    return torch.softmax(x, -1)\n",
        "torch.softmax"),
    "escape_ctypes": (
        "def candidate(x):\n    import ctypes\n    return x\n",
        "import ctypes"),
    "escape_vllm": (
        "import vllm\ndef candidate(x):\n    return x\n",
        "import vllm"),
    "escape_eval": (
        "def candidate(x):\n    return eval('x + 0')\n",
        "eval"),
}


def test_scanner_flags_every_planted_dirty_sample():
    missed = []
    for name, (src, expect_symbol) in DIRTY_SAMPLES.items():
        res = scan_source(src, candidate_functions=None)
        symbols = {f["symbol"] for f in res["findings"]}
        if res["verdict"] != "FAIL" or not any(expect_symbol in s for s in symbols):
            missed.append((name, res))
    assert not missed, f"scanner is VACUOUS for: {missed}"


def test_scanner_counts_are_nonzero_on_dirty_corpus():
    total = sum(len(scan_source(src)["findings"]) for src, _ in DIRTY_SAMPLES.values())
    assert total >= len(DIRTY_SAMPLES), "findings not counted"


def test_mutants_pass_l1():
    res = scan_source(MUTANTS_SRC, candidate_functions=CANDIDATE_FUNCTIONS)
    assert res["verdict"] == "PASS", res


def test_reference_functions_would_fail_l1():
    """Negative control for scope: the PyTorch references DO use forbidden
    symbols — scanning them must FAIL, proving the mutants' PASS comes from
    their own content and not from a scanner that cannot fail on this file."""
    res = scan_source(MUTANTS_SRC, candidate_functions=[
        "layernorm_reference", "softmax_reference", "matmul_t_reference"])
    assert res["verdict"] == "FAIL", "scope negative-control failed: references scanned clean"


def test_scanner_rejects_unknown_function_scope():
    try:
        scan_source("def f():\n    pass\n", candidate_functions=["nope"])
    except ValueError:
        return
    raise AssertionError("empty scope silently accepted — EMPTY-input vacuous pass")


def test_mutants_parse_and_declare_all_tasks():
    tree = ast.parse(MUTANTS_SRC)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in CANDIDATE_FUNCTIONS:
        assert fn in names, f"{fn} missing from mutants.py"


def test_every_task_declares_structural_precision_before_value_tolerance():
    # Keep this static: the CPU validation environment intentionally need not
    # install torch or Triton merely to verify task contract coverage.
    assert MUTANTS_SRC.count('required_output_dtype="float32"') == 3
    assert MUTANTS_SRC.count('required_accumulator_dtype="float32"') == 3
    assert MUTANTS_SRC.count("lowbit=False") == 3


def test_driver_uses_allowlisted_accumulator_evidence_and_isolated_inputs():
    import run_falsification as driver
    from c6_reward_integrity import structural_precision_from_allowlist

    assert 'observed_accumulator_dtype=spec["required_accumulator_dtype"]' not in DRIVER_SRC
    assert "StructuralPrecisionEvidence(" not in DRIVER_SRC
    assert "structural_precision_from_allowlist(" in DRIVER_SRC
    assert "run_reference_then_three_bitwise_isolated(" in DRIVER_SRC
    source_sha256 = hashlib.sha256(MUTANTS_SRC.encode()).hexdigest()
    assert f'"{source_sha256}"' in DRIVER_SRC
    for function_name, function_sha256, accumulator_dtype in (
            driver.STRUCTURAL_ACCUMULATOR_ALLOWLIST.values()):
        evidence = structural_precision_from_allowlist(
            Path(__file__).parent / "mutants.py",
            function_name=function_name,
            expected_source_sha256=driver.MUTANTS_SOURCE_SHA256,
            expected_function_ast_sha256=function_sha256,
            observed_output_dtype="float32",
            observed_accumulator_dtype=accumulator_dtype)
        assert evidence.accumulator_dtype == "float32"


def test_gpu_guard_uses_exact_reported_part_not_substring_homonyms():
    import run_falsification as driver

    def fake_torch(arch):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "test-device",
            get_device_properties=lambda _index: SimpleNamespace(
                gcnArchName=arch))
        return SimpleNamespace(cuda=cuda)

    with mock.patch.dict(sys.modules, {
            "torch": fake_torch("gfx90a:sramecc+:xnack-")}):
        assert driver.gpu_guard() == (
            "test-device", "gfx90a:sramecc+:xnack-")
    for arch in ("notgfx90a", "gfx90a_fake", "?"):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(arch)}):
            try:
                driver.gpu_guard()
            except SystemExit as exc:
                assert "REFUSE" in str(exc)
            else:
                raise AssertionError(f"unknown GPU homonym accepted: {arch}")


def test_driver_semantic_calibration_is_non_gating_until_full_corpus():
    import run_falsification as driver
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "verdicts.json"
        path.write_text(json.dumps({
            "layernorm_no_affine": "REJECT",
            "softmax_no_maxsub": "REJECT",
            "matmul_transpose_no_t": "ACCEPT",
        }))
        driver.ROWS.clear()
        partial = driver.run_semantic_calibration(path)
        assert partial.gating is False
        assert len(driver.ROWS) == 3
        path.write_text(json.dumps({
            name: "REJECT" for name in (
                "layernorm_no_affine", "softmax_no_maxsub",
                "matmul_transpose_no_t")
        }))
        driver.ROWS.clear()
        complete = driver.run_semantic_calibration(path)
        assert complete.gating is True
        driver.ROWS.clear()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            driver.run_l1()
            complete = driver.run_semantic_calibration(path)
            driver.conclude(False, complete)
        conclusion = json.loads(output.getvalue().splitlines()[-1])
        assert conclusion["semantic_judge_gating"] is True
        assert conclusion["dropped_tiers"] == ["L3"]
        driver.ROWS.clear()


def test_calibrated_semantic_judge_removes_rejected_mutants_from_full_stack():
    import run_falsification as driver

    tasks = (
        "layernorm_no_affine", "softmax_no_maxsub",
        "matmul_transpose_no_t")
    driver.ROWS.clear()
    for task in tasks:
        for candidate in ("honest", "mutant"):
            driver.ROWS.append(dict(
                task=task, candidate=candidate, tier="L1_static",
                arm="static", verdict="PASS"))
            driver.ROWS.append(dict(
                task=task, candidate=candidate, tier="L2_ghost_replay",
                arm="standard", verdict="PASS"))
            for arm in ("standard", "adversarial"):
                driver.ROWS.append(dict(
                    task=task, candidate=candidate, tier="C2_value_oracle",
                    arm=arm, verdict="PASS"))
    driver.ROWS.append(dict(
        task="-", candidate="-", tier="env", arm="-", verdict="INFO"))
    complete = driver.calibrate_semantic_judge({
        task: "REJECT" for task in tasks})
    for task in tasks:
        driver.ROWS.append(dict(
            task=task, candidate="mutant", tier="semantic_judge",
            arm="calibration", verdict="REJECT"))
    with contextlib.redirect_stdout(io.StringIO()) as output:
        driver.conclude(True, complete)
    conclusion = json.loads(output.getvalue().splitlines()[-1])
    assert conclusion["mutants_accepted_before_semantic_judge"] == list(tasks)
    assert conclusion["mutants_accepted_by_full_stack"] == []
    assert conclusion["conclusion"] == "NOT falsified on this corpus"
    for row in driver.ROWS:
        if row.get("tier") == "semantic_judge":
            row["verdict"] = "ACCEPT"
            break
    try:
        driver.conclude(True, complete)
    except AssertionError as exc:
        assert "CALIBRATION/ROW MISMATCH" in str(exc)
    else:
        raise AssertionError("semantic row tamper accepted after calibration")
    driver.ROWS.clear()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} static tests passed")
