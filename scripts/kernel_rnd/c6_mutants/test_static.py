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
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l1_scan import CANDIDATE_FUNCTIONS, scan_source  # noqa: E402

MUTANTS_SRC = (Path(__file__).parent / "mutants.py").read_text()

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} static tests passed")
