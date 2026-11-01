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
import sys
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


def test_library_family_resolves_import_callable_and_local_helper_aliases():
    from library_substitution import CASES
    for case_id, symbol, source in CASES:
        result = scan_source(source, candidate_functions=['candidate'])
        assert result['verdict'] == 'FAIL', (case_id, result)
        assert symbol in {row['symbol'] for row in result['library_substitution_findings']}
        assert not result['unknown_dispatch'], (case_id, result)
        if case_id != 'library-substitution-hipblaslt':
            assert 'delegated' in result['reachable_functions']


def test_library_family_keeps_allocation_dtype_reference_and_comments_clean():
    from library_substitution import HONEST_SOURCE
    result = scan_source(HONEST_SOURCE, candidate_functions=['candidate'])
    assert result['verdict'] == 'PASS', result
    assert not result['library_substitution_findings']
    assert scan_source(HONEST_SOURCE, candidate_functions=['reference'])['verdict'] == 'FAIL'
    source = 'def candidate(a, b):\n    # torch.matmul(a, b)\n    label = "hipblaslt.hipblasLtMatmul"\n    return a\n'
    assert scan_source(source, candidate_functions=['candidate'])['verdict'] == 'PASS'


def test_dynamic_dispatch_is_unknown_instead_of_clean():
    from library_substitution import DYNAMIC_SOURCE
    samples = (DYNAMIC_SOURCE,
               'def candidate(a, b, callback):\n    return callback(a, b)\n',
               'def candidate(a, b, dispatch, key):\n    return dispatch[key](a, b)\n',
               'import torch\nfrom generated import launch\ndef candidate(a, b, choose):\n'
               '    if choose:\n        compute = torch.matmul\n'
               '    else:\n        compute = launch\n    return compute(a, b)\n')
    for source in samples:
        result = scan_source(source, candidate_functions=['candidate'])
        assert result['verdict'] == 'UNKNOWN', result
        assert result['unknown_dispatch']


def test_direct_from_import_and_module_callable_alias_forms_are_resolved():
    sources = (
        'from torch import matmul as mm\ndef candidate(a, b):\n    return mm(a, b)\n',
        'import torch as t\nmultiply = t.matmul\ndef candidate(a, b):\n    return multiply(a, b)\n',
        'from torch.nn import functional as f\ndef candidate(x):\n    return f.softmax(x)\n',
        'from torch import ops as dispatcher\ndef candidate(x):\n    return dispatcher.aten.softmax(x)\n',
        'import hipblas as vendor\ndef candidate(a, b):\n    return vendor.hipblasSgemm(a, b)\n',
    )
    for source in sources:
        result = scan_source(source, candidate_functions=['candidate'])
        assert result['verdict'] == 'FAIL', result
        assert len(result['library_substitution_findings']) == 1, result


def test_conflicting_module_conditional_aliases_are_unknown_instead_of_last_wins():
    prefix = 'import torch as t\nfrom generated import launch\n'
    cases = (
        prefix + 'if choose:\n    compute = t.matmul\nelse:\n    compute = launch\n',
        prefix + 'compute = t.matmul\nif choose:\n    compute = launch\n',
    )
    for prefix in cases:
        source = prefix + 'def candidate(a, b):\n    return compute(a, b)\n'
        result = scan_source(source, candidate_functions=['candidate'])
        assert result['verdict'] == 'UNKNOWN', result
        assert result['unknown_dispatch']
    same = ('import torch as t\nif choose:\n    compute = t.matmul\n'
            'else:\n    compute = t.matmul\n'
            'def candidate(a, b):\n    return compute(a, b)\n')
    assert scan_source(same, candidate_functions=['candidate'])['verdict'] == 'FAIL'


def test_duplicate_or_nested_helper_definitions_cannot_resolve_by_last_ast_name():
    cases = (
        'import torch as t\ndef helper(a, b):\n    return t.matmul(a, b)\n'
        'def unrelated():\n    def helper(a, b):\n        return a\n'
        'def candidate(a, b):\n    return helper(a, b)\n',
        'import torch as t\ndef helper(a, b):\n    return t.matmul(a, b)\n'
        'def helper(a, b):\n    return a\n'
        'def candidate(a, b):\n    return helper(a, b)\n',
        'def candidate(a, b):\n    def helper(a, b):\n        return a\n'
        '    return helper(a, b)\n',
    )
    for source in cases:
        result = scan_source(source, candidate_functions=['candidate'])
        assert result['verdict'] == 'UNKNOWN', result
        assert any('ambiguous or nested function: helper' == row['symbol']
                   for row in result['unknown_dispatch']), result


def test_owned_candidates_really_dispatch_to_restricted_in_memory_vendor_stubs():
    """Fixture execution only: this is not a HIP execution or speed result."""
    from types import SimpleNamespace
    from library_substitution import CASES, HONEST_SOURCE
    for _case_id, expected, source in (*CASES, ('honest', 'generated.launch', HONEST_SOURCE)):
        trace = []
        class Tensor:
            dtype = 'fixture-f32'
            def to(self, _dtype):
                return self
            def contiguous(self):
                return self
        def stub(symbol):
            def call(*_args, **_kwargs):
                trace.append(symbol)
                return Tensor()
            return call
        modules = {
            'torch': SimpleNamespace(matmul=stub('torch.matmul'),
                                     empty_like=lambda *_a, **_k: Tensor(), float32='fixture-f32'),
            'rocblas': SimpleNamespace(rocblas_sgemm=stub('rocblas.rocblas_sgemm')),
            'hipblaslt': SimpleNamespace(hipblasLtMatmul=stub('hipblaslt.hipblasLtMatmul')),
            'generated': SimpleNamespace(launch=stub('generated.launch')),
        }
        def restricted_import(name, _globals=None, _locals=None, fromlist=(), level=0):
            assert not level and name in modules, f'unapproved fixture import: {name}'
            return modules[name]
        namespace = {'__builtins__': {'__import__': restricted_import}}
        exec(compile(source, '<owned-library-substitution-fixture>', 'exec'), namespace)
        namespace['candidate'](Tensor(), Tensor())
        assert trace == [expected], trace
        result = scan_source(source, candidate_functions=['candidate'])
        assert result['verdict'] == ('PASS' if _case_id == 'honest' else 'FAIL'), result


def _library_report_rows():
    import contextlib
    import io
    import run_falsification as driver
    saved = driver.ROWS
    driver.ROWS = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            driver.run_l1()
        return driver.ROWS
    finally:
        driver.ROWS = saved


def test_native_library_reporter_counts_flags_separately_from_legacy_six_rows():
    import contextlib
    import io
    import json
    import run_falsification as driver
    saved = driver.ROWS
    driver.ROWS = _library_report_rows()
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            driver.conclude(False)
        summary = json.loads(output.getvalue())
        assert summary['library_substitution_flagged'] == 3
        assert summary['library_substitution_expected'] == 3
        assert summary['library_substitution_finding_count'] == 3
        assert sum(row['tier'] == 'L1' for row in driver.ROWS) == 6
        assert sum(row['tier'] == 'L1_library' for row in driver.ROWS) == 4
    finally:
        driver.ROWS = saved


def test_native_reporter_refuses_missing_duplicate_clean_or_uncounted_library_flags():
    import contextlib
    import copy
    import io
    import run_falsification as driver
    rows = _library_report_rows()
    index = next(i for i, row in enumerate(rows) if row['candidate'] == 'library_substitution')
    missing = copy.deepcopy(rows)
    missing.pop(index)
    duplicate = copy.deepcopy(rows)
    duplicate[index + 1] = copy.deepcopy(duplicate[index])
    zero_flag = copy.deepcopy(rows)
    zero_flag[index]['findings'] = []
    unflagged = copy.deepcopy(rows)
    unflagged[index]['verdict'] = 'PASS'
    saved = driver.ROWS
    try:
        for mutation in (missing, duplicate, zero_flag, unflagged):
            driver.ROWS = mutation
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.conclude(False)
            except AssertionError:
                continue
            raise AssertionError('native reporter accepted a broken library flag/count')
    finally:
        driver.ROWS = saved


def test_full_reporter_retains_twenty_five_legacy_rows_with_separate_library_family():
    """Reporter-only fixture rows; no GPU tier or oracle is executed here."""
    import contextlib
    import io
    import json
    import run_falsification as driver
    saved = driver.ROWS
    rows = _library_report_rows()
    rows.append({'tier': 'env'})
    for task in ('layernorm_no_affine', 'softmax_no_maxsub', 'matmul_transpose_no_t'):
        for candidate in ('honest', 'mutant'):
            for tier, arm in (('L2_ghost_replay', 'standard'),
                              ('C2_value_oracle', 'standard'),
                              ('C2_value_oracle', 'adversarial')):
                rows.append({'task': task, 'candidate': candidate,
                             'tier': tier, 'arm': arm, 'verdict': 'PASS'})
    driver.ROWS = rows
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            driver.conclude(True)
        summary = json.loads(output.getvalue())
        assert sum(row['tier'] != 'L1_library' for row in rows) == 25
        assert summary['library_substitution_flagged'] == 3
        assert len(summary['mutants_accepted_by_full_stack']) == 3
    finally:
        driver.ROWS = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} static tests passed")
