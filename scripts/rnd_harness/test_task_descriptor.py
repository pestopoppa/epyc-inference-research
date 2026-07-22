#!/usr/bin/env python3
"""Tiny tests for task_descriptor.TaskDescriptor (no inference, pure stdlib).

Proves the two contract-critical properties:
  * validate() ACCEPTS a well-formed descriptor.
  * validate()/load() REJECT a descriptor whose always-valid fallback
    (seed_or_baseline_solution) is missing.
  * fallback() materializes a valid solution.json from an inline baseline
    (baseline-commit-first), and refuses when no baseline can be produced.

Runs under pytest OR as a plain script (`python test_task_descriptor.py`) so it
works in the research venv, which has no pytest installed.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_descriptor as td  # noqa: E402


def _well_formed(**overrides):
    base = dict(
        task_id="kernel-q8-8x8-sweep",
        seed_or_baseline_solution="/mnt/raid0/llm/seed/baseline_kernel",
        run_script="scripts/kernel_rnd/kernel_eval.sh",
        objective_scorer="rnd_harness.scorers:tps_delta",
        is_correct="rnd_harness.gates:byte_coherent_ok",
        target={"compute": "cpu", "hardware": "epyc-9655"},
        dependencies=[],
        objective_direction="max",
    )
    base.update(overrides)
    return td.TaskDescriptor(**base)


def test_validate_accepts_well_formed():
    desc = _well_formed()
    assert desc.validate(strict=True) == []  # no errors, does not raise


def test_validate_rejects_missing_fallback():
    # Empty seed_or_baseline_solution == no always-valid fallback -> reject.
    desc = _well_formed(seed_or_baseline_solution="")
    errors = desc.validate(strict=False)
    assert any("seed_or_baseline_solution" in e for e in errors), errors
    try:
        desc.validate(strict=True)
        raised = False
    except td.TaskDescriptorError:
        raised = True
    assert raised, "strict validate must raise on a missing fallback"


def test_validate_rejects_bad_direction_and_target():
    bad = _well_formed(objective_direction="down", target={"hardware": "x"})
    errors = bad.validate(strict=False)
    assert any("objective_direction" in e for e in errors), errors
    assert any("target" in e for e in errors), errors


def test_load_rejects_unknown_field(tmp_path=None):
    d = tmp_path if tmp_path is not None else _mk_tmpdir()
    path = os.path.join(str(d), "desc.json")
    payload = _well_formed().to_dict()
    payload["bogus"] = 1
    with open(path, "w") as fh:
        json.dump(payload, fh)
    try:
        td.load(path)
        raised = False
    except td.TaskDescriptorError as exc:
        raised = "unknown" in str(exc).lower()
    assert raised, "load must reject unknown descriptor fields"


def test_fallback_materializes_baseline(tmp_path=None):
    d = tmp_path if tmp_path is not None else _mk_tmpdir()
    ws = os.path.join(str(d), "workspace")
    desc = _well_formed(baseline_solution_json={"params": {"threads": 96}, "note": "baseline"})
    sol = desc.fallback(ws)
    assert os.path.isfile(sol), "fallback() must create a solution.json"
    with open(sol) as fh:
        data = json.load(fh)
    assert data["params"]["threads"] == 96


def test_fallback_refuses_without_any_baseline(tmp_path=None):
    d = tmp_path if tmp_path is not None else _mk_tmpdir()
    ws = os.path.join(str(d), "ws2")
    # seed path does not exist and no inline baseline -> cannot guarantee validity.
    desc = _well_formed(
        seed_or_baseline_solution="/nonexistent/seed/dir",
        baseline_solution_json=None,
    )
    try:
        desc.fallback(ws)
        raised = False
    except td.TaskDescriptorError:
        raised = True
    assert raised, "fallback must refuse to start a search with no valid baseline"


def test_emit_schema_is_wellformed():
    schema = td.emit_json_schema()
    assert schema["type"] == "object"
    for req in ("task_id", "seed_or_baseline_solution", "run_script",
                "objective_scorer", "is_correct"):
        assert req in schema["required"], req


def _mk_tmpdir():
    return tempfile.mkdtemp(prefix="rnd_harness_test_")


def _run_all():
    """Plain-script runner (used when pytest is not installed)."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
