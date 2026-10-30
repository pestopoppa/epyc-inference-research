"""The shared T0 defaults preserve campaign values without loading its driver."""

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_fresh_live_loop_import_does_not_load_campaign_driver():
    package = __package__.rsplit(".", 1)[0]
    code = (
        "import importlib,sys; "
        f"sys.path[:0] = {sys.path!r}; "
        f"importlib.import_module({package + '.loop.run'!r}); "
        f"assert {package + '.campaign'!r} not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


@pytest.mark.parametrize(
    "caller",
    [
        "loop.runtime_admission",
        "loop.direct_historical_control",
        "loop.direct_gpu_control",
        "execution.live_controls",
    ],
)
def test_fresh_actual_runtime_caller_import_does_not_load_campaign(caller):
    package = __package__.rsplit(".", 1)[0]
    code = (
        "import importlib,sys; "
        f"sys.path[:0] = {sys.path!r}; "
        f"importlib.import_module({package + '.loop.run'!r}); "
        f"importlib.import_module({package + '.' + caller!r}); "
        f"assert {package + '.campaign'!r} not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_shared_tools_preserve_original_recipe_root():
    from . import recipes, t0_policy
    from .. import campaign

    tools = t0_policy.t0_tools()
    assert dataclasses.asdict(tools) == {
        "bash": "/bin/bash",
        "cmake": "/usr/bin/cmake",
        "git": None,
        "verify_ggml_linkage_sh": str(recipes.REPO_ROOT / "scripts/utils/verify_ggml_linkage.sh"),
    }
    assert tools == campaign.HostOps._t0_tools()


@pytest.mark.parametrize("backend", ["llama_cpu", "llama_gpu"])
def test_fixed_policy_outputs_and_legacy_campaign_entry(backend):
    from . import t0_policy
    from .. import campaign

    policy = t0_policy.t0_evaluator_policy(backend)
    assert json.loads(json.dumps(dataclasses.asdict(policy))) == {
        "required_backend_ops": ["MUL_MAT", "MUL_MAT_ID"],
        "symbol_shrinkage_reject_ratio": 0.6,
        "diff_ceiling": {
            "backend": backend,
            "max_changed_lines": 400,
            "max_files_touched": 10,
            "shared_core_forces_review": True,
        },
        "determinism_min_runs": 2,
        "coherence_tolerance_floor": 0.98,
        "policy_ref": "ak-policy/v1",
    }
    assert policy == campaign.HostOps._t0_evaluator_policy(SimpleNamespace(backend=backend))


def test_generation_output_and_legacy_campaign_entry():
    from . import t0_policy
    from .. import campaign

    plan = t0_policy.t0_generation_plan("/MOCK_ONLY/model.gguf")
    assert json.loads(json.dumps(dataclasses.asdict(plan))) == {
        "prompt": "The capital of France is",
        "prompt_ref": "ak-prompt-001",
        "n_predict": 32,
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
        "threads": None,
        "extra_argv": ["-m", "/MOCK_ONLY/model.gguf"],
        "timeout_s": 1800.0,
    }
    assert plan == campaign.HostOps._t0_generation_plan(
        SimpleNamespace(model="/MOCK_ONLY/model.gguf")
    )


@pytest.mark.parametrize(
    "values",
    [(899.95, 894.70, 867.16, 886.16), (52.76, 52.31, 51.62, 50.52), (1, 1, 1), (1, 2, 0.5)],
)
def test_consecutive_arithmetic_keeps_original_public_functions(values):
    from . import anchor_drift
    from .. import campaign

    expected = tuple(
        (float(values[i + 1]) - float(values[i])) / float(values[i]) for i in range(len(values) - 1)
    )
    assert anchor_drift.adjacent_relative_steps(values) == expected
    assert anchor_drift.drift_bound_from(values) == max(abs(v) for v in expected)
    assert campaign.adjacent_relative_steps is anchor_drift.adjacent_relative_steps
    assert campaign.drift_bound_from is anchor_drift.drift_bound_from
    assert campaign.drift_bound_from.__annotations__ == {"series": "Sequence[float]", "return": "float"}
    assert campaign.adjacent_relative_steps.__annotations__ == {"series": "Sequence[float]", "return": "tuple"}


def test_all_new_leaf_bytes_enter_actual_control_fingerprints(tmp_path, monkeypatch):
    from . import anchor_drift, t0_policy
    from .. import production_pins
    from ..loop import direct_gpu_control, direct_historical_control, runtime_admission
    import hashlib

    for caller in (direct_gpu_control, direct_historical_control, runtime_admission):
        baseline = caller.source_identity()
        recorded = baseline["files"]
        for leaf in (production_pins, anchor_drift, t0_policy):
            digest = hashlib.sha256(Path(leaf.__file__).read_bytes()).hexdigest()
            assert digest in recorded.values(), (caller.__name__, leaf.__name__)
            changed = tmp_path / Path(leaf.__file__).name
            changed.write_bytes(
                Path(leaf.__file__).read_bytes() + b"\n# MOCK_ONLY source mutation\n"
            )
            with monkeypatch.context() as patch:
                patch.setattr(leaf, "__file__", str(changed))
                assert caller.source_identity() != baseline
                assert (
                    hashlib.sha256(changed.read_bytes()).hexdigest()
                    in caller.source_identity()["files"].values()
                )


def test_declared_pins_keep_every_original_value_and_campaign_export():
    from .. import campaign, production_pins

    expected = {
        "PRODUCTION_REPO": "/mnt/raid0/llm/llama.cpp",
        "PRODUCTION_BRANCH": "production-consolidated-v10",
        "PRODUCTION_COMMIT": "ffc1bac82eeca6f9099e1ccd9ba49703c460a115",
        "MEASUREMENT_REPO": "/mnt/raid0/llm/llama.cpp",
        "MEASUREMENT_BRANCH": "production-consolidated-v10",
        "MEASUREMENT_COMMIT": "ffc1bac82eeca6f9099e1ccd9ba49703c460a115",
        "MEASUREMENT_BUILD_ROOT": "/mnt/raid0/llm/tmp/build-ak-t0-cpu-ffc1bac82",
        "POST_T0_QUIET_BARRIER_S": 65.0,
        "POST_T0_QUIET_SAMPLES": 3,
        "POST_T0_QUIET_SAMPLE_INTERVAL_S": 5.0,
        "IQK_MATCHED_PAIR_REPS": 1,
    }
    assert {k: getattr(production_pins, k) for k in expected} == expected
    assert {k: getattr(campaign, k) for k in expected} == expected


def test_actual_runtime_gate_uses_the_shared_backend_policy(monkeypatch):
    from . import t0_policy
    from ..loop import runtime_admission as runtime

    class PolicyEntered(Exception):
        pass

    calls = []
    original = t0_policy.t0_evaluator_policy

    def enter(backend):
        calls.append(original(backend))
        raise PolicyEntered

    monkeypatch.setattr(t0_policy, "t0_evaluator_policy", enter)
    gates = runtime.DirectGates(rows=(), operations=(), raw_ref="MOCK_ONLY", original_prompts=None)
    with pytest.raises(PolicyEntered):
        gates.run_gates(SimpleNamespace(backend="llama_gpu"))
    assert calls == [original("llama_gpu")]


def test_actual_historical_t0_plan_uses_shared_tools_and_generation(tmp_path, monkeypatch):
    from . import t0_policy
    from ..execution import live_controls as lc
    from ..loop import direct_historical_control as historical

    monkeypatch.setattr(lc, "INSTRUMENT_ROOT", tmp_path / "MOCK_ONLY-source")
    monkeypatch.setattr(lc, "INSTRUMENT_BINARY", tmp_path / "MOCK_ONLY-build/bin/llama-bench")
    plan = historical._t0_plan(hashlib.sha256(b"MOCK_ONLY source").hexdigest(), arm="anchor")
    assert plan.tools == t0_policy.t0_tools()
    assert plan.generation == t0_policy.t0_generation_plan(str(lc.MODEL))
    assert plan.parameter_env == (("GGML_IQK", "0"),)


def test_actual_gpu_t0_plan_preserves_its_gpu_only_overrides():
    from . import t0_policy
    from ..execution import t0_provider as tp
    from ..loop import direct_gpu_control as gpu

    frame = {
        "source_root": "/MOCK_ONLY/source",
        "anchor_build": "/MOCK_ONLY/anchor",
        "candidate_build": "/MOCK_ONLY/candidate",
        "anchor_commit": hashlib.sha1(b"MOCK_ONLY anchor").hexdigest(),
        "candidate_commit": hashlib.sha1(b"MOCK_ONLY candidate").hexdigest(),
        "model": "/MOCK_ONLY/model.gguf",
    }
    plan = gpu._t0_plan(
        frame,
        {"candidate": {"source_sha256": hashlib.sha256(b"MOCK_ONLY source").hexdigest()}},
        "candidate",
    )
    assert plan.tools == t0_policy.t0_tools()
    assert plan.generation == tp.GenerationPlan(
        prompt="The capital of France is",
        prompt_ref="ak-prompt-001",
        n_predict=32,
        seed=42,
        threads=8,
        extra_argv=("-m", frame["model"], "-ngl", "99"),
    )


def test_actual_live_anchor_authority_uses_the_unchanged_shared_drift(tmp_path):
    from . import anchor_drift
    from ..execution import live_controls as lc

    values = tuple(float(i + 1) for i in range(lc.ANCHOR_MOTION_WINDOW_BLOCKS))
    material = SimpleNamespace(
        label=lc.ANCHOR_MOTION_LABEL,
        blocks=tuple(SimpleNamespace(anchor_samples=(value,)) for value in values),
    )
    raw = tmp_path / "raw" / (lc.ANCHOR_MOTION_LABEL + ".json")
    raw.parent.mkdir()
    raw.write_text('{"MOCK_ONLY": true}')
    authority = lc._anchor_motion_authority(
        tmp_path, material=material, settling_receipt={"receipt_sha256": "d" * 64}
    )
    assert authority["anchor_medians"] == list(values)
    assert authority["bound"] == anchor_drift.drift_bound_from(values)


def test_actual_campaign_evaluator_seal_includes_each_new_source_leaf():
    from . import anchor_drift, t0_policy
    from .. import campaign, production_pins

    paths = campaign.HostOps._evaluator_bundle_files()
    for leaf in (production_pins, anchor_drift, t0_policy):
        assert Path(leaf.__file__) in paths
    assert len(paths) == len(set(paths))
