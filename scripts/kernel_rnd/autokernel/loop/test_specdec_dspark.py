"""DSpark speculative decoding must be EXPRESSIBLE as a recipe, and must refuse the
three ways it silently measured the wrong thing on 2026-09-23.

Before this, `SPECULATION_TYPES` named only {none, draft-dflash, draft-mtp}, so the
prepared DS41 campaign recipe was forced to `spec_decode: {"type": "none"}` -- it would
have optimised the NO-DRAFTER surface we have already beaten by 1.56x.

The three refusals, each tied to a measured failure:

* `--parallel 1` -- the server refuses draft-dspark at more than one slot, so a recipe
  that asks for np>1 is refused rather than silently overridden (a rewritten np is a
  different measured condition wearing this recipe's hash).
* `LLAMA_SPEC_EXACT` -- unset, a greedy launch falls back to the SERIAL verification
  path: exactly one token per target decode, so never above 1x. Measured 7.18 t/s with a
  drafter against an 11.43 t/s no-drafter control. This is the failure that cost a
  measurement, and it cost it SILENTLY.
* the drafter itself -- half the identity of a speculative measurement.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from . import resolved_recipe as rr
from . import serving


BUILD = Path("/build")
DRAFTER = "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf"
TARGET = "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash.gguf"
EXACT = "batched-greedy-inexact"


def _artifact(role: str, path: str, digit: str) -> dict:
    return {"schema": rr.ARTIFACT_SCHEMA, "role": role, "path": path, "sha256": digit * 64}


def _artifacts(recipe: serving.Recipe, *, build: Path = BUILD) -> dict:
    drafter = recipe.spec_decode.get("drafter")
    return {"model": _artifact("model", recipe.model, "a"),
            "drafter": (_artifact("drafter", drafter, "b") if drafter else None),
            "executable": _artifact("executable", str(build / "bin" / "llama-server"), "c"),
            "dsos": [_artifact("dso", str(build / "bin" / "libggml.so"), "d")]}


def _policy(*keys: str, witness: str = "process_environ") -> dict:
    return {"schema": rr.ENVIRONMENT_POLICY_SCHEMA, "version": "ds41-v1",
            "measurement_keys": list(keys), "allowed_inherit_keys": [],
            "witnesses": {key: witness for key in keys}}


def _recipe(**kw) -> serving.Recipe:
    """The MEASURED operating point: CPU, block 2, greedy, batched verification."""
    base = dict(
        name="ds41-flash-dspark-b2-cpu", model=TARGET, device="none", ngl=0,
        spec_decode={"type": "draft-dspark", "drafter": DRAFTER, "ngld": 0,
                     "draft_n_max": 2},
        np=1, ctx=4096, threads=96, batch=2048, ubatch=512, ctk="f16", ctv="f16", fa="on",
        extra_flags=("--device-draft", "none"),
        n_predict=256, temperature=0.0, top_p=1.0, top_k=1,
        env={serving.SPEC_EXACT_ENV: EXACT})
    base.update(kw)
    return serving.Recipe(**base)


def _resolve(recipe: serving.Recipe, *, backend="cpu", policy=None, artifacts=None,
             build=BUILD, port=18317):
    return rr.resolve_recipe(
        recipe, build_dir=build,
        artifact_identities=artifacts if artifacts is not None else _artifacts(recipe, build=build),
        backend=backend, environment_policy=policy or _policy(serving.SPEC_EXACT_ENV),
        inherited_environment={}, port=port)


# --------------------------------------------------------------------------- expressible

def test_dspark_is_a_recipe_type_at_all():
    assert "draft-dspark" in rr.SPECULATION_TYPES


def test_operating_point_resolves_and_emits_exactly_the_measured_argv_and_env():
    resolved = _resolve(_recipe())
    assert resolved.capability.supported, [r.to_dict() for r in resolved.capability.reasons]
    assert resolved.capability.speculation == "external_draft"
    assert list(resolved.argv) == [
        "/build/bin/llama-server",
        "-m", TARGET, "-np", "1", "-c", "4096",
        "-t", "96", "-tb", "96", "-b", "2048", "-ub", "512",
        "-ctk", "f16", "-ctv", "f16", "--device", "none", "-ngl", "0", "-fa", "on",
        "--host", "127.0.0.1", "--port", "18317", "--metrics", "--slots",
        "-md", DRAFTER, "-ngld", "0",
        "--spec-type", "draft-dspark", "--spec-draft-n-max", "2",
        "--no-kv-unified", "--device-draft", "none"]
    assert dict(resolved.launch_env) == {"LD_LIBRARY_PATH": "/build/bin",
                                         serving.SPEC_EXACT_ENV: EXACT}
    assert dict(resolved.relevant_environment) == {serving.SPEC_EXACT_ENV: EXACT}
    assert resolved.absent_environment == ()


def test_block_three_at_temperature_zero_seven_is_expressible_and_needs_no_exactness_env():
    """The 44.4%/10.26 t/s point sampled at temp 0.7 -- not greedy, so the verification
    path is not selectable and the env is not required. Guard against a blanket rule."""
    recipe = _recipe(spec_decode={"type": "draft-dspark", "drafter": DRAFTER, "ngld": 0,
                                  "draft_n_max": 3},
                     temperature=0.7, top_p=0.95, top_k=20, env=None)
    resolved = _resolve(recipe, policy=_policy())
    assert resolved.capability.supported
    assert resolved.argv[resolved.argv.index("--spec-draft-n-max") + 1] == "3"


# --------------------------------------------------------------------------- np > 1

def test_np_greater_than_one_refuses_rather_than_overriding():
    with pytest.raises(rr.ResolutionError, match="requires --parallel 1"):
        _resolve(_recipe(np=4))


def test_np_refusal_is_not_vacuous_for_other_speculation_types():
    """Control: draft-dflash at np=4 is still perfectly legal, so the refusal above is
    about draft-dspark and not about np."""
    resolved = _resolve(_recipe(np=4, env=None,
                                spec_decode={"type": "draft-dflash", "drafter": DRAFTER,
                                             "ngld": 0, "draft_n_max": 2}),
                        policy=_policy())
    assert resolved.capability.supported


def test_resolved_argv_is_rechecked_for_parallel_one_independently_of_the_template():
    resolved = _resolve(_recipe())
    argv = list(resolved.argv)
    argv[argv.index("-np") + 1] = "4"
    tampered = replace(resolved, argv=tuple(argv))
    with pytest.raises(rr.ResolutionError, match="--parallel != 1"):
        rr._validate_resolved_consistency(tampered)


# --------------------------------------------------------------------------- exactness env

def test_greedy_dspark_without_the_exactness_env_is_a_refusal_not_a_fallback():
    with pytest.raises(rr.ResolutionError, match="SERIAL verification path"):
        _resolve(_recipe(env=None), policy=_policy())


def test_greedy_by_top_k_one_also_requires_the_exactness_env():
    with pytest.raises(rr.ResolutionError, match="SERIAL verification path"):
        _resolve(_recipe(env=None, temperature=0.6, top_k=1), policy=_policy())


def test_explicitly_unsetting_the_exactness_env_is_also_a_refusal_for_greedy():
    with pytest.raises(rr.ResolutionError, match="SERIAL verification path"):
        _resolve(_recipe(env=None, explicit_unsets=(serving.SPEC_EXACT_ENV,)),
                 policy=_policy(serving.SPEC_EXACT_ENV))


def test_an_unknown_verification_path_is_refused():
    with pytest.raises(rr.ResolutionError, match="not a verification path"):
        _resolve(_recipe(env={serving.SPEC_EXACT_ENV: "serial-greedy-exact"}))


def test_declaring_the_exactness_env_with_no_speculation_is_refused():
    with pytest.raises(rr.ResolutionError, match="speculation type 'none'"):
        _resolve(_recipe(spec_decode={"type": "none"}, extra_flags=()))


# --------------------------------------------------------------------------- the witness

def test_the_exactness_witness_is_declared_and_says_what_it_records():
    resolved = _resolve(_recipe())
    witness, = resolved.capability.witnesses
    assert (witness.key, witness.kind, witness.status) == (
        serving.SPEC_EXACT_ENV, "process_environ", "declared")
    assert (witness.field, witness.expected) == (f"environ:{serving.SPEC_EXACT_ENV}", EXACT)


def test_a_policy_that_cannot_witness_the_exactness_key_makes_the_recipe_unsupported():
    """`runtime_set` has no sampler, so it can never prove the process got the value.
    Naming the key is not witnessing it."""
    resolved = _resolve(_recipe(), policy=_policy(serving.SPEC_EXACT_ENV,
                                                  witness="runtime_set"))
    assert not resolved.capability.supported
    assert "spec_exactness_witness_missing" in {r.code for r in resolved.capability.reasons}


def test_the_witness_expectation_is_cross_checked_against_the_frozen_launch():
    resolved = _resolve(_recipe())
    witness, = resolved.capability.witnesses
    tampered = replace(
        resolved,
        capability=replace(resolved.capability,
                           witnesses=(replace(witness, expected="something-else"),)))
    with pytest.raises(rr.ResolutionError, match="process_environ witness disagrees"):
        rr._validate_resolved_consistency(tampered)


def test_process_environ_witness_refuses_in_both_directions(tmp_path):
    recipe = _recipe()
    expectations = ((f"environ:{serving.SPEC_EXACT_ENV}", EXACT),)
    blob = f"PATH=/usr/bin\0{serving.SPEC_EXACT_ENV}={EXACT}\0"
    assert serving.verify_process_environ(recipe, 1, expectations=expectations,
                                          environ_text=blob) == {
        f"environ:{serving.SPEC_EXACT_ENV}": EXACT}
    with pytest.raises(serving.EnvReadbackFailed):
        serving.verify_process_environ(recipe, 1, expectations=expectations,
                                       environ_text="PATH=/usr/bin\0")
    # the control direction: a recipe that UNSETS the knob must not find it set
    with pytest.raises(serving.EnvReadbackFailed):
        serving.verify_process_environ(
            recipe, 1,
            expectations=((f"environ:{serving.SPEC_EXACT_ENV}", serving.UNSET),),
            environ_text=blob)


def test_process_environ_witness_refuses_an_unreadable_proc():
    with pytest.raises(serving.EnvReadbackFailed, match="unwitnessed"):
        serving.verify_process_environ(
            _recipe(), 999999999,
            expectations=((f"environ:{serving.SPEC_EXACT_ENV}", EXACT),))


# --------------------------------------------------------------------------- the drafter

def test_dspark_without_a_drafter_is_refused_because_it_is_never_self_drafting():
    with pytest.raises(rr.ResolutionError, match="needs a separate drafter GGUF"):
        _resolve(_recipe(spec_decode={"type": "draft-dspark", "draft_n_max": 2}))


def test_the_drafter_is_part_of_the_identity_by_path_and_by_content():
    base = _resolve(_recipe())
    moved = _resolve(_recipe(spec_decode={"type": "draft-dspark",
                                          "drafter": "/other/drafter.gguf", "ngld": 0,
                                          "draft_n_max": 2}))
    # The PATH moves the recipe hash and the snapshot, and deliberately NOT the execution
    # digest: that one is relocation-invariant by construction (`-md` is normalised to
    # `<drafter:sha256>`), so a drafter moved on disk is the same measurement.
    assert moved.template_hash != base.template_hash
    assert moved.snapshot_digest != base.snapshot_digest
    assert moved.execution_digest == base.execution_digest
    artifacts = _artifacts(_recipe())
    artifacts["drafter"]["sha256"] = "e" * 64
    rebuilt = _resolve(_recipe(), artifacts=artifacts)
    # SAME path, SAME template hash -- only the drafter BYTES changed, and the execution
    # identity must still move. A recipe hash alone cannot see this.
    assert rebuilt.template_hash == base.template_hash
    assert rebuilt.execution_digest != base.execution_digest


def test_block_size_is_part_of_the_identity():
    base = _resolve(_recipe())
    block_five = _resolve(_recipe(spec_decode={"type": "draft-dspark", "drafter": DRAFTER,
                                               "ngld": 0, "draft_n_max": 5}))
    assert block_five.execution_digest != base.execution_digest


def test_launch_refuses_a_missing_or_unreadable_drafter(tmp_path):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "llama-server").write_text("#!/bin/true\n")
    drafter = tmp_path / "drafter.gguf"
    drafter.write_bytes(b"GGUF")
    recipe = _recipe(spec_decode={"type": "draft-dspark", "drafter": str(drafter),
                                  "ngld": 0, "draft_n_max": 2})
    resolved = _resolve(recipe, build=build, port=18317)
    resolved.validate_launch(recipe, build, 18317)  # present and readable: admitted
    drafter.unlink()
    with pytest.raises(rr.ResolutionError, match="missing or unreadable"):
        resolved.validate_launch(recipe, build, 18317)


def test_launch_refuses_a_drafter_that_is_a_directory(tmp_path):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "llama-server").write_text("#!/bin/true\n")
    drafter = tmp_path / "drafter.gguf"
    drafter.mkdir()
    recipe = _recipe(spec_decode={"type": "draft-dspark", "drafter": str(drafter),
                                  "ngld": 0, "draft_n_max": 2})
    resolved = _resolve(recipe, build=build, port=18317)
    with pytest.raises(rr.ResolutionError, match="missing or unreadable"):
        resolved.validate_launch(recipe, build, 18317)


# --------------------------------------------------------------------------- round trip

def test_the_dspark_launch_round_trips_through_its_serialized_form():
    resolved = _resolve(_recipe())
    assert rr.resolved_recipe_from_dict(resolved.to_dict()) == resolved


def test_a_cpu_dspark_recipe_with_gpu_draft_placement_is_unsupported():
    resolved = _resolve(_recipe(extra_flags=("--device-draft", "ROCm0")))
    assert not resolved.capability.supported
    assert "mixed_cpu_gpu_draft_unsupported" in {r.code for r in resolved.capability.reasons}
