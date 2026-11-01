"""Opt-in readiness regressions: intentionally red until U3 gaps are implemented.

Run explicitly; excluded from normal pytest discovery. No process or inference is launched.
"""
import json
import unittest
from pathlib import Path
from scripts.kernel_rnd.autokernel.loop import serving, resolved_recipe as rr, runtime_arms, actors

BUILD = Path("/build")
def recipe(**kw):
    return serving.Recipe(name="u3", model="/models/main.gguf", **kw)
def artifact(role, path, digit):
    return {"schema": rr.ARTIFACT_SCHEMA, "role": role, "path": path, "sha256": digit * 64}
def resolve(r, additional_artifacts=None):
    draft = r.spec_decode.get("drafter")
    arts = {"model": artifact("model", r.model, "a"),
            "drafter": artifact("drafter", draft, "b") if draft else None,
            "executable": artifact("executable", "/build/bin/llama-server", "c"),
            "dsos": [artifact("dso", "/build/bin/libggml.so", "d")]}
    arts.update(additional_artifacts or {})
    policy = {"schema": rr.ENVIRONMENT_POLICY_SCHEMA, "version": "u3",
              "measurement_keys": list((r.env or {}).keys()), "allowed_inherit_keys": [],
              "witnesses": {k: "runtime_set" for k in (r.env or {})}}
    return rr.resolve_recipe(r, build_dir=BUILD, artifact_identities=arts, backend="gpu",
                             environment_policy=policy, inherited_environment={}, port=18311)
def canonical_anchor():
    template = recipe(env={"U3_A": "0", "U3_B": "0"})
    base = resolve(template)
    return rr.resolve_canonical_launch(
        template, build_dir=BUILD, command_argv=template.server_argv(BUILD, 18311),
        topology_prefix=(), launch_environment=dict(base.launch_env),
        artifact_identities={"model": base.model.to_dict(), "drafter": None,
            "executable": base.executable.to_dict(), "dsos": [x.to_dict() for x in base.dsos]},
        backend="gpu", environment_policy=base.environment_policy, port=18311,
        runtime_binary_dir=None, runtime_ld_paths=(), provenance={"export_sha256":"e" * 64,
            "instance_mode":"full", "source:fixture":"offline"})
def env_arm(candidate):
    return runtime_arms.RuntimeArm.from_dict({"arm_id":"u3_env", "kind":"env",
        "candidate":candidate, "numerics":"bit_exact", "rationale":"compare environment"})
def env_pair(arm):
    base = canonical_anchor()
    return actors._runtime_pair(arm.treatment(), {"runtime_anchor": base.to_dict(),
                    "runtime_env_keys": ["U3_A", "U3_B"]}, "u3_env")
def checked(r, expected):
    roundtrip = serving.Recipe.from_dict(r.to_dict())
    assert roundtrip.recipe_hash == r.recipe_hash
    argv = roundtrip.server_argv(BUILD, 18311)
    for token in expected:
        assert token in argv, (token, argv)
    for i, token in enumerate(expected[:-1]):
        if token.startswith("-") and not expected[i + 1].startswith("-"):
            assert argv[argv.index(token) + 1] == expected[i + 1], (token, argv)
    resolved = resolve(roundtrip)
    assert resolved.capability.supported, [(x.code, x.detail) for x in resolved.capability.reasons]
    assert list(resolved.argv) == argv
    assert rr.ResolvedRecipe.from_dict(resolved.to_dict()).execution_digest == resolved.execution_digest
    return resolved

class Readiness(unittest.TestCase):
    def test_multigpu(self): checked(recipe(device="ROCm0,ROCm1"), ["ROCm0,ROCm1"])
    def test_split_mode(self): checked(recipe(extra_flags=("--split-mode", "row")), ["--split-mode", "row"])
    def test_tensor_split(self): checked(recipe(extra_flags=("--tensor-split", "1,1")), ["--tensor-split", "1,1"])
    def test_per_device_ngl(self):
        # Representation and honest lowering: no invented per-device server flag.
        placement = {"ROCm0": 40, "ROCm1": 20}
        r = serving.Recipe.from_dict({**recipe().to_dict(), "per_device_ngl": placement})
        self.assertEqual(r.to_dict()["per_device_ngl"], placement)
        restored = serving.Recipe.from_dict(r.to_dict())
        self.assertEqual(restored.to_dict()["per_device_ngl"], placement)
        self.assertEqual(restored.recipe_hash, r.recipe_hash)
        self.assertNotEqual(r.recipe_hash, recipe().recipe_hash)
        out = resolve(restored)
        if out.capability.supported:
            baseline = resolve(recipe())
            self.assertNotEqual(out.argv, baseline.argv)
            self.assertNotEqual(out.execution_digest, baseline.execution_digest)
        else:
            self.assertIn("per_device_ngl_unsupported", {x.code for x in out.capability.reasons})
    def test_lora(self): checked(recipe(extra_flags=("--lora", "/models/adapter.gguf")), ["--lora", "/models/adapter.gguf"])
    def test_control_vector(self): checked(recipe(extra_flags=("--control-vector", "/models/control.gguf")), ["--control-vector", "/models/control.gguf"])
    def test_lora_content_identity(self):
        r = recipe(extra_flags=("--lora", "/models/adapter.gguf"))
        # Proposed pinned set: same path/argv, two different content digests.
        one = resolve(r, {"loras": [artifact("lora", "/models/adapter.gguf", "e")]})
        two = resolve(r, {"loras": [artifact("lora", "/models/adapter.gguf", "f")]})
        self.assertTrue(one.capability.supported)
        self.assertEqual(one.argv, two.argv)
        self.assertNotEqual(one.execution_digest, two.execution_digest)
    def test_control_vector_content_identity(self):
        r = recipe(extra_flags=("--control-vector", "/models/control.gguf"))
        one = resolve(r, {"control_vectors": [artifact("control_vector", "/models/control.gguf", "e")]})
        two = resolve(r, {"control_vectors": [artifact("control_vector", "/models/control.gguf", "f")]})
        self.assertTrue(one.capability.supported)
        self.assertEqual(one.argv, two.argv)
        self.assertNotEqual(one.execution_digest, two.execution_digest)
    def test_rpc(self): checked(recipe(extra_flags=("--rpc", "127.0.0.1:50052")), ["--rpc", "127.0.0.1:50052"])
    def test_draft_p_min(self): checked(recipe(spec_decode={"type":"draft-mtp", "draft_p_min":0.5}), ["--spec-draft-p-min", "0.5"])
    def test_draft_n_min(self): checked(recipe(spec_decode={"type":"draft-mtp", "draft_n_min":1}), ["--spec-draft-n-min", "1"])
    def test_draft_own_cache(self): checked(recipe(spec_decode={"type":"draft-dflash", "drafter":"/models/draft.gguf", "ctk":"q8_0"}), ["--spec-draft-type-k", "q8_0"])
    def test_grammar(self): checked(recipe(extra_flags=("--grammar", 'root ::= "x"')), ["--grammar", 'root ::= "x"'])
    def test_sampler_variant(self): checked(recipe(extra_flags=("--samplers", "top_k;temperature")), ["--samplers", "top_k;temperature"])
    def test_no_kv_offload_cache_combination(self): checked(recipe(ctk="q8_0", ctv="q4_0", extra_flags=("--no-kv-offload",)), ["-ctk", "q8_0", "-ctv", "q4_0", "--no-kv-offload"])
    def test_runtime_arm_single_env_control(self):
        pair = env_pair(env_arm({"key":"U3_A", "value":"1"}))
        self.assertEqual(dict(pair.anchor.launch_env)["U3_A"], "0")
        self.assertEqual(dict(pair.candidate.launch_env)["U3_A"], "1")
        self.assertEqual(dict(pair.candidate.launch_env)["U3_B"], "0")
        self.assertNotEqual(pair.candidate.execution_digest, pair.anchor.execution_digest)
    def test_runtime_arm_multi_env(self):
        # Same arm metadata, canonical anchor and planner as the passing single-key control.
        pair = env_pair(env_arm({"set":{"U3_A":"1", "U3_B":"2"}}))
        self.assertEqual(dict(pair.candidate.launch_env)["U3_A"], "1")
        self.assertEqual(dict(pair.candidate.launch_env)["U3_B"], "2")
        self.assertEqual(dict(pair.anchor.launch_env)["U3_A"], "0")
        self.assertNotEqual(pair.candidate.execution_digest, pair.anchor.execution_digest)
    def test_multienv_control(self):
        a = recipe(env={"U3_A":"1", "U3_B":"2"})
        x = checked(a, ["--device", "ROCm0"])
        self.assertEqual(dict(x.launch_env)["U3_A"], "1")
        self.assertEqual(dict(x.launch_env)["U3_B"], "2")
        self.assertNotEqual(a.recipe_hash, recipe(env={"U3_A":"1", "U3_B":"3"}).recipe_hash)
    def test_selfdraft_absence_control(self):
        r = recipe(spec_decode={"type":"draft-mtp", "draft_n_max":4})
        x = checked(r, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"])
        self.assertNotIn("-md", x.argv)
        self.assertNotIn("-ngld", x.argv)
        self.assertNotEqual(r.recipe_hash, recipe(spec_decode={"type":"draft-mtp", "drafter":"/models/draft.gguf", "draft_n_max":4}).recipe_hash)
    def test_no_speculation_absence_control(self):
        r = recipe(spec_decode={"type":"none"})
        x = checked(r, ["--device", "ROCm0"])
        for flag in ("--spec-type", "-md", "-ngld", "--spec-draft-n-max"):
            self.assertNotIn(flag, x.argv)
        self.assertNotEqual(r.recipe_hash, recipe(spec_decode={"type":"draft-mtp"}).recipe_hash)
    def test_cache_types_control(self):
        x = checked(recipe(ctk="q8_0", ctv="q4_0"), ["-ctk", "q8_0", "-ctv", "q4_0"])
        self.assertEqual(x.argv[x.argv.index("-ctk") + 1], "q8_0")
        self.assertEqual(x.argv[x.argv.index("-ctv") + 1], "q4_0")
        changed = checked(recipe(ctk="q8_0", ctv="f16"), ["-ctk", "q8_0", "-ctv", "f16"])
        self.assertNotEqual(x.execution_digest, changed.execution_digest)
    def test_frozen_request_grammar_and_sampler_control(self):
        a = recipe(np=1)
        rows = (("p1", json.dumps({"prompt": "x", "grammar": 'root ::= "x"',
                                    "samplers": ["top_k"]}).encode()),)
        other = (("p1", json.dumps({"prompt": "x", "grammar": 'root ::= "y"',
                                     "samplers": ["top_k"]}).encode()),)
        self.assertEqual(serving._frozen_requests(a, rows), rows)
        self.assertNotEqual(serving.request_digest(a, rows), serving.request_digest(a, other))
    def test_draft_p_min_extra_flag_control(self):
        x = checked(recipe(spec_decode={"type":"draft-mtp"},
                           extra_flags=("--draft-p-min", "0.5")), ["--draft-p-min", "0.5"])
        self.assertEqual(x.argv[x.argv.index("--draft-p-min") + 1], "0.5")
    def test_unset_differs_from_inherit_control(self):
        inherit = recipe()
        unset = recipe(explicit_unsets=("U3_A",))
        self.assertNotEqual(inherit.recipe_hash, unset.recipe_hash)
        self.assertEqual(inherit.server_env(BUILD, base={"U3_A":"1"})["U3_A"], "1")
        self.assertNotIn("U3_A", unset.server_env(BUILD, base={"U3_A":"1"}))

if __name__ == "__main__": unittest.main(verbosity=2)
