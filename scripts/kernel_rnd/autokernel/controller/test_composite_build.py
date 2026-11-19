import json
import os
import unittest

from autokernel.controller import build_recipe as br
from autokernel.controller import composite_build as cb

C14_STORE_IDENTITY = "/mnt/raid0/llm/kernels/builds/gpu-20261118-eee3da838-c14/BUILD-IDENTITY.json"


def _argv(recipe=br.GFX90A_ROCM62_C14):
    return ["/usr/bin/cmake", "-S", "x", "-B", "y", "-DCMAKE_C_FLAGS:STRING=-O3"] + [
        f"-D{f.name}:STRING={f.value}" for f in recipe.flags]


class CompositeBuildTest(unittest.TestCase):
    def test_c14_digest_matches_receipt(self):
        self.assertEqual(br.GFX90A_ROCM62_C14.sha256(), br.GFX90A_ROCM62_C14_DIGEST)
        self.assertIs(br.recipe_for(br.GFX90A_ROCM62_C14.name), br.GFX90A_ROCM62_C14)

    def test_c14_contains_v10_and_six_extras(self):
        names = [f.name for f in br.GFX90A_ROCM62_C14.flags]
        self.assertEqual(names[:13], [f.name for f in br.GFX90A_ROCM62_V10_RECIPE.flags])
        self.assertEqual(len(names), 19)
        self.assertIn("LLAMA_LLGUIDANCE", names)

    def test_default_is_c14(self):
        self.assertEqual(cb.DEFAULT_COMPOSITE_RECIPE, br.GFX90A_ROCM62_C14.name)

    def test_pin_stamps_name_and_digest(self):
        out = cb.pin_identity({"configure_argv": _argv()})
        self.assertEqual(out["build_recipe"], br.GFX90A_ROCM62_C14.name)
        self.assertEqual(out["build_recipe_digest"], br.GFX90A_ROCM62_C14_DIGEST)
        self.assertEqual(cb.verify_identity(out), [])

    def test_pin_refuses_changed_value(self):
        argv = _argv() + ["-DGGML_HIP_GRAPHS:BOOL=OFF"]
        with self.assertRaises(cb.RecipeDriftError):
            cb.pin_identity({"configure_argv": argv})

    def test_pin_refuses_missing_flag(self):
        argv = [a for a in _argv() if "LLAMA_LLGUIDANCE" not in a]
        with self.assertRaisesRegex(cb.RecipeDriftError, "LLAMA_LLGUIDANCE"):
            cb.pin_identity({"configure_argv": argv})

    def test_pin_refuses_empty_argv_and_unknown_recipe(self):
        with self.assertRaises(cb.RecipeDriftError):
            cb.pin_identity({})
        with self.assertRaises(br.BuildRecipeError):
            cb.pin_identity({"configure_argv": _argv()}, "nope")

    def test_unnamed_flags_may_vary(self):
        out = cb.pin_identity({"configure_argv": _argv() + ["-DCMAKE_C_FLAGS:STRING=other"]})
        self.assertEqual(out["build_recipe"], br.GFX90A_ROCM62_C14.name)

    def test_verify_detects_tampered_stamp(self):
        out = cb.pin_identity({"configure_argv": _argv()})
        out["build_recipe_digest"] = "0" * 64
        self.assertTrue(cb.verify_identity(out))
        self.assertTrue(cb.verify_identity({"configure_argv": _argv()}))

    @unittest.skipUnless(os.path.exists(C14_STORE_IDENTITY), "C14 store not present")
    def test_real_c14_store_realises_recipe(self):
        ident = json.load(open(C14_STORE_IDENTITY))
        self.assertEqual(cb.drift(ident["configure_argv"], br.GFX90A_ROCM62_C14), [])


if __name__ == "__main__":
    unittest.main()
