"""A self-drafting model (MTP) has no separate drafter GGUF — the recipe must not pass `-md`.

Found 2026-09-08: `Recipe.server_argv` required `spec_decode["drafter"]` unconditionally, so
every self-drafting model was inexpressible as a recipe and raised KeyError. The absence of
`drafter` IS the declaration that the model drafts for itself.
"""
import dataclasses
import unittest
from pathlib import Path

from . import serving


def _recipe(**kw):
    base = dict(name="t", model="/m.gguf", device="ROCm0", ngl=99, np=1, ctx=4096,
                threads=8, batch=512, ubatch=512, ctk="f16", ctv="f16", fa="on")
    base.update(kw)
    return serving.Recipe(**base)


class SelfDraftingRecipe(unittest.TestCase):
    def test_self_draft_passes_spec_type_but_no_md(self):
        r = _recipe(spec_decode={"type": "draft-mtp", "draft_n_max": 4})
        argv = r.server_argv(Path("/b"), 18317)
        self.assertIn("--spec-type", argv)
        self.assertEqual("draft-mtp", argv[argv.index("--spec-type") + 1])
        self.assertNotIn("-md", argv)
        self.assertNotIn("-ngld", argv)
        self.assertIn("--spec-draft-n-max", argv)

    def test_separate_drafter_still_passes_md_and_ngld(self):
        r = _recipe(spec_decode={"type": "draft-dflash", "drafter": "/d.gguf", "ngld": 99})
        argv = r.server_argv(Path("/b"), 18317)
        self.assertIn("-md", argv)
        self.assertEqual("/d.gguf", argv[argv.index("-md") + 1])
        self.assertEqual("99", argv[argv.index("-ngld") + 1])

    def test_spec_none_passes_neither(self):
        r = _recipe(spec_decode={"type": "none"})
        argv = r.server_argv(Path("/b"), 18317)
        self.assertNotIn("--spec-type", argv)
        self.assertNotIn("-md", argv)

    def test_control_the_assertion_can_fail(self):
        """Guard: prove -md IS present for a separate drafter, so 'notIn' above is not vacuous."""
        r = _recipe(spec_decode={"type": "draft-dflash", "drafter": "/d.gguf"})
        self.assertIn("-md", r.server_argv(Path("/b"), 18317))

    def test_self_draft_recipe_hash_differs_from_separate_drafter(self):
        a = _recipe(spec_decode={"type": "draft-mtp"})
        b = _recipe(spec_decode={"type": "draft-mtp", "drafter": "/d.gguf"})
        self.assertNotEqual(a.recipe_hash, b.recipe_hash)


if __name__ == "__main__":
    unittest.main()
