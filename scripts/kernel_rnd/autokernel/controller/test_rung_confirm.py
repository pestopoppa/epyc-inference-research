"""The confirm rung's configuration contract (§5.3, D1-D6).

The gate's decision arithmetic is exercised at the loop boundary
(`loop/test_confirm.py`); this suite pins what `configure` may and may not build:
a confirm rung that is not production-shaped, points at surfaces the loop cannot
drive, or has nothing to gate over must be refused at STARTUP -- a misconfigured
confirm gate that runs anyway is a screen with extra steps.
"""
import tempfile
import unittest
from pathlib import Path

from autokernel.controller import rung_confirm, workload_contract as wc
from autokernel.controller.test_workload_contract import _census, _production

KNOWN = ("pp512", "tg128", "dec-b2", "dec-b4", "dec-b8")


def _fixture_production(tmp: Path) -> Path:
    """The production reference as a FILE (configure censuses it itself)."""
    return Path(wc.write_minimal_gguf(
        tmp / "Qwen3.8-27B-Q8_0.gguf", architecture="qwen35", n_embd=5120,
        tensor_types={"Q8_0": 506, "F32": 360}))


def _screen_census(tmp: Path):
    return _census(tmp, "screen.gguf", architecture="qwen2", n_embd=1536,
                   tensors={12: 169, 0: 141, 14: 29})       # Q4_K/F32/Q6_K


def _configure(tmp: Path, **overrides):
    production = _fixture_production(tmp)
    confirm_model = overrides.pop("model", None)
    if confirm_model is None:
        confirm_model = Path(wc.write_minimal_gguf(
            tmp / "confirm.gguf", architecture="qwen35", n_embd=5120,
            tensor_types={"Q8_0": 400, "F32": 300}))
    kwargs = dict(model=confirm_model, pairs=5, surfaces="dec-b4,dec-b8",
                  store=tmp / "store", screen_census=_screen_census(tmp),
                  known_surfaces=KNOWN, floor_for=lambda s: 0.751,
                  production_model=production)
    kwargs.update(overrides)
    return rung_confirm.configure(**kwargs)


class ConfigureBuildsAProductionShapedGate(unittest.TestCase):

    def test_a_production_shaped_confirm_model_configures(self):
        with tempfile.TemporaryDirectory() as tmp:
            confirm = _configure(Path(tmp))
            self.assertEqual(confirm.surfaces, ("dec-b4", "dec-b8"))
            self.assertTrue(confirm.confirm_parity.exact)
            self.assertTrue(confirm.screen_parity.waived,
                            "the 1.5B screen's mismatch is RECORDED, not refused")
            self.assertIn("WAIVED", confirm.describe())

    def test_surfaces_parse_from_the_comma_separated_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            confirm = _configure(Path(tmp), surfaces=" dec-b4 , tg128 ")
            self.assertEqual(confirm.surfaces, ("dec-b4", "tg128"))

    def test_floors_are_taken_per_surface_from_the_injected_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            confirm = _configure(
                Path(tmp), floor_for={"dec-b4": 0.7, "dec-b8": None}.get)
            self.assertEqual(confirm.floors["dec-b4"], 0.7)
            self.assertIsNone(confirm.floors["dec-b8"])
            self.assertIn("UNCALIBRATED", confirm.describe(),
                          "an unmeasured confirm floor must be visible at startup")


class ConfigureRefusesAMisconfiguredGate(unittest.TestCase):

    def test_a_non_production_shaped_confirm_model_is_refused(self):
        """Deleting or inverting the parity refusal must fail here: a screen-shaped
        confirm rung is the screen's job done twice and the confirm's not at all."""
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(wc.write_minimal_gguf(
                Path(tmp) / "DeepSeek-1.5B.gguf", architecture="qwen2",
                n_embd=1536, tensor_types={"Q4_K": 169, "F32": 141}))
            with self.assertRaises(wc.WorkloadContractError) as caught:
                _configure(Path(tmp), model=wrong)
            self.assertIn("not production-shaped", str(caught.exception))

    def test_an_unknown_surface_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(wc.WorkloadContractError) as caught:
                _configure(Path(tmp), surfaces="dec-b4,dec-b16")
            self.assertIn("dec-b16", str(caught.exception))

    def test_an_empty_surface_list_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(wc.WorkloadContractError):
                _configure(Path(tmp), surfaces=" , ")

    def test_zero_pairs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(wc.WorkloadContractError):
                _configure(Path(tmp), pairs=0)

    def test_an_unreadable_production_reference_refuses_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(wc.WorkloadContractError) as caught:
                _configure(Path(tmp),
                           production_model=Path(tmp) / "absent.gguf")
            self.assertIn("declared production model", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
