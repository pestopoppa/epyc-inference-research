"""SPEC-COPY-COV-1: synthetic checks of the exact longest-suffix coverage replay (no tokenizer, no inference)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location("copy_cov_replay", pathlib.Path(__file__).with_name("copy_cov_replay.py"))
ccr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccr)


def test_selftest():
    assert ccr.selftest() == 0


def test_fire_trigger_on_cross_request_copy():
    shared = list(range(200, 260))
    reqs = [{"id": "a", "suite": "debugbench", "prompt_ids": [1], "out_ids": shared},
            {"id": "b", "suite": "debugbench", "prompt_ids": [2], "out_ids": shared},
            {"id": "c", "suite": "general", "prompt_ids": [3], "out_ids": list(range(300, 340))}]
    res = ccr.run(reqs)
    assert res["variants"]["loo"]["code"]["hit_share"] > 0.05
    assert res["verdict"] == "FIRE_TRIGGER"
