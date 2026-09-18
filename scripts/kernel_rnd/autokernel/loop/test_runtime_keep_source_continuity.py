"""Real existing run/pool/keep closures; synthetic runtime-admission/measurement boundary.

Strict admission is independently exercised by test_runtime_admission. This test
does not claim hardware or control qualification for the fixture owner below.
"""
import hashlib
from pathlib import Path

from . import pool, runtime_admission, serial_run, serving
from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion as run_fixture


def test_runtime_keep_then_source_reuses_actual_recipe_and_fresh_source_floor(monkeypatch):
    floors, keeps = [], []
    original_calibrate = serving.calibrate_floor
    protected = []

    def calibration(recipe, *args, **kwargs):
        floors.append((recipe.recipe_hash, recipe.threads))
        return original_calibrate(recipe, *args, **kwargs)

    class FixtureAdmission:
        def __init__(self, **kwargs):
            self.original = kwargs["original"]
            self.state = {"selected": None}
            self.default = kwargs["store"].write("fixture-default", self.original.to_dict())

        def selected(self):
            return self.original

        def compare(self, pair):
            self.pair = pair
            return {"recipe": pair.anchor.template.name, "recipe_hash": pair.anchor.template.recipe_hash,
                    "pairs": 2, "effect": .1, "decisive": True, "noise_floor_pct": 1,
                    "runtime_pair": pair.to_dict(), "runtime_admission": {"fixture": True}}

        def retain(self, row, current):
            assert current == self.pair.anchor
            keeps.append(self.pair)
            return self.pair.candidate

        def selection_reference(self, *args, **kwargs):
            return {"locator": "fixture-runtime-selection", "sha256": "b" * 64,
                    "verified": True}

        def retained_build(self, reference):
            return Path(keeps[0].anchor.build_dir) if keeps else None

        def pending_pair(self):
            return None

    def prune(*args, **kwargs):
        original_runtime_build = Path(keeps[0].anchor.build_dir)
        assert original_runtime_build in kwargs["protect"]
        # Only the preexisting COR and this exact original are protected.
        assert len(kwargs["protect"]) == 2
        protected.append(original_runtime_build)
        return pool.PruneReport("complete")  # Observe owning arguments; delete no builds.

    def checked(result, measured, builds):
        # Four source candidates plus existing kept-anchor verification builds.
        # The fixture rejects ANY compile during the first recipe-only iteration.
        assert len(keeps) == 1 and len(builds) >= 4 and protected
        rows = result["iterations"]
        assert rows[0]["status"] == "kept"
        assert rows[0]["runtime_pair"] == keeps[0].to_dict()
        following = rows[1]["comparison"]
        assert following["recipe_hash"] == keeps[0].candidate.template.recipe_hash
        assert following["floor_request_digest"] == following["request_digest"]
        assert following["noise_floor_pct"] is not None
        assert len(floors) == 2 and floors[0][0] != floors[1][0]
        assert floors[1][0] == keeps[0].candidate.template.recipe_hash
        assert measured
        continuation = result["continuation"]
        assert continuation["runtime_recipe_reference"] == {
            "locator": "fixture-runtime-selection", "sha256": "b" * 64,
            "verified": True}
        # The source keep above installed a fresh build.  Its terminal carries
        # the current operational recipe, and the actual serial owner restores
        # that target-local reference on the next batch without its old floor.
        original = continuation["input_argv"]
        prior_path = Path(serial_run.option(original, "--out")) / "loop-continuation.json"
        next_dir = prior_path.parent.parent / "serial-resume"
        next_dir.mkdir()
        resumed = serial_run._batch_argv(original, {
            "path": str(prior_path),
            "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest()}, 1, next_dir)
        reference_path = Path(serial_run.option(resumed, "--runtime-recipe-reference"))
        assert reference_path.parent == next_dir
        assert serial_run.option(resumed, "--resume-run") == str(prior_path)
        assert serial_run.option(resumed, "--cpu-calibrate-serving") is None
        assert serial_run._json(reference_path, limit=4096)[0] == \
            continuation["runtime_recipe_reference"]

    monkeypatch.setattr(runtime_admission, "RuntimeAdmission", FixtureAdmission)
    monkeypatch.setattr(serving, "calibrate_floor", calibration)
    monkeypatch.setattr(pool, "prune_anchor_generations", prune)
    run_fixture(False, runtime_transition=checked)
