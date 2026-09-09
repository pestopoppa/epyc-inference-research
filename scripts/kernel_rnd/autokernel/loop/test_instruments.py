"""The hand-run instruments: floor identity, refusals, argument handling, dry-run posture.

WHY THESE TESTS EXIST. `serving_gate`, `recal_serving_floor` and `seed_bundle` produced
this campaign's most consequential numbers while living in `/mnt/raid0/llm/tmp/`, outside
the package and therefore outside its safety layer:

  * the GATE built `serving-floor.<name>.json` itself and read `["floor_pct"]` off it --
    the reader half of the defect `run.py` was fixed for, so a hand-run gate could be
    judged against a floor calibrated under a DIFFERENT condition with nothing raised;
  * the RECALIBRATION wrote that same path with a bare `write_text` -- no atomic replace
    and no identity stamp, so one arm's A/A could be filed under another arm's name.

So what is asserted here is that there is now exactly ONE path to a floor from every
caller (`serving.load_floor` / `serving.write_floor`), that a mismatch REFUSES rather than
degrading to "no floor", and that nothing spends GPU time or writes decision state without
an explicit flag. The measurement calls are thin seams; every one of them is stubbed.
"""
import ast
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import (accumulate, fold2_gates, instruments,
                             recal_serving_floor, seed_bundle, serving, serving_gate)


SHIPPED_RECIPE = instruments.DEFAULT_RECIPE
RECIPE = serving.Recipe(name="t", model="/m/target.gguf", np=4, ctx=16384)


def _floor_row(recipe: serving.Recipe, floor_pct: float = 3.536) -> dict:
    """What `serving.calibrate_floor` returns, without launching anything."""
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(), "metric": recipe.metric,
            "np": recipe.np, "samples": 5, "median_tok_s": 71.2,
            "floor_pct": floor_pct, "cv_pct": 1.2, "runs": [71.0, 71.2, 71.4],
            "spread": {"p95_dev_pct": floor_pct}}


def _serving_row(effect_pct: float, decisive: bool | None) -> dict:
    """What `serving.compare` returns."""
    return {"schema": "epyc.autokernel.serving_ab.v1", "recipe": RECIPE.name,
            "recipe_hash": RECIPE.recipe_hash, "anchor_tok_s": 70.0,
            "candidate_tok_s": 70.0 * (1 + effect_pct / 100.0),
            "effect": effect_pct / 100.0, "effect_pct": effect_pct,
            "noise_floor_pct": 3.536, "decisive": decisive, "pairs": 5}


def _code_only(path: Path) -> str:
    """The module's EXECUTABLE lines: docstrings and comments removed.

    These structural tests are about what the code DOES. A usage example in a docstring
    naming a scratch build directory is documentation of how the fold window was driven,
    not a dependency on scratch -- scanning raw text would make the guard fire on its own
    explanation, and the usual fix for that is to delete the explanation.
    """
    source = path.read_text(encoding="utf-8")
    skip: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    return "\n".join(line for number, line in enumerate(source.splitlines(), 1)
                      if number not in skip and not line.lstrip().startswith("#"))


def _build(root: Path, *names: str) -> Path:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / "bin" / name).write_text("#!/bin/false\n")
    return root


class Posture(unittest.TestCase):
    def test_the_default_is_dry(self):
        posture = instruments.Posture(execute=False, apply=False)
        self.assertTrue(posture.dry_run)
        self.assertIn("DRY", posture.describe())

    def test_apply_implies_execute(self):
        parser = serving_gate.build_parser()
        args = parser.parse_args(
            ["--cor-build", "/a", "--tip-build", "/b", "--tip", "abc", "--apply"])
        posture = instruments.resolve_posture(args)
        self.assertTrue(posture.execute)
        self.assertTrue(posture.apply)

    def test_execute_alone_does_not_apply(self):
        parser = serving_gate.build_parser()
        args = parser.parse_args(
            ["--cor-build", "/a", "--tip-build", "/b", "--tip", "abc", "--execute"])
        posture = instruments.resolve_posture(args)
        self.assertTrue(posture.execute)
        self.assertFalse(posture.apply)

    def test_an_instrument_with_no_decision_state_has_no_apply_flag(self):
        """A gate battery writes an evidence record and nothing else: an `--apply` there
        would imply there is state it could settle."""
        args = fold2_gates.build_parser().parse_args(["--candidate-build", "/c"])
        self.assertFalse(hasattr(args, "apply"))
        self.assertFalse(instruments.resolve_posture(args).execute)


class FloorPathResolution(unittest.TestCase):
    def test_no_instrument_builds_a_floor_filename_of_its_own(self):
        """The reader half of the defect: a hand-built
        `store / f"serving-floor.{recipe.name}.json"` is what let a stale floor judge a new
        condition. Nothing under the loop package may spell that pattern again."""
        offenders = [path.name for path in
                     sorted(Path(__file__).resolve().parent.glob("*.py"))
                     if path.name != "serving.py"
                     and not path.name.startswith("test_")   # tests build it to prove
                     and "serving-floor.{" in _code_only(path)]   # the refusal fires
        self.assertEqual(offenders, [])

    def test_that_scan_would_catch_the_pattern_it_looks_for(self):
        """Control: the scan above is not a permanently-green no-op."""
        with tempfile.TemporaryDirectory() as tmp:
            offender = Path(tmp) / "bad.py"
            offender.write_text(
                'p = store / f"serving-floor.{recipe.name}.json"\n', encoding="utf-8")
            self.assertIn("serving-floor.{", _code_only(offender))

    def test_the_shipped_recipe_keeps_the_path_it_has_today(self):
        recipe = serving.Recipe.load(SHIPPED_RECIPE)
        self.assertEqual(serving.floor_path("/S", recipe).name,
                         "serving-floor.qwen3.8-27b-q8-gpu-dflash2-np4.json")

    def test_the_recalibration_targets_exactly_that_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            store.mkdir()
            build = _build(Path(tmp) / "b", "llama-server")
            printed = []
            with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(
                    str(x) for x in a))):
                rc = recal_serving_floor.main(
                    ["--store", str(store), "--recipe", str(SHIPPED_RECIPE),
                     "--build", str(build)])
            self.assertEqual(rc, 0)
            recipe = serving.Recipe.load(SHIPPED_RECIPE)
            expected = str(serving.floor_path(store, recipe))
            self.assertTrue(any(expected in line for line in printed), printed)


class FloorIdentityRefusal(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Path(self._tmp.name)

    def test_a_floor_from_another_recipe_is_refused_not_used(self):
        other = dataclasses.replace(RECIPE, np=8)
        serving.floor_path(self.store, RECIPE).write_text(
            json.dumps(_floor_row(other)), encoding="utf-8")
        with self.assertRaises(serving.ServingFloorMismatch):
            instruments.read_floor(self.store, RECIPE, echo=lambda *_: None)

    def test_an_absent_floor_is_refused_rather_than_bought_with_a_full_gate(self):
        """`decisive` would come back None, `classify_serving` would read that as
        DIVERGED: a verdict decided before the measurement started."""
        with self.assertRaises(instruments.InstrumentRefusal):
            instruments.read_floor(self.store, RECIPE, echo=lambda *_: None)

    def test_an_unstamped_floor_is_used_but_announced(self):
        row = _floor_row(RECIPE)
        row.pop("recipe_hash")
        serving.floor_path(self.store, RECIPE).write_text(json.dumps(row),
                                                          encoding="utf-8")
        said = []
        reading = instruments.read_floor(self.store, RECIPE, echo=said.append)
        self.assertEqual(reading.provenance, "unverified")
        self.assertEqual(reading.floor_pct, 3.536)
        self.assertTrue(any("recipe_hash" in line for line in said), said)

    def test_a_matching_floor_is_verified(self):
        serving.write_floor(self.store, RECIPE, _floor_row(RECIPE))
        reading = instruments.read_floor(self.store, RECIPE, echo=lambda *_: None)
        self.assertTrue(reading.verified)


class RecalibrationWritesThroughWriteFloor(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"
        self.store.mkdir()
        self.build = _build(self.root / "b", "llama-server")
        self.recipe = serving.Recipe.load(SHIPPED_RECIPE)

    def _run(self, *extra, row=None):
        row = row if row is not None else _floor_row(self.recipe, 2.111)
        with mock.patch.object(recal_serving_floor, "measure",
                               lambda *a, **k: row) as seam, \
             mock.patch("builtins.print"):
            rc = recal_serving_floor.main(
                ["--store", str(self.store), "--recipe", str(SHIPPED_RECIPE),
                 "--build", str(self.build), *extra])
        return rc, seam

    def test_the_default_posture_measures_nothing_and_writes_nothing(self):
        def explode(*_a, **_k):
            raise AssertionError("a dry run must not launch a server")

        with mock.patch.object(recal_serving_floor, "measure", explode), \
             mock.patch("builtins.print"):
            rc = recal_serving_floor.main(
                ["--store", str(self.store), "--recipe", str(SHIPPED_RECIPE),
                 "--build", str(self.build)])
        self.assertEqual(rc, 0)
        self.assertFalse(serving.floor_path(self.store, self.recipe).exists())

    def test_execute_measures_but_does_not_write_the_floor(self):
        rc, _ = self._run("--execute")
        self.assertEqual(rc, 0)
        self.assertFalse(serving.floor_path(self.store, self.recipe).exists())

    def test_apply_writes_an_identity_stamped_floor(self):
        rc, _ = self._run("--apply")
        self.assertEqual(rc, 0)
        target = serving.floor_path(self.store, self.recipe)
        body = json.loads(target.read_text())
        self.assertEqual(body["recipe_hash"], self.recipe.recipe_hash)
        self.assertEqual(body["recipe"], self.recipe.name)
        self.assertEqual(body["floor_pct"], 2.111)
        self.assertIn("calibrated_at", body["conditions"])
        # The atomic write leaves no scratch file behind.
        self.assertEqual([p.name for p in self.store.glob(".sv-floor-*")], [])

    def test_the_previous_floor_is_backed_up_not_overwritten_blind(self):
        target = serving.floor_path(self.store, self.recipe)
        serving.write_floor(self.store, self.recipe, _floor_row(self.recipe, 3.536))
        self._run("--apply")
        backups = list(self.store.glob("*.bak"))
        self.assertEqual(len(backups), 1, backups)
        self.assertEqual(json.loads(backups[0].read_text())["floor_pct"], 3.536)
        self.assertEqual(json.loads(target.read_text())["floor_pct"], 2.111)

    def test_a_row_produced_by_another_recipe_is_refused(self):
        """The copy-paste `write_floor` exists to stop: one arm's A/A filed under
        another arm's name is a bar nobody would question."""
        foreign = _floor_row(dataclasses.replace(self.recipe, np=8), 9.9)
        rc, _ = self._run("--apply", row=foreign)
        self.assertEqual(rc, instruments.REFUSED)
        self.assertFalse(serving.floor_path(self.store, self.recipe).exists())

    def test_require_cpu_list_refuses_an_unpinned_recipe(self):
        unpinned = self.root / "unpinned.json"
        body = json.loads(SHIPPED_RECIPE.read_text())
        body.pop("cpu_list", None)
        body["name"] = "unpinned-arm"
        unpinned.write_text(json.dumps(body))
        with mock.patch.object(recal_serving_floor, "measure",
                               lambda *a, **k: self.fail("must not measure")), \
             mock.patch("builtins.print"):
            rc = recal_serving_floor.main(
                ["--store", str(self.store), "--recipe", str(unpinned),
                 "--build", str(self.build), "--require-cpu-list", "--execute"])
        self.assertEqual(rc, instruments.REFUSED)

    def test_a_build_without_llama_server_is_refused(self):
        empty = _build(self.root / "empty")
        with mock.patch.object(recal_serving_floor, "measure",
                               lambda *a, **k: self.fail("must not measure")), \
             mock.patch("builtins.print"):
            rc = recal_serving_floor.main(
                ["--store", str(self.store), "--recipe", str(SHIPPED_RECIPE),
                 "--build", str(empty), "--execute"])
        self.assertEqual(rc, instruments.REFUSED)

    def test_the_backup_name_is_derived_from_the_floor_path(self):
        target = serving.floor_path(self.store, self.recipe)
        backup = recal_serving_floor.backup_path(target, "prev", day="20260908")
        self.assertEqual(
            backup.name,
            "serving-floor.qwen3.8-27b-q8-gpu-dflash2-np4.prev-20260908.json.bak")


class ServingGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"
        self.store.mkdir()
        self.recipe = serving.Recipe.load(SHIPPED_RECIPE)
        serving.write_floor(self.store, self.recipe, _floor_row(self.recipe, 3.536))
        self.cor = _build(self.root / "cor", "llama-server")
        self.tip = _build(self.root / "tip", "llama-server")
        self._bundle(compounded=12.0, keeps=["akm-a", "akm-b"])

    def _bundle(self, *, compounded: float, keeps: list, cor="c" * 40, tip="t" * 40):
        accumulate.Bundle(champion_of_record=cor, tip=tip, keeps=list(keeps),
                          compounded_bench_pct=compounded).save(self.store)

    def _argv(self, *extra):
        return ["--store", str(self.store), "--recipe", str(SHIPPED_RECIPE),
                "--cor-build", str(self.cor), "--tip-build", str(self.tip),
                "--tip", "t" * 40, "--champion-worktree", str(self.root), *extra]

    def _run(self, *extra, row=None):
        row = row if row is not None else _serving_row(5.0, True)
        calls = []

        def seam(*args, **kwargs):
            calls.append((args, kwargs))
            return row

        with mock.patch.object(serving_gate, "measure", seam), \
             mock.patch.object(serving_gate, "_is_ancestor", return_value=True), \
             mock.patch("builtins.print"):
            rc = serving_gate.main(self._argv(*extra))
        return rc, calls

    def test_the_default_posture_measures_nothing(self):
        rc, calls = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertFalse((self.store / "serving").exists())

    def test_bundle_output_names_validity_without_inventing_measurement_source(self):
        source = Path(serving_gate.__file__).read_text(encoding="utf-8")
        self.assertNotIn("(MEASURED by seed_bundle)", source)
        self.assertIn("validity={bundle.measurement_validity}", source)
        self.assertIn("historical-only magnitude; threshold disabled", source)

    def test_dry_legacy_replay_does_not_import_or_rewrite(self):
        shutil.rmtree(self.store / accumulate.JOURNAL_DIRNAME)
        legacy = {
            "schema": accumulate.Bundle.LEGACY_SCHEMA,
            "champion_of_record": "c" * 40,
            "tip": "t" * 40,
            "keeps": ["akm-a", "akm-b"],
            "compounded_bench_pct": 12.0,
            "keeps_since_serving_gate": 2,
        }
        path = self.store / accumulate.Bundle.FILENAME
        original = json.dumps(legacy)
        path.write_text(original, encoding="utf-8")
        rc, calls = self._run("--force")
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertFalse((self.store / accumulate.JOURNAL_DIRNAME).exists())
        self.assertEqual(path.read_text(), original)

    def test_missing_bundle_state_refuses_without_measuring(self):
        shutil.rmtree(self.store / accumulate.JOURNAL_DIRNAME)
        (self.store / accumulate.Bundle.FILENAME).unlink()
        rc, calls = self._run("--execute")
        self.assertEqual(rc, instruments.REFUSED)
        self.assertEqual(calls, [])

    def test_corrupt_journal_refuses_instead_of_trusting_projection(self):
        events = (self.store / accumulate.JOURNAL_DIRNAME
                  / "events.jsonl")
        with events.open("a", encoding="utf-8") as stream:
            stream.write("{corrupt}\n")
        rc, calls = self._run("--execute")
        self.assertEqual(rc, instruments.REFUSED)
        self.assertEqual(calls, [])

    def test_execute_writes_the_record_but_does_not_advance_the_champion(self):
        rc, calls = self._run("--execute")
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        records = list((self.store / "serving").glob("bundle-*.json"))
        self.assertEqual(len(records), 1, records)
        body = json.loads(records[0].read_text())
        self.assertEqual(body["outcome"], "promote")
        self.assertEqual(body["floor_provenance"], "verified")
        bundle = accumulate.Bundle.from_dict(
            json.loads((self.store / accumulate.Bundle.FILENAME).read_text()))
        self.assertEqual(bundle.champion_of_record, "c" * 40)

    def test_apply_advances_the_champion_of_record_on_a_promote(self):
        rc, _ = self._run("--apply")
        self.assertEqual(rc, 0)
        bundle = accumulate.Bundle.from_dict(
            json.loads((self.store / accumulate.Bundle.FILENAME).read_text()))
        self.assertEqual(bundle.champion_of_record, "t" * 40)
        self.assertEqual(bundle.keeps, [])

    def test_a_divergence_holds_the_champion_even_under_apply(self):
        rc, _ = self._run("--apply", row=_serving_row(0.4, False))
        self.assertEqual(rc, 0)
        bundle = accumulate.Bundle.from_dict(
            json.loads((self.store / accumulate.Bundle.FILENAME).read_text()))
        self.assertEqual(bundle.champion_of_record, "c" * 40)
        self.assertEqual(bundle.keeps, ["akm-a", "akm-b"])

    def test_a_bundle_below_the_scheduling_threshold_spends_nothing(self):
        self._bundle(compounded=1.0, keeps=["akm-a"])
        rc, calls = self._run("--execute")
        self.assertEqual(rc, instruments.REFUSED)
        self.assertEqual(calls, [])

    def test_force_spends_the_gate_below_the_scheduling_threshold(self):
        """The 2026-09-08 one-off operator permission. It bypasses the loop's SCHEDULING
        heuristic only -- PROMOTE still requires a decisive positive serving effect,
        which lives in `accumulate.resolve` and is not reachable from here."""
        self._bundle(compounded=1.0, keeps=["akm-a"])
        rc, calls = self._run("--execute", "--force")
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)

    def test_force_cannot_promote_an_indecisive_result(self):
        self._bundle(compounded=1.0, keeps=["akm-a"])
        rc, _ = self._run("--apply", "--force", row=_serving_row(9.0, False))
        self.assertEqual(rc, 0)
        bundle = accumulate.Bundle.from_dict(
            json.loads((self.store / accumulate.Bundle.FILENAME).read_text()))
        self.assertEqual(bundle.champion_of_record, "c" * 40)

    def test_a_stale_floor_refuses_the_gate(self):
        foreign = dataclasses.replace(self.recipe, np=8)
        row = _floor_row(foreign)
        row["recipe"] = self.recipe.name
        serving.floor_path(self.store, self.recipe).write_text(json.dumps(row),
                                                               encoding="utf-8")
        rc, calls = self._run("--execute")
        self.assertEqual(rc, instruments.REFUSED)
        self.assertEqual(calls, [])

    def test_an_uncalibrated_store_refuses_the_gate(self):
        serving.floor_path(self.store, self.recipe).unlink()
        rc, calls = self._run("--execute")
        self.assertEqual(rc, instruments.REFUSED)
        self.assertEqual(calls, [])

    def test_a_build_without_llama_server_is_refused(self):
        self.tip = _build(self.root / "notip")
        rc, calls = self._run("--execute")
        self.assertEqual(rc, instruments.REFUSED)
        self.assertEqual(calls, [])

    def test_the_floor_passed_to_the_measurement_is_the_one_on_disk(self):
        _, calls = self._run("--execute")
        self.assertEqual(calls[0][1]["floor_pct"], 3.536)


def _commit(repo: Path, subject: str) -> str:
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "--allow-empty",
                    "-m", subject], capture_output=True, text=True, timeout=60)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=60).stdout.strip()


class SeedBundle(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"
        self.store.mkdir()
        self.repo = self.root / "tree"
        self.repo.mkdir()
        subprocess.run(["git", "-C", str(self.repo), "init", "-q", "-b", "main"],
                       capture_output=True, text=True, timeout=60)
        # LOCAL identity: a runner with no global gitconfig would otherwise fail with
        # "Author identity unknown" (the reason test_champion.py does the same).
        for key, value in (("user.email", "t@t"), ("user.name", "t")):
            subprocess.run(["git", "-C", str(self.repo), "config", key, value],
                           capture_output=True, text=True, timeout=60)
        self.cor = _commit(self.repo, "champion of record")
        _commit(self.repo, "akm-fast-dequant: bit-deposit rewrite")
        _commit(self.repo, "chore: not a keep")
        self.tip = _commit(self.repo, "akm-tiled-gemv: LDS tiling")
        self.cor_build = _build(self.root / "corb", "llama-bench")
        self.tip_build = _build(self.root / "tipb", "llama-bench")

    def _argv(self, *extra):
        return ["--cor", self.cor, "--cor-build", str(self.cor_build),
                "--tip-build", str(self.tip_build), "--store", str(self.store),
                "--champion-worktree", str(self.repo), "--model",
                str(self.root / "m.gguf"), *extra]

    def _run(self, *extra, effect=6.13):
        calls = []

        def seam(*args, **kwargs):
            calls.append((args, kwargs))
            return {"effect_pct": effect, "decisive": True, "drifting": False,
                    "surface": "tg128", "pairs": 20}

        with mock.patch.object(seed_bundle, "measure", seam), \
             mock.patch("builtins.print"):
            rc = seed_bundle.main(self._argv(*extra))
        return rc, calls

    def test_the_keeps_are_read_from_commit_topology(self):
        self.assertEqual(seed_bundle.keeps_between(self.repo, self.cor, self.tip),
                         ["akm-tiled-gemv", "akm-fast-dequant"])

    def test_the_ancestry_check_reads_the_exit_code(self):
        """The scratch original asserted on stdout being empty, which is true of BOTH
        answers -- the check could not fail."""
        self.assertTrue(seed_bundle.is_ancestor(self.repo, self.cor, self.tip))
        self.assertFalse(seed_bundle.is_ancestor(self.repo, self.tip, self.cor))

    def test_a_cor_that_is_not_an_ancestor_is_refused(self):
        with mock.patch.object(seed_bundle, "measure",
                               lambda *a, **k: self.fail("must not measure")), \
             mock.patch("builtins.print"):
            rc = seed_bundle.main(
                ["--cor", self.tip, "--tip", self.cor, "--cor-build", str(self.cor_build),
                 "--tip-build", str(self.tip_build), "--store", str(self.store),
                 "--champion-worktree", str(self.repo), "--execute"])
        self.assertEqual(rc, instruments.REFUSED)

    def test_the_default_posture_measures_nothing_and_writes_no_bundle(self):
        rc, calls = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertFalse((self.store / accumulate.Bundle.FILENAME).exists())

    def test_execute_records_the_measurement_but_writes_no_bundle(self):
        rc, calls = self._run("--execute")
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.store / accumulate.Bundle.FILENAME).exists())
        self.assertEqual(
            json.loads((self.store / "seed-measurement.json").read_text())["effect_pct"],
            6.13)

    def test_apply_writes_a_bundle_carrying_the_MEASURED_compounded_number(self):
        """Never a product of solos: the value is one paired A/B of the whole bundle."""
        rc, _ = self._run("--apply")
        self.assertEqual(rc, 0)
        bundle = accumulate.Bundle.from_dict(
            json.loads((self.store / accumulate.Bundle.FILENAME).read_text()))
        self.assertEqual(bundle.champion_of_record, self.cor)
        self.assertEqual(bundle.tip, self.tip)
        self.assertEqual(bundle.compounded_bench_pct, 6.13)
        self.assertEqual(sorted(bundle.keeps), ["akm-fast-dequant", "akm-tiled-gemv"])

    def test_a_build_without_llama_bench_is_refused(self):
        empty = _build(self.root / "empty")
        with mock.patch.object(seed_bundle, "measure",
                               lambda *a, **k: self.fail("must not measure")), \
             mock.patch("builtins.print"):
            rc = seed_bundle.main(
                ["--cor", self.cor, "--cor-build", str(empty), "--tip-build",
                 str(self.tip_build), "--store", str(self.store),
                 "--champion-worktree", str(self.repo), "--execute"])
        self.assertEqual(rc, instruments.REFUSED)

    def test_the_measurement_is_taken_on_the_calibrated_tg128_floor(self):
        _, calls = self._run("--execute")
        self.assertEqual(calls[0][1]["floor_pct"], 0.638)
        self.assertEqual(calls[0][1]["pairs"], 20)


class NoScratchPathsSurvived(unittest.TestCase):
    """The instruments were promoted so they stop depending on scratch. A module that
    still hardcodes `/mnt/raid0/llm/tmp/...` or re-inserts a worktree onto `sys.path` is
    one `rm` from unreproducible again."""

    INSTRUMENTS = ("instruments.py", "serving_gate.py", "recal_serving_floor.py",
                   "fold2_gates.py", "seed_bundle.py")

    def test_no_instrument_hardcodes_a_scratch_path_or_a_syspath_shim(self):
        here = Path(__file__).resolve().parent
        offenders = []
        for name in self.INSTRUMENTS:
            body = _code_only(here / name)
            if "sys.path.insert" in body:
                offenders.append(f"{name}: sys.path shim")
            for line in body.splitlines():
                if "/mnt/raid0/llm/tmp/" in line:
                    offenders.append(f"{name}: {line.strip()[:70]}")
        self.assertEqual(offenders, [])

    def test_that_scan_would_catch_what_it_looks_for(self):
        """Control: the two patterns really do match the shims that were removed."""
        with tempfile.TemporaryDirectory() as tmp:
            offender = Path(tmp) / "bad.py"
            offender.write_text(
                'import sys\n'
                'sys.path.insert(0, "/mnt/raid0/llm/worktrees/x/scripts/kernel_rnd")\n'
                'CAND = "/mnt/raid0/llm/tmp/build-fold-ef81196d5"\n', encoding="utf-8")
            body = _code_only(offender)
            self.assertIn("sys.path.insert", body)
            self.assertIn("/mnt/raid0/llm/tmp/", body)


if __name__ == "__main__":
    unittest.main()
