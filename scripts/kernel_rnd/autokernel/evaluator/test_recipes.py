#!/usr/bin/env python3
"""test_recipes.py — the regression barrier for the codified T1 recipe constructor.

WHY THIS FILE EXISTS
--------------------
Recipe drift is not hypothetical here. It cost this project a day on 2026-05-02
(missing `taskset`, mmap defaulted ON, AOCC libomp) and another on 2026-05-28
(seven compounding drift bugs in one run, including an experimental binary
resolving production libraries). Both were *visible in the source*; neither was
ASSERTED anywhere. `canonical_recipe.py` was written in response, and this suite
does the same job for the operator-microbenchmark constructor that
`P-AK-SEARCH-1` precondition 6 requires and that §2.6 of the owning design
records as the substrate blocking T1a on every backend.

The properties under test, each traceable to a clause of
`measurement/protocols/kernel-research.md` (P-AK-SEARCH-1, RATIFIED 2026-08-03):

  * an unregistered `recipe_id` is REFUSED, never constructed (precondition 6:
    *"hand-typed argv voids the run"*);
  * the ratified constants are IMPORTED, not retyped — the suite mutates the
    canonical constants and proves the emitted argv follows;
  * argv is CONSTRUCTED: no parameter is interpolated into a token, and a value
    that would shift the tool's own option parsing is refused;
  * the resource footprint is DERIVED from the argv's own `taskset -c` list
    (precondition 1: *"a CPU region claim covering the exact footprint measured"*);
  * a CANDIDATE arm cannot be measured out of a frozen production tree
    (denial 2), while the ANCHOR arm can;
  * where the ratified discipline is unsatisfiable for a tool, the gap is
    RECORDED as a `DisciplineFinding`, not smoothed over (denial 6);
  * the receipt plugs into `api.RecipeReceipt` / `api.WindowAttestations` and
    renders the grammar's `recipe=<id>@<sha[:12]>` field;
  * the module constructs and executes nothing, proved from its own AST.

NO inference, NO benchmark, NO build. No process is started or signalled. Files
are written only inside per-test temporary directories this suite creates and
removes.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_recipes.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_recipes.py
"""
from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Import through the PACKAGE so `recipes.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel import storage as ST  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.evaluator import recipes as R  # noqa: E402
from autokernel.resource import device_claim as DC  # noqa: E402

NOW = "2026-08-03T12:00:00+00:00"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"


def _sha(seed: str) -> str:
    import hashlib
    return hashlib.sha256(seed.encode()).hexdigest()


class FakeTreeMixin:
    """A throwaway git-shaped worktree with executable tool stubs.

    The stubs are never executed by anything in this suite; they exist so the
    constructor's read-only `stat` checks have something real to look at. That is
    deliberate: a fixture that removed the inputs under test would make
    `verify_inputs=True` untestable, which is the
    `feedback_fixture_must_not_remove_signal_under_test` failure.
    """

    TOOLS = ("llama-bench", "test-backend-ops", "test-quantize-perf")

    def setUp(self) -> None:  # noqa: D102
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-recipes-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "llama.cpp-ak-campaign"
        self.bindir = self.root / "build" / "bin"
        self.bindir.mkdir(parents=True)
        # A worktree records `.git` as a FILE; a clone as a directory. Use the
        # worktree form, because that is what AK2 creates.
        (self.root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/ak\n")
        self.tools = {}
        for name in self.TOOLS:
            path = self.bindir / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
            self.tools[name] = path
        self.model = self.root / "tiny-Q4_K_M.gguf"
        self.model.write_bytes(b"GGUF" * 64)

    def binding(self, tool: str = "llama-bench") -> R.ToolBinding:
        return R.ToolBinding(binary=str(self.tools[tool]),
                             source_root=str(self.root),
                             library_path=str(self.bindir))

    def decode_params(self, **overrides):
        params = {"model": str(self.model), "n_gen": 64, "reps": 10}
        params.update(overrides)
        return params

    def ops_params(self, **overrides):
        params = {"phase": "prefill", "ops": ["MUL_MAT"],
                  "min_measurable_us": self.duration_floor()}
        params.update(overrides)
        return params

    @staticmethod
    def duration_floor():
        """Synthetic local-A/A fixture; never represented as measured evidence."""
        return api.MinimumMeasurableDuration(
            min_measurable_us=10.0, aa_absolute_spread_us=0.3,
            relative_noise_budget=0.03, aa_pair_count=20,
            samples_ref="fixture:paired-local-aa-durations")

    def quant_params(self, **overrides):
        params = {"phase": "decode", "op": "vec_dot_q", "types": ["q4_K"],
                  "iterations": 10, "min_measurable_us": self.duration_floor()}
        params.update(overrides)
        return params

    def all_cells(self):
        """One constructible (recipe_id, binding, params) triple per REGISTERED recipe.

        `test_every_registered_recipe_is_covered_by_this_suite` asserts this covers
        the whole registry, so adding a recipe without a construction case fails
        here rather than shipping a recipe nothing proves is constructible.
        """
        gpu = {"device_index": 0, "device_id": "mi210_0", "n_gpu_layers": 99}
        return [
            ("t1a.llama_cpu.backend_ops_perf.v1", self.binding("test-backend-ops"),
             self.ops_params()),
            ("t1a.llama_gpu.backend_ops_perf.v1", self.binding("test-backend-ops"),
             self.ops_params(device_index=0, device_id="mi210_0")),
            ("t1a.llama_cpu.quantize_perf.v1", self.binding("test-quantize-perf"),
             self.quant_params()),
            ("t1b.llama_cpu.llama_bench_decode.v1", self.binding(),
             self.decode_params()),
            ("t1b.llama_cpu.llama_bench_prefill.v1", self.binding(),
             {"model": str(self.model), "n_prompt": 512, "reps": 10}),
            ("t1b.llama_gpu.llama_bench_decode.v1", self.binding(),
             dict(self.decode_params(), **gpu)),
            ("t1b.llama_gpu.llama_bench_prefill.v1", self.binding(),
             dict({"model": str(self.model), "n_prompt": 512, "reps": 10}, **gpu)),
        ]


# =============================================================================
# The registry, and the refusal of anything not in it
# =============================================================================

class RegistryTest(FakeTreeMixin, unittest.TestCase):

    def test_recipe_ids_are_pinned(self):
        """A recipe id is a citation key: renaming one orphans every record citing it.

        This is deliberately a hard-coded list. Adding a recipe is a one-line edit
        here; RENAMING one has to be a conscious act, because P-AK-SEARCH-1's record
        grammar carries `recipe=<recipe_constructor_id>@<sha[:12]>` and a record whose
        recipe id no longer resolves cannot be audited.
        """
        self.assertEqual(R.RECIPE_IDS, (
            "t1a.llama_cpu.backend_ops_perf.v1",
            "t1a.llama_cpu.quantize_perf.v1",
            "t1a.llama_gpu.backend_ops_perf.v1",
            "t1b.llama_cpu.llama_bench_decode.v1",
            "t1b.llama_cpu.llama_bench_prefill.v1",
            "t1b.llama_gpu.llama_bench_decode.v1",
            "t1b.llama_gpu.llama_bench_prefill.v1",
        ))

    def test_every_registered_id_resolves_and_is_sorted(self):
        self.assertEqual(R.RECIPE_IDS, tuple(sorted(R.RECIPE_IDS)))
        self.assertTrue(R.RECIPE_IDS)
        for recipe_id in R.RECIPE_IDS:
            self.assertIs(R.get_recipe(recipe_id), R.REGISTRY[recipe_id])

    def test_unregistered_recipe_id_is_refused_not_constructed(self):
        with self.assertRaises(R.UnregisteredRecipe) as ctx:
            R.construct("t1a.llama_cpu.made_up.v9", binding=self.binding())
        message = str(ctx.exception)
        self.assertIn("not registered", message)
        self.assertIn("hand-typed argv voids the run", message)
        # The refusal lists what IS registered, so the caller can correct it.
        self.assertIn("t1b.llama_cpu.llama_bench_decode.v1", message)

    def test_close_miss_on_a_registered_id_is_still_refused(self):
        # A version bump or a typo must not fall through to "something similar".
        for near in ("t1b.llama_cpu.llama_bench_decode.v2",
                     "t1b.llama_cpu.llama_bench_decode",
                     "T1B.LLAMA_CPU.LLAMA_BENCH_DECODE.V1"):
            with self.subTest(near=near):
                with self.assertRaises(R.UnregisteredRecipe):
                    R.get_recipe(near)

    def test_non_string_recipe_id_is_a_type_error(self):
        with self.assertRaises(TypeError):
            R.get_recipe(None)

    def test_list_recipes_filters(self):
        self.assertEqual(R.list_recipes(backend="llama_gpu", family=R.RECIPE_FAMILY_T1A),
                         ("t1a.llama_gpu.backend_ops_perf.v1",))
        self.assertEqual(len(R.list_recipes(tier="T1b")), 4)
        self.assertEqual(R.list_recipes(backend="whisper_stt"), ())

    def test_registry_covers_both_families_on_both_llama_backends(self):
        for backend in ("llama_cpu", "llama_gpu"):
            for family in R.RECIPE_FAMILIES:
                with self.subTest(backend=backend, family=family):
                    self.assertTrue(R.list_recipes(backend=backend, family=family),
                                    f"no {family} recipe for {backend}")

    def test_every_spec_declares_a_tier_this_evaluator_owns(self):
        for spec in R.REGISTRY.values():
            with self.subTest(recipe_id=spec.recipe_id):
                self.assertEqual(api.admit_tier(spec.tier), spec.tier)
                self.assertNotIn(spec.tier, api.RELEASE_TIERS)

    def test_release_tier_recipe_cannot_be_registered(self):
        # T3 is AK5's, outside P-AK-SEARCH-1's scope; a recipe naming it must not
        # be constructible even as a dataclass.
        with self.assertRaises(api.TierNotOwned):
            R.RecipeSpec(
                recipe_id="t3.llama_cpu.freeze.v1", family=R.RECIPE_FAMILY_T1B,
                tier="T3", backend="llama_cpu", phase="decode",
                cell_class=R.CELL_CLASS_TINY_GRAPH, tool="llama-bench",
                metric="decode_tokens_per_s", metric_direction="higher_better",
                params=(), builder="_builder_llama_bench_decode_cpu", summary="x")

    def test_metric_must_be_commensurable_with_its_backend(self):
        # MEASUREMENT.md:23-30 — task_rate belongs to the serving_runtime scope.
        with self.assertRaises(ValueError) as ctx:
            R.RecipeSpec(
                recipe_id="t1b.llama_cpu.wrong_metric.v1", family=R.RECIPE_FAMILY_T1B,
                tier="T1b", backend="llama_cpu", phase="decode",
                cell_class=R.CELL_CLASS_TINY_GRAPH, tool="llama-bench",
                metric="task_rate", metric_direction="higher_better",
                params=(), builder="_builder_llama_bench_decode_cpu", summary="x")
        self.assertIn("commensurable", str(ctx.exception))

    def test_required_param_may_not_carry_a_default(self):
        with self.assertRaises(ValueError) as ctx:
            R.ParamSpec(name="reps", kind="int", required=True, default=5, doc="x")
        self.assertIn("unrecorded", str(ctx.exception))


# =============================================================================
# The ratified constants are SOURCED, not retyped
# =============================================================================

class SourcedConstantsTest(FakeTreeMixin, unittest.TestCase):

    def test_constants_are_the_canonical_recipe_objects(self):
        cr = R.canonical_recipe
        self.assertEqual(list(R.CANONICAL_PREFIX), list(cr.CANONICAL_PREFIX))
        self.assertEqual(list(R.CANONICAL_BENCH_FLAGS),
                         list(cr.CANONICAL_BENCH_FLAGS_LLAMA_BENCH))
        self.assertEqual(R.CANONICAL_OMP_ENV, dict(cr.CANONICAL_OMP_ENV))
        self.assertEqual(R.LLVM20_LIBDIR, cr.LLVM20_LIBDIR)

    def test_bound_canonical_recipe_is_the_repo_file(self):
        self.assertEqual(Path(R.canonical_recipe.__file__).resolve(),
                         R.CANONICAL_RECIPE_PATH)
        self.assertTrue(R.CANONICAL_RECIPE_PATH.is_file())

    def test_argv_follows_the_canonical_prefix_rather_than_a_local_copy(self):
        """Move the ratified constant; the emitted argv and footprint move with it."""
        moved = ("taskset", "-c", "0-47", "numactl", "--interleave=all")
        with mock.patch.object(R, "CANONICAL_PREFIX", moved), \
                mock.patch.object(R.canonical_recipe, "CANONICAL_PREFIX", list(moved)):
            command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                                  binding=self.binding("test-backend-ops"),
                                  params=self.ops_params())
            self.assertEqual(command.argv[:5], moved)
            self.assertEqual(command.claim_footprint.cpu_list, "0-47")
            self.assertEqual(command.claim_footprint.cpu_count, 48)

    def test_a_prefix_that_diverges_from_the_ratified_one_is_refused(self):
        """The two sources must agree; a local-only edit raises rather than emits."""
        with mock.patch.object(R, "CANONICAL_PREFIX",
                               ("taskset", "-c", "0-3", "numactl", "--interleave=all")):
            with self.assertRaises(R.RecipeDriftError) as ctx:
                R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                            binding=self.binding("test-backend-ops"),
                            params=self.ops_params())
        self.assertIn("do not retype it", str(ctx.exception))

    def test_argv_follows_the_canonical_bench_flags(self):
        moved = ("-t", "48", "-fa", "1", "-mmp", "0")
        with mock.patch.object(R, "CANONICAL_BENCH_FLAGS", moved):
            command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                                  binding=self.binding(), params=self.decode_params())
        self.assertIn("-t", command.argv)
        self.assertEqual(command.argv[command.argv.index("-t") + 1], "48")

    def test_canonical_flags_are_all_present_and_explicit(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        argv = list(command.argv)
        for flag, value in (("-t", "96"), ("-fa", "1"), ("-mmp", "0")):
            with self.subTest(flag=flag):
                self.assertIn(flag, argv)
                self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_env_is_the_canonical_omp_stack_plus_pinned_library_path(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        for key, value in R.CANONICAL_OMP_ENV.items():
            self.assertEqual(command.env[key], value)
        entries = command.env["LD_LIBRARY_PATH"].split(":")
        self.assertEqual(entries[0], str(self.bindir),
                         "the candidate library path must be FIRST or the binary can "
                         "resolve production libraries (2026-05-28)")
        self.assertIn(R.LLVM20_LIBDIR, entries)

    def test_emitted_env_carries_nothing_ambient(self):
        with mock.patch.dict("os.environ", {"OMP_NUM_THREADS": "7", "LEAKED": "yes"}):
            command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                                  binding=self.binding(), params=self.decode_params())
        self.assertNotIn("LEAKED", command.env)
        self.assertNotIn("OMP_NUM_THREADS", command.env)
        self.assertEqual(set(command.env), set(R.CANONICAL_OMP_ENV) | {"LD_LIBRARY_PATH"})

    def test_gpu_host_cores_are_read_from_the_codified_launcher(self):
        value = R.gpu_host_cpu_list()
        self.assertTrue(R.GPU_BENCH_LIB_PATH.is_file())
        self.assertNotEqual(
            value, "88-95",
            "88-95 is the SUPERSEDED MI210 host pinning per the launcher's own "
            "provenance note")
        self.assertEqual(R.MI210_NUMA_NODE, 1)
        self.assertEqual(R.GPU_HOST_THREADS_NUMA_NODE, 3)
        self.assertFalse(R.GPU_HOST_THREADS_ARE_NUMA_LOCAL,
                         "184-191 are node-3 SMT siblings, not MI210-local threads")

    def test_gpu_host_cores_really_come_from_the_file(self):
        # Patching `_MODULE_HASHES` with a copy keeps the fake file's hash from
        # leaking into another test's receipt.
        fake = self.tmp / "fake_gpu_lib.sh"
        fake.write_text('BIN="x"\nCORES="${GPU_BENCH_CORES:-12-19}"   # note\n')
        with mock.patch.object(R, "GPU_BENCH_LIB_PATH", fake), \
                mock.patch.object(R, "_gpu_host_cpu_list_cache", None), \
                mock.patch.object(R, "_MODULE_HASHES", dict(R._MODULE_HASHES)):
            self.assertEqual(R.gpu_host_cpu_list(), "12-19")

    def test_missing_gpu_launcher_raises_and_never_guesses(self):
        missing = self.tmp / "nope.sh"
        with mock.patch.object(R, "GPU_BENCH_LIB_PATH", missing), \
                mock.patch.object(R, "_gpu_host_cpu_list_cache", None), \
                mock.patch.object(R, "_MODULE_HASHES", dict(R._MODULE_HASHES)):
            with self.assertRaises(R.SourcedConstantUnavailable) as ctx:
                R.gpu_host_cpu_list()
        self.assertIn("retyping the value here", str(ctx.exception))

    def test_changed_shape_of_the_gpu_constant_fails_closed(self):
        fake = self.tmp / "reshaped.sh"
        fake.write_text('CORES=184-191\n')  # no ${GPU_BENCH_CORES:-...} form
        with mock.patch.object(R, "GPU_BENCH_LIB_PATH", fake), \
                mock.patch.object(R, "_gpu_host_cpu_list_cache", None), \
                mock.patch.object(R, "_MODULE_HASHES", dict(R._MODULE_HASHES)):
            with self.assertRaises(R.SourcedConstantUnavailable) as ctx:
                R.gpu_host_cpu_list()
        self.assertIn("fails closed", str(ctx.exception))

    def test_a_cached_gpu_constant_still_carries_its_provenance_hash(self):
        R.gpu_host_cpu_list()
        with mock.patch.object(R, "_MODULE_HASHES", dict(R._MODULE_HASHES)):
            R._MODULE_HASHES.pop(R._GPU_LIB_REL_PATH, None)
            R.gpu_host_cpu_list()  # cache hit
            self.assertIn(R._GPU_LIB_REL_PATH, R._MODULE_HASHES,
                          "a cache hit must not hand back a value with no recorded "
                          "source; the receipt would then cite an unhashed constant")

    def test_sourced_constants_are_recorded_with_their_content_hashes(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        names = {item["name"] for item in command.sourced_constants}
        self.assertEqual(names, {"CANONICAL_PREFIX", "CANONICAL_BENCH_FLAGS_LLAMA_BENCH",
                                 "CANONICAL_OMP_ENV", "LLVM20_LIBDIR"})
        for item in command.sourced_constants:
            self.assertEqual(item["source"], "scripts/lib/canonical_recipe.py")
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")

    def test_gpu_recipe_records_the_launcher_provenance(self):
        command = R.construct(
            "t1a.llama_gpu.backend_ops_perf.v1", binding=self.binding("test-backend-ops"),
            params=self.ops_params(device_index=0, device_id="mi210_0"))
        names = {item["name"] for item in command.sourced_constants}
        self.assertIn("GPU_BENCH_CORES", names)


# =============================================================================
# argv CONSTRUCTION — no interpolation of caller input
# =============================================================================

class ArgvConstructionTest(FakeTreeMixin, unittest.TestCase):

    def test_every_parameter_lands_as_its_own_argv_element(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=self.decode_params(n_gen=128, reps=11))
        argv = list(command.argv)
        for flag, value in (("-m", str(self.model)), ("-p", "0"), ("-n", "128"),
                            ("-r", "11"), ("-o", "json")):
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        # No token is a concatenation of a flag and its value.
        self.assertNotIn(f"-m{self.model}", argv)
        self.assertNotIn("-n128", argv)

    def test_a_model_path_that_looks_like_a_flag_is_refused(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(model="-fa"))
        self.assertIn("would be parsed as an option", str(ctx.exception))

    def test_relative_and_dotdot_model_paths_are_refused(self):
        for bad in ("models/tiny.gguf", "/mnt/raid0/../etc/tiny.gguf"):
            with self.subTest(bad=bad):
                with self.assertRaises(R.RecipeParameterError):
                    R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                                binding=self.binding(),
                                params=self.decode_params(model=bad))

    def test_model_must_be_a_gguf(self):
        other = self.root / "weights.safetensors"
        other.write_bytes(b"x" * 16)
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(model=str(other)))
        self.assertIn(".gguf", str(ctx.exception))

    def test_control_characters_are_refused_everywhere(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params=self.ops_params(params_filter="type=f16\nrm -rf /"))
        self.assertIn("control character", str(ctx.exception))

    def test_op_names_must_be_bare_ggml_ops(self):
        for bad in ("mul_mat", "MUL MAT", "ADD(type=f16,ne=[1,1,8,1])", "-o"):
            with self.subTest(bad=bad):
                with self.assertRaises(R.RecipeParameterError):
                    R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                                binding=self.binding("test-backend-ops"),
                                params=self.ops_params(ops=[bad]))

    def test_no_recipe_emits_a_flag_the_canonical_protocol_forbids(self):
        """`bench-cpu.md:10-11`: no `--numa distribute`, and mmap is never re-enabled."""
        for recipe_id, binding, params in self.all_cells():
            with self.subTest(recipe_id=recipe_id):
                argv = list(R.construct(recipe_id, binding=binding, params=params).argv)
                self.assertNotIn("--numa", argv)
                self.assertNotIn("--mmap", argv)
                if "-mmp" in argv:
                    self.assertEqual(argv[argv.index("-mmp") + 1], "0")

    def test_the_backend_filter_matches_the_recipes_backend(self):
        cpu = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                          binding=self.binding("test-backend-ops"),
                          params=self.ops_params())
        argv = list(cpu.argv)
        self.assertEqual(argv[argv.index("-b") + 1], "CPU")
        gpu = R.construct("t1a.llama_gpu.backend_ops_perf.v1",
                          binding=self.binding("test-backend-ops"),
                          params=self.ops_params(device_index=1, device_id="mi210_1"))
        argv = list(gpu.argv)
        # NOT `ROCm1`. `ROCR_VISIBLE_DEVICES` removes the other devices rather than
        # renumbering the chosen one, and ggml names what survives by its index in
        # the VISIBLE set (ggml-cuda.cu:5358), which is always 0.
        self.assertEqual(argv[argv.index("-b") + 1], "ROCm0")

    def test_backend_filter_never_names_an_ordinal_the_mask_makes_impossible(self):
        """`-b ROCm<physical>` is not a loud failure — it is a silent zero-measurement.

        `test-backend-ops` skips every device whose name does not equal the filter,
        counts each skip as OK, and returns 0 (tests/test-backend-ops.cpp:10366-10371,
        :10413-10417). A filter naming an ordinal that cannot exist inside the mask
        the same recipe emits produces a success-shaped run with no perf row in it —
        a status without its evidence.
        """
        for index in (0, 1, 7, 15):
            with self.subTest(device_index=index):
                command = R.construct(
                    "t1a.llama_gpu.backend_ops_perf.v1",
                    binding=self.binding("test-backend-ops"),
                    params=self.ops_params(device_index=index,
                                           device_id=f"mi210_{index}"))
                argv = list(command.argv)
                self.assertEqual(argv[argv.index("-b") + 1], R.GPU_VISIBLE_DEVICE_NAME)
                # The physical ordinal is still carried — by the outer mask.
                self.assertEqual(command.env["ROCR_VISIBLE_DEVICES"], str(index))
                self.assertTrue(
                    any("ROCR_VISIBLE_DEVICES" in note for note in command.bounded),
                    "where the physical ordinal actually travels must be stated")

    def test_the_two_device_masks_compose_and_the_inner_one_indexes_the_masked_set(self):
        """ROCR filters the HSA agents; HIP then indexes what SURVIVED that filter.

        Setting both to the same physical ordinal `n` asks for index `n` of a
        one-element list: for any n >= 1 the process sees NO device at all. The
        codified launcher's literal `0 0 0` is safe only because it selects the
        first device at both levels (architect_bench_gpu_lib.sh:33).
        """
        for index in (0, 1, 15):
            with self.subTest(device_index=index):
                command = R.construct(
                    "t1b.llama_gpu.llama_bench_decode.v1", binding=self.binding(),
                    params=dict(self.decode_params(), device_index=index,
                                device_id=f"mi210_{index}", n_gpu_layers=99))
                self.assertEqual(command.env["ROCR_VISIBLE_DEVICES"], str(index))
                self.assertEqual(command.env["HIP_VISIBLE_DEVICES"], "0")
                self.assertEqual(command.env["CUDA_VISIBLE_DEVICES"], "0")

    def test_device_index_zero_is_byte_identical_to_the_codified_gpu_launcher(self):
        """The one arm that can be checked against the sourced file must match it.

        `architect_bench_gpu_lib.sh:33-35` runs
        `ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 …
        --device ROCm0`. Parameterising the ordinal must not change that arm.
        """
        command = R.construct("t1b.llama_gpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=dict(self.decode_params(), device_index=0,
                                          device_id="mi210_0", n_gpu_layers=99))
        for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES"):
            self.assertEqual(command.env[key], "0")
        argv = list(command.argv)
        self.assertEqual(argv[argv.index("-dev") + 1], "ROCm0")

    def test_a_params_filter_that_looks_like_a_flag_is_refused(self):
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params=self.ops_params(params_filter="--output"))

    def test_an_over_long_params_filter_is_refused_not_truncated(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params=self.ops_params(
                            params_filter="a" * (R.MAX_PARAMS_FILTER_CHARS + 1)))
        self.assertIn("above the declared bound", str(ctx.exception))

    def test_the_declared_phase_is_required_and_reaches_the_command(self):
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params={"ops": ["MUL_MAT"]})
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params=self.ops_params(phase="operator"))
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params(phase="prefill"))
        self.assertEqual(command.phase, "prefill")
        self.assertEqual(command.cell_class, R.CELL_CLASS_OPERATOR)

    def test_op_list_is_joined_from_validated_tokens_only(self):
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params(ops=["MUL_MAT", "MUL_MAT_ID"]))
        argv = list(command.argv)
        self.assertEqual(argv[argv.index("-o") + 1], "MUL_MAT,MUL_MAT_ID")

    def test_duplicate_and_empty_op_lists_are_refused(self):
        for bad in ([], ["MUL_MAT", "MUL_MAT"], "MUL_MAT"):
            with self.subTest(bad=bad):
                with self.assertRaises(R.RecipeParameterError):
                    R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                                binding=self.binding("test-backend-ops"),
                                params=self.ops_params(ops=bad))

    def test_an_over_long_op_list_is_refused_not_truncated(self):
        too_many = [f"OP_{i}" for i in range(R.MAX_OPS_PER_INVOCATION + 1)]
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params=self.ops_params(ops=too_many))
        self.assertIn("never silently truncated", str(ctx.exception))

    def test_quant_types_come_from_a_declared_enum(self):
        binding = self.binding("test-quantize-perf")
        base = self.quant_params(iterations=100)
        good = R.construct("t1a.llama_cpu.quantize_perf.v1", binding=binding,
                           params=dict(base, types=["q4_K", "q8_0"]))
        argv = list(good.argv)
        self.assertEqual(argv.count("--type"), 2)
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1a.llama_cpu.quantize_perf.v1", binding=binding,
                        params=dict(base, types=["q4_k_m"]))

    def test_size_not_divisible_by_32_is_refused_before_the_window_opens(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1a.llama_cpu.quantize_perf.v1",
                        binding=self.binding("test-quantize-perf"),
                        params=self.quant_params(
                            iterations=100, size_elements=100))
        self.assertIn("divisible by 32", str(ctx.exception))

    def test_unknown_parameters_are_refused(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(extra_flags=["--numa", "distribute"]))
        self.assertIn("hand-typed argv with extra steps", str(ctx.exception))

    def test_missing_required_parameter_is_refused(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params={"model": str(self.model), "n_gen": 64})
        self.assertIn("'reps'", str(ctx.exception))

    def test_bool_is_not_an_int(self):
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(reps=True))

    def test_reps_below_one_is_refused(self):
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(reps=0))

    def test_optional_flags_are_omitted_when_unset_and_emitted_when_set(self):
        without = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        self.assertNotIn("-d", without.argv)
        self.assertNotIn("-ub", without.argv)
        with_extras = R.construct(
            "t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
            params=self.decode_params(n_depth=512, ubatch=2048, batch=4096))
        argv = list(with_extras.argv)
        self.assertEqual(argv[argv.index("-d") + 1], "512")
        self.assertEqual(argv[argv.index("-ub") + 1], "2048")
        self.assertEqual(argv[argv.index("-b") + 1], "4096")

    def test_prefill_and_decode_recipes_pin_the_other_axis_to_zero(self):
        decode = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                             binding=self.binding(), params=self.decode_params(n_gen=32))
        argv = list(decode.argv)
        self.assertEqual(argv[argv.index("-p") + 1], "0")
        self.assertEqual(argv[argv.index("-n") + 1], "32")
        prefill = R.construct("t1b.llama_cpu.llama_bench_prefill.v1",
                              binding=self.binding(),
                              params={"model": str(self.model), "n_prompt": 512,
                                      "reps": 10})
        argv = list(prefill.argv)
        self.assertEqual(argv[argv.index("-p") + 1], "512")
        self.assertEqual(argv[argv.index("-n") + 1], "0")

    def test_argv_and_env_are_all_strings(self):
        for recipe_id, binding, params in self.all_cells():
            with self.subTest(recipe_id=recipe_id):
                command = R.construct(recipe_id, binding=binding, params=params)
                for token in command.argv:
                    self.assertIsInstance(token, str)
                for key, value in command.env.items():
                    self.assertIsInstance(key, str)
                    self.assertIsInstance(value, str)

    def test_every_registered_recipe_constructs(self):
        for recipe_id, binding, params in self.all_cells():
            with self.subTest(recipe_id=recipe_id):
                command = R.construct(recipe_id, binding=binding, params=params)
                self.assertEqual(command.recipe_id, recipe_id)
                self.assertTrue(command.argv)
                self.assertTrue(command.render_human_readable())

    def test_every_registered_recipe_is_covered_by_this_suite(self):
        covered = {recipe_id for recipe_id, _, _ in self.all_cells()}
        self.assertEqual(covered, set(R.RECIPE_IDS),
                         "a registered recipe with no construction test is a recipe "
                         "nothing proves is constructible")


# =============================================================================
# The binding — which build actually runs
# =============================================================================

class BindingTest(FakeTreeMixin, unittest.TestCase):

    def test_library_path_must_be_the_binarys_own_directory(self):
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.ToolBinding(binary=str(self.tools["llama-bench"]),
                          source_root=str(self.root), library_path=str(self.root))
        self.assertIn("resolve someone else's libggml", str(ctx.exception))

    def test_binary_outside_the_source_root_is_refused(self):
        other = self.tmp / "elsewhere" / "bin"
        other.mkdir(parents=True)
        stray = other / "llama-bench"
        stray.write_text("#!/bin/sh\n")
        stray.chmod(0o755)
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.ToolBinding(binary=str(stray), source_root=str(self.root),
                          library_path=str(other))
        self.assertIn("outside binding.source_root", str(ctx.exception))

    def test_relative_binding_paths_are_refused(self):
        with self.assertRaises(R.RecipeParameterError):
            R.ToolBinding(binary="build/bin/llama-bench", source_root="build",
                          library_path="build/bin")

    def test_binary_name_must_match_the_recipes_tool(self):
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.construct("t1a.llama_cpu.quantize_perf.v1",
                        binding=self.binding("test-backend-ops"),
                        params=self.quant_params())
        self.assertIn("emits argv for 'test-quantize-perf'", str(ctx.exception))

    def test_missing_binary_is_refused_when_inputs_are_verified(self):
        self.tools["llama-bench"].unlink()
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        self.assertIn("cannot run", str(ctx.exception))

    def test_non_executable_binary_is_refused(self):
        self.tools["llama-bench"].chmod(0o644)
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        self.assertIn("not executable", str(ctx.exception))

    def test_source_root_without_git_is_refused(self):
        (self.root / ".git").unlink()
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        self.assertIn("no .git entry", str(ctx.exception))

    def test_missing_model_is_refused(self):
        self.model.unlink()
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        self.assertIn("model file not found", str(ctx.exception))

    def test_empty_model_is_refused(self):
        self.model.write_bytes(b"")
        with self.assertRaises(R.RecipeBindingError):
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())

    def test_disabling_verification_yields_could_not_check_never_pass(self):
        self.tools["llama-bench"].unlink()
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params(),
                              verify_inputs=False)
        self.assertEqual([c.outcome for c in command.input_checks],
                         [S.COULD_NOT_CHECK])
        self.assertFalse(command.inputs_verified)
        self.assertIn("never PASS", command.input_checks[0].reasons[0])

    def test_verified_inputs_report_pass_for_each_checked_thing(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        self.assertTrue(command.inputs_verified)
        self.assertEqual(len(command.input_checks), 5)  # binary, libdir, root, .git, model


# =============================================================================
# Denial 2 — a candidate is never measured out of a production tree
# =============================================================================

class ProductionTreeDenialTest(FakeTreeMixin, unittest.TestCase):

    def _production_binding(self):
        production = Path(ST.PRODUCTION_TREES[0])
        return R.ToolBinding(binary=str(production / "build" / "bin" / "llama-bench"),
                             source_root=str(production),
                             library_path=str(production / "build" / "bin"))

    def test_candidate_arm_out_of_a_production_tree_is_refused(self):
        with self.assertRaises(R.RecipeBindingError) as ctx:
            R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                        binding=self._production_binding(),
                        params=self.decode_params(), arm="candidate",
                        verify_inputs=False)
        message = str(ctx.exception)
        self.assertIn("FROZEN production tree", message)
        self.assertIn("denial 2", message)

    def test_anchor_arm_out_of_a_production_tree_is_allowed(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self._production_binding(),
                              params=self.decode_params(), arm="anchor",
                              verify_inputs=False)
        self.assertEqual(command.arm, "anchor")
        self.assertTrue(command.argv)

    def test_the_guard_follows_symlinks(self):
        # CLAUDE.md's working-tree identity rule makes /workspace/repos/<name> a
        # symlink to /mnt/raid0/llm/<name>; a literal-string guard would fail OPEN.
        link_root = self.tmp / "repos"
        link_root.mkdir()
        link = link_root / "epyc-llama"
        link.symlink_to(ST.PRODUCTION_TREES[0])
        binding = R.ToolBinding(binary=str(link / "build" / "bin" / "llama-bench"),
                                source_root=str(link),
                                library_path=str(link / "build" / "bin"))
        with self.assertRaises(R.RecipeBindingError):
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=binding,
                        params=self.decode_params(), arm="candidate",
                        verify_inputs=False)

    def test_unknown_arm_is_refused(self):
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(), arm="baseline")

    def test_production_roots_are_sourced_from_storage(self):
        self.assertIn("/mnt/raid0/llm/llama.cpp", ST.PRODUCTION_TREES)
        self.assertTrue(set(ST.PRODUCTION_TREES).issubset(
            set(ST.production_tree_forms())))


# =============================================================================
# The derived resource footprint (precondition 1)
# =============================================================================

class FootprintTest(FakeTreeMixin, unittest.TestCase):

    def test_cpu_footprint_is_derived_from_the_argv_taskset_list(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        argv = list(command.argv)
        self.assertEqual(command.claim_footprint.cpu_list,
                         argv[argv.index("taskset") + 2])
        self.assertEqual(command.claim_footprint.cpu_count, 96)
        self.assertIn("taskset -c", command.claim_footprint.derived_from)

    def test_cpu_cell_is_full_machine(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        self.assertEqual(command.scope_denominator.machine_subset, "full")
        self.assertEqual(command.scope_denominator.cores, 96)
        self.assertEqual(command.scope_denominator.devices, ())

    def test_gpu_cell_is_partial_and_names_its_device(self):
        command = R.construct("t1b.llama_gpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=dict(self.decode_params(), device_index=1,
                                          device_id="mi210_1", n_gpu_layers=99))
        scope = command.scope_denominator
        self.assertEqual(scope.machine_subset, "partial")
        self.assertEqual(scope.devices, ("mi210_1",))
        self.assertEqual(command.claim_footprint.cpu_list, R.gpu_host_cpu_list())
        self.assertEqual(command.claim_footprint.devices, ("mi210_1",))
        self.assertEqual(scope.render(), "partial/devmi210_1/8c")

    def test_gpu_device_index_reaches_the_mask_and_argv_names_the_masked_device(self):
        """The claimed device travels in the mask; argv names what the mask leaves.

        `-dev ROCm1` inside a process masked to one device is not merely wrong, it
        aborts: llama-bench raises `invalid device: ROCm1`
        (tools/llama-bench/llama-bench.cpp:166), burning the measurement window the
        exclusive device claim was acquired for.
        """
        command = R.construct("t1b.llama_gpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=dict(self.decode_params(), device_index=1,
                                          device_id="mi210_1", n_gpu_layers=99))
        argv = list(command.argv)
        self.assertEqual(argv[argv.index("-dev") + 1], R.GPU_VISIBLE_DEVICE_NAME)
        self.assertEqual(command.env["ROCR_VISIBLE_DEVICES"], "1")
        self.assertNotIn("ROCm1", argv)

    def test_gpu_device_id_must_match_the_device_claim_pattern(self):
        with self.assertRaises(R.RecipeParameterError) as ctx:
            R.construct("t1b.llama_gpu.llama_bench_decode.v1", binding=self.binding(),
                        params=dict(self.decode_params(), device_index=0,
                                    device_id="mi210 0", n_gpu_layers=99))
        self.assertIn("device-claim id pattern", str(ctx.exception))

    def test_device_id_pattern_agrees_with_the_device_claim_module(self):
        # Two copies of one boundary is how one of them quietly loses an entry.
        self.assertEqual(R._DEVICE_ID_RE.pattern, DC._DEVICE_ID_RE.pattern,
                         "recipes._DEVICE_ID_RE has drifted from "
                         "resource.device_claim._DEVICE_ID_RE; a recipe could then name "
                         "a device the claim layer would refuse")

    def test_gpu_threads_default_to_the_sourced_mask_width_and_stay_explicit(self):
        command = R.construct("t1b.llama_gpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=dict(self.decode_params(), device_index=0,
                                          device_id="mi210_0", n_gpu_layers=99))
        argv = list(command.argv)
        self.assertEqual(argv[argv.index("-t") + 1], "8")
        self.assertTrue(any("width of the sourced GPU host-thread mask" in b
                            for b in command.bounded))

    def test_gpu_prefix_does_not_interleave_all_nodes(self):
        command = R.construct("t1b.llama_gpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=dict(self.decode_params(), device_index=0,
                                          device_id="mi210_0", n_gpu_layers=99))
        self.assertNotIn("numactl", command.argv)
        self.assertNotIn("--interleave=all", command.argv)

    def test_malformed_cpu_lists_are_refused(self):
        for bad in ("", "0-", "5-1", "a-b", "0,,3", "999999"):
            with self.subTest(bad=bad):
                with self.assertRaises(R.RecipeParameterError):
                    R._cpu_list_members(bad, field="test")

    def test_argv_without_a_taskset_prefix_cannot_produce_a_footprint(self):
        with self.assertRaises(R.RecipeDriftError) as ctx:
            R._footprint_from_argv(["/bin/llama-bench", "-m", "x"], ())
        self.assertIn("precondition 1", str(ctx.exception))


# =============================================================================
# Discipline findings — recorded, not smoothed over (denial 6)
# =============================================================================

class DisciplineTest(FakeTreeMixin, unittest.TestCase):

    def test_backend_ops_records_that_the_thread_discipline_is_unsatisfiable(self):
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params())
        finding = command.finding("explicit_threads")
        self.assertEqual(finding.check.outcome, S.FAIL)
        joined = " ".join(finding.check.reasons)
        self.assertIn("hardware_concurrency", joined)
        self.assertIn("COVERAGE GAP, recorded not patched", joined)
        self.assertEqual(command.discipline_outcome, S.FAIL)

    def test_a_failing_discipline_finding_still_yields_a_usable_command(self):
        # "What voids a run" does not include a non-canonical thread count; the
        # finding must travel with the record, not block construction.
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params())
        self.assertTrue(command.argv)
        self.assertEqual(command.receipt.constructor_id,
                         "t1a.llama_cpu.backend_ops_perf.v1")

    def test_gpu_env_gap_is_recorded_with_its_provenance(self):
        command = R.construct("t1a.llama_gpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params(device_index=0,
                                                     device_id="mi210_0"))
        finding = command.finding("canonical_env_stack")
        self.assertEqual(finding.check.outcome, S.COULD_NOT_CHECK)
        joined = " ".join(finding.check.reasons)
        self.assertIn("architect_bench_gpu_lib.sh@", joined)
        self.assertIn("COVERAGE GAP, recorded not patched", joined)

    def test_llama_bench_json_retains_raw_samples_and_md_does_not(self):
        good = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                           params=self.decode_params(output_format="jsonl"))
        self.assertEqual(good.finding("raw_samples_retained").check.outcome, S.PASS)
        self.assertIn("samples_ns", good.raw_samples_source)

        bad = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                          params=self.decode_params(output_format="md"))
        finding = bad.finding("raw_samples_retained")
        self.assertEqual(finding.check.outcome, S.FAIL)
        joined = " ".join(finding.check.reasons)
        self.assertIn("raw samples from which the reduction is reproducible", joined)
        self.assertIn("NONE", bad.raw_samples_source)

    def test_quant_types_the_tool_silently_skips_are_flagged_not_passed(self):
        """`--type q8_1` measures nothing and exits 0. A recipe must say so.

        test-quantize-perf runs a type's op blocks only under
        `qfns_cpu->from_float && qfns->to_float` (tests/test-quantize-perf.cpp:273)
        and has no else-branch: a type failing the guard produces not one line of
        output, and the process still returns 0. Ten of the twenty-six declared
        `--type` names fail it in the frozen production tree.
        """
        def quant(types):
            return R.construct("t1a.llama_cpu.quantize_perf.v1",
                               binding=self.binding("test-quantize-perf"),
                               params=self.quant_params(types=types))

        for bad in ("q8_1", "q8_K", "f32", "iq1_s", "iq2_s", "iq3_s"):
            with self.subTest(type=bad):
                self.assertIn(bad, R.QUANTIZE_PERF_UNMEASURABLE_TYPES)
                finding = quant([bad]).finding("measurable_types")
                self.assertEqual(finding.check.outcome, S.FAIL)
                joined = " ".join(finding.check.reasons)
                self.assertIn("exits 0", joined)
                self.assertIn("ASYMMETRIC", joined)

        good = quant(["q4_K", "q6_K"]).finding("measurable_types")
        self.assertEqual(good.check.outcome, S.PASS)

        # A partly-affected list is still a FAIL: the record must not imply that
        # every type in the argv produced a row.
        mixed = quant(["q4_K", "q8_1"]).finding("measurable_types")
        self.assertEqual(mixed.check.outcome, S.FAIL)

    def test_the_unmeasurable_type_list_matches_the_reference_trees_traits(self):
        """Pin the list to its source so a kernel-tree update cannot rot it silently.

        Every name in `QUANTIZE_PERF_UNMEASURABLE_TYPES` must be a declared type
        name; if the production tree later gains the missing trait, this list is
        wrong in the safe direction (a false FAIL a human sees) rather than the
        unsafe one — but it must never name a type the enum does not offer.
        """
        for name in R.QUANTIZE_PERF_UNMEASURABLE_TYPES:
            self.assertIn(name, R.GGML_TYPE_NAMES)
        self.assertTrue(
            set(R.GGML_TYPE_NAMES) - set(R.QUANTIZE_PERF_UNMEASURABLE_TYPES),
            "if every declared type were unmeasurable the recipe would be unusable")
        joined = " ".join(
            R.construct("t1a.llama_cpu.quantize_perf.v1",
                        binding=self.binding("test-quantize-perf"),
                        params=self.quant_params()).bounded)
        self.assertIn("still exits 0", joined)

    def test_backend_ops_csv_output_carries_no_metric_and_says_so(self):
        """`--output csv` emits a well-formed table with the measurement removed.

        `csv_printer` filters every row through `get_fields_csv()` — op_name,
        op_params, supported, error_message, test_mode, backend_reg_name,
        backend_name (tests/test-backend-ops.cpp:1091-1100). None of `time_us`,
        `flops`, `bandwidth_gb_s` or `n_runs` survives, so a recipe declaring
        `op_throughput_gflops` cannot be reduced from it at all. That is a FAIL,
        not a taste — the same rule llama-bench's `-o md` already gets.
        """
        bad = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                          binding=self.binding("test-backend-ops"),
                          params=self.ops_params(output_format="csv"))
        finding = bad.finding("raw_samples_retained")
        self.assertEqual(finding.check.outcome, S.FAIL)
        joined = " ".join(finding.check.reasons)
        self.assertIn("get_fields_csv()", joined)
        self.assertIn("op_throughput_gflops", joined)
        self.assertIn("NONE", bad.raw_samples_source)

        for fmt in R.BACKEND_OPS_METRIC_BEARING_FORMATS:
            with self.subTest(output_format=fmt):
                good = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                                   binding=self.binding("test-backend-ops"),
                                   params=self.ops_params(output_format=fmt))
                self.assertEqual(
                    good.finding("raw_samples_retained").check.outcome,
                    S.COULD_NOT_CHECK)
                self.assertNotIn("NONE", good.raw_samples_source)

    def test_the_default_backend_ops_output_format_can_carry_the_declared_metric(self):
        """A default that cannot express the recipe's own metric is a silent gap.

        The default is what a caller who states nothing gets, so it is the one
        value that must not need a discipline finding to be safe.
        """
        default = R.REGISTRY["t1a.llama_cpu.backend_ops_perf.v1"].param_map[
            "output_format"].default
        self.assertIn(default, R.BACKEND_OPS_METRIC_BEARING_FORMATS)
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params())
        argv = list(command.argv)
        self.assertEqual(argv[argv.index("--output") + 1], default)
        self.assertNotEqual(command.finding("raw_samples_retained").check.outcome, S.FAIL)

    def test_t1a_cache_state_is_declared_recorded_and_receipt_bound(self):
        binding = self.binding("test-backend-ops")
        cold = R.construct("t1a.llama_cpu.backend_ops_perf.v1", binding=binding,
                           params=self.ops_params())
        warm = R.construct("t1a.llama_cpu.backend_ops_perf.v1", binding=binding,
                           params=self.ops_params(cache_state="warm"))
        self.assertEqual(cold.params["cache_state"], "cold")
        self.assertEqual(warm.params["cache_state"], "warm")
        self.assertNotEqual(cold.receipt.argv_sha256, warm.receipt.argv_sha256)
        with self.assertRaisesRegex(R.RecipeParameterError, "cache_state"):
            R.construct("t1a.llama_cpu.backend_ops_perf.v1", binding=binding,
                        params=self.ops_params(cache_state="unknown"))

    def test_t1a_requires_a_local_aa_derived_minimum_duration(self):
        binding = self.binding("test-backend-ops")
        command = R.construct(
            "t1a.llama_cpu.backend_ops_perf.v1", binding=binding,
            params=self.ops_params())
        floor = command.params["min_measurable_us"]
        self.assertIsInstance(floor, api.MinimumMeasurableDuration)
        self.assertEqual(
            command.finding("minimum_measurable_duration").check.outcome, S.PASS)
        payload = command.to_dict()
        self.assertEqual(
            payload["params"]["min_measurable_us"]["samples_ref"],
            "fixture:paired-local-aa-durations")
        with self.assertRaisesRegex(R.RecipeParameterError, "bare microsecond"):
            R.construct(
                "t1a.llama_cpu.backend_ops_perf.v1", binding=binding,
                params=self.ops_params(min_measurable_us=10.0))

    def test_linkage_host_and_cli_checks_are_delegated_not_assumed(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        for finding_id, needle in (
                ("binary_linkage_resolution", "assert_explicit_bench_identity"),
                ("worktree_identity", "git rev-parse"),
                ("host_environment", "validate_host_environment"),
                ("tool_cli_contract", "without executing it")):
            with self.subTest(finding_id=finding_id):
                finding = command.finding(finding_id)
                self.assertEqual(finding.check.outcome, S.COULD_NOT_CHECK)
                self.assertIn(needle, " ".join(finding.check.reasons))

    def test_worst_outcome_ordering_and_empty_vector(self):
        pass_f = R.DisciplineFinding("a", S.Check(S.PASS), "clause")
        cnc_f = R.DisciplineFinding("b", S.Check(S.COULD_NOT_CHECK, ("x",)), "clause")
        fail_f = R.DisciplineFinding("c", S.Check(S.FAIL, ("y",)), "clause")
        self.assertEqual(R.worst_outcome([pass_f]), S.PASS)
        self.assertEqual(R.worst_outcome([pass_f, cnc_f]), S.COULD_NOT_CHECK)
        self.assertEqual(R.worst_outcome([pass_f, cnc_f, fail_f]), S.FAIL)
        self.assertEqual(R.worst_outcome([]), S.COULD_NOT_CHECK,
                         "an empty discipline vector checked nothing; that is the third "
                         "outcome, never a pass")

    def test_unknown_finding_id_raises_rather_than_returning_none(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        with self.assertRaises(KeyError):
            command.finding("no_such_finding")

    def test_env_flag_variant_is_declared_and_every_other_key_still_enforced(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(),
                              params=self.decode_params(ggml_iqk="0"))
        self.assertEqual(command.env["GGML_IQK"], "0")
        finding = command.finding("canonical_env_stack")
        self.assertEqual(finding.check.outcome, S.PASS)
        joined = " ".join(finding.check.reasons)
        self.assertIn("declared env-flag variant under test: ['GGML_IQK']", joined)
        self.assertIn("every other canonical key was still enforced", joined)
        for key, value in R.CANONICAL_OMP_ENV.items():
            if key != "GGML_IQK":
                self.assertEqual(command.env[key], value)

    def test_ggml_iqk_outside_its_domain_is_refused(self):
        with self.assertRaises(R.RecipeParameterError):
            R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(ggml_iqk="on"))

    def test_a_drifted_canonical_env_raises_rather_than_emitting(self):
        broken = dict(R.CANONICAL_OMP_ENV)
        broken["OMP_DYNAMIC"] = "true"
        with mock.patch.object(R, "CANONICAL_OMP_ENV", broken):
            with self.assertRaises(R.RecipeDriftError) as ctx:
                R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                            binding=self.binding(), params=self.decode_params())
        self.assertIn("ratified canonical env validator", str(ctx.exception))

    def test_a_drifted_canonical_cmd_raises_rather_than_emitting(self):
        # Drop `-mmp 0`: mmap=ON defeats --interleave=all striping on EPYC.
        with mock.patch.object(R, "CANONICAL_BENCH_FLAGS", ("-t", "96", "-fa", "1")):
            with self.assertRaises(R.RecipeDriftError) as ctx:
                R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                            binding=self.binding(), params=self.decode_params())
        self.assertIn("ratified canonical cmd validator", str(ctx.exception))

    def test_a_reordered_prefix_raises(self):
        # numactl BEFORE taskset is the 2026-05-02 shape; taskset must wrap numactl.
        with mock.patch.object(R, "CANONICAL_PREFIX", ("numactl", "--interleave=all",
                                                       "taskset", "-c", "0-95")):
            with self.assertRaises(R.RecipeDriftError):
                R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                            binding=self.binding("test-backend-ops"),
                            params=self.ops_params())

    def test_every_recipe_states_the_bounds_it_applied(self):
        for recipe_id, binding, params in self.all_cells():
            with self.subTest(recipe_id=recipe_id):
                command = R.construct(recipe_id, binding=binding, params=params)
                self.assertTrue(command.bounded,
                                "a recipe that bounds something silently is the "
                                "'no silent caps' failure")
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params())
        joined = " ".join(command.bounded)
        self.assertIn("REFUSED, never truncated", joined)
        self.assertIn("NOT constructible", joined)


# =============================================================================
# The receipt, and the seam into api.py
# =============================================================================

class ReceiptTest(FakeTreeMixin, unittest.TestCase):

    def test_receipt_is_an_api_recipe_receipt_and_renders_the_grammar_field(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        self.assertIsInstance(command.receipt, api.RecipeReceipt)
        rendered = command.receipt.render()
        self.assertTrue(rendered.startswith("t1b.llama_cpu.llama_bench_decode.v1@"))
        self.assertEqual(len(rendered.split("@")[1]), 12)

    def test_argv_hash_changes_with_the_argv(self):
        a = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(n_gen=64))
        b = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(n_gen=65))
        self.assertNotEqual(a.receipt.argv_sha256, b.receipt.argv_sha256)

    def test_argv_hash_changes_with_the_env(self):
        a = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(ggml_iqk="1"))
        b = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(ggml_iqk="0"))
        self.assertNotEqual(a.receipt.argv_sha256, b.receipt.argv_sha256)

    def test_argv_hash_distinguishes_the_two_arms(self):
        a = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(), arm="candidate")
        b = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params(), arm="anchor")
        self.assertNotEqual(a.receipt.argv_sha256, b.receipt.argv_sha256)

    def test_argv_hash_is_stable_for_an_identical_construction(self):
        a = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        b = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        self.assertEqual(a.receipt.argv_sha256, b.receipt.argv_sha256)
        self.assertEqual(a.receipt.constructor_sha256, b.receipt.constructor_sha256)

    def test_constructor_hash_follows_the_sourced_constants(self):
        before = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                             binding=self.binding(), params=self.decode_params())
        patched = dict(R._MODULE_HASHES)
        patched["scripts/lib/canonical_recipe.py"] = _sha("edited")
        with mock.patch.object(R, "_MODULE_HASHES", patched):
            after = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                                binding=self.binding(), params=self.decode_params())
        self.assertNotEqual(before.receipt.constructor_sha256,
                            after.receipt.constructor_sha256,
                            "an edit to canonical_recipe.py must change the recipe hash "
                            "of every record that cites it")

    def test_a_cpu_receipt_does_not_change_because_a_gpu_recipe_ran_first(self):
        """A receipt must be a function of its recipe, not of process history.

        The GPU recipe resolves and hashes `architect_bench_gpu_lib.sh`, growing the
        module-hash table. If the constructor hash covered the whole table, the same
        CPU recipe would carry two different `recipe=` fields depending on ordering.
        """
        before = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                             binding=self.binding(), params=self.decode_params())
        R.construct("t1a.llama_gpu.backend_ops_perf.v1",
                    binding=self.binding("test-backend-ops"),
                    params=self.ops_params(device_index=0, device_id="mi210_0"))
        after = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                            binding=self.binding(), params=self.decode_params())
        self.assertEqual(before.receipt.constructor_sha256,
                         after.receipt.constructor_sha256)

    def test_two_recipes_have_different_constructor_hashes(self):
        a = R.construct("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                        params=self.decode_params())
        b = R.construct("t1b.llama_cpu.llama_bench_prefill.v1", binding=self.binding(),
                        params={"model": str(self.model), "n_prompt": 512, "reps": 10})
        self.assertNotEqual(a.receipt.constructor_sha256, b.receipt.constructor_sha256)

    def test_receipt_satisfies_the_windows_hand_typed_argv_check(self):
        """The absence of this receipt is what `HAND_TYPED_ARGV` detects."""
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        request = _request(command)
        scan = api.check_void_conditions(request, _window(recipe=command.receipt),
                                         rate_comparison=False)
        self.assertNotIn("HAND_TYPED_ARGV", [f.reason for f in scan.findings])

        scan = api.check_void_conditions(request, _window(recipe=None),
                                         rate_comparison=False)
        self.assertIn("HAND_TYPED_ARGV", [f.reason for f in scan.findings])

    def test_receipt_satisfies_precondition_6(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        request = _request(command)
        scan = api.check_preconditions(request, _window(recipe=command.receipt))
        self.assertEqual(scan.get("codified_recipe").outcome, S.PASS)
        self.assertNotIn("codified_recipe", scan.unsatisfied)
        without = api.check_preconditions(request, _window(recipe=None))
        self.assertEqual(without.get("codified_recipe").outcome, S.FAIL)
        self.assertIn("recipe constructor",
                      " ".join(without.get("codified_recipe").reasons))

    def test_the_grammar_line_carries_the_recipe_id_and_hash(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        window = _window(recipe=command.receipt)
        request = _request(command)
        gates = (api.GateResult(gate_id="op_reference_parity", gate_class="correctness",
                                check=S.Check(S.PASS)),)
        preconditions = api.check_preconditions(request, window)
        void_scan = api.check_void_conditions(request, window, rate_comparison=False)
        grammar_complete = api.check_record_grammar_complete(
            request=request, window=window, effect=None)
        search_grade = api.evaluate_search_grade(
            request=request, window=window, preconditions=preconditions, effect=None,
            grammar_complete=grammar_complete)
        verdict = api.compute_verdict(tier=request.tier, gates=gates,
                                      void_scan=void_scan, search_grade=search_grade,
                                      anchor=request.anchor, effect=None)
        line = api.render_search_record_grammar(request=request, window=window,
                                                verdict=verdict, effect=None)
        self.assertIn(f"recipe={command.receipt.render()}", line)
        self.assertIn("SEARCH RECORD, NOT A CLAIM", line)
        self.assertEqual(grammar_complete.outcome, S.PASS, grammar_complete.reasons)

    def test_a_hand_typed_run_cannot_produce_a_complete_grammar_line(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        request = _request(command)
        complete = api.check_record_grammar_complete(
            request=request, window=_window(recipe=None), effect=None)
        self.assertEqual(complete.outcome, S.FAIL)
        self.assertIn("recipe: no recipe-constructor identity was recorded",
                      complete.reasons)


class ConstructorSeamTest(FakeTreeMixin, unittest.TestCase):

    def _constructor(self, recipe_id="t1b.llama_cpu.llama_bench_decode.v1", **kwargs):
        params = kwargs.pop("params", None) or self.decode_params()
        return R.AutoKernelRecipeConstructor(recipe_id, binding=self.binding(),
                                             params=params, **kwargs)

    def test_constructor_id_is_the_recipe_id_a_record_cites(self):
        constructor = self._constructor()
        self.assertEqual(constructor.constructor_id,
                         "t1b.llama_cpu.llama_bench_decode.v1")

    def test_construct_returns_the_protocol_tuple(self):
        constructor = self._constructor()
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        argv, env, receipt = constructor.construct(_request(command))
        self.assertEqual(argv, command.argv)
        self.assertEqual(env, command.env)
        self.assertEqual(receipt.argv_sha256, command.receipt.argv_sha256)

    def test_a_request_for_a_different_backend_is_refused(self):
        constructor = self._constructor()
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        request = _request(command, backend="llama_gpu")
        with self.assertRaises(R.RecipeRequestMismatch) as ctx:
            constructor.construct(request)
        self.assertIn("backend", str(ctx.exception))

    def test_a_request_for_a_different_phase_is_refused(self):
        constructor = self._constructor()
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        with self.assertRaises(R.RecipeRequestMismatch):
            constructor.construct(_request(command, phase="prefill"))

    def test_a_request_for_a_different_metric_is_refused(self):
        constructor = self._constructor()
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        with self.assertRaises(R.RecipeRequestMismatch):
            constructor.construct(_request(command, metric="prefill_tokens_per_s"))

    def test_check_request_on_a_non_request_is_could_not_check(self):
        self.assertEqual(self._constructor().check_request(object()).outcome,
                         S.COULD_NOT_CHECK)

    def test_the_seam_matches_the_api_protocol_shape(self):
        # `api.RecipeConstructor` is a plain (non-runtime-checkable) Protocol, so
        # the shape is asserted structurally rather than with isinstance.
        constructor = self._constructor()
        self.assertIsInstance(constructor.constructor_id, str)
        self.assertTrue(callable(constructor.construct))
        for name in api.RecipeConstructor.__annotations__:
            self.assertTrue(hasattr(constructor, name),
                            f"seam is missing Protocol attribute {name!r}")


# =============================================================================
# dry_run and rendering
# =============================================================================

class DryRunTest(FakeTreeMixin, unittest.TestCase):

    def test_dry_run_returns_the_exact_argv_and_is_canonicalizable(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        payload = R.dry_run("t1b.llama_cpu.llama_bench_decode.v1", binding=self.binding(),
                            params=self.decode_params())
        self.assertEqual(payload["argv"], list(command.argv))
        self.assertEqual(payload["env"], command.env)
        self.assertTrue(payload["dry_run"])
        S.canonical_json(payload)  # raises if anything is not canonicalizable

    def test_dry_run_payload_contains_no_tuples(self):
        payload = R.dry_run("t1a.llama_gpu.backend_ops_perf.v1",
                            binding=self.binding("test-backend-ops"),
                            params=self.ops_params(device_index=0, device_id="mi210_0"))
        self._assert_no_tuples(payload, "$")

    def _assert_no_tuples(self, value, path):
        self.assertNotIsInstance(value, tuple, f"{path} is a tuple")
        if isinstance(value, dict):
            for key, item in value.items():
                self._assert_no_tuples(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._assert_no_tuples(item, f"{path}[{index}]")

    def test_dry_run_of_an_unregistered_recipe_is_refused(self):
        with self.assertRaises(R.UnregisteredRecipe):
            R.dry_run("nope.v1", binding=self.binding())

    def test_human_readable_quotes_a_token_that_needs_it(self):
        command = R.construct("t1a.llama_cpu.backend_ops_perf.v1",
                              binding=self.binding("test-backend-ops"),
                              params=self.ops_params(params_filter="ne=[4096,1,1,1]"))
        rendered = command.render_human_readable()
        self.assertIn("'ne=[4096,1,1,1]'", rendered)

    def test_human_readable_shows_env_then_argv(self):
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        rendered = command.render_human_readable()
        self.assertLess(rendered.index("OMP_PROC_BIND=spread"), rendered.index("taskset"))
        self.assertIn("-t 96 -fa 1 -mmp 0", rendered)

    def test_the_payload_is_embeddable_in_an_evaluation_events_free_form_block(self):
        """The discipline vector has to travel with the record, not stay in a log.

        `evaluation_event.v3` has no top-level home for it, so it rides inside
        `performance.search_discipline` — the same block `api.build_evaluation_event`
        uses for the grammar fields the schema does not name.
        """
        command = R.construct("t1b.llama_cpu.llama_bench_decode.v1",
                              binding=self.binding(), params=self.decode_params())
        payload = command.to_dict()
        block = {"raw_samples": ["akr-1"], "paired_blocks": 10, "estimate": None,
                 "uncertainty": None,
                 "search_discipline": {"recipe": payload["recipe"],
                                       "discipline": payload["discipline"],
                                       "claim_footprint": payload["claim_footprint"]}}
        S.canonical_json(block)
        self.assertEqual(block["search_discipline"]["recipe"]["constructor_id"],
                         command.recipe_id)

    def test_to_dict_round_trips_through_the_content_hasher(self):
        command = R.construct("t1a.llama_cpu.quantize_perf.v1",
                              binding=self.binding("test-quantize-perf"),
                              params=self.quant_params(iterations=100))
        self.assertRegex(S.content_hash(command.to_dict()), r"^[0-9a-f]{64}$")


# =============================================================================
# The module runs nothing — proved from its own AST
# =============================================================================

class SelfAuditTest(unittest.TestCase):

    def test_module_has_no_execution_or_write_path(self):
        result = R.audit_no_execution_paths()
        self.assertEqual(result.outcome, S.PASS, result.reasons)

    def test_the_audit_detects_a_subprocess_import(self):
        result = R.audit_no_execution_paths(
            "import subprocess\ndef go():\n    return subprocess\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("subprocess", " ".join(result.reasons))

    def test_the_audit_detects_a_process_launch(self):
        result = R.audit_no_execution_paths(
            "def go(runner):\n    return runner.check_output(['ls'])\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_the_audit_detects_a_write(self):
        result = R.audit_no_execution_paths(
            "from pathlib import Path\n"
            "def go(p):\n    Path(p).write_text('x')\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_the_audit_detects_an_ambient_environment_read(self):
        result = R.audit_no_execution_paths(
            "import sys\ndef go(mod):\n    return mod.environ.get('LD_LIBRARY_PATH')\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("fully declared", " ".join(result.reasons))

    def test_the_audit_reports_could_not_check_on_unparseable_source(self):
        result = R.audit_no_execution_paths("def (:\n")
        self.assertEqual(result.outcome, S.COULD_NOT_CHECK)

    def test_the_audit_cannot_be_passed_by_deleting_what_it_inspects(self):
        """An empty AST contains no forbidden node because it contains no node.

        This is the "can I pass this check by removing the thing it inspects?"
        test: PASS on a blank body would make the guarantee "constructs nothing,
        executes nothing" satisfiable by having no code at all.
        """
        for empty in ("", "   ", "\n\n", "# only a comment\n", '"""docstring only"""'):
            with self.subTest(source=repr(empty)):
                result = R.audit_no_execution_paths(empty)
                self.assertEqual(
                    result.outcome, S.COULD_NOT_CHECK,
                    "an empty or comment-only body must never certify the module")

    def test_the_audit_refuses_to_certify_a_file_that_is_not_the_running_module(self):
        """`source=None` reads from DISK; the running module is in memory.

        Auditing bytes that were edited after import would report on a file nobody
        loaded — "verify THE consumer, not A consumer". The audit binds what it
        reads to the import-time hash and degrades to COULD_NOT_CHECK on drift.
        """
        drifted = dict(R._MODULE_HASHES)
        drifted[R._SELF_REL_PATH] = _sha("some other revision of recipes.py")
        with mock.patch.object(R, "_MODULE_HASHES", drifted):
            result = R.audit_no_execution_paths()
        self.assertEqual(result.outcome, S.COULD_NOT_CHECK)
        self.assertIn("not the running module", " ".join(result.reasons))
        # ...and the undrifted call still PASSes, naming what it audited.
        clean = R.audit_no_execution_paths()
        self.assertEqual(clean.outcome, S.PASS)
        self.assertIn(R._MODULE_HASHES[R._SELF_REL_PATH][:12], " ".join(clean.reasons))

    def test_every_refusal_class_exists_before_the_first_statement_that_raises_one(self):
        """`_MODULE_HASHES` is built at import time and its handler raises.

        When the error classes were defined BELOW that statement, an unreadable
        `canonical_recipe.py` produced `NameError: SourcedConstantUnavailable`
        instead of the refusal the handler was written to make — the failure mode
        hidden inside a `# pragma: no cover` branch. Definition order is the fix,
        so assert the order.
        """
        tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
        class_lines = {node.name: node.lineno for node in tree.body
                       if isinstance(node, ast.ClassDef)}
        targets_of = {
            ast.Assign: lambda n: n.targets,
            ast.AnnAssign: lambda n: [n.target],
        }
        hashes_line = min(
            node.lineno for node in tree.body
            if type(node) in targets_of
            and any(getattr(t, "id", None) == "_MODULE_HASHES"
                    for t in targets_of[type(node)](node)))
        for name in ("RecipeError", "SourcedConstantUnavailable"):
            self.assertIn(name, class_lines)
            self.assertLess(class_lines[name], hashes_line,
                            f"{name} must be defined before _MODULE_HASHES, whose "
                            f"import-time OSError handler raises it")

    def test_an_unhashable_sourced_constant_raises_the_declared_refusal(self):
        tmp = Path(tempfile.mkdtemp(prefix="ak-recipes-hash-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with self.assertRaises(R.SourcedConstantUnavailable):
            R._sha256_file(tmp / "definitely-not-here" / "canonical_recipe.py")

    def test_no_module_level_call_launches_anything(self):
        """Belt and braces: walk the AST for any name resembling an executor."""
        tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
        banned = {"run", "call", "check_output", "check_call", "Popen", "system",
                  "spawnv", "popen"}
        offenders = [
            f"line {node.lineno}: {node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in banned
        ]
        self.assertEqual(offenders, [])


# =============================================================================
# Fixtures for the api seam
# =============================================================================

def _anchor() -> api.AnchorIdentity:
    return api.AnchorIdentity(source_commit=V8_COMMIT, binary_sha256=_sha("bin"),
                              linkage_sha256=_sha("link"),
                              measurement_event_ids=("ake-anchor-1",))


def _controls() -> api.ControlPanel:
    return api.ControlPanel(
        positive=S.Check(S.PASS), neutral=S.Check(S.PASS),
        degraded_negative=S.Check(S.PASS), aa=S.Check(S.PASS),
        historical_replay=S.Check(S.PASS))


def _window(*, recipe) -> api.WindowAttestations:
    ok = S.Check(S.PASS)
    return api.WindowAttestations(
        resource_claim_receipt="akc-claim-1", resource_claim_open=ok,
        resource_claim_close=ok, resource_claim_same_holder=ok,
        no_concurrent_inference=ok, preflight_attestation_ref="akp-1",
        host_receipt="akh-1", host_health=ok,
        anchor_at_open=_anchor(), anchor_at_close=_anchor(), anchor_gate=ok,
        evaluator_bundle=ok, runtime_source_label=ok,
        recipe=recipe, storage_open=ok, storage_close=ok, strata=ok,
        stopping_rule_id="stop-v1", rule_immutability=ok, order_randomized=ok,
        order_seed="seed-1", aa_cadence=ok, controls=_controls(), calibration=ok,
        control_definitions_immutable=ok,
        raw_evidence_ref="akr-1")


def _request(command: R.ConstructedCommand, **overrides) -> api.EvaluationRequest:
    kwargs = dict(
        event_id="ake-1", campaign_id="ak-1", candidate_id="akc-1",
        tier=command.tier, backend=command.backend, phase=command.phase,
        cell_class=command.cell_class, protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(source_sha256=_sha("src"),
                                      binary_sha256=_sha("cbin"),
                                      linkage_sha256=_sha("clink")),
        anchor=_anchor(),
        evaluator=api.EvaluatorIdentity(id="ak-evaluator/v1",
                                        bundle_sha256=_sha("bundle"),
                                        runtime_source_label_ref="aksl-1"),
        scope_denominator=command.scope_denominator,
        scope_manifest_sha256=_sha("scope"), co_residency="single",
        determinism=api.DeterminismReport(determinism_class="not_measured",
                                          same_seed_repeat_runs=0),
        metric=command.metric, metric_direction=command.metric_direction,
        reps=10, change_class="parameter", anchor_tier=command.tier,
        transfer_ratio_to=(), created_at=NOW, campaign_controls=None, calibration=None)
    kwargs.update(overrides)
    return api.EvaluationRequest(**kwargs)


class WorstOutcomeIsTheOneLatticeTest(unittest.TestCase):
    """`recipes.worst_outcome` delegates to `schemas.Check.worst_of`.

    This reducer already answered the empty case correctly. The delegation keeps
    it correct by construction rather than by a local `if`, and makes it give the
    same answer as every other reducer.
    """

    @staticmethod
    def _finding(outcome, reasons=()):
        return R.DisciplineFinding(finding_id="f", check=S.Check(outcome, reasons),
                                   clause="c")

    def test_an_empty_discipline_vector_is_could_not_check_and_never_pass(self):
        self.assertEqual(R.worst_outcome([]), S.COULD_NOT_CHECK)
        self.assertEqual(R.worst_outcome(iter([])), S.COULD_NOT_CHECK)

    def test_the_delegation_is_real_and_not_a_reimplementation(self):
        for outcomes in ([], [S.PASS], [S.PASS, S.COULD_NOT_CHECK],
                         [S.COULD_NOT_CHECK, S.FAIL], [S.FAIL, S.PASS]):
            with self.subTest(outcomes=outcomes):
                findings = [self._finding(o, ("r",)) for o in outcomes]
                self.assertEqual(
                    R.worst_outcome(findings),
                    S.Check.worst_of(f.check for f in findings).outcome)


if __name__ == "__main__":
    unittest.main(verbosity=2)
