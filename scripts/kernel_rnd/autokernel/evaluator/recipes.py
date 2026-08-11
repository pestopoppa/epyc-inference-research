#!/usr/bin/env python3
"""recipes.py — the codified benchmark-recipe constructor for AK3 tier T1.

WHY THIS MODULE EXISTS
----------------------
`measurement/protocols/kernel-research.md` (Annex K, **P-AK-SEARCH-1**, RATIFIED
2026-08-03), *"Preconditions (all enforced or attested per run)"* item 6:

    **Codified recipe.** Every measurement command line emitted inside this
    protocol's scope is emitted by a recipe constructor; the constructor's
    identifier and content hash are recorded with the record. **Hand-typed argv
    voids the run** (`bench-cpu.md:8-10`, `MEASUREMENT_POLICY.md:37`).

and *"What voids a run"* lists `hand-typed argv` among the twelve void
conditions. `OPERATING_CONSTRAINTS.md:38` states the same rule for the whole
project — *"Throughput numbers only via the codified recipes … never hand-typed
bench commands"* — and `bench-cpu.md:8-10` names the entry point.

The project already has that constructor **for whole-model llama-bench runs**:
`scripts/lib/canonical_recipe.py`, wrapped by `scripts/benchmark/bench_canonical.sh`.
It has none for **operator-level microbenchmarks**, and the owning design records
that absence as substrate that blocks the loop:

    | **Codified microbenchmark recipe** | `OPERATING_CONSTRAINTS.md:38` requires
    *all* throughput numbers to go through `bench_canonical.sh`/`canonical_recipe.py`;
    no constructor exists for operator-level microbenchmarks | T1a on every backend |
    — `handoffs/active/autokernel-research-loop.md` §2.6

That gap is why **T1 could not legally run**: §9.3 requires *"Argv is constructed
by the codified microbenchmark recipe (§2.5), never hand-typed"*, and there was
nothing to construct it with. This module closes it.

WHICH FAILURE IT PREVENTS
-------------------------
Recipe drift, which has cost this project measurable days:

  * 2026-05-02 — the launcher had drifted off the recipe: missing `taskset`,
    `mmap` defaulted to ON (which defeats `--interleave=all` striping), AOCC
    libomp resolved instead of clang-20.
  * 2026-05-28 — **seven** compounding drift bugs in a single bench run: wrong
    binary, wrong libomp, missing `OMP_DYNAMIC=false`, THP defrag reset,
    `perf_event_paranoid` reset, and a RUNPATH-vs-`LD_LIBRARY_PATH` mismatch that
    silently resolved an experimental binary against production libraries.

Both are documented in `canonical_recipe.py`'s own module docstring, and the fix
in both cases was *"use the codified recipe, don't invent the command."* This
module therefore **imports** the ratified CPU constants (`CANONICAL_PREFIX`,
`CANONICAL_BENCH_FLAGS_LLAMA_BENCH`, `CANONICAL_OMP_ENV`, `LLVM20_LIBDIR`) and
re-validates its output through `canonical_recipe`'s own assertions rather than
retyping any of them. A drift in the ratified constants propagates here on the
next import; a divergence between the two is not expressible.

WHICH PROTOCOL CLAUSES IT IMPLEMENTS (by section name)
------------------------------------------------------
`measurement/protocols/kernel-research.md`:

  * *"Preconditions (all enforced or attested per run)"* precondition 6 —
    `construct()` is the only way to obtain an argv, `RecipeReceipt` carries the
    constructor id and content hash, and an unregistered `recipe_id` is refused
    with `UnregisteredRecipe` rather than constructed on the fly.
  * *"Preconditions …"* precondition 1 — `ClaimFootprint` is **derived from the
    constructed argv's own `taskset -c` list**, not declared beside it, so the
    region claim the runner acquires is provably the footprint the command pins
    (*"A CPU region claim covering the exact footprint measured"*). The same
    derivation is what `bench_canonical.sh` does at its A0 region-lock step.
  * *"What voids a run"* — `hand-typed argv`: the receipt's `argv_sha256` binds
    the exact argv+env, so a record can be checked against the command that
    produced it.
  * *"Record grammar"* — the `recipe=<recipe_constructor_id>@<recipe_sha256[:12]>`
    field is `RecipeReceipt.render()`, and `recipe_id` is the stable key a
    P-AK-SEARCH-1 record cites.
  * *"What this protocol does NOT authorize"* denial 2 (*"No production write of
    any kind, including building in … any production tree"*) — a **candidate**
    arm whose binary or source root resolves inside a frozen production tree is
    refused. The **anchor** arm is allowed there, because the anchor IS the
    frozen production binary and executing it read-only is not a write.
  * *"What this protocol does NOT authorize"* denial 6 (*"a controller that
    discovers a coverage gap in its evaluator RECORDS the gap … it does not patch
    the instrument"*) — every place the ratified discipline is unsatisfiable for a
    tool is emitted as a `DisciplineFinding`, not silently smoothed over. Two are
    real and known: `test-backend-ops` has no thread-count flag, and there is no
    codified GPU env stack. Both are reported, neither is invented around.

Design context: `handoffs/active/autokernel-research-loop.md` §2.6 (the missing
substrate), §9.3 (T1a — target operator discriminator), §9.4 (T1b — tiny
real-graph translation), §13.1/§13.2 (backend adapter responsibilities), phase
AK3 (*"Build the codified operator-microbenchmark recipe constructor so T1 argv
is constructed, and bind it to a recipe id `P-AK-SEARCH-1` can cite"*).

WHAT THIS MODULE IS NOT
-----------------------
**It executes nothing.** It constructs argv and validates it. It runs no
inference, no benchmark and no build; it starts, stops and signals no process;
it writes no file. Those are not promises in prose: `audit_no_execution_paths()`
parses this module's own AST and FAILs on any write-capable call, process call,
or import of `os`/`subprocess`/`shutil`/`signal`/… — and it additionally FAILs on
any read of `os.environ`, because **the emitted environment is fully declared**.
`canonical_recipe.build_canonical_env()` starts from `os.environ.copy()`, which
is right for a launcher and wrong for a hashed recipe: an ambient variable would
make `argv_sha256` a function of whoever's shell invoked the constructor.

Every filesystem touch is a read-only `stat`/`read_bytes`, and every one of them
is reported as a `schemas.Check` in `ConstructedCommand.input_checks`. Disabling
input verification does not produce PASS; it produces `COULD_NOT_CHECK` with the
reason.

The checks this constructor **cannot** perform without executing something —
`ldd` linkage resolution, `git rev-parse` worktree identity, the host-environment
sysfs sweep, and whether the candidate binary's CLI actually accepts these flags
— are `COULD_NOT_CHECK` findings naming the ratified validator the runner must
call (`canonical_recipe.assert_explicit_bench_identity`,
`canonical_recipe.validate_host_environment`). They are delegated in the return
value, never assumed away.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .. import schemas, storage
from . import api

# =============================================================================
# Errors — every one is a refusal, never a degraded command
# =============================================================================
#
# These are declared FIRST, before any module-level statement that can raise one.
# `_MODULE_HASHES` is populated at import time by `_sha256_file`, whose OSError
# handler raises `SourcedConstantUnavailable`; when the error classes lived below
# that statement a genuinely unreadable `canonical_recipe.py` produced
# `NameError: SourcedConstantUnavailable` instead of the refusal the handler was
# written to make. Definition order is load-bearing here, not cosmetic.


class RecipeError(api.EvaluatorError):
    """Base class for every refusal this module makes."""


class UnregisteredRecipe(RecipeError):
    """A `recipe_id` that is not in the registry was handed to `construct()`.

    Precondition 6 admits exactly one source of argv. Constructing "something
    reasonable" for an unknown id would make the constructor a hand-typing
    surface with extra steps, and the resulting record would cite a recipe id
    that resolves to nothing.
    """


class RecipeParameterError(RecipeError):
    """A parameter was missing, of the wrong type, or outside its declared domain.

    Parameters are the only caller-controlled input to argv construction, so this
    is where argv injection would enter. Every value is validated against a
    declared type/domain and appended as its **own argv element**; no parameter
    is ever interpolated into a token.
    """


class RecipeBindingError(RecipeError):
    """The binary / source-root / library-path triple is incomplete or illegal."""


class RecipeDriftError(RecipeError):
    """A constructed command failed the ratified canonical validators.

    This should be unreachable: it means this module's own construction disagrees
    with `canonical_recipe`'s assertions. It raises rather than warning, because a
    command that fails the recipe validators is exactly the 2026-05-02 /
    2026-05-28 defect class.
    """


class RecipeRequestMismatch(RecipeError):
    """An `EvaluationRequest` and the selected recipe describe different cells.

    Backend, tier, phase, metric, metric direction and cell class must agree.
    Emitting a CPU recipe for a GPU request, or a decode recipe for a prefill
    cell, would attach a number to a cell it was not measured in — the
    `MEASUREMENT.md:25-26` substitution the protocol's Metric clause forbids.
    """


class SourcedConstantUnavailable(RecipeError):
    """A codified constant this recipe depends on could not be resolved.

    Explicit failure over silent fallback: retyping the value here is precisely
    the drift this module exists to prevent.
    """


# =============================================================================
# Where the ratified constants live, and proof that we bound the right file
# =============================================================================

_HERE = Path(__file__).resolve()

#: `<repo>/scripts/kernel_rnd/autokernel/evaluator/recipes.py` -> `<repo>`.
REPO_ROOT = _HERE.parents[4]

#: The single source of truth for the canonical CPU bench recipe (`bench-cpu.md:8-10`).
CANONICAL_RECIPE_PATH = REPO_ROOT / "scripts" / "lib" / "canonical_recipe.py"

#: The codified GPU launcher. Sourced lazily, for the GPU host-thread pinning
#: only — see `gpu_host_cpu_list()`.
GPU_BENCH_LIB_PATH = REPO_ROOT / "scripts" / "benchmark" / "architect_bench_gpu_lib.sh"


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes. Raises rather than returning a placeholder."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - exercised via the injected-path tests
        raise SourcedConstantUnavailable(
            f"cannot hash {path}: {exc}. A recipe whose provenance cannot be hashed is "
            f"not a codified recipe."
        ) from exc


def _load_canonical_recipe():
    """Bind `scripts/lib/canonical_recipe.py`, or refuse to import at all.

    The ordinary package import is tried first so that a process which already
    holds `scripts.lib.canonical_recipe` gets THAT module object rather than a
    second execution of the same file — one source of truth that exists twice is
    not one source of truth (the `autokernel/storage.py` seam defect, README).
    The file-location fallback exists because this package is imported by putting
    `scripts/kernel_rnd` on `sys.path` (README, "Import convention"), which does
    not put the repository root there.

    Either way the bound module's `__file__` MUST resolve to
    `CANONICAL_RECIPE_PATH`; a foreign module is an ImportError, never a
    fallback.
    """
    if not CANONICAL_RECIPE_PATH.is_file():
        raise ImportError(
            f"the ratified canonical bench recipe is missing: {CANONICAL_RECIPE_PATH}. "
            f"This module exists to emit argv from that file's constants; without it "
            f"there is no codified recipe and P-AK-SEARCH-1 precondition 6 cannot be "
            f"satisfied. Refusing to import rather than retyping the recipe."
        )
    module = sys.modules.get("scripts.lib.canonical_recipe")
    if module is None:
        try:
            module = importlib.import_module("scripts.lib.canonical_recipe")
        except ImportError:
            spec = importlib.util.spec_from_file_location(
                "_autokernel_canonical_recipe", CANONICAL_RECIPE_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"could not build an import spec for {CANONICAL_RECIPE_PATH}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            # An import is not a write. The path was resolved from this file's own
            # location and is re-asserted below before anything is read from it.
            spec.loader.exec_module(module)
    bound = Path(getattr(module, "__file__", "")).resolve()
    if bound != CANONICAL_RECIPE_PATH:
        raise ImportError(
            f"autokernel.evaluator.recipes bound a foreign canonical_recipe module: "
            f"{bound} is not {CANONICAL_RECIPE_PATH}"
        )
    return module


canonical_recipe = _load_canonical_recipe()

if Path(schemas.__file__).resolve() != _HERE.parents[1] / "schemas.py":  # pragma: no cover
    raise ImportError(
        f"autokernel.evaluator.recipes bound a foreign schemas module: {schemas.__file__}")

_SELF_REL_PATH = "scripts/kernel_rnd/autokernel/evaluator/recipes.py"
_CANONICAL_REL_PATH = "scripts/lib/canonical_recipe.py"
_GPU_LIB_REL_PATH = "scripts/benchmark/architect_bench_gpu_lib.sh"

#: Content hash of every file that supplied a constant to a constructed command.
#: Recorded inside `RecipeReceipt.constructor_sha256`, so "which recipe ran" is a
#: checkable fact rather than an inference from an import statement — the same
#: shape precondition 5 requires of the evaluator bundle. Entries are added as
#: constants are resolved; `_constructor_sha256` selects the subset a given recipe
#: actually used, so the table's growth never changes an unrelated receipt.
_MODULE_HASHES: dict = {
    _SELF_REL_PATH: _sha256_file(_HERE),
    _CANONICAL_REL_PATH: _sha256_file(CANONICAL_RECIPE_PATH),
}


# =============================================================================
# Identity
# =============================================================================

#: Bumping this is a schema-version event: it changes the constructor hash of
#: every recipe and therefore the `recipe=` field of every record cited against
#: one. Records cite `<recipe_id>@<constructor_sha256[:12]>`, so a bump does not
#: silently re-label old records — it makes the difference visible.
REGISTRY_ID = "ak-recipe-registry/v1"

#: The `RecipeConstructor.constructor_id` of the seam implementation. Individual
#: recipes are cited by their own `recipe_id`, which is what lands in the record.
CONSTRUCTOR_MODULE_ID = "autokernel.evaluator.recipes/v1"

RECIPE_FAMILY_T1A = "T1a_operator_microbench"
RECIPE_FAMILY_T1B = "T1b_tiny_real_graph"
RECIPE_FAMILIES = (RECIPE_FAMILY_T1A, RECIPE_FAMILY_T1B)

#: An arm is the anchor or the candidate. There is deliberately no third value:
#: a measurement that is neither is not a paired-block measurement.
ARMS = ("anchor", "candidate")

#: Cell classes. The calibration block is solved per (backend, phase, cell class)
#: and *"Values calibrated under a different host state, backend, phase, or cell
#: class MUST NOT be reused"* — so the cell class is a property of the recipe,
#: not a label the caller chooses.
CELL_CLASS_OPERATOR = "operator_microbench"
CELL_CLASS_TINY_GRAPH = "tiny_real_graph"


# =============================================================================
# Codified constants, sourced — never retyped
# =============================================================================

# The four names below are IMPORT-TIME SNAPSHOTS of the ratified constants. They
# are what the builders emit; the ratified module's own validators
# (`assert_canonical_cmd`, `assert_canonical_env`) then check the result against
# the LIVE values. Two sources that must agree is the point: an edit to
# `canonical_recipe.py` propagates here on the next import, and any divergence
# between what this module emits and what is ratified raises `RecipeDriftError`
# instead of producing a quietly different measurement.

#: `taskset -c 0-95 numactl --interleave=all` — imported, not retyped.
CANONICAL_PREFIX: tuple = tuple(canonical_recipe.CANONICAL_PREFIX)

#: `-t 96 -fa 1 -mmp 0` — the explicit thread/flash-attention/mmap discipline
#: (`bench-cpu.md:21-22`: *"`-fa 1` always explicit (8-10% swing; llama-bench
#: defaults to 0)"*).
CANONICAL_BENCH_FLAGS: tuple = tuple(canonical_recipe.CANONICAL_BENCH_FLAGS_LLAMA_BENCH)

#: `OMP_PROC_BIND/PLACES/WAIT_POLICY/DYNAMIC` + `GGML_IQK`.
CANONICAL_OMP_ENV: dict = dict(canonical_recipe.CANONICAL_OMP_ENV)

#: clang-20's libomp directory (drift-trap 4: AOCC libomp costs throughput).
LLVM20_LIBDIR: str = canonical_recipe.LLVM20_LIBDIR

#: Parsed out of `architect_bench_gpu_lib.sh`, whose own provenance note records
#: that `88-95` is the SUPERSEDED pinning and `184-191` (node-3 SMT siblings)
#: is correct. The MI210 itself is attached to NUMA node 1; the host-thread
#: placement is therefore deliberately cross-node and must not be described as
#: device-local. Sourcing the CPU list means a correction there reaches here.
MI210_NUMA_NODE = 1
GPU_HOST_THREADS_NUMA_NODE = 3
GPU_HOST_THREADS_ARE_NUMA_LOCAL = MI210_NUMA_NODE == GPU_HOST_THREADS_NUMA_NODE
_GPU_CORES_RE = re.compile(
    r'^CORES="\$\{GPU_BENCH_CORES:-(?P<value>[0-9,\-]+)\}"', re.MULTILINE)

_gpu_host_cpu_list_cache: Optional[str] = None


def gpu_host_cpu_list() -> str:
    """The GPU host-thread pinning, read from the codified GPU launcher.

    Lazy on purpose: a CPU-only campaign must not be blocked at import time by an
    unrelated file. Resolution failure at the point of use is still explicit
    failure — it raises `SourcedConstantUnavailable` and never guesses.
    """
    global _gpu_host_cpu_list_cache
    if _gpu_host_cpu_list_cache is not None:
        # The parsed value is cached; the provenance hash is re-established if it is
        # missing, so a cache hit can never hand back a value with no recorded source.
        if _GPU_LIB_REL_PATH not in _MODULE_HASHES:
            _MODULE_HASHES[_GPU_LIB_REL_PATH] = _sha256_file(GPU_BENCH_LIB_PATH)
        return _gpu_host_cpu_list_cache
    if not GPU_BENCH_LIB_PATH.is_file():
        raise SourcedConstantUnavailable(
            f"the codified GPU launcher is missing: {GPU_BENCH_LIB_PATH}. It is the only "
            f"in-repo home of the MI210 host-thread pinning; retyping the value here "
            f"would re-create the 88-95 vs 184-191 divergence that file's provenance "
            f"note exists to record."
        )
    try:
        text = GPU_BENCH_LIB_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourcedConstantUnavailable(f"cannot read {GPU_BENCH_LIB_PATH}: {exc}") from exc
    matches = _GPU_CORES_RE.findall(text)
    if len(matches) != 1:
        raise SourcedConstantUnavailable(
            f"expected exactly one `CORES=\"${{GPU_BENCH_CORES:-<list>}}\"` assignment in "
            f"{GPU_BENCH_LIB_PATH}, found {len(matches)}. The constant's shape changed; "
            f"this parser fails closed rather than picking one."
        )
    value = matches[0]
    _cpu_list_members(value, field="GPU_BENCH_CORES")  # validates, raises on garbage
    _MODULE_HASHES[_GPU_LIB_REL_PATH] = _sha256_file(GPU_BENCH_LIB_PATH)
    _gpu_host_cpu_list_cache = value
    return value


#: Mirrors `autokernel/resource/device_claim._DEVICE_ID_RE`. Two copies of one
#: boundary is how one of them quietly loses an entry, so `test_recipes.py`
#: imports that module and asserts the two patterns agree.
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

#: ggml quantization / float type names accepted by `test-quantize-perf --type`.
#: An enum rather than a free string: `--type` is caller-supplied and this is the
#: boundary that keeps it from becoming an interpolation surface.
GGML_TYPE_NAMES = (
    "f32", "f16", "bf16",
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1",
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
    "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs",
    "tq1_0", "tq2_0",
)

#: `test-quantize-perf --op` accepts exactly these five (tests/test-quantize-perf.cpp).
QUANTIZE_PERF_OPS = (
    "quantize_row_q_reference", "quantize_row_q", "dequantize_row_q",
    "quantize_row_q_dot", "vec_dot_q",
)

#: Declared type names that the REFERENCE tree cannot measure, and does not say so.
#:
#: `test-quantize-perf` runs a type's op blocks only under
#: `if (qfns_cpu->from_float && qfns->to_float)` (tests/test-quantize-perf.cpp:273).
#: There is no `else`: a type failing that guard is skipped with no line of output,
#: and the tool still **exits 0**. An invocation whose whole `--type` list fails the
#: guard therefore produces an empty stdout indistinguishable from a clean run.
#:
#: Resolved against the frozen production tree `/mnt/raid0/llm/llama.cpp`
#: @67a433bf45a8: `f32`/`q8_1`/`q8_K` have no `to_float` in `ggml.c`'s
#: `type_traits`; `iq1_s`/`iq1_m`/`iq2_xxs`/`iq2_xs` carry `from_float = NULL` and
#: `iq2_s`/`iq3_xxs`/`iq3_s` carry no `from_float` entry at all in `ggml-cpu.c`'s
#: `type_traits_cpu` (consistent with CLAUDE.md's *"IQ1 remains stubbed"*).
#:
#: These are NOT refused, unlike a `--size` the tool rejects outright. "Give this
#: quant type a CPU `from_float`" is a legitimate candidate, and refusing the type
#: would make the instrument unable to measure the change it exists to evaluate.
#: What must not happen silently is the ASYMMETRIC paired block such a candidate
#: creates: the candidate emits rows and the frozen anchor emits none, so the FAIL
#: travels with the record (denial 6 — record the gap, do not patch).
QUANTIZE_PERF_UNMEASURABLE_TYPES = (
    "f32", "q8_1", "q8_K",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s",
)

#: llama-bench output formats. Only `json` and `jsonl` carry `samples_ns` /
#: `samples_ts`; `csv`, `md` and `sql` print `get_fields()`, which stops at
#: `avg_ns/stddev_ns/avg_ts/stddev_ts` (tools/llama-bench/llama-bench.cpp).
#: P-AK-SEARCH-1 requires *"raw samples from which the reduction is
#: reproducible"*, so the format choice is a discipline finding, not a taste.
LLAMA_BENCH_OUTPUT_FORMATS = ("json", "jsonl", "csv", "md", "sql")
LLAMA_BENCH_SAMPLE_BEARING_FORMATS = ("json", "jsonl")

#: `test-backend-ops --output <console|sql|csv>` (tests/test-backend-ops.cpp usage()).
BACKEND_OPS_OUTPUT_FORMATS = ("console", "csv", "sql")

#: Which of those formats actually carry a NUMBER. `csv_printer` filters every
#: row through `get_fields_csv()` — `{op_name, op_params, supported,
#: error_message, test_mode, backend_reg_name, backend_name}`
#: (tests/test-backend-ops.cpp:1091-1100) — which contains none of `time_us`,
#: `flops`, `bandwidth_gb_s`, `n_runs`. `--output csv` therefore emits a
#: well-formed table with the measurement removed, while `sql` prints the full
#: `get_fields()` list (`:589-596`) and `console` prints `print_perf_console`.
#: A recipe whose declared metric is `op_throughput_gflops` cannot be reduced
#: from csv at all, so the format choice is a PASS/FAIL discipline finding
#: exactly as it is for llama-bench, not a taste.
BACKEND_OPS_METRIC_BEARING_FORMATS = ("console", "sql")

#: The ggml device name a masked GPU process sees.
#:
#: `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` do not renumber a device — they
#: REMOVE the others. ggml then names what remains by its index in the VISIBLE
#: set (`dev_ctx->name = GGML_CUDA_NAME + std::to_string(i)`,
#: `ggml/src/ggml-cuda/ggml-cuda.cu:5358`), so a process masked to one device
#: always calls it `ROCm0`, whatever its physical ordinal. The codified GPU
#: launcher pairs `*_VISIBLE_DEVICES=0` with `--device ROCm0` for this reason
#: (`scripts/benchmark/architect_bench_gpu_lib.sh:33-35`).
#:
#: Emitting `ROCm<physical ordinal>` instead is not a loud failure: with
#: `-b ROCm1` `test-backend-ops` skips every enumerated backend, increments
#: `n_ok` for each skip, and **exits 0** (`tests/test-backend-ops.cpp:10366-10371`,
#: `:10413-10417`) — a success-shaped run that measured nothing.
GPU_VISIBLE_DEVICE_NAME = "ROCm0"

#: Bounds. Every one of these is stated in `ConstructedCommand.bounded`; none is
#: applied silently.
MAX_OPS_PER_INVOCATION = 64
MAX_TYPES_PER_INVOCATION = 32
MAX_PARAMS_FILTER_CHARS = 200
MAX_QUANTIZE_ITERATIONS = 100_000_000  # MAX_ITERATIONS, tests/test-quantize-perf.cpp:24
MAX_ARGV_TOKENS = 128


# =============================================================================
# Token and parameter validation — the argv-construction boundary
# =============================================================================

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_GGML_OP_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")
_PARAMS_FILTER_RE = re.compile(r"^[\x20-\x7e]+$")


def _shell_quote(token: str) -> str:
    """POSIX single-quote a token for the human-readable rendering only.

    Equivalent to `shlex.quote`, written out because `shlex` is on this module's
    forbidden-import list — not for safety (argv is a list, so nothing is ever
    handed to a shell) but so the AST audit can keep a single, blunt rule.
    Control characters cannot reach here: every validator rejects them.
    """
    if _SAFE_TOKEN_RE.match(token):
        return token
    return "'" + token.replace("'", "'\"'\"'") + "'"


def _require_str(value: Any, field: str) -> str:
    """`schemas.require.str`, PLUS the rule that is this module's alone.

    Composition, not a copy: the shared predicate ("a non-empty string") comes
    from the field type and raises this module's error; the control-character
    rule stays here because it is about argv and nowhere else.
    """
    schemas.require.str(value, field, error=RecipeParameterError)
    if _CONTROL_CHARS_RE.search(value):
        raise RecipeParameterError(
            f"{field}: contains a control character; an argv token carrying one cannot "
            f"be reproduced from the record"
        )
    return value


def _require_argv_token(value: Any, field: str) -> str:
    """A single argv element. Refuses anything a flag parser could mistake for a flag.

    argv is built as a **list**, so shell metacharacters are inert — but a value
    beginning with `-` shifts the tool's own option parsing, which silently
    changes what was measured. That is the injection this boundary blocks.
    """
    token = _require_str(value, field)
    if token.startswith("-"):
        raise RecipeParameterError(
            f"{field}: {token!r} begins with '-' and would be parsed as an option by the "
            f"measurement tool, changing what was measured"
        )
    return token


def _require_int(value: Any, field: str, *, minimum: Optional[int] = None,
                 maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeParameterError(f"{field}: expected an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise RecipeParameterError(f"{field}: {value} is below the minimum {minimum}")
    if maximum is not None and value > maximum:
        raise RecipeParameterError(f"{field}: {value} is above the maximum {maximum}")
    return value


def _require_abs_path(value: Any, field: str, *, suffix: Optional[str] = None) -> str:
    token = _require_argv_token(value, field)
    path = Path(token)
    if not path.is_absolute():
        raise RecipeParameterError(
            f"{field}: {token!r} is not absolute. A relative path makes the emitted argv "
            f"depend on the runner's working directory, which the record does not carry."
        )
    if ".." in path.parts:
        raise RecipeParameterError(
            f"{field}: {token!r} contains a '..' segment; the recorded path would not "
            f"identify what was measured"
        )
    if suffix is not None and not token.endswith(suffix):
        raise RecipeParameterError(f"{field}: {token!r} does not end with {suffix!r}")
    return token


def _cpu_list_members(spec: Any, *, field: str) -> tuple:
    """Parse `0-95` / `184-191` / `0-3,8-11` into the sorted tuple of cpu ids."""
    text = _require_str(spec, field)
    members: set = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise RecipeParameterError(f"{field}: {text!r} has an empty range element")
        if "-" in part:
            lo_txt, _, hi_txt = part.partition("-")
            if not lo_txt.isdigit() or not hi_txt.isdigit():
                raise RecipeParameterError(f"{field}: {part!r} is not a cpu range")
            lo, hi = int(lo_txt), int(hi_txt)
            if lo > hi:
                raise RecipeParameterError(f"{field}: {part!r} is an inverted range")
        elif part.isdigit():
            lo = hi = int(part)
        else:
            raise RecipeParameterError(f"{field}: {part!r} is not a cpu id or range")
        if hi >= 4096:
            raise RecipeParameterError(f"{field}: cpu id {hi} is implausible (>= 4096)")
        members.update(range(lo, hi + 1))
    if not members:
        raise RecipeParameterError(f"{field}: {text!r} names no cpus")
    return tuple(sorted(members))


# =============================================================================
# Typed parameter declarations
# =============================================================================

@dataclass(frozen=True)
class ParamSpec:
    """One caller-controlled input, with the domain it is validated against."""

    name: str
    kind: str
    doc: str
    required: bool = False
    default: Any = None
    choices: Optional[tuple] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    suffix: Optional[str] = None

    _KINDS = ("int", "path", "enum", "op_list", "type_list", "params_filter", "device_id")

    def __post_init__(self) -> None:
        _require_str(self.name, "param.name")
        if self.kind not in self._KINDS:
            raise ValueError(f"param {self.name}: unknown kind {self.kind!r}")
        if self.kind == "enum" and not self.choices:
            raise ValueError(f"param {self.name}: an enum param needs choices")
        if self.required and self.default is not None:
            raise ValueError(
                f"param {self.name}: a required param must not carry a default — a default "
                f"is how an unstated measurement condition becomes an unrecorded one"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "doc": self.doc,
            "required": self.required, "default": self.default,
            "choices": list(self.choices) if self.choices else None,
            "minimum": self.minimum, "maximum": self.maximum, "suffix": self.suffix,
        }

    def validate(self, value: Any) -> Any:
        field = f"param.{self.name}"
        if self.kind == "int":
            return _require_int(value, field, minimum=self.minimum, maximum=self.maximum)
        if self.kind == "path":
            return _require_abs_path(value, field, suffix=self.suffix)
        if self.kind == "enum":
            token = _require_str(value, field)
            if token not in (self.choices or ()):  # pragma: no branch
                raise RecipeParameterError(
                    f"{field}: {token!r} is not one of {list(self.choices or ())}")
            return token
        if self.kind == "device_id":
            token = _require_argv_token(value, field)
            if not _DEVICE_ID_RE.match(token):
                raise RecipeParameterError(
                    f"{field}: {token!r} does not match the device-claim id pattern "
                    f"{_DEVICE_ID_RE.pattern}; the recipe's device must be the one the "
                    f"exclusive device claim was acquired for (precondition 1)")
            return token
        if self.kind == "op_list":
            return self._validate_list(value, field, _validate_op, MAX_OPS_PER_INVOCATION)
        if self.kind == "type_list":
            return self._validate_list(value, field, _validate_ggml_type,
                                       MAX_TYPES_PER_INVOCATION)
        if self.kind == "params_filter":
            token = _require_argv_token(value, field)
            if len(token) > MAX_PARAMS_FILTER_CHARS:
                raise RecipeParameterError(
                    f"{field}: filter is {len(token)} chars, above the declared bound "
                    f"{MAX_PARAMS_FILTER_CHARS}")
            if not _PARAMS_FILTER_RE.match(token):
                raise RecipeParameterError(
                    f"{field}: filter must be printable ASCII with no control characters")
            return token
        raise AssertionError(f"unhandled param kind {self.kind!r}")  # pragma: no cover

    @staticmethod
    def _validate_list(value: Any, field: str, item_validator, bound: int) -> tuple:
        if not isinstance(value, (list, tuple)):
            raise RecipeParameterError(
                f"{field}: expected a list or tuple of items, got {type(value).__name__}")
        items = tuple(value)
        if not items:
            raise RecipeParameterError(f"{field}: is empty; a filter matching nothing "
                                       f"measures nothing")
        if len(items) > bound:
            raise RecipeParameterError(
                f"{field}: {len(items)} items exceeds the declared bound {bound}. The "
                f"bound is refused, never silently truncated.")
        seen: list = []
        for index, item in enumerate(items):
            token = item_validator(item, f"{field}[{index}]")
            if token in seen:
                raise RecipeParameterError(f"{field}: duplicate entry {token!r}")
            seen.append(token)
        return tuple(seen)


def _validate_op(value: Any, field: str) -> str:
    token = _require_str(value, field)
    if not _GGML_OP_RE.match(token):
        raise RecipeParameterError(
            f"{field}: {token!r} is not a bare ggml op name (pattern "
            f"{_GGML_OP_RE.pattern}). test-backend-ops also accepts a full test-case "
            f"selector string; constructing one is NOT supported in {REGISTRY_ID} and is "
            f"refused rather than passed through unvalidated."
        )
    return token


def _validate_ggml_type(value: Any, field: str) -> str:
    token = _require_str(value, field)
    if token not in GGML_TYPE_NAMES:
        raise RecipeParameterError(
            f"{field}: {token!r} is not one of the declared ggml type names "
            f"{list(GGML_TYPE_NAMES)}")
    return token


# =============================================================================
# Tool binding — the explicit A/B arm identity triple
# =============================================================================

@dataclass(frozen=True)
class ToolBinding:
    """Which build the argv runs, as `canonical_recipe`'s explicit-arm triple.

    All three are required together, exactly as
    `canonical_recipe.build_canonical_bench_command` requires them: *"--binary,
    --source-root, and --library-path must be supplied together"*. The binary's
    directory MUST be the library path, and both must live under the source root
    — the 2026-05-28 defect was an experimental binary resolving production
    libraries through an ambient loader path.
    """

    binary: str
    source_root: str
    library_path: str

    def __post_init__(self) -> None:
        for name in ("binary", "source_root", "library_path"):
            _require_abs_path(getattr(self, name), f"binding.{name}")
        binary = Path(self.binary)
        library_path = Path(self.library_path)
        source_root = Path(self.source_root)
        if binary.parent != library_path:
            raise RecipeBindingError(
                f"binding.library_path must be the binary's own directory:\n"
                f"  binary directory: {binary.parent}\n"
                f"  library path:     {library_path}\n"
                f"Anything else lets the binary resolve someone else's libggml/libllama.")
        for name, path in (("binary", binary), ("library_path", library_path)):
            if source_root != path and source_root not in path.parents:
                raise RecipeBindingError(
                    f"binding.{name} ({path}) is outside binding.source_root "
                    f"({source_root}); the arm's source identity would not cover what ran")

    def to_dict(self) -> dict:
        return {"binary": self.binary, "source_root": self.source_root,
                "library_path": self.library_path}


# =============================================================================
# Derived footprint and scope
# =============================================================================

@dataclass(frozen=True)
class ClaimFootprint:
    """The exact resource footprint the constructed argv pins.

    Precondition 1 requires *"A CPU region claim covering the exact footprint
    measured"*. This is DERIVED from the argv's own `taskset -c` list rather than
    declared next to it, so the claim the runner acquires cannot drift from the
    mask the command applies. `bench_canonical.sh` derives its region-lock cpu
    list the same way and for the same reason.
    """

    cpu_list: str
    cpu_count: int
    devices: tuple
    derived_from: str

    def to_dict(self) -> dict:
        return {"cpu_list": self.cpu_list, "cpu_count": self.cpu_count,
                "devices": list(self.devices), "derived_from": self.derived_from}


def _footprint_from_argv(argv: Sequence[str], devices: Sequence[str]) -> ClaimFootprint:
    argv = list(argv)
    try:
        index = argv.index("taskset")
    except ValueError:
        raise RecipeDriftError(
            "the constructed argv has no `taskset` prefix, so the footprint it pins "
            "cannot be derived and no region claim can be shown to cover it "
            "(P-AK-SEARCH-1 precondition 1)")
    if index + 2 >= len(argv) or argv[index + 1] != "-c":
        raise RecipeDriftError(
            f"`taskset` at argv[{index}] is not followed by `-c <cpu-list>`; refusing to "
            f"guess the pinned footprint")
    cpu_list = argv[index + 2]
    members = _cpu_list_members(cpu_list, field="argv.taskset.cpu_list")
    return ClaimFootprint(
        cpu_list=cpu_list,
        cpu_count=len(members),
        devices=tuple(devices),
        derived_from=f"`taskset -c {cpu_list}` at argv[{index}] of the constructed command",
    )


def _scope_from_footprint(footprint: ClaimFootprint,
                          argv: Sequence[str]) -> api.ScopeDenominator:
    """Full-machine only when the argv pins the canonical mask AND interleaves all nodes.

    `feedback_gate_scope_must_match_measured_subset`: a full-machine gate applied
    to a partial-machine cell is a category error. The denominator is therefore
    derived from what the command actually pins, not from the backend's name.

    NUMA node ids are deliberately NOT synthesised for the partial case: deriving
    them needs a live topology read, which would make the emitted argv a function
    of host state at construct time. The device list plus the core count is what
    this constructor can state truthfully.
    """
    canonical_cpu_list = CANONICAL_PREFIX[CANONICAL_PREFIX.index("-c") + 1]
    interleaves_all = "--interleave=all" in list(argv)
    is_full = footprint.cpu_list == canonical_cpu_list and interleaves_all and not footprint.devices
    if not is_full and not footprint.devices:
        raise RecipeDriftError(
            f"the constructed command pins `taskset -c {footprint.cpu_list}`, which is "
            f"neither the canonical full-machine mask ({canonical_cpu_list} with "
            f"--interleave=all) nor a device cell. A partial CPU cell must name the NUMA "
            f"nodes it measured, and deriving them needs a live topology read that would "
            f"make the emitted argv depend on host state. Refusing rather than declaring "
            f"an unknown denominator.")
    return api.ScopeDenominator(
        machine_subset="full" if is_full else "partial",
        numa_nodes=(),
        devices=footprint.devices,
        cores=footprint.cpu_count,
    )


# =============================================================================
# Discipline findings — where the ratified recipe is and is not satisfiable
# =============================================================================

@dataclass(frozen=True)
class DisciplineFinding:
    """One property of the constructed command, measured against a named clause.

    A `FAIL` here does NOT by itself void a run: *"What voids a run"* enumerates
    twelve conditions and "the tool has no thread flag" is not one of them. What
    it does is travel with the record, so a reader can never be told the canonical
    discipline held when it could not. Denial 6: *"a controller that discovers a
    coverage gap in its evaluator RECORDS the gap … it does not patch the
    instrument."*
    """

    finding_id: str
    check: schemas.Check
    clause: str

    def __post_init__(self) -> None:
        _require_str(self.finding_id, "finding.finding_id")
        if not isinstance(self.check, schemas.Check):
            raise TypeError("finding.check must be a schemas.Check")
        _require_str(self.clause, "finding.clause")

    def to_dict(self) -> dict:
        return {"finding_id": self.finding_id, "outcome": self.check.outcome,
                "reasons": list(self.check.reasons), "clause": self.clause}


def worst_outcome(findings: Sequence[DisciplineFinding]) -> str:
    """`FAIL` > `COULD_NOT_CHECK` > `PASS`. An empty vector is `COULD_NOT_CHECK`.

    An empty discipline vector is NOT a pass: nothing was checked, which is the
    third outcome and never the first. This module already answered the empty
    case correctly; delegating to `schemas.Check.worst_of` keeps it correct by
    construction rather than by a local `if`, and makes the answer the same one
    every other reducer gives.

    Returns the outcome STRING, not a `Check` — this is the `RecipeReceipt`
    summary field and its callers compare it to `schemas.PASS`.
    """
    return schemas.Check.worst_of(f.check for f in findings).outcome


_DELEGATED_LINKAGE = DisciplineFinding(
    finding_id="binary_linkage_resolution",
    check=schemas.Check(schemas.COULD_NOT_CHECK, (
        "the constructor runs no process, so `ldd` linkage resolution is not performed "
        "here; before executing this argv the runner MUST call "
        "canonical_recipe.assert_explicit_bench_identity(binary, source_root, "
        "library_path, env), which is the ratified guard for the 2026-05-28 "
        "RUNPATH-vs-LD_LIBRARY_PATH defect",)),
    clause="bench-cpu.md:8-14 (canonical recipe entry point and linkage guard)",
)

_DELEGATED_HOST_ENV = DisciplineFinding(
    finding_id="host_environment",
    check=schemas.Check(schemas.COULD_NOT_CHECK, (
        "THP / scaling_governor / numa_balancing / perf_event_paranoid are live host "
        "state; reading them here would make the emitted argv a function of when it was "
        "constructed. The runner satisfies P-AK-SEARCH-1 precondition 3 by calling "
        "canonical_recipe.validate_host_environment() inside the measurement window",)),
    clause="P-AK-SEARCH-1 preconditions, item 3 (host-health tier per bench-cpu.md:17-19)",
)

_DELEGATED_GIT_IDENTITY = DisciplineFinding(
    finding_id="worktree_identity",
    check=schemas.Check(schemas.COULD_NOT_CHECK, (
        "`git rev-parse --show-toplevel` is a process launch; the constructor checks only "
        "that source_root exists and contains a `.git` entry. The runner MUST call "
        "canonical_recipe.assert_explicit_bench_identity() for the authoritative check",)),
    clause="bench-cpu.md:38-45 (candidate arm source/binary/linkage identity)",
)


def _tool_cli_finding(tool: str) -> DisciplineFinding:
    return DisciplineFinding(
        finding_id="tool_cli_contract",
        check=schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the flags emitted for {tool} were derived from the production tree's source "
            f"at /mnt/raid0/llm/llama.cpp; whether the CANDIDATE build accepts them cannot "
            f"be established without executing it, which this module does not do. A "
            f"non-zero exit with an unknown-argument message is the runner's signal",)),
        clause="P-AK-SEARCH-1 preconditions, item 6 (codified recipe)",
    )


# =============================================================================
# Recipe specifications and the registry
# =============================================================================

@dataclass(frozen=True)
class RecipeSpec:
    """One registered recipe. `recipe_id` is the stable key a record cites.

    `phase` is the cell's graph phase. For the T1b tiny-real-graph recipes it is a
    property of the recipe (a decode slice measures decode). For the T1a operator
    recipes it is only a placeholder that satisfies the backend's phase vocabulary:
    those recipes declare a REQUIRED `phase` parameter, so the phase in
    `ConstructedCommand.phase` is always the caller's declaration. An operator cell
    has no intrinsic phase — which graph phase a kernel belongs to is a claim about
    the workload, and the calibration block is solved per (backend, phase, cell
    class), so a mis-declared phase scores the cell against a floor it was never
    measured under.
    """

    recipe_id: str
    family: str
    tier: str
    backend: str
    phase: str
    cell_class: str
    tool: str
    metric: str
    metric_direction: str
    params: tuple
    builder: str
    summary: str

    def __post_init__(self) -> None:
        _require_str(self.recipe_id, "recipe.recipe_id")
        if self.family not in RECIPE_FAMILIES:
            raise ValueError(f"{self.recipe_id}: unknown family {self.family!r}")
        api.admit_tier(self.tier)
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"{self.recipe_id}: unknown backend {self.backend!r}")
        allowed_phases = schemas.PHASES_BY_BACKEND.get(self.backend)
        if allowed_phases is not None and self.phase not in allowed_phases:
            raise ValueError(
                f"{self.recipe_id}: phase {self.phase!r} is not one of "
                f"{sorted(allowed_phases)} for backend {self.backend!r}")
        if self.metric_direction not in schemas.METRIC_DIRECTIONS:
            raise ValueError(f"{self.recipe_id}: bad metric_direction "
                             f"{self.metric_direction!r}")
        commensurable = schemas.check_metric_commensurability(
            self.backend, {"metric": self.metric})
        if commensurable.outcome == schemas.FAIL:
            raise ValueError(
                f"{self.recipe_id}: metric {self.metric!r} is not commensurable with "
                f"backend {self.backend!r}: {list(commensurable.reasons)}")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.recipe_id}: duplicate parameter names {names}")

    @property
    def param_map(self) -> dict:
        return {p.name: p for p in self.params}

    def to_dict(self) -> dict:
        return {
            "recipe_id": self.recipe_id, "family": self.family, "tier": self.tier,
            "backend": self.backend, "phase": self.phase, "cell_class": self.cell_class,
            "tool": self.tool, "metric": self.metric,
            "metric_direction": self.metric_direction,
            "params": [p.to_dict() for p in self.params],
            "builder": self.builder, "summary": self.summary,
        }


# --- shared parameter declarations -------------------------------------------

_P_OPS = ParamSpec(
    name="ops", kind="op_list", required=True,
    doc="Bare ggml op names to measure, e.g. ['MUL_MAT', 'MUL_MAT_ID']. §9.3: run "
        "only captured target shapes that occur in the selected real workload.")
_P_PARAMS_FILTER = ParamSpec(
    name="params_filter", kind="params_filter", required=False, default=None,
    doc="test-backend-ops `-p` test-case filter, restricting the op's parameter surface.")
_P_BACKEND_OPS_OUTPUT = ParamSpec(
    name="output_format", kind="enum", default="sql", choices=BACKEND_OPS_OUTPUT_FORMATS,
    doc="test-backend-ops `--output`. `sql` is the default because it is the only "
        "format that is BOTH machine-parseable and metric-bearing: the csv printer "
        "drops time_us/flops/bandwidth_gb_s/n_runs, and `console` keeps them only in "
        "prose. Choosing a non-metric-bearing format is a FAIL discipline finding.")
_P_CACHE_STATE = ParamSpec(
    name="cache_state", kind="enum", default="cold", choices=("warm", "cold"),
    doc="Declared T1a cache state. It is recorded in the recipe parameters and receipt; "
        "candidate and anchor arms must use the same value. The default is explicitly cold, "
        "never unknown.")
_P_MODEL = ParamSpec(
    name="model", kind="path", required=True, suffix=".gguf",
    doc="Absolute path to the production-representative GGUF (§9.4: one model/quant/"
        "shape that actually dispatches the changed path).")
_P_REPS = ParamSpec(
    name="reps", kind="int", required=True, minimum=1, maximum=1000,
    doc="llama-bench `-r`. Required, never defaulted: the rep count is a calibrated "
        "quantity (`B_min`, bench-cpu.md:21-22), not a convenience.")
_P_LB_OUTPUT = ParamSpec(
    name="output_format", kind="enum", default="json", choices=LLAMA_BENCH_OUTPUT_FORMATS,
    doc="llama-bench `-o`. Only json/jsonl carry samples_ns/samples_ts, which the "
        "protocol's raw-sample reproducibility requirement needs.")
_P_DEPTH = ParamSpec(
    name="n_depth", kind="int", required=False, default=None, minimum=0, maximum=1_000_000,
    doc="llama-bench `-d`: pre-existing context depth before the measured slice.")
_P_UBATCH = ParamSpec(
    name="ubatch", kind="int", required=False, default=None, minimum=1, maximum=1_048_576,
    doc="llama-bench `-ub`.")
_P_BATCH = ParamSpec(
    name="batch", kind="int", required=False, default=None, minimum=1, maximum=1_048_576,
    doc="llama-bench `-b`.")
_P_GGML_IQK = ParamSpec(
    name="ggml_iqk", kind="enum", default="1", choices=("0", "1"),
    doc="The GGML_IQK runtime gate. bench-cpu.md:12-13 allows a GGML_* env deviation "
        "ONLY when the variant under test IS an env flag, one flag per arm; setting "
        "'0' records a declared env-flag variant.")
_P_DEVICE_INDEX = ParamSpec(
    name="device_index", kind="int", required=True, minimum=0, maximum=15,
    doc="The ROCm device ordinal. Becomes `-b ROCm<n>` / `-dev ROCm<n>` and the "
        "*_VISIBLE_DEVICES env values.")
_P_DEVICE_ID = ParamSpec(
    name="device_id", kind="device_id", required=True,
    doc="The device-claim id of the SAME device, e.g. 'mi210_0'. Required separately "
        "because the mapping from claim id to ROCm ordinal is not derivable here, and "
        "inventing it would attach a measurement to a device that was never claimed.")
_P_NGL = ParamSpec(
    name="n_gpu_layers", kind="int", required=True, minimum=0, maximum=1024,
    doc="llama-bench `-ngl`. Required: an unstated offload split is an unrecorded "
        "measurement condition.")
_P_GPU_THREADS = ParamSpec(
    name="threads", kind="int", required=False, default=None, minimum=1, maximum=192,
    doc="llama-bench `-t` for the GPU arm's host-side threads. Defaults to the width "
        "of the sourced GPU host-thread mask, so it is always emitted explicitly.")


def _phase_param(choices: tuple) -> ParamSpec:
    return ParamSpec(
        name="phase", kind="enum", required=True, choices=choices,
        doc="Which graph phase's kernels this operator cell belongs to. Declared by the "
            "caller, never inferred: the calibration block is solved per (backend, "
            "phase, cell class) and a mis-declared phase scores a cell against a floor "
            "it was not measured under.")


# =============================================================================
# Environment construction — fully declared, nothing ambient
# =============================================================================

def _ld_library_path(entries: Sequence[str]) -> str:
    seen: list = []
    for entry in entries:
        resolved = str(Path(entry))
        if resolved not in seen:
            seen.append(resolved)
    return ":".join(seen)


def _cpu_env(binding: ToolBinding, *, ggml_iqk: str) -> tuple:
    """The canonical CPU env stack, plus the candidate library path pinned first.

    Returns `(env, deviations)`. `deviations` names every key that differs from
    `CANONICAL_OMP_ENV`; there is at most one, because `bench-cpu.md:12-13` allows
    a GGML_* deviation only when the variant under test IS an env flag, one flag
    per arm.
    """
    env = {
        "LD_LIBRARY_PATH": _ld_library_path([binding.library_path, LLVM20_LIBDIR]),
    }
    env.update(CANONICAL_OMP_ENV)
    deviations: list = []
    if ggml_iqk != CANONICAL_OMP_ENV.get("GGML_IQK"):
        env["GGML_IQK"] = ggml_iqk
        deviations.append("GGML_IQK")
    return env, tuple(deviations)


def _gpu_env(binding: ToolBinding, *, device_index: int, ggml_iqk: str) -> dict:
    """The GPU launch env, declared here with its provenance recorded.

    There is no ratified GPU analogue of `CANONICAL_OMP_ENV`. The shape below is
    the one the in-repo codified GPU launcher uses
    (`scripts/benchmark/architect_bench_gpu_lib.sh`), with one deliberate change:
    the visible-device ordinal is a PARAMETER rather than that file's literal `0`,
    because it must name the device the exclusive claim was acquired for. The
    launcher's `LD_LIBRARY_PATH=$(dirname $BIN)` is followed exactly — clang-20's
    libomp is NOT prepended, because the codified GPU launcher does not prepend it
    and inventing a difference here would be the drift this module prevents.

    THE THREE MASKS COMPOSE; they are not three spellings of one selector.
    `ROCR_VISIBLE_DEVICES` filters at the ROCr/HSA agent level, and HIP then
    enumerates only the agents that survived; `HIP_VISIBLE_DEVICES` (and its
    `CUDA_VISIBLE_DEVICES` alias) indexes into THAT already-filtered list. The
    launcher's literal `0 0 0` is safe only because it selects the first device
    at both levels. Parameterising all three to the same physical ordinal `n`
    selects, for any `n >= 1`, index `n` of a one-element list — no device at
    all. The physical ordinal therefore belongs to the outer mask only; the inner
    masks index the masked set and stay `0`, which is what makes the surviving
    device `GPU_VISIBLE_DEVICE_NAME`. At `device_index=0` this is byte-identical
    to the launcher.
    """
    return {
        "LD_LIBRARY_PATH": _ld_library_path([binding.library_path]),
        "GGML_IQK": ggml_iqk,
        "ROCR_VISIBLE_DEVICES": str(device_index),
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
    }


def _assert_canonical_env(env: Mapping, deviations: Sequence[str]) -> DisciplineFinding:
    """Run the RATIFIED env validator, neutralising only the declared deviations.

    A declared env-flag arm restores that one key to its canonical value for the
    purpose of the check, so every OTHER key is still enforced by
    `canonical_recipe.assert_canonical_env` rather than by a local reimplementation.
    """
    probe = dict(env)
    for key in deviations:
        if key in CANONICAL_OMP_ENV:
            probe[key] = CANONICAL_OMP_ENV[key]
    try:
        canonical_recipe.assert_canonical_env(probe)
    except canonical_recipe.CanonicalRecipeViolation as exc:
        raise RecipeDriftError(
            f"the constructed environment failed the ratified canonical env validator: "
            f"{exc}") from exc
    reasons = ("canonical OMP stack + clang-20 libomp path verified by "
               "canonical_recipe.assert_canonical_env",)
    if deviations:
        reasons = reasons + (
            f"declared env-flag variant under test: {list(deviations)} (bench-cpu.md:12-13 "
            f"allows one GGML_* flag per arm when the variant IS the flag); every other "
            f"canonical key was still enforced",)
    return DisciplineFinding(
        finding_id="canonical_env_stack",
        check=schemas.Check(schemas.PASS, reasons),
        clause="bench-cpu.md:10-14 (core recipe: OMP env stack and libomp resolution)",
    )


_GPU_ENV_FINDING_CLAUSE = "P-AK-SEARCH-1 denials, item 6 (record the gap, do not patch)"


def _gpu_env_finding(sourced_sha: str) -> DisciplineFinding:
    return DisciplineFinding(
        finding_id="canonical_env_stack",
        check=schemas.Check(schemas.COULD_NOT_CHECK, (
            "canonical_recipe.CANONICAL_OMP_ENV is the CPU baseline stack (Annex B, "
            "bench-cpu.md); the project has no ratified GPU env stack to source from, so "
            "the GPU env here is declared in recipes.py with its provenance recorded as "
            f"scripts/benchmark/architect_bench_gpu_lib.sh@{sourced_sha[:12]}",
            "COVERAGE GAP, recorded not patched: closing it means ratifying a GPU "
            "canonical recipe in Annex G, which is a human-only trust-boundary write")),
        clause=_GPU_ENV_FINDING_CLAUSE,
    )


# =============================================================================
# The constructed command
# =============================================================================

@dataclass(frozen=True)
class ConstructedCommand:
    """One codified measurement command. Constructed, validated, never executed."""

    recipe_id: str
    registry_id: str
    tier: str
    backend: str
    phase: str
    cell_class: str
    metric: str
    metric_direction: str
    arm: str
    tool: str
    argv: tuple
    env: dict
    binding: ToolBinding
    params: dict
    claim_footprint: ClaimFootprint
    scope_denominator: api.ScopeDenominator
    receipt: api.RecipeReceipt
    sourced_constants: tuple
    discipline: tuple
    input_checks: tuple
    raw_samples_source: str
    bounded: tuple

    @property
    def discipline_outcome(self) -> str:
        return worst_outcome(self.discipline)

    @property
    def inputs_verified(self) -> bool:
        """True only when every input check PASSed. COULD_NOT_CHECK is not a pass."""
        return bool(self.input_checks) and all(
            c.outcome == schemas.PASS for c in self.input_checks)

    def finding(self, finding_id: str) -> DisciplineFinding:
        for item in self.discipline:
            if item.finding_id == finding_id:
                return item
        raise KeyError(f"{self.recipe_id}: no discipline finding {finding_id!r}; "
                       f"present: {[f.finding_id for f in self.discipline]}")

    def render_human_readable(self) -> str:
        """A readable rendering of env + argv, for logs and operator review.

        **This string is for READING, not for pasting.** The execution path
        consumes `argv` as a list; a human who copies this line back into a shell
        has hand-typed the command, which is precondition 6's void condition. It
        is emitted anyway because an unreadable command cannot be reviewed, and
        `bench_canonical.sh` prints the same thing for the same reason.
        """
        parts = [f"{k}={_shell_quote(v)}" for k, v in sorted(self.env.items())]
        parts += [_shell_quote(token) for token in self.argv]
        return " ".join(parts)

    def to_dict(self) -> dict:
        """Canonical-JSON-able. Lists, never tuples (schemas.canonical_json refuses tuples)."""
        return {
            "recipe_id": self.recipe_id,
            "registry_id": self.registry_id,
            "tier": self.tier,
            "backend": self.backend,
            "phase": self.phase,
            "cell_class": self.cell_class,
            "metric": self.metric,
            "metric_direction": self.metric_direction,
            "arm": self.arm,
            "tool": self.tool,
            "argv": list(self.argv),
            "env": dict(self.env),
            "binding": self.binding.to_dict(),
            "params": _jsonable(self.params),
            "claim_footprint": self.claim_footprint.to_dict(),
            "scope_denominator": self.scope_denominator.to_dict(),
            "recipe": self.receipt.to_dict(),
            "recipe_render": self.receipt.render(),
            "sourced_constants": [dict(item) for item in self.sourced_constants],
            "discipline": [f.to_dict() for f in self.discipline],
            "discipline_outcome": self.discipline_outcome,
            "input_checks": [{"outcome": c.outcome, "reasons": list(c.reasons)}
                             for c in self.input_checks],
            "inputs_verified": self.inputs_verified,
            "raw_samples_source": self.raw_samples_source,
            "bounded": list(self.bounded),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# =============================================================================
# Builders — one per registered recipe
# =============================================================================

def _cpu_prefix() -> list:
    return list(CANONICAL_PREFIX)


def _gpu_prefix() -> list:
    """`taskset -c <gpu host cores>` — no `numactl --interleave=all`.

    The GPU arm's host threads are pinned to the MI210's node-local SMT siblings
    (`architect_bench_gpu_lib.sh`); striping them across all four nodes would move
    them off the device's node, which is the correction that file's provenance
    note records.
    """
    return ["taskset", "-c", gpu_host_cpu_list()]


def _assert_canonical_prefix(argv: Sequence[str]) -> DisciplineFinding:
    # Compared against `canonical_recipe`'s LIVE constant, not this module's
    # import-time snapshot: the snapshot is what the builder emits, the live value
    # is what is ratified, and the whole point is that a divergence between the two
    # is a refusal rather than a quietly different measurement.
    ratified = list(canonical_recipe.CANONICAL_PREFIX)
    head = list(argv)[:len(ratified)]
    if head != ratified:
        raise RecipeDriftError(
            f"constructed argv does not start with the ratified canonical prefix.\n"
            f"  expected: {ratified}\n  got:      {head}\n"
            f"Fix: route through canonical_recipe.CANONICAL_PREFIX; do not retype it.")
    return DisciplineFinding(
        finding_id="canonical_prefix",
        check=schemas.Check(schemas.PASS, (
            f"argv starts with canonical_recipe.CANONICAL_PREFIX {list(CANONICAL_PREFIX)} "
            f"(taskset before numactl, --interleave=all)",)),
        clause="bench-cpu.md:10-11 (core recipe: taskset/NUMA policy)",
    )


def _assert_canonical_bench_cmd(argv: Sequence[str]) -> tuple:
    """Full `assert_canonical_cmd` — prefix AND the mmap guard — for llama-bench."""
    try:
        canonical_recipe.assert_canonical_cmd(list(argv))
    except canonical_recipe.CanonicalRecipeViolation as exc:
        raise RecipeDriftError(
            f"the constructed llama-bench argv failed the ratified canonical cmd "
            f"validator: {exc}") from exc
    findings = [
        DisciplineFinding(
            finding_id="canonical_prefix",
            check=schemas.Check(schemas.PASS, (
                "verified by canonical_recipe.assert_canonical_cmd",)),
            clause="bench-cpu.md:10-11 (core recipe: taskset/NUMA policy)"),
        DisciplineFinding(
            finding_id="mmap_disabled",
            check=schemas.Check(schemas.PASS, (
                "`-mmp 0` present; mmap=ON pulls weights through file-cache first-touch "
                "and defeats --interleave=all striping on EPYC",)),
            clause="bench-cpu.md:10-11 / canonical_recipe.assert_canonical_cmd"),
    ]
    argv_list = list(argv)
    for flag, finding_id, clause in (
            ("-t", "explicit_threads", "bench-cpu.md:10 (core recipe: `-t 96` explicit)"),
            ("-fa", "explicit_flash_attention",
             "bench-cpu.md:21-22 (`-fa 1` always explicit; 8-10% swing)")):
        if flag in argv_list and argv_list.index(flag) + 1 < len(argv_list):
            value = argv_list[argv_list.index(flag) + 1]
            findings.append(DisciplineFinding(
                finding_id=finding_id,
                check=schemas.Check(schemas.PASS, (f"`{flag} {value}` is explicit in argv",)),
                clause=clause))
        else:  # pragma: no cover - unreachable while CANONICAL_BENCH_FLAGS carries both
            findings.append(DisciplineFinding(
                finding_id=finding_id,
                check=schemas.Check(schemas.FAIL, (f"`{flag}` is absent from the argv",)),
                clause=clause))
    return tuple(findings)


_NO_THREAD_FLAG_FINDING = DisciplineFinding(
    finding_id="explicit_threads",
    check=schemas.Check(schemas.FAIL, (
        "test-backend-ops exposes no thread-count flag: it calls "
        "ggml_backend_set_n_threads with std::thread::hardware_concurrency() "
        "(tests/test-backend-ops.cpp:10387, macro at :51), which libstdc++ derives from "
        "_SC_NPROCESSORS_ONLN and NOT from the taskset affinity mask",
        "on this host that is 192 threads regardless of `taskset -c 0-95` (96 logical "
        "cpus in the mask), i.e. 2x oversubscription inside the pinned footprint",
        "the canonical `-t` discipline is therefore UNSATISFIABLE for this tool; the "
        "comparison stays valid because the calibration block runs under the identical "
        "recipe, but no record may claim canonical thread discipline",
        "COVERAGE GAP, recorded not patched (P-AK-SEARCH-1 denial 6): closing it needs a "
        "thread-count flag in the tool, which is a source change and a normal candidate")),
    clause="bench-cpu.md:10 (core recipe: `-t 96` explicit)",
)


def _build_backend_ops(*, spec: RecipeSpec, binding: ToolBinding, params: Mapping,
                       gpu: bool) -> tuple:
    """`test-backend-ops perf` — the T1a operator discriminator (§9.3)."""
    bounded: list = [
        f"op filter accepts at most {MAX_OPS_PER_INVOCATION} bare ggml op names per "
        f"invocation; a longer list is REFUSED, never truncated",
        f"`-p` test-case filter is bounded at {MAX_PARAMS_FILTER_CHARS} printable ASCII "
        f"characters",
        "full test-case selector strings (e.g. 'ADD(type=f16,ne=[...])') are NOT "
        f"constructible in {REGISTRY_ID}; only bare op names are",
    ]
    if gpu:
        device_index = params["device_index"]
        prefix = _gpu_prefix()
        # The masked process sees ONE device and ggml calls it ROCm0 regardless of
        # its physical ordinal; `-b ROCm<physical>` would skip every backend and
        # still exit 0. See GPU_VISIBLE_DEVICE_NAME.
        backend_filter = GPU_VISIBLE_DEVICE_NAME
        env = _gpu_env(binding, device_index=device_index,
                       ggml_iqk=params.get("ggml_iqk", "1"))
        devices = (params["device_id"],)
        bounded.append(
            f"the physical device ordinal is carried by ROCR_VISIBLE_DEVICES="
            f"{device_index}, not by the argv device name: after masking, ggml names "
            f"the single visible device {GPU_VISIBLE_DEVICE_NAME}")
    else:
        prefix = _cpu_prefix()
        backend_filter = "CPU"
        env, deviations = _cpu_env(binding, ggml_iqk=params.get("ggml_iqk", "1"))
        devices = ()

    argv = prefix + [binding.binary, "perf", "-o", ",".join(params["ops"]),
                     "-b", backend_filter]
    if params.get("params_filter") is not None:
        argv += ["-p", params["params_filter"]]
    argv += ["--output", params["output_format"]]

    findings: list = []
    if gpu:
        findings.append(DisciplineFinding(
            finding_id="canonical_prefix",
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the GPU arm pins `taskset -c {prefix[2]}` (the MI210's node-local SMT "
                f"siblings, sourced from architect_bench_gpu_lib.sh) and deliberately "
                f"does NOT apply `numactl --interleave=all`, which would move the host "
                f"threads off the device's node; the ratified canonical prefix is the CPU "
                f"baseline and does not describe this cell",)),
            clause=_GPU_ENV_FINDING_CLAUSE))
        findings.append(_gpu_env_finding(
            _MODULE_HASHES[_GPU_LIB_REL_PATH]))
    else:
        findings.append(_assert_canonical_prefix(argv))
        findings.append(_assert_canonical_env(env, deviations))
    findings.append(_NO_THREAD_FLAG_FINDING)
    fmt = params["output_format"]
    if fmt in BACKEND_OPS_METRIC_BEARING_FORMATS:
        samples_check = schemas.Check(schemas.COULD_NOT_CHECK, (
            f"`--output {fmt}` carries time_us/flops/bandwidth_gb_s/n_runs, but "
            f"test-backend-ops emits one aggregate figure per op per invocation and no "
            f"per-repetition sample vector, so one invocation is one paired-block "
            f"sample; the raw evidence is the CAPTURED STDOUT of each invocation and "
            f"the constructor cannot verify the runner persists it",
            "P-AK-SEARCH-1 'Record grammar' requires raw=<raw_samples_ref>, and 'a record "
            "whose reduction cannot be recomputed from its raw samples is INVALID'"))
        raw = ("captured stdout of each invocation; one invocation = one paired-block "
               "sample (test-backend-ops emits no per-repetition sample vector)")
    else:
        samples_check = schemas.Check(schemas.FAIL, (
            f"`--output {fmt}` filters every row through get_fields_csv() = "
            f"{{op_name, op_params, supported, error_message, test_mode, "
            f"backend_reg_name, backend_name}} (tests/test-backend-ops.cpp:1091-1100), "
            f"which contains NONE of time_us/flops/bandwidth_gb_s/n_runs",
            f"the recipe's declared metric is {spec.metric!r}; this output carries no "
            f"number it could be reduced from, so the run would emit a well-formed "
            f"table with the measurement removed",
            f"use one of {list(BACKEND_OPS_METRIC_BEARING_FORMATS)}"))
        raw = f"NONE — `--output {fmt}` carries no timing or throughput field"
    findings.append(DisciplineFinding(
        finding_id="raw_samples_retained", check=samples_check,
        clause="P-AK-SEARCH-1 search-grade conjunction (raw samples reproducible)"))
    findings.append(_tool_cli_finding("test-backend-ops"))
    findings.append(_DELEGATED_LINKAGE)
    findings.append(_DELEGATED_GIT_IDENTITY)
    findings.append(_DELEGATED_HOST_ENV)

    return argv, env, tuple(findings), raw, tuple(bounded), devices


def _build_quantize_perf(*, spec: RecipeSpec, binding: ToolBinding,
                         params: Mapping) -> tuple:
    """`test-quantize-perf` — the T1a quantization-kernel discriminator (§9.5, arithmetic)."""
    size = params.get("size_elements")
    if size is not None and size % 32 != 0:
        raise RecipeParameterError(
            f"param.size_elements: {size} is not divisible by 32; test-quantize-perf "
            f"refuses it at tests/test-quantize-perf.cpp:156. Refused here so the failure "
            f"is a construction error rather than a burnt measurement window.")
    env, deviations = _cpu_env(binding, ggml_iqk=params.get("ggml_iqk", "1"))
    argv = _cpu_prefix() + [binding.binary, "--op", params["op"]]
    for type_name in params["types"]:
        argv += ["--type", type_name]
    argv += ["-i", str(params["iterations"])]
    if size is not None:
        argv += ["--size", str(size)]
    if params.get("alignment_offset") is not None:
        argv += ["--alignment-offset", str(params["alignment_offset"])]

    unmeasurable = tuple(t for t in params["types"]
                         if t in QUANTIZE_PERF_UNMEASURABLE_TYPES)
    if unmeasurable:
        types_check = schemas.Check(schemas.FAIL, (
            f"{list(unmeasurable)} fail test-quantize-perf's "
            f"`qfns_cpu->from_float && qfns->to_float` guard in the reference tree "
            f"(tests/test-quantize-perf.cpp:273); the tool prints NO line for such a "
            f"type and still exits 0",
            f"{'every' if len(unmeasurable) == len(params['types']) else 'part'} of "
            f"this `--type` list is affected, so the invocation's stdout "
            f"{'is empty and looks like a clean run' if len(unmeasurable) == len(params['types']) else 'silently omits those types'}",
            "if only the CANDIDATE build supplies the missing trait the paired block "
            "is ASYMMETRIC — candidate rows against no anchor rows — which "
            "P-AK-SEARCH-1 precondition 4 makes INVALID, not favourable",
            f"unaffected declared types: "
            f"{[t for t in GGML_TYPE_NAMES if t not in QUANTIZE_PERF_UNMEASURABLE_TYPES]}"))
    else:
        types_check = schemas.Check(schemas.PASS, (
            f"every requested type {list(params['types'])} passes the reference tree's "
            f"from_float/to_float guard, so each produces output",))
    findings = [
        _assert_canonical_prefix(argv),
        _assert_canonical_env(env, deviations),
        DisciplineFinding(
            finding_id="measurable_types", check=types_check,
            clause="P-AK-SEARCH-1 search-grade conjunction (raw samples reproducible) "
                   "and preconditions item 4 (explicit immutable anchor)"),
        DisciplineFinding(
            finding_id="explicit_threads",
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                "test-quantize-perf exposes no thread-count flag; the production tree's "
                "tests/test-quantize-perf.cpp contains no threading at all, but whether "
                "the CANDIDATE build is single-threaded cannot be established without "
                "executing it",)),
            clause="bench-cpu.md:10 (core recipe: `-t 96` explicit)"),
        DisciplineFinding(
            finding_id="raw_samples_retained",
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                "test-quantize-perf prints one GB/s figure per (op, type, size); one "
                "invocation is one paired-block sample and the raw evidence is the "
                "captured stdout, which the constructor cannot verify is persisted",)),
            clause="P-AK-SEARCH-1 record grammar (raw samples reference)"),
        _tool_cli_finding("test-quantize-perf"),
        _DELEGATED_LINKAGE,
        _DELEGATED_GIT_IDENTITY,
        _DELEGATED_HOST_ENV,
    ]
    bounded = (
        f"`--type` accepts at most {MAX_TYPES_PER_INVOCATION} names from the declared "
        f"ggml type enum; anything else is REFUSED, never dropped",
        f"`-i` is bounded at {MAX_QUANTIZE_ITERATIONS} (MAX_ITERATIONS, "
        f"tests/test-quantize-perf.cpp:24)",
        "`--size` must be divisible by 32 (the tool rejects anything else at :156); "
        "refused here rather than discovered after the window opened",
        "the tool's `-3` / `-4` cache-size quick-selects are NOT constructible: they "
        "expand to several sizes in one invocation, which is several cells in one record",
        f"{list(QUANTIZE_PERF_UNMEASURABLE_TYPES)} are constructible but produce NO "
        f"output in the reference tree (from_float/to_float guard, :273) and the tool "
        f"still exits 0; requesting one is a FAIL discipline finding, not a refusal, "
        f"because supplying the missing trait is itself a legitimate candidate",
    )
    raw = ("captured stdout of each invocation; one invocation = one paired-block sample "
           "(test-quantize-perf emits no per-repetition sample vector)")
    return argv, env, tuple(findings), raw, bounded, ()


def _build_llama_bench(*, spec: RecipeSpec, binding: ToolBinding, params: Mapping,
                       n_prompt: int, n_gen: int, gpu: bool) -> tuple:
    """`llama-bench` — the T1b tiny real-graph translation (§9.4).

    The CPU form is the ratified canonical recipe verbatim: `CANONICAL_PREFIX` +
    `CANONICAL_BENCH_FLAGS_LLAMA_BENCH` (`-t 96 -fa 1 -mmp 0`), with the model,
    the bounded prompt/decode slice, the rep count, and the output format appended
    in the same order `canonical_recipe.build_canonical_bench_command` uses.
    """
    bounded: list = []
    if gpu:
        device_index = params["device_index"]
        prefix = _gpu_prefix()
        threads = params.get("threads")
        if threads is None:
            threads = len(_cpu_list_members(prefix[2], field="gpu_host_cpu_list"))
            bounded.append(
                f"`-t {threads}` defaults to the width of the sourced GPU host-thread "
                f"mask ({prefix[2]}) so the thread count is always explicit in argv")
        bench_flags = ["-t", str(threads), "-fa", "1", "-mmp", "0",
                       "-ngl", str(params["n_gpu_layers"]),
                       # ggml names the single visible device by its index in the
                       # MASKED set, so this is ROCm0 for every physical ordinal.
                       # `-dev ROCm1` under a one-device mask raises
                       # "invalid device: ROCm1" (tools/llama-bench/llama-bench.cpp:166).
                       "-dev", GPU_VISIBLE_DEVICE_NAME]
        env = _gpu_env(binding, device_index=device_index,
                       ggml_iqk=params.get("ggml_iqk", "1"))
        devices = (params["device_id"],)
        bounded.append(
            f"the physical device ordinal is carried by ROCR_VISIBLE_DEVICES="
            f"{device_index}, not by `-dev`: after masking, ggml names the single "
            f"visible device {GPU_VISIBLE_DEVICE_NAME}")
    else:
        prefix = _cpu_prefix()
        bench_flags = list(CANONICAL_BENCH_FLAGS)
        env, deviations = _cpu_env(binding, ggml_iqk=params.get("ggml_iqk", "1"))
        devices = ()

    argv = prefix + [binding.binary] + bench_flags + [
        "-m", params["model"],
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(params["reps"]),
    ]
    for flag, key in (("-d", "n_depth"), ("-ub", "ubatch"), ("-b", "batch")):
        if params.get(key) is not None:
            argv += [flag, str(params[key])]
    argv += ["-o", params["output_format"]]

    findings: list = []
    if gpu:
        findings.append(DisciplineFinding(
            finding_id="canonical_prefix",
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the GPU arm pins `taskset -c {prefix[2]}` and deliberately omits "
                f"`numactl --interleave=all`; the ratified canonical prefix is the CPU "
                f"baseline and does not describe this cell",)),
            clause=_GPU_ENV_FINDING_CLAUSE))
        findings.append(_gpu_env_finding(
            _MODULE_HASHES[_GPU_LIB_REL_PATH]))
        for flag, finding_id, clause in (
                ("-t", "explicit_threads", "bench-cpu.md:10 (`-t` explicit)"),
                ("-fa", "explicit_flash_attention", "bench-cpu.md:21-22 (`-fa 1` explicit)"),
                ("-mmp", "mmap_disabled", "bench-cpu.md:10-11 (mmap defeats NUMA interleave)")):
            index = argv.index(flag)
            findings.append(DisciplineFinding(
                finding_id=finding_id,
                check=schemas.Check(schemas.PASS,
                                    (f"`{flag} {argv[index + 1]}` is explicit in argv",)),
                clause=clause))
    else:
        findings.extend(_assert_canonical_bench_cmd(argv))
        findings.append(_assert_canonical_env(env, deviations))

    fmt = params["output_format"]
    if fmt in LLAMA_BENCH_SAMPLE_BEARING_FORMATS:
        samples_check = schemas.Check(schemas.PASS, (
            f"`-o {fmt}` emits samples_ns and samples_ts, one entry per repetition "
            f"(tools/llama-bench/llama-bench.cpp json/jsonl printers)",))
        raw = f"per-repetition samples_ns/samples_ts inside the `-o {fmt}` output"
    else:
        samples_check = schemas.Check(schemas.FAIL, (
            f"`-o {fmt}` prints get_fields(), which stops at avg_ns/stddev_ns/avg_ts/"
            f"stddev_ts and carries NO per-repetition samples "
            f"(tools/llama-bench/llama-bench.cpp)",
            "P-AK-SEARCH-1 'Search-grade requires ALL of' includes 'raw samples from "
            "which the reduction is reproducible'; a record reduced from this output "
            "cannot satisfy it",
            f"use one of {list(LLAMA_BENCH_SAMPLE_BEARING_FORMATS)}"))
        raw = f"NONE — `-o {fmt}` carries no per-repetition samples"
    findings.append(DisciplineFinding(
        finding_id="raw_samples_retained", check=samples_check,
        clause="P-AK-SEARCH-1 search-grade conjunction (raw samples reproducible)"))
    findings.append(_tool_cli_finding("llama-bench"))
    findings.append(_DELEGATED_LINKAGE)
    findings.append(_DELEGATED_GIT_IDENTITY)
    findings.append(_DELEGATED_HOST_ENV)

    bounded.append(
        "the measured slice is exactly one (n_prompt, n_gen) point per invocation; "
        "llama-bench's multi-value sweep syntax is not constructible here, because a "
        "sweep row is a different cell and would share one record")
    return argv, env, tuple(findings), raw, tuple(bounded), devices


# --- builder dispatch ---------------------------------------------------------

def _builder_backend_ops_cpu(spec, binding, params):
    return _build_backend_ops(spec=spec, binding=binding, params=params, gpu=False)


def _builder_backend_ops_gpu(spec, binding, params):
    return _build_backend_ops(spec=spec, binding=binding, params=params, gpu=True)


def _builder_quantize_perf(spec, binding, params):
    return _build_quantize_perf(spec=spec, binding=binding, params=params)


def _builder_llama_bench_decode_cpu(spec, binding, params):
    return _build_llama_bench(spec=spec, binding=binding, params=params,
                              n_prompt=0, n_gen=params["n_gen"], gpu=False)


def _builder_llama_bench_prefill_cpu(spec, binding, params):
    return _build_llama_bench(spec=spec, binding=binding, params=params,
                              n_prompt=params["n_prompt"], n_gen=0, gpu=False)


def _builder_llama_bench_decode_gpu(spec, binding, params):
    return _build_llama_bench(spec=spec, binding=binding, params=params,
                              n_prompt=0, n_gen=params["n_gen"], gpu=True)


def _builder_llama_bench_prefill_gpu(spec, binding, params):
    return _build_llama_bench(spec=spec, binding=binding, params=params,
                              n_prompt=params["n_prompt"], n_gen=0, gpu=True)


_BUILDERS = {
    "_builder_backend_ops_cpu": _builder_backend_ops_cpu,
    "_builder_backend_ops_gpu": _builder_backend_ops_gpu,
    "_builder_quantize_perf": _builder_quantize_perf,
    "_builder_llama_bench_decode_cpu": _builder_llama_bench_decode_cpu,
    "_builder_llama_bench_prefill_cpu": _builder_llama_bench_prefill_cpu,
    "_builder_llama_bench_decode_gpu": _builder_llama_bench_decode_gpu,
    "_builder_llama_bench_prefill_gpu": _builder_llama_bench_prefill_gpu,
}


# =============================================================================
# The registry
# =============================================================================

_P_N_GEN = ParamSpec(
    name="n_gen", kind="int", required=True, minimum=1, maximum=65536,
    doc="llama-bench `-n`: the bounded decode slice (§9.4, 'a bounded prompt/decode or "
        "prefill slice').")
_P_N_PROMPT = ParamSpec(
    name="n_prompt", kind="int", required=True, minimum=1, maximum=1_048_576,
    doc="llama-bench `-p`: the bounded prefill slice.")

_SPECS = (
    RecipeSpec(
        recipe_id="t1a.llama_cpu.backend_ops_perf.v1",
        family=RECIPE_FAMILY_T1A, tier="T1a", backend="llama_cpu", phase="decode",
        cell_class=CELL_CLASS_OPERATOR, tool="test-backend-ops",
        metric="op_throughput_gflops", metric_direction="higher_better",
        params=(_phase_param(("prefill", "decode")), _P_OPS, _P_PARAMS_FILTER,
                _P_BACKEND_OPS_OUTPUT, _P_CACHE_STATE, _P_GGML_IQK),
        builder="_builder_backend_ops_cpu",
        summary="CPU target-operator discriminator: test-backend-ops perf under the "
                "canonical taskset/NUMA/OMP baseline."),
    RecipeSpec(
        recipe_id="t1a.llama_gpu.backend_ops_perf.v1",
        family=RECIPE_FAMILY_T1A, tier="T1a", backend="llama_gpu", phase="decode",
        cell_class=CELL_CLASS_OPERATOR, tool="test-backend-ops",
        metric="op_throughput_gflops", metric_direction="higher_better",
        params=(_phase_param(("prefill", "decode")), _P_OPS, _P_PARAMS_FILTER,
                _P_BACKEND_OPS_OUTPUT, _P_CACHE_STATE, _P_GGML_IQK,
                _P_DEVICE_INDEX, _P_DEVICE_ID),
        builder="_builder_backend_ops_gpu",
        summary="MI210 target-operator discriminator: test-backend-ops perf on ROCm<n>, "
                "host threads pinned to the device's node-local SMT siblings."),
    RecipeSpec(
        recipe_id="t1a.llama_cpu.quantize_perf.v1",
        family=RECIPE_FAMILY_T1A, tier="T1a", backend="llama_cpu", phase="decode",
        cell_class=CELL_CLASS_OPERATOR, tool="test-quantize-perf",
        metric="quant_kernel_throughput_gbps", metric_direction="higher_better",
        params=(
            _phase_param(("prefill", "decode")),
            ParamSpec(name="op", kind="enum", required=True, choices=QUANTIZE_PERF_OPS,
                      doc="test-quantize-perf `--op`."),
            ParamSpec(name="types", kind="type_list", required=True,
                      doc="test-quantize-perf `--type`, repeated once per name."),
            ParamSpec(name="iterations", kind="int", required=True, minimum=1,
                      maximum=MAX_QUANTIZE_ITERATIONS,
                      doc="test-quantize-perf `-i`."),
            ParamSpec(name="size_elements", kind="int", required=False, default=None,
                      minimum=32, maximum=1 << 34,
                      doc="test-quantize-perf `--size`, in elements; must be divisible "
                          "by 32 or the tool refuses it."),
            ParamSpec(name="alignment_offset", kind="int", required=False, default=None,
                      minimum=0, maximum=64,
                      doc="test-quantize-perf `--alignment-offset` (MAX_ALIGNMENT=64)."),
            _P_CACHE_STATE, _P_GGML_IQK),
        builder="_builder_quantize_perf",
        summary="CPU quantization-kernel discriminator for arithmetic/layout change "
                "classes: test-quantize-perf under the canonical baseline."),
    RecipeSpec(
        recipe_id="t1b.llama_cpu.llama_bench_decode.v1",
        family=RECIPE_FAMILY_T1B, tier="T1b", backend="llama_cpu", phase="decode",
        cell_class=CELL_CLASS_TINY_GRAPH, tool="llama-bench",
        metric="decode_tokens_per_s", metric_direction="higher_better",
        params=(_P_MODEL, _P_N_GEN, _P_REPS, _P_LB_OUTPUT, _P_DEPTH, _P_UBATCH, _P_BATCH,
                _P_GGML_IQK),
        builder="_builder_llama_bench_decode_cpu",
        summary="Tiny real-graph decode slice on CPU: the ratified canonical llama-bench "
                "recipe with a bounded -n and an explicit rep count."),
    RecipeSpec(
        recipe_id="t1b.llama_cpu.llama_bench_prefill.v1",
        family=RECIPE_FAMILY_T1B, tier="T1b", backend="llama_cpu", phase="prefill",
        cell_class=CELL_CLASS_TINY_GRAPH, tool="llama-bench",
        metric="prefill_tokens_per_s", metric_direction="higher_better",
        params=(_P_MODEL, _P_N_PROMPT, _P_REPS, _P_LB_OUTPUT, _P_DEPTH, _P_UBATCH,
                _P_BATCH, _P_GGML_IQK),
        builder="_builder_llama_bench_prefill_cpu",
        summary="Tiny real-graph prefill slice on CPU: the ratified canonical llama-bench "
                "recipe with a bounded -p and an explicit rep count."),
    RecipeSpec(
        recipe_id="t1b.llama_gpu.llama_bench_decode.v1",
        family=RECIPE_FAMILY_T1B, tier="T1b", backend="llama_gpu", phase="decode",
        cell_class=CELL_CLASS_TINY_GRAPH, tool="llama-bench",
        metric="decode_tokens_per_s", metric_direction="higher_better",
        params=(_P_MODEL, _P_N_GEN, _P_REPS, _P_LB_OUTPUT, _P_DEPTH, _P_UBATCH, _P_BATCH,
                _P_GGML_IQK, _P_DEVICE_INDEX, _P_DEVICE_ID, _P_NGL, _P_GPU_THREADS),
        builder="_builder_llama_bench_decode_gpu",
        summary="Tiny real-graph decode slice on MI210: explicit -t/-fa/-mmp/-ngl/-dev "
                "with host threads on the device's node-local SMT siblings."),
    RecipeSpec(
        recipe_id="t1b.llama_gpu.llama_bench_prefill.v1",
        family=RECIPE_FAMILY_T1B, tier="T1b", backend="llama_gpu", phase="prefill",
        cell_class=CELL_CLASS_TINY_GRAPH, tool="llama-bench",
        metric="prefill_tokens_per_s", metric_direction="higher_better",
        params=(_P_MODEL, _P_N_PROMPT, _P_REPS, _P_LB_OUTPUT, _P_DEPTH, _P_UBATCH,
                _P_BATCH, _P_GGML_IQK, _P_DEVICE_INDEX, _P_DEVICE_ID, _P_NGL,
                _P_GPU_THREADS),
        builder="_builder_llama_bench_prefill_gpu",
        summary="Tiny real-graph prefill slice on MI210."),
)

REGISTRY: dict = {}
for _spec in _SPECS:
    if _spec.recipe_id in REGISTRY:
        raise ValueError(f"duplicate recipe_id {_spec.recipe_id!r}")
    if _spec.builder not in _BUILDERS:
        raise ValueError(f"{_spec.recipe_id}: builder {_spec.builder!r} is not defined")
    REGISTRY[_spec.recipe_id] = _spec
del _spec

RECIPE_IDS: tuple = tuple(sorted(REGISTRY))


def list_recipes(*, backend: Optional[str] = None, tier: Optional[str] = None,
                 family: Optional[str] = None) -> tuple:
    """Registered recipe ids, optionally filtered. Sorted, so callers are stable."""
    out = []
    for recipe_id in RECIPE_IDS:
        spec = REGISTRY[recipe_id]
        if backend is not None and spec.backend != backend:
            continue
        if tier is not None and spec.tier != tier:
            continue
        if family is not None and spec.family != family:
            continue
        out.append(recipe_id)
    return tuple(out)


def get_recipe(recipe_id: str) -> RecipeSpec:
    """The spec for `recipe_id`, or `UnregisteredRecipe`.

    Precondition 6 admits exactly one source of argv; an id that resolves to
    nothing is a hand-typed command with a citation attached.
    """
    if not isinstance(recipe_id, str):
        raise TypeError(f"recipe_id must be a string, got {type(recipe_id).__name__}")
    try:
        return REGISTRY[recipe_id]
    except KeyError:
        raise UnregisteredRecipe(
            f"recipe_id {recipe_id!r} is not registered in {REGISTRY_ID}. "
            f"P-AK-SEARCH-1 precondition 6 requires every measurement command line to be "
            f"emitted by a recipe constructor and 'hand-typed argv voids the run', so an "
            f"unregistered id is refused rather than constructed. Registered ids: "
            f"{list(RECIPE_IDS)}"
        ) from None


# =============================================================================
# Input verification — read-only, and never silently skipped
# =============================================================================

def _check_binding_inputs(binding: ToolBinding, params: Mapping,
                          spec: RecipeSpec, verify: bool) -> tuple:
    if not verify:
        return (schemas.Check(schemas.COULD_NOT_CHECK, (
            "input verification was disabled by the caller (verify_inputs=False); no "
            "path was stat'ed. This is COULD_NOT_CHECK, never PASS.",)),)

    checks: list = []
    failures: list = []

    binary = Path(binding.binary)
    try:
        stat = binary.stat()
    except FileNotFoundError:
        # ENOENT is a confirmed negative, not an inability to check. Reporting it
        # as COULD_NOT_CHECK would let a missing binary construct a command.
        failures.append(f"binary does not exist: {binary}")
        stat = None
    except OSError as exc:
        checks.append(schemas.Check(schemas.COULD_NOT_CHECK,
                                    (f"cannot stat binary {binary}: {exc}",)))
        stat = None
    if stat is not None:
        if not binary.is_file():
            failures.append(f"binary is not a regular file: {binary}")
        elif not stat.st_mode & 0o111:
            failures.append(f"binary is not executable: {binary}")
        else:
            checks.append(schemas.Check(schemas.PASS,
                                        (f"binary exists and is executable: {binary}",)))

    for name in ("library_path", "source_root"):
        path = Path(getattr(binding, name))
        try:
            is_dir = path.is_dir()
        except OSError as exc:
            checks.append(schemas.Check(schemas.COULD_NOT_CHECK,
                                        (f"cannot stat {name} {path}: {exc}",)))
            continue
        if not is_dir:
            failures.append(f"{name} is not a directory: {path}")
        else:
            checks.append(schemas.Check(schemas.PASS, (f"{name} is a directory: {path}",)))

    git_marker = Path(binding.source_root) / ".git"
    try:
        exists = git_marker.exists()
    except OSError as exc:
        checks.append(schemas.Check(schemas.COULD_NOT_CHECK,
                                    (f"cannot stat {git_marker}: {exc}",)))
    else:
        if exists:
            checks.append(schemas.Check(schemas.PASS, (
                f"{git_marker} exists (a worktree records it as a file, a clone as a "
                f"directory); the authoritative `git rev-parse` check is delegated",)))
        else:
            failures.append(
                f"source_root has no .git entry: {binding.source_root}. An arm whose "
                f"source identity cannot be resolved cannot name its candidate commit.")

    if "model" in spec.param_map and params.get("model") is not None:
        model = Path(params["model"])
        try:
            size = model.stat().st_size if model.is_file() else None
        except OSError as exc:
            checks.append(schemas.Check(schemas.COULD_NOT_CHECK,
                                        (f"cannot stat model {model}: {exc}",)))
        else:
            if size is None:
                failures.append(f"model file not found: {model}")
            elif size == 0:
                failures.append(f"model file is empty: {model}")
            else:
                checks.append(schemas.Check(schemas.PASS,
                                            (f"model exists, {size} bytes: {model}",)))

    if failures:
        raise RecipeBindingError(
            "the recipe was asked to construct a command against inputs that cannot run:\n"
            + "\n".join(f"  - {reason}" for reason in failures)
            + "\nPass verify_inputs=False to construct anyway; the result then carries "
              "COULD_NOT_CHECK for every input rather than a pass."
        )
    return tuple(checks)


def _assert_arm_allows_binding(arm: str, binding: ToolBinding) -> None:
    """Denial 2: a CANDIDATE may not be built in, or measured from, a production tree.

    The anchor arm IS the frozen production binary, so it is allowed there —
    executing it read-only is not a production write. Production roots come from
    `storage.production_tree_forms()`, which resolves symlinks, rather than from a
    literal list retyped here.
    """
    if arm == "anchor":
        return
    roots = storage.production_tree_forms()
    for name in ("binary", "source_root", "library_path"):
        resolved = str(Path(getattr(binding, name)).resolve())
        for root in roots:
            if resolved == root or resolved.startswith(root.rstrip("/") + "/"):
                raise RecipeBindingError(
                    f"arm='candidate' but binding.{name} ({resolved}) is inside the FROZEN "
                    f"production tree {root!r}. P-AK-SEARCH-1 denial 2 forbids 'building "
                    f"in, committing to, or modifying any production tree'; a candidate "
                    f"measured out of production either was built there or is mislabelled. "
                    f"Use arm='anchor' to measure the frozen anchor."
                )


# =============================================================================
# construct() — the one entry point
# =============================================================================

def _sourced_constants(used_gpu_lib: bool) -> tuple:
    items = [
        {"name": "CANONICAL_PREFIX", "value": list(CANONICAL_PREFIX),
         "source": _CANONICAL_REL_PATH,
         "sha256": _MODULE_HASHES[_CANONICAL_REL_PATH]},
        {"name": "CANONICAL_BENCH_FLAGS_LLAMA_BENCH", "value": list(CANONICAL_BENCH_FLAGS),
         "source": _CANONICAL_REL_PATH,
         "sha256": _MODULE_HASHES[_CANONICAL_REL_PATH]},
        {"name": "CANONICAL_OMP_ENV", "value": dict(CANONICAL_OMP_ENV),
         "source": _CANONICAL_REL_PATH,
         "sha256": _MODULE_HASHES[_CANONICAL_REL_PATH]},
        {"name": "LLVM20_LIBDIR", "value": LLVM20_LIBDIR,
         "source": _CANONICAL_REL_PATH,
         "sha256": _MODULE_HASHES[_CANONICAL_REL_PATH]},
    ]
    if used_gpu_lib:
        items.append({
            "name": "GPU_BENCH_CORES", "value": _gpu_host_cpu_list_cache,
            "source": _GPU_LIB_REL_PATH,
            "sha256": _MODULE_HASHES[_GPU_LIB_REL_PATH]})
    return tuple(items)


def _constructor_sha256(spec: RecipeSpec, sourced: Sequence[Mapping]) -> str:
    """Bind the receipt to the recipe definition AND the bytes that supplied it.

    Precondition 6 requires *"the constructor's identifier and content hash"*. The
    hash covers this module's own bytes (which contain the builder), the spec's
    declarative form, and the content hash of every file a constant came from — so
    an edit to `canonical_recipe.py` changes the recipe hash of every record that
    cites it, exactly as precondition 5 requires of the evaluator bundle.

    The module list is derived from THIS recipe's own sourced constants, never from
    the whole `_MODULE_HASHES` table. That table grows when a GPU recipe resolves
    the GPU launcher, and hashing the table would have made a CPU recipe's receipt
    depend on whether a GPU recipe happened to be constructed earlier in the same
    process — a receipt that is not a function of the recipe is not a receipt.
    """
    paths = [_SELF_REL_PATH]
    for item in sourced:
        if item["source"] not in paths:
            paths.append(item["source"])
    return schemas.content_hash({
        "registry_id": REGISTRY_ID,
        "constructor_module_id": CONSTRUCTOR_MODULE_ID,
        "spec": spec.to_dict(),
        "modules": [{"path": path, "sha256": _MODULE_HASHES[path]}
                    for path in sorted(paths)],
        "sourced_constants": [dict(item) for item in sourced],
    })


def construct(recipe_id: str, *, binding: ToolBinding, params: Optional[Mapping] = None,
              arm: str = "candidate", verify_inputs: bool = True) -> ConstructedCommand:
    """Construct — and only construct — the argv for `recipe_id`.

    NOTHING is executed. The return value is the exact argv, the complete declared
    environment, the derived resource footprint the runner must claim, and the
    receipt that goes into the record's `recipe=` field.

    Raises `UnregisteredRecipe` for an unknown id, `RecipeParameterError` for a
    parameter outside its declared domain, `RecipeBindingError` for an arm/binding
    combination the protocol forbids or inputs that cannot run, and
    `RecipeDriftError` if the constructed command fails `canonical_recipe`'s own
    ratified validators.
    """
    spec = get_recipe(recipe_id)
    if arm not in ARMS:
        raise RecipeParameterError(f"arm: {arm!r} is not one of {list(ARMS)}")
    if not isinstance(binding, ToolBinding):
        raise TypeError(f"binding must be a ToolBinding, got {type(binding).__name__}")
    binary_name = Path(binding.binary).name
    if binary_name != spec.tool:
        raise RecipeBindingError(
            f"{recipe_id} emits argv for {spec.tool!r}, but binding.binary is named "
            f"{binary_name!r}. The flags below are that tool's flags; handing them to a "
            f"different program either fails loudly or measures something else under this "
            f"recipe's id. A renamed binary is a registry change, not a runtime override."
        )
    _assert_arm_allows_binding(arm, binding)

    supplied = dict(params or {})
    declared = spec.param_map
    unknown = sorted(set(supplied) - set(declared))
    if unknown:
        raise RecipeParameterError(
            f"{recipe_id}: unknown parameters {unknown}. Accepted: {sorted(declared)}. "
            f"A recipe that accepts arbitrary keys is a hand-typed argv with extra steps.")
    resolved: dict = {}
    for name, param in declared.items():
        if name in supplied:
            resolved[name] = param.validate(supplied[name])
        elif param.required:
            raise RecipeParameterError(
                f"{recipe_id}: required parameter {name!r} is missing ({param.doc})")
        else:
            resolved[name] = param.default

    used_gpu_lib = spec.backend == "llama_gpu"
    if used_gpu_lib:
        gpu_host_cpu_list()  # resolve + hash the sourced constant before building

    input_checks = _check_binding_inputs(binding, resolved, spec, verify_inputs)

    builder = _BUILDERS[spec.builder]
    argv, env, discipline, raw_samples_source, bounded, devices = builder(
        spec, binding, resolved)

    if len(argv) > MAX_ARGV_TOKENS:
        raise RecipeDriftError(
            f"{recipe_id}: constructed argv has {len(argv)} tokens, above the declared "
            f"bound {MAX_ARGV_TOKENS}; refusing rather than truncating")
    for index, token in enumerate(argv):
        _require_str(token, f"argv[{index}]")
    for key, value in env.items():
        _require_str(key, "env key")
        _require_str(value, f"env[{key}]")

    footprint = _footprint_from_argv(argv, devices)
    scope = _scope_from_footprint(footprint, argv)
    sourced = _sourced_constants(used_gpu_lib)
    receipt = api.RecipeReceipt(
        constructor_id=spec.recipe_id,
        constructor_sha256=_constructor_sha256(spec, sourced),
        argv_sha256=schemas.content_hash({
        "recipe_id": spec.recipe_id,
        "registry_id": REGISTRY_ID,
        "arm": arm,
        "argv": list(argv),
        "env": dict(env),
        "params": _jsonable(resolved),
    }),
    )
    phase = resolved.get("phase") or spec.phase
    return ConstructedCommand(
        recipe_id=spec.recipe_id,
        registry_id=REGISTRY_ID,
        tier=spec.tier,
        backend=spec.backend,
        phase=phase,
        cell_class=spec.cell_class,
        metric=spec.metric,
        metric_direction=spec.metric_direction,
        arm=arm,
        tool=spec.tool,
        argv=tuple(argv),
        env=dict(env),
        binding=binding,
        params=dict(resolved),
        claim_footprint=footprint,
        scope_denominator=scope,
        receipt=receipt,
        sourced_constants=sourced,
        discipline=tuple(discipline),
        input_checks=tuple(input_checks),
        raw_samples_source=raw_samples_source,
        bounded=tuple(bounded),
    )


def dry_run(recipe_id: str, *, binding: ToolBinding, params: Optional[Mapping] = None,
            arm: str = "candidate", verify_inputs: bool = True) -> dict:
    """The exact argv and env, as a canonical-JSON-able dict, with nothing executed.

    This module has no execution path at all, so `dry_run` is not a mode that
    suppresses one — it is `construct()` rendered for a human or a shell wrapper,
    in the same shape `canonical_recipe.py emit-bench-command` hands to
    `bench_canonical.sh`.
    """
    command = construct(recipe_id, binding=binding, params=params, arm=arm,
                        verify_inputs=verify_inputs)
    payload = command.to_dict()
    payload["dry_run"] = True
    payload["human_readable"] = command.render_human_readable()
    schemas.canonical_json(payload)  # refuses anything not canonicalizable
    return payload


# =============================================================================
# The api.RecipeConstructor seam
# =============================================================================

class AutoKernelRecipeConstructor:
    """`api.RecipeConstructor` implementation bound to one recipe and one binding.

    `api.py` never builds a command line; it only checks that a `RecipeReceipt`
    exists. This is the object that produces one. `construct(request)` additionally
    REFUSES a request whose cell does not match the recipe's, so a CPU recipe can
    never be recorded against a GPU cell, nor a decode recipe against a prefill one.
    """

    def __init__(self, recipe_id: str, *, binding: ToolBinding,
                 params: Optional[Mapping] = None, arm: str = "candidate",
                 verify_inputs: bool = True) -> None:
        self.spec = get_recipe(recipe_id)
        self.binding = binding
        self.params = dict(params or {})
        self.arm = arm
        self.verify_inputs = verify_inputs

    @property
    def constructor_id(self) -> str:
        """What the record cites: the recipe id, not a single opaque module id."""
        return self.spec.recipe_id

    def check_request(self, request: api.EvaluationRequest) -> schemas.Check:
        """PASS only when the request and the recipe describe the same cell."""
        if not isinstance(request, api.EvaluationRequest):
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"not an EvaluationRequest: {type(request).__name__}",))
        reasons = []
        for name, mine in (("backend", self.spec.backend),
                           ("tier", self.spec.tier),
                           ("cell_class", self.spec.cell_class),
                           ("metric", self.spec.metric),
                           ("metric_direction", self.spec.metric_direction)):
            theirs = getattr(request, name)
            if theirs != mine:
                reasons.append(f"{name}: request says {theirs!r}, recipe "
                               f"{self.spec.recipe_id} emits {mine!r}")
        phase = self.params.get("phase") or self.spec.phase
        if request.phase != phase:
            reasons.append(f"phase: request says {request.phase!r}, recipe emits {phase!r}")
        if reasons:
            return schemas.Check(schemas.FAIL, tuple(reasons))
        return schemas.Check(schemas.PASS)

    def construct(self, request: api.EvaluationRequest) -> tuple:
        """Return `(argv, env, receipt)` for `request`, or refuse.

        The tuple shape is the `api.RecipeConstructor` Protocol's; the full
        `ConstructedCommand` (footprint, scope, discipline vector) is available
        from `construct_full()`.
        """
        return self.construct_full(request).as_protocol_tuple()

    def construct_full(self, request: api.EvaluationRequest) -> "_ProtocolResult":
        check = self.check_request(request)
        if check.outcome != schemas.PASS:
            raise RecipeRequestMismatch(
                f"{self.spec.recipe_id} cannot serve this request: {list(check.reasons)}")
        command = construct(self.spec.recipe_id, binding=self.binding, params=self.params,
                            arm=self.arm, verify_inputs=self.verify_inputs)
        return _ProtocolResult(command)


@dataclass(frozen=True)
class _ProtocolResult:
    """Wraps a `ConstructedCommand` with the Protocol's tuple projection."""

    command: ConstructedCommand

    def as_protocol_tuple(self) -> tuple:
        return (self.command.argv, self.command.env, self.command.receipt)


# =============================================================================
# Self-audit — "constructs, never executes", proved not promised
# =============================================================================

_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input"})

_FORBIDDEN_CALL_ATTRS = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "fsync",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "rmtree", "rename", "chmod",
    "chown", "utime", "symlink", "link", "touch", "move", "copy", "copyfile", "copytree",
    "system", "popen", "Popen", "spawnv", "fork", "kill", "killpg", "send_signal",
    "terminate", "check_call", "check_output", "communicate", "run", "call", "setxattr",
})

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "shlex", "asyncio",
})


def audit_no_execution_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that it constructs commands and runs none.

    A recipe constructor that could also *run* the recipe would be a benchmark
    launcher, and every launcher this project has had drifted off the recipe it
    was supposed to enforce (2026-05-02, 2026-05-28). Separating construction from
    execution is what makes `argv_sha256` a property of the recipe rather than of
    the run.

    The audit also FAILs on any read of `os.environ` — not merely on importing
    `os` — because the emitted environment is **fully declared**. An ambient
    variable would make the argv hash a function of the invoking shell, and two
    arms measured from two shells would silently differ.

    `importlib`'s `exec_module` is NOT forbidden: binding
    `scripts/lib/canonical_recipe.py` is an import, not an execution of measurement
    work, and the bound file's identity is asserted before anything is read from it.

    COULD_NOT_CHECK when the source cannot be read or parsed — an unreadable module
    is not an audited one.

    Two ways this audit used to be passable by REMOVING what it inspects, both
    closed here:

      * an empty or comment-only `source` parses to an empty AST, finds nothing,
        and returned PASS. "Nothing forbidden appears in no code" is vacuous, and
        a check that a blank string passes is not a check;
      * with `source=None` the audit read the file FROM DISK, which is not
        necessarily the module that is running. It now binds the bytes it audits
        to the import-time hash in `_MODULE_HASHES`, so an on-disk edit after
        import reports COULD_NOT_CHECK instead of certifying a file nobody loaded
        — the same "verify THE consumer, not A consumer" rule the receipt obeys.
    """
    audited_self = source is None
    if source is None:
        try:
            raw = _HERE.read_bytes()
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {_HERE}: {exc}",))
        on_disk = hashlib.sha256(raw).hexdigest()
        expected = _MODULE_HASHES.get(_SELF_REL_PATH)
        if expected is not None and on_disk != expected:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{_HERE} has changed since it was imported (on disk "
                f"{on_disk[:12]}, loaded {expected[:12]}); auditing the file would "
                f"certify bytes that are not the running module",))
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - a .py that is not utf-8
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not decode {_HERE}: {exc}",))
    if not isinstance(source, str) or not source.strip():
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no source to audit: an empty or whitespace-only body yields an empty AST, "
            "in which nothing forbidden appears because nothing appears at all",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module: {exc}",))
    substantive = [
        node for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))
    ]
    if not substantive:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the audited source parses to a module body with no statement in it "
            "(comments and a docstring are not code); there is nothing to audit and "
            "PASS would be vacuous",))

    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                findings.append(f"line {node.lineno}: calls {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALL_ATTRS:
                findings.append(f"line {node.lineno}: calls .{func.attr}()")
        elif isinstance(node, ast.Attribute) and node.attr == "environ":
            findings.append(
                f"line {node.lineno}: reads .environ — the emitted env must be fully "
                f"declared, or argv_sha256 becomes a function of the invoking shell")

    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    subject = (f"{_SELF_REL_PATH}@{_MODULE_HASHES[_SELF_REL_PATH][:12]}, the bytes this "
               f"process imported" if audited_self else "the supplied source")
    return schemas.Check(schemas.PASS, (
        f"no write, process, or ambient-environment path in the AST of {subject}",))


__all__ = [
    "REGISTRY_ID", "CONSTRUCTOR_MODULE_ID", "REPO_ROOT", "CANONICAL_RECIPE_PATH",
    "GPU_BENCH_LIB_PATH", "MI210_NUMA_NODE", "GPU_HOST_THREADS_NUMA_NODE",
    "GPU_HOST_THREADS_ARE_NUMA_LOCAL",
    "RECIPE_FAMILY_T1A", "RECIPE_FAMILY_T1B", "RECIPE_FAMILIES", "ARMS",
    "CELL_CLASS_OPERATOR", "CELL_CLASS_TINY_GRAPH",
    "CANONICAL_PREFIX", "CANONICAL_BENCH_FLAGS", "CANONICAL_OMP_ENV", "LLVM20_LIBDIR",
    "GGML_TYPE_NAMES", "QUANTIZE_PERF_OPS", "QUANTIZE_PERF_UNMEASURABLE_TYPES",
    "LLAMA_BENCH_OUTPUT_FORMATS",
    "LLAMA_BENCH_SAMPLE_BEARING_FORMATS", "BACKEND_OPS_OUTPUT_FORMATS",
    "BACKEND_OPS_METRIC_BEARING_FORMATS", "GPU_VISIBLE_DEVICE_NAME",
    "MAX_OPS_PER_INVOCATION", "MAX_TYPES_PER_INVOCATION", "MAX_PARAMS_FILTER_CHARS",
    "MAX_QUANTIZE_ITERATIONS", "MAX_ARGV_TOKENS",
    "RecipeError", "UnregisteredRecipe", "RecipeParameterError", "RecipeBindingError",
    "RecipeDriftError", "RecipeRequestMismatch", "SourcedConstantUnavailable",
    "ParamSpec", "RecipeSpec", "ToolBinding", "ClaimFootprint", "DisciplineFinding",
    "ConstructedCommand", "AutoKernelRecipeConstructor",
    "REGISTRY", "RECIPE_IDS", "list_recipes", "get_recipe", "construct", "dry_run",
    "gpu_host_cpu_list", "worst_outcome", "audit_no_execution_paths",
]
