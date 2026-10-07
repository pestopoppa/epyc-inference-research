"""FLASH_ATTN_EXT case set, anchor bit-identity gate and paired perf screen for the
`cpu_fa_schedule` route (audit 2026-10-04 `ak-longctx-audit` C2/C3).

`cpu_fa_schedule` admits work-split and traversal changes to the CPU flash-attention
bodies (`ops.cpp`) that leave every output row's reduction order unchanged: e.g. the
G query heads sharing one KV head read that KV in ONE pass instead of G passes. Such a
change is bit-exact by construction, so its reference is the ANCHOR itself:

1. **Anchor bit identity** (`check_anchor_identity`). `cpu_fa_reference_probe.cpp` is
   compiled twice, once linked against each arm's ggml, and runs every case of
   `PROBE_CASES` (the case set below plus layout/prefill guards) with fixed inputs on
   the same thread team, under the arm's launch env and topology prefix. Each run
   repeats the graph `REPS` times in one process. The arms must agree bit for bit on
   every case, at every thread count, and under both settings of `GGML_FA_SPLIT_KV`
   (the AK recipes pin 0, production `:8074` leaves it on, and the split path's
   reduction order depends on the team size). Each repetition must reproduce the first
   (race detector). The probe names the `libggml-cpu` that actually served it (dladdr),
   which must be the arm's own: a probe that ran the other arm's kernel proves nothing.
   An anchor that disagrees with itself, or any infrastructure fault, is `unavailable`,
   never `wrong`.

2. **The test-backend-ops case set** (`CASE_SET_ID`, selected by the reviewed
   `AUTOKERNEL_CORRECTNESS_CASE_SET` env selector; precedent `odd_gqa7_d64_q1_v1`).
   The cases are registered by a llama-tree patch (`backend_ops_patch_block`) in both
   `make_test_cases_eval` (correctness corpus) and `make_test_cases_perf` (paired perf
   screen). A test-backend-ops binary that does not carry the literal has no such cases;
   the loop then skips both uses with a recorded reason (the probe above covers the
   same shapes) instead of reading 0/0 as evidence.

   - Q38FN: D=256, 2 KV heads, 12 query heads per KV head, kv 8k/64k/128k, nb 1..5 (every
     decode/verify query-row count the route's own admitted-text names -- "Decode/verify
     steps have N <= 5 query rows", `gates.CPU_SOURCE_ROUTES` cpu_fa_schedule -- 2026-10-07
     widened from {1, 5} so N=2,3,4 are not only covered by the identity gate's greedy
     serving requests, which never pin a query-row count).
   - DS41: D=512, 1 KV head, 64 query heads, attention sinks, kv n/2..n for n = 8k and
     64k (4k/8k/32k/64k), nb 1..5 (same widening). The stock mask cannot model DS41's
     sparse top-k mask; the probe approximates it with a fixed sparse mask, and the long
     serving surface (audit C1) is DS41's real judge.

3. **Paired perf screen** (`perf_screen`): `test-backend-ops perf` on the case set,
   anchor and candidate alternated ABAB on the recipe's thread team
   (`AUTOKERNEL_CPU_N_THREADS`, from the same patch), median per case per arm. The
   candidate must take at least `1 - MAX_SCREEN_RATIO` off the geometric mean before
   the serving A/B is paid for. This is a screen, not a measurement of record: nothing
   here is a keep or a headline.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
from typing import Callable, Literal, Mapping, Sequence

MARKER = "AK_CPU_FA_REFERENCE_V1"
PROBE = Path(__file__).with_name("cpu_fa_reference_probe.cpp")
CASE_SET_ID = "cpu_fa_longctx_v1"
CASE_SET_ENV = "AUTOKERNEL_CORRECTNESS_CASE_SET"
THREADS_ENV = "AUTOKERNEL_CPU_N_THREADS"
SEED = 20261004
REPS = 3
#: A team that divides neither 24 nor 64 query rows, besides the recipe's own team.
ODD_THREADS = 7
DEFAULT_THREADS = 48
PROBE_TIMEOUT_S = 900
PERF_TIMEOUT_S = 1800
SCREEN_ROUNDS = 3
#: Attention is at most ~68% of Q38FN decode wall at depth (audit §3.1: 55-68% of
#: per-token decode time scales with context). The enforced decode floor is 1.544%, so
#: an attention-only change must take >= 1.544 / 0.68 = 2.3% off attention before a
#: serving A/B can resolve it: the screen asks for 2.5%.
MAX_SCREEN_RATIO = 0.975
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class FaCase:
    name: str
    hsk: int
    hsv: int
    n_kv_heads: int
    gqa: int                    # query heads per KV head
    kv: int
    nb: int                     # query rows (tokens) per eval
    sinks: bool = False
    mask: str = "causal"        # probe only: causal | sparse (DS41 top-k stand-in)
    layout: str = "cache"       # probe only: cache (llama KV view) | plain
    backend_ops: bool = True    # registered in the test-backend-ops case set

    @property
    def n_q_heads(self) -> int:
        return self.n_kv_heads * self.gqa

    def vars(self) -> str:
        """`test_flash_attn_ext::vars()` of the registered case, byte for byte."""
        return (f"hsk={self.hsk},hsv={self.hsv},nh={self.n_kv_heads},nr23=[{self.gqa},1],"
                f"kv={self.kv},nb={self.nb},mask=1,sinks={int(self.sinks)},"
                "max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,"
                "permute=[0,1,2,3]")

    def cpp(self) -> str:
        return (f"test_cases.emplace_back(new test_flash_attn_ext({self.hsk}, {self.hsv}, "
                f"{self.n_kv_heads}, {{{self.gqa}, 1}}, {self.kv}, {self.nb}, true, "
                f"{'true' if self.sinks else 'false'}, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, "
                "GGML_TYPE_F16));")


#: Every query-row count a CPU decode/verify step can present (the route's own admitted
#: text: "Decode/verify steps have N <= 5 query rows" -- MTP/dflash verification batches
#: 2..5 draft tokens in one eval; greedy serving alone never exercises nb 2-4).
SERVED_QUERY_ROWS = (1, 2, 3, 4, 5)

CASE_SET = (
    *(FaCase(f"q38fn_kv{kv // 1024}k_nb{nb}", 256, 256, 2, 12, kv, nb)
      for kv in (8192, 65536, 131072) for nb in SERVED_QUERY_ROWS),
    *(FaCase(f"ds41_kv{kv // 1024}k_nb{nb}", 512, 512, 1, 64, kv, nb, sinks=True,
             mask="sparse")
      for kv in (4096, 8192, 32768, 65536) for nb in SERVED_QUERY_ROWS),
)
#: Probe-only guards: the plain (non-view) layout test-backend-ops uses, and a 64-row
#: prefill that takes the tiled path, so a dispatch change cannot reroute it unseen.
GUARD_CASES = (
    FaCase("q38fn_plain_kv4k_nb5", 256, 256, 2, 12, 4096, 5, layout="plain",
           backend_ops=False),
    FaCase("ds41_plain_kv4k_nb3", 512, 512, 1, 64, 4096, 3, sinks=True, mask="sparse",
           layout="plain", backend_ops=False),
    FaCase("q38fn_prefill_kv4k_nb64", 256, 256, 2, 12, 4096, 64, backend_ops=False),
)
PROBE_CASES = (*CASE_SET, *GUARD_CASES)
#: `test-backend-ops -p` selects by `std::regex_search` over vars(): anchored exact
#: alternation, so no other FLASH_ATTN_EXT case can ride along.
CASE_SET_REGEX = "^(" + "|".join(re.escape(case.vars()) for case in CASE_SET) + ")$"


def backend_ops_patch_block() -> str:
    """The C++ the llama-tree patch adds (static helper; called from eval and perf)."""
    lines = "\n".join(f"        {case.cpp()}" for case in CASE_SET)
    return (
        "// AutoKernel cpu_fa_schedule case set (epyc-inference-research\n"
        "// scripts/kernel_rnd/autokernel/loop/cpu_fa_reference.py CASE_SET, generated).\n"
        "// Long-context decode/verify shapes: Q38FN D=256 2 KV heads x GQA 12, DS41 D=512\n"
        "// 1 KV head x 64 with sinks. Registered only when the reviewed selector names this\n"
        "// set, so the generic FLASH_ATTN_EXT corpus keeps its size.\n"
        "static void autokernel_add_cpu_fa_longctx_cases(std::vector<std::unique_ptr<test_case>> & test_cases) {\n"
        f"    const char * case_set = std::getenv(\"{CASE_SET_ENV}\");\n"
        f"    if (case_set != nullptr && std::strcmp(case_set, \"{CASE_SET_ID}\") == 0) {{\n"
        f"{lines}\n"
        "    }\n"
        "}\n")


# ------------------------------------------------------------------ anchor identity

@dataclass(frozen=True)
class FaResult:
    status: Literal["pass", "wrong", "unavailable", "slower"]
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ProbeRun:
    library: str
    input_hash: str
    digests: tuple[str, ...]
    rows: tuple[str, ...]


def probe_argv(binary: Path, case: FaCase, threads: int, reps: int = REPS,
               seed: int = SEED) -> list[str]:
    return [str(binary), str(case.hsk), str(case.hsv), str(case.n_kv_heads), str(case.gqa),
            str(case.kv), str(case.nb), str(int(case.sinks)), case.mask, case.layout,
            str(threads), str(reps), str(seed)]


def parse_probe(output: str, case: FaCase, threads: int, reps: int = REPS,
                seed: int = SEED) -> ProbeRun:
    """Strict parse; anything unexpected raises ValueError (-> unavailable)."""
    lines = output.splitlines()
    header = (f"{MARKER} {case.hsk} {case.hsv} {case.n_kv_heads} {case.gqa} {case.kv} "
              f"{case.nb} {int(case.sinks)} {case.mask} {case.layout} {threads} {reps} {seed}")
    if not lines or lines.count(header) != 1 or lines[0] != header:
        raise ValueError("probe marker missing, duplicated or for another case")
    library = input_hash = None
    digests: dict[int, str] = {}
    rows: dict[int, str] = {}
    n_rows = case.nb * case.n_q_heads
    for line in lines[1:]:
        item = line.split(" ")
        if item[0] == "L" and len(item) == 2 and library is None:
            library = item[1]
        elif item[0] == "I" and len(item) == 2 and input_hash is None and \
                re.fullmatch(r"[0-9a-f]{16}", item[1]):
            input_hash = item[1]
        elif item[0] in ("D", "R") and len(item) == 3 and \
                re.fullmatch(r"\d+", item[1]) and re.fullmatch(r"[0-9a-f]{16}", item[2]):
            index, table, bound = int(item[1]), (digests if item[0] == "D" else rows), \
                (reps if item[0] == "D" else n_rows)
            if index >= bound or index in table:
                raise ValueError("out-of-range or duplicate digest line")
            table[index] = item[2]
        else:
            raise ValueError(f"unknown probe line {line[:80]!r}")
    if library is None or input_hash is None or len(digests) != reps or len(rows) != n_rows:
        raise ValueError("incomplete probe payload")
    return ProbeRun(library, input_hash, tuple(digests[i] for i in range(reps)),
                    tuple(rows[i] for i in range(n_rows)))


def split_kv_enabled(env: Mapping[str, str]) -> bool:
    """`GGML_FA_SPLIT_KV` as ops.cpp reads it: unset = on, else atoi(value) != 0."""
    value = env.get("GGML_FA_SPLIT_KV")
    if value is None:
        return True
    match = re.match(r"\s*([+-]?\d+)", value)
    return bool(match) and int(match.group(1)) != 0


def probe_configs(env: Mapping[str, str], threads: int) -> list[tuple[str, str, int]]:
    """(label, GGML_FA_SPLIT_KV value, team): the recipe's setting on an odd team and on
    the recipe team, then the other setting on the recipe team. The value is always set
    explicitly ("1" is what an unset variable means)."""
    here = "1" if split_kv_enabled(env) else "0"
    other = "0" if here == "1" else "1"
    configs = [(f"recipe split_kv={here} t{ODD_THREADS}", here, ODD_THREADS)]
    if threads != ODD_THREADS:
        configs.append((f"recipe split_kv={here} t{threads}", here, threads))
    configs.append((f"split_kv={other} t{threads}", other, threads))
    return configs


def compare_runs(case: FaCase, label: str, anchor: ProbeRun, candidate: ProbeRun,
                 anchor_lib: Path, candidate_lib: Path) -> FaResult | None:
    """None when the arms agree; otherwise the verdict (wrong or unavailable)."""
    where = f"{case.name} [{label}]"
    for role, run, lib in (("anchor", anchor, anchor_lib), ("candidate", candidate, candidate_lib)):
        if Path(run.library).resolve() != lib.resolve():
            return FaResult("unavailable", f"{where}: the {role} probe was served by "
                            f"{run.library}, not the {role} build's {lib}")
    if anchor.input_hash != candidate.input_hash:
        return FaResult("unavailable", f"{where}: the two probes generated different inputs")
    if len(set(anchor.digests)) != 1:
        return FaResult("unavailable", f"{where}: the ANCHOR disagrees with itself across "
                        f"{len(anchor.digests)} repetitions; this instrument cannot judge")
    want = anchor.digests[0]
    if candidate.digests[0] == want and len(set(candidate.digests)) != 1:
        bad = [i for i, digest in enumerate(candidate.digests) if digest != want]
        return FaResult("wrong", f"{where}: candidate repetitions {bad} of "
                        f"{len(candidate.digests)} differ from its first, which matched the "
                        "anchor (nondeterministic: a race)")
    if candidate.digests[0] != want or candidate.rows != anchor.rows:
        differing = [i for i, (a, b) in enumerate(zip(anchor.rows, candidate.rows)) if a != b]
        first = differing[0] if differing else None
        token, head = (divmod(first, case.n_q_heads) if first is not None else (None, None))
        return FaResult("wrong", f"{where}: output is not bit-identical to the anchor "
                        f"({len(differing)} of {len(anchor.rows)} rows differ; first at token "
                        f"{token}, query head {head}); cpu_fa_schedule admits only changes "
                        "that keep every row's reduction order")
    return None


def _arm_env(recipe, build_dir: Path) -> dict[str, str]:
    env = dict(getattr(recipe, "launch_env", None) or os.environ)
    lib_dir = str(Path(build_dir) / "bin")
    env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"]
                                        if env.get("LD_LIBRARY_PATH") else "")
    return env


def recipe_threads(recipe) -> int:
    template = getattr(recipe, "template", None)
    threads = getattr(template, "threads", None)
    return int(threads) if isinstance(threads, int) and threads > 0 else DEFAULT_THREADS


def check_anchor_identity(anchor_build: Path, candidate_build: Path, source_root: Path, *,
                          anchor_recipe, candidate_recipe,
                          cases: Sequence[FaCase] = PROBE_CASES, reps: int = REPS,
                          window: Callable[[], object] | None = None,
                          runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
                          ) -> FaResult:
    """Compile the probe against both arms and require bit identity on every case."""
    anchor_build, candidate_build = Path(anchor_build), Path(candidate_build)
    source_root = Path(source_root)
    needed = [build / "bin" / name for build in (anchor_build, candidate_build)
              for name in ("libggml.so", "libggml-base.so", "libggml-cpu.so")]
    needed += [source_root / "ggml/include/ggml.h", PROBE]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        return FaResult("unavailable", "FA probe: a ggml library, header or the probe is "
                        "missing", ", ".join(missing))
    threads = recipe_threads(candidate_recipe)
    if recipe_threads(anchor_recipe) != threads:
        return FaResult("unavailable", "anchor and candidate recipes use different teams")
    guard = window if window is not None else nullcontext
    passes = []
    try:
        with tempfile.TemporaryDirectory(prefix="ak-cpu-fa-ref-") as temp:
            binaries = {}
            for role, build in (("anchor", anchor_build), ("candidate", candidate_build)):
                lib_dir = build / "bin"
                binary = Path(temp) / f"fa-reference-probe-{role}"
                command = ["c++", "-std=c++17", "-O2", "-I", str(source_root / "ggml/include"),
                           str(PROBE), "-L", str(lib_dir), "-Wl,-rpath," + str(lib_dir),
                           "-lggml-cpu", "-lggml-base", "-lggml", "-ldl", "-o", str(binary)]
                # Headers come from the candidate tree: the route admits ops.cpp only.
                built = runner(command, capture_output=True, text=True, timeout=180,
                               env=dict(os.environ))
                if built.returncode:
                    return FaResult("unavailable", f"FA probe compile failed ({role})",
                                    built.stderr[-2000:])
                binaries[role] = binary
            with guard():
                for case in cases:
                    for label, split_kv, team in probe_configs(
                            _arm_env(candidate_recipe, candidate_build), threads):
                        runs = {}
                        for role, build, recipe in (
                                ("anchor", anchor_build, anchor_recipe),
                                ("candidate", candidate_build, candidate_recipe)):
                            env = _arm_env(recipe, build)
                            env["GGML_FA_SPLIT_KV"] = split_kv
                            prefix = tuple(getattr(recipe, "topology_prefix", ()) or ())
                            done = runner([*prefix, *probe_argv(binaries[role], case, team, reps)],
                                          capture_output=True, text=True,
                                          timeout=PROBE_TIMEOUT_S, env=env)
                            if done.returncode:
                                return FaResult("unavailable", f"{case.name} [{label}] {role} "
                                                "probe did not complete",
                                                f"exit={done.returncode}; {done.stderr[-1500:]}")
                            try:
                                runs[role] = parse_probe(done.stdout, case, team, reps)
                            except ValueError as exc:
                                return FaResult("unavailable", f"{case.name} [{label}] {role} "
                                                f"probe output invalid: {exc}")
                        verdict = compare_runs(case, label, runs["anchor"], runs["candidate"],
                                               anchor_build / "bin/libggml-cpu.so",
                                               candidate_build / "bin/libggml-cpu.so")
                        if verdict is not None:
                            return verdict
                        passes.append({"case": case.name, "config": label,
                                       "digest": runs["anchor"].digests[0]})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FaResult("unavailable", "FA probe infrastructure fault", str(exc))
    return FaResult("pass", f"{len(cases)} FLASH_ATTN_EXT cases bit-identical to the anchor "
                    f"under {len(passes) // max(1, len(cases))} team/env configurations, "
                    f"each repetition ({reps}x) identical",
                    json.dumps({"schema": "epyc.autokernel.cpu_fa_identity.v1",
                                "case_set": CASE_SET_ID, "passes": passes}, sort_keys=True))


# ------------------------------------------------------------------ case set + screen

def binary_has_case_set(build_dir: Path) -> bool:
    """True when the build's test-backend-ops carries the case-set selector literal
    (a 0/0 suite from a binary that lacks it is not evidence about anything)."""
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        return CASE_SET_ID.encode() in binary.read_bytes()
    except OSError:
        return False


_PERF_LINE = re.compile(r"^\s*FLASH_ATTN_EXT\(([^()]*)\):\s+(\d+) runs -\s+([0-9.]+) us/run")


def parse_perf(output: str) -> dict[str, float]:
    """{vars: us/run} for exactly the case set; ValueError otherwise."""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    found: dict[str, float] = {}
    for line in plain.splitlines():
        match = _PERF_LINE.match(line)
        if match is None:
            continue
        if match.group(1) in found:
            raise ValueError(f"case reported twice: {match.group(1)[:80]}")
        found[match.group(1)] = float(match.group(3))
    expected = {case.vars() for case in CASE_SET}
    if set(found) != expected:
        raise ValueError(f"perf output selected {len(found)} cases, not the "
                         f"{len(expected)} of {CASE_SET_ID}")
    if any(not math.isfinite(us) or us <= 0 for us in found.values()):
        raise ValueError("non-positive or non-finite us/run")
    return found


def perf_argv(build_dir: Path, recipe) -> list[str]:
    prefix = tuple(getattr(recipe, "topology_prefix", ()) or ())
    return [*prefix, str(Path(build_dir) / "bin" / "test-backend-ops"), "perf",
            "-o", "FLASH_ATTN_EXT", "-b", "CPU", "-p", CASE_SET_REGEX]


def perf_env(build_dir: Path, recipe) -> dict[str, str]:
    env = _arm_env(recipe, build_dir)
    env[CASE_SET_ENV] = CASE_SET_ID
    env[THREADS_ENV] = str(recipe_threads(recipe))
    return env


def perf_screen(anchor_build: Path, candidate_build: Path, *, anchor_recipe,
                candidate_recipe, rounds: int = SCREEN_ROUNDS,
                max_ratio: float = MAX_SCREEN_RATIO,
                window: Callable[[], object] | None = None,
                runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
                ) -> FaResult:
    """Paired ABAB `test-backend-ops perf` on the case set; `slower` refuses the serving
    A/B. A binary without the case set is `unavailable` (the caller records a skip)."""
    for role, build in (("anchor", anchor_build), ("candidate", candidate_build)):
        if not binary_has_case_set(build):
            return FaResult("unavailable", f"{role} test-backend-ops does not carry the "
                            f"{CASE_SET_ID} case set (llama-tree patch not applied)")
    if recipe_threads(anchor_recipe) != recipe_threads(candidate_recipe):
        return FaResult("unavailable", "anchor and candidate recipes use different teams")
    arms = {"anchor": (anchor_build, anchor_recipe), "candidate": (candidate_build,
                                                                   candidate_recipe)}
    samples: dict[str, list[dict[str, float]]] = {"anchor": [], "candidate": []}
    guard = window if window is not None else nullcontext
    try:
        with guard():
            for index in range(max(1, int(rounds))):
                order = ("anchor", "candidate") if index % 2 == 0 else ("candidate", "anchor")
                for role in order:
                    build, recipe = arms[role]
                    done = runner(perf_argv(build, recipe), capture_output=True, text=True,
                                  timeout=PERF_TIMEOUT_S, env=perf_env(build, recipe))
                    if done.returncode:
                        return FaResult("unavailable", f"{role} FA perf run failed "
                                        f"(exit {done.returncode})", done.stderr[-1500:])
                    try:
                        samples[role].append(parse_perf(done.stdout))
                    except ValueError as exc:
                        return FaResult("unavailable", f"{role} FA perf output invalid: {exc}",
                                        done.stdout[-1500:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FaResult("unavailable", "FA perf screen infrastructure fault", str(exc))
    rows = []
    for case in CASE_SET:
        anchor_us = statistics.median(sample[case.vars()] for sample in samples["anchor"])
        candidate_us = statistics.median(sample[case.vars()] for sample in samples["candidate"])
        rows.append({"case": case.name, "anchor_us": anchor_us,
                     "candidate_us": candidate_us, "ratio": candidate_us / anchor_us})
    geomean = math.exp(statistics.fmean(math.log(row["ratio"]) for row in rows))
    receipt = json.dumps({"schema": "epyc.autokernel.cpu_fa_perf_screen.v1",
                          "case_set": CASE_SET_ID, "rounds": len(samples["anchor"]),
                          "threads": recipe_threads(candidate_recipe),
                          "geomean_ratio": geomean, "max_ratio": max_ratio,
                          "rows": rows}, sort_keys=True)
    best = min(rows, key=lambda row: row["ratio"])
    summary = (f"FA case set candidate/anchor geomean {geomean:.4f} over {len(rows)} cases "
               f"(best {best['case']} {best['ratio']:.4f}), threshold {max_ratio}")
    if geomean > max_ratio:
        return FaResult("slower", summary + ": not enough attention speedup to justify the "
                        "serving A/B", receipt)
    return FaResult("pass", summary, receipt)


__all__ = ["CASE_SET", "CASE_SET_ENV", "CASE_SET_ID", "CASE_SET_REGEX", "FaCase", "FaResult",
           "GUARD_CASES", "PROBE_CASES", "THREADS_ENV", "backend_ops_patch_block",
           "binary_has_case_set", "check_anchor_identity", "compare_runs", "parse_perf",
           "parse_probe", "perf_screen", "probe_configs", "split_kv_enabled"]
