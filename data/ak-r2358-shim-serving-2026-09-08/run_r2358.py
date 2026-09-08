#!/usr/bin/env python3
"""R23-58 runner — does the whole-process THP shim improve the GPU SERVING FLOOR?

IMPLEMENTS EXACTLY the design registered in PREREGISTRATION.md, in this directory. If the
two ever disagree, the pre-registration wins and this file is the bug.

  * Two interleaved A/A calibrations, ONE build, arms differing only in
    `GGML_NOHUGEPAGE_PROCESS`. Not a paired A/B: `serving.compare` varies the BUILD from one
    recipe so it cannot express a same-build env arm at all, and its decision statistic is a
    median contrast -- exactly the statistic that would miss the compressed downside tail
    this experiment is hunting.
  * The unit is the SESSION (one llama-server launch). The shim is an
    `__attribute__((constructor))` `prctl(PR_SET_THP_DISABLE)` that runs before main(), and
    `PR_SET_THP_DISABLE` governs FUTURE faults, so it cannot be switched inside a live
    process. Any arm-unit number for it is meaningless by construction -- substituting one
    earlier today gave a 1200-fold sizing error.
  * Dispersion is measured with `serving._spread`, the SAME function `calibrate_floor` uses
    to define `floor_pct`, so the tested statistic and the campaign's keep gate cannot drift
    apart.
  * The positive control is NOT implemented here. It is a declarative `Recipe.env_readback`
    enforced inside `serving._measure_once` by `serving.verify_env_readback`, fail-closed in
    both directions. This runner DECLARES it and ASSERTS THAT IT FIRED via a delegating
    wrapper; it does not re-implement it. The AnonHugePages second control IS implemented
    here, because it is not in serving.py.

SAFETY. `--dry-run` is the default and launches nothing. `--run` is required to spend GPU.
The runner refuses to start on a busy host or a non-empty GPU, holds the loop's own device
flock for the whole window, never matches processes by name, writes every launch to disk as
it completes, prints the pre-registered verdict at the stop rule and never past it, and
never writes to `loop-memory` decision state.

Exit codes:  0 verdict issued   2 preflight refusal   3 API not yet available
            67 positive control failed / never fired   68 the two controls disagreed
            69 run inadmissible (claim lost, replacement budget exhausted)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

RESEARCH = Path("/mnt/raid0/llm/worktrees/mains/ak-rebuild-research")
sys.path.insert(0, str(RESEARCH / "scripts" / "kernel_rnd"))
sys.path.insert(1, str(RESEARCH))

from autokernel.loop import serving, residency, bench  # noqa: E402
from autokernel.loop import claim  # noqa: E402  (the device flock; acquired, never observed)

# --------------------------------------------------------------------------------------
# Registered constants. Changing any of these changes the pre-registration, not this file.
# --------------------------------------------------------------------------------------

#: The knob. NOT `GGML_NOHUGEPAGE` (the already-on madvise on ggml buffers) -- different
#: knob, different scope. This one is the whole-process prctl shim, default OFF / opt-in.
KNOB = "GGML_NOHUGEPAGE_PROCESS"

#: Control 0: proof the shim is COMPILED IN. A null from a knob that is not in the binary is
#: not evidence about the knob (INF-70 C9).
#:
#: SCAN THE ARTIFACT THAT ACTUALLY CARRIES THE CODE. The marker is an
#: `__attribute__((used))` static in `common/common.cpp`, which compiles into
#: **libllama-common.so** -- NOT into the `llama-server` executable, which on these builds
#: is an 18 KB stub that links everything. Searching the executable finds nothing even when
#: the shim is fully present.
#:
#: And it must be the LINKED library, resolved the way the loader will resolve it, not any
#: file matching a glob: `champ2/build-hip/bin` alone holds five vintages of
#: `libllama-common.so.0.0.*` from five different dates, so a glob-any-match check could
#: pass on a stale sibling the loader would never load. That is the three-ggml-generations
#: hazard wearing a different hat.
MARKER = ("INF70_CHAMPION3_PROCESS_THP_DISABLE=DEFAULT_OFF"
          ";OPT_IN=GGML_NOHUGEPAGE_PROCESS=1")

#: A build KNOWN not to contain the shim. Control 0 must REJECT it, or Control 0 is not a
#: check -- a scan that cannot fail says nothing about the scan that passed. Override with
#: --negative-control if this tree is ever reclaimed.
NEGATIVE_CONTROL_BUILD = Path("/mnt/raid0/llm/tmp/build-cor-445e93a8")

#: The build R23-58 runs on: the FOLD-2 candidate build of the current champion. HIP,
#: gfx90a, built 2026-09-08 09:25 from /mnt/raid0/llm/tmp/fold-ef81196d5-src @ ef81196d5.
DEFAULT_BUILD = Path("/mnt/raid0/llm/tmp/build-fold-ef81196d5")

#: `gfx90a-house-v1` -- the four flags the loop's own GPU build recipe declares
#: divergence-free against production.
HOUSE_FLAGS = {"GGML_HIP": "ON", "AMDGPU_TARGETS": "gfx90a",
               "GGML_HIP_ROCWMMA_FATTN": "ON", "GGML_NATIVE": "ON"}

#: Control 1, declared -- not implemented -- here. One declaration, BOTH directions.
#: An env state it does not cover is refused at Recipe construction, so a one-sided
#: declaration makes the control arm impossible to build rather than silently unchecked.
ENV_READBACK = ({"field": "THP_enabled", "env": KNOB, "expect": {"1": "0", "unset": "1"}},)

CHAMPION_COMMIT = "ef81196d5bdd4190b46dff4ae7eecc333a46c8ce"
RECIPE_PATH = RESEARCH / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json"
REQUIRED_CPU_LIST = "184-191"
STANDING_FLOOR_PCT = 4.581           # window-comparability reference only; never a gate.

COUPLES = 24                          # valid couples required
MAX_COUPLES = 28                      # 4-couple replacement budget; couple 29 is refused
ALPHA = 0.05                          # two-sided, claim (ii)
PERM_SEED = 2358                      # registered in advance
PERM_DRAWS = 200_000                  # used when exhaustive enumeration is too large
PERM_EXACT_MAX_N = 18                 # <= this many couples -> enumerate all 2^n

START_LOAD1_MAX = 8.0                 # refuse to start above this
LAUNCH_LOAD1_MAX = 12.0               # a launch above this is invalid (couple replaced)
SCLK_DRIFT_MAX_MHZ = 50               # a launch across a governor transition is invalid
ANON_HUGE_PCT_MAX_ON = 1.0            # ON arm above this => the two controls DISAGREE
MAX_INVALID_LAUNCH_EVENTS = 4

PORT = 18317                          # not the loop's 18311
LOOP_MEMORY = Path("/mnt/raid0/llm/autokernel/loop-memory")


# --------------------------------------------------------------------------------------
# API guard. A parallel change (U3, research lane b5f58b74) added env arms to serving.py.
# --------------------------------------------------------------------------------------

def require_serving_api() -> None:
    """Refuse clearly rather than fail obscurely three launches in."""
    fields = {f.name for f in dataclasses.fields(serving.Recipe)}
    missing = []
    for name in ("env", "env_readback"):
        if name not in fields:
            missing.append(f"serving.Recipe.{name} (dataclass field)")
    for name in ("with_env", "recipe_hash", "readback_expectations", "server_env", "to_dict"):
        if not hasattr(serving.Recipe, name):
            missing.append(f"serving.Recipe.{name}")
    for name in ("verify_env_readback", "EnvReadbackFailed", "RecipeError", "_spread",
                 "_measure_once", "UNSET",
                 # R23-60: this runner consumes serving's own residency record rather
                 # than sampling a second time. Against an older serving.py the sink is
                 # ignored, every launch records nothing, and `classify` fails them all.
                 "ServingNotResident", "RESIDENCY_PROVEN", "covers_request_phase"):
        if not hasattr(serving, name):
            missing.append(f"serving.{name}")
    if missing:
        sys.exit("R23-58 REFUSES TO RUN: autokernel.loop.serving is missing "
                 + ", ".join(missing)
                 + "\n  -> run after the U3 enabling change lands (research lane b5f58b74: "
                   "Recipe.env / Recipe.with_env / recipe_hash / env_readback / spread).")


# --------------------------------------------------------------------------------------
# Host state. Name-blind throughout: NEVER pkill/pgrep a name pattern on this shared host.
# --------------------------------------------------------------------------------------

def host_state() -> dict:
    """Everything about the host that can make a launch inadmissible.

    `kfd_processes` is the name-blind foreign-GPU-tenant check the loop already owns, and
    `vram_bytes`/`sclk_mhz` come from the same module the bench path proves residency with.
    `HSA_OVERRIDE_GFX_VERSION` is recorded because `serving._measure_once` builds its env
    from `Recipe.server_env`, which pins LD_LIBRARY_PATH but -- unlike bench/gates/hotspots
    -- does NOT go through `residency.loader_env`, so serving is the one path where a stray
    value would reach the process. Left as-is deliberately; recorded so it stays
    interpretable if it ever turns out to matter.
    """
    return {
        "at": time.time(),
        "load1": os.getloadavg()[0],
        "kfd_processes": residency.kfd_processes(),
        "vram_bytes": residency.vram_bytes(),
        "sclk_mhz": residency.sclk_mhz(),
        "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmake_cache(build_dir: Path) -> dict[str, str]:
    """`CMakeCache.txt` as {NAME: VALUE}, dropping the `:TYPE` suffix."""
    out: dict[str, str] = {}
    path = build_dir / "CMakeCache.txt"
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.split(":", 1)[0]] = value
    return out


def linked_common_library(build_dir: Path, recipe) -> tuple[Path | None, str]:
    """The `libllama-common.so` the loader WILL load, resolved exactly as the run resolves it.

    Uses `ldd` under `Recipe.server_env(build_dir)` -- the same environment
    `serving._measure_once` launches with -- so this answers "which file will actually be
    mapped", not "which files happen to be lying in bin/". `ldd` runs the dynamic loader
    only; `main()` never executes, no device is touched.

    A library resolved from OUTSIDE the build directory is REFUSED: that is the
    three-ggml-generations hazard in another form, and no exit code would ever report it.
    """
    server = build_dir / "bin" / "llama-server"
    if not server.is_file():
        return None, f"{server} is not a file"
    try:
        proc = subprocess.run(["ldd", str(server)], capture_output=True, text=True,
                              timeout=60, env=recipe.server_env(build_dir))
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"ldd failed: {exc}"
    for line in proc.stdout.splitlines():
        if "libllama-common.so" not in line or "=>" not in line:
            continue
        resolved = line.split("=>", 1)[1].strip().split(" (")[0].strip()
        if not resolved or resolved == "not found":
            return None, f"libllama-common.so unresolved: {line.strip()}"
        real = Path(resolved).resolve()
        try:
            real.relative_to(build_dir.resolve())
        except ValueError:
            return None, (f"REFUSED: llama-server resolves libllama-common.so to {real}, "
                          f"OUTSIDE {build_dir} -- a foreign library would be measured and "
                          f"no exit code would say so")
        return real, str(real)
    return None, "llama-server declares no libllama-common.so dependency"


def marker_present(build_dir: Path, recipe) -> tuple[bool, str]:
    """CONTROL 0: is the shim compiled into the artifact that will actually be LOADED?

    Scans the resolved `libllama-common.so`, because `common/common.cpp` compiles into that
    library and NOT into the `llama-server` executable (an 18 KB stub here). Scanning the
    executable returns a false negative on a build that fully contains the shim -- which is
    exactly what happened while this design was being written, and nearly bought an
    unnecessary rebuild that would have contended with a live measurement on this host.
    """
    lib, detail = linked_common_library(build_dir, recipe)
    if lib is None:
        return False, detail
    try:
        blob = lib.read_bytes()
    except OSError as exc:
        return False, f"cannot read {lib}: {exc}"
    if MARKER.encode() not in blob:
        return False, f"marker ABSENT from the linked library {lib}"
    return True, f"marker present in {lib}"


def build_identity(build_dir: Path, recipe) -> dict:
    """Everything that pins WHICH artifact produced a number.

    The linked common library is digested by name AND sha256, so the run record identifies
    the exact file that carried the shim -- not merely a build directory that contained one
    somewhere.
    """
    binaries: dict[str, str] = {}
    server = build_dir / "bin" / "llama-server"
    if server.is_file():
        binaries[str(server)] = file_sha256(server)
    lib, _ = linked_common_library(build_dir, recipe)
    if lib is not None:
        binaries[str(lib)] = file_sha256(lib)
    cache = cmake_cache(build_dir)
    source_dir = cache.get("CMAKE_HOME_DIRECTORY")
    source: dict = {"dir": source_dir, "head": None, "dirty": None}
    if source_dir and Path(source_dir).is_dir():
        for key, argv in (("head", ["rev-parse", "HEAD"]),
                          ("dirty", ["status", "--porcelain"])):
            try:
                got = subprocess.run(["git", "-C", source_dir, *argv], capture_output=True,
                                     text=True, timeout=60)
                source[key] = (got.stdout.strip() if key == "head"
                               else bool(got.stdout.strip()))
            except (OSError, subprocess.SubprocessError):
                pass
    prov = build_dir / "provenance.json"
    return {"build_dir": str(build_dir),
            "linked_common_library": str(lib) if lib else None,
            "binaries": binaries,
            "source": source,
            "cmake_flags": {k: cache.get(k) for k in HOUSE_FLAGS},
            "provenance_json": json.loads(prov.read_text()) if prov.is_file() else None}


def provenance_ok(ident: dict) -> tuple[bool, str]:
    """Is this build the champion, built the house way?

    `provenance.json` is accepted when present, but it is only an assertion. The structural
    route is stronger and is what this build supports: CMakeCache names the source tree, the
    source tree's HEAD is the champion commit, the tree is clean, and the four
    `gfx90a-house-v1` flags match. Residual, stated rather than hidden: a source tree can
    move AFTER a build, so HEAD describes the build only as well as the tree's mtime allows.
    """
    prov = ident.get("provenance_json") or {}
    if str(prov.get("champion_commit", "")) == CHAMPION_COMMIT:
        return True, f"provenance.json champion_commit={CHAMPION_COMMIT[:12]}"
    src = ident.get("source") or {}
    flags = ident.get("cmake_flags") or {}
    bad_flags = {k: flags.get(k) for k, v in HOUSE_FLAGS.items() if flags.get(k) != v}
    if src.get("head") != CHAMPION_COMMIT:
        return False, (f"source {src.get('dir')} HEAD={str(src.get('head'))[:12] or '<none>'}"
                       f", expected {CHAMPION_COMMIT[:12]}")
    if src.get("dirty"):
        return False, f"source tree {src.get('dir')} is DIRTY; HEAD does not describe it"
    if bad_flags:
        return False, f"gfx90a-house-v1 flag mismatch: {bad_flags}"
    return True, (f"CMakeCache source {src.get('dir')} @ {CHAMPION_COMMIT[:12]}, clean, "
                  f"house flags match")


# --------------------------------------------------------------------------------------
# CONTROL 2 -- AnonHugePages. Independent of the THP_enabled flag, and NOT in serving.py.
# --------------------------------------------------------------------------------------

def thp_sample(pid: int) -> dict:
    """`THP_enabled` + the AnonHugePages fraction of Rss, for one pid, right now.

    The flag proves the prctl took. The fraction proves it did to the ALLOCATION what it was
    meant to do (~0% of Rss when ON, because prctl also stops khugepaged collapsing later).
    AnonHugePages alone is NOT a discriminator -- the CPU session measured 0.06% of Rss at
    load and ~6% minutes later on the same process -- which is why both are carried and why
    the sampling points are fixed in advance.
    """
    out: dict = {"pid": pid, "at": time.time(), "thp_enabled": None,
                 "rss_kb": None, "anon_huge_kb": None, "anon_huge_pct": None}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "THP_enabled":
                out["thp_enabled"] = value.strip()
    except OSError:
        pass
    try:
        rss = anon = None
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key = key.strip()
            if key == "Rss":
                rss = int(value.split()[0])
            elif key == "AnonHugePages":
                anon = int(value.split()[0])
        out["rss_kb"], out["anon_huge_kb"] = rss, anon
        if rss and anon is not None:
            out["anon_huge_pct"] = round(100.0 * anon / rss, 4)
    except (OSError, ValueError, IndexError):
        pass
    return out


class AnonHugeProbe:
    """Polls one pid until it dies, keeping the last successful reading.

    The last reading is the pre-teardown sample the registered conflict rule is evaluated
    on. Daemon thread, same shape as `residency.Sampler`, so it can never hold the run open.
    """

    def __init__(self, pid: int, interval: float = 1.0) -> None:
        self.pid = pid
        self.interval = interval
        self.last: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = thp_sample(self.pid)
            if sample["thp_enabled"] is not None or sample["rss_kb"] is not None:
                self.last = sample
            self._stop.wait(self.interval)

    def stop(self) -> dict | None:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.last


# --------------------------------------------------------------------------------------
# CONTROL 1 -- delegating wrapper. Records that serving's own check FIRED. No logic here.
# --------------------------------------------------------------------------------------

_CURRENT: dict | None = None
_ORIG_VERIFY = serving.verify_env_readback


def _verify_wrapper(recipe, pid, *, status_text=None):
    """Delegate to serving.verify_env_readback, recording that it ran and what it saw.

    This is deliberately NOT a re-implementation: every decision -- the expectations, the
    both-directions assertion, the fail-closed behaviour on an unreadable /proc -- stays in
    serving.py. All that happens here is bookkeeping, plus grabbing the server pid (which
    serving hands us) so CONTROL 2 can start sampling the same process at the same moment.
    """
    rec = _CURRENT
    if rec is not None:
        rec["readback_fired"] = True
        rec["readback_pid"] = pid
        rec["readback_expectations"] = [list(t) for t in recipe.readback_expectations()]
        rec["thp_first_health"] = thp_sample(pid)
        rec["_probe"] = AnonHugeProbe(pid)
    try:
        observed = _ORIG_VERIFY(recipe, pid, status_text=status_text)
    except Exception as exc:
        if rec is not None:
            rec["readback_error"] = f"{type(exc).__name__}: {exc}"
        raise
    if rec is not None:
        rec["readback_observed"] = observed
    return observed


serving.verify_env_readback = _verify_wrapper


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------

def build_arms(recipe_path: Path) -> tuple:
    """The two arms, from ONE recipe file, via `with_env`. No JSON is edited.

    `env_readback` is attached BEFORE the split so both arms carry the same declaration;
    `Recipe.__post_init__` resolves it at construction, so an arm whose env state the
    declaration does not cover cannot be built at all.
    """
    base = serving.Recipe.load(recipe_path)
    declared = dataclasses.replace(
        base, env_readback=tuple(dict(c) for c in ENV_READBACK))
    off = declared.with_env(name=f"{base.name}+thp-shim-off", **{KNOB: None})
    on = declared.with_env(name=f"{base.name}+thp-shim-on", **{KNOB: "1"})
    return off, on


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------

def preflight(args, off, on) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    out = args.out.resolve()
    check("out dir is not loop-memory decision state",
          LOOP_MEMORY not in out.parents and out != LOOP_MEMORY,
          f"{out}")

    build = args.build
    check("build dir exists", (build / "bin" / "llama-server").is_file(),
          str(build / "bin" / "llama-server"))

    ok, detail = marker_present(build, off) if (build / "bin").is_dir() else \
        (False, f"{build}/bin is not a directory")
    check("CONTROL 0: shim present in the LINKED libllama-common.so", ok,
          detail if ok else
          f"{detail}. Refusing: a null from a knob that is not in the binary is not "
          f"evidence about the knob. (Scan the library, never the 18 KB llama-server stub.)")

    # A scan that cannot fail is not a scan. Prove Control 0 DISCRIMINATES on a build known
    # to predate the fold, before trusting the pass above.
    neg = args.negative_control
    if (neg / "bin").is_dir():
        neg_ok, neg_detail = marker_present(neg, off)
        check("CONTROL 0 negative control REJECTS a pre-fold build", not neg_ok,
              f"{neg.name}: {neg_detail}")
    else:
        check("CONTROL 0 negative control REJECTS a pre-fold build", False,
              f"{neg} is missing -- point --negative-control at any pre-fold build. "
              f"Control 0 is not trusted until it has been shown to fail on one.")

    ident = build_identity(build, off) if (build / "bin").is_dir() else {}
    prov_ok, prov_detail = provenance_ok(ident) if ident else (False, "no build")
    check("build provenance is the champion, built the house way", prov_ok, prov_detail)

    check("recipe cpu_list is pinned", off.cpu_list == REQUIRED_CPU_LIST,
          f"cpu_list={off.cpu_list!r} (the 4.581% floor was calibrated under "
          f"{REQUIRED_CPU_LIST}; unpinned is a different measured condition, R23-49)")

    check("arms differ in recipe_hash", off.recipe_hash != on.recipe_hash,
          f"off={off.recipe_hash[:12]} on={on.recipe_hash[:12]}")
    check("arms differ ONLY in the knob",
          {k: v for k, v in off.to_dict().items() if k not in ("name", "env")}
          == {k: v for k, v in on.to_dict().items() if k not in ("name", "env")}
          and dict(on.env or {}) == {KNOB: "1"} and KNOB not in (off.env or {}),
          f"off.env={dict(off.env or {})} on.env={dict(on.env or {})}")

    off_rb, on_rb = off.readback_expectations(), on.readback_expectations()
    check("CONTROL 1 declared, OFF arm expects THP_enabled=1",
          ("THP_enabled", "1") in off_rb, f"{off_rb}")
    check("CONTROL 1 declared, ON arm expects THP_enabled=0",
          ("THP_enabled", "0") in on_rb, f"{on_rb}")

    check("kernel exposes THP_enabled",
          "THP_enabled" in Path("/proc/self/status").read_text(),
          "/proc/self/status")
    check("kernel exposes smaps_rollup AnonHugePages",
          Path("/proc/self/smaps_rollup").is_file(), "/proc/self/smaps_rollup")

    state = host_state()
    check("host is quiet", state["load1"] <= START_LOAD1_MAX,
          f"load1={state['load1']:.2f} (limit {START_LOAD1_MAX})")
    check("no foreign GPU tenant (KFD, name-blind)", state["kfd_processes"] == 0,
          f"kfd_processes={state['kfd_processes']}")
    check("GPU has no resident model",
          0 <= state["vram_bytes"] < residency.RESIDENT_FLOOR_BYTES,
          f"vram={state['vram_bytes']} bytes (floor {residency.RESIDENT_FLOOR_BYTES})")

    check("GPU host-thread pin matches the loop's own constant",
          off.cpu_list == bench.CPU_LIST, f"bench.CPU_LIST={bench.CPU_LIST}")

    if not args.resume:
        prior = (out / "launches.jsonl").exists()
        check("no prior results in the output directory", not prior,
              f"{out/'launches.jsonl'} EXISTS -- pass --resume deliberately or use a new "
              f"--out" if prior else "clean")
    return checks


# --------------------------------------------------------------------------------------
# Verdict arithmetic. Registered in PREREGISTRATION.md sections 3.1 and 3.2.
# --------------------------------------------------------------------------------------

def sign_verdict(signs: list[int]) -> dict:
    """Claim (i): the CPU session's frozen two-look boundary. Exact two-sided alpha 0.0430.

        LOOK 1  after  6 valid couples : CALL iff 6/6 agree
        LOOK 2  after 10 valid couples : CALL iff >= 9/10 agree

    The verdict FREEZES at whichever look first calls; collection continues to n=24 because
    claim (ii) needs the launches. Alpha is a property of the rule, not of when the machine
    stops. A call is a DIRECTION, never a magnitude.
    """
    if len(signs) >= 6:
        first6 = signs[:6]
        if all(s == 1 for s in first6):
            return {"verdict": "T+", "look": 1, "detail": "6/6 ON faster", "alpha": 0.0430}
        if all(s == -1 for s in first6):
            return {"verdict": "T-", "look": 1, "detail": "6/6 OFF faster", "alpha": 0.0430}
    if len(signs) >= 10:
        first10 = signs[:10]
        up = sum(1 for s in first10 if s == 1)
        if up >= 9:
            return {"verdict": "T+", "look": 2, "detail": f"{up}/10 ON faster",
                    "alpha": 0.0430}
        if (10 - up) >= 9:
            return {"verdict": "T-", "look": 2, "detail": f"{10-up}/10 OFF faster",
                    "alpha": 0.0430}
        return {"verdict": "T0", "look": 2, "detail": f"{up}/10 ON faster", "alpha": 0.0430}
    return {"verdict": "PENDING", "look": None,
            "detail": f"{len(signs)} valid couples, look 1 needs 6", "alpha": 0.0430}


def _p95_dev(runs) -> float:
    """Delegates to serving._spread so the tested statistic IS the floor's statistic."""
    return serving._spread(list(runs))["p95_dev_pct"]


def _sd_log(runs) -> float:
    return statistics.pstdev([math.log(r) for r in runs])


def dispersion_verdict(off_runs, on_runs, *, alpha=ALPHA, seed=PERM_SEED) -> dict:
    """Claim (ii): ratio of p95 deviations, exact within-couple label permutation.

    Each arm is divided by its OWN median first, so a location difference is removed exactly
    and cannot leak into a scale test. The label is then swapped WITHIN couples, which
    preserves the pairing and therefore preserves host drift. `p95_dev_pct` decides -- it is
    the floor's own formula, so the statistic and the keep gate cannot drift apart. The sd
    ratio is reported alongside; it is better behaved in the abstract but measures something
    adjacent to, not identical to, the number the campaign gates on.
    """
    n = len(off_runs)
    off_med, on_med = statistics.median(off_runs), statistics.median(on_runs)
    a = [r / off_med for r in off_runs]
    b = [r / on_med for r in on_runs]

    def stat_p95(x, y) -> float:
        return math.log(_p95_dev(x) / _p95_dev(y))

    def stat_sd(x, y) -> float:
        return math.log(_sd_log(x) / _sd_log(y))

    def permute_p(stat) -> tuple[float, float, int]:
        obs = stat(a, b)
        target = abs(obs) - 1e-12
        if n <= PERM_EXACT_MAX_N:
            draws, exhaustive = 1 << n, True
            def masks():
                return range(draws)
        else:
            rng = random.Random(seed)
            draws, exhaustive = PERM_DRAWS, False
            def masks():
                return (rng.getrandbits(n) for _ in range(draws))
        hits = 0
        for mask in masks():
            x, y = [], []
            for i in range(n):
                if (mask >> i) & 1:
                    x.append(b[i]); y.append(a[i])
                else:
                    x.append(a[i]); y.append(b[i])
            if abs(stat(x, y)) >= target:
                hits += 1
        p = hits / draws if exhaustive else (hits + 1) / (draws + 1)
        return obs, p, draws

    obs_p95, p_p95, draws = permute_p(stat_p95)
    obs_sd, p_sd, _ = permute_p(stat_sd)
    rho_p95, rho_sd = math.exp(obs_p95), math.exp(obs_sd)

    if p_p95 < alpha:
        verdict = "D+" if rho_p95 > 1.0 else "D-"
    else:
        verdict = "D0"
    return {"verdict": verdict, "alpha": alpha, "n_per_arm": n,
            "statistic": "ratio of spread.p95_dev_pct (OFF/ON), serving._spread",
            "rho_p95": rho_p95, "p_p95": p_p95,
            "rho_sd_log": rho_sd, "p_sd_log": p_sd,
            "permutation": {"scheme": "within-couple label swap on median-normalised runs",
                            "draws": draws, "exhaustive": n <= PERM_EXACT_MAX_N,
                            "seed": seed if n > PERM_EXACT_MAX_N else None}}


#: The 3x3 action table, registered in PREREGISTRATION.md section 4.
VERDICT_ACTIONS = {
    ("T+", "D+"): "ADOPT (strongest form): set env GGML_NOHUGEPAGE_PROCESS=1 on the GPU "
                  "serving recipe, recalibrate the floor under ON, publish both. Quote no "
                  "throughput magnitude.",
    ("T+", "D0"): "ADOPT as a launcher default (zero-cost, reversible, no rebuild). "
                  "PUBLISH NO MAGNITUDE -- a sign test gives direction only. Floor "
                  "unchanged; the campaign gains no resolution.",
    ("T+", "D-"): "REJECT. A faster arm that is noisier raises the bar every future keep "
                  "must clear. Report D- to the CPU side (INF-70).",
    ("T0", "D+"): "SUCCESS -- THE TARGET OUTCOME. ADOPT. Null level, real resolution gain. "
                  "Recalibrate the serving floor under ON and re-run the 6-keep bundle "
                  "against the new floor.",
    ("T0", "D0"): "NULL. Do not adopt. Close R23-58. Report as a BOUNDED null (see the "
                  "achieved-power line) -- 'no effect' unqualified is not permitted.",
    ("T0", "D-"): "REJECT. Report D- to the CPU side (INF-70).",
    ("T-", "D+"): "HARD CELL -- apply the registered T-/D+ rule, printed below. Never "
                  "adjudicated after the fact.",
    ("T-", "D0"): "REJECT. Direction is against and nothing is bought. Record and close.",
    ("T-", "D-"): "REJECT, unambiguously -- worse on both axes. Report D- to the CPU side.",
}


def hard_cell_rule(off_runs, on_runs) -> dict:
    """The T-/D+ arithmetic, registered before any data existed."""
    off_sp, on_sp = serving._spread(off_runs), serving._spread(on_runs)
    level_cost = 100.0 * (1.0 - on_sp["median"] / off_sp["median"])
    floor_saving = off_sp["p95_dev_pct"] - on_sp["p95_dev_pct"]
    if level_cost >= STANDING_FLOOR_PCT:
        action = (f"DO NOT ADOPT in any form: level_cost {level_cost:.3f}% >= the standing "
                  f"floor {STANDING_FLOOR_PCT}%. A change costing more level than the "
                  f"instrument can resolve is a regression wearing a measurement.")
    elif floor_saving > level_cost:
        action = (f"ADOPT for serving: floor_saving {floor_saving:.3f}% > level_cost "
                  f"{level_cost:.3f}% -- it buys strictly more resolution than it costs.")
    else:
        action = (f"DO NOT ADOPT for production serving; ADOPT AS A MEASUREMENT INSTRUMENT "
                  f"ONLY (floor_saving {floor_saving:.3f}% <= level_cost "
                  f"{level_cost:.3f}%). Champion is still SERVED with the shim OFF and "
                  f"every headline stays an OFF-arm number.")
    return {"level_cost_pct": level_cost, "floor_saving_pct": floor_saving,
            "action": action}


# --------------------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------------------

class ControlFailure(RuntimeError):
    """The positive control failed or never fired. Fatal, exit 67."""


class ControlsDisagree(RuntimeError):
    """THP_enabled and AnonHugePages disagree. Registered as the finding. Exit 68."""


def run_launch(recipe, build_dir: Path, port: int, *, couple: int, arm: str,
               position: int, ident: dict, out_dir: Path) -> dict:
    """One session. One observation. Written to disk before this function returns."""
    global _CURRENT
    rec: dict = {
        "schema": "epyc.r2358.launch.v1",
        "couple": couple, "arm": arm, "position_in_couple": position,
        "recipe": recipe.name, "recipe_hash": recipe.recipe_hash,
        "recipe_env": dict(recipe.env or {}), "recipe_describe": recipe.describe(),
        "build": ident, "port": port,
        "readback_fired": False, "readback_pid": None, "readback_observed": None,
        "readback_expectations": None, "readback_error": None,
        "thp_first_health": None, "thp_pre_teardown": None,
        "aggregate_tok_s": None, "error": None,
        "host_pre": host_state(), "host_post": None, "residency": None,
        "started_at": time.time(), "wall_s": None,
        "valid": False, "invalid_reason": None,
    }
    _CURRENT = rec
    t0 = time.time()
    # R23-60: `serving._measure_once` is now THE residency sampler for the serving path.
    # This runner used to wrap each launch in its own `residency.Sampler` because
    # serving.py sampled nothing; wrapping one around the other now would run TWO sampling
    # threads over the same window. So we consume serving's record instead of collecting
    # our own. The recorded FIELDS are a superset of what `sampler.proof` gave (same keys,
    # plus `status`, `sampled`, the window/request timestamps and `median_vram_bytes`), so
    # `classify` below is byte-for-byte the same admissibility rule it was pre-registered
    # as -- it reads `resident`, `peak_vram_bytes`, `peak_kfd_processes`, `sclk_*`, and all
    # five are still present and still mean what they meant.
    launch_residency: list[dict] = []
    try:
        try:
            rec["aggregate_tok_s"] = serving._measure_once(
                recipe, build_dir, port, evidence=launch_residency)
        except serving.EnvReadbackFailed as exc:
            rec["error"] = f"EnvReadbackFailed: {exc}"
        except Exception as exc:                           # ServerDied and anything else
            rec["error"] = f"{type(exc).__name__}: {exc}"
        # Appended by `_measure_once` in a `finally`, so it is present on a failed launch
        # too. Empty would mean serving.py stopped recording -- left as None so `classify`
        # marks the launch inadmissible rather than silently admitting an unproven one.
        rec["residency"] = launch_residency[-1] if launch_residency else None
    finally:
        probe = rec.pop("_probe", None)
        if probe is not None:
            rec["thp_pre_teardown"] = probe.stop()
        rec["wall_s"] = round(time.time() - t0, 2)
        rec["host_post"] = host_state()
        _CURRENT = None

    rec["valid"], rec["invalid_reason"] = classify(rec)
    append_launch(out_dir, rec)
    return rec


def classify(rec: dict) -> tuple[bool, str | None]:
    """Admissibility, exactly as registered. Raises on the two FATAL classes."""
    if rec["error"] and rec["error"].startswith("EnvReadbackFailed"):
        raise ControlFailure(
            f"couple {rec['couple']} arm {rec['arm']}: {rec['error']}\n"
            f"  Setting the knob is not evidence it took effect. In the OFF arm this means "
            f"the CONTROL was secretly the TREATMENT, which is unrecoverable after the fact."
        )
    if not rec["readback_fired"] and not rec["error"]:
        raise ControlFailure(
            f"couple {rec['couple']} arm {rec['arm']}: serving.verify_env_readback never "
            f"fired -- the arm is UNCHECKED. An unfired control is not a passed control.")

    on_arm = rec["recipe_env"].get(KNOB) == "1"
    tail = rec.get("thp_pre_teardown") or {}
    if on_arm and tail.get("thp_enabled") == "0":
        pct = tail.get("anon_huge_pct")
        if pct is not None and pct >= ANON_HUGE_PCT_MAX_ON:
            raise ControlsDisagree(
                f"couple {rec['couple']} ON arm: THP_enabled=0 but AnonHugePages is "
                f"{pct:.3f}% of Rss (limit {ANON_HUGE_PCT_MAX_ON}%). The two independent "
                f"controls disagree. Registered outcome: the disagreement IS the finding; "
                f"the run stops and no T/D verdict is issued.")

    if rec["error"]:
        return False, rec["error"]
    if rec["aggregate_tok_s"] is None or rec["aggregate_tok_s"] <= 0:
        return False, "no aggregate_tok_s"
    if rec["host_pre"]["load1"] > LAUNCH_LOAD1_MAX:
        return False, (f"host not quiet: pre-launch load1 "
                       f"{rec['host_pre']['load1']:.2f} > {LAUNCH_LOAD1_MAX}")
    proof = rec.get("residency") or {}
    # R23-60: `serving._measure_once` REFUSES a launch it sampled as non-resident over a
    # covering window, so that case now arrives as `rec["error"] = "ServingNotResident:
    # ..."` and is already inadmissible above. What remains here is the unproven case --
    # an unreadable instrument, a window that did not cover the request phase, or a
    # serving.py that recorded nothing -- and it stays inadmissible, exactly as before.
    if not proof.get("resident"):
        return False, (f"GPU residency not proven: peak_vram="
                       f"{proof.get('peak_vram_bytes')} < "
                       f"{residency.RESIDENT_FLOOR_BYTES}")
    if proof.get("status") != serving.RESIDENCY_PROVEN:
        return False, (f"GPU residency not proven: status="
                       f"{proof.get('status')!r} covers_request_phase="
                       f"{proof.get('covers_request_phase')!r}")
    if proof.get("peak_kfd_processes") != 1:
        return False, f"peak_kfd_processes={proof.get('peak_kfd_processes')}, expected 1"
    lo, hi = proof.get("sclk_min_mhz") or 0, proof.get("sclk_max_mhz") or 0
    if lo and hi - lo > SCLK_DRIFT_MAX_MHZ:
        return False, f"clock moved {hi-lo} MHz during the launch (limit {SCLK_DRIFT_MAX_MHZ})"
    return True, None


# --------------------------------------------------------------------------------------
# Incremental persistence. Never only at the end.
# --------------------------------------------------------------------------------------

def append_launch(out_dir: Path, rec: dict) -> None:
    with (out_dir / "launches.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    env = ",".join(f"{k}={v}" for k, v in sorted(rec["recipe_env"].items())) or "unset"
    first = (rec.get("thp_first_health") or {})
    tail = (rec.get("thp_pre_teardown") or {})
    line = (f"c{rec['couple']:02d}{rec['arm']} pid={rec.get('readback_pid')} "
            f"THP_enabled={first.get('thp_enabled')} "
            f"sessenv=[{KNOB}={env}] "
            f"fired={rec['readback_fired']} observed={rec.get('readback_observed')} "
            f"anon_huge_pct_first={first.get('anon_huge_pct')} "
            f"anon_huge_pct_tail={tail.get('anon_huge_pct')} "
            f"rss_kb_tail={tail.get('rss_kb')} valid={rec['valid']}")
    with (out_dir / "thp_proof.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def arm_record(name: str, recipe, runs: list[float]) -> dict:
    """`epyc.autokernel.serving_floor.v1`-shaped, but marked interleaved: these launches were
    NOT a consecutive block, which is the whole point (drift cannot alias into the arm)."""
    sp = serving._spread(runs)
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(), "metric": recipe.metric, "np": recipe.np,
            "samples": len(runs), "median_tok_s": sp["median"],
            "floor_pct": sp["p95_dev_pct"], "runs": runs, "cv_pct": sp["cv_pct"],
            "spread": sp, "interleaved": True, "arm": name,
            "conditions": {"cpu_list": recipe.cpu_list,
                           "harness": "R23-58 interleaved order-balanced A/A; fresh server "
                                      "per sample; serving._spread",
                           "note": "NOT a consecutive-block calibrate_floor run"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R23-58: THP shim vs the GPU serving floor")
    parser.add_argument("--run", action="store_true",
                        help="actually launch servers. Without it, this is a dry run and "
                             "nothing touches the GPU.")
    parser.add_argument("--build", type=Path, required=False, default=DEFAULT_BUILD,
                        help="HIP build of ef81196d5 whose LINKED libllama-common.so "
                             "carries the shim marker (default: %(default)s)")
    parser.add_argument("--negative-control", type=Path, default=NEGATIVE_CONTROL_BUILD,
                        help="a build known NOT to contain the shim; Control 0 must reject "
                             "it, or Control 0 is not a check (default: %(default)s)")
    parser.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    parser.add_argument("--out", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/r2358-shim-serving-20260908"))
    parser.add_argument("--couples", type=int, default=COUPLES)
    parser.add_argument("--max-couples", type=int, default=MAX_COUPLES)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    require_serving_api()
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        off, on = build_arms(args.recipe)
    except serving.RecipeError as exc:
        print(f"REFUSED at recipe construction: {exc}")
        return 2

    print("R23-58 — does the whole-process THP shim improve the GPU serving floor?")
    print(f"  registration : {args.out/'PREREGISTRATION.md'}")
    print(f"  build        : {args.build}")
    print(f"  OFF arm      : {off.describe()}")
    print(f"  ON  arm      : {on.describe()}")
    print(f"  plan         : {args.couples} valid couples "
          f"({args.couples} launches per arm, {2*args.couples} total), "
          f"max {args.max_couples} couples launched")
    print(f"  alpha        : {ALPHA} two-sided (claim ii); 0.0430 (claim i, two looks)")
    print(f"  est. wall    : {2*args.couples*75/60:.0f}–{2*args.max_couples*82/60:.0f} min "
          f"at 75–82 s per launch (derived from loop-memory serving artifacts)")
    print()

    checks = preflight(args, off, on)
    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print()

    if not args.run:
        print("DRY RUN — nothing launched, no GPU touched, no core in 0-95 used.")
        print("  OFF argv:", " ".join(off.server_argv(args.build, args.port)))
        print("  ON  argv:", " ".join(on.server_argv(args.build, args.port)))
        print(f"  ON env delta: {KNOB}={dict(on.env or {}).get(KNOB)}")
        print("  schedule: couple k odd -> (OFF, ON); k even -> (ON, OFF)")
        print(f"  re-run with --run once every check above is [ok ]"
              + ("" if not failed else f"  ({failed} still failing)"))
        return 0 if not failed else 2

    if failed:
        print(f"REFUSING TO RUN: {failed} preflight check(s) failed.")
        return 2

    off_runs: list[float] = []
    on_runs: list[float] = []
    signs: list[int] = []
    couples_launched = 0
    invalid_events = 0
    ident = build_identity(args.build, off)
    started = time.time()
    exit_code = 0

    try:
        with claim.hold() as receipt:
            print(f"claim     held on {receipt['device_id']} (acquired, never observed)\n")
            while len(signs) < args.couples and couples_launched < args.max_couples:
                couples_launched += 1
                k = couples_launched
                order = [("off", off), ("on", on)] if k % 2 == 1 else \
                        [("on", on), ("off", off)]
                pair: dict[str, dict] = {}
                for position, (name, recipe) in enumerate(order, start=1):
                    rec = run_launch(recipe, args.build, args.port, couple=k, arm=name,
                                     position=position, ident=ident, out_dir=args.out)
                    pair[name] = rec
                    flag = "ok" if rec["valid"] else f"INVALID: {rec['invalid_reason']}"
                    print(f"  c{k:02d} {name:<3} pos{position} "
                          f"{rec['aggregate_tok_s'] if rec['aggregate_tok_s'] else float('nan'):>8.3f} tok/s  "
                          f"{rec['wall_s']:>6.1f}s  {flag}")
                    if not rec["valid"]:
                        invalid_events += 1

                if pair["off"]["valid"] and pair["on"]["valid"]:
                    off_runs.append(pair["off"]["aggregate_tok_s"])
                    on_runs.append(pair["on"]["aggregate_tok_s"])
                    signs.append(1 if on_runs[-1] > off_runs[-1] else -1)
                    look = sign_verdict(signs)
                    if look["look"] is not None and len(signs) in (6, 10):
                        print(f"    claim (i) LOOK {look['look']} at {len(signs)} couples: "
                              f"{look['verdict']} ({look['detail']}) — evaluated, "
                              f"collection continues")
                else:
                    print(f"    couple {k} REPLACED (screen-driven, blind to the rate)")

                write_json(args.out / "state.json", {
                    "schema": "epyc.r2358.state.v1",
                    "couples_launched": couples_launched, "valid_couples": len(signs),
                    "signs": signs, "replacements_used": couples_launched - len(signs),
                    "invalid_launch_events": invalid_events,
                    "claim_i_look": sign_verdict(signs),
                    "elapsed_s": round(time.time() - started, 1)})

                if invalid_events > MAX_INVALID_LAUNCH_EVENTS:
                    print(f"\nINADMISSIBLE: {invalid_events} invalid launches exceeds the "
                          f"registered budget of {MAX_INVALID_LAUNCH_EVENTS}. The window "
                          f"was not clean enough to measure in.")
                    return 69
    except ControlFailure as exc:
        print(f"\nFATAL — POSITIVE CONTROL: {exc}")
        return 67
    except ControlsDisagree as exc:
        print(f"\nSTOP — THE CONTROLS DISAGREE, AND THAT IS THE FINDING: {exc}")
        return 68
    except claim.ClaimRefused as exc:
        print(f"\nINADMISSIBLE — device claim: {exc}")
        return 69

    # ---- the stop rule. Verdict computed ONCE, here, and never past it. ----
    if len(signs) < args.couples:
        print(f"\nINADMISSIBLE: {len(signs)} valid couples after {couples_launched} "
              f"launched; the registered plan needs {args.couples}.")
        return 69

    write_json(args.out / "arm-off.json", arm_record("off", off, off_runs))
    write_json(args.out / "arm-on.json", arm_record("on", on, on_runs))

    t = sign_verdict(signs)
    d = dispersion_verdict(off_runs, on_runs)
    cell = (t["verdict"] if t["verdict"] != "PENDING" else "T0", d["verdict"])
    off_sp, on_sp = serving._spread(off_runs), serving._spread(on_runs)
    verdict = {
        "schema": "epyc.r2358.verdict.v1",
        "registration": str(args.out / "PREREGISTRATION.md"),
        "champion_commit": CHAMPION_COMMIT, "build": ident,
        "n_valid_couples": len(signs), "couples_launched": couples_launched,
        "claim_i_level": t, "claim_ii_dispersion": d,
        "cell": f"{cell[0]}/{cell[1]}", "action": VERDICT_ACTIONS[cell],
        "arm_off": off_sp, "arm_on": on_sp,
        "window_comparability": {
            "standing_floor_pct": STANDING_FLOOR_PCT,
            "off_arm_p95_dev_pct": off_sp["p95_dev_pct"],
            "note": "descriptive, NOT a gate: the standing floor was calibrated on "
                    "bff30cebe under a different recipe_hash (no env, no readback)."},
        "achieved_power_note": "at n=24 per arm the primary p95_dev permutation test has "
                               "~0.97 power against a 3x sd ratio and ~0.69 against 2x "
                               "(coarse simulation, PREREGISTRATION.md 3.3). A null is "
                               "BOUNDED by those, never unqualified.",
        "elapsed_s": round(time.time() - started, 1),
    }
    if cell == ("T-", "D+"):
        verdict["hard_cell"] = hard_cell_rule(off_runs, on_runs)

    write_json(args.out / "VERDICT.json", verdict)
    (args.out / "VERDICT.md").write_text(
        f"# R23-58 VERDICT — {cell[0]}/{cell[1]}\n\n"
        f"Registered plan: {args.couples} valid couples, alpha {ALPHA}. "
        f"Achieved {len(signs)} valid of {couples_launched} launched.\n\n"
        f"* **Claim (i) LEVEL**: `{t['verdict']}` — {t['detail']} "
        f"(two-look sign boundary, exact two-sided alpha 0.0430). Direction only; "
        f"no magnitude may be quoted.\n"
        f"* **Claim (ii) DISPERSION**: `{d['verdict']}` — p95_dev ratio OFF/ON = "
        f"{d['rho_p95']:.3f}, p = {d['p_p95']:.4f}; sd(log) ratio = {d['rho_sd_log']:.3f}, "
        f"p = {d['p_sd_log']:.4f}.\n"
        f"* OFF arm: median {off_sp['median']:.3f} tok/s, p95_dev "
        f"{off_sp['p95_dev_pct']}%, cv {off_sp['cv_pct']}%\n"
        f"* ON  arm: median {on_sp['median']:.3f} tok/s, p95_dev "
        f"{on_sp['p95_dev_pct']}%, cv {on_sp['cv_pct']}%\n\n"
        f"## Registered action\n\n{VERDICT_ACTIONS[cell]}\n"
        + (f"\n### T-/D+ hard-cell rule\n\n{verdict['hard_cell']['action']}\n"
           if "hard_cell" in verdict else "")
        + "\n*This runner does not promote, fold, or write loop-memory decision state. "
          "Recalibrating the serving floor under ON is a separate, separately-authorised "
          "run.*\n")

    print(f"\n=== VERDICT {cell[0]}/{cell[1]} ===")
    print(f"  claim (i)  LEVEL      : {t['verdict']}  ({t['detail']})")
    print(f"  claim (ii) DISPERSION : {d['verdict']}  rho_p95={d['rho_p95']:.3f} "
          f"p={d['p_p95']:.4f}   [sd: rho={d['rho_sd_log']:.3f} p={d['p_sd_log']:.4f}]")
    print(f"  OFF p95_dev {off_sp['p95_dev_pct']}%  |  ON p95_dev {on_sp['p95_dev_pct']}%")
    print(f"  ACTION: {VERDICT_ACTIONS[cell]}")
    if "hard_cell" in verdict:
        print(f"  T-/D+ : {verdict['hard_cell']['action']}")
    print(f"  written: {args.out/'VERDICT.json'}, {args.out/'VERDICT.md'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
