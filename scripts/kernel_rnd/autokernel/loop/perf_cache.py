"""Content-addressed cache for `perf report` / `perf annotate` subprocess output.

DS41-C20d: in the seat A/B (DS41-C20c) the bounded seat lost ~12 minutes of wall
time to `actor_tools_mcp`'s `profile_top` / `symbol_annotate` re-shelling out to
`perf report --stdio` / `perf annotate --stdio` over a large `perf.data` on every
single tool call -- including the internal `_dso_symbols` lookup that
`symbol_annotate` uses to resolve a short/typed symbol name (DS41-C23), which is
itself one more full `perf report` pass hidden inside what looks like one call.

Design (see the DS41-C20d task text for the acceptance bar):

* A perf.data's **identity** is its realpath + size + mtime_ns + a cheap content
  hash (first/last 1 MiB) + the `perf --version` string in effect. Any one of
  these changing means "this is not the profile you cached" -- there is no
  separate invalidation path, a changed identity simply misses and refills.
  `extra` lets a caller fold in something this module cannot see on its own
  (e.g. the profiled binary's build-id/sha256, if the caller resolved one) --
  unused by `actor_tools_mcp.py` today because this campaign's profile *path*
  already changes per binary variant (`cpu-raw-<digest(request)>`, where the
  request digest includes the build's execution digest -- see
  `cpu_profile.py:CpuProfileCapture._initialize`), so path+size+mtime already
  discriminates every binary this campaign ever profiles. A caller whose
  perf.data path is stable across rebuilds (e.g. a future orchestrator tool,
  INF-78 R4) MUST pass a real `extra`, or a rebuilt binary at the same path
  will serve annotate output disassembled against the wrong binary.

* A **call's identity** is the exact `perf` argv used, MINUS the profile path
  itself (already covered by the profile identity) -- so `(profile, dso, sort,
  percent-limit)` for a report call and `(profile, dso, symbol)` for an
  annotate call each get one cache entry, exactly the granularity the task
  text asks for ("once per (profile, dso, sort)"). `limit` is never part of
  perf's own argv (it is applied to the parsed rows *after* the call), so one
  cached report call answers every `limit` value for that `(profile, dso)` --
  this is the "structured, sorted symbol table" the design asks for: the
  table is just the cached, already-parsed row list, and `profile_top` slices
  it locklessly.

* Cache **hit path never invokes perf**, so a hit's output is not just
  "equivalent" but the literal bytes a prior real invocation produced --
  byte-identical by construction, not by re-implementing perf's own
  percent/DSO filtering.

* Storage: one JSON file per (profile identity, call signature), under a
  directory keyed by the profile's identity hash so a changed perf.data can
  never collide with an old entry. Default location is a `.actor_tools_perf_cache/`
  directory *beside* the profile's containing directory (a sibling of
  `cpu-raw-<digest>/`, never inside it -- that directory is integrity-checked
  by `CpuProfileCapture._open` via `_stat_identity`, and dropping cache files
  into it would be an unrelated write into a tamper-checked directory).
  Writes are atomic (temp file + `os.replace`) so a reader never observes a
  partial entry.

This module is plain stdlib and imports nothing from the MCP SDK or from
`actor_tools_mcp.py`, so it stays importable and unit-testable standalone, and
safe for `cpu_profile.py` (the profile producer) to import for the optional
prewarm hook without pulling in opencode/MCP machinery.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

DEFAULT_CACHE_DIRNAME = ".actor_tools_perf_cache"
FASTHASH_SAMPLE_BYTES = 1024 * 1024  # 1 MiB from each end; cheap even on a 200 MB perf.data
PREWARM_TIMEOUT_S = 300  # the loop can afford more patience here than the interactive planner


class PerfCacheError(Exception):
    """Raised only for caller misuse (e.g. a profile that does not exist)."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def _fast_hash(path: str, size: int, sample_bytes: int = FASTHASH_SAMPLE_BYTES) -> str:
    """size + first/last `sample_bytes` of the file's content.

    Catches a same-size rewrite whose mtime lands in the same second (coarse
    filesystem mtime resolution, or a tool that preserves mtime on copy) --
    path+size+mtime alone would otherwise silently serve the old content.
    """
    h = hashlib.sha256()
    h.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            h.update(f.read(sample_bytes))
            if size > sample_bytes:
                f.seek(max(0, size - sample_bytes))
                h.update(f.read(sample_bytes))
    except OSError:
        pass  # identity still discriminates on size/mtime; a hash of "" is fine
    return h.hexdigest()[:16]


def perf_version(perf_bin: str = "perf") -> str:
    """`perf --version`, or 'unknown' if it cannot be run. One subprocess call;
    callers that already know the version (tests, a caller amortising it across
    many profiles) should pass `perf_ver=` to `PerfCache`/`identify` instead."""
    try:
        proc = subprocess.run([perf_bin, "--version"], capture_output=True, text=True,
                              timeout=10, stdin=subprocess.DEVNULL, check=False)
        return (proc.stdout or proc.stderr or "unknown").strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


@dataclass(frozen=True)
class ProfileIdentity:
    """Everything that must match for a cached perf call to still be valid."""
    path: str
    size: int
    mtime_ns: int
    fasthash: str
    perf_ver: str
    extra: str = ""

    @property
    def key(self) -> str:
        raw = "|".join((self.path, str(self.size), str(self.mtime_ns), self.fasthash,
                        self.perf_ver, self.extra))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns,
                "fasthash": self.fasthash, "perf_ver": self.perf_ver, "extra": self.extra}

    def matches(self, other: Dict[str, object]) -> bool:
        return self.to_dict() == other


def identify(profile_path: str, perf_bin: str = "perf", *, perf_ver: Optional[str] = None,
            extra: str = "") -> ProfileIdentity:
    real = os.path.realpath(profile_path)
    try:
        st = os.stat(real)
    except OSError as e:
        raise PerfCacheError(f"cannot stat profile {profile_path!r}: {e}") from e
    return ProfileIdentity(path=real, size=st.st_size, mtime_ns=st.st_mtime_ns,
                           fasthash=_fast_hash(real, st.st_size),
                           perf_ver=perf_ver if perf_ver is not None else perf_version(perf_bin),
                           extra=extra)


def cache_dir_for(identity: ProfileIdentity, cache_root: Optional[str] = None) -> str:
    """Where entries for this exact profile identity live. `cache_root`, when
    given, overrides the default "beside the profile's cpu-raw-<digest>/ dir"
    placement -- e.g. a single shared cache dir under the campaign store."""
    if cache_root:
        base = os.path.realpath(cache_root)
    else:
        base = _default_cache_base(identity.path)
    return os.path.join(base, os.path.basename(identity.path) + "-" + identity.key)


def _default_cache_base(profile_real: str) -> str:
    """Mirrors `actors.py:_profile_dirs`'s "peel off cpu-raw-<digest>/" convention,
    so the cache sits beside the raw-capture directory, never inside it."""
    parent = os.path.dirname(profile_real)
    base = os.path.dirname(parent) if os.path.basename(parent).startswith("cpu-raw-") else parent
    return os.path.join(base, DEFAULT_CACHE_DIRNAME)


# ---------------------------------------------------------------------------
# Call signature (perf argv minus the profile path)
# ---------------------------------------------------------------------------
def sig_from_cmd(cmd: Sequence[str], profile_real: str) -> Tuple[str, ...]:
    """The perf argv with the profile path elided (identity already covers it),
    so the same logical call from two different tmp-relocated copies of the
    same content still hits -- though in practice the two are already the
    same identity iff content matches, so this mostly just keeps entries
    readable on disk."""
    return tuple("<profile>" if a == profile_real else a for a in cmd)


def _sig_hash(sig: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(sig).encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _atomic_write_json(path: str, body: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(body, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class PerfCache:
    """Caches `(rc, stdout, stderr)` for exact perf invocations, keyed by a
    profile's identity plus the call's own argv signature. See module
    docstring for the full design.
    """

    def __init__(self, cache_root: Optional[str] = None, perf_bin: str = "perf",
                perf_ver: Optional[str] = None):
        self.cache_root = cache_root
        self.perf_bin = perf_bin
        self._perf_ver = perf_ver
        self._mem: Dict[Tuple[str, str], Tuple[int, str, str]] = {}
        self.hits = 0
        self.misses = 0

    def _identity(self, profile_real: str, extra: str = "") -> ProfileIdentity:
        if self._perf_ver is None:
            # Resolved once per PerfCache instance (a real `perf --version` subprocess,
            # negligible cost) and reused for every profile/call this instance ever
            # sees -- never once per identify() call, which would otherwise cost one
            # extra perf subprocess per cache lookup, hit or miss.
            self._perf_ver = perf_version(self.perf_bin)
        return identify(profile_real, self.perf_bin, perf_ver=self._perf_ver, extra=extra)

    def entry_dir(self, profile_real: str, extra: str = "") -> str:
        return cache_dir_for(self._identity(profile_real, extra), self.cache_root)

    def get_or_compute(self, profile_real: str, sig: Sequence[str],
                        compute: Callable[[], Tuple[int, str, str]], *,
                        extra: str = "") -> Tuple[int, str, str]:
        """Return the cached `(rc, stdout, stderr)` for this profile+sig, computing
        (and caching) it via `compute()` on a miss. `compute` is called at most
        once per distinct (profile identity, sig); an exception from `compute`
        (e.g. the caller's own timeout) propagates uncached -- a failed/timed-out
        run is never remembered as an answer."""
        real = os.path.realpath(profile_real)
        ident = self._identity(real, extra)
        sig_key = _sig_hash(sig)
        mem_key = (ident.key, sig_key)
        if mem_key in self._mem:
            self.hits += 1
            return self._mem[mem_key]
        entry_path = os.path.join(cache_dir_for(ident, self.cache_root), sig_key + ".json")
        cached = _read_json(entry_path)
        if cached is not None and ident.matches(cached.get("identity")) and \
                cached.get("sig") == list(sig):
            result = (cached["rc"], cached["stdout"], cached["stderr"])
            self._mem[mem_key] = result
            self.hits += 1
            return result
        self.misses += 1
        result = compute()
        rc, stdout, stderr = result
        _atomic_write_json(entry_path, {"identity": ident.to_dict(), "sig": list(sig),
                                        "rc": rc, "stdout": stdout, "stderr": stderr,
                                        "cached_at": time.time()})
        self._mem[mem_key] = result
        return result

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._mem)}


# ---------------------------------------------------------------------------
# Prewarm (optional, off by default -- see cpu_profile.py's AK_PERF_CACHE_PREWARM)
# ---------------------------------------------------------------------------
def prewarm_profile(profile_path: str, *, cache_root: Optional[str] = None,
                    perf_bin: str = "perf", percent_limit: str = "0.3",
                    sort: str = "dso,symbol", timeout_s: int = PREWARM_TIMEOUT_S) -> Dict[str, object]:
    """Build the base (no-dso) `perf report` cache entry for one freshly-written
    profile, so the planner's first `profile_top()` call is already a cache hit.

    Costs one full `perf report --stdio` pass over the whole perf.data -- wall
    time roughly tracks file size/sample count (the DS41 run-7 25.7 MB / 114K
    sample profile took ~2-3 s; see DS41-C20d's measured numbers). Call this
    only between loop iterations, on the loop's own CPUs, AFTER the profile
    round has fully ended and BEFORE anything starts a new measurement window --
    never overlapping a floor/measurement capture, which the loop's own
    'idle compute is a reportable condition' + observation-window discipline
    would otherwise misattribute as noise on the CPUs perf shares with the
    server under measurement (`agents/shared/OPERATING_CONSTRAINTS.md`
    Observation Windows). This is exactly why it is gated off by default.
    """
    cache = PerfCache(cache_root=cache_root, perf_bin=perf_bin)
    real = os.path.realpath(profile_path)
    cmd = [perf_bin, "report", "--stdio", "--no-children", "--force", "-i", real,
           "--percent-limit", percent_limit, "--sort", sort]
    sig = sig_from_cmd(cmd, real)
    started = time.monotonic()

    def compute() -> Tuple[int, str, str]:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              timeout=timeout_s, stdin=subprocess.DEVNULL, check=False)
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    rc, stdout, _stderr = cache.get_or_compute(real, sig, compute)
    return {"profile": real, "rc": rc, "elapsed_s": time.monotonic() - started,
            "cache_dir": cache.entry_dir(real), "stdout_bytes": len(stdout),
            "cache_stats": cache.stats()}
