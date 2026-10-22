#!/usr/bin/env python3
"""SERVING-FLOOR REGISTRY keyed by execution EQUIVALENCE (inference-backlog audit 2026-10-21, change E).

WHY. A matched serving floor (`serving.calibrate_floor`, schema `epyc.autokernel.serving_floor.v2`)
is filed under the anchor's `execution_digest`, which hashes the executable AND every DSO. Two
byte-equivalent trees -- `dd3d9721c` and `06b169e27` (= dd3d9721c plus one tests-only commit) --
therefore never share a floor, so the same A/A (24 pairs = 48 launches, quiet host) was booked twice
on the same day. A floor is a property of the measured CONDITION, not of a tree's commit id
(MEASUREMENT.md Annex K / P-AK-SEARCH-1-A3/A4): this registry keys it by what the condition is.

KEY. (equivalence_key, recipe_hash, request_digest, model_sha, [frame_digest, pairs]).
`equivalence_key` = sha256 over {source_tree_hash, cmake_recipe_digest, compiler}:
  * `source_tree_hash` -- the git tree of the build's commit WITH tests/docs paths removed
    (any path component `tests`/`docs`, or a `.md`/`.rst` file). A tests-only commit maps to the
    same hash. Known limit: the root CMakeLists/build flags are NOT excluded, so a change there
    is a different key (conservative).
  * `cmake_recipe_digest` -- the build's CMakeCache GGML_*/LLAMA_*/compiler/flags entries, path-
    valued and tests-only entries dropped, type tags stripped.
  * `compiler` -- the `built with ...` line of the store build's VERSION.txt.
The exact DSO/executable shas and the anchor commit are RECORDED (never keyed): the floor row's own
`baseline_resolved_recipe` is authoritative, SHA256SUMS the fallback.

ERA RULES (MEASUREMENT.md A4: "the serving floor recomputes under the adopted recipe"; "a source-only
anchor change may carry it, a runtime-recipe change may not"). A registered floor is NEVER served
  * across a CHAMPION ADVANCE: the entry stores the champion's `source_tree_hash`; lookup resolves the
    current `ak/champion/...` ref and refuses (`stale_champion`) when the trees differ. A tests-only
    advance is not an advance (same hash). This is exactly the 2026-10-20 Q38FN case: the champion
    moved b0ba1d427 -> dd3d9721c but the campaign still anchors b0ba1d427 -- `status`/`check`
    report it, nothing silently reuses it;
  * across a NUMERICS CHANGE: `numerics_epoch` is an opaque label the caller states (e.g. the W5.1a
    FP8-QDQ default flipping); it must be EQUAL. Champion unresolvable = refuse (fail closed).

GRADE. Every entry carries {threads, co_tenancy: quiet|co-tenant, claim_class: decision|indicative}.
`decision` REQUIRES `quiet` (register refuses the combination otherwise). `lookup(require_grade=
"decision")` only ever returns decision entries; an indicative floor (e.g. a co-tenant 24-thread
re-anchor floor) is retrievable with `require_grade="indicative"` and can never satisfy a decision.

ONE RECALIBRATION PER (champion advance x condition): `check_launch` is the standalone launcher hook.
It answers `reuse` (a fresh registered floor exists) or `calibrate`, and on `calibrate` ATOMICALLY
CLAIMS the condition (O_EXCL claim file) so a second launcher that arrives while the first is
calibrating gets `claimed` + the owner instead of burning another 48 launches. It touches no AK code
path (AKX-AUTO-6, the re-anchor/relaunch watchdog, is a separate unchecked task; this hook is the
floor half only).

stdlib ONLY and no relative imports: the module is loadable by file path from root launchers.
Records are plain JSON (`entries/<id>.json` immutable, `index.jsonl` append-only) so a belief adapter
can project them; the floor row is embedded verbatim and sealed by `floor_sha256`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

ENTRY_SCHEMA = "epyc.autokernel.floor_registry_entry.v1"
CLAIM_SCHEMA = "epyc.autokernel.floor_registry_claim.v1"
EQUIVALENCE_SCHEMA = "epyc.autokernel.build_equivalence.v1"
FLOOR_SCHEMAS = ("epyc.autokernel.serving_floor.v1", "epyc.autokernel.serving_floor.v2")
MATCHED_SCHEMA = "epyc.autokernel.serving_floor.v2"

DEFAULT_ROOT = Path(os.environ.get("FLOOR_REGISTRY_ROOT", "/mnt/raid0/llm/autokernel/floor-registry"))
DEFAULT_SOURCE_REPO = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_CHAMPION_BRANCH = "ak/champion/llama-cpp-ffc1bac82eec"

#: Path components / suffixes that never enter the source-tree hash.
EXCLUDED_COMPONENTS = frozenset({"tests", "docs"})
EXCLUDED_SUFFIXES = (".md", ".rst")
#: CMakeCache entries that define the build's behaviour (paths and tests-only knobs removed).
_CMAKE_KEEP = re.compile(
    r"^(CMAKE_BUILD_TYPE|CMAKE_(C|CXX)_COMPILER|CMAKE_(C|CXX)_FLAGS(_RELEASE)?|"
    r"CMAKE_(EXE|SHARED)_LINKER_FLAGS|BUILD_SHARED_LIBS|GGML_[A-Z0-9_]+|LLAMA_[A-Z0-9_]+)$")
_CMAKE_DROP = re.compile(r"(_DIR|_PATH|_ROOT|_DIRECTORY|BUILD_TESTS|CURL_|RPATH)")

GRADES = ("decision", "indicative")
CO_TENANCY = ("quiet", "co-tenant")

STATUS_HIT = "hit"
STATUS_MISS = "miss"
STATUS_STALE_CHAMPION = "stale_champion"
STATUS_STALE_NUMERICS = "stale_numerics"
STATUS_GRADE = "grade_insufficient"
STATUS_UNRESOLVABLE = "champion_unresolvable"
STATUS_CORRUPT = "corrupt"

#: exit codes of `check-launch` (the launcher hook)
EXIT_REUSE, EXIT_ERROR, EXIT_CALIBRATE, EXIT_CLAIMED = 0, 3, 10, 11


class FloorRegistryError(ValueError):
    """A registry input that cannot be admitted. Always names what is wrong."""


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode()).hexdigest()


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _write_atomic(path: Path, data: Mapping, *, exclusive: bool = False) -> None:
    """Whole-file atomic write. `exclusive` refuses to replace (claims, immutable entries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    try:
        if exclusive:
            os.link(tmp, path)  # fails with FileExistsError if present: no TOCTOU
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- equivalence
def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise FloorRegistryError(f"git {' '.join(args[:2])} failed in {repo}: {done.stderr.strip()[:300]}")
    return done.stdout.strip()


def excluded_path(path: str) -> bool:
    parts = path.split("/")
    return (any(part in EXCLUDED_COMPONENTS for part in parts[:-1])
            or path.endswith(EXCLUDED_SUFFIXES))


def source_tree_hash(repo: Path | str, commit: str) -> str:
    """sha256 over the (path, mode, blob) triples of `commit`'s tree minus tests/docs paths."""
    done = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit],
                          capture_output=True, timeout=300)
    if done.returncode != 0:
        raise FloorRegistryError(f"cannot list tree of {commit} in {repo}: {done.stderr.decode()[:200]}")
    rows = []
    for raw in done.stdout.split(b"\0"):
        if not raw:
            continue
        meta, _, path = raw.decode("utf-8", "surrogateescape").partition("\t")
        mode, kind, blob = meta.split()
        if excluded_path(path):
            continue
        rows.append((path, mode, kind, blob))
    if not rows:
        raise FloorRegistryError(f"commit {commit} has an empty tree after tests/docs exclusion")
    return digest(sorted(rows))


def cmake_recipe(build_dir: Path | str) -> dict[str, str]:
    """The behaviour-defining CMakeCache entries of a build, type tags stripped."""
    cache = Path(build_dir) / "CMakeCache.txt"
    if not cache.is_file():
        raise FloorRegistryError(f"{cache} missing: the build's CMake recipe cannot be keyed")
    out: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        name_type, _, value = line.partition("=")
        name = name_type.split(":", 1)[0]
        if _CMAKE_KEEP.match(name) and not _CMAKE_DROP.search(name):
            out[name] = value
    if not out:
        raise FloorRegistryError(f"{cache} carries no GGML_/LLAMA_/compiler entries")
    return dict(sorted(out.items()))


def _version_info(build_dir: Path) -> tuple[str, str]:
    """(short commit, compiler line) from a store build's VERSION.txt: `10345 (dd3d9721c)`."""
    path = build_dir / "VERSION.txt"
    if not path.is_file():
        raise FloorRegistryError(f"{path} missing: pass commit= explicitly")
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\(([0-9a-f]{7,40})\)", text)
    built = re.search(r"^built with .*$", text, re.M)
    if not match or not built:
        raise FloorRegistryError(f"{path} names no commit or no compiler line")
    return match.group(1), built.group(0).strip()


def _sha256sums(build_dir: Path) -> dict[str, str]:
    path = build_dir / "SHA256SUMS"
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            sha, _, name = line.partition("  ")
            if re.fullmatch(r"[0-9a-f]{64}", sha) and (
                    re.search(r"(^|/)lib[^/]*\.so", name) or name.endswith("/llama-server")):
                out[name.removeprefix("./")] = sha
    return dict(sorted(out.items()))


def equivalence_for_build(build_dir: Path | str, *, source_repo: Path | str = DEFAULT_SOURCE_REPO,
                          commit: str | None = None) -> dict:
    """The execution-equivalence record of a build: key + the facts it was derived from."""
    build = Path(build_dir)
    short, compiler = _version_info(build)
    commit = commit or short
    full = _git(Path(source_repo), "rev-parse", "--verify", f"{commit}^{{commit}}")
    tree = source_tree_hash(source_repo, full)
    recipe = cmake_recipe(build)
    facts = {"source_tree_hash": tree, "cmake_recipe_digest": digest(recipe), "compiler": compiler}
    return {"schema": EQUIVALENCE_SCHEMA, "equivalence_key": digest(facts), **facts,
            "commit": full, "build_dir": str(build), "cmake_recipe": recipe,
            "file_shas": _sha256sums(build)}


def champion_state(repo: Path | str = DEFAULT_SOURCE_REPO,
                   branch: str = DEFAULT_CHAMPION_BRANCH) -> dict:
    """The CURRENT champion ref's commit and tests/docs-excluded tree hash (read only)."""
    commit = _git(Path(repo), "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}")
    return {"branch": branch, "commit": commit, "source_tree_hash": source_tree_hash(repo, commit)}


def is_store_build(build_dir: Path | str) -> bool:
    """A kernel-store build image: VERSION.txt + SHA256SUMS + PROVENANCE.md and no provenance.json."""
    b = Path(build_dir)
    return (all((b / n).is_file() for n in ("VERSION.txt", "SHA256SUMS", "PROVENANCE.md"))
            and not (b / "provenance.json").exists())


def store_build_commit_is_descendant(build_dir: Path | str, worktree: Path | str, head: str) -> str:
    """Attestation for a kernel-STORE build (no provenance.json): its VERSION.txt commit must be
    ancestor-or-equal of the champion head. Returns the full commit; raises otherwise."""
    short, _ = _version_info(Path(build_dir))
    full = _git(Path(worktree), "rev-parse", "--verify", f"{short}^{{commit}}")
    done = subprocess.run(["git", "-C", str(worktree), "merge-base", "--is-ancestor", full, head],
                          capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise FloorRegistryError(f"store build {build_dir} commit {full[:12]} is not an ancestor-or-"
                                 f"equal of champion head {head[:12]}")
    return full


# ----------------------------------------------------------------------------- floor facts
def _unseal_check(floor: Mapping) -> None:
    """A v2 floor is sealed: content_sha256 over the row minus {content_sha256, conditions}
    (mirrors `serving._digest` / `_validate_matched_floor`)."""
    body = {k: v for k, v in floor.items() if k not in {"content_sha256", "conditions"}}
    if floor.get("content_sha256") != digest(body):
        raise FloorRegistryError("matched floor does not re-derive its own content_sha256")


def floor_facts(floor: Mapping, *, model_sha: str | None = None) -> dict:
    schema = floor.get("schema")
    if schema not in FLOOR_SCHEMAS:
        raise FloorRegistryError(f"unregistrable floor schema {schema!r}")
    if floor.get("unit") != "process":
        raise FloorRegistryError(f"floor unit {floor.get('unit')!r}: only process-unit floors gate "
                                 f"the loop's between-process effects (R23-55)")
    if type(floor.get("n")) is not int or floor["n"] < 1:
        raise FloorRegistryError("floor states no n (R23-61)")
    for key in ("recipe_hash", "request_digest", "floor_pct"):
        if floor.get(key) is None:
            raise FloorRegistryError(f"floor lacks {key}")
    facts: dict[str, Any] = {"recipe_hash": floor["recipe_hash"],
                             "request_digest": floor["request_digest"],
                             "recipe": floor.get("recipe"), "floor_pct": floor["floor_pct"],
                             "n": floor["n"], "schema": schema,
                             "pairs": floor.get("comparison_pairs"), "frame_digest": None,
                             "model_sha": model_sha, "threads": None, "dsos": {}, "executable": None,
                             "execution_digest": None}
    if schema == MATCHED_SCHEMA:
        _unseal_check(floor)
        frame = floor.get("frame")
        if not isinstance(frame, dict) or not isinstance(frame.get("model"), dict):
            raise FloorRegistryError("matched floor carries no frame.model")
        facts["frame_digest"] = digest(frame)
        facts["model_sha"] = frame["model"].get("sha256")
        base = floor.get("baseline_resolved_recipe") or {}
        facts["execution_digest"] = base.get("execution_digest")
        facts["dsos"] = {Path(d["path"]).name: d["sha256"] for d in base.get("dsos", ())}
        facts["executable"] = (base.get("executable") or {}).get("sha256")
        argv = base.get("argv") or base.get("command_argv") or []
        if "-t" in argv and argv.index("-t") + 1 < len(argv) and argv[argv.index("-t") + 1].isdigit():
            facts["threads"] = int(argv[argv.index("-t") + 1])
    if not facts["model_sha"]:
        raise FloorRegistryError("floor names no model sha (matched floors carry frame.model.sha256; "
                                 "a v1 floor needs model_sha=)")
    residency = floor.get("residency")
    end = residency.get("window_end") if isinstance(residency, dict) else None
    facts["calibrated_at"] = float(end) if type(end) in (int, float) and end > 0 else None
    return facts


def validate_grade(grade: Mapping, *, floor_threads: int | None = None) -> dict:
    out = {"threads": grade.get("threads"), "co_tenancy": grade.get("co_tenancy"),
           "claim_class": grade.get("claim_class")}
    if type(out["threads"]) is not int or out["threads"] < 1:
        raise FloorRegistryError("grade.threads must be a positive int")
    if out["co_tenancy"] not in CO_TENANCY:
        raise FloorRegistryError(f"grade.co_tenancy must be one of {CO_TENANCY}")
    if out["claim_class"] not in GRADES:
        raise FloorRegistryError(f"grade.claim_class must be one of {GRADES}")
    if out["claim_class"] == "decision" and out["co_tenancy"] != "quiet":
        raise FloorRegistryError("a decision-grade floor requires a QUIET host: a co-tenant floor "
                                 "is indicative only")
    if floor_threads is not None and floor_threads != out["threads"]:
        raise FloorRegistryError(f"grade.threads {out['threads']} != the floor's launch -t "
                                 f"{floor_threads}")
    return out


# ----------------------------------------------------------------------------- registry
@dataclass
class LookupResult:
    status: str
    floor: dict | None = None
    entry: dict | None = None
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.floor is not None


class FloorRegistry:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    # ---- write
    def register(self, floor: Mapping, *, equivalence: Mapping, grade: Mapping,
                 numerics_epoch: str, model_sha: str | None = None,
                 registered_by: str = "unknown",
                 validator: Callable[[Mapping], None] | None = None) -> Path:
        """Admit one calibrated floor. Idempotent on identical content; entries are immutable.

        `validator` is the AK-side frame check (`serving._validate_matched_floor` bound to its
        recipe/requests/pairs) -- the registry stays stdlib-only and cannot run it itself.
        """
        if not isinstance(numerics_epoch, str) or not numerics_epoch:
            raise FloorRegistryError("numerics_epoch must be a non-empty label")
        if equivalence.get("schema") != EQUIVALENCE_SCHEMA or not equivalence.get("equivalence_key"):
            raise FloorRegistryError("equivalence must come from equivalence_for_build")
        facts = floor_facts(floor, model_sha=model_sha)
        if validator is not None:
            validator(floor)
        clean = validate_grade(grade, floor_threads=facts["threads"])
        now = _now()
        floor_doc = json.loads(_canon(floor))  # detached, JSON-clean copy
        entry = {
            "schema": ENTRY_SCHEMA, "registered_at": _iso(now), "registered_ts": now,
            "registered_by": registered_by,
            "key": {"equivalence_key": equivalence["equivalence_key"],
                    "recipe_hash": facts["recipe_hash"], "request_digest": facts["request_digest"],
                    "model_sha": facts["model_sha"], "frame_digest": facts["frame_digest"],
                    "pairs": facts["pairs"]},
            "era": {"champion_commit": equivalence["commit"],
                    "champion_source_tree_hash": equivalence["source_tree_hash"],
                    "numerics_epoch": numerics_epoch},
            "grade": clean,
            "anchor": {"commit": equivalence["commit"], "build_dir": equivalence["build_dir"],
                       "execution_digest": facts["execution_digest"],
                       "executable_sha256": facts["executable"],
                       "dso_sha256": facts["dsos"], "file_shas": equivalence.get("file_shas", {})},
            "equivalence": {k: equivalence[k] for k in
                            ("source_tree_hash", "cmake_recipe_digest", "compiler", "cmake_recipe")},
            "summary": {"recipe": facts["recipe"], "floor_pct": facts["floor_pct"], "n": facts["n"],
                        "unit": "process", "schema": facts["schema"],
                        "calibrated_at": facts["calibrated_at"]},
            "floor": floor_doc, "floor_sha256": digest(floor_doc)}
        entry_id = digest({"key": entry["key"], "era": entry["era"], "grade": clean,
                           "floor_sha256": entry["floor_sha256"]})[:32]
        path = self.root / "entries" / f"{entry_id}.json"
        try:
            _write_atomic(path, entry, exclusive=True)
        except FileExistsError:
            return path  # same content already registered
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as idx:
            idx.write(_canon({"id": entry_id, "registered_at": entry["registered_at"],
                              "key": entry["key"], "era": entry["era"], "grade": clean,
                              "floor_pct": facts["floor_pct"], "n": facts["n"]}) + "\n")
        self.release_claim_for(entry, note=f"registered {entry_id}")
        return path

    # ---- read
    def entries(self) -> list[dict]:
        out = []
        for path in sorted((self.root / "entries").glob("*.json")):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            entry["_path"] = str(path)
            out.append(entry)
        return out

    @staticmethod
    def _intact(entry: Mapping) -> bool:
        return (entry.get("schema") == ENTRY_SCHEMA
                and entry.get("floor_sha256") == digest(entry.get("floor")))

    def lookup_detail(self, *, equivalence_key: str, recipe_hash: str, request_digest: str,
                      model_sha: str, numerics_epoch: str, require_grade: str = "decision",
                      frame_digest: str | None = None, pairs: int | None = None,
                      threads: int | None = None,
                      champion: Mapping | None = None,
                      check_champion: bool = True) -> LookupResult:
        """`champion` = `champion_state(...)`. With `check_champion` (the default) a lookup that
        cannot name the current champion REFUSES -- an unresolvable era is not a fresh one."""
        if require_grade not in GRADES:
            raise FloorRegistryError(f"require_grade must be one of {GRADES}")
        want = {"equivalence_key": equivalence_key, "recipe_hash": recipe_hash,
                "request_digest": request_digest, "model_sha": model_sha}
        if frame_digest is not None:
            want["frame_digest"] = frame_digest
        if pairs is not None:
            want["pairs"] = pairs
        keyed = [e for e in self.entries() if all(e.get("key", {}).get(k) == v for k, v in want.items())]
        if not keyed:
            return LookupResult(STATUS_MISS, reasons=["no entry for this key"])
        sound = [e for e in keyed if self._intact(e)]
        if not sound:
            return LookupResult(STATUS_CORRUPT, reasons=[f"{len(keyed)} keyed entr(ies) fail their "
                                                         f"floor_sha256 seal"])
        reasons: list[str] = []
        live = [e for e in sound if e["era"]["numerics_epoch"] == numerics_epoch]
        if not live:
            have = sorted({e["era"]["numerics_epoch"] for e in sound})
            return LookupResult(STATUS_STALE_NUMERICS, reasons=[
                f"registered under numerics_epoch {have}, caller is {numerics_epoch!r}: a floor is "
                f"never reused across a numerics change (MEASUREMENT.md A4)"])
        if check_champion:
            if champion is None:
                return LookupResult(STATUS_UNRESOLVABLE, reasons=["current champion not supplied"])
            fresh = [e for e in live
                     if e["era"]["champion_source_tree_hash"] == champion.get("source_tree_hash")]
            if not fresh:
                anchors = sorted({e["era"]["champion_commit"][:12] for e in live})
                return LookupResult(STATUS_STALE_CHAMPION, reasons=[
                    f"anchored on {anchors}, current champion {champion.get('branch')} is "
                    f"{str(champion.get('commit'))[:12]} (different source tree): a floor is never "
                    f"reused across a champion advance -- recalibrate once"])
            live = fresh
        rank = {"decision": 1, "indicative": 0}
        graded = [e for e in live if rank[e["grade"]["claim_class"]] >= rank[require_grade]]
        if threads is not None:
            graded = [e for e in graded if e["grade"]["threads"] == threads]
        if not graded:
            have = sorted({(e["grade"]["claim_class"], e["grade"]["threads"], e["grade"]["co_tenancy"])
                           for e in live})
            return LookupResult(STATUS_GRADE, reasons=[
                f"need {require_grade}" + (f" threads={threads}" if threads else "")
                + f"; registered: {have}"])
        # newest only -- never shop among floors for a wider or narrower bar (cf. DS41-C69)
        best = max(graded, key=lambda e: (e["summary"].get("calibrated_at") or 0, e["registered_ts"]))
        return LookupResult(STATUS_HIT, floor=json.loads(_canon(best["floor"])), entry=best)

    def lookup(self, **kwargs) -> dict | None:
        """The floor row or None (see `lookup_detail` for the reason a miss happened)."""
        return self.lookup_detail(**kwargs).floor

    def status(self, champion: Mapping) -> list[dict]:
        """Every entry with its freshness against the CURRENT champion (the Q38FN flag)."""
        rows = []
        for e in self.entries():
            fresh = e.get("era", {}).get("champion_source_tree_hash") == champion.get("source_tree_hash")
            rows.append({"id": Path(e["_path"]).stem, "recipe": e["summary"].get("recipe"),
                         "floor_pct": e["summary"]["floor_pct"], "n": e["summary"]["n"],
                         "grade": e["grade"], "anchor": e["era"]["champion_commit"][:12],
                         "numerics_epoch": e["era"]["numerics_epoch"],
                         "state": "fresh" if fresh else "STALE_CHAMPION_ADVANCED",
                         "intact": self._intact(e)})
        return rows

    # ---- one recalibration per (champion advance x condition)
    def condition_key(self, *, champion_tree_hash: str, equivalence_key: str, recipe_hash: str,
                      request_digest: str, model_sha: str, pairs: int | None, numerics_epoch: str,
                      grade: str) -> str:
        return digest({"champion_tree": champion_tree_hash, "equivalence_key": equivalence_key,
                       "recipe_hash": recipe_hash, "request_digest": request_digest,
                       "model_sha": model_sha, "pairs": pairs, "numerics_epoch": numerics_epoch,
                       "claim_class": grade})[:32]

    def claim_calibration(self, condition: str, *, owner: str, note: str = "") -> dict:
        """Atomically take the right to calibrate `condition`. {granted, claim}; a held claim is
        reported, never stolen (release is explicit: `release_claim` or registering the floor)."""
        path = self.root / "claims" / f"{condition}.json"
        claim = {"schema": CLAIM_SCHEMA, "condition": condition, "owner": owner, "pid": os.getpid(),
                 "claimed_at": _iso(_now()), "claimed_ts": _now(), "note": note}
        try:
            _write_atomic(path, claim, exclusive=True)
            return {"granted": True, "claim": claim}
        except FileExistsError:
            held = json.loads(path.read_text(encoding="utf-8"))
            return {"granted": False, "claim": held,
                    "age_s": round(_now() - held.get("claimed_ts", _now()))}

    def claim_holder(self, condition: str) -> dict | None:
        """Read-only: the live claim for `condition`, or None (never creates or steals)."""
        path = self.root / "claims" / f"{condition}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def release_claim(self, condition: str) -> bool:
        path = self.root / "claims" / f"{condition}.json"
        if path.exists():
            path.rename(path.with_suffix(f".released-{int(_now())}"))
            return True
        return False

    def release_claim_for(self, entry: Mapping, *, note: str = "") -> None:
        """Registering a floor settles every claim of its condition (any champion tree matching)."""
        k, era, g = entry["key"], entry["era"], entry["grade"]
        for grade in {g["claim_class"], "indicative"}:
            self.release_claim(self.condition_key(
                champion_tree_hash=era["champion_source_tree_hash"],
                equivalence_key=k["equivalence_key"], recipe_hash=k["recipe_hash"],
                request_digest=k["request_digest"], model_sha=k["model_sha"], pairs=k["pairs"],
                numerics_epoch=era["numerics_epoch"], grade=grade))


def check_launch(registry: FloorRegistry, *, equivalence: Mapping, recipe_hash: str,
                 request_digest: str, model_sha: str, pairs: int | None, numerics_epoch: str,
                 require_grade: str, champion: Mapping, owner: str,
                 frame_digest: str | None = None, threads: int | None = None) -> dict:
    """THE LAUNCHER HOOK. `{action: reuse|calibrate|claimed, ...}`.

    reuse     -- a fresh registered floor satisfies this condition; run NO A/A. `floor_path` names it.
    calibrate -- none does and this caller now HOLDS the condition's claim: run the A/A once, then
                 `register` (which releases the claim).
    claimed   -- another launcher already holds the claim for this exact condition; wait for its
                 floor (poll `lookup`), do not calibrate.
    """
    result = registry.lookup_detail(
        equivalence_key=equivalence["equivalence_key"], recipe_hash=recipe_hash,
        request_digest=request_digest, model_sha=model_sha, numerics_epoch=numerics_epoch,
        require_grade=require_grade, frame_digest=frame_digest, pairs=pairs, threads=threads,
        champion=champion)
    if result:
        return {"action": "reuse", "status": result.status, "floor_path": result.entry["_path"],
                "floor_pct": result.floor["floor_pct"], "n": result.floor["n"],
                "grade": result.entry["grade"]}
    if result.status in (STATUS_CORRUPT,):
        return {"action": "error", "status": result.status, "reasons": result.reasons}
    if equivalence.get("source_tree_hash") != champion.get("source_tree_hash"):
        # The caller's own anchor is not the current champion: calibrating it would only file a
        # floor that is stale at birth (the Q38FN b0ba1d427 case). Re-anchor first.
        return {"action": "error", "status": STATUS_STALE_CHAMPION, "reasons": [
            f"anchor build {equivalence.get('commit', '')[:12]} is not the current champion "
            f"{str(champion.get('commit'))[:12]}: re-anchor (--new-anchor-epoch) before calibrating"]}
    condition = registry.condition_key(
        champion_tree_hash=champion["source_tree_hash"], equivalence_key=equivalence["equivalence_key"],
        recipe_hash=recipe_hash, request_digest=request_digest, model_sha=model_sha, pairs=pairs,
        numerics_epoch=numerics_epoch, grade=require_grade)
    taken = registry.claim_calibration(condition, owner=owner,
                                       note=f"miss status={result.status}: {'; '.join(result.reasons)}")
    return {"action": "calibrate" if taken["granted"] else "claimed", "status": result.status,
            "reasons": result.reasons, "condition": condition, "claim": taken["claim"]}


# ----------------------------------------------------------------------------- CLI
def _grade_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--co-tenancy", choices=CO_TENANCY, required=True)
    p.add_argument("--claim-class", choices=GRADES, required=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floor_registry", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    p.add_argument("--champion-branch", default=DEFAULT_CHAMPION_BRANCH)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("equivalence", help="print a build's equivalence record")
    s.add_argument("build", type=Path)
    s = sub.add_parser("register", help="register a calibrated floor json")
    s.add_argument("floor", type=Path)
    s.add_argument("--build", type=Path, required=True, help="the build the A/A ran on")
    s.add_argument("--numerics-epoch", required=True)
    s.add_argument("--model-sha")
    s.add_argument("--by", default=os.environ.get("USER", "unknown"))
    _grade_args(s)
    for name, helptext in (("lookup", "print the registered floor path, or the miss reason"),
                           ("check-launch", "launcher hook: reuse | calibrate (claim) | claimed")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--build", type=Path, required=True)
        s.add_argument("--floor", type=Path, help="any floor json of the recipe/requests/model "
                       "(recipe_hash, request_digest, model_sha, pairs, frame are read from it)")
        s.add_argument("--recipe-hash"); s.add_argument("--request-digest"); s.add_argument("--model-sha")
        s.add_argument("--pairs", type=int)
        s.add_argument("--numerics-epoch", required=True)
        s.add_argument("--require-grade", choices=GRADES, default="decision")
        s.add_argument("--threads", type=int)
        s.add_argument("--owner", default=os.environ.get("USER", "unknown"))
        s.add_argument("--write-floor", type=Path, help="on a hit, write the registered floor json here")
    s = sub.add_parser("status", help="every entry vs the CURRENT champion (flags advanced anchors)")
    s = sub.add_parser("release", help="release a held calibration claim")
    s.add_argument("condition")
    return p


def _identity(args) -> dict:
    ident = {"recipe_hash": args.recipe_hash, "request_digest": args.request_digest,
             "model_sha": args.model_sha, "pairs": args.pairs, "frame_digest": None}
    if args.floor:
        facts = floor_facts(json.loads(args.floor.read_text(encoding="utf-8")), model_sha=args.model_sha)
        ident.update({k: facts[k] for k in ("recipe_hash", "request_digest", "model_sha", "frame_digest")})
        ident["pairs"] = args.pairs if args.pairs is not None else facts["pairs"]
    missing = [k for k in ("recipe_hash", "request_digest", "model_sha") if not ident[k]]
    if missing:
        raise FloorRegistryError(f"identity incomplete: pass --floor or --{missing[0].replace('_', '-')}")
    return ident


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reg = FloorRegistry(args.root)
    try:
        if args.cmd == "equivalence":
            print(json.dumps(equivalence_for_build(args.build, source_repo=args.source_repo), indent=1))
            return 0
        if args.cmd == "status":
            champ = champion_state(args.source_repo, args.champion_branch)
            rows = reg.status(champ)
            print(f"champion {champ['branch']} = {champ['commit'][:12]}")
            for row in rows:
                print(json.dumps(row, sort_keys=True))
            return EXIT_ERROR if any(r["state"] != "fresh" or not r["intact"] for r in rows) else 0
        if args.cmd == "release":
            print("released" if reg.release_claim(args.condition) else "no such claim")
            return 0
        equiv = equivalence_for_build(args.build, source_repo=args.source_repo)
        if args.cmd == "register":
            floor = json.loads(args.floor.read_text(encoding="utf-8"))
            path = reg.register(floor, equivalence=equiv, numerics_epoch=args.numerics_epoch,
                                model_sha=args.model_sha, registered_by=args.by,
                                grade={"threads": args.threads, "co_tenancy": args.co_tenancy,
                                       "claim_class": args.claim_class})
            print(path)
            return 0
        ident = _identity(args)
        champ = champion_state(args.source_repo, args.champion_branch)
        if args.cmd == "lookup":
            res = reg.lookup_detail(equivalence_key=equiv["equivalence_key"], numerics_epoch=args.numerics_epoch,
                                    require_grade=args.require_grade, threads=args.threads,
                                    champion=champ, **ident)
            print(res.entry["_path"] if res else f"{res.status}: {'; '.join(res.reasons)}")
            if res and args.write_floor:
                _write_atomic(args.write_floor, res.floor)
            return 0 if res else 1
        out = check_launch(reg, equivalence=equiv, numerics_epoch=args.numerics_epoch,
                           require_grade=args.require_grade, champion=champ, owner=args.owner,
                           threads=args.threads, **ident)
        if out["action"] == "reuse" and args.write_floor:
            _write_atomic(args.write_floor, json.loads(Path(out["floor_path"]).read_text())["floor"])
        print(json.dumps(out, indent=1, sort_keys=True))
        return {"reuse": EXIT_REUSE, "calibrate": EXIT_CALIBRATE, "claimed": EXIT_CLAIMED}.get(
            out["action"], EXIT_ERROR)
    except FloorRegistryError as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
