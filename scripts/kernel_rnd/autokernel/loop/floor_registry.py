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

STORAGE (operator decision 2026-10-22): the MASTER model registry (research
`orchestration/model_registry.yaml`, top-level `serving_floors:` keyed by role) is the ONLY floor store.
Readers read only it -- no pending file, no fallback. This module is READ-ONLY over it: a calibration
seals its record under durable disk (campaign store), uses the fresh floor in-process, and
`plan_registration` emits an entry + the applier command; the owning session adds the entry in a normal
commit through `apply_registry_serving_floors.py` (comment-preserving, round-trip proven) from a clean
worktree of origin/main. Only the in-flight calibration CLAIMS (locks, not floors) live outside it.

stdlib + yaml and no relative imports: the module is loadable by file path (root shim, the registry
validator).
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

#: The research repo THIS FILE lives in (scripts/kernel_rnd/autokernel/loop/floor_registry.py -> repo
#: root). The default master is that repo's registry, so a run from a pinned worktree reads the
#: registry of EXACTLY that commit and never a shared clone that may be behind (operator 2026-10-22).
CODE_REPO = Path(__file__).resolve().parents[4]
DEFAULT_MASTER = CODE_REPO / "orchestration" / "model_registry.yaml"
#: `--floor-registry @code` / `--master @code` spell the default explicitly (serial_run forwards
#: flag values verbatim and cannot carry a value-less flag).
CODE_REPO_TOKEN = "@code"
DEFAULT_CLAIMS = Path(os.environ.get("FLOOR_REGISTRY_CLAIMS", "/mnt/raid0/llm/autokernel/floor-claims"))
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
#: Entries the tree declares deprecated and ignores (llama.cpp CMakeLists.txt:170
#: `llama_option_depr(WARNING LLAMA_CURL)`): a store build passes -DLLAMA_CURL=OFF, an in-tree build
#: does not, and the two are the same executable. Measured 2026-10-22 on dd3d9721c vs 06b169e27.
_CMAKE_DEPRECATED_NOOP = frozenset({"LLAMA_CURL"})

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
def is_scratch(path: str) -> bool:
    """The registry's documented scratch root (check_evidence_durability): never citable evidence."""
    return path.startswith("/mnt/raid0/llm/tmp/") or "/llm/tmp/" in path


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
        if (_CMAKE_KEEP.match(name) and not _CMAKE_DROP.search(name)
                and name not in _CMAKE_DEPRECATED_NOOP):
            out[name] = value
    if not out:
        raise FloorRegistryError(f"{cache} carries no GGML_/LLAMA_/compiler entries")
    return dict(sorted(out.items()))


def _compiler(build_dir: Path, version_text: str | None) -> str:
    """`<ID> <version>` (e.g. `GNU 15.2.0`) -- from VERSION.txt's `built with` line when the build is a
    store image, else from CMake's own compiler record, so an in-tree build compares equal to a store one."""
    if version_text:
        match = re.search(r"^built with (\S+) (\S+)", version_text, re.M)
        if match:
            return f"{match.group(1)} {match.group(2)}"
    for path in sorted(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")):
        text = path.read_text(encoding="utf-8", errors="replace")
        cid = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)', text)
        ver = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)', text)
        if cid and ver:
            return f"{cid.group(1)} {ver.group(1)}"
    raise FloorRegistryError(f"{build_dir}: no compiler identity (VERSION.txt `built with` or "
                             f"CMakeFiles/*/CMakeCXXCompiler.cmake)")


def _version_info(build_dir: Path, commit: str | None = None) -> tuple[str, str]:
    """(commit, compiler). A store build names its commit in VERSION.txt (`10345 (dd3d9721c)`); an
    in-tree build has no VERSION.txt, so the caller must pass `commit`."""
    path = build_dir / "VERSION.txt"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    if commit is None:
        match = re.search(r"\(([0-9a-f]{7,40})\)", text or "")
        if not match:
            raise FloorRegistryError(f"{path} missing or names no commit: pass commit= explicitly")
        commit = match.group(1)
    return commit, _compiler(build_dir, text)


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
    commit, compiler = _version_info(build, commit)
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


# ----------------------------------------------------------------------------- which registry, and is it stale
class RegistryBehindUpstream(FloorRegistryError):
    """The local registry lacks floors that origin/main has: refusing is the only safe answer."""


def resolve_master(explicit: Path | str | None = None) -> tuple[Path, str]:
    """(path, source). Order: explicit arg (`--master`/`--floor-registry`) > `FLOOR_REGISTRY_MASTER` >
    the registry of the repo this code runs from. Any override is LOUD (stderr): an overridden
    registry is the one way a run can read a different commit's floors than its own code."""
    if explicit is not None and str(explicit) != CODE_REPO_TOKEN:
        path, source = Path(explicit), "explicit"
    elif explicit is None and os.environ.get("FLOOR_REGISTRY_MASTER"):
        path, source = Path(os.environ["FLOOR_REGISTRY_MASTER"]), "env:FLOOR_REGISTRY_MASTER"
    else:
        return DEFAULT_MASTER, "code-repo"
    try:
        same = path.resolve() == DEFAULT_MASTER.resolve()
    except OSError:
        same = False
    if not same:
        print(f"WARNING   floor registry master OVERRIDDEN ({source}): {path} is NOT the registry of the "
              f"code repo ({DEFAULT_MASTER}); floors are read from a different file than this code's own "
              f"commit carries", file=sys.stderr)
    return path, source


def _git_out(repo: Path, *args: str) -> str | None:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    return done.stdout.strip() if done.returncode == 0 else None


def registry_provenance(master: Path | str) -> dict:
    """Where the registry file came from: path, repo, the commit it is checked out at, whether the file
    has uncommitted edits, and origin/main as THIS repo last saw it (no fetch is performed)."""
    master = Path(master)
    out: dict[str, Any] = {"path": str(master), "git_root": None, "commit": None, "dirty": None,
                           "origin_main_commit": None, "origin_main_age_s": None}
    top = _git_out(master.parent, "rev-parse", "--show-toplevel")
    if not top:
        return out
    out["git_root"] = top
    out["commit"] = _git_out(Path(top), "rev-parse", "HEAD")
    rel = os.path.relpath(master.resolve(), Path(top).resolve())
    status = _git_out(Path(top), "status", "--porcelain", "--", rel)
    out["dirty"] = bool(status) if status is not None else None
    out["origin_main_commit"] = _git_out(Path(top), "rev-parse", "--verify", "origin/main^{commit}")
    stamp = _git_out(Path(top), "log", "-1", "--format=%ct", "origin/main")
    if stamp and stamp.isdigit():
        out["origin_main_age_s"] = round(time.time() - int(stamp))
    return out


def upstream_check(master: Path | str, *, role: str | None, model_sha: str | None,
                   show: Callable[[Path, str], str | None] | None = None) -> dict:
    """Compare the LOCAL registry's serving_floors with origin/main's for this target.

    `behind`  -- origin/main carries entries (matching `role`, or `model_sha`) whose ids the local file
                 lacks: the local registry is behind and a lookup miss/hit would be silently wrong.
    `ok`      -- every upstream entry for the target is present locally.
    `skipped` -- origin/main cannot be read (no remote / ref); the caller WARNS, never refuses.
    `show` is the seam for tests: (repo_root, relpath) -> origin/main's file text or None."""
    master = Path(master)
    top = _git_out(master.parent, "rev-parse", "--show-toplevel")
    if not top:
        return {"status": "skipped", "reason": f"{master} is not inside a git repository"}
    rel = os.path.relpath(master.resolve(), Path(top).resolve())
    text = (show or (lambda repo, r: _git_out(repo, "show", f"origin/main:{r}")))(Path(top), rel)
    if text is None:
        return {"status": "skipped", "reason": f"origin/main:{rel} is not readable in {top} "
                                                f"(no origin remote or no origin/main ref)"}
    upstream = _section_from_text(text)
    local = load_section(master)

    def matching(section: Mapping) -> dict[str, dict]:
        out = {}
        for r, rows in section.items():
            for e in rows:
                if isinstance(e, dict) and (r == role or (model_sha and e.get("model_sha") == model_sha)):
                    out[f"{r}/{e.get('id')}"] = e
        return out

    up, here = matching(upstream), matching(local)
    missing = sorted(set(up) - set(here))
    return {"status": "behind" if missing else "ok", "upstream_entries": sorted(up),
            "local_entries": sorted(here), "missing_locally": missing}


def preflight_or_refuse(master: Path | str, *, role: str | None, model_sha: str | None,
                        log: Callable[[str], None] = print,
                        show: Callable[[Path, str], str | None] | None = None) -> dict:
    """The AK launch gate: refuse (RegistryBehindUpstream) if origin/main has floors for this target
    that the local registry lacks; WARN and continue only when origin/main cannot be read."""
    result = upstream_check(master, role=role, model_sha=model_sha, show=show)
    if result["status"] == "behind":
        raise RegistryBehindUpstream(
            f"registry at {master} is behind origin/main; floor for {role or model_sha} exists upstream "
            f"({', '.join(result['missing_locally'])}). Update/rebase this repo (or run from a worktree "
            f"of origin/main) -- refusing rather than calibrating or gating on a stale registry.")
    if result["status"] == "skipped":
        log(f"WARNING   floor registry upstream check SKIPPED: {result['reason']}")
    else:
        log(f"serving   floor registry upstream check ok ({len(result['local_entries'])} local / "
            f"{len(result['upstream_entries'])} upstream entries for {role or model_sha})")
    return result


# ----------------------------------------------------------------------------- master registry I/O
SECTION = "serving_floors"
ENTRY_FIELDS = ("id", "recipe", "recipe_hash", "frame_digest", "request_digest", "model_sha",
                "equivalence_key", "numerics_epoch", "champion_tree_hash", "build", "schema",
                "floor_pct", "unit", "n", "k", "interval", "grade", "measured_at", "record",
                "record_sha256", "content_sha256", "annotation")


def repo_root_of(master_path: Path | str) -> Path:
    """The research repo root that relative `record:` paths resolve against."""
    return Path(master_path).resolve().parent.parent


def section_slice(lines: list[str]) -> tuple[int, int] | None:
    """[start, end) of the top-level `serving_floors:` block (end = next unindented non-comment line)."""
    start = next((i for i, ln in enumerate(lines) if ln == f"{SECTION}:"), None)
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i].startswith((" ", "#")):
            end = i
            break
    return start, end


def _section_from_text(text: str, origin: str = "registry") -> dict[str, list[dict]]:
    import yaml
    lines = text.split("\n")
    span = section_slice(lines)
    if span is None:
        return {}
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    body = yaml.load("\n".join(lines[span[0]:span[1]]), Loader=loader) or {}
    section = body.get(SECTION) or {}
    if not isinstance(section, dict) or any(not isinstance(v, list) for v in section.values()):
        raise FloorRegistryError(f"{origin}: `{SECTION}` must map role -> list of entries")
    return section


def load_section(master_path: Path | str) -> dict[str, list[dict]]:
    """role -> entries, read from ONLY the `serving_floors:` slice of the master (no 13k-line parse)."""
    return _section_from_text(Path(master_path).read_text(encoding="utf-8"), str(master_path))


def _iso_z(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _iso(float(value))
    text = str(value)
    if text.endswith("+00:00"):
        text = text[:-6] + "Z"
    return text.split(".")[0].rstrip("Z") + "Z" if "T" in text else text


def _epoch(value: Any) -> float:
    import calendar
    try:
        return float(calendar.timegm(time.strptime(_iso_z(value).rstrip("Z"), "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return 0.0


def plan_registration(floor: Mapping, *, floor_file: Path | str, role: str, equivalence: Mapping,
                      grade: Mapping, numerics_epoch: str, annotation: str = "",
                      model_sha: str | None = None, registered_by: str = "unknown") -> dict:
    """The REGISTRY ENTRY for a calibrated floor plus the applier invocation. Writes NOTHING: the
    master registry is changed only by `apply_registry_serving_floors.py`, run by the owning session
    from a clean worktree on origin/main in a normal commit. `floor_file` is the sealed record already
    on durable disk (the campaign store); it is cited in place (never copied into git) and its sha pinned."""
    if not role or not isinstance(role, str):
        raise FloorRegistryError("a registry role/key is required")
    if not isinstance(numerics_epoch, str) or not numerics_epoch:
        raise FloorRegistryError("numerics_epoch must be a non-empty label")
    if equivalence.get("schema") != EQUIVALENCE_SCHEMA or not equivalence.get("equivalence_key"):
        raise FloorRegistryError("equivalence must come from equivalence_for_build")
    raw = Path(floor_file).read_bytes()
    if json.loads(raw) != json.loads(json.dumps(floor)):
        raise FloorRegistryError("floor_file does not contain the floor being registered")
    facts = floor_facts(floor, model_sha=model_sha)
    clean = validate_grade(grade, floor_threads=facts["threads"])
    base = floor.get("baseline_resolved_recipe") or {}
    measured = (floor.get("conditions") or {}).get("recorded_at") or facts.get("calibrated_at") or _now()
    interval = floor.get("interval") or {}
    if is_scratch(str(Path(floor_file).resolve())):
        raise FloorRegistryError(f"{floor_file} is in scratch: seal the record under durable disk "
                                 f"(campaign store) before registering it")
    entry = {
        "id": str(floor.get("content_sha256") or hashlib.sha256(raw).hexdigest())[:16],
        "recipe": facts["recipe"], "recipe_hash": facts["recipe_hash"],
        "frame_digest": facts["frame_digest"], "request_digest": facts["request_digest"],
        "model_sha": facts["model_sha"], "equivalence_key": equivalence["equivalence_key"],
        "numerics_epoch": numerics_epoch, "champion_tree_hash": equivalence["source_tree_hash"],
        "build": {"commit": equivalence["commit"], "build_dir": equivalence["build_dir"],
                  "execution_digest": facts["execution_digest"], "executable_sha256": facts["executable"],
                  "dso_sha256": dict(sorted(facts["dsos"].items()))},
        "schema": facts["schema"], "floor_pct": facts["floor_pct"], "unit": "process", "n": facts["n"],
        "k": facts["pairs"],
        "interval": {"level": interval.get("level"), "low_pct": interval.get("low_pct"),
                     "high_pct": interval.get("high_pct")},
        "grade": clean, "measured_at": _iso_z(measured),
        # The record stays WHERE IT IS (durable, local, untracked): a matched floor carries its full
        # residency series (the DS41-1017 one is 336 MB) and raw measurement material reaches git only
        # as distilled knowledge (operator ruling 2026-08-03), so it is cited by absolute path.
        "record": str(Path(floor_file).resolve()),
        "record_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": floor.get("content_sha256"), "annotation": annotation,
    }
    plan = {"schema": "epyc.autokernel.floor_registry_plan.v1", "role": role, "entry": entry,
            "record_src": str(Path(floor_file).resolve()),
            "registered_by": registered_by}
    return plan


def applier_command(plan_path: Path | str, master: Path | str | None = None) -> str:
    return (f"python3 -m autokernel.loop.apply_registry_serving_floors --registry <CLEAN-WORKTREE>/"
            f"orchestration/model_registry.yaml --plan {plan_path}   # from scripts/kernel_rnd, on a clean "
            f"worktree of origin/main; then commit; canonical master: {master or DEFAULT_MASTER}")


def _load_record(master_path: Path, entry: Mapping) -> dict:
    """The entry's sealed record, verified against its pinned sha and against the entry's own claims."""
    path = repo_root_of(master_path) / str(entry.get("record", ""))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FloorRegistryError(f"record {entry.get('record')!r} unreadable: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.get("record_sha256"):
        raise FloorRegistryError(f"record {path} does not match its pinned record_sha256")
    floor = json.loads(raw)
    facts = floor_facts(floor)
    for mine, theirs in (("recipe_hash", "recipe_hash"), ("request_digest", "request_digest"),
                         ("floor_pct", "floor_pct"), ("n", "n"), ("frame_digest", "frame_digest"),
                         ("model_sha", "model_sha")):
        if facts[mine] != entry.get(theirs):
            raise FloorRegistryError(f"record {path}: {mine} {facts[mine]!r} != entry {entry.get(theirs)!r}")
    if floor.get("content_sha256") != entry.get("content_sha256"):
        raise FloorRegistryError(f"record {path}: content_sha256 differs from the entry's")
    return floor


# ----------------------------------------------------------------------------- registry
@dataclass
class LookupResult:
    status: str
    floor: dict | None = None
    entry: dict | None = None
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.floor is not None


def _normalize(role: str, e: Mapping) -> dict:
    return {"_role": role, "_id": e.get("id"), "_path": e.get("record"),
            "key": {"equivalence_key": e.get("equivalence_key"), "recipe_hash": e.get("recipe_hash"),
                    "request_digest": e.get("request_digest"), "model_sha": e.get("model_sha"),
                    "frame_digest": e.get("frame_digest"), "pairs": e.get("k")},
            "era": {"champion_commit": (e.get("build") or {}).get("commit", ""),
                    "champion_source_tree_hash": e.get("champion_tree_hash"),
                    "numerics_epoch": e.get("numerics_epoch")},
            "grade": e.get("grade") or {},
            "summary": {"recipe": e.get("recipe"), "floor_pct": e.get("floor_pct"), "n": e.get("n"),
                        "calibrated_at": _epoch(e.get("measured_at"))},
            "registered_ts": _epoch(e.get("measured_at")), "raw": dict(e)}


class FloorRegistry:
    """READ-ONLY view of the master registry's `serving_floors` section, plus the in-flight
    calibration CLAIMS (locks, not floors). It never writes a floor: see `plan_registration`."""

    def __init__(self, master: Path | str | None = None, claims_root: Path | str = DEFAULT_CLAIMS,
                 log: Callable[[str], None] | None = None):
        # None -> the registry of the repo THIS code runs from (see resolve_master); an override
        # (argument or FLOOR_REGISTRY_MASTER) is logged loudly there.
        self.master, self.master_source = resolve_master(master)
        self.claims_root = Path(claims_root)
        self.log = log if log is not None else (lambda m: print(m, file=sys.stderr))
        self.lookups: list[dict] = []   # every lookup this object served, for the run receipt
        self._provenance: dict | None = None

    def provenance(self) -> dict:
        if self._provenance is None:
            self._provenance = {**registry_provenance(self.master), "source": self.master_source}
        return self._provenance

    # ---- read
    def entries(self, role: str | None = None) -> list[dict]:
        out = []
        for r, rows in load_section(self.master).items():
            if role is None or r == role:
                out.extend(_normalize(r, e) for e in rows if isinstance(e, dict))
        return out

    def lookup_detail(self, **kwargs) -> LookupResult:
        """`_lookup` plus the audit trail: EVERY lookup is logged (registry path, git commit, the matched
        entry or the miss reason) and kept in `self.lookups` for the run receipt."""
        result = self._lookup(**kwargs)
        prov = self.provenance()
        entry = result.entry or {}
        rec = {"at": _iso(_now()), "registry": prov["path"], "registry_source": prov["source"],
               "registry_commit": prov["commit"], "registry_dirty": prov["dirty"],
               "role": kwargs.get("role"), "numerics_epoch": kwargs.get("numerics_epoch"),
               "require_grade": kwargs.get("require_grade", "decision"), "status": result.status,
               "entry_id": entry.get("_id"), "entry_role": entry.get("_role"),
               "floor_pct": (result.floor or {}).get("floor_pct"), "reasons": list(result.reasons)}
        self.lookups.append(rec)
        self.log(f"serving   floor registry lookup: {result.status}"
                 + (f" entry {rec['entry_role']}/{rec['entry_id']} floor {rec['floor_pct']}%"
                    if result else f" ({'; '.join(result.reasons)})")
                 + f" | registry {prov['path']} @ {str(prov['commit'])[:12]}"
                 + (" DIRTY" if prov["dirty"] else "") + f" [{prov['source']}]")
        return result

    def receipt(self, limit: int = 20) -> dict:
        """The block AK writes into its serving-gate records/status (`floor_registry` key)."""
        return {"schema": "epyc.autokernel.floor_registry_receipt.v1", **self.provenance(),
                "lookups": self.lookups[-limit:], "lookup_count": len(self.lookups)}

    def _lookup(self, *, equivalence_key: str, recipe_hash: str, request_digest: str,
                model_sha: str, numerics_epoch: str, require_grade: str = "decision",
                frame_digest: str | None = None, pairs: int | None = None,
                threads: int | None = None, role: str | None = None,
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
        keyed = [e for e in self.entries(role) if all(e["key"].get(k) == v for k, v in want.items())]
        if not keyed:
            return LookupResult(STATUS_MISS, reasons=["no entry for this key"])
        reasons: list[str] = []
        live = [e for e in keyed if e["era"]["numerics_epoch"] == numerics_epoch]
        if not live:
            have = sorted({str(e["era"]["numerics_epoch"]) for e in keyed})
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
        graded = [e for e in live if rank.get(e["grade"].get("claim_class"), -1) >= rank[require_grade]]
        if threads is not None:
            graded = [e for e in graded if e["grade"].get("threads") == threads]
        if not graded:
            have = sorted({(str(e["grade"].get("claim_class")), e["grade"].get("threads"),
                            str(e["grade"].get("co_tenancy"))) for e in live})
            return LookupResult(STATUS_GRADE, reasons=[
                f"need {require_grade}" + (f" threads={threads}" if threads else "")
                + f"; registered: {have}"])
        # newest only -- never shop among floors for a wider or narrower bar (cf. DS41-C69)
        best = max(graded, key=lambda e: (e["summary"].get("calibrated_at") or 0, e["registered_ts"]))
        try:
            floor = _load_record(self.master, best["raw"])
        except FloorRegistryError as exc:
            return LookupResult(STATUS_CORRUPT, entry=best, reasons=[str(exc)])
        best["_path"] = str(repo_root_of(self.master) / best["raw"]["record"])
        return LookupResult(STATUS_HIT, floor=floor, entry=best)

    def lookup(self, **kwargs) -> dict | None:
        """The floor row or None (see `lookup_detail` for the reason a miss happened)."""
        return self.lookup_detail(**kwargs).floor

    def status(self, champion: Mapping) -> list[dict]:
        """Every entry with its freshness against the CURRENT champion (the Q38FN flag)."""
        rows = []
        for e in self.entries():
            fresh = e["era"]["champion_source_tree_hash"] == champion.get("source_tree_hash")
            try:
                _load_record(self.master, e["raw"])
                intact = True
            except FloorRegistryError:
                intact = False
            rows.append({"role": e["_role"], "id": e["_id"], "recipe": e["summary"]["recipe"],
                         "floor_pct": e["summary"]["floor_pct"], "n": e["summary"]["n"],
                         "grade": e["grade"], "anchor": e["era"]["champion_commit"][:12],
                         "numerics_epoch": e["era"]["numerics_epoch"],
                         "state": "fresh" if fresh else "STALE_CHAMPION_ADVANCED", "intact": intact})
        return rows

    # ---- one recalibration per (champion advance x condition): claims are locks, not floors
    def condition_key(self, *, champion_tree_hash: str, equivalence_key: str, recipe_hash: str,
                      request_digest: str, model_sha: str, pairs: int | None, numerics_epoch: str,
                      grade: str) -> str:
        return digest({"champion_tree": champion_tree_hash, "equivalence_key": equivalence_key,
                       "recipe_hash": recipe_hash, "request_digest": request_digest,
                       "model_sha": model_sha, "pairs": pairs, "numerics_epoch": numerics_epoch,
                       "claim_class": grade})[:32]

    def claim_calibration(self, condition: str, *, owner: str, note: str = "") -> dict:
        """Atomically take the right to calibrate `condition`. {granted, claim}; a held claim is
        reported, never stolen (release is explicit: `release_claim`)."""
        path = self.claims_root / f"{condition}.json"
        claim = {"schema": CLAIM_SCHEMA, "condition": condition, "owner": owner, "pid": os.getpid(),
                 "claimed_at": _iso(_now()), "claimed_ts": _now(), "note": note, "state": "calibrating"}
        try:
            _write_atomic(path, claim, exclusive=True)
            return {"granted": True, "claim": claim}
        except FileExistsError:
            held = json.loads(path.read_text(encoding="utf-8"))
            return {"granted": False, "claim": held,
                    "age_s": round(_now() - held.get("claimed_ts", _now()))}

    def claim_holder(self, condition: str) -> dict | None:
        """Read-only: the live claim for `condition`, or None (never creates or steals)."""
        try:
            return json.loads((self.claims_root / f"{condition}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def mark_awaiting_registration(self, condition: str, plan_path: Path | str) -> bool:
        """The holder finished calibrating and has emitted its registry plan; the claim now says so
        (so a waiter knows a floor EXISTS and is only waiting for the owner's commit)."""
        held = self.claim_holder(condition)
        if held is None:
            return False
        held.update({"state": "awaiting_registration", "plan": str(plan_path),
                     "calibrated_at": _iso(_now())})
        _write_atomic(self.claims_root / f"{condition}.json", held)
        return True

    def release_claim(self, condition: str) -> bool:
        path = self.claims_root / f"{condition}.json"
        if path.exists():
            path.rename(path.with_suffix(f".released-{int(_now())}"))
            return True
        return False


def check_launch(registry: FloorRegistry, *, equivalence: Mapping, recipe_hash: str,
                 request_digest: str, model_sha: str, pairs: int | None, numerics_epoch: str,
                 require_grade: str, champion: Mapping, owner: str,
                 frame_digest: str | None = None, threads: int | None = None,
                 role: str | None = None) -> dict:
    """THE LAUNCHER HOOK. `{action: reuse|calibrate|claimed, ...}`.

    reuse     -- a fresh floor in the MASTER registry satisfies this condition; run NO A/A.
                 `floor_path` names its sealed record.
    calibrate -- none does and this caller now HOLDS the condition's claim: run the A/A once, use the
                 floor in-process, then emit the registry plan (`plan_registration`) and
                 `mark_awaiting_registration`; the owning session commits it.
    claimed   -- another launcher already holds the claim for this exact condition (its `state` says
                 calibrating or awaiting_registration); wait for the master to carry its floor, do
                 not calibrate.
    """
    result = registry.lookup_detail(
        equivalence_key=equivalence["equivalence_key"], recipe_hash=recipe_hash,
        request_digest=request_digest, model_sha=model_sha, numerics_epoch=numerics_epoch,
        require_grade=require_grade, frame_digest=frame_digest, pairs=pairs, threads=threads,
        role=role, champion=champion)
    condition = registry.condition_key(
        champion_tree_hash=champion["source_tree_hash"], equivalence_key=equivalence["equivalence_key"],
        recipe_hash=recipe_hash, request_digest=request_digest, model_sha=model_sha, pairs=pairs,
        numerics_epoch=numerics_epoch, grade=require_grade)
    if result:
        registry.release_claim(condition)  # the floor is in the master: any awaiting claim is settled
        return {"action": "reuse", "status": result.status, "floor_path": result.entry["_path"],
                "role": result.entry["_role"], "floor_pct": result.floor["floor_pct"],
                "n": result.floor["n"], "grade": result.entry["grade"]}
    if result.status in (STATUS_CORRUPT,):
        return {"action": "error", "status": result.status, "reasons": result.reasons}
    if equivalence.get("source_tree_hash") != champion.get("source_tree_hash"):
        # The caller's own anchor is not the current champion: calibrating it would only file a
        # floor that is stale at birth (the Q38FN b0ba1d427 case). Re-anchor first.
        return {"action": "error", "status": STATUS_STALE_CHAMPION, "reasons": [
            f"anchor build {equivalence.get('commit', '')[:12]} is not the current champion "
            f"{str(champion.get('commit'))[:12]}: re-anchor (--new-anchor-epoch) before calibrating"]}
    taken = registry.claim_calibration(condition, owner=owner,
                                       note=f"miss status={result.status}: {'; '.join(result.reasons)}")
    return {"action": "calibrate" if taken["granted"] else "claimed", "status": result.status,
            "reasons": result.reasons, "condition": condition, "claim": taken["claim"]}


# ----------------------------------------------------------------------------- validator rules
def validate_section(master_data: Mapping, repo_root: Path | str) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the `serving_floors` section of a parsed master registry.
    Called by `scripts/validate_model_registry.py`."""
    errors: list[str] = []
    warnings: list[str] = []
    section = master_data.get(SECTION)
    if section is None:
        return errors, warnings
    if not isinstance(section, dict):
        return [f"{SECTION}: must map role -> list of entries"], warnings
    roles = master_data.get("roles") or {}
    seen: set[tuple] = set()
    for role, rows in section.items():
        if not isinstance(rows, list):
            errors.append(f"{SECTION}.{role}: must be a list")
            continue
        if role not in roles:
            warnings.append(f"{SECTION}.{role}: not a roles.* id (a campaign key with no served role is "
                            f"allowed for catalogue measurements; bind it when the model is served)")
        elif roles[role].get("deprecated"):
            warnings.append(f"{SECTION}.{role}: role is deprecated; its floors are history")
        for i, e in enumerate(rows):
            where = f"{SECTION}.{role}[{i}]"
            if not isinstance(e, dict):
                errors.append(f"{where}: not a mapping")
                continue
            missing = [f for f in ENTRY_FIELDS if f not in e]
            if missing:
                errors.append(f"{where}: missing {missing}")
                continue
            if e["unit"] != "process":
                errors.append(f"{where}: unit {e['unit']!r} (only between-process floors gate; R23-55)")
            if type(e["n"]) is not int or (e["schema"] == MATCHED_SCHEMA and e["n"] < 24):
                errors.append(f"{where}: n {e['n']!r} (matched floors need an int n >= 24)")
            g = e["grade"] if isinstance(e["grade"], dict) else {}
            if (g.get("claim_class") not in GRADES or g.get("co_tenancy") not in CO_TENANCY
                    or type(g.get("threads")) is not int):
                errors.append(f"{where}: grade needs claim_class {GRADES}, co_tenancy {CO_TENANCY}, int threads")
            elif g["claim_class"] == "decision" and g["co_tenancy"] != "quiet":
                errors.append(f"{where}: decision-grade requires co_tenancy quiet")
            iv = e["interval"] if isinstance(e["interval"], dict) else {}
            if not (type(iv.get("low_pct")) in (int, float) and type(iv.get("high_pct")) in (int, float)
                    and iv["low_pct"] <= iv["high_pct"]):
                errors.append(f"{where}: interval needs low_pct <= high_pct")
            ident = (role, e["equivalence_key"], e["recipe_hash"], e["request_digest"], e["k"],
                     e["numerics_epoch"], g.get("claim_class"), e["measured_at"])
            if ident in seen:
                errors.append(f"{where}: duplicate of an earlier entry")
            seen.add(ident)
            pinned = ((roles.get(role) or {}).get("model") or {}).get("sha256")
            if pinned and pinned != e["model_sha"] and not (
                    isinstance(pinned, str) and e["model_sha"] in pinned):
                errors.append(f"{where}: model_sha differs from roles.{role}.model.sha256")
            try:
                _load_record(Path(repo_root) / "orchestration" / "model_registry.yaml", e)
            except (FloorRegistryError, ValueError) as exc:
                errors.append(f"{where}: {exc}")
            rec = str(e["record"])
            if is_scratch(rec):
                errors.append(f"{where}: record is in scratch ({rec})")
    return errors, warnings


# ----------------------------------------------------------------------------- CLI
def _grade_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--co-tenancy", choices=CO_TENANCY, required=True)
    p.add_argument("--claim-class", choices=GRADES, required=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floor_registry", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", type=Path, default=None,
                   help="the MASTER model registry (the only floor store). Default: the registry of the "
                        f"repo this code runs from ({DEFAULT_MASTER}); an override is logged loudly. "
                        f"`{CODE_REPO_TOKEN}` spells the default.")
    p.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS, help="in-flight calibration claim locks")
    p.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    p.add_argument("--champion-branch", default=DEFAULT_CHAMPION_BRANCH)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("equivalence", help="print a build's equivalence record")
    s.add_argument("build", type=Path)
    s.add_argument("--commit", help="the build's commit (required for an in-tree build with no VERSION.txt)")
    s = sub.add_parser("register", help="PLAN the registry entry for a calibrated floor (writes only "
                       "the plan file; the master is changed by apply_registry_serving_floors)")
    s.add_argument("floor", type=Path, help="the sealed floor json, already on durable disk")
    s.add_argument("--build", type=Path, required=True, help="the build the A/A ran on")
    s.add_argument("--commit", help="the build's commit (in-tree build with no VERSION.txt)")
    s.add_argument("--role", required=True, help="registry role id / campaign key to file it under")
    s.add_argument("--numerics-epoch", required=True)
    s.add_argument("--model-sha")
    s.add_argument("--annotation", default="")
    s.add_argument("--plan-out", type=Path, help="default: <floor>.registry-plan.json")
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
        s.add_argument("--role")
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
    reg = FloorRegistry(args.master, args.claims)
    try:
        if args.cmd == "equivalence":
            print(json.dumps(equivalence_for_build(args.build, source_repo=args.source_repo,
                                                         commit=args.commit), indent=1))
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
        equiv = equivalence_for_build(args.build, source_repo=args.source_repo,
                                      commit=getattr(args, "commit", None))
        if args.cmd == "register":
            floor = json.loads(args.floor.read_text(encoding="utf-8"))
            plan = plan_registration(
                floor, floor_file=args.floor, role=args.role, equivalence=equiv,
                numerics_epoch=args.numerics_epoch, model_sha=args.model_sha,
                annotation=args.annotation, registered_by=args.by,
                grade={"threads": args.threads, "co_tenancy": args.co_tenancy,
                       "claim_class": args.claim_class})
            out_path = args.plan_out or args.floor.with_name(args.floor.name + ".registry-plan.json")
            _write_atomic(out_path, plan)
            print(f"plan written: {out_path}\nnext (owning session, clean origin/main worktree):\n  "
                  f"{applier_command(out_path, args.master)}")
            return 0
        ident = _identity(args)
        champ = champion_state(args.source_repo, args.champion_branch)
        if args.cmd == "lookup":
            preflight_or_refuse(reg.master, role=args.role, model_sha=ident["model_sha"],
                                log=lambda m: print(m, file=sys.stderr))
            res = reg.lookup_detail(equivalence_key=equiv["equivalence_key"], numerics_epoch=args.numerics_epoch,
                                    require_grade=args.require_grade, threads=args.threads,
                                    role=args.role, champion=champ, **ident)
            print(res.entry["_path"] if res else f"{res.status}: {'; '.join(res.reasons)}")
            if res and args.write_floor:
                _write_atomic(args.write_floor, res.floor)
            return 0 if res else 1
        preflight_or_refuse(reg.master, role=args.role, model_sha=ident["model_sha"],
                            log=lambda m: print(m, file=sys.stderr))
        out = check_launch(reg, equivalence=equiv, numerics_epoch=args.numerics_epoch,
                           require_grade=args.require_grade, champion=champ, owner=args.owner,
                           threads=args.threads, role=args.role, **ident)
        if out["action"] == "reuse" and args.write_floor:
            _write_atomic(args.write_floor, json.loads(Path(out["floor_path"]).read_text()))
        print(json.dumps(out, indent=1, sort_keys=True))
        return {"reuse": EXIT_REUSE, "calibrate": EXIT_CALIBRATE, "claimed": EXIT_CLAIMED}.get(
            out["action"], EXIT_ERROR)
    except FloorRegistryError as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
