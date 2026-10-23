#!/usr/bin/env python3
"""AKX-AUTO-6: detect a CHAMPION ADVANCE, re-anchor, relaunch, and recalibrate AT MOST ONCE.

SUPPORTED INVOCATION: run from scripts/kernel_rnd, or as `kernel_rnd.autokernel.loop.champion_reanchor`
from scripts/. NOT `-m loop.champion_reanchor` from autokernel/: that makes `loop` top-level, the `..`
imports of plan/apply fail, and main() refuses it with a message.

    python3 -m autokernel.loop.champion_reanchor detect --store S --worktree W
    python3 -m autokernel.loop.champion_reanchor plan   --store S --worktree W --registry R [...]
    python3 -m autokernel.loop.champion_reanchor apply  --store S --worktree W --registry R \
        --relaunch-cmd '["/path/launcher.sh"]' --execute

OFF BY DEFAULT in every sense: nothing imports this module, `run.py`/`serial_run.py` are untouched,
and `apply` is a dry run unless `--execute` is given.

ORIGIN. 2026-10-20 18:55Z the champion advanced b0ba1d427 -> dd3d9721c; the Q38FN campaign's
`accumulator-bundle.json` / `loop-status.json` still said champion_of_record b0ba1d427 a day later,
because nothing noticed and nothing re-anchored. AKX-AUTO-6 is that noticing. The same night the
DS41-1017 floor was booked twice (06:30Z gate and 18:00Z AK window) on byte-equivalent builds; the
floor half is `floor_registry.check_launch`, called here so ONE recalibration happens per advance
(or none when a registered decision-grade floor exists).

THE FOUR STEPS
1. DETECT (read-only). The campaign's champion_of_record (store `accumulator-bundle.json`, short
   hashes resolved in the worktree's repo) is compared with the CURRENT `ak/champion/...` ref by
   SOURCE-TREE HASH with tests/docs paths excluded (`floor_registry.source_tree_hash`), so a
   tests-only advance is NOT an advance. States:
     current     same tree                                  -> nothing to do
     behind      trees differ and COR is an ancestor of the champion head -> a genuine advance
     diverged    trees differ and COR is NOT an ancestor    -> reported, never acted on
     unreadable  bundle/ref cannot be read                  -> reported, never acted on
2. PLAN (read-only). Finds the kernel-store build whose equivalence tree hash is the champion's
   (`kernels/builds/*`, newest), runs `accumulate.would_refuse_anchor` (an ancestry refusal is the
   documented trigger for `--new-anchor-epoch`; any other refusal blocks), and asks the floor
   registry what it would do (reuse / calibrate / claimed) WITHOUT claiming.
3. APPLY (`--execute`). Refuses while the campaign is live (loop-status fresh and running; it never
   kills anything -- stopping a lane is its owner's act). Calls `check_launch` (which claims the
   condition on `calibrate`), writes `<store>/reanchor/reanchor-<ts>.json` (the SPEC), and runs the
   owner-supplied `--relaunch-cmd` with `REANCHOR_SPEC=<path>`. If the relaunch fails and no floor
   was registered, the claim is released so the next launcher can calibrate.
4. The SPEC carries the exact run.py argv additions: `--anchor-build <store build>`,
   `--new-anchor-epoch <champion commit>`, `--floor-registry/--floor-numerics-epoch/
   --floor-require-grade/--floor-co-tenancy`, plus `calibrate_serving_pairs` only when the
   registry said `calibrate`.

TWO RELAUNCH PATHS. With `--resolved-campaign/--owned-targets/--alias/--target-root` (and, to validate,
`--rebound-launch`), plan/apply also emit BOTH commands, built by `serial_roster.build_targets` so the
roster's own checks run: `serial_run` (patched owned-targets + common-args; `serial_run` now accepts
`--new-anchor-epoch` and the `--floor-*` flags, forwarding the epoch to the FIRST child only) and
`direct_run_py` (the exact run.py argv, for hand launches as in RUNBOOK-ak.md s1).

WHAT THIS DELIBERATELY DOES NOT DO: regenerate the campaign's `launch.json`. A CanonicalResolvedRecipe
binds `build_dir`, executable and DSO digests, and `serial_roster` refuses a launch whose build differs
from the owner's anchor_build; producing the rebound launch is the campaign's input build
(`<campaign>/inputs/build_inputs.py`), which the owner's `--relaunch-cmd` runs. The SPEC tells that
script which build and epoch to bind.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import floor_registry as fr

SPEC_SCHEMA = "epyc.autokernel.reanchor_spec.v1"
DEFAULT_BUILDS_ROOT = Path("/mnt/raid0/llm/kernels/builds")
STATES = ("current", "behind", "diverged", "unreadable")


class ReanchorRefused(RuntimeError):
    """A condition under which no re-anchor may proceed. Always names why."""


def _git_ok(repo: Path, *args: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          timeout=300).returncode == 0


def _resolve(repo: Path, rev: str) -> str:
    return fr._git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")


def _iso_epoch(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return float(calendar.timegm(time.strptime(text.split(".")[0].rstrip("Z"), "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None


def read_campaign(store: Path | str) -> dict:
    """The campaign's own account of its champion: bundle COR, status anchor and liveness."""
    store = Path(store)
    out: dict[str, Any] = {"store": str(store), "cor": None, "status_anchor": None, "live": False,
                           "state": None, "errors": []}
    try:
        bundle = json.loads((store / "accumulator-bundle.json").read_text(encoding="utf-8"))
        out["cor"] = bundle.get("champion_of_record")
        out["tip"] = bundle.get("tip")
    except (OSError, ValueError) as exc:
        out["errors"].append(f"accumulator-bundle.json: {exc}")
    try:
        status = json.loads((store / "loop-status.json").read_text(encoding="utf-8"))
        out["status_anchor"] = status.get("anchor_commit")
        out["status_cor"] = (status.get("accumulator") or {}).get("champion_of_record")
        out["state"] = status.get("state")
        generated = _iso_epoch(status.get("generated_at"))
        stale_after = status.get("stale_after_s")
        out["live"] = bool(status.get("state") == "running" and generated is not None
                           and type(stale_after) in (int, float)
                           and time.time() - generated < stale_after)
    except (OSError, ValueError) as exc:
        out["errors"].append(f"loop-status.json: {exc}")
    return out


def detect(store: Path | str, worktree: Path | str, branch: str = fr.DEFAULT_CHAMPION_BRANCH) -> dict:
    """Read-only. `{state, champion, campaign, reasons}` -- see the module docstring for states."""
    worktree = Path(worktree)
    campaign = read_campaign(store)
    result: dict[str, Any] = {"state": "unreadable", "campaign": campaign, "champion": None,
                              "reasons": []}
    if not campaign["cor"]:
        result["reasons"].append("campaign names no champion_of_record: " + "; ".join(campaign["errors"]))
        return result
    try:
        champion = fr.champion_state(worktree, branch)
        cor_full = _resolve(worktree, campaign["cor"])
        cor_tree = fr.source_tree_hash(worktree, cor_full)
    except fr.FloorRegistryError as exc:
        result["reasons"].append(str(exc))
        return result
    campaign.update({"cor_full": cor_full, "cor_tree_hash": cor_tree})
    result["champion"] = champion
    if cor_tree == champion["source_tree_hash"]:
        result["state"] = "current"
        result["reasons"].append(
            "same source tree (tests/docs excluded)"
            + ("" if cor_full == champion["commit"] else
               f"; commits differ ({cor_full[:12]} vs {champion['commit'][:12]}) but that is not an advance"))
    elif _git_ok(worktree, "merge-base", "--is-ancestor", cor_full, champion["commit"]):
        result["state"] = "behind"
        result["reasons"].append(f"champion advanced {cor_full[:12]} -> {champion['commit'][:12]}")
    else:
        result["state"] = "diverged"
        result["reasons"].append(f"champion_of_record {cor_full[:12]} is not an ancestor of the champion "
                                 f"head {champion['commit'][:12]}: not a plain advance; not acted on")
    return result


def _image_defect(build: Path) -> str | None:
    """Why `build` is not a kernel-store image (None when it is one). Mirrors `fr.is_store_build`."""
    missing = [n for n in ("VERSION.txt", "SHA256SUMS", "PROVENANCE.md") if not (build / n).is_file()]
    if missing:
        return "missing " + ", ".join(missing)
    if (build / "provenance.json").exists():
        return "has provenance.json (an in-tree build, not a store image)"
    return None


def diagnose_store_builds(champion: dict, worktree: Path | str,
                          builds_root: Path | str = DEFAULT_BUILDS_ROOT, *,
                          reference_build: Path | str | None = None) -> dict:
    """`{match, rejected, other}`. `match` is the newest acceptable build's equivalence record (or None).
    `rejected` lists every `cpu-*` image that carries the champion's SOURCE TREE yet was refused, as
    `{build, commit, reason}` -- so a blocker can name the image that is one defect away from usable
    instead of saying none exists. Images with another tree (or no readable commit) are only counted."""
    reference = None
    if reference_build is not None:
        reference = fr.equivalence_for_build(reference_build, source_repo=worktree)["cmake_recipe_digest"]
    rejected: list[dict] = []
    other = 0
    match = None
    for build in sorted(Path(builds_root).glob("cpu-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        defect = _image_defect(build)
        try:
            if defect is not None:
                # tree identity needs only VERSION.txt's commit; no recipe/SHA256SUMS required
                commit, _ = fr._version_info(build)
                full = fr._git(Path(worktree), "rev-parse", "--verify", f"{commit}^{{commit}}")
                tree, eq = fr.source_tree_hash(worktree, full), None
            else:
                eq = fr.equivalence_for_build(build, source_repo=worktree)
                tree, full = eq["source_tree_hash"], eq["commit"]
        except fr.FloorRegistryError:
            other += 1  # commit not in this repo, or an unkeyable build: not a candidate
            continue
        if tree != champion["source_tree_hash"]:
            other += 1
        elif defect is not None:
            rejected.append({"build": str(build), "commit": full, "reason": defect})
        elif reference is not None and eq["cmake_recipe_digest"] != reference:
            rejected.append({"build": str(build), "commit": full,
                             "reason": "CMake recipe digest differs from the reference build's"})
        elif match is None:
            match = eq
    return {"match": match, "rejected": rejected, "other": other}


def find_store_build(champion: dict, worktree: Path | str, builds_root: Path | str = DEFAULT_BUILDS_ROOT,
                     *, reference_build: Path | str | None = None) -> dict | None:
    """Newest kernel-store `cpu-*` build whose equivalence tree hash is the champion's (and, when a
    reference build is given, whose CMake recipe digest equals the reference's). See
    `diagnose_store_builds` for the tree-matching images that were rejected, and why."""
    return diagnose_store_builds(champion, worktree, builds_root,
                                 reference_build=reference_build)["match"]


def find_exact_store_build(commit: str, worktree: Path | str,
                           builds_root: Path | str = DEFAULT_BUILDS_ROOT) -> str | None:
    """Newest kernel-store `cpu-*` image built from EXACTLY `commit` (VERSION.txt), or None.

    CH-27 (2026-10-23): on a plain advance (no new anchor epoch) run.py restores the campaign's
    champion_of_record and refuses to start unless `--cor-build` names that commit's exact build;
    the champion's tree-equivalent image is not enough there (exact arm, never an ancestor)."""
    for build in sorted(Path(builds_root).glob("cpu-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if _image_defect(build) is not None:
            continue
        try:
            short, _ = fr._version_info(build)
            if _resolve(Path(worktree), short) == commit:
                return str(build)
        except fr.FloorRegistryError:
            continue
    return None


def store_floor_identity(store: Path | str) -> dict | None:
    """Identity (recipe_hash, request_digest, model_sha, frame_digest, pairs) of the NEWEST matched
    floor this campaign has on disk -- the condition it would recalibrate."""
    best = None
    for path in Path(store).glob("runtime-source-floors/*/*/serving-floor.*.matched_process_v2.pairs-*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            facts = fr.floor_facts(row)
        except (OSError, ValueError, fr.FloorRegistryError):
            continue
        key = facts.get("calibrated_at") or path.stat().st_mtime
        if best is None or key > best[0]:
            best = (key, {k: facts[k] for k in ("recipe_hash", "request_digest", "model_sha",
                                                "frame_digest", "pairs")} | {"floor_path": str(path)})
    return best[1] if best else None


def _would_refuse(store: Path, worktree: Path, anchor_commit: str) -> dict:
    from . import accumulate
    refuses, reason, kind = accumulate.would_refuse_anchor(
        store, anchor_commit=anchor_commit,
        is_ancestor=lambda older, newer: _git_ok(worktree, "merge-base", "--is-ancestor", older, newer))
    return {"refuses": refuses, "reason": reason, "kind": kind,
            "needs_new_anchor_epoch": bool(refuses and kind == accumulate.REFUSAL_KIND_ANCESTRY),
            "blocking": bool(refuses and kind != accumulate.REFUSAL_KIND_ANCESTRY)}


def launch_paths(*, anchor_build: str, run_argv_additions: list[str], floor_action: str,
                 launch: dict, out_dir: Path | str) -> dict:
    """BOTH ways to relaunch, built from the owner's own inputs and VALIDATED by `serial_roster`.

    `launch` = {resolved_campaign, owned_targets, alias, target_root, [common_args], [rebound_launch],
    [iterations], [state_dir]}. The owner map is copied with `anchor_build` swapped to the store build
    (and `launch` -> `rebound_launch` when given -- required for a validating build, because the
    CanonicalResolvedRecipe binds build_dir); `calibrate_serving` is dropped on `reuse` (no A/A when a
    floor exists) and raised to 24 on `calibrate`. Written under `out_dir` (never over the originals).

      serial_run     -- common-args + the re-anchor/registry flags and the patched owned-targets;
                        `serial_run` forwards `--new-anchor-epoch` to the FIRST child only.
      direct_run_py  -- the exact `run.py` argv/cwd (what `serial_run` would spawn), for operators
                        that launch run.py by hand (RUNBOOK-ak.md s1).
    Raises `ReanchorRefused` with the roster's own message when the inputs do not validate."""
    import sys as _sys
    from . import serial_roster
    from . import serial_run as sr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    owners = json.loads(Path(launch["owned_targets"]).read_text(encoding="utf-8"))
    alias = launch["alias"]
    if alias not in owners:
        raise ReanchorRefused(f"alias {alias!r} not in {launch['owned_targets']}")
    owner = dict(owners[alias], anchor_build=str(anchor_build))
    if launch.get("rebound_launch"):
        owner["launch"] = str(launch["rebound_launch"])
    if "--cor-build" in run_argv_additions:
        owner["cor_build"] = run_argv_additions[run_argv_additions.index("--cor-build") + 1]
    if floor_action == "reuse":
        owner.pop("calibrate_serving", None)
    elif floor_action == "calibrate":
        owner["calibrate_serving"] = max(24, int(owner.get("calibrate_serving") or 0))
    patched = out_dir / "owned-targets.reanchor.json"
    fr._write_atomic(patched, {**owners, alias: owner})
    # anchor (and COR) travel via the owner map
    flags = _drop_pair(_drop_pair(run_argv_additions, "--anchor-build"), "--cor-build")
    common = []
    if launch.get("common_args"):
        common = json.loads(Path(launch["common_args"]).read_text(encoding="utf-8"))
    common_path = out_dir / "common-args.reanchor.json"
    fr._write_atomic(common_path, [*common, *flags])
    try:
        argvs, _skipped, _cpus = serial_roster.build_targets(
            launch["resolved_campaign"], patched, target_root=launch["target_root"], common_path=common_path)
    except sr.SerialRefused as refusal:
        raise ReanchorRefused(
            f"owner inputs do not validate for the new anchor ({refusal}); "
            + ("" if launch.get("rebound_launch") else
               "pass a rebound launch.json (rebound_launch): the launch binds the build digest")) from refusal
    chosen = [a for a in argvs if sr.option(a, "--target-id") == alias]
    if len(chosen) != 1:
        raise ReanchorRefused(f"alias {alias!r} did not resolve to exactly one ready target")
    iterations = str(launch.get("iterations", 1))
    state_dir = str(launch.get("state_dir", out_dir / "serial-state"))
    repo_root = Path(sr.__file__).resolve().parents[4]
    return {
        "serial_run": {
            "cwd": str(repo_root),
            "argv": [_sys.executable, "-m", "scripts.kernel_rnd.autokernel.loop.serial_run",
                     "--resolved-campaign", str(launch["resolved_campaign"]),
                     "--owned-targets", str(patched), "--common-args", str(common_path),
                     "--target-root", str(launch["target_root"]), "--batch-iterations", iterations,
                     "--rounds", "1", "--state-dir", state_dir],
            "note": "--new-anchor-epoch reaches the FIRST child only (serial_run._batch_argv)"},
        "direct_run_py": {
            "cwd": str(repo_root),
            "argv": [_sys.executable, "-m", "scripts.kernel_rnd.autokernel.loop.run", *chosen[0],
                     "--iterations", iterations, "--out", str(out_dir / "run-out")]}}


def _drop_pair(argv: list[str], flag: str) -> list[str]:
    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
        elif item == flag:
            skip = True
        else:
            out.append(item)
    return out


def plan(store: Path | str, worktree: Path | str, *, master: Path | str, role: str | None = None, claims_root: Path | str = fr.DEFAULT_CLAIMS,
         branch: str = fr.DEFAULT_CHAMPION_BRANCH, builds_root: Path | str = DEFAULT_BUILDS_ROOT,
         numerics_epoch: str = "default", require_grade: str = "decision",
         co_tenancy: str = "quiet", floor_identity: dict | None = None,
         reference_build: Path | str | None = None, launch: dict | None = None,
         allow_stale_upstream: bool = False, preview_dir: Path | str | None = None) -> dict:
    """Read-only (except `preview_dir`, the one place launch artifacts may be written). Everything
    `apply` would do, with the floor decision made WITHOUT claiming. With `launch` (see
    `launch_paths`) the output also carries both relaunch paths, validated by `serial_roster`."""
    store, worktree = Path(store), Path(worktree)
    found = detect(store, worktree, branch)
    out: dict[str, Any] = {"schema": SPEC_SCHEMA, "detect": found, "actionable": False, "blockers": []}
    if found["state"] != "behind":
        out["blockers"].append(f"state {found['state']}: " + "; ".join(found["reasons"]))
        return out
    champion = found["champion"]
    diag = diagnose_store_builds(champion, worktree, builds_root, reference_build=reference_build)
    equivalence = diag["match"]
    if equivalence is None:
        msg = (f"no usable kernel-store cpu-* build has the champion's source tree "
               f"({champion['commit'][:12]}) under {builds_root}")
        if diag["rejected"]:
            msg += ("; tree-matching image(s) REJECTED: "
                    + "; ".join(f"{r['build']} (commit {r['commit'][:12]}): {r['reason']}"
                                for r in diag["rejected"])
                    + " -- repair the image, do not rebuild")
            out["rejected_builds"] = diag["rejected"]
        else:
            msg += ": build and store it first"
        out["blockers"].append(msg)
        return out
    out["anchor_build"] = equivalence["build_dir"]
    out["equivalence_key"] = equivalence["equivalence_key"]
    try:
        out["anchor_preflight"] = _would_refuse(store, worktree, champion["commit"])
    except Exception as exc:  # noqa: BLE001 -- preflight failure blocks, never crashes the watcher
        out["blockers"].append(f"anchor preflight failed: {type(exc).__name__}: {exc}")
        return out
    if out["anchor_preflight"]["blocking"]:
        out["blockers"].append("store refuses the anchor for a non-ancestry reason (corrupt journal, "
                               "symlinked store, ...): " + out["anchor_preflight"]["reason"])
        return out
    cor_full = found["campaign"].get("cor_full")
    if not out["anchor_preflight"]["needs_new_anchor_epoch"] and cor_full and cor_full != champion["commit"]:
        # Plain advance: the store keeps its champion_of_record and run.py refuses without that
        # commit's exact build ("supply its original --cor-build"). Name its store image here.
        out["cor_build"] = find_exact_store_build(cor_full, worktree, builds_root)
        if out["cor_build"] is None:
            out["blockers"].append(f"plain advance keeps champion_of_record {cor_full[:12]}, but no kernel-"
                                   f"store cpu-* image built from exactly that commit exists under "
                                   f"{builds_root} (run.py needs it as --cor-build)")
            return out
    identity = floor_identity or store_floor_identity(store)
    if identity is None:
        out["blockers"].append("no floor identity: the campaign has no matched floor on disk; pass "
                               "floor_identity (--floor FILE) so the registry can be consulted")
        return out
    registry = fr.FloorRegistry(master, claims_root)
    out["registry"] = registry.provenance()
    out["upstream"] = fr.upstream_check(registry.master, role=role, model_sha=identity["model_sha"],
                                        allow_stale=bool(allow_stale_upstream))
    if out["upstream"]["status"] == "stale_upstream":
        out["blockers"].append(out["upstream"]["reason"] + " (pass --allow-stale-upstream to override)")
        return out
    if out["upstream"]["status"] == "behind":
        out["blockers"].append(
            f"registry at {registry.master} is behind origin/main; floor for {role or identity['model_sha']} "
            f"exists upstream ({', '.join(out['upstream']['missing_locally'])})")
        return out
    kw = dict(equivalence_key=equivalence["equivalence_key"], recipe_hash=identity["recipe_hash"],
              request_digest=identity["request_digest"], model_sha=identity["model_sha"],
              pairs=identity["pairs"], numerics_epoch=numerics_epoch)
    found_floor = registry.lookup_detail(require_grade=require_grade, champion=champion,
                                         frame_digest=identity.get("frame_digest"), role=role, **kw)
    condition = registry.condition_key(champion_tree_hash=champion["source_tree_hash"],
                                       equivalence_key=equivalence["equivalence_key"],
                                       recipe_hash=identity["recipe_hash"],
                                       request_digest=identity["request_digest"],
                                       model_sha=identity["model_sha"], pairs=identity["pairs"],
                                       numerics_epoch=numerics_epoch, grade=require_grade)
    holder = None if found_floor else registry.claim_holder(condition)
    out["floor"] = {"would": ("reuse" if found_floor else "claimed" if holder else "calibrate"),
                    "registry_status": found_floor.status, "condition": condition,
                    "holder": (holder or {}).get("owner"), "identity": identity}
    out["run_argv_additions"] = _run_argv(equivalence["build_dir"], champion["commit"], registry.master,
                                          numerics_epoch, require_grade, co_tenancy,
                                          needs_epoch=out["anchor_preflight"]["needs_new_anchor_epoch"],
                                          role=role, cor_build=out.get("cor_build"))
    if allow_stale_upstream:
        out["run_argv_additions"].append("--floor-allow-stale-upstream")
    out["actionable"] = True
    if launch is not None and preview_dir is not None:
        try:
            out["launch_paths"] = launch_paths(
                anchor_build=equivalence["build_dir"], run_argv_additions=out["run_argv_additions"],
                floor_action=out["floor"]["would"], launch=launch, out_dir=preview_dir)
        except ReanchorRefused as refusal:
            out["launch_paths_error"] = str(refusal)
    return out


def _run_argv(build: str, champion_commit: str, master, numerics_epoch: str, grade: str,
              co_tenancy: str, *, needs_epoch: bool, role: str | None = None,
              cor_build: str | None = None) -> list[str]:
    argv = ["--anchor-build", str(build)]
    if needs_epoch:
        argv += ["--new-anchor-epoch", champion_commit]
    if cor_build:
        argv += ["--cor-build", str(cor_build)]
    return argv + ["--floor-registry", str(master), "--floor-registry-role", str(role or "ROLE"),
                   "--floor-numerics-epoch", numerics_epoch,
                   "--floor-require-grade", grade, "--floor-co-tenancy", co_tenancy]


def apply(store: Path | str, worktree: Path | str, *, master: Path | str, relaunch_cmd: list[str],
          owner: str, execute: bool = False, **plan_kwargs) -> dict:
    """Claim the (single) recalibration, write the SPEC, run the owner's relaunch command.
    A dry run (the default) returns the plan and does none of that."""
    store = Path(store)
    planned = plan(store, worktree, master=master, **plan_kwargs)
    if not planned["actionable"] or not execute:
        planned["executed"] = False
        return planned
    campaign = planned["detect"]["campaign"]
    if campaign["live"]:
        raise ReanchorRefused(f"campaign at {store} is LIVE (loop-status running and fresh): stopping a lane "
                              f"is its owner's act; this tool never kills a process")
    champion = planned["detect"]["champion"]
    equivalence = fr.equivalence_for_build(planned["anchor_build"], source_repo=worktree)
    ident = planned["floor"]["identity"]
    registry = fr.FloorRegistry(master, plan_kwargs.get("claims_root", fr.DEFAULT_CLAIMS))
    decision = fr.check_launch(
        registry, equivalence=equivalence, recipe_hash=ident["recipe_hash"],
        request_digest=ident["request_digest"], model_sha=ident["model_sha"], pairs=ident["pairs"],
        numerics_epoch=plan_kwargs.get("numerics_epoch", "default"),
        require_grade=plan_kwargs.get("require_grade", "decision"), champion=champion, owner=owner,
        frame_digest=ident.get("frame_digest"), role=plan_kwargs.get("role"))
    if decision["action"] == "error":
        raise ReanchorRefused("floor registry refused: " + "; ".join(decision.get("reasons", [decision["status"]])))
    if decision["action"] == "claimed":
        raise ReanchorRefused(f"another launcher ({decision['claim'].get('owner')}) already holds the "
                              f"recalibration for this champion advance; wait for its floor, do not calibrate")
    # nanoseconds in the stamp: two applies inside one second must not collide on the exclusive spec file
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"{time.time_ns() % 10**9:09d}Z"
    paths = None
    if plan_kwargs.get("launch") is not None:
        try:
            paths = launch_paths(anchor_build=planned["anchor_build"],
                                 run_argv_additions=planned["run_argv_additions"],
                                 floor_action=decision["action"], launch=plan_kwargs["launch"],
                                 out_dir=store / "reanchor" / f"launch-{stamp}")
        except ReanchorRefused:
            if decision["action"] == "calibrate":
                registry.release_claim(decision["condition"])  # nothing was launched: hand it back
            raise
    spec = {"schema": SPEC_SCHEMA, "created": fr._iso(time.time()), "owner": owner, "store": str(store),
            "launch_paths": paths,
            "previous_champion_of_record": campaign.get("cor_full"),
            "champion": champion, "anchor_build": planned["anchor_build"],
            "cor_build": planned.get("cor_build"),
            "equivalence_key": planned["equivalence_key"], "anchor_preflight": planned["anchor_preflight"],
            "floor_action": decision["action"], "floor": {k: v for k, v in decision.items() if k != "claim"},
            "calibrate_serving_pairs": 24 if decision["action"] == "calibrate" else None,
            "run_argv_additions": planned["run_argv_additions"]}
    spec_path = store / "reanchor" / f"reanchor-{stamp}.json"
    fr._write_atomic(spec_path, spec, exclusive=True)
    done = subprocess.run(relaunch_cmd, env={**os.environ, "REANCHOR_SPEC": str(spec_path)})
    spec_after = {"spec": str(spec_path), "relaunch_rc": done.returncode, "action": decision["action"]}
    if decision["action"] == "calibrate" and done.returncode != 0:
        # The calibration never produced a floor: hand the condition back (register would have).
        spec_after["claim_released"] = registry.release_claim(decision["condition"])
    if decision["action"] == "calibrate" and done.returncode == 0:
        # The run uses its fresh floor in-process and leaves a registry PLAN next to the sealed floor.
        # Say so on the claim: waiters learn a floor EXISTS and only awaits the owner's registry commit.
        plans = sorted(store.glob("runtime-source-floors/**/*.registry-plan.json"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        if plans:
            spec_after["registry_plan"] = str(plans[0])
            spec_after["claim_marked"] = registry.mark_awaiting_registration(decision["condition"], plans[0])
    planned.update({"executed": True, **spec_after})
    return planned


SUPPORTED_INVOCATION = ("from scripts/kernel_rnd:  python -m autokernel.loop.champion_reanchor ...   or   "
                        "from scripts/:  python -m kernel_rnd.autokernel.loop.champion_reanchor ...")


def _invocation_ok(package: str | None) -> bool:
    """`plan`/`apply` import `..journal`, which needs `loop` to live inside the `autokernel` package.
    `python -m loop.champion_reanchor` (cwd = autokernel/) makes `loop` top-level, so detect works but
    plan/apply die with an opaque relative-import error -- refuse that up front."""
    return bool(package) and package.count(".") >= 1


def main(argv: list[str] | None = None) -> int:
    if not _invocation_ok(__package__):
        sys.stderr.write("champion_reanchor: unsupported invocation (`loop` is a top-level package, so "
                         "`..` imports fail). Supported: " + SUPPORTED_INVOCATION + "\n")
        return 2
    p = argparse.ArgumentParser(prog="champion_reanchor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("detect", "plan", "apply"))
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--worktree", type=Path, required=True, help="a worktree of the champion's repo")
    p.add_argument("--champion-branch", default=fr.DEFAULT_CHAMPION_BRANCH)
    p.add_argument("--master", type=Path, default=None,
                   help="the MASTER model registry (the only floor store). Default: the registry of the "
                        "repo this code runs from; an override is logged loudly")
    p.add_argument("--role", help="the serving_floors.<role> key of this campaign's target")
    p.add_argument("--claims", type=Path, default=fr.DEFAULT_CLAIMS)
    p.add_argument("--allow-stale-upstream", action="store_true",
                   help="proceed although origin/main cannot be refreshed and the last fetch is >24h old")
    p.add_argument("--builds-root", type=Path, default=DEFAULT_BUILDS_ROOT)
    p.add_argument("--reference-build", type=Path,
                   help="require the store build's CMake recipe to equal this build's (the old anchor)")
    p.add_argument("--numerics-epoch", default="default")
    p.add_argument("--require-grade", choices=fr.GRADES, default="decision")
    p.add_argument("--co-tenancy", choices=fr.CO_TENANCY, default="quiet")
    p.add_argument("--floor", type=Path, help="a floor json naming the condition (default: the store's newest)")
    p.add_argument("--resolved-campaign", type=Path, help="with --owned-targets/--alias/--target-root: "
                   "also emit both relaunch paths (serial_run and direct run.py), validated")
    p.add_argument("--owned-targets", type=Path)
    p.add_argument("--alias", help="the owned-targets alias (== --target-id) being re-anchored")
    p.add_argument("--target-root", type=Path)
    p.add_argument("--common-args", type=Path)
    p.add_argument("--rebound-launch", type=Path, help="launch.json rebound to the new anchor build")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--state-dir", type=Path)
    p.add_argument("--preview-dir", type=Path, help="plan: where launch artifacts may be written")
    p.add_argument("--relaunch-cmd", help="JSON argv array run by `apply --execute` with REANCHOR_SPEC set")
    p.add_argument("--owner", default=os.environ.get("USER", "unknown"))
    p.add_argument("--execute", action="store_true", help="apply: actually claim, write the spec, relaunch")
    args = p.parse_args(argv)
    try:
        if args.cmd == "detect":
            out = detect(args.store, args.worktree, args.champion_branch)
            print(json.dumps(out, indent=1, sort_keys=True, default=str))
            return 0 if out["state"] in ("current",) else 3 if out["state"] != "behind" else 10
        identity = None
        if args.floor:
            facts = fr.floor_facts(json.loads(args.floor.read_text(encoding="utf-8")))
            identity = {k: facts[k] for k in ("recipe_hash", "request_digest", "model_sha",
                                              "frame_digest", "pairs")}
        kw = dict(master=args.master, role=args.role, allow_stale_upstream=args.allow_stale_upstream, claims_root=args.claims, branch=args.champion_branch, builds_root=args.builds_root,
                  numerics_epoch=args.numerics_epoch, require_grade=args.require_grade,
                  co_tenancy=args.co_tenancy, floor_identity=identity,
                  reference_build=args.reference_build)
        if args.resolved_campaign:
            need = ("owned_targets", "alias", "target_root")
            if any(getattr(args, n) is None for n in need):
                p.error("--resolved-campaign needs --owned-targets, --alias and --target-root")
            kw["launch"] = {"resolved_campaign": args.resolved_campaign, "owned_targets": args.owned_targets,
                            "alias": args.alias, "target_root": args.target_root,
                            "common_args": args.common_args, "rebound_launch": args.rebound_launch,
                            "iterations": args.iterations,
                            **({"state_dir": args.state_dir} if args.state_dir else {})}
        if args.cmd == "plan":
            out = plan(args.store, args.worktree, preview_dir=args.preview_dir, **kw)
        else:
            if args.execute and not args.relaunch_cmd:
                p.error("apply --execute needs --relaunch-cmd")
            out = apply(args.store, args.worktree, owner=args.owner, execute=args.execute,
                        relaunch_cmd=json.loads(args.relaunch_cmd) if args.relaunch_cmd else [], **kw)
        print(json.dumps(out, indent=1, sort_keys=True, default=str))
        return 0 if out.get("actionable") or out["detect"]["state"] == "current" else 3
    except (ReanchorRefused, fr.FloorRegistryError) as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
