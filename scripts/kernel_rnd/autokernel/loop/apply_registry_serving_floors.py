#!/usr/bin/env python3
"""Add serving-floor entries to the MASTER model registry WITHOUT disturbing its comments.

    python3 -m autokernel.loop.apply_registry_serving_floors \
        --registry <CLEAN-WORKTREE>/orchestration/model_registry.yaml --plan PLAN.json [--dry-run]

The master registry is a 13k-line hand-annotated YAML file; a `yaml.dump` round-trip would destroy the
comments that carry its history (the same reason `apply_registry_integrity_pins.py` inserts line by
line). This inserts one entry into the top-level `serving_floors:` section (creating the section at the
end of the file if absent), cites the sealed floor RECORD in place (it can be hundreds of MB and is never copied into
git), and then PROVES the edit:

  * every original line is still present, in order (the diff is pure insertion);
  * re-parsed, the document equals the original plus EXACTLY the planned entry (no other change, no
    duplicate keys);
  * the copied record's sha256 is the one the entry pins, and `floor_registry.validate_section` is clean
    for the new document.

PLAN comes from `floor_registry.py register` (`plan_registration`). Nothing here talks to a campaign
store. The owning session runs this from a CLEAN worktree of origin/main (refused otherwise: a dirty
tree would carry another session's hunks into your commit), then commits the registry hunk and the
record in a normal commit. Idempotent: an entry whose `id` is already present is reported and left alone.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from . import floor_registry as fr

SECTION_HEADER = """
# =============================================================================
# Serving floors — measured run-to-run noise floors, keyed by role (AKX-AUTO-6 / floor registry)
# =============================================================================
#
# THE ONLY STORE of serving noise floors. A floor is a property of the measured CONDITION: it is keyed by
# (equivalence_key = source tree minus tests/docs + CMake recipe + compiler, recipe_hash, request_digest,
# model_sha, k) and is never reused across a champion advance (champion_tree_hash differs) or a numerics
# change (numerics_epoch differs). `grade.claim_class: decision` requires `co_tenancy: quiet`.
# `record` is the sealed A/A JSON (durable, non-scratch); `record_sha256` pins its bytes.
#
# DO NOT hand-edit. Entries are added by
#     scripts/kernel_rnd/autokernel/loop/apply_registry_serving_floors.py
# (comment-preserving, round-trip proven) from a clean worktree of origin/main. Readers:
# autokernel.loop.floor_registry (lookup / check_launch). Validator: scripts/validate_model_registry.py.
# The lean compiler drops this section (it is research-side evidence, not runtime input).
#
"""


class ApplyError(RuntimeError):
    pass


def _render_entry(entry: dict) -> list[str]:
    """YAML list-item lines for one entry, indented to sit under `  <role>:` (item dash at 4)."""
    body = yaml.safe_dump(entry, sort_keys=False, default_flow_style=False, allow_unicode=True,
                          width=10 ** 6).rstrip("\n").split("\n")
    out = []
    for i, line in enumerate(body):
        out.append(("    - " if i == 0 else "      ") + line)
    return out


def _git_clean(path: Path) -> None:
    repo = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if repo.returncode != 0:
        raise ApplyError(f"{path} is not inside a git worktree: refusing (use --allow-dirty to override)")
    dirty = subprocess.run(["git", "-C", repo.stdout.strip(), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ApplyError("worktree is not clean (a dirty tree carries other sessions' hunks into your "
                         "commit); use a clean worktree of origin/main:\n" + dirty[:500])


def insert_entry(lines: list[str], role: str, entry: dict) -> list[str]:
    """New line list with `entry` appended under `serving_floors.<role>` (sections/roles created)."""
    new = list(lines)
    rendered = _render_entry(entry)
    span = fr.section_slice(new)
    if span is None:
        while new and new[-1] == "":
            new.pop()
        return new + SECTION_HEADER.rstrip("\n").split("\n") + [f"{fr.SECTION}:", f"  {role}:"] \
            + rendered + [""]
    lo, hi = span
    child = next((i for i in range(lo + 1, hi) if new[i].startswith(f"  {role}:")
                  and new[i][2 + len(role) + 1:].strip() in ("", ) ), None)
    if child is None:
        last = hi
        while last > lo + 1 and not new[last - 1].strip():
            last -= 1
        return new[:last] + [f"  {role}:"] + rendered + new[last:]
    end = hi
    for i in range(child + 1, hi):
        ln = new[i]
        if ln.strip() and not ln.lstrip().startswith("#") and (len(ln) - len(ln.lstrip())) <= 2:
            end = i
            break
    while end > child + 1 and not new[end - 1].strip():
        end -= 1
    return new[:end] + rendered + new[end:]


def apply_plan(registry: Path | str, plan: dict, *, dry_run: bool = False,
               allow_dirty: bool = False) -> dict:
    registry = Path(registry)
    role, entry = plan["role"], plan["entry"]
    if plan.get("schema") != "epyc.autokernel.floor_registry_plan.v1":
        raise ApplyError("not a floor_registry plan")
    if not allow_dirty and not dry_run:
        _git_clean(registry)
    text = registry.read_text(encoding="utf-8")
    original = yaml.safe_load(text)
    existing = (original.get(fr.SECTION) or {}).get(role, [])
    if any(e.get("id") == entry["id"] for e in existing):
        return {"status": "already_present", "id": entry["id"], "role": role}

    # 1. the record: cited IN PLACE (absolute, durable, untracked -- it can be hundreds of MB and raw
    #    measurement material never goes into git); verify it is the one the entry pins.
    root = fr.repo_root_of(registry)
    cited = Path(entry["record"])
    cited = cited if cited.is_absolute() else root / cited
    if fr.is_scratch(str(cited)):
        raise ApplyError(f"record {cited} is in scratch: refusing")
    h = hashlib.sha256()
    with open(cited, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != entry["record_sha256"]:
        raise ApplyError(f"record {cited} does not match the entry's pinned record_sha256")

    # 2. the edit, then the proof
    lines = text.split("\n")
    new_lines = insert_entry(lines, role, entry)
    new_text = "\n".join(new_lines)
    ops = [op for op in difflib.SequenceMatcher(a=lines, b=new_lines, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    if any(tag != "insert" for tag, *_ in ops):
        raise ApplyError(f"edit is not a pure insertion: {ops[:3]}")
    reparsed = yaml.safe_load(new_text)
    expected = copy.deepcopy(original)
    expected.setdefault(fr.SECTION, {}).setdefault(role, []).append(json.loads(json.dumps(entry)))
    if reparsed != expected:
        raise ApplyError("re-parsed registry != original + exactly the planned entry")
    # validate_section resolves the record relative to <repo>/..., so run it against the final layout
    problems: tuple[list[str], list[str]]
    if dry_run:
        problems = ([], [])
    else:
        problems = fr.validate_section(reparsed, root)
        if problems[0]:
            raise ApplyError("validate_section refuses the new document: " + "; ".join(problems[0]))
        tmp = registry.with_name(f".{registry.name}.{os.getpid()}.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, registry)
    diff = "\n".join(difflib.unified_diff(lines, new_lines, "a/model_registry.yaml",
                                          "b/model_registry.yaml", n=2, lineterm=""))
    return {"status": "dry_run" if dry_run else "applied", "id": entry["id"], "role": role,
            "record": entry["record"], "warnings": problems[1], "diff": diff}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-dirty", action="store_true",
                   help="skip the clean-worktree check (tests only; never for a real commit)")
    args = p.parse_args(argv)
    try:
        result = apply_plan(args.registry, json.loads(args.plan.read_text(encoding="utf-8")),
                            dry_run=args.dry_run, allow_dirty=args.allow_dirty)
    except (ApplyError, fr.FloorRegistryError, OSError, ValueError, KeyError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(result.get("diff", ""))
    print(f"{result['status']}: {result['role']} {result['id']}"
          + (f" (record {result['record']})" if "record" in result else ""))
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
