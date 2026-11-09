#!/usr/bin/env python3
"""Mechanisms already answered elsewhere, so a planner does not re-propose them.

`generate` writes `<store>/retired-mechanisms.jsonl` and `<store>/inbox/00-retired-mechanisms.md`
from `seeds/retired_manifest.json` (each row re-verified against git or its filed verdict, dropped
if not), `seeds/negatives.json` and the `akm-*` commit subjects of the champion and experimental
branches. `refusal` is the formation-guard check: only `enforcement: hard` entries refuse.

Hard means the identity is an exact symbol and the answer is conclusive. Inconclusive nulls, [inf]
rows and anything whose symbol is prose stay advisory: rendered for the planner, never refused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping

LIST_NAME = "retired-mechanisms.jsonl"
NOTE_NAME = "00-retired-mechanisms.md"
SEEDS = Path(__file__).with_name("seeds")
STATUSES = ("in-champion", "tried-null", "superseded", "in-flight")
FIELDS = ("target_surface", "target_symbol", "mechanism", "status", "receipt", "reopen_if")
#: Id tokens naming the lane or era rather than the change, so they cannot identify a mechanism.
GENERIC_TOKENS = {"akm", "ak", "hist", "cpu", "gpu", "c6", "c7", "c8", "c9", "c10", "q38fn",
                  "iqk", "moe", "27b"}
INCONCLUSIVE = re.compile(r"\bn=1\b|INCONCLUSIVE|no effect size|NOT MEASURED", re.I)
IDENTIFIER = re.compile(r"^[A-Za-z_][\w:]*$")
PATH = re.compile(r"^[\w.-]+(/[\w.-]+)+$")
REOPEN_RECHECK = "re-measure on the current champion with a floor below the expected effect"


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower())
            if t and not t.isdigit()} - GENERIC_TOKENS


def _surface(text: str) -> str:
    return str(text or "").split(" (")[0].strip().rstrip("/")


def _surface_agrees(entry: str, proposed: str) -> bool:
    entry, proposed = _surface(entry), _surface(proposed)
    return bool(entry and proposed) and (
        entry == proposed or proposed.endswith("/" + entry) or entry.endswith("/" + proposed)
        or proposed.startswith(entry + "/"))


def _mechanism_agrees(entry: Mapping[str, Any], mechanism_id: str) -> bool:
    """Scoped entries need their id tokens; an entry with none owns the whole symbol."""
    wanted = {str(t).lower() for t in entry.get("mechanism_tokens") or ()}
    if str(entry.get("mechanism")) == mechanism_id or not wanted:
        return True
    return len(wanted & _tokens(mechanism_id)) >= min(2, len(wanted))


def load(path: Path) -> list[dict[str, Any]]:
    """Every entry, or raises OSError/ValueError: one bad line voids the list."""
    entries = [json.loads(line) for line in
               Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(e, dict) and all(k in e for k in FIELDS) for e in entries):
        raise ValueError(f"{path}: entry without the required fields")
    return entries


def refusal(hypothesis, path: Path | str | None,
            note: Callable[[str], object] = print) -> str | None:
    """The refusal text for a hard match. Fails open, loudly, on a missing or corrupt list."""
    if path is None:
        return None
    try:
        entries = load(Path(path))
    except (OSError, ValueError) as exc:
        note(f"WARNING retired_mechanisms_unavailable path={path} guard=OPEN: {exc}")
        return None
    for entry in entries:
        if (entry.get("enforcement") == "hard"
                and entry["target_symbol"] == hypothesis.target_symbol
                and _surface_agrees(entry["target_surface"], hypothesis.target_surface)
                and _mechanism_agrees(entry, hypothesis.mechanism_id)):
            return (f"retired_mechanism {hypothesis.mechanism_id} on "
                    f"{hypothesis.target_surface}::{hypothesis.target_symbol}: "
                    f"{entry['status']} as {entry['mechanism']} ({entry['receipt']}); "
                    f"reopen_if: {entry['reopen_if']}")
    return None


def _git(repo: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True)


def _verified(row: Mapping[str, Any], repo: Path, champion: str) -> str | None:
    """None when the row's evidence holds, otherwise why it does not."""
    commit, verdict = row.get("commit"), row.get("verdict")
    if commit:
        if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
            return f"commit {commit} not found"
        in_champion = not _git(repo, "merge-base", "--is-ancestor", commit, champion).returncode
        if in_champion != (row["status"] == "in-champion"):
            return f"commit {commit} champion ancestry contradicts status {row['status']}"
        if row.get("ref") and _git(repo, "merge-base", "--is-ancestor", commit,
                                   row["ref"]).returncode:
            return f"commit {commit} is not on {row['ref']}"
        needle = row.get("diff_contains")
        if needle and needle not in _git(repo, "show", "--format=", commit).stdout:
            return f"commit {commit} diff lacks {needle}"
    if verdict:
        try:
            found = json.loads(Path(verdict["path"]).read_text(encoding="utf-8")).get("verdict")
        except (OSError, ValueError) as exc:
            return f"verdict unreadable: {exc}"
        if found != verdict["expect"]:
            return f"verdict {found} is not {verdict['expect']}"
    if not commit and not verdict:
        return "row names no commit or verdict"
    return None


def _from_manifest(repo: Path, champion: str, seeds: Path) -> tuple[list[dict], list[str]]:
    kept, dropped = [], []
    body = json.loads((seeds / "retired_manifest.json").read_text(encoding="utf-8"))
    for row in body["entries"]:
        why = _verified(row, repo, champion)
        if why is not None:
            dropped.append(f"{row['mechanism']}: {why}")
            continue
        entry = {key: row[key] for key in FIELDS}
        entry.update(enforcement=row["enforcement"], evidence=row.get("evidence", "checked"),
                     mechanism_tokens=row.get("mechanism_tokens", []))
        kept.append(entry)
    return kept, dropped


def _from_negatives(seeds: Path) -> list[dict]:
    """Hard only for a conclusive current-lineage result on an exact symbol."""
    out = []
    for row in json.loads((seeds / "negatives.json").read_text(encoding="utf-8"))["records"]:
        mechanism, reason = row["mechanism_id"], str(row.get("reason", ""))
        symbol, surface = str(row["target_symbol"]).split(" (")[0], _surface(row["target_surface"])
        inconclusive = bool(INCONCLUSIVE.search(reason)) or row.get("effect_fraction") is None
        hard = (not inconclusive and "-hist-" not in mechanism and bool(IDENTIFIER.match(symbol))
                and bool(PATH.match(surface)))
        out.append({
            "target_surface": surface, "target_symbol": symbol, "mechanism": mechanism,
            "status": "tried-null", "receipt": f"negatives.json {mechanism}: {reason[:160]}",
            "reopen_if": ("a design that removes the measured cause" if hard
                          else REOPEN_RECHECK),
            "enforcement": "hard" if hard else "advisory", "evidence": "negatives.json",
            "mechanism_tokens": sorted(_tokens(mechanism))})
    return out


def _hunk_symbol(repo: Path, commit: str) -> str:
    """Best-effort enclosing symbol, which is why subject-derived entries stay advisory."""
    for line in _git(repo, "show", "--format=", commit).stdout.splitlines():
        if line.startswith("@@"):
            names = re.findall(r"([A-Za-z_]\w*)\s*(?:\(|$|\{)", line.split("@@")[-1])
            if names:
                return names[0]
    return "unknown"


def _from_log(repo: Path, champion: str, refs: list[str]) -> list[dict]:
    """Advisory rows from `akm-*` subjects: in the champion, or tips of experimental branches."""
    out = []
    for status, spec in (("in-champion", [champion]),
                         ("in-flight", [*refs, "--not", champion])):
        if status == "in-flight" and not refs:
            continue
        log = _git(repo, "log", "--format=%H%x00%s", *spec).stdout
        for line in log.splitlines():
            commit, _, subject = line.partition("\0")
            match = re.match(r"(akm-[\w-]+):", subject)
            if not match:
                continue
            names = _git(repo, "show", "--name-only", "--format=", commit).stdout.split()
            out.append({
                "target_surface": names[0] if names else "unknown",
                "target_symbol": _hunk_symbol(repo, commit), "mechanism": match[1],
                "status": status, "receipt": f"{commit[:9]} {subject[:100]}",
                "reopen_if": "a different mechanism, or this commit is dropped",
                "enforcement": "advisory", "evidence": "git-subject",
                "mechanism_tokens": sorted(_tokens(match[1]))})
    return out


def lane_of(surface: str) -> str | None:
    if "ggml-cpu" in surface:
        return "cpu"
    return "gpu" if "ggml-cuda" in surface or "ggml-hip" in surface else None


def experimental_refs(repo: Path) -> list[str]:
    listed = _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads/experimental/",
                  "refs/heads/llama.cpp-experimental-c10-*").stdout.split()
    return [ref for ref in listed if re.search(r"(^|[-/])(akm-|c9|c10)", ref)]


def build(repo: Path, champion: str, lane: str, *,
          seeds: Path = SEEDS) -> tuple[list[dict], list[str]]:
    """Entries for one lane, manifest first so a checked row wins over a derived duplicate."""
    entries, dropped = _from_manifest(repo, champion, seeds)
    seen: set[str] = set()
    out = []
    for entry in (*entries, *_from_negatives(seeds),
                  *_from_log(repo, champion, experimental_refs(repo))):
        if entry["mechanism"] in seen or lane_of(entry["target_surface"]) not in (lane, None):
            continue
        seen.add(entry["mechanism"])
        out.append(entry)
    return out, dropped


def render(entries: list[dict], champion: str) -> str:
    hard = [e for e in entries if e["enforcement"] == "hard"]
    soft = [e for e in entries if e["enforcement"] != "hard"]
    head = "| surface | symbol | mechanism | status | receipt | reopen_if |\n|---|---|---|---|---|---|\n"

    def table(rows: list[dict]) -> str:
        return head + "".join(
            "| " + " | ".join(str(e[k]).replace("|", "/").replace("\n", " ") for k in FIELDS)
            + " |\n" for e in rows)
    return (f"# Retired mechanisms (generated, champion {champion}; do not hand-edit)\n\n"
            "Answered outside this store's own experiments, so do not re-propose these under "
            "a new id. To propose one anyway, state how its `reopen_if` is met.\n\n"
            f"## Hard: refused at formation ({len(hard)})\n\n{table(hard)}\n"
            f"## Advisory: inconclusive, inferred or unverified identity, not refused "
            f"({len(soft)})\n\n{table(soft)}")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/raid0/llm/llama.cpp"))
    parser.add_argument("--champion", required=True)
    parser.add_argument("--lane", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--store", type=Path, required=True,
                        help="writes <store>/retired-mechanisms.jsonl and the inbox note")
    args = parser.parse_args(argv)
    entries, dropped = build(args.repo, args.champion, args.lane)
    for why in dropped:
        print(f"WARNING dropped_unverified {why}", file=sys.stderr)
    _write(args.store / LIST_NAME, "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    _write(args.store / "inbox" / NOTE_NAME, render(entries, args.champion))
    hard = sum(e["enforcement"] == "hard" for e in entries)
    print(f"{args.store}: {len(entries)} entries, {hard} hard, {len(dropped)} dropped")
    return 0


__all__ = ["LIST_NAME", "NOTE_NAME", "STATUSES", "build", "load", "main", "refusal", "render"]

if __name__ == "__main__":
    raise SystemExit(main())
