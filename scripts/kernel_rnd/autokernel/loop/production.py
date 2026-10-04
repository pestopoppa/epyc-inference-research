#!/usr/bin/env python3
"""Publish the champion's gain over the FROZEN PRODUCTION kernel, at every advance.

WHY THIS EXISTS
---------------
The Kernel R&D dashboard's headline is defined as one thing and one thing only: the
champion tree's measured gain over frozen production v9. It was taken ONCE, by hand,
for champion `5ad3e36d` (+8.524% decode). The champion has advanced twice since, so
the panel renders SUPERSEDED with no current figure -- correct, and useless. A
headline that only a person can refresh is a headline that is wrong by default.

It cannot be computed from what the loop already publishes. Every per-iteration
effect on that page is a MARGINAL against an anchor that advances on every keep, so
each has a different baseline; composing them arithmetically would claim a
measurement no run ever took. This program has made that error once already. The
number can only ever come from ONE direct A/B, both arms in the same session --
absolute throughput on this host is not comparable across sessions.

WHY IT IS AFFORDABLE. Run 19 produced 2 keeps in 11 hours, and a paired A/B at 20
pairs is ~181 s of device time: about 0.09%. The operator's ruling: "the champion
advances so rarely that performing the proper A/B measurement whenever the champion
advances is totally reasonable."

TWO BUILDS THAT DO NOT HAPPEN HERE
  * The CHAMPION arm is the anchor slot `pool.promote_anchor` just built from the
    champion commit, and which `anchor.verify` just proved IS the champion. Building
    it again would pay a second full build for a binary already on disk and already
    A/A-verified -- and a second build is a second chance to bench something that is
    not the champion.
  * The BASELINE arm is frozen production, resolved LIVE from the canonical frozen
    tree at every refresh (a promotion advances that tree, and the headline must
    follow it -- never a stale pinned sha). Its build is cached per commit: built at
    most once per freeze, then reused. A missing cache means "build it once", never
    "rebuild every time"; a new promotion means one cache miss, one build.

WHY A FAILURE HERE MUST NOT STOP THE RUN. This is a REPORTING refresh, not a
correctness gate. `anchor.verify` aborts because a wrong anchor voids every
measurement after it; nothing downstream of this depends on it at all. If the
baseline is missing or the A/B fails, the previous bundle stays exactly where it is
and the panel keeps reading SUPERSEDED -- which is the correct degraded state, and
strictly better than killing a run that is otherwise producing science. So
`refresh` NEVER raises and never uses `loop.RunAborted`.

THE EFFECT IS NOT WRITTEN INTO THE ATTEMPT ROW. `to_attempt` records the refresh in
durable memory with `effect_fraction: None` and the number in the prose. Every other
row in that table is a marginal against the advancing anchor, and the planner reads
that table back as its memory; dropping a CUMULATIVE +8.5% into it, in the column
the planner compares against, is the composition error one level down.

THE RECORDED PRODUCTION BASELINE (carry-over, 2026-10-04). Operator: "v10 was once a
champion. We have those measurements. The production promotion should have
transferred those results." A promoted champion's last standing (or fold) samples ARE
production's numbers the moment it is frozen, so the promotion writes them as
`production-baseline.<sha12>.json` (`write_baseline`, CLI `write-baseline`), and
`refresh` then measures the CHAMPION ALONE against that record -- no baseline build, no
production arm, half the device time. The record stands in for a production arm only
when it was taken under the SAME protocol (the exact llama-bench argv, model and
surface), is younger than `BASELINE_MAX_AGE_DAYS`, predates no recorded host change
(`HOST_CHANGES`) and its captured host facts still match this host. Anything else falls
back to the paired A/B above, with the refusal recorded in the bundle; and a paired
fallback re-records the production arm it just measured, so the NEXT refresh is
single-arm. A carry-over number is UNPAIRED -- two sessions, and this host drifts ~3%
over hours -- and the bundle and the attempt reason both say so.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from . import bench, headline_admissibility, status

#: The contract `dashboard/loop_status.py` reads. Its reader REFUSES a bundle whose
#: `baseline.commit` is not the frozen production commit rather than relabelling it,
#: so these two constants are load bearing: they are the only reason the number
#: shown under that heading is the number that heading names.
SCHEMA = "epyc.autokernel.champion_vs_production.v1"
FILENAME = "champion-vs-production.json"
MECHANISM_ID = "champion-vs-production"

#: The canonical FROZEN production tree. The baseline commit is resolved LIVE from its
#: HEAD, never pinned here: the promotion process is what advances that tree, so the
#: headline's baseline follows a promotion with no constant for anyone to forget.
#: (Operator, 2026-08-31: "once we promote a new frozen version in the future, the
#: comparison should be against the newly promoted version, NOT stale v9. This is a
#: classic mistake." The previous revision of this file had made it: a hardcoded
#: `BASELINE_COMMIT` that would have silently kept measuring v9 past a v10 freeze.)
FROZEN_TREE = Path("/mnt/raid0/llm/llama.cpp")
#: The production branch contract (`production-consolidated-v9`, future `-v10`, ...) —
#: the same family `scripts/session/verify_llama_cpp.sh` enforces in the root repo. A
#: tree on any other branch is NOT the freeze, and no headline is measured against it.
FROZEN_BRANCH_PREFIX = "production-consolidated-"
#: Baseline builds are cached PER COMMIT under this root, `production-baseline-<sha12>`.
#: A promotion is then a cache miss that builds once, never an overwrite.
BASELINE_ROOT = Path("/mnt/raid0/llm/tmp")
#: v9's verified prebuilt, ADOPTED as the cache entry for exactly this sha (verified
#: 2026-08-30: 584/584 CPU symbols, 918/918 GPU device kernels identical to the shipped
#: libraries). A legacy-path fallback rather than a rebuild or a symlink; it is never
#: used for any other commit.
LEGACY_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
LEGACY_BUILD = Path("/mnt/raid0/llm/tmp/v9v-build-base")
#: Source for the at-most-once build. NOT `FROZEN_TREE`: the production tree is frozen
#: and may not be built in. The builder checks this copy's HEAD against the LIVE
#: resolved commit, so after a promotion a stale copy refuses loudly instead of
#: publishing a number under the new freeze's name.
BASELINE_TREE = Path("/mnt/raid0/llm/tmp/v9v-base-tree")
#: Written into a baseline build this module drove, so a cache that is later pointed
#: at some other tree is caught instead of published under production's name.
PROVENANCE = "baseline-provenance.json"
#: Days, not minutes: a cumulative A/B is a deliberate, expensive act rather than a
#: per-iteration beat. Matches the reader's own default envelope.
STALE_AFTER_S = 14 * 86400

#: ---- The recorded production baseline (see the module docstring) ----
BASELINE_SCHEMA = "epyc.autokernel.production_baseline.v1"
#: Default store, where `champion-vs-production.json` lives and `refresh` looks.
DEFAULT_STORE = Path("/mnt/raid0/llm/autokernel/loop-memory")
#: A record older than this is not trusted to describe today's production arm.
BASELINE_MAX_AGE_DAYS = 30.0
#: Measured: absolute throughput on this host drifts ~3% over hours, same window
#: alternating (2026-09 host-drift ABA). The size of what an unpaired number cannot see.
HOST_DRIFT_CAVEAT_PCT = 3.0
UNPAIRED_CAVEAT = (
    "UNPAIRED: the champion was measured alone and compared against production "
    "samples RECORDED in an earlier session, not in a same-session alternating A/B. "
    f"Absolute throughput on this host drifts ~{HOST_DRIFT_CAVEAT_PCT:g}% over hours, "
    "so an effect inside that band is not resolved by this comparison, and the "
    "confidence interval covers launch scatter only, not the cross-session drift.")
#: Recorded host changes. A baseline measured on or before a change's `applied` date
#: is stale: the host it describes no longer exists. Settings-only BIOS changes leave
#: the DMI BIOS version untouched, so `read_host_facts` alone cannot see them -- this
#: ledger is the record. Append a row whenever the host's BIOS settings, firmware,
#: kernel or memory configuration change.
HOST_CHANGES: tuple[dict, ...] = (
    {"applied": "2026-09-21",
     "what": "BIOS: memory interleave ENABLED + memory speed 5600 MT/s (was 4800); "
             "NPS4 unchanged",
     "source": "epyc-root artifacts/operator/v10-qualification-20260921-ffc1bac82.json "
               "host.bios_change"},
)
#: The protocol fields that must be IDENTICAL for a record to stand in for a
#: production arm. `argv` subsumes the rest, but each is named so a refusal says WHICH.
PROTOCOL_MATCH_KEYS = ("harness", "model", "surface", "pp", "tg", "ubatch", "reps",
                       "warmup_launches", "sample_unit", "hardened", "argv")


class Unavailable(RuntimeError):
    """The measurement could not be taken. Reporting only -- the run continues."""


def _read(tree: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise Unavailable(f"cannot resolve the frozen production kernel from {tree}: "
                          f"{done.stderr.strip()[:200]}")
    return done.stdout.strip()


def resolve_frozen(tree: Path = FROZEN_TREE) -> tuple[str, str]:
    """`(commit, label)` of the frozen production kernel, resolved LIVE from the tree.

    The branch is cross-checked against the production contract FIRST: a resolver that
    returned whatever HEAD it found would headline against an experimental checkout the
    day someone left the tree on one. Failing the check raises `Unavailable`, so the
    refresh is refused and recorded while the run continues.
    """
    label = _read(tree, "branch", "--show-current")
    # Refusing to publish a headline measured against an unknown tree.
    if not label.startswith(FROZEN_BRANCH_PREFIX):
        raise Unavailable(f"{tree} is on branch '{label or '(detached)'}', not the "
                          f"production contract '{FROZEN_BRANCH_PREFIX}*'")
    return _read(tree, "rev-parse", "HEAD"), label


def baseline_slot(commit: str, *, root: Path = BASELINE_ROOT,
                  legacy: Path = LEGACY_BUILD) -> Path:
    """The cache directory for one frozen commit's baseline build.

    Keyed by commit, so a promotion is a cache miss that builds once. The v9 prebuilt
    is adopted in place for exactly `LEGACY_COMMIT` (see the constant), because it is
    already verified against production's shipped libraries and rebuilding a
    known-good tree is the opposite of the cache contract.
    """
    if commit == LEGACY_COMMIT and is_built(legacy):
        return legacy
    return root / f"production-baseline-{commit[:12]}"


def is_built(build_dir: Path | str) -> bool:
    """Whether a build directory holds the binary this comparison needs."""
    return (Path(build_dir) / "bin" / "llama-bench").is_file()


def declared_commit(build_dir: Path | str) -> str | None:
    """The commit a cached baseline build declares, or None if it declares nothing.

    None is ACCEPTED: the verified prebuilt predates this stamp, and refusing it
    would force a rebuild of a tree that is known-good -- the opposite of the cache
    contract. A stamp that DISAGREES is refused, because that is the only reading
    that means someone re-pointed the cache.
    """
    try:
        body = json.loads((Path(build_dir) / PROVENANCE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return body.get("commit") if isinstance(body, dict) else None


@dataclass(frozen=True)
class Refresh:
    """One attempt at the headline. `published` False is a normal, survivable state."""
    published: bool
    reason: str
    path: Path | None = None
    effect_fraction: float | None = None
    #: What happened to the recorded-baseline carry-over: why it was or was not used.
    carryover: str | None = None

    def to_attempt(self) -> dict[str, Any]:
        return {"status": ("champion_vs_production" if self.published
                           else "champion_vs_production_unavailable"),
                "mechanism_id": MECHANISM_ID,
                # Deliberately None -- see the module docstring. The number is in the
                # bundle and in `reason`, never in the planner's marginals column.
                "effect_fraction": None,
                "reason": self.reason}


def _ensure_baseline(baseline_build: Path, commit: str, label: str,
                     build_baseline: Callable[[Path, str], Any] | None,
                     on_step: Callable[[str], Any]) -> None:
    """Frozen production, built at most once PER COMMIT. Raises `Unavailable`, never
    aborts. `commit`/`label` are the LIVE-resolved freeze, and the builder receives
    the commit so it can refuse a source tree that has not followed the promotion."""
    declared = declared_commit(baseline_build)
    if declared is not None and declared != commit:
        raise Unavailable(
            f"the cached baseline build at {baseline_build} declares commit "
            f"{declared[:12]}, not the frozen production kernel {commit[:12]}; "
            f"publishing it would put a number under a heading it does not measure")
    if is_built(baseline_build):
        return
    if build_baseline is None:
        raise Unavailable(
            f"there is no baseline build at {baseline_build} and no builder was "
            f"wired, so there is nothing to measure the champion against")
    on_step("champion-vs-production: building the frozen production baseline (once)")
    verdict = build_baseline(baseline_build, commit)
    if not getattr(verdict, "passed", False) or not is_built(baseline_build):
        raise Unavailable(
            f"the frozen production kernel {commit[:12]} would not build into "
            f"{baseline_build} ({getattr(verdict, 'reason', '') or 'no reason given'})")
    status.write_json(Path(baseline_build), PROVENANCE,
                      {"commit": commit, "label": label},
                      prefix=".prov-")


# --------------------------------------------------------------------------- #
# The recorded production baseline
# --------------------------------------------------------------------------- #


def baseline_filename(commit: str) -> str:
    """`production-baseline.<sha12>.json` -- keyed by the FROZEN commit it describes."""
    return f"production-baseline.{commit[:12]}.json"


def protocol(*, model: Path | str, surface: str, reps: int = 9,
             warmup_launches: int = bench.WARMUP_PAIRS,
             argv: Sequence[str] | None = None,
             argv_provenance: str = "bench.llama_bench_argv (live)") -> dict:
    """The measurement protocol a production arm was (or will be) taken under.

    `argv` defaults to the instrument's OWN argv (`bench.llama_bench_argv`) with the
    binary and hardening seed as placeholders, so the recorded protocol is what runs.
    Pass `argv` explicitly only to describe a historical measurement whose instrument
    has since changed -- and say so in `argv_provenance`.

    `surface` must be one of `bench.SURFACES`: those are the only shapes this protocol
    (and the llama-bench harness it describes) knows how to run or record. A surface
    from a different serving instrument (e.g. an `ab_probe_*` surface) is refused here
    rather than let `bench.SURFACES[surface]` raise an opaque `KeyError` -- this module
    does not support that instrument, and the caller (`write-baseline`) needs a message
    naming the problem, not a traceback.
    """
    if surface not in bench.SURFACES:
        raise ValueError(
            f"unknown surface {surface!r}; production.protocol only knows the "
            f"llama-bench surfaces {sorted(bench.SURFACES)}")
    pp, tg, ubatch = bench.SURFACES[surface]
    argv = (list(argv) if argv is not None else
            bench.llama_bench_argv("<llama-bench>", model, pp=pp, tg=tg, reps=reps,
                                   ubatch=ubatch))
    return {"harness": "llama-bench", "model": str(model), "surface": surface,
            "pp": pp, "tg": tg, "ubatch": ubatch, "reps": reps,
            "warmup_launches": warmup_launches, "sample_unit": bench.FLOOR_UNIT,
            "estimator": "median", "hardened": "--autokernel-harden" in argv,
            "argv": [str(item) for item in argv], "argv_provenance": argv_provenance}


def protocol_mismatch(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """The `PROTOCOL_MATCH_KEYS` on which two protocols differ, each with both values."""
    return [f"{key} (recorded {recorded.get(key)!r}, now {current.get(key)!r})"
            for key in PROTOCOL_MATCH_KEYS if recorded.get(key) != current.get(key)]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def read_host_facts(*, dmi_root: Path | str = "/sys/class/dmi/id",
                    proc_root: Path | str = "/proc",
                    node_root: Path | str = "/sys/devices/system/node",
                    drm_root: Path | str = "/sys/class/drm") -> dict:
    """Identity facts about this host that a BIOS, firmware or kernel change moves.

    Readable without root. An unreadable fact is None, and None is never compared:
    "could not read" is not "changed". Settings-only BIOS changes (memory interleave,
    memory speed) do NOT move any of these -- `HOST_CHANGES` is their record.
    """
    dmi = Path(dmi_root)
    facts: dict[str, Any] = {key: _read_text(dmi / key) for key in (
        "bios_vendor", "bios_version", "bios_date", "bios_release",
        "board_name", "product_name")}
    facts["kernel_release"] = os.uname().release if proc_root == "/proc" else \
        _read_text(Path(proc_root) / "sys" / "kernel" / "osrelease")
    cpu_model = None
    cpuinfo = _read_text(Path(proc_root) / "cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    facts["cpu_model"] = cpu_model
    try:
        nodes = [entry for entry in Path(node_root).iterdir()
                 if entry.name.startswith("node") and entry.name[4:].isdigit()]
        facts["numa_nodes"] = len(nodes) or None
    except OSError:
        facts["numa_nodes"] = None
    try:
        vbios = sorted({value for card in Path(drm_root).glob("card*")
                        if (value := _read_text(card / "device" / "vbios_version"))})
    except OSError:
        vbios = []
    facts["gpu_vbios"] = vbios or None
    return facts


def host_fact_changes(recorded: Mapping[str, Any] | None,
                      current: Mapping[str, Any]) -> list[str]:
    """Facts both sides could read that now differ."""
    if not recorded:
        return []
    return [f"{key} (recorded {recorded[key]!r}, now {current[key]!r})"
            for key in sorted(recorded)
            if recorded.get(key) is not None and current.get(key) is not None
            and recorded[key] != current[key]]


def _parse_when(text: str) -> datetime:
    when = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _admissible_samples(samples: Any) -> list[float]:
    if (not isinstance(samples, list)
            or len(samples) < headline_admissibility.MIN_LAUNCHES_PER_ARM
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                   for x in samples)):
        raise ValueError(
            f"a production baseline needs >= "
            f"{headline_admissibility.MIN_LAUNCHES_PER_ARM} finite positive launch "
            f"rates (the headline's per-arm minimum); got {samples!r:.200}")
    return [float(x) for x in samples]


def commits_between(tree: Path | str, measured: str, production: str) -> list[dict]:
    """Every commit in `measured..production`, with the files each touches. Read-only.

    The measured binary must be an ANCESTOR of the frozen commit: a record of some
    sibling's samples is not production's numbers. What the list does to the
    measured surface is a judgement the caller records in the lineage note.
    """
    done = subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor",
                           measured, production], capture_output=True, text=True,
                          timeout=120)
    if done.returncode != 0:
        raise ValueError(f"{measured[:12]} is not an ancestor of {production[:12]} in "
                         f"{tree}; its samples are not production's numbers")
    log = _read(Path(tree), "log", "--format=%x1e%H%x1f%s", "--name-only",
                f"{measured}..{production}")
    commits = []
    for record in log.split("\x1e"):
        if not record.strip():
            continue
        head, _, files = record.partition("\n")
        commit, _, subject = head.partition("\x1f")
        commits.append({"commit": commit.strip(), "subject": subject.strip(),
                        "files": [line.strip() for line in files.splitlines()
                                  if line.strip()]})
    return commits


def write_baseline(store: Path | str, *, production_commit: str, production_label: str,
                   measured_commit: str, samples: Sequence[float],
                   protocol: Mapping[str, Any], measured_at: str,
                   source: Mapping[str, Any], host_facts: Mapping[str, Any] | None,
                   host_facts_note: str, lineage: Mapping[str, Any] | None = None,
                   residency: Mapping[str, Any] | None = None, overwrite: bool = False,
                   now: Callable[[], str] = status._now) -> Path:
    """Write `production-baseline.<sha12>.json`: production's numbers, carried over.

    The promotion's entry point (CLI: `write-baseline`). Refuses an under-sized or
    non-finite sample vector, a non-40-hex commit, an unparseable date, and -- unless
    `overwrite` -- an existing record for the same freeze (an overwritten record is
    first preserved beside it as `.superseded-<ts>`, never discarded).
    """
    if len(production_commit) != 40 or any(c not in "0123456789abcdef"
                                            for c in production_commit):
        raise ValueError(f"production_commit must be a full 40-hex sha, got "
                         f"{production_commit!r}")
    values = _admissible_samples(list(samples))
    _parse_when(measured_at)
    missing = [key for key in PROTOCOL_MATCH_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}; build it with production.protocol")
    store = Path(store)
    target = store / baseline_filename(production_commit)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already records this freeze; pass "
                                  f"overwrite to supersede it")
        stamp = now().replace(":", "").replace("-", "")
        target.rename(target.with_name(f"{target.name}.superseded-{stamp}"))
    body = {
        "schema": BASELINE_SCHEMA,
        "production": {"commit": production_commit, "label": production_label},
        "measured_commit": measured_commit,
        "lineage": dict(lineage or {}),
        "protocol": dict(protocol),
        "samples": values,
        "launches": len(values),
        "median": statistics.median(values),
        "sample_unit": protocol.get("sample_unit", bench.FLOOR_UNIT),
        "measured_at": measured_at,
        "host": {"facts": None if host_facts is None else dict(host_facts),
                 "note": host_facts_note},
        "source": dict(source),
        "residency": None if residency is None else dict(residency),
        "written_at": now(),
        "writer": "autokernel.loop.production.write_baseline",
    }
    return status.write_json(store, target.name, body, prefix=".pbl-")


@dataclass(frozen=True)
class BaselineDecision:
    """Whether a recorded baseline may stand in for the production arm, and why."""
    usable: bool
    reason: str
    path: Path | None = None
    body: dict | None = None


def baseline_decision(store: Path | str, frozen_commit: str, *,
                      protocol: Mapping[str, Any],
                      max_age_days: float = BASELINE_MAX_AGE_DAYS,
                      host_facts: Mapping[str, Any] | None = None,
                      host_changes: Sequence[Mapping[str, Any]] = HOST_CHANGES,
                      clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                      ) -> BaselineDecision:
    """Match, staleness and integrity checks, in that order. Never raises."""
    path = Path(store) / baseline_filename(frozen_commit)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BaselineDecision(False, f"no recorded production baseline at {path}")
    except (OSError, ValueError) as exc:
        return BaselineDecision(False, f"the recorded baseline {path} is unreadable "
                                       f"({type(exc).__name__}: {exc})", path)
    try:
        if not isinstance(body, dict) or body.get("schema") != BASELINE_SCHEMA:
            return BaselineDecision(False, f"{path} is not a {BASELINE_SCHEMA} record",
                                    path)
        recorded = (body.get("production") or {}).get("commit")
        if recorded != frozen_commit:
            return BaselineDecision(False, f"{path} records production {recorded!r}, not "
                                           f"the frozen kernel {frozen_commit[:12]}", path)
        differs = protocol_mismatch(body.get("protocol") or {}, protocol)
        if differs:
            return BaselineDecision(False, f"protocol mismatch: {'; '.join(differs)}",
                                    path, body)
        measured = _parse_when(body["measured_at"])
        age_days = (clock() - measured).total_seconds() / 86400.0
        if age_days > max_age_days:
            return BaselineDecision(False, f"stale: measured {body['measured_at']}, "
                                           f"{age_days:.1f} days ago, past the "
                                           f"{max_age_days:g}-day limit",
                                    path, body)
        for change in host_changes:
            applied = _parse_when(change["applied"])
            if measured.date() <= applied.date():
                return BaselineDecision(
                    False, f"stale: measured {body['measured_at']}, on or before the "
                           f"recorded host change of {change['applied']} "
                           f"({change['what']}; {change.get('source')})", path, body)
        changed = host_fact_changes((body.get("host") or {}).get("facts"),
                                    host_facts if host_facts is not None
                                    else read_host_facts())
        if changed:
            return BaselineDecision(False, f"stale: host changed since the record: "
                                           f"{'; '.join(changed)}", path, body)
        _admissible_samples(body.get("samples"))
    except Exception as exc:  # noqa: BLE001 -- a malformed record is a fallback, never a crash
        return BaselineDecision(False, f"the recorded baseline {path} is malformed "
                                       f"({type(exc).__name__}: {exc})", path)
    return BaselineDecision(True, f"recorded production baseline {path.name} "
                                  f"({body['launches']} launches, measured "
                                  f"{body['measured_at']}) matches protocol and host",
                            path, body)


def _record_paired_baseline(store: Path, frozen_commit: str, frozen_label: str,
                            comparison: Any, protocol: Mapping[str, Any],
                            refused: str, now: Callable[[], str],
                            host_facts: Callable[[], dict]) -> str:
    """After a paired fallback: its production arm IS a fresh record. Never raises."""
    try:
        if getattr(comparison, "model", None) not in (None, protocol.get("model")) or \
                comparison.surface != protocol.get("surface"):
            return (f"not re-recorded: the comparison measured {comparison.model} "
                    f"{comparison.surface}, not the protocol's")
        path = write_baseline(
            store, production_commit=frozen_commit, production_label=frozen_label,
            measured_commit=frozen_commit, samples=comparison.anchor_samples,
            protocol=protocol, measured_at=now(),
            source={"kind": "paired champion-vs-production refresh, production arm",
                    "arm": "anchor_samples", "replaced_because": refused},
            host_facts=host_facts(), host_facts_note="captured at the measurement",
            residency=comparison.residency, overwrite=True, now=now)
        return f"re-recorded production's arm as {path.name}"
    except Exception as exc:  # noqa: BLE001 -- reporting only
        return f"not re-recorded ({type(exc).__name__}: {exc})"


def _publish_unpaired(store: Path, decision: BaselineDecision, *, champion_commit: str,
                      champion_build: Path, measure: Callable[[Path], Any],
                      frozen_commit: str, frozen_label: str, note: str | None,
                      now: Callable[[], str], on_step: Callable[[str], Any]) -> Refresh:
    """Champion measured ALONE against the recorded production samples."""
    body = decision.body or {}
    baseline = [float(x) for x in body["samples"]]
    on_step("champion-vs-production: champion alone vs the RECORDED production baseline")
    measured = measure(Path(champion_build))
    model = getattr(measured, "model", None)
    if model is not None and model != body["protocol"]["model"]:
        raise Unavailable(f"the champion was measured on {model}, the record on "
                          f"{body['protocol']['model']}")
    if measured.surface != body["protocol"]["surface"]:
        raise Unavailable(f"the champion was measured on {measured.surface}, the record "
                          f"on {body['protocol']['surface']}")
    champion = list(measured.samples)
    ci = headline_admissibility.unpaired_confidence_interval(baseline, champion)
    effect = statistics.median(champion) / statistics.median(baseline) - 1.0
    record_sha = _sha256(decision.path)
    evidence = status.write_json(Path(store),
                                 f"champion-vs-production.{champion_commit[:12]}.json", {
        "comparison": "unpaired_recorded_baseline",
        "effect": effect, "effect_pct": effect * 100.0,
        "estimator": "median_over_median",
        "baseline_record": str(decision.path), "baseline_record_sha256": record_sha,
        "baseline_measured_at": body["measured_at"], "baseline_samples": baseline,
        "champion": measured.to_dict(), "caveat": UNPAIRED_CAVEAT,
        "host_drift_caveat_pct": HOST_DRIFT_CAVEAT_PCT}, prefix=".cvp-")
    surface = measured.surface
    target = status.write_json(Path(store), FILENAME, {
        "schema": SCHEMA, "generated_at": now(), "stale_after_s": STALE_AFTER_S,
        "baseline": {"commit": frozen_commit, "label": frozen_label,
                     "build": None, "recorded": str(decision.path),
                     "recorded_sha256": record_sha,
                     "recorded_measured_at": body["measured_at"],
                     "recorded_measured_commit": body.get("measured_commit")},
        "champion": {"commit": champion_commit, "build": str(champion_build)},
        "effect_fraction": float(effect),
        "metric": f"{surface}_tok_s", "metric_direction": "higher_better",
        "model": body["protocol"]["model"], "surface": surface,
        # No pairs: nothing alternated. A count here would claim an A/B nobody ran.
        "pairs": None, "comparison": "unpaired_recorded_baseline",
        "launches": len(baseline) + len(champion),
        "champion_launches": len(champion), "baseline_launches": len(baseline),
        "headline_admissibility": {**headline_admissibility.contract(),
                                   "ci_method": headline_admissibility.UNPAIRED_CI_METHOD},
        "confidence_interval": ci,
        # The paired floor does not apply to an unpaired number; the drift caveat does.
        "noise_floor_pct": None,
        "host_drift_caveat_pct": HOST_DRIFT_CAVEAT_PCT, "caveat": UNPAIRED_CAVEAT,
        "baseline_carryover": {"used": True, "reason": decision.reason},
        "evidence": str(evidence), "mechanism_id": MECHANISM_ID,
        **({"anchor_guard_excursion": note} if note else {}),
    }, prefix=".cvp-")
    return Refresh(
        True,
        f"champion {champion_commit[:12]} measures {effect * 100.0:+.3f}% against the "
        f"RECORDED frozen production {frozen_commit[:12]} ({frozen_label}) baseline of "
        f"{body['measured_at'][:10]}, UNPAIRED: {len(champion)} champion launches vs "
        f"{len(baseline)} recorded {surface} launches, host-drift caveat "
        f"~{HOST_DRIFT_CAVEAT_PCT:g}%",
        target, float(effect), carryover=decision.reason)


def refresh(*, store: Path, champion_commit: str, champion_build: Path,
            compare: Callable[[Path, Path], Any],
            baseline_build: Path | None = None,
            baseline_root: Path = BASELINE_ROOT,
            resolve: Callable[[], tuple[str, str]] = resolve_frozen,
            build_baseline: Callable[[Path, str], Any] | None = None,
            on_step: Callable[[str], Any] = lambda _label: None, note: str | None = None,
            now: Callable[[], str] = status._now,
            measure: Callable[[Path], Any] | None = None,
            protocol: Mapping[str, Any] | None = None,
            baseline_max_age_days: float = BASELINE_MAX_AGE_DAYS,
            host_facts: Callable[[], dict] = read_host_facts,
            host_changes: Sequence[Mapping[str, Any]] = HOST_CHANGES,
            clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
            ) -> Refresh:
    """Measure the champion against frozen production and publish it. NEVER raises.

    CARRY-OVER FIRST. With `measure(champion_build) -> bench.Measurement` and the
    `protocol` it runs under both wired, a `production-baseline.<sha12>.json` for the
    LIVE frozen commit that `baseline_decision` admits replaces the production arm:
    the champion is measured alone, nothing is built, and the bundle is marked
    UNPAIRED with the host-drift caveat. Any refusal (absent, other protocol, older
    than `baseline_max_age_days`, across a recorded host change, host facts moved) is
    recorded in the bundle's `baseline_carryover` and the paired A/B below runs as
    before -- after which its production arm is re-recorded for the next refresh.

    `compare(baseline_build, champion_build) -> bench.Comparison` is injected, so this
    is exercised with no GPU, no ROCm and no build. The BASELINE is the first arm, so
    a positive effect reads as "the champion is faster than production" -- the sign
    the headline is stated in.

    Everything the bundle says about the measurement is taken from the `Comparison`
    the benchmark returned, never from a caller's parameters: a surface or pair count
    passed in alongside the comparison is a second source of truth for one fact, and
    the one that gets published is the one nobody measured.
    """
    try:
        # Resolved LIVE, inside the containment: a promotion moves the frozen tree, so
        # a resolver failure (or a tree off the production branch contract) refuses
        # THIS refresh and never ends the run.
        frozen_commit, frozen_label = resolve()
        if measure is not None and protocol is not None:
            decision = baseline_decision(
                store, frozen_commit, protocol=protocol,
                max_age_days=baseline_max_age_days, host_facts=host_facts(),
                host_changes=host_changes, clock=clock)
            if decision.usable:
                return _publish_unpaired(
                    Path(store), decision, champion_commit=champion_commit,
                    champion_build=Path(champion_build), measure=measure,
                    frozen_commit=frozen_commit, frozen_label=frozen_label, note=note,
                    now=now, on_step=on_step)
            carryover = f"recorded baseline NOT used, paired A/B instead: {decision.reason}"
        else:
            carryover = ("recorded baseline NOT consulted: the caller wired no "
                         "single-arm measurer and protocol")
        on_step(f"champion-vs-production: {carryover}")
        slot = (Path(baseline_build) if baseline_build is not None
                else baseline_slot(frozen_commit, root=baseline_root))
        _ensure_baseline(slot, frozen_commit, frozen_label, build_baseline, on_step)
        on_step("champion-vs-production: A/B against the frozen production kernel")
        comparison = compare(slot, Path(champion_build))
        headline_ci = headline_admissibility.confidence_interval(
            comparison.anchor_samples, comparison.candidate_samples)
        # Per-champion, so publishing a new bundle never overwrites the raw record the
        # PREVIOUS bundle points at. `evidence` in a superseded bundle must still
        # resolve, or the number it carries stops being auditable the moment it ages.
        evidence = status.write_json(
            Path(store), f"champion-vs-production.{champion_commit[:12]}.json",
            comparison.to_dict(), prefix=".cvp-")
        target = status.write_json(Path(store), FILENAME, {
            "schema": SCHEMA,
            "generated_at": now(),
            "stale_after_s": STALE_AFTER_S,
            "baseline": {"commit": frozen_commit, "label": frozen_label,
                         "build": str(slot)},
            "champion": {"commit": champion_commit, "build": str(champion_build)},
            "effect_fraction": float(comparison.effect),
            "metric": f"{comparison.surface}_tok_s",
            "metric_direction": "higher_better",
            # The rung the headline was measured on, off the comparison like every
            # other fact here (§5.3): an off-production-shape headline must SAY so.
            "model": comparison.model,
            "surface": comparison.surface,
            "pairs": comparison.pairs,
            "launches": len(comparison.anchor_samples) + len(comparison.candidate_samples),
            "headline_admissibility": headline_admissibility.contract(),
            "confidence_interval": headline_ci,
            "noise_floor_pct": comparison.noise_floor_pct,
            "evidence": str(evidence),
            "mechanism_id": MECHANISM_ID,
            "comparison": "paired_ab",
            "baseline_carryover": {"used": False, "reason": carryover},
            # R22-3: present only when the promotion's anchor guard recorded an
            # instrument excursion. The headline still publishes -- the anchor is
            # hash-proven -- but the number was taken in a session whose A/A read
            # above the floor, and a reader weighing the number needs to know.
            **({"anchor_guard_excursion": note} if note else {}),
        }, prefix=".cvp-")
        # Built INSIDE the containment. The headline sentence formats fields off the
        # comparison, and a formatting error on the SUCCESS path would otherwise be
        # the one exception in this module that still reached the loop.
        published = Refresh(
            True,
            f"champion {champion_commit[:12]} measures "
            f"{comparison.effect * 100.0:+.3f}% against frozen production "
            f"{frozen_commit[:12]} ({frozen_label}) over {comparison.pairs} "
            f"{comparison.surface} pairs, floor {comparison.noise_floor_pct}%",
            target, float(comparison.effect), carryover=carryover)
        if protocol is not None:
            # The production arm just measured IS production's numbers under this
            # protocol, on this host, today: record it so the next refresh is
            # single-arm. Outside the bundle's success path; never raises.
            recorded = _record_paired_baseline(
                Path(store), frozen_commit, frozen_label, comparison, protocol,
                carryover, now, host_facts)
            published = Refresh(published.published, published.reason, published.path,
                                published.effect_fraction,
                                carryover=f"{carryover}; {recorded}")
    except (Unavailable, headline_admissibility.HeadlineInadmissible) as exc:
        return Refresh(False, f"champion-vs-production NOT refreshed: {exc}. The "
                              f"previous bundle stands, so the headline reads "
                              f"SUPERSEDED rather than a number measured against "
                              f"something that is not frozen production")
    except Exception as exc:  # noqa: BLE001 -- reporting must never end a run
        return Refresh(False, f"champion-vs-production refresh FAILED "
                              f"({type(exc).__name__}: {exc}). The previous bundle "
                              f"stands and the run continues; this is a reporting "
                              f"refresh, not a correctness gate")
    return published


def _dig(body: Any, dotted: str) -> tuple[Any, Any]:
    """`(value, parent)` at a dotted path such as `g5_full.candidate_samples`."""
    parent, value = None, body
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"no {dotted!r} in the source (stopped at {part!r})")
        parent, value = value, value[part]
    return value, parent


def main(argv: Sequence[str] | None = None) -> int:
    """`python3 -m autokernel.loop.production write-baseline ...` (cwd scripts/kernel_rnd).

    The kernel-promotion step: record the promoted champion's last standing or fold
    samples as the new freeze's production baseline. Prints the written record's path
    and a one-line summary; exit 0 written, 2 refused.
    """
    parser = argparse.ArgumentParser(prog="autokernel.loop.production")
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write-baseline", help="record production-baseline.<sha12>.json")
    write.add_argument("--source", required=True, type=Path,
                       help="JSON holding the samples (a fold result, a standing "
                            "champion-vs-production.<sha12>.json evidence file, ...)")
    write.add_argument("--samples-path", default="candidate_samples",
                       help="dotted path to the champion's sample list in --source "
                            "(e.g. g5_full.candidate_samples); its parent supplies "
                            "model/surface/residency when present")
    write.add_argument("--measured-commit", required=True,
                       help="commit of the binary that produced the samples")
    write.add_argument("--production-commit",
                       help="the frozen commit (default: resolved live from the frozen tree)")
    write.add_argument("--production-label", help="e.g. production-consolidated-v11")
    write.add_argument("--lineage-tree", type=Path, default=FROZEN_TREE,
                       help="git repo for `git log measured..production` (read-only)")
    write.add_argument("--lineage-note", default="",
                       help="why the commits between do not move the measured surface")
    write.add_argument("--model", help="default: the samples' parent `model`")
    write.add_argument("--surface", help="default: the samples' parent `surface`")
    write.add_argument("--reps", type=int, default=9)
    write.add_argument("--argv-json", help="the exact argv list, as JSON, when the "
                                           "measurement predates the live instrument")
    write.add_argument("--argv-provenance", default="bench.llama_bench_argv (live)")
    write.add_argument("--measured-at", help="ISO-8601 (default: --source mtime, UTC)")
    write.add_argument("--host-facts", choices=("capture", "none"), default="capture",
                       help="capture this host's facts now (only when the samples were "
                            "taken on this host as it is today), or record none")
    write.add_argument("--host-facts-note", default="")
    write.add_argument("--store", type=Path, default=DEFAULT_STORE)
    write.add_argument("--overwrite", action="store_true")
    write.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        source_path = args.source.resolve()
        body = json.loads(source_path.read_text(encoding="utf-8"))
        samples, parent = _dig(body, args.samples_path)
        parent = parent if isinstance(parent, dict) else {}
        model = args.model or parent.get("model")
        surface = args.surface or parent.get("surface")
        if not model or not surface:
            raise ValueError("model and surface must come from the source or the flags")
        for key, given in (("model", model), ("surface", surface)):
            if parent.get(key) not in (None, given):
                raise ValueError(f"--{key} {given!r} contradicts the source's "
                                 f"{parent.get(key)!r}")
        if args.production_commit:
            commit = _read(args.lineage_tree, "rev-parse", args.production_commit) \
                if len(args.production_commit) != 40 else args.production_commit
            label = args.production_label or ""
        else:
            commit, label = resolve_frozen()
            label = args.production_label or label
        if not label.startswith(FROZEN_BRANCH_PREFIX):
            raise ValueError(f"production label {label!r} is not a "
                             f"'{FROZEN_BRANCH_PREFIX}*' freeze")
        measured = (args.measured_commit if len(args.measured_commit) == 40 else
                    _read(args.lineage_tree, "rev-parse", args.measured_commit))
        between = commits_between(args.lineage_tree, measured, commit)
        measured_at = args.measured_at or datetime.fromtimestamp(
            source_path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
        record = dict(
            production_commit=commit, production_label=label, measured_commit=measured,
            samples=samples,
            protocol=protocol(model=model, surface=surface, reps=args.reps,
                              argv=(json.loads(args.argv_json) if args.argv_json
                                    else None),
                              argv_provenance=args.argv_provenance),
            measured_at=measured_at,
            source={"path": str(source_path), "sha256": _sha256(source_path),
                    "samples_path": args.samples_path,
                    "measured_at_source": ("--measured-at" if args.measured_at
                                           else "source mtime")},
            host_facts=read_host_facts() if args.host_facts == "capture" else None,
            host_facts_note=args.host_facts_note or (
                "captured at baseline write" if args.host_facts == "capture"
                else "not captured"),
            lineage={"tree": str(args.lineage_tree), "commits_between": between,
                     "note": args.lineage_note,
                     "verified_by": f"git log --name-only {measured[:12]}..{commit[:12]}"},
            residency=parent.get("residency"))
        if args.dry_run:
            _admissible_samples(list(samples))
            print(json.dumps(record, indent=2, sort_keys=True, default=str))
            return 0
        path = write_baseline(args.store, overwrite=args.overwrite, **record)
    except (ValueError, OSError, Unavailable) as exc:
        print(f"write-baseline REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"{path}  production {commit[:12]} ({label})  {len(samples)} launches, "
          f"median {statistics.median(samples):.3f}  measured {measured[:12]} "
          f"+{len(between)} commits  at {measured_at}")
    return 0


__all__ = ["BASELINE_MAX_AGE_DAYS", "BASELINE_ROOT", "BASELINE_SCHEMA", "BASELINE_TREE",
           "BaselineDecision", "DEFAULT_STORE", "FILENAME", "FROZEN_BRANCH_PREFIX",
           "FROZEN_TREE", "HOST_CHANGES", "HOST_DRIFT_CAVEAT_PCT", "LEGACY_BUILD",
           "LEGACY_COMMIT", "MECHANISM_ID", "PROTOCOL_MATCH_KEYS", "PROVENANCE",
           "SCHEMA", "STALE_AFTER_S", "UNPAIRED_CAVEAT", "Refresh", "Unavailable",
           "baseline_decision", "baseline_filename", "baseline_slot", "commits_between",
           "declared_commit", "host_fact_changes", "is_built", "main", "protocol",
           "protocol_mismatch", "read_host_facts", "refresh", "resolve_frozen",
           "write_baseline"]


if __name__ == "__main__":
    raise SystemExit(main())
