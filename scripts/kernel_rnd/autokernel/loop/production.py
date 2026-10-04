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

THE SERVING-PROBE PROTOCOL KIND (2026-10-04). Operator: measure v10 ONCE on the current
DFlash2 serving protocol with >= 14 paired launches (champion vs production); future
champions then run single-arm against that baseline. `llama-bench` cannot drive DFlash2
speculative decoding, so a second protocol kind exists beside the llama-bench one:
`serving_probe` (`serving_protocol`). Its identity is the server argv (the :8083
production argv, binary swapped for `<llama-server>`), the probe tool and version, the
request set (canonical-JSON digest of the request manifest: prompts, n_predict,
sampling), the metric (`pp` = timings.prompt_per_second, `tg` =
timings.predicted_per_second) and the unit (one server launch = one sample). A record
stands in for production only on the same normalised argv, request digest and metric
(`SERVING_PROBE_MATCH_KEYS`), under the same >= 14-launch floor and the same staleness
rules (age, `HOST_CHANGES`, host facts). This module never launches a server: the stack
owner's runner produces the samples, and `ingest_serving_result` (CLI `ingest-serving`)
turns its result JSON (`SERVING_RESULT_SCHEMA`, documented at that function) into the
production baselines and the champion's standing receipt. `refresh` takes the serving
path when handed a serving protocol, and has NO llama-bench fallback there -- a paired
llama-bench A/B would measure a different protocol under the serving headline's name.
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

#: ---- Protocol kinds (see the module docstring's SERVING-PROBE section) ----
PROTOCOL_KIND_LLAMA_BENCH = "llama_bench"
PROTOCOL_KIND_SERVING_PROBE = "serving_probe"
PROTOCOL_KINDS = (PROTOCOL_KIND_LLAMA_BENCH, PROTOCOL_KIND_SERVING_PROBE)
#: One server launch is one sample: the probe's per-launch median over its reps.
SERVING_SAMPLE_UNIT = "server_launch"
#: metric -> the llama-server `timings` field it is read from.
SERVING_METRICS = {"pp": "prompt_per_second", "tg": "predicted_per_second"}
#: metric -> the per-launch key in a runner result's `speed` block (ab_probe.py's names).
SERVING_LAUNCH_FIELDS = {"pp": "pp_tps_median", "tg": "tg_tps_median"}
#: The binary is the one argv element that may differ between arms and runs ...
SERVING_BINARY_PLACEHOLDER = "<llama-server>"
#: ... except `--slot-save-path`, which the runner (`ab_probe.py args --slot-dir
#: <run>/slots`) points into each run's own directory. Slot files are written only on
#: an explicit slot save, which the probe never issues (cache_prompt=false), so the
#: value cannot move a timing -- and left in, it would make every run a new protocol.
SERVING_SLOT_PLACEHOLDER = "<slot-save-path>"
#: A serving surface is a free-form label the runner names (`serving_ab_probe_tg256`,
#: `serving_ab_probe_pp6465`, ...) under this shape; it is matched like any key.
SERVING_SURFACE_PREFIX = "serving_"
#: The runner result schemas `ingest_serving_result` reads (one shape, two names: the
#: first is what the stack owner's `paired_receipt.py` emits today).
SERVING_RESULT_SCHEMAS = ("epyc.fold.serving_ab_samples_for_production_writer.v1",
                          "epyc.autokernel.serving_probe_result.v1")
#: Per-launch sample units a runner may declare. One server launch IS one process
#: launch -- the floor unit `bench.FLOOR_UNIT` names -- so "process" is the same unit.
SERVING_ACCEPTED_UNITS = (SERVING_SAMPLE_UNIT, bench.FLOOR_UNIT)
#: The serving standing receipt. NOT `FILENAME`: the llama-bench headline bundle keeps
#: its own file, and a serving number never silently replaces it.
SERVING_FILENAME = "champion-vs-production.serving_probe.json"
#: What must be IDENTICAL for a serving record to stand in for production. The argv is
#: compared with the binary normalised away; `request_digest` covers prompts, n_predict
#: and sampling; probe tool/version are recorded identity but NOT matched (operator
#: rule: same argv-but-binary, same request digest, same metric). `placement` (the
#: numactl/taskset/env the runner wraps the server in, absent from argv) IS matched:
#: a different CPU placement is a different measurement of the same binary.
SERVING_PROBE_MATCH_KEYS = ("kind", "harness", "model", "surface", "metric", "timing_field",
                            "sample_unit", "server_argv", "placement", "request_digest")


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


def baseline_filename(commit: str, protocol: Mapping[str, Any] | None = None) -> str:
    """`production-baseline.<sha12>.json` -- keyed by the FROZEN commit it describes.

    A serving-probe record is keyed by commit AND metric
    (`production-baseline.<sha12>.serving_probe-<pp|tg>.json`), so it never collides
    with the llama-bench record of the same freeze, nor pp with tg."""
    if protocol is not None and protocol_kind(protocol) == PROTOCOL_KIND_SERVING_PROBE:
        return f"production-baseline.{commit[:12]}.serving_probe-{protocol['metric']}.json"
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
    return {"kind": PROTOCOL_KIND_LLAMA_BENCH,
            "harness": "llama-bench", "model": str(model), "surface": surface,
            "pp": pp, "tg": tg, "ubatch": ubatch, "reps": reps,
            "warmup_launches": warmup_launches, "sample_unit": bench.FLOOR_UNIT,
            "estimator": "median", "hardened": "--autokernel-harden" in argv,
            "argv": [str(item) for item in argv], "argv_provenance": argv_provenance}


def protocol_kind(protocol: Mapping[str, Any]) -> str:
    """A protocol's kind. A record that names none predates the field: llama-bench."""
    return protocol.get("kind") or PROTOCOL_KIND_LLAMA_BENCH


#: `write_baseline` takes a `protocol_kind` PARAMETER (the CLI's `--protocol-kind`),
#: which shadows the function inside it.
_kind_of = protocol_kind


def match_keys(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    """The keys that must be identical for a record under this protocol to stand in."""
    kind = protocol_kind(protocol)
    if kind == PROTOCOL_KIND_SERVING_PROBE:
        return SERVING_PROBE_MATCH_KEYS
    if kind == PROTOCOL_KIND_LLAMA_BENCH:
        return PROTOCOL_MATCH_KEYS
    raise ValueError(f"unknown protocol kind {kind!r}; known: {list(PROTOCOL_KINDS)}")


def protocol_mismatch(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """The match keys on which two protocols differ, each with both values.

    A different KIND is reported alone: two instruments share no comparable field."""
    if protocol_kind(recorded) != protocol_kind(current):
        return [f"kind (recorded {protocol_kind(recorded)!r}, "
                f"now {protocol_kind(current)!r})"]
    return [f"{key} (recorded {recorded.get(key)!r}, now {current.get(key)!r})"
            for key in match_keys(current) if recorded.get(key) != current.get(key)]


def canonical_digest(value: Any) -> str:
    """sha256 of `value` as canonical JSON (sorted keys, no whitespace, UTF-8).

    THE request-digest definition: a runner that computes it itself must use exactly
    `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_hex(text: Any, length: int | None = None) -> bool:
    return (isinstance(text, str) and bool(text)
            and (length is None or len(text) == length)
            and all(c in "0123456789abcdef" for c in text))


def normalize_server_argv(argv: Sequence[Any]) -> tuple[list[str], str]:
    """`(argv with the binary and the slot-save path normalised, the binary)`.

    The binary is the FIRST element whose basename is `llama-server` (or the
    placeholder itself); anything before it (a numactl/taskset spawn prefix) stays and
    is therefore matched -- placement is part of the protocol. An argv naming no
    `llama-server` is refused rather than guessing which element is the binary. The
    value after `--slot-save-path` becomes `SERVING_SLOT_PLACEHOLDER` (see there).
    Every other element, the port included, is compared verbatim."""
    items = [str(item) for item in argv]
    for index, item in enumerate(items):
        if item == SERVING_BINARY_PLACEHOLDER or Path(item).name == "llama-server":
            out = items[:index] + [SERVING_BINARY_PLACEHOLDER] + items[index + 1:]
            for at in range(index + 1, len(out) - 1):
                if out[at] == "--slot-save-path":
                    out[at + 1] = SERVING_SLOT_PLACEHOLDER
            return out, item
    raise ValueError(f"server argv names no llama-server binary: {items!r:.200}")


def _argv_model(argv: Sequence[str]) -> str | None:
    for flag in ("-m", "--model"):
        if flag in argv[:-1]:
            return argv[list(argv).index(flag) + 1]
    return None


def request_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Validate a request manifest and return its canonical digest (`request_digest`).

    A manifest is any non-empty JSON object; EVERY key in it is pinned by the digest.
    Recommended keys: `prompts` (or `prompt_set_digest`, 64 hex), `n_predict`,
    `sampling` -- read off into the protocol when present. A runner whose prompts and
    sampling live in its own source (ab_probe.py) pins them through the probe command
    plus `probe_version` instead; see `serving_protocol_from_runner`."""
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError("a request manifest must be a non-empty JSON object")
    if "sampling" in manifest and not isinstance(manifest["sampling"], Mapping):
        raise ValueError("request manifest `sampling` must be an object")
    if "prompt_set_digest" in manifest and not _is_hex(manifest["prompt_set_digest"], 64):
        raise ValueError("request manifest `prompt_set_digest` must be 64 lowercase hex")
    return canonical_digest(dict(manifest))


def serving_protocol(*, server_argv: Sequence[Any], metric: str, probe_tool: str,
                     probe_version: str,
                     request_manifest: Mapping[str, Any] | None = None,
                     request_digest: str | None = None,
                     n_predict: Any = None, sampling: Mapping[str, Any] | None = None,
                     surface: str | None = None, placement: str | None = None,
                     model: str | None = None,
                     argv_provenance: str = "runner result server_argv",
                     request_manifest_source: str = "runner manifest") -> dict:
    """The `serving_probe` protocol one metric of a serving probe is taken under.

    Pass the request manifest (preferred: its digest is computed here, and n_predict /
    sampling are read off it) or, for a run whose manifest is not at hand, its
    `request_digest` with the n_predict / sampling it pinned. Both may be given only
    when they agree. `surface` defaults to `serving_probe_<metric>`; a runner's own
    label (`serving_ab_probe_tg256`) is accepted when it starts `serving_`. `model`
    defaults to the argv's `-m` and, when given, must agree with it."""
    if metric not in SERVING_METRICS:
        raise ValueError(f"unknown serving metric {metric!r}; known: "
                         f"{sorted(SERVING_METRICS)}")
    for name, value in (("probe_tool", probe_tool), ("probe_version", probe_version)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    argv, _binary = normalize_server_argv(server_argv)
    surface = surface or f"{SERVING_SURFACE_PREFIX}probe_{metric}"
    if (not isinstance(surface, str) or not surface.startswith(SERVING_SURFACE_PREFIX)
            or not surface.replace("_", "").replace("-", "").isalnum()):
        raise ValueError(f"serving surface {surface!r} must be '{SERVING_SURFACE_PREFIX}*' "
                         f"in [A-Za-z0-9_-]")
    argv_model = _argv_model(argv)
    if model is not None and argv_model is not None and str(model) != argv_model:
        raise ValueError(f"model {model!r} contradicts the server argv's -m {argv_model!r}")
    prompt_set_digest = None
    if request_manifest is not None:
        digest = request_manifest_digest(request_manifest)
        if request_digest is not None and request_digest != digest:
            raise ValueError(f"request_digest {request_digest!r} is not the manifest's "
                             f"digest {digest}")
        request_digest = digest
        n_predict = request_manifest.get("n_predict", n_predict)
        sampling = request_manifest.get("sampling", sampling)
        prompt_set_digest = request_manifest.get("prompt_set_digest") or (
            canonical_digest(request_manifest["prompts"])
            if "prompts" in request_manifest else None)
    elif not _is_hex(request_digest, 64):
        raise ValueError("a serving protocol needs a request manifest or a 64-hex "
                         f"request_digest, got {request_digest!r}")
    return {"kind": PROTOCOL_KIND_SERVING_PROBE, "harness": "serving_probe",
            "probe_tool": probe_tool, "probe_version": probe_version,
            "model": argv_model or (None if model is None else str(model)),
            "surface": surface, "placement": placement,
            "metric": metric, "timing_field": SERVING_METRICS[metric],
            "unit": "tok/s (server timings)", "sample_unit": SERVING_SAMPLE_UNIT,
            "estimator": "median", "server_argv": argv,
            "server_argv_sha256": canonical_digest(argv),
            "request_digest": request_digest, "prompt_set_digest": prompt_set_digest,
            "n_predict": n_predict, "sampling": None if sampling is None else dict(sampling),
            "request_manifest": None if request_manifest is None else dict(request_manifest),
            "request_manifest_source": (request_manifest_source if request_manifest
                                        is not None else "digest only"),
            "argv_provenance": argv_provenance}


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
                   now: Callable[[], str] = status._now,
                   protocol_kind: str | None = None) -> Path:
    """Write `production-baseline.<sha12>.json`: production's numbers, carried over.

    The promotion's entry point (CLI: `write-baseline`). Refuses an under-sized or
    non-finite sample vector, a non-40-hex commit, an unparseable date, and -- unless
    `overwrite` -- an existing record for the same freeze (an overwritten record is
    first preserved beside it as `.superseded-<ts>`, never discarded).

    `protocol_kind` (`llama_bench` | `serving_probe`) is read off `protocol`; passing
    it explicitly asserts it, and a disagreement is refused. A serving-probe record is
    built with `serving_protocol` and lands under its own per-metric filename.
    """
    if len(production_commit) != 40 or any(c not in "0123456789abcdef"
                                            for c in production_commit):
        raise ValueError(f"production_commit must be a full 40-hex sha, got "
                         f"{production_commit!r}")
    kind = _kind_of(protocol)
    if protocol_kind is not None and protocol_kind != kind:
        raise ValueError(f"protocol_kind {protocol_kind!r} contradicts the protocol's "
                         f"kind {kind!r}")
    values = _admissible_samples(list(samples))
    _parse_when(measured_at)
    missing = [key for key in match_keys(protocol) if key not in protocol]
    if missing:
        builder = ("production.serving_protocol" if kind == PROTOCOL_KIND_SERVING_PROBE
                   else "production.protocol")
        raise ValueError(f"protocol lacks {missing}; build it with {builder}")
    store = Path(store)
    target = store / baseline_filename(production_commit, protocol)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already records this freeze; pass "
                                  f"overwrite to supersede it")
        stamp = now().replace(":", "").replace("-", "")
        target.rename(target.with_name(f"{target.name}.superseded-{stamp}"))
    body = {
        "schema": BASELINE_SCHEMA,
        "protocol_kind": kind,
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
    """Match, staleness and integrity checks, in that order. Never raises.

    Kind-aware: a serving protocol reads the per-metric serving record and matches on
    `SERVING_PROBE_MATCH_KEYS`; the age, host-change and host-fact rules are shared."""
    try:
        path = Path(store) / baseline_filename(frozen_commit, protocol)
    except Exception as exc:  # noqa: BLE001 -- a malformed protocol is a refusal
        return BaselineDecision(False, f"cannot name the record for this protocol "
                                       f"({type(exc).__name__}: {exc})")
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

    SERVING TARGETS. A `protocol` of kind `serving_probe` (`serving_protocol`,
    `read_serving_result(...).protocols[metric]`) takes the serving path: `measure`
    returns the champion's serving samples (`serving_measurement`), the matching
    `production-baseline.<sha12>.serving_probe-<metric>.json` must be admitted, and the
    UNPAIRED receipt lands in `SERVING_FILENAME`. There is NO paired fallback -- a
    refusal is a non-published `Refresh` telling the operator to run the paired runner
    and `ingest-serving` it; `compare` and `build_baseline` are never called.
    """
    try:
        # Resolved LIVE, inside the containment: a promotion moves the frozen tree, so
        # a resolver failure (or a tree off the production branch contract) refuses
        # THIS refresh and never ends the run.
        frozen_commit, frozen_label = resolve()
        if protocol is not None and _kind_of(protocol) == PROTOCOL_KIND_SERVING_PROBE:
            return _refresh_serving(
                store=Path(store), protocol=protocol, frozen_commit=frozen_commit,
                frozen_label=frozen_label, champion_commit=champion_commit,
                champion_build=Path(champion_build), measure=measure,
                max_age_days=baseline_max_age_days, host_facts=host_facts,
                host_changes=host_changes, clock=clock, now=now, note=note,
                on_step=on_step)
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


# --------------------------------------------------------------------------- #
# The serving-probe protocol kind: runner results in, baselines and receipts out
# --------------------------------------------------------------------------- #


def serving_protocol_from_runner(runner: Mapping[str, Any], *, metric: str, surface: str,
                                 probe_version: str | None = None) -> dict:
    """The `serving_probe` protocol for one metric of a runner's `protocol` block.

    Read off the runner's block AS IT IS (the stack owner's `paired_receipt.py` shape):
      * `argv`            -> server_argv (binary / slot-save path normalised)
      * `placement`       -> placement (matched; optional)
      * `model`           -> model (must agree with argv `-m`)
      * `sample_unit`     -> must be "process" or "server_launch" (the same unit)
      * `probe`           -> the probe command; its first token's basename is the probe
                             tool unless `probe_tool` is given
      * `probe_version`   -> REQUIRED here or as the `probe_version` argument (CLI
                             `--probe-version`); recommended: sha256 of the probe source
      * `requests` / `request_digest` -> the request manifest / its digest, when the
                             runner has one. Absent both, the manifest is DERIVED from
                             the block's own `probe`, `pp`, `tg`, `reps`,
                             `warmup_launches` -- ab_probe.py's prompts and sampling are
                             constants in its source, pinned by the command plus
                             `probe_version` -- and the protocol says so in
                             `request_manifest_source`.
    """
    if not isinstance(runner, Mapping):
        raise ValueError("the runner's `protocol` must be a JSON object")
    argv = runner.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError("the runner's protocol carries no `argv` list")
    unit = runner.get("sample_unit", SERVING_SAMPLE_UNIT)
    if unit not in SERVING_ACCEPTED_UNITS:
        raise ValueError(f"runner sample_unit {unit!r} is not one server launch per "
                         f"sample ({list(SERVING_ACCEPTED_UNITS)})")
    command = runner.get("probe")
    tool = runner.get("probe_tool") or (Path(str(command).split()[0]).name
                                        if isinstance(command, str) and command.strip()
                                        else None)
    recorded_version = runner.get("probe_version")
    if recorded_version and probe_version and recorded_version != probe_version:
        raise ValueError(f"--probe-version {probe_version!r} contradicts the runner's "
                         f"protocol.probe_version {recorded_version!r}")
    version = recorded_version or probe_version
    if not tool:
        raise ValueError("the runner's protocol names no probe: add `protocol.probe` "
                         "(the probe command) or `protocol.probe_tool`")
    if not version:
        raise ValueError("the runner's protocol carries no `probe_version`: add "
                         "`protocol.probe_version` (e.g. the sha256 of the probe "
                         "source) or pass --probe-version")
    manifest, digest = runner.get("requests"), runner.get("request_digest")
    source = "runner protocol.requests"
    if manifest is None and digest is None:
        missing = [key for key in ("probe", "pp", "tg", "reps") if key not in runner]
        if missing:
            raise ValueError(f"the runner's protocol has no `requests` manifest and lacks "
                             f"{missing} to derive one from")
        manifest = {key: runner[key] for key in
                    ("probe", "pp", "tg", "reps", "warmup_launches") if key in runner}
        source = ("derived from the runner protocol's probe/pp/tg/reps/warmup_launches; "
                  "the probe's prompts and sampling are pinned by its command and "
                  "probe_version")
    return serving_protocol(
        server_argv=argv, metric=metric, probe_tool=str(tool), probe_version=str(version),
        request_manifest=manifest, request_digest=digest, surface=surface,
        placement=runner.get("placement"), model=runner.get("model"),
        argv_provenance=str(runner.get("argv_provenance") or "runner protocol.argv"),
        request_manifest_source=source)


@dataclass(frozen=True)
class ServingMeasurement:
    """ONE arm's serving-probe samples for one metric: one sample per server launch."""
    surface: str
    metric: str
    samples: list[float]
    model: str | None
    launches: int
    residency: Any = None

    def to_dict(self) -> dict:
        return {"surface": self.surface, "metric": self.metric, "model": self.model,
                "launches": self.launches, "samples": self.samples,
                "sample_unit": SERVING_SAMPLE_UNIT, "estimator": "median",
                "median": statistics.median(self.samples), "residency": self.residency}


@dataclass(frozen=True)
class ServingRun:
    """A runner result, validated. `arms` maps "P"/"C" to the arm's record."""
    path: Path | None
    sha256: str | None
    arms: dict
    samples: dict
    surfaces: dict
    runner_protocol: dict
    protocols: dict

    def measurement(self, metric: str = "tg", arm: str = "C") -> ServingMeasurement:
        if arm not in self.samples:
            raise ValueError(f"the runner result has no {arm} arm")
        values = list(self.samples[arm][metric])
        return ServingMeasurement(
            surface=self.surfaces[metric], metric=metric, samples=values,
            model=self.protocols[metric]["model"], launches=len(values),
            residency=self.arms[arm].get("residency"))


def _samples(value: Any, where: str) -> list[float]:
    try:
        return _admissible_samples(value)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


def read_serving_result(source: Path | str | Mapping[str, Any], *,
                        probe_version: str | None = None) -> ServingRun:
    """Parse and validate a runner result (`SERVING_RESULT_SCHEMAS`). Raises ValueError.

    THE SCHEMA a runner emits (the stack owner's `paired_receipt.py` writer-samples.json,
    accepted as-is; `SERVING_RESULT_SCHEMAS[1]` names the same shape for new runners)::

      {"schema": "epyc.fold.serving_ab_samples_for_production_writer.v1",
       "p_arm": {"commit": "<40-hex production>", "build": "<dir>",
                 "surface": "serving_<...>_tg<N>",
                 "samples": [<tg tok/s, one per server launch, timings.predicted_per_second
                             median over the launch's reps>, ... >= 14],
                 "model": "<gguf>", "measured_at": "<ISO-8601>",
                 "protocol": {"argv": ["<llama-server>"|<path>, ...], "model": ...,
                              "probe": "<probe command>", "pp": <int>, "tg": <int>,
                              "reps": <int>, "warmup_launches": <int>,
                              "sample_unit": "process"|"server_launch",
                              "placement": "<env/numactl/taskset>" (optional, matched),
                              "probe_version": "<str>" (or --probe-version),
                              "requests": {...} | "request_digest": "<64-hex>" (optional),
                              ...any other keys, recorded not matched},
                 "host_facts": {<production.read_host_facts()>} | null,
                 "source": {...}, "residency": {...}},
       "c_arm":    {same keys, "commit": "<champion sha, >= 12 hex>"},
       "p_arm_pp": {"surface": "serving_<...>_pp<N>", "model": ..., "samples": [pp tok/s
                    (timings.prompt_per_second), one per launch, index-aligned]},
       "c_arm_pp": {same},
       "comparison": {"anchor_samples": == p_arm.samples,
                      "candidate_samples": == c_arm.samples, ...} (optional)}

    `p_arm.samples[i]` and `c_arm.samples[i]` are the i-th alternating pair (warm-up
    pair already discarded). Either arm may be absent: P alone records production's
    baseline, C alone is a single-arm champion run against the recorded baseline. Both
    arms' `protocol` blocks must be identical.
    """
    path = sha = None
    if isinstance(source, Mapping):
        body = dict(source)
    else:
        path = Path(source).resolve()
        body = json.loads(path.read_text(encoding="utf-8"))
        sha = _sha256(path)
    if not isinstance(body, dict) or body.get("schema") not in SERVING_RESULT_SCHEMAS:
        raise ValueError(f"not a serving-probe runner result: schema "
                         f"{(body or {}).get('schema') if isinstance(body, dict) else None!r}"
                         f", expected one of {list(SERVING_RESULT_SCHEMAS)}")
    arms, samples, protocols_seen = {}, {}, []
    surfaces: dict[str, set] = {"tg": set(), "pp": set()}
    for arm, key in (("P", "p_arm"), ("C", "c_arm")):
        record = body.get(key)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"`{key}` must be an object")
        missing = [field for field in ("commit", "samples", "surface", "protocol")
                   if field not in record]
        prefill = body.get(f"{key}_pp")
        if not isinstance(prefill, dict) or "samples" not in prefill:
            missing.append(f"{key}_pp.samples")
        if missing:
            raise ValueError(f"`{key}` lacks {missing}")
        if not _is_hex(str(record["commit"])) or len(str(record["commit"])) < 12:
            raise ValueError(f"`{key}.commit` must be a sha (>= 12 hex), got "
                             f"{record['commit']!r}")
        for model in (record.get("model"), prefill.get("model")):
            declared = record["protocol"].get("model") if isinstance(
                record["protocol"], dict) else None
            if model is not None and declared is not None and model != declared:
                raise ValueError(f"`{key}` model {model!r} contradicts its protocol's "
                                 f"{declared!r}")
        arms[arm] = record
        samples[arm] = {"tg": _samples(record["samples"], f"{key}.samples"),
                        "pp": _samples(prefill["samples"], f"{key}_pp.samples")}
        surfaces["tg"].add(record["surface"])
        surfaces["pp"].add(prefill.get("surface") or f"{SERVING_SURFACE_PREFIX}probe_pp")
        protocols_seen.append(record["protocol"])
    if not arms:
        raise ValueError("the runner result has neither a `p_arm` nor a `c_arm`")
    if len({canonical_digest(item) for item in protocols_seen}) != 1:
        raise ValueError("`p_arm.protocol` and `c_arm.protocol` differ: the two arms "
                         "were not measured under one protocol")
    runner = protocols_seen[0]
    for metric, seen in surfaces.items():
        if len(seen) != 1:
            raise ValueError(f"the arms name different {metric} surfaces: {sorted(seen)}")
    surface = {metric: next(iter(seen)) for metric, seen in surfaces.items()}
    if isinstance(runner, dict) and runner.get("surface") not in (None, surface["tg"]):
        raise ValueError(f"the arms' surface {surface['tg']!r} contradicts the protocol's "
                         f"{runner.get('surface')!r}")
    if len(arms) == 2:
        for metric in ("tg", "pp"):
            if len(samples["P"][metric]) != len(samples["C"][metric]):
                raise ValueError(
                    f"{metric}: {len(samples['P'][metric])} P vs "
                    f"{len(samples['C'][metric])} C launches -- a paired run pairs "
                    f"launch i of P with launch i of C")
    comparison = body.get("comparison")
    if comparison is not None:
        for arm, key in (("P", "anchor_samples"), ("C", "candidate_samples")):
            if arm in samples and key in comparison and \
                    [float(x) for x in comparison[key]] != samples[arm]["tg"]:
                raise ValueError(f"`comparison.{key}` is not the {arm} arm's tg samples")
        if comparison.get("surface") not in (None, surface["tg"]):
            raise ValueError(f"`comparison.surface` {comparison.get('surface')!r} is not "
                             f"{surface['tg']!r}")
    protocols = {metric: serving_protocol_from_runner(
        runner, metric=metric, surface=surface[metric], probe_version=probe_version)
        for metric in ("tg", "pp")}
    return ServingRun(path, sha, arms, samples, surface, dict(runner), protocols)


def serving_measurement(source: Path | str | Mapping[str, Any], metric: str = "tg",
                        arm: str = "C", *, probe_version: str | None = None
                        ) -> ServingMeasurement:
    """One arm of a runner result as a measurement -- `refresh`'s `measure` for a
    serving target: `measure=lambda _build: serving_measurement(result_json)`."""
    return read_serving_result(source, probe_version=probe_version).measurement(metric, arm)


def _serving_leg(metric: str, protocol: Mapping[str, Any], baseline: Sequence[float],
                 champion: Sequence[float], *, paired: bool,
                 decision: BaselineDecision | None = None) -> dict:
    """Effect and CI for one metric. Raises `HeadlineInadmissible` under the floor."""
    base, champ = [float(x) for x in baseline], [float(x) for x in champion]
    ci = (headline_admissibility.confidence_interval(base, champ) if paired else
          headline_admissibility.unpaired_confidence_interval(base, champ))
    effect = statistics.median(champ) / statistics.median(base) - 1.0
    leg = {"metric": metric, "surface": protocol["surface"],
           "timing_field": protocol["timing_field"], "unit": protocol.get("unit"),
           "effect_fraction": effect, "effect_pct": effect * 100.0,
           "estimator": "median_over_median",
           "baseline_median": statistics.median(base),
           "champion_median": statistics.median(champ),
           "baseline_launches": len(base), "champion_launches": len(champ),
           "confidence_interval": ci, "baseline_samples": base, "champion_samples": champ,
           "baseline_record": None, "baseline_record_sha256": None,
           "baseline_measured_at": None}
    if paired:
        leg["pair_ratios"] = [c / b for b, c in zip(base, champ)]
    if decision is not None and decision.path is not None:
        leg.update(baseline_record=str(decision.path),
                   baseline_record_sha256=_sha256(decision.path),
                   baseline_measured_at=(decision.body or {}).get("measured_at"))
    return leg


def _write_serving_receipt(store: Path, *, legs: Mapping[str, dict],
                           protocols: Mapping[str, Mapping[str, Any]], paired: bool,
                           primary: str, champion_commit: str, champion_build: Any,
                           frozen_commit: str, frozen_label: str, production_build: Any,
                           carryover: Mapping[str, Any], source: Mapping[str, Any] | None,
                           bundle_name: str, now: Callable[[], str],
                           note: str | None = None) -> tuple[Path, Path, str]:
    """The champion's serving standing receipt: per-champion evidence + the bundle."""
    if primary not in legs:
        raise ValueError(f"primary metric {primary!r} was not measured ({sorted(legs)})")
    order = [primary] + sorted(m for m in legs if m != primary)
    comparison = "paired_ab" if paired else "unpaired_recorded_baseline"
    head = legs[primary]
    evidence = status.write_json(
        store, f"champion-vs-production.{champion_commit[:12]}.serving_probe.json", {
            "comparison": comparison, "protocol_kind": PROTOCOL_KIND_SERVING_PROBE,
            "champion_commit": champion_commit, "production_commit": frozen_commit,
            "legs": {m: legs[m] for m in order},
            "protocols": {m: dict(protocols[m]) for m in order},
            "source": None if source is None else dict(source),
            **({} if paired else {"caveat": UNPAIRED_CAVEAT,
                                  "host_drift_caveat_pct": HOST_DRIFT_CAVEAT_PCT})},
        prefix=".cvp-")
    summary = {m: {k: v for k, v in legs[m].items()
                   if k not in ("baseline_samples", "champion_samples", "pair_ratios")}
               for m in order}
    bundle = {
        "schema": SCHEMA, "generated_at": now(), "stale_after_s": STALE_AFTER_S,
        "protocol_kind": PROTOCOL_KIND_SERVING_PROBE,
        "baseline": {"commit": frozen_commit, "label": frozen_label,
                     "build": None if production_build is None else str(production_build),
                     "recorded": head["baseline_record"],
                     "recorded_sha256": head["baseline_record_sha256"],
                     "recorded_measured_at": head["baseline_measured_at"]},
        "champion": {"commit": champion_commit,
                     "build": None if champion_build is None else str(champion_build)},
        "effect_fraction": float(head["effect_fraction"]),
        "metric": f"{head['surface']}_tok_s", "metric_direction": "higher_better",
        "model": protocols[primary]["model"], "surface": head["surface"],
        "pairs": head["baseline_launches"] if paired else None,
        "comparison": comparison,
        "launches": head["baseline_launches"] + head["champion_launches"],
        "champion_launches": head["champion_launches"],
        "baseline_launches": head["baseline_launches"],
        "headline_admissibility": {**headline_admissibility.contract(),
                                   **({} if paired else {
                                       "ci_method": headline_admissibility.UNPAIRED_CI_METHOD})},
        "confidence_interval": head["confidence_interval"],
        "noise_floor_pct": None,
        "metrics": summary, "protocol": dict(protocols[primary]),
        "baseline_carryover": dict(carryover),
        "evidence": str(evidence), "mechanism_id": MECHANISM_ID,
        **({} if paired else {"host_drift_caveat_pct": HOST_DRIFT_CAVEAT_PCT,
                              "caveat": UNPAIRED_CAVEAT}),
        **({"anchor_guard_excursion": note} if note else {}),
    }
    target = status.write_json(store, bundle_name, bundle, prefix=".cvp-")
    effects = " / ".join(f"{legs[m]['effect_pct']:+.3f}% {legs[m]['surface']}"
                         for m in order)
    if paired:
        reason = (f"champion {champion_commit[:12]} measures {effects} against frozen "
                  f"production {frozen_commit[:12]} ({frozen_label}) over "
                  f"{head['baseline_launches']} PAIRED serving-probe launches")
    else:
        reason = (f"champion {champion_commit[:12]} measures {effects} against the "
                  f"RECORDED frozen production {frozen_commit[:12]} ({frozen_label}) "
                  f"serving-probe baseline of {str(head['baseline_measured_at'])[:10]}, "
                  f"UNPAIRED: {head['champion_launches']} champion launches vs "
                  f"{head['baseline_launches']} recorded launches, host-drift caveat "
                  f"~{HOST_DRIFT_CAVEAT_PCT:g}%")
    return target, evidence, reason


@dataclass(frozen=True)
class ServingIngest:
    """What `ingest_serving_result` wrote (or, on a dry run, would write)."""
    mode: str
    reason: str
    baselines: dict
    bundle: Path | None = None
    evidence: Path | None = None
    effects: dict | None = None


def ingest_serving_result(source: Path | str | Mapping[str, Any], *, store: Path | str,
                          production_commit: str | None = None,
                          production_label: str | None = None,
                          resolve: Callable[[], tuple[str, str]] = resolve_frozen,
                          probe_version: str | None = None, primary_metric: str = "tg",
                          bundle_name: str = SERVING_FILENAME, overwrite: bool = False,
                          host_facts: Callable[[], dict | None] = read_host_facts,
                          host_changes: Sequence[Mapping[str, Any]] = HOST_CHANGES,
                          max_age_days: float = BASELINE_MAX_AGE_DAYS,
                          clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                          now: Callable[[], str] = status._now,
                          dry_run: bool = False) -> ServingIngest:
    """A serving runner's result -> production baselines and/or the champion receipt.

    CLI: `ingest-serving --result <writer-samples.json>`. Raises ValueError / OSError /
    `Unavailable` and writes NOTHING on any refusal (everything is validated, and every
    CI computed, before the first write).

      * P and C (the paired runner): writes `production-baseline.<sha12>.serving_probe-
        {tg,pp}.json` from the P arm, then the PAIRED receipt (`bundle_name` + per-
        champion `champion-vs-production.<c12>.serving_probe.json`).
      * P only: the two baselines.
      * C only (a future champion, single-arm): `baseline_decision` must admit the
        recorded baseline for EVERY metric (same normalised argv, request digest,
        metric; age, host changes, host facts -- the C arm's `host_facts`, else this
        host's now); then the UNPAIRED receipt, with the host-drift caveat.

    The P arm's commit must be the frozen production kernel: `production_commit` /
    `production_label` when given, else resolved live from the frozen tree.
    """
    run = read_serving_result(source, probe_version=probe_version)
    if production_commit:
        commit, label = production_commit, production_label or ""
    else:
        commit, label = resolve()
        label = production_label or label
    if not _is_hex(commit, 40):
        raise ValueError(f"production commit must be a full 40-hex sha, got {commit!r}")
    if not label.startswith(FROZEN_BRANCH_PREFIX):
        raise ValueError(f"production label {label!r} is not a "
                         f"'{FROZEN_BRANCH_PREFIX}*' freeze")
    if "P" in run.arms and str(run.arms["P"]["commit"]) != commit:
        raise ValueError(f"the P arm measured {run.arms['P']['commit']!r}, not the frozen "
                         f"production kernel {commit}; its samples are not production's")
    metrics = list(run.protocols)
    store = Path(store)
    source_ref = {"path": None if run.path is None else str(run.path), "sha256": run.sha256}
    plan: dict[str, Any] = {"baselines": {}, "legs": {}}
    if "P" in run.arms:
        p = run.arms["P"]
        measured_at = p.get("measured_at")
        if not measured_at:
            raise ValueError("`p_arm.measured_at` is required to record a baseline")
        _parse_when(measured_at)
        for metric in metrics:
            target = store / baseline_filename(commit, run.protocols[metric])
            if target.exists() and not overwrite:
                raise FileExistsError(f"{target} already records this freeze; pass "
                                      f"overwrite to supersede it")
            plan["baselines"][metric] = dict(
                production_commit=commit, production_label=label, measured_commit=commit,
                samples=run.samples["P"][metric], protocol=run.protocols[metric],
                measured_at=measured_at,
                source={**source_ref, "kind": "serving-probe runner, production (P) arm",
                        "samples_path": ("p_arm.samples" if metric == "tg"
                                         else "p_arm_pp.samples"),
                        "runner_protocol": run.runner_protocol,
                        "runner_source": p.get("source")},
                host_facts=p.get("host_facts"),
                host_facts_note=("captured by the runner at the measurement"
                                 if p.get("host_facts") else "not captured by the runner"),
                lineage={"commits_between": [],
                         "note": "the frozen production binary itself, measured by the "
                                 "serving probe"},
                residency=p.get("residency"))
    mode = "baseline_only"
    if "C" in run.arms:
        if "P" in run.arms:
            mode = "paired"
            for metric in metrics:
                plan["legs"][metric] = _serving_leg(
                    metric, run.protocols[metric], run.samples["P"][metric],
                    run.samples["C"][metric], paired=True)
            carryover = {"used": False, "reason": "paired serving-probe runner: the "
                         "production arm was measured in the same session"}
        else:
            mode = "single_arm"
            facts = run.arms["C"].get("host_facts") or host_facts() or {}
            refused = []
            for metric in metrics:
                decision = baseline_decision(
                    store, commit, protocol=run.protocols[metric], max_age_days=max_age_days,
                    host_facts=facts, host_changes=host_changes, clock=clock)
                if not decision.usable:
                    refused.append(f"{metric}: {decision.reason}")
                    continue
                plan["legs"][metric] = _serving_leg(
                    metric, run.protocols[metric], (decision.body or {})["samples"],
                    run.samples["C"][metric], paired=False, decision=decision)
            if refused:
                raise Unavailable(
                    "single-arm champion run REFUSED: no usable recorded serving-probe "
                    f"baseline ({'; '.join(refused)}). Run the paired runner (P and C) "
                    "and ingest that instead")
            carryover = {"used": True, "reason": "; ".join(
                f"{m}: recorded {Path(plan['legs'][m]['baseline_record']).name}"
                for m in metrics)}
    effects = {m: leg["effect_fraction"] for m, leg in plan["legs"].items()}
    if dry_run:
        return ServingIngest(mode, f"DRY RUN ({mode}): nothing written",
                             {m: str(store / baseline_filename(commit, run.protocols[m]))
                              for m in plan["baselines"]}, effects=effects or None)
    baselines = {m: write_baseline(store, overwrite=overwrite, now=now, **record)
                 for m, record in plan["baselines"].items()}
    if not plan["legs"]:
        return ServingIngest(mode, f"recorded production {commit[:12]} ({label}) serving-"
                                   f"probe baselines for {sorted(baselines)}", baselines)
    champion = run.arms["C"]
    bundle, evidence, reason = _write_serving_receipt(
        store, legs=plan["legs"], protocols=run.protocols, paired=(mode == "paired"),
        primary=primary_metric, champion_commit=str(champion["commit"]),
        champion_build=champion.get("build"), frozen_commit=commit, frozen_label=label,
        production_build=(run.arms.get("P") or {}).get("build"), carryover=carryover,
        source=source_ref, bundle_name=bundle_name, now=now)
    return ServingIngest(mode, reason, baselines, bundle, evidence, effects)


def _refresh_serving(*, store: Path, protocol: Mapping[str, Any], frozen_commit: str,
                     frozen_label: str, champion_commit: str, champion_build: Path,
                     measure: Callable[[Path], Any] | None, max_age_days: float,
                     host_facts: Callable[[], dict], host_changes: Sequence[Mapping],
                     clock: Callable[[], datetime], now: Callable[[], str],
                     note: str | None, on_step: Callable[[str], Any]) -> Refresh:
    """`refresh` for a SERVING target: the champion alone vs the recorded serving
    baseline, or a refusal. Never a paired llama-bench fallback (wrong protocol)."""
    if measure is None:
        raise Unavailable("a serving target needs a single-arm measurer (`measure`, e.g. "
                          "`serving_measurement(<runner result>)`); this module never "
                          "launches a server")
    decision = baseline_decision(store, frozen_commit, protocol=protocol,
                                 max_age_days=max_age_days, host_facts=host_facts(),
                                 host_changes=host_changes, clock=clock)
    if not decision.usable:
        raise Unavailable(
            f"serving target: the recorded serving-probe baseline is not usable "
            f"({decision.reason}). llama-bench cannot drive this protocol, so there is no "
            f"paired fallback here: run the stack owner's paired runner and ingest its "
            f"result (`ingest-serving`)")
    on_step("champion-vs-production: champion alone vs the RECORDED serving-probe baseline")
    measured = measure(Path(champion_build))
    model = getattr(measured, "model", None)
    if model is not None and model != protocol.get("model"):
        raise Unavailable(f"the champion was measured on {model}, the protocol names "
                          f"{protocol.get('model')}")
    if measured.surface != protocol["surface"]:
        raise Unavailable(f"the champion was measured on {measured.surface}, the protocol "
                          f"on {protocol['surface']}")
    metric = protocol["metric"]
    leg = _serving_leg(metric, protocol, (decision.body or {})["samples"],
                       list(measured.samples), paired=False, decision=decision)
    bundle, _evidence, reason = _write_serving_receipt(
        Path(store), legs={metric: leg}, protocols={metric: protocol}, paired=False,
        primary=metric, champion_commit=champion_commit, champion_build=champion_build,
        frozen_commit=frozen_commit, frozen_label=frozen_label, production_build=None,
        carryover={"used": True, "reason": decision.reason}, source=None,
        bundle_name=SERVING_FILENAME, now=now, note=note)
    return Refresh(True, reason, bundle, float(leg["effect_fraction"]),
                   carryover=decision.reason)


def _dig(body: Any, dotted: str) -> tuple[Any, Any]:
    """`(value, parent)` at a dotted path such as `g5_full.candidate_samples`."""
    parent, value = None, body
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"no {dotted!r} in the source (stopped at {part!r})")
        parent, value = value, value[part]
    return value, parent


def _ingest_main(args: argparse.Namespace) -> int:
    try:
        commit = args.production_commit
        if commit and len(commit) != 40:
            commit = _read(args.frozen_tree, "rev-parse", commit)
        outcome = ingest_serving_result(
            args.result, store=args.store, production_commit=commit,
            production_label=args.production_label, probe_version=args.probe_version,
            primary_metric=args.primary_metric, bundle_name=args.bundle_name,
            overwrite=args.overwrite, dry_run=args.dry_run,
            host_facts=(read_host_facts if args.host_facts == "capture" else lambda: None))
    except (ValueError, OSError, Unavailable) as exc:
        print(f"ingest-serving REFUSED: {exc}", file=sys.stderr)
        return 2
    print(outcome.reason)
    for metric, path in sorted(outcome.baselines.items()):
        print(f"  baseline {metric}: {path}")
    if outcome.bundle is not None:
        print(f"  receipt: {outcome.bundle}  evidence: {outcome.evidence}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """`python3 -m autokernel.loop.production {write-baseline,ingest-serving} ...`
    (cwd scripts/kernel_rnd).

    `write-baseline` is the kernel-promotion step: record the promoted champion's last
    standing or fold samples as the new freeze's production baseline (llama-bench by
    default; `--protocol-kind serving_probe` with `--argv-json`, `--metric`,
    `--probe-tool`, `--probe-version` and `--request-manifest` | `--request-digest` for
    a serving probe). `ingest-serving --result <writer-samples.json>` takes a serving
    runner's result (schema at `read_serving_result`) and writes the production
    baselines and/or the champion's receipt (`ingest_serving_result`). Each prints what
    it wrote; exit 0 written, 2 refused.
    """
    parser = argparse.ArgumentParser(prog="autokernel.loop.production")
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write-baseline", help="record production-baseline.<sha12>.json")
    write.add_argument("--source", type=Path,
                       help="JSON holding the samples (a fold result, a standing "
                            "champion-vs-production.<sha12>.json evidence file, a "
                            "serving runner's writer-samples.json, ...)")
    write.add_argument("--samples-json", help="the sample list inline, as JSON, in "
                                              "place of --source (needs --measured-at)")
    write.add_argument("--protocol-kind", choices=PROTOCOL_KINDS,
                       default=PROTOCOL_KIND_LLAMA_BENCH)
    write.add_argument("--metric", choices=sorted(SERVING_METRICS),
                       help="serving_probe: pp (prompt_per_second) or tg "
                            "(predicted_per_second)")
    write.add_argument("--probe-tool", help="serving_probe: the probe tool (ab_probe.py)")
    write.add_argument("--probe-version", help="serving_probe: its version (sha256 of "
                                               "its source recommended)")
    write.add_argument("--request-manifest", type=Path,
                       help="serving_probe: request manifest JSON (digest computed)")
    write.add_argument("--request-digest", help="serving_probe: the manifest's 64-hex "
                                                "canonical digest, when no manifest")
    write.add_argument("--n-predict", help="serving_probe, digest only: n_predict as JSON")
    write.add_argument("--sampling-json", help="serving_probe, digest only: sampling params")
    write.add_argument("--placement", help="serving_probe: the env/numactl/taskset the "
                                           "server ran under (matched)")
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
                                           "measurement predates the live instrument; "
                                           "serving_probe: the server argv (required)")
    write.add_argument("--argv-provenance", default="bench.llama_bench_argv (live)")
    write.add_argument("--measured-at", help="ISO-8601 (default: --source mtime, UTC)")
    write.add_argument("--host-facts", choices=("capture", "none"), default="capture",
                       help="capture this host's facts now (only when the samples were "
                            "taken on this host as it is today), or record none")
    write.add_argument("--host-facts-note", default="")
    write.add_argument("--store", type=Path, default=DEFAULT_STORE)
    write.add_argument("--overwrite", action="store_true")
    write.add_argument("--dry-run", action="store_true")
    ingest = sub.add_parser("ingest-serving", help="a serving runner's result -> "
                                                   "production baselines + champion receipt")
    ingest.add_argument("--result", required=True, type=Path,
                        help="the runner's result JSON (writer-samples.json)")
    ingest.add_argument("--production-commit",
                        help="the frozen commit (default: resolved live from the frozen tree)")
    ingest.add_argument("--production-label", help="e.g. production-consolidated-v10")
    ingest.add_argument("--frozen-tree", type=Path, default=FROZEN_TREE,
                        help="git repo resolving a short --production-commit (read-only)")
    ingest.add_argument("--probe-version", help="when the result's protocol carries none")
    ingest.add_argument("--primary-metric", choices=sorted(SERVING_METRICS), default="tg")
    ingest.add_argument("--bundle-name", default=SERVING_FILENAME)
    ingest.add_argument("--host-facts", choices=("capture", "none"), default="capture",
                        help="single-arm only, when the C arm carries no host_facts")
    ingest.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ingest.add_argument("--overwrite", action="store_true")
    ingest.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "ingest-serving":
        return _ingest_main(args)
    serving = args.protocol_kind == PROTOCOL_KIND_SERVING_PROBE
    try:
        if args.source is not None:
            source_path = args.source.resolve()
            body = json.loads(source_path.read_text(encoding="utf-8"))
            samples, parent = _dig(body, args.samples_path)
            parent = parent if isinstance(parent, dict) else {}
        elif args.samples_json:
            if not args.measured_at:
                raise ValueError("--samples-json needs --measured-at")
            source_path, samples, parent = None, json.loads(args.samples_json), {}
        else:
            raise ValueError("give --source (with --samples-path) or --samples-json")
        model = args.model or parent.get("model")
        surface = args.surface or parent.get("surface")
        if not serving and (not model or not surface):
            raise ValueError("model and surface must come from the source or the flags")
        for key, given in (("model", model), ("surface", surface)):
            if given is not None and parent.get(key) not in (None, given):
                raise ValueError(f"--{key} {given!r} contradicts the source's "
                                 f"{parent.get(key)!r}")
        if serving:
            # A runner arm (writer-samples.json `p_arm`) carries its own placement and
            # measurement time; the flags override, never the other way round.
            runner = parent.get("protocol") if isinstance(parent.get("protocol"),
                                                          dict) else {}
            args.placement = args.placement or runner.get("placement")
            args.measured_at = args.measured_at or parent.get("measured_at")
            if not args.argv_json or not args.metric:
                raise ValueError("--protocol-kind serving_probe needs --argv-json (the "
                                 "server argv) and --metric")
            measured_protocol = serving_protocol(
                server_argv=json.loads(args.argv_json), metric=args.metric,
                probe_tool=args.probe_tool or "", probe_version=args.probe_version or "",
                request_manifest=(json.loads(args.request_manifest.read_text(
                    encoding="utf-8")) if args.request_manifest else None),
                request_digest=args.request_digest,
                n_predict=json.loads(args.n_predict) if args.n_predict else None,
                sampling=json.loads(args.sampling_json) if args.sampling_json else None,
                surface=surface, placement=args.placement, model=model,
                argv_provenance=(args.argv_provenance if args.argv_provenance
                                 != "bench.llama_bench_argv (live)" else "--argv-json"),
                request_manifest_source=(str(args.request_manifest)
                                         if args.request_manifest else "digest only"))
        else:
            measured_protocol = protocol(
                model=model, surface=surface, reps=args.reps,
                argv=json.loads(args.argv_json) if args.argv_json else None,
                argv_provenance=args.argv_provenance)
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
            samples=samples, protocol=measured_protocol, measured_at=measured_at,
            source=({"path": str(source_path), "sha256": _sha256(source_path),
                     "samples_path": args.samples_path} if source_path is not None
                    else {"path": None, "inline": "--samples-json"})
            | {"measured_at_source": ("--measured-at" if args.measured_at
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
        path = write_baseline(args.store, overwrite=args.overwrite,
                              protocol_kind=args.protocol_kind, **record)
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
           "LEGACY_COMMIT", "MECHANISM_ID", "PROTOCOL_KIND_LLAMA_BENCH",
           "PROTOCOL_KIND_SERVING_PROBE", "PROTOCOL_KINDS", "PROTOCOL_MATCH_KEYS",
           "PROVENANCE", "SCHEMA", "SERVING_FILENAME", "SERVING_METRICS",
           "SERVING_PROBE_MATCH_KEYS", "SERVING_RESULT_SCHEMAS", "SERVING_SAMPLE_UNIT",
           "STALE_AFTER_S", "UNPAIRED_CAVEAT", "Refresh", "ServingIngest",
           "ServingMeasurement", "ServingRun", "Unavailable", "baseline_decision",
           "baseline_filename", "baseline_slot", "canonical_digest", "commits_between",
           "declared_commit", "host_fact_changes", "ingest_serving_result", "is_built",
           "main", "match_keys", "normalize_server_argv", "protocol", "protocol_kind",
           "protocol_mismatch", "read_host_facts", "read_serving_result", "refresh",
           "request_manifest_digest", "resolve_frozen", "serving_measurement",
           "serving_protocol", "serving_protocol_from_runner", "write_baseline"]


if __name__ == "__main__":
    raise SystemExit(main())
