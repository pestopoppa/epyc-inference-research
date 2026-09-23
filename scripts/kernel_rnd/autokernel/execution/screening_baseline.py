"""Immutable amortized baseline bank for non-promotable discovery screens.

The bank deliberately carries only an exact-frame anchor vector.  It is never
accepted by strict T1 and cannot be converted into a candidate/archive record.
"""
from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .. import schemas
from ..evaluator import recipes
from . import microbench
from ..resource import preflight

SCHEMA = "epyc.autokernel.screening_baseline_bank.v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BaselineBankError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineBank:
    frame: Mapping[str, Any]
    anchor_samples: tuple[float, ...]
    sentinel_before: float
    anchor_command: Mapping[str, Any]
    anchor_artifacts: Mapping[str, Any]
    sentinel_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": SCHEMA, "frame": dict(self.frame),
                "anchor_samples": list(self.anchor_samples),
                "anchor_command": dict(self.anchor_command),
                "anchor_artifacts": dict(self.anchor_artifacts),
                "sentinel_before": self.sentinel_before,
                "sentinel_after": self.sentinel_after}
        return {**body, "baseline_sha256": schemas.content_hash(body)}

    def admit(self, frame: Mapping[str, Any]) -> None:
        if dict(frame) != dict(self.frame):
            raise BaselineBankError("screening baseline frame differs from candidate frame")
        if self.sentinel_after is not None:
            raise BaselineBankError("screening baseline is closed; create a fresh bank")

    def nominate(self, candidate_samples: tuple[float, ...]) -> dict[str, Any]:
        """Noise-tolerant directional summary, never a pass/fail decision."""
        if not candidate_samples:
            raise BaselineBankError("screening candidate has no samples")
        center = sum(self.anchor_samples) / len(self.anchor_samples)
        values = tuple((x - center) / center for x in candidate_samples)
        return {"baseline_center": center, "candidate_samples": list(candidate_samples),
                "relative_effects": list(values),
                "median_relative": sorted(values)[len(values) // 2],
                "uncertainty": "screening_noise_unquantified_nonpromotable",
                "nomination": "top_k_candidate_only_not_a_keep"}


def load(path: str | Path) -> BaselineBank:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BaselineBankError("baseline bank must be an object")
    body = {key: raw.get(key) for key in ("schema", "frame", "anchor_samples", "anchor_command", "anchor_artifacts",
                                          "sentinel_before", "sentinel_after")}
    if raw.get("baseline_sha256") != schemas.content_hash(body) or body["schema"] != SCHEMA:
        raise BaselineBankError("baseline bank schema/hash is invalid")
    values = body["anchor_samples"]
    command = body["anchor_command"]
    artifacts = body["anchor_artifacts"]
    if not isinstance(body["frame"], Mapping) or not isinstance(values, list) or len(values) != 3:
        raise BaselineBankError("baseline bank needs exact frame and exactly three anchor samples")
    if body["frame"].get("anchor_ggml_iqk") != "0" \
            or not isinstance(command, Mapping) \
            or command.get("arm") != "anchor" \
            or command.get("env", {}).get("GGML_IQK") != "0" \
            or command.get("params", {}).get("ggml_iqk") != "0" \
            or not isinstance(artifacts, Mapping):
        raise BaselineBankError(
            "baseline bank must seal an anchor command with GGML_IQK=0")
    return BaselineBank(dict(body["frame"]), tuple(float(x) for x in values),
                        float(body["sentinel_before"]),
                        dict(command),
                        dict(artifacts),
                        None if body["sentinel_after"] is None else float(body["sentinel_after"]))


def create(*, frame: Mapping[str, Any], anchor_command: Mapping[str, Any],
           invoke_anchor, anchor_count: int = 3) -> BaselineBank:
    """Seal O(1) anchor invocations once for a whole discovery batch."""
    if anchor_count != 3:
        raise BaselineBankError("baseline bank requires exactly three anchor invocations")
    samples = tuple(float(invoke_anchor()) for _ in range(anchor_count))
    if frame.get("anchor_ggml_iqk") != "0" \
            or anchor_command.get("env", {}).get("GGML_IQK") != "0" \
            or anchor_command.get("params", {}).get("ggml_iqk") != "0":
        raise BaselineBankError("baseline creation requires a bound GGML_IQK=0 anchor")
    return BaselineBank(dict(frame), samples, samples[-1], dict(anchor_command),
                        command_artifacts(anchor_command))


def command_artifacts(command: Mapping[str, Any]) -> dict[str, Any]:
    binding = command.get("binding", {})
    binary = Path(str(binding.get("binary", "")))
    library_root = Path(str(binding.get("library_path", "")))
    if not binary.is_file() or not library_root.is_dir():
        raise BaselineBankError("screening command artifact paths are unavailable")
    libraries = {}
    for path in sorted(library_root.glob("*.so*")):
        if path.is_file():
            libraries[path.name] = _sha256_file(path)
    return {"binary_sha256": _sha256_file(binary), "libraries": libraries}


def _semantic_command(command: Mapping[str, Any]) -> dict[str, Any]:
    env = dict(command.get("env", {}))
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("GGML_IQK", None)
    params = dict(command.get("params", {}))
    params.pop("ggml_iqk", None)
    params.pop("autokernel_seed", None)
    recipe = command.get("recipe", {})
    return {key: command.get(key) for key in (
        "recipe_id", "registry_id", "backend", "phase", "cell_class",
        "metric", "metric_direction", "tool") } | {
            "recipe": {key: recipe.get(key) for key in (
                "constructor_id", "constructor_sha256")},
            "env": env, "params": params,
        }


def screen(*, bank: BaselineBank, frame: Mapping[str, Any], invoke_candidate,
           competing_inference: bool,
           candidate_command: Mapping[str, Any],
           close_span: Optional[Callable[[], Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Three candidate-only calls; ordinary host load is intentionally not input.

    The caller must provide the claim-scoped competing-inference witness. That
    is the one discovery blocker; service/build/load noise is reflected in the
    uncertainty label, not converted into a false refusal.
    """
    bank.admit(frame)
    if competing_inference:
        raise BaselineBankError("competing model inference occupies claimed screening compute")
    if candidate_command.get("arm") != "candidate" \
            or candidate_command.get("env", {}).get("GGML_IQK") != "1" \
            or candidate_command.get("params", {}).get("ggml_iqk") != "1":
        raise BaselineBankError(
            "screening candidate command must seal candidate GGML_IQK=1")
    anchor_semantic = _semantic_command(bank.anchor_command)
    candidate_semantic = _semantic_command(candidate_command)
    candidate_artifacts = command_artifacts(candidate_command)
    if anchor_semantic != candidate_semantic \
            or dict(bank.anchor_artifacts) != candidate_artifacts:
        raise BaselineBankError(
            "screening arm commands differ beyond the sole intended GGML_IQK factor: "
            f"semantic_equal={anchor_semantic == candidate_semantic}, "
            f"artifact_equal={dict(bank.anchor_artifacts) == candidate_artifacts}")
    samples = tuple(float(invoke_candidate()) for _ in range(3))
    # Close the bracket.  `competing_inference` above only proved the host was
    # quiet BEFORE the three calls; this proves it stayed quiet ACROSS them.
    closing = close_span() if close_span is not None else None
    if closing is not None and closing.get("competing"):
        raise BaselineBankError(
            "unowned model inference did work during the screening window: "
            + violation_summary(closing))
    report = bank.nominate(samples)
    report.update({"candidate_invocations": 3, "anchor_invocations": 0,
                   "closing_inference_witness": closing,
                   "host_noise_policy": "recorded_not_blocking",
                   "candidate_command_sha256": schemas.content_hash(candidate_command),
                   "sole_intended_factor": {"name": "GGML_IQK",
                                             "anchor": "0", "candidate": "1"},
                   "non_promotable": True})
    return report


def invoke_command(*, command: recipes.ConstructedCommand, spawner: microbench.Spawner,
                   timeout_s: float = 300.0) -> float:
    """Run exactly one bound llama-bench command and reduce its own samples."""
    env = microbench.assemble_env(command.env).env
    spawned = spawner.run(command.argv, env, timeout_s=timeout_s)
    if spawned.timed_out or spawned.returncode != 0:
        raise BaselineBankError("screening invocation failed or timed out")
    rows = microbench.parse_llama_bench_json(spawned.stdout)
    if len(rows) != 1:
        raise BaselineBankError("screening invocation must emit exactly one result row")
    check = microbench.LlamaBenchExpectation.from_command(command).check_row(rows[0])
    if check.outcome != schemas.PASS:
        raise BaselineBankError("screening command/result frame mismatch: " + "; ".join(check.reasons))
    values = rows[0].metric_samples
    return sum(values) / len(values)


# =============================================================================
# Unowned-inference idleness — what this gate measures, and why it changed
# =============================================================================
#
# The original rule was: an UNOWNED `llama-server` exists  ->  refuse.
# On this host that makes every campaign unrunnable whenever the serving stack
# is up, and the operator's position is that the stack stays up.  Ignoring
# those pids (an allowlist, a flag, a "known pids" file) would give up exactly
# the property the gate exists for: concurrent inference poisons measurement.
#
# So the safety property is kept and the MEASURED QUANTITY is changed.  An
# unowned inference process is admissible only while it is proven to be doing
# NO WORK, and it is refused the moment it does any.
#
# The instrument is the per-process CPU-time counter in /proc/<pid>/stat
# (utime + stime).  It is MONOTONE and CUMULATIVE, so two reads that BRACKET a
# measurement window observe the WHOLE window and not an instant inside it:
# work done between the two samples cannot hide, because the counter that
# recorded it is still there when the closing sample is read.  That is the one
# property a sampled liveness probe lacks, and it is what CLAUDE.md's
# "Observation Windows" rule demands - a measurement whose window does not
# overlap the phenomenon is not evidence of its absence.
#
# Endpoint state (`/health`, `/slots`, `/metrics`) was considered and rejected
# as a GATE input: it needs a port map the campaign does not have, it is
# optional per server (`--metrics`), an unreachable endpoint would have to
# fail open, and issuing an HTTP request into another session's server is an
# action on a shared host.  The /proc counter needs no network, no privilege
# and no configuration, and it fails closed.
#
# There is no flag, no allowlist and nothing for the operator to remember.

IDLENESS_SCHEMA = "epyc.autokernel.unowned_inference_idleness.v1"
IDLENESS_BASIS = "interim_inference_executable_scan+unowned_cpu_time_idleness"

# Fixed cost tolerated per span whatever its length: a health scrape, a log
# line, an accept()/close() on an idle listener, one timer wakeup.
IDLE_SPAN_ALLOWANCE_CORE_S = 0.5
# Sustained rate tolerated, as a fraction of ONE core.
# Measured basis (2026-09-23, this host, all 13 resident unowned servers -- the
# six bge embedders, the three Qwen3.6-35B instances, Qwen3.8-27B, VL-30B, the
# :8074 AutoKernel planner and sd-server, with the stack up and idle): over a
# 279.06 s span the LARGEST accrual by any one of them was 6 ticks = 0.06
# core-seconds, i.e. 0.000215 cores; seven of the thirteen moved at all and
# none by more than that.  0.02 cores is ~93x that measured idle rate, while
# ONE busy decode thread is 1.0 -- fifty times over the line.  The gate
# therefore has no realistic false refusal and no realistic false admission.
# Re-derive these two numbers if the stack's launch flags change: a server
# started with a spinning wait policy would burn cores while idle, and this
# gate would then refuse -- correctly, because a spinning server really does
# consume the cores the campaign claimed.
IDLE_SPAN_ALLOWANCE_CORES = 0.02

_CLOCK_TICKS_PER_S = os.sysconf("SC_CLK_TCK")


def _read_process_cpu(pid: int) -> tuple[int, int] | None:
    """`(cpu_ticks, starttime_ticks)` for one pid, or None if it vanished.

    `comm` may contain spaces and parentheses, so every field is taken after
    the LAST ')': utime/stime/starttime are fields 14/15/22 of stat, i.e.
    offsets 11/12/19 in that tail.
    """
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:  # present but unreadable is NOT absent
        raise BaselineBankError(
            f"unowned inference pid {pid} is present but unreadable: {exc}") from exc
    try:
        tail = text[text.rindex(")") + 2:].split()
        return int(tail[11]) + int(tail[12]), int(tail[19])
    except (ValueError, IndexError) as exc:
        raise BaselineBankError(
            f"unowned inference pid {pid} has an unparsable /proc stat line") from exc


def read_cpu_ledger(findings: Any) -> dict[str, Any]:
    """Snapshot the CPU-time counter of every UNOWNED inference process."""
    entries: dict[str, Any] = {}
    for item in findings:
        pid = int(item.pid) if hasattr(item, "pid") else int(item["pid"])
        sample = _read_process_cpu(pid)
        if sample is None:
            # It was enumerated a moment ago and is gone now.  Its final
            # accrual is unknowable, so it cannot be certified idle.
            raise BaselineBankError(
                f"unowned inference pid {pid} vanished between enumeration and "
                "its CPU-time read; the span cannot be certified idle")
        cpu_ticks, starttime_ticks = sample
        cmdline = tuple(getattr(item, "cmdline", ()) or ())
        entries[str(pid)] = {
            "cpu_ticks": cpu_ticks,
            "starttime_ticks": starttime_ticks,
            "argv0_basename": getattr(item, "argv0_basename", None),
            "cmdline_head": " ".join(cmdline)[:200],
        }
    return {
        "schema": IDLENESS_SCHEMA,
        "clock_ticks_per_s": _CLOCK_TICKS_PER_S,
        "read_at_monotonic_s": time.monotonic(),
        "read_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }


def idleness_verdict(previous: Mapping[str, Any] | None,
                     current: Mapping[str, Any]) -> dict[str, Any]:
    """Decide the span between two ledgers.  Unknown is always a violation."""
    allowance_cores = IDLE_SPAN_ALLOWANCE_CORES
    base_core_s = IDLE_SPAN_ALLOWANCE_CORE_S
    if previous is None:
        # Nothing is asserted about a span with no start.  The window that
        # matters is closed by the NEXT call, which will have this ledger.
        return {"schema": IDLENESS_SCHEMA, "span": "opened", "span_s": None,
                "busy": False, "tolerated": [], "violations": [],
                "allowance_core_s": base_core_s, "allowance_cores": allowance_cores,
                "bracketed_pids": sorted(int(p) for p in current["entries"])}
    span_s = float(current["read_at_monotonic_s"]) - float(previous["read_at_monotonic_s"])
    ticks = float(current["clock_ticks_per_s"])
    allowance = base_core_s + allowance_cores * max(span_s, 0.0)
    tolerated: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    if span_s <= 0.0 or float(previous["clock_ticks_per_s"]) != ticks:
        violations.append({"pid": None, "reason": "non_monotonic_span", "span_s": span_s})
    before = dict(previous["entries"])
    after = dict(current["entries"])
    for key, now in sorted(after.items(), key=lambda kv: int(kv[0])):
        record = {"pid": int(key), "argv0_basename": now.get("argv0_basename"),
                  "cmdline_head": now.get("cmdline_head")}
        was = before.get(key)
        if was is None:
            violations.append({**record, "reason": "appeared_mid_span"})
            continue
        if was["starttime_ticks"] != now["starttime_ticks"]:
            # Same pid number, different process: the original one's accrual
            # was never closed out.
            violations.append({**record, "reason": "pid_reused_mid_span"})
            continue
        delta = int(now["cpu_ticks"]) - int(was["cpu_ticks"])
        core_s = delta / ticks
        record.update({"cpu_core_seconds": core_s, "span_s": span_s,
                       "cores_equivalent": (core_s / span_s) if span_s > 0 else None,
                       "allowance_core_s": allowance})
        if delta < 0 or core_s > allowance:
            violations.append({**record, "reason": "cpu_work"})
        else:
            tolerated.append(record)
    for key, was in sorted(before.items(), key=lambda kv: int(kv[0])):
        if key not in after:
            violations.append({"pid": int(key), "reason": "vanished_mid_span",
                               "argv0_basename": was.get("argv0_basename"),
                               "cmdline_head": was.get("cmdline_head")})
    return {"schema": IDLENESS_SCHEMA, "span": "closed", "span_s": span_s,
            "busy": bool(violations), "tolerated": tolerated,
            "violations": violations, "allowance_core_s": allowance,
            "allowance_cores": allowance_cores}


def violation_summary(witness: Mapping[str, Any]) -> str:
    """One readable line naming every unowned process that broke the span."""
    parts = []
    for item in witness.get("idleness", {}).get("violations", ()):
        core_s = item.get("cpu_core_seconds")
        measured = "" if core_s is None else f" {core_s:.3f} core-s"
        parts.append(f"pid {item.get('pid')} ({item.get('reason')}{measured}): "
                     f"{item.get('cmdline_head')}")
    return " | ".join(parts) or "no violation recorded"


def competing_inference_witness(
        *, previous_ledger: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Read only model-inference identities; ordinary CPU activity is excluded.

    `competing` is no longer "an unowned inference process EXISTS" but "an
    unowned inference process DID WORK in the span since `previous_ledger`".
    With no `previous_ledger` the call merely OPENS a span: it can refuse for
    an unreadable or vanishing process, never for a busy one, because nothing
    is known about a span with no start.  Chain the returned `cpu_ledger` into
    the next call at the far side of the measurement and the whole window is
    bracketed by a monotone counter.
    """
    try:
        owned = preflight.read_own_scope()
        scan = preflight.interim_process_scan(owned=owned)
    except preflight.PreflightUnavailable as exc:
        raise BaselineBankError("screening inference witness unavailable") from exc
    if scan.unreadable_pids:
        raise BaselineBankError("screening inference witness unreadable")
    unowned = scan.inference_like()
    findings = [item.to_dict() for item in unowned]
    ledger = read_cpu_ledger(unowned)
    verdict = idleness_verdict(previous_ledger, ledger)
    return {"basis": IDLENESS_BASIS, "competing": bool(verdict["busy"]),
            "findings": findings, "ordinary_processes_ignored": True,
            "resident_unowned_servers": len(findings),
            "idleness": verdict, "cpu_ledger": ledger}
