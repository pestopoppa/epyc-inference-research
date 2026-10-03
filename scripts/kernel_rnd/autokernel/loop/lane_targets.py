#!/usr/bin/env python3
"""Lane -> target binding: one lane per model, one shared champion lineage (2026-10-03).

WHY. Operator directive 2026-10-03: of the DS41 loop's two lanes, one keeps exploring
DeepSeek-V4.1-Flash and the other explores Qwen3.8-Flash-Next (Q38FN), whose token
path runs at ~17% of the host's read ceiling ("that autokernel lane must focus on
significantly increasing that abysmal 17% bandwidth ceiling usage"). A `run.py` child
owns exactly ONE target end to end -- its launch, anchor build, frozen requests, store,
accumulator, serving floor and serving gate -- so a lane bound to another model is
another child, not another thread: each lane is its own serial instance (`--workers
1`) on its own campaign/store, and the two instances' build+measure tails serialize
across processes through the CPU-region claim (`cpu_window`: a tail re-acquires the
claim, waiting on a peer, never pre-empting it), exactly as two lanes of one pool
serialize through `pipeline.SerializedTail`.

THE BINDING FILE (`SCHEMA`), shared by both instances and named by each one's common
args `--lane-targets <file> --lane <name>`:

    {"schema": "epyc.autokernel.lane_targets.v1",
     "lanes": {"lane0": {"target_id": ..., "owned_targets": <abs path>,
                         "exclusive_paths": [<fnmatch globs, source-tree relative>],
                         "as_peer_floor_file": <abs path> | null,
                         "as_peer_max_regression_pct": <number> | null,
                         "cpu_window_path": <abs path>   (optional)},
               "lane1": {...}}}

* the child refuses to run a target on a lane the file binds to another target, and
  takes `<file dir>/.lane-<name>.lock` for its lifetime, so two instances can never
  run one lane (a serial instance's children are sequential, so they never collide);
* `cpu_window_path` gives a non-default lane its own published window file (both
  instances writing the well-known path would overwrite each other's window; the
  region claim, not the file, is what serializes them);
* KEEPS. A keep must clear its own target's gates (unchanged) AND, when it touches a
  path outside its own lane's `exclusive_paths` (anything under ggml/ is shared by
  construction), must not regress any PEER target: the peer's launch and frozen
  requests are A/B'd with this lane's anchor build vs the candidate build (matched
  serving instrument, the builds are the ones the keep was measured with, so no extra
  build). The bar is the PEER entry's `as_peer_floor_file` (the peer's own matched
  floor record, validated by `serving.compare`); without one, its explicit
  `as_peer_max_regression_pct` (a point-estimate bar, recorded as such); with
  neither the keep is refused -- an absent bar never passes silently.

The flags are keep POLICY, not identity: the launch, requests, campaign and lineage of
each target are unchanged, so they are excluded from the serial continuation binding
(`serial_run.LANE_BINDING_FLAGS`) like the runtime-arm and pool-actor flags.
"""
from __future__ import annotations

import fcntl
import fnmatch
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.lane_targets.v1"
CHECK_SCHEMA = "epyc.autokernel.cross_target_check.v1"
FLAGS = ("--lane-targets", "--lane")
CHECK_DIR = "cross-target-checks"
MAX_BYTES = 64 * 1024
_ENTRY_FIELDS = {"target_id", "owned_targets", "exclusive_paths", "as_peer_floor_file",
                 "as_peer_max_regression_pct"}
_OPTIONAL_FIELDS = {"cpu_window_path"}


class LaneBindingError(ValueError):
    pass


@dataclass(frozen=True)
class LaneEntry:
    name: str
    target_id: str
    owned_targets: Path
    exclusive_paths: tuple[str, ...]
    as_peer_floor_file: Path | None
    as_peer_max_regression_pct: float | None
    cpu_window_path: Path | None = None


@dataclass(frozen=True)
class PeerTarget:
    """A peer lane's target as this lane cross-checks it: its own owned paths."""
    entry: LaneEntry
    launch_path: Path
    frozen_prompts: Path
    store: Path | None


@dataclass
class Bound:
    path: Path
    lane: LaneEntry
    peers: tuple[PeerTarget, ...]
    _lock: Any = field(default=None, repr=False)

    def release(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None


def _abs(value: Any, what: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\0" in value \
            or not Path(value).is_absolute():
        raise LaneBindingError(f"{what} must be an absolute path")
    return Path(value)


def _entry(name: str, body: Any) -> LaneEntry:
    if not isinstance(body, Mapping) or _ENTRY_FIELDS - body.keys() \
            or body.keys() - _ENTRY_FIELDS - _OPTIONAL_FIELDS:
        raise LaneBindingError(f"{name}: expected fields {sorted(_ENTRY_FIELDS)} "
                               f"plus optional {sorted(_OPTIONAL_FIELDS)}")
    target = body["target_id"]
    if not isinstance(target, str) or not target.strip():
        raise LaneBindingError(f"{name}: target_id must be nonempty text")
    globs = body["exclusive_paths"]
    if not isinstance(globs, list) or not all(
            isinstance(g, str) and g.strip() and not g.startswith("/") for g in globs):
        raise LaneBindingError(f"{name}: exclusive_paths must be relative glob strings")
    if any(g.startswith("ggml") for g in globs):
        raise LaneBindingError(f"{name}: ggml/ is shared by every target, never exclusive")
    floor = body["as_peer_floor_file"]
    bar = body["as_peer_max_regression_pct"]
    if bar is not None and (isinstance(bar, bool) or not isinstance(bar, (int, float))
                            or not math.isfinite(bar) or not 0 < bar <= 20):
        raise LaneBindingError(f"{name}: as_peer_max_regression_pct must be in (0, 20]")
    window = body.get("cpu_window_path")
    return LaneEntry(name=name, target_id=target, owned_targets=_abs(body["owned_targets"],
                     f"{name}.owned_targets"), exclusive_paths=tuple(globs),
                     as_peer_floor_file=(None if floor is None
                                             else _abs(floor, f"{name}.as_peer_floor_file")),
                     as_peer_max_regression_pct=None if bar is None else float(bar),
                     cpu_window_path=None if window is None else _abs(window,
                                                                      f"{name}.cpu_window_path"))


def load(path: Path) -> dict[str, LaneEntry]:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise LaneBindingError("lane binding file exceeds 64 KiB")
    body = json.loads(raw)
    if not isinstance(body, Mapping) or set(body) != {"schema", "lanes"} \
            or body["schema"] != SCHEMA or not isinstance(body["lanes"], Mapping):
        raise LaneBindingError(f"lane binding must be {{schema: {SCHEMA}, lanes: {{...}}}}")
    lanes = {name: _entry(name, entry) for name, entry in body["lanes"].items()}
    if len(lanes) < 2:
        raise LaneBindingError("a lane binding names at least two lanes")
    if any(not name.startswith("lane") or not name[4:].isdigit() for name in lanes):
        raise LaneBindingError("lane names are lane<N>")
    targets = [entry.target_id for entry in lanes.values()]
    if len(set(targets)) != len(targets):
        raise LaneBindingError("one target is bound to two lanes")
    return lanes


def _owned(entry: LaneEntry) -> Mapping[str, Any]:
    body = json.loads(entry.owned_targets.read_bytes())
    owner = body.get(entry.target_id) if isinstance(body, Mapping) else None
    if not isinstance(owner, Mapping) or "frozen_prompts" not in owner or "launch" not in owner:
        raise LaneBindingError(f"{entry.name}: {entry.owned_targets} owns no launch/frozen "
                               f"prompts for {entry.target_id}")
    return owner


def resolve(path: Path | None, lane: str | None, *, target_id: str | None, workers: int,
            lock: bool = True) -> Bound:
    """The child's lane, after every refusal that can be made before measuring."""
    if path is None or lane is None:
        raise LaneBindingError("--lane-targets and --lane are given together")
    lanes = load(path)
    if lane not in lanes:
        raise LaneBindingError(f"--lane {lane} is not bound in {path}")
    entry = lanes[lane]
    if target_id != entry.target_id:
        raise LaneBindingError(f"{lane} is bound to {entry.target_id}, not {target_id}")
    if int(workers) != 1:
        raise LaneBindingError("a bound lane runs ONE worker (--workers 1): the other lane "
                               "is the other target's instance")
    peers = []
    for name, other in sorted(lanes.items()):
        if name == lane:
            continue
        owner = _owned(other)
        peers.append(PeerTarget(entry=other, launch_path=_abs(owner["launch"], f"{name} launch"),
                                frozen_prompts=_abs(owner["frozen_prompts"], f"{name} prompts"),
                                store=(None if owner.get("store") is None
                                       else _abs(owner["store"], f"{name} store"))))
    # A peer with no bar is not a startup refusal: a keep touching only this lane's
    # exclusive paths needs none, and any other keep is refused at `decide`.
    bound = Bound(path=Path(path), lane=entry, peers=tuple(peers))
    if lock:
        handle = (Path(path).parent / f".lane-{lane}.lock").open("a")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise LaneBindingError(f"{lane} already has a running instance") from None
        bound._lock = handle
    return bound


def peer_serving(peer: PeerTarget):
    """(canonical launch, frozen requests) of a peer target, through the loop's own
    validators -- called at startup so a malformed peer refuses before any work."""
    from . import planned_serving, resolved_recipe as rr
    launch = rr.CanonicalResolvedRecipe.from_dict(json.loads(peer.launch_path.read_bytes()))
    manifest = planned_serving.FrozenPromptManifest.from_dict(
        json.loads(peer.frozen_prompts.read_bytes()))
    requests = manifest.requests(tuple(p.prompt_id for p in manifest.prompts), launch.template)
    if len(requests) != launch.template.np:
        raise LaneBindingError(f"{peer.entry.name}: peer requests differ from its concurrency")
    return launch, requests


def compare_kwargs(bar: Mapping[str, Any], template, requests) -> dict:
    """`serving.compare` keyword arguments for a bar: the floor record (matched, request
    bound) or nothing (uncalibrated point-estimate comparison)."""
    if bar["mode"] != "floor":
        return {"floor_pct": None}
    from . import serving
    return {"floor_pct": bar["floor_pct"], "floor_unit": bar["floor_unit"],
            "floor_record": bar["floor_record"],
            "floor_request_digest": serving.request_digest(template, requests)}


def needs_cross_check(changed: Iterable[str], exclusive: Sequence[str]) -> bool:
    """True unless EVERY changed path is inside this lane's exclusive globs. An empty
    change set cannot prove exclusivity, so it is checked."""
    paths = [p for p in changed if p]
    if not paths:
        return True
    return not all(any(fnmatch.fnmatchcase(p, glob) for glob in exclusive) for p in paths)


def peer_bar(peer: PeerTarget) -> dict:
    """The bar for one peer: its matched floor record, else its explicit point bar."""
    entry = peer.entry
    if entry.as_peer_floor_file is not None:
        record = json.loads(entry.as_peer_floor_file.read_bytes())
        if not isinstance(record, Mapping) or record.get("unit") != "process" \
                or not isinstance(record.get("floor_pct"), (int, float)):
            raise LaneBindingError(f"{entry.name}: {entry.as_peer_floor_file} is not a "
                                   "process-unit serving floor record")
        return {"mode": "floor", "floor_pct": float(record["floor_pct"]),
                "floor_unit": "process", "floor_record": dict(record),
                "floor_file": str(entry.as_peer_floor_file)}
    if entry.as_peer_max_regression_pct is not None:
        return {"mode": "point", "max_regression_pct": entry.as_peer_max_regression_pct}
    return {"mode": "absent"}


def decide(row: Mapping[str, Any] | None, bar: Mapping[str, Any]) -> tuple[bool, str]:
    """(passed, reason) for one peer comparison row (`serving.compare` output)."""
    if bar["mode"] == "absent":
        return False, "no cross-check bar declared for the peer (floor file or point bar)"
    if row is None or not isinstance(row.get("effect"), (int, float)) \
            or not math.isfinite(row["effect"]):
        return False, "peer comparison produced no finite effect"
    pct = row["effect"] * 100.0
    if bar["mode"] == "floor":
        if row.get("decisive") is True and pct < 0:
            return False, (f"peer regressed {pct:+.3f}% beyond its {bar['floor_pct']:.3f}% "
                           "matched floor")
        return True, f"peer {pct:+.3f}% within/above its {bar['floor_pct']:.3f}% floor"
    limit = bar["max_regression_pct"]
    if pct < -limit:
        return False, f"peer regressed {pct:+.3f}% past the declared -{limit:.3f}% point bar"
    return True, f"peer {pct:+.3f}% within the declared -{limit:.3f}% point bar"


def cross_check(bound: Bound, *, changed: Iterable[str],
                compare: Callable[[PeerTarget, dict], Mapping[str, Any]],
                store: Path, mechanism_id: str, now: Callable[[], float] = time.time) -> dict:
    """Run the cross-target non-regression check for one keep; always records it.

    `compare(peer, bar)` runs the peer A/B (injected: the loop owns builds/launches)
    and returns the `serving.compare` row. Returns the verdict (`passed`); the caller
    vetoes the keep on `passed is False`. A comparison that raises is a failed check
    (fail closed), recorded with its error."""
    changed = sorted({p for p in changed if p})
    verdict: dict[str, Any] = {"schema": CHECK_SCHEMA, "lane": bound.lane.name,
                               "target_id": bound.lane.target_id,
                               "mechanism_id": mechanism_id, "changed_paths": changed,
                               "exclusive_paths": list(bound.lane.exclusive_paths),
                               "at": now(), "peers": []}
    if not needs_cross_check(changed, bound.lane.exclusive_paths):
        verdict.update(required=False, passed=True,
                       reason="every changed path is exclusive to this lane's target")
    else:
        verdict["required"] = True
        passed = True
        for peer in bound.peers:
            row_out: dict[str, Any] = {"lane": peer.entry.name,
                                       "target_id": peer.entry.target_id}
            try:
                bar = peer_bar(peer)
                row_out["bar"] = {k: v for k, v in bar.items() if k != "floor_record"}
                row = None if bar["mode"] == "absent" else dict(compare(peer, bar))
                ok, reason = decide(row, bar)
                row_out.update(effect=None if row is None else row.get("effect"),
                               decisive=None if row is None else row.get("decisive"),
                               pairs=None if row is None else row.get("pairs"))
            except Exception as exc:  # noqa: BLE001 -- fail closed, recorded
                ok, reason = False, f"peer check failed: {type(exc).__name__}: {exc}"
            row_out.update(passed=ok, reason=reason)
            verdict["peers"].append(row_out)
            passed = passed and ok
        verdict["passed"] = passed
        verdict["reason"] = "; ".join(f"{p['lane']}: {p['reason']}" for p in verdict["peers"])
    directory = Path(store) / CHECK_DIR
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{int(verdict['at'] * 1000)}-{mechanism_id[:80].replace('/', '_')}.json"
    (directory / name).write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return verdict


__all__ = ["Bound", "CHECK_SCHEMA", "FLAGS", "LaneBindingError", "LaneEntry", "PeerTarget",
           "SCHEMA", "cross_check", "decide", "load", "needs_cross_check", "peer_bar",
           "resolve"]
