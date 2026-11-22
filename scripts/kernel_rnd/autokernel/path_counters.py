"""In-window path counters for T1/T2 timing records (AK-WIN-CNT-1). Non-gating.

Annex K's record grammar has no field proving the timed window ran the intended
path. A llama.cpp-experimental build carrying the AK-WIN-CNT-1 port
(`GGML_CUDA_PATH_COUNTERS=1`, patch under the row dir
`/mnt/raid0/llm/tmp/ak-win-cnt-1-20261119-r1/patches/path-counters.patch`) makes
`llama-bench` reset the CUDA/HIP backend's graph_compute counters after warmup and
print, on stderr, after the timed repetitions::

    autokernel_path_counters: {"enabled":1,"calls":..,"eager":..,"capture":..,"replay":..,"graph_resets":..}

This module parses that line and classifies one window. It is a DIAGNOSTIC SIDECAR:
nothing here may change an effect, a floor, `decisive` or a keep. Making a path
mismatch VOID a window is an Annex K amendment, which is human-amendment-only
(`docs/design/ak-win-cnt-1-annexk-path-counters-amendment-draft.md` in epyc-root).

A binary without the port prints nothing; that is "unavailable", never "eager".
Code-object load / JIT events are not counted in-process; capture and graph_resets
after warmup are the stated proxy.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Mapping

SCHEMA = "epyc.autokernel.path_counters.v1"
FIELDS = ("calls", "eager", "capture", "replay", "graph_resets")
_LINE = re.compile(r"^autokernel_path_counters: (\{.*\})\s*$", re.MULTILINE)

#: The row rule (ak-win-cnt-1-20261119-r1, RULE): a graphed window is VALID iff
#: replay/calls >= 0.90, eager/calls <= 0.05 and (capture+graph_resets)/calls <= 0.05.
MIN_REPLAY_SHARE = 0.90
MAX_EAGER_SHARE = 0.05
MAX_REBUILD_SHARE = 0.05


def parse_stderr(stderr: str | bytes | None) -> dict | None:
    """The LAST counters line on stderr, or None when the binary printed none."""
    if not stderr:
        return None
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    found = _LINE.findall(stderr)
    if not found:
        return None
    try:
        body = json.loads(found[-1])
    except ValueError:
        return {"enabled": 0, "malformed": True}
    if not isinstance(body, dict):
        return {"enabled": 0, "malformed": True}
    out: dict = {"enabled": int(bool(body.get("enabled")))}
    if body.get("unbound"):
        out["unbound"] = True
    for key in FIELDS:
        value = body.get(key)
        if type(value) is int and value >= 0:
            out[key] = value
        elif key in body:
            out["malformed"] = True
    return out


def of(done) -> dict | None:
    """`parse_stderr` of a finished `subprocess.CompletedProcess`."""
    return parse_stderr(getattr(done, "stderr", None))


def classify(counters: Mapping | None) -> dict:
    """One window: state in {unavailable, disabled, graphed, eager, mixed} + reasons.

    `graphed` is the row rule's VALID; `eager` is a graphs-off window (replay 0,
    eager == calls) -- the degraded-negative control must land here.
    """
    if counters is None:
        return {"state": "unavailable", "reasons": ["no counters line (binary lacks the port)"]}
    if counters.get("malformed"):
        return {"state": "unavailable", "reasons": ["malformed counters line"]}
    if not counters.get("enabled") or not all(k in counters for k in FIELDS):
        return {"state": "disabled", "reasons": ["counters not enabled"]}
    calls = counters["calls"]
    if calls <= 0:
        return {"state": "disabled", "reasons": ["zero graph_compute calls in window"]}
    if counters["replay"] == 0 and counters["eager"] == calls:
        return {"state": "eager", "reasons": ["no graph replay in window"]}
    reasons = []
    if counters["replay"] / calls < MIN_REPLAY_SHARE:
        reasons.append(f"replay/calls {counters['replay'] / calls:.3f} < {MIN_REPLAY_SHARE}")
    if counters["eager"] / calls > MAX_EAGER_SHARE:
        reasons.append(f"eager/calls {counters['eager'] / calls:.3f} > {MAX_EAGER_SHARE}")
    rebuild = counters["capture"] + counters["graph_resets"]
    if rebuild / calls > MAX_REBUILD_SHARE:
        reasons.append(f"(capture+graph_resets)/calls {rebuild / calls:.3f} > {MAX_REBUILD_SHARE}")
    return {"state": "mixed" if reasons else "graphed", "reasons": reasons}


def sidecar(per_invocation: Iterable[Mapping | None]) -> dict:
    """The record field: every window's raw counters kept, in invocation order.

    `consistent` is False when the measured windows did not all run one path --
    the condition the draft amendment would make VOID. Report-only here.
    """
    raw = [None if c is None else dict(c) for c in per_invocation]
    states = [classify(c)["state"] for c in raw]
    seen = sorted(set(states))
    return {"schema": SCHEMA, "authority": "report_only", "gating": False,
            "windows": raw, "states": states,
            "consistent": len(seen) == 1 and seen[0] in ("graphed", "eager"),
            "available": bool(raw) and "unavailable" not in seen}
