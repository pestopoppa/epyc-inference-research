"""OAB-10/11 A/B for the AutoKernel opencode PLAIN planner seat (DS41 run-8 prompt).

Successor of `/mnt/raid0/llm/tmp/ak-ctx-ab/driver.py` (OAB-9, inline vs variable; inline
won). Here the seat AND the context placement are fixed (plain `opencode run --auto`,
inline context: the campaign default) and the OAB-10/11 knobs are the treatment:

  baseline : `ActorSeat(bounded=False, context_mode="inline")` -- the historical seat.
             The prompt is the OAB-9 control, byte for byte (run 8's planner prompt +
             node_profile, sha a265a03a); opencode loads the lane AGENTS.md and the
             ~/.claude skill catalog, all native tools are offered, nothing is fenced.
  trimmed  : the same seat with `trim_instructions`, `trim_tools` and `lane_guard` on:
             OPENCODE_DISABLE_PROJECT_CONFIG/_CLAUDE_CODE/_EXTERNAL_SKILLS, a per-call
             permission-only OPENCODE_CONFIG (skill/unused tools/edit denied, builds and
             anchor-source reads denied, all writes denied), and the lane block ahead of
             the same inline context.

Both arms run the REAL `AgentPlanner.propose` of `lane/ak-oab1011-20260925`, patched
exactly as OAB-9 was: `render_context` returns the control bundle text and
`_with_backoff` makes ONE attempt. The target context carries run 8's real anchor build
dir (`/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu`) in both arms;
the baseline prompt ignores it (knobs off), the guard fences its tree.

LOOP GATE (default `--loop-control gate`). DS41 run 9c measures on the CPU. Before EVERY
call the driver waits until the live loop (state dir `STATE_DIR`) is not measuring or
calibrating: no process under its `run.py` child is a `llama-server`/`llama-bench`
(read from /proc only; nothing is signalled). Every wait is logged to `gate-log.jsonl`
and stdout with its duration and the blocking pids. The loop's state is also SAMPLED
every 60 s during each call and recorded (`loop_during_call`): a call that overlapped a
measurement is flagged, never hidden.

`--loop-control pause` pauses the serial owner between batches through its authenticated
control listener, runs every call, and resumes it. Available only if the owner was
launched with `--control-listen` (and the token is in AUTOKERNEL_CONTROL_TOKEN); run 9c
was NOT, so this mode refuses and prints the fallback the MAIN session executes (see
`pause_plan`). `--loop-control none` skips the gate (never on a live campaign).

Same :8083 server generation for every call of a pair (OAB-4a), `/slots` sampled every
30 s during each call, exactly as OAB-9.

OAB-11 acceptance columns per call (from the session export's tool inputs and the
opencode log's `evaluated permission=...` lines): reads under the lane vs under the
anchor source root, compiler/build invocations attempted, and permission denies.

usage:
  python3 driver.py dry-run                       # no opencode, no model port, no signal
  python3 driver.py run --pairs 3                 # ABBA: baseline,trimmed / trimmed,baseline ...
  python3 driver.py run --pairs 2 --budget-h 3.5
  python3 driver.py summarize
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock
import urllib.request

WORKTREE = Path("/mnt/raid0/llm/worktrees/research-ak-oab1011-20260925")
sys.path[:0] = [str(WORKTREE / "scripts/kernel_rnd"), str(WORKTREE / "scripts/lib"),
                str(WORKTREE / "scripts"), str(WORKTREE)]
from autokernel.loop import actor_metrics, actor_opencode_config as aoc, actors  # noqa: E402
from autokernel.loop import test_actor_context as fx  # noqa: E402  (fixture helpers)

AB = Path("/mnt/raid0/llm/tmp/ak-oab1011-ab")
#: The OAB-9 lane (a detached llama.cpp worktree at the run-8 anchor). The workspace is
#: a LINK in AB, so configs, replies and metrics land beside it here.
LANE_TARGET = Path("/mnt/raid0/llm/tmp/ak-seat-ab/lane")
LANE = AB / "lane"
ANCHOR = "ebb68dc55d5f6af4a4a5dccdd2a013fa76c63bee"   # run 8's COR anchor
BUILD_DIR = "/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu"
MODEL = "qwen-gpu/qwen3.8-27b"
EFFORT = "high"
PORT = 8083
TIMEOUT_S = 5400
KNOBS = {"trim_instructions": True, "trim_tools": True, "lane_guard": True}
ARMS = {"baseline": {}, "trimmed": KNOBS}
RESULTS = AB / "results.jsonl"
GATE_LOG = AB / "gate-log.jsonl"
CALL_LOG = AB / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
OPENCODE_LOG = Path.home() / ".local/share/opencode/log/opencode.log"
STATIC = WORKTREE / "artifacts/autokernel_oab1011_20260925/fixed_overhead.json"
CHARS_PER_TOKEN = 2.89
KV_ON = ("-kvu", "--kv-unified")
KV_OFF = ("-no-kvu", "--no-kv-unified")
CONTEXT = {"target": {"recipe": {"backend": "cpu", "build_dir": BUILD_DIR}}}

#: The live campaign the gate protects.
CAMPAIGN = Path("/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923")
STATE_DIR = CAMPAIGN / "state-run9c"
STORE = CAMPAIGN / "store"
MEASURING = ("llama-server", "llama-bench")
ANCHOR_TREE = "/mnt/raid0/llm/llama.cpp-experimental-deepseek41-"
GATE_POLL_S = 20.0
COMPILERS = re.compile(r"(?:^|[\s/;&|(])(cmake|make|ninja|gcc|g\+\+|clang\+?\+?|cc|c\+\+|ld)"
                       r"(?:\s|$)|\.o(?:\s|$)")


def control_text() -> str:
    return fx._real_context_text()


def planner(workspace: Path, arm: str) -> actors.AgentPlanner:
    return actors.AgentPlanner(workspace=workspace, backend=actors.backend_for(MODEL, EFFORT),
                               timeout_s=TIMEOUT_S,
                               seat=actors.ActorSeat(bounded=False, context_mode="inline",
                                                     **ARMS[arm]))


def one_attempt(call, **_kw):
    return call(), 0


def est_tokens(chars: int) -> int:
    return round(chars / CHARS_PER_TOKEN)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- /proc

def _procs() -> dict[int, dict]:
    """pid -> {ppid, argv, comm}: one /proc snapshot (inspection only)."""
    out = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [a.decode(errors="replace")
                    for a in (proc / "cmdline").read_bytes().split(b"\0") if a]
            status = (proc / "status").read_text()
            comm = (proc / "comm").read_text().strip()
        except OSError:
            continue
        ppid = int(re.search(r"^PPid:\s+(\d+)", status, re.M).group(1))
        out[int(proc.name)] = {"ppid": ppid, "argv": argv, "comm": comm}
    return out


def _descendants(procs: dict[int, dict], root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, info in procs.items():
        children.setdefault(info["ppid"], []).append(pid)
    out, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def loop_state() -> dict:
    """Is the DS41 loop measuring/calibrating right now? From /proc only."""
    procs = _procs()
    state_arg = str(STATE_DIR)
    owners = [pid for pid, info in procs.items()
              if "autokernel.loop.serial_run" in " ".join(info["argv"])
              and state_arg in info["argv"]]
    runs = [pid for owner in owners for pid in _descendants(procs, owner)
            if re.search(r"autokernel\.loop\.run(\s|$)", " ".join(procs[pid]["argv"]))]
    measuring = []
    for run in runs:
        for pid in _descendants(procs, run):
            info = procs[pid]
            name = Path(info["argv"][0]).name if info["argv"] else info["comm"]
            if name in MEASURING or info["comm"] in MEASURING:
                measuring.append({"pid": pid, "name": name, "parent": info["ppid"]})
    # Belt and braces: a measuring process of THIS campaign that is not (or no longer)
    # under run.py -- reparented after a double fork -- still blocks the gate.
    seen = {m["pid"] for m in measuring}
    for pid, info in procs.items():
        name = Path(info["argv"][0]).name if info["argv"] else info["comm"]
        text = " ".join(info["argv"])
        if pid not in seen and name in MEASURING and (
                str(CAMPAIGN) in text or ANCHOR_TREE in text):
            measuring.append({"pid": pid, "name": name, "parent": info["ppid"],
                              "matched": "campaign/anchor path in argv"})
    listen = None
    for owner in owners:
        argv = procs[owner]["argv"]
        if "--control-listen" in argv:
            listen = argv[argv.index("--control-listen") + 1]
    return {"ts": stamp(), "owner_pids": owners, "run_pids": runs,
            "measuring": measuring, "busy": bool(measuring), "loop_running": bool(owners),
            "control_listen": listen}


def gate(label: str, *, max_wait_s: float | None) -> dict:
    """Block until the loop is not measuring; log every wait. Returns the gate record."""
    started = time.time()
    first = state = loop_state()
    logged = False
    while state["busy"]:
        if not logged:
            print(f"  gate[{label}]: DS41 loop is measuring {[m['name'] for m in state['measuring']]} "
                  f"(pids {[m['pid'] for m in state['measuring']]}); waiting", flush=True)
            logged = True
        if max_wait_s is not None and time.time() - started > max_wait_s:
            break
        time.sleep(GATE_POLL_S)
        state = loop_state()
    record = {"label": label, "started": first["ts"], "ended": stamp(),
              "waited_s": round(time.time() - started, 1), "blocked_by": first["measuring"],
              "cleared": not state["busy"], "loop_running": state["loop_running"]}
    with open(GATE_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if logged:
        print(f"  gate[{label}]: {'cleared' if record['cleared'] else 'GAVE UP'} after "
              f"{record['waited_s']:.0f}s", flush=True)
    return record


class LoopSampler(threading.Thread):
    """Samples the loop state every `period` s DURING a call: overlap is recorded."""

    def __init__(self, period: float = 60.0):
        super().__init__(daemon=True)
        self.period, self.samples = period, []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(loop_state()["busy"])
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=10)
        return {"samples": len(self.samples), "samples_measuring": sum(self.samples),
                "overlapped_measurement": any(self.samples)}


# --------------------------------------------------------------------------- pause mode

def pause_plan(state: dict) -> dict:
    """What `--loop-control pause` can do for this loop, and the exact commands."""
    token = bool(os.environ.get("AUTOKERNEL_CONTROL_TOKEN"))
    listen = state.get("control_listen")
    if listen:
        base = f"http://{listen}"
        return {"available": token, "reason": None if token else
                "AUTOKERNEL_CONTROL_TOKEN is not set in this environment",
                "endpoint": base,
                "commands": [
                    f"curl -s -H 'Authorization: Bearer $AUTOKERNEL_CONTROL_TOKEN' {base}/snapshot",
                    f"curl -s -X POST -H 'Authorization: Bearer $AUTOKERNEL_CONTROL_TOKEN' "
                    f"-H 'Content-Type: application/json' {base}/commands -d "
                    "'{\"schema\":\"epyc.autokernel.serial_command.v1\",\"config_digest\":<snapshot>,"
                    "\"owner_id\":<snapshot>,\"request_id\":\"oab1011-pause-1\",\"operation\":\"pause\","
                    "\"expected_revision\":<snapshot.revision>,\"expected_batch\":<snapshot.batch_number>,"
                    "\"expected_target\":<snapshot.active_target>}'",
                    "poll /snapshot until observed_state == \"paused\" (the owner finishes the "
                    "current batch, then idles with no child and no claims); resume = same POST "
                    "with operation \"resume\" and a new request_id"]}
    return {"available": False,
            "reason": ("the serial owner was launched without --control-listen, so it has no "
                       "control listener or token; pause/resume cannot be requested"
                       if state.get("loop_running") else "the loop is not running"),
            "fallback_for_main_session": [
                "1. wait for a forming boundary: the gate condition (no llama-server/"
                "llama-bench under the run.py child) -- `python3 driver.py gate-status`",
                f"2. touch {STATE_DIR}/STOP   (serial owner starts no further batch)",
                f"3. touch {STORE}/STOP   (run.py: forming lanes abandon before their next "
                "actor call; a lane holding the measurement tail finishes and publishes first)",
                "4. wait until the serial_run and run.py pids have exited (ps -p <pid>)",
                "5. run the A/B with --loop-control none",
                f"6. rm {STATE_DIR}/STOP {STORE}/STOP and relaunch run 9c with its exact argv "
                "(or a new state dir, as run 9 -> 9b -> 9c were), floors reused",
            ]}


def _control(base: str, path: str, body: dict | None = None) -> dict:
    token = os.environ["AUTOKERNEL_CONTROL_TOKEN"]
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{base}{path}", data=data, method="GET" if body is None else "POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          **({"Content-Type": "application/json"} if body else {})})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def pause_loop(plan: dict, operation: str, *, wait_s: float = 6 * 3600) -> dict:
    base = plan["endpoint"]
    snap = _control(base, "/snapshot")
    body = {"schema": "epyc.autokernel.serial_command.v1",
            "config_digest": snap["config_digest"], "owner_id": snap["owner_id"],
            "request_id": f"oab1011-{operation}-{int(time.time())}", "operation": operation,
            "expected_revision": snap["revision"], "expected_batch": snap["batch_number"],
            "expected_target": snap["active_target"]}
    result = _control(base, "/commands", body)
    want = "paused" if operation == "pause" else "running"
    deadline = time.time() + wait_s
    while time.time() < deadline:
        snap = _control(base, "/snapshot")
        if snap["observed_state"] == want:
            return {"operation": operation, "result": result, "observed": want}
        time.sleep(GATE_POLL_S)
    raise RuntimeError(f"loop did not reach {want} within {wait_s}s")


# --------------------------------------------------------------------------- server

def server_process() -> dict | None:
    """The :8083 llama-server, read from /proc (inspection only; nothing is signalled)."""
    for pid, info in _procs().items():
        argv = info["argv"]
        if not argv or not argv[0].endswith("llama-server"):
            continue
        if "--port" in argv and argv[argv.index("--port") + 1:argv.index("--port") + 2] == [str(PORT)]:
            proc = Path("/proc") / str(pid)
            try:
                start = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
            except (OSError, IndexError):
                start = None
            log = None
            for fd in ("1", "2"):
                try:
                    target = os.readlink(proc / "fd" / fd)
                except OSError:
                    continue
                if target.startswith("/") and Path(target).is_file():
                    log = target
                    break
            return {"pid": pid, "start_ticks": start, "argv": argv, "log": log}
    return None


def kv_unified_from_argv(argv: list[str]) -> bool:
    for token in reversed(argv):
        if token in KV_ON:
            return True
        if token in KV_OFF:
            return False
    return not ("-np" in argv or "--parallel" in argv)


def http_json(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generation(*, live: bool) -> dict:
    """Fingerprint of the :8083 server. `live=False` never touches the port."""
    proc = server_process()
    if proc is None:
        return {"found": False}
    out = {"found": True, "pid": proc["pid"], "start_ticks": proc["start_ticks"],
           "cmdline_sha256": hashlib.sha256("\0".join(proc["argv"]).encode()).hexdigest(),
           "binary": proc["argv"][0], "kv_unified_cmdline": kv_unified_from_argv(proc["argv"]),
           "launch_log": proc["log"],
           "np": (proc["argv"][proc["argv"].index("-np") + 1] if "-np" in proc["argv"] else None),
           "c": (proc["argv"][proc["argv"].index("-c") + 1] if "-c" in proc["argv"] else None)}
    if live:
        try:
            props = http_json("/props")
            dgs = props.get("default_generation_settings") or {}
            out["props_n_ctx"] = dgs.get("n_ctx", props.get("n_ctx"))
            out["props_total_slots"] = props.get("total_slots")
            out["props_build_info"] = props.get("build_info")
        except Exception as exc:  # noqa: BLE001
            out["props_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def same_generation(a: dict, b: dict) -> bool:
    keys = ("pid", "start_ticks", "cmdline_sha256")
    return all(a.get(k) == b.get(k) for k in keys) and a.get("found") and b.get("found")


class SlotSampler(threading.Thread):
    def __init__(self, period: float = 30.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.errors = period, [], 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                slots = http_json("/slots")
                self.samples.append(sum(1 for s in slots if s.get("is_processing")))
            except Exception:  # noqa: BLE001
                self.errors += 1
            self._stop.wait(self.period)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=10)
        return {"samples": len(self.samples), "errors": self.errors,
                "max_busy": max(self.samples) if self.samples else None,
                "samples_with_other_busy": sum(1 for n in self.samples if n > 1)}


# --------------------------------------------------------------------------- lane

def ensure_lane_link() -> None:
    if not LANE.is_symlink():
        LANE.symlink_to(LANE_TARGET)


def lane_state() -> dict:
    head = subprocess.run(["git", "-C", str(LANE), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(LANE), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return {"link": str(LANE), "real": str(LANE.resolve()), "head": head,
            "at_anchor": head == ANCHOR, "clean": not dirty}


def reset_lane() -> None:
    subprocess.run(["git", "-C", str(LANE), "reset", "-q", "--hard", ANCHOR], check=True)
    subprocess.run(["git", "-C", str(LANE), "clean", "-qfd"], check=True)


# --------------------------------------------------------------------------- metrics

def new_rows(offset: int) -> list[dict]:
    if not CALL_LOG.is_file():
        return []
    with open(CALL_LOG, "rb") as handle:
        handle.seek(offset)
        text = handle.read().decode(errors="replace")
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fence_columns(metrics_row: dict | None) -> dict:
    """OAB-11 acceptance from the session exports: where the planner read, and whether
    it tried to build."""
    root, _builds, _ = aoc.anchor_fence(BUILD_DIR, LANE_TARGET)
    root = str(root) if root else None
    lane_paths = (str(LANE), str(LANE_TARGET))
    out = {"tool_calls": 0, "calls_naming_lane": 0, "calls_naming_anchor_source": 0,
           "calls_naming_anchor_build": 0, "build_like_commands": [], "anchor_source_examples": []}
    for session in (((metrics_row or {}).get("opencode") or {}).get("sessions") or []):
        path = (session.get("export") or {}).get("path")
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        for message in data.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") != "tool":
                    continue
                out["tool_calls"] += 1
                inp = (part.get("state") or {}).get("input") or {}
                text = json.dumps(inp)
                if any(p in text for p in lane_paths):
                    out["calls_naming_lane"] += 1
                if root and root in text:
                    if re.search(re.escape(root) + r"/build", text) and not re.search(
                            re.escape(root) + r"/(?!build)", text):
                        out["calls_naming_anchor_build"] += 1
                    else:
                        out["calls_naming_anchor_source"] += 1
                        if len(out["anchor_source_examples"]) < 5:
                            out["anchor_source_examples"].append(text[:200])
                command = inp.get("command") if part.get("tool") == "bash" else None
                if command and COMPILERS.search(command):
                    out["build_like_commands"].append(command[:200])
    return out


def opencode_denies(offset: int, session_ids: list[str]) -> dict:
    """`evaluated permission=... action.action=deny` lines the call produced."""
    if not OPENCODE_LOG.is_file() or not session_ids:
        return {"denies": None}
    runs: set[str] = set()
    denies = []
    with open(OPENCODE_LOG, "rb") as handle:
        handle.seek(offset)
        lines = handle.read().decode(errors="replace").splitlines()
    for line in lines:
        if any(s in line for s in session_ids):
            found = re.search(r"\brun=(\w+)", line)
            if found:
                runs.add(found.group(1))
    for line in lines:
        found = re.search(r"\brun=(\w+)", line)
        if found and found.group(1) in runs and "message=evaluated" in line \
                and "action.action=deny" in line:
            denies.append(line.split("message=evaluated", 1)[1][:240])
    return {"denies": len(denies), "deny_examples": denies[:10]}


def metrics_summary(row: dict | None) -> dict:
    if not row:
        return {"metrics_row": False}
    oc = row.get("opencode") or {}
    totals = oc.get("totals") or {}
    tools: dict[str, int] = {}
    for session in oc.get("sessions") or []:
        for name, n in (session.get("tools") or {}).items():
            tools[name] = tools.get(name, 0) + n
    return {"metrics_row": True, "seat_arm": row.get("seat_arm"),
            "seat_config": row.get("seat_config"), "seat_env": row.get("seat_env"),
            "metrics_error": row.get("metrics_error"),
            "sessions": oc.get("session_ids"), "steps": totals.get("steps"),
            "tool_calls": totals.get("tool_calls"), "tools": tools,
            "decoded_tokens": totals.get("decoded_tokens"),
            "prompt_tokens": totals.get("prompt_tokens"),
            "cache_read_tokens": totals.get("cache_read_tokens"),
            "compactions": totals.get("compactions"),
            "context_first_tokens": oc.get("context_first_tokens"),
            "context_max_tokens": oc.get("context_max_tokens"),
            "tool_output_chars": totals.get("tool_output_chars"),
            "wall_s": row.get("wall_s"), "returncode": row.get("returncode"),
            "timed_out": row.get("timed_out"), "salvaged": row.get("salvaged"),
            "schema_valid": row.get("schema_valid"), "repair_ran": row.get("repair_ran")}


# --------------------------------------------------------------------------- one call

def run_call(pair: int, arm: str, pair_generation: dict, args) -> dict:
    result: dict = {"pair": pair, "arm": arm, "knobs": ARMS[arm], "timeout_s": TIMEOUT_S}
    if args.loop_control == "gate":
        result["gate"] = gate(f"p{pair}-{arm}", max_wait_s=args.gate_max_wait_h * 3600
                              if args.gate_max_wait_h else None)
        if not result["gate"]["cleared"]:
            result["gate_gave_up"] = True
            return result
    gen = generation(live=True)
    result.update(started=stamp(), server=gen)
    if not same_generation(gen, pair_generation):
        result["generation_changed"] = True
        return result
    reset_lane()
    result["lane"] = lane_state()
    offset = CALL_LOG.stat().st_size if CALL_LOG.is_file() else 0
    log_offset = OPENCODE_LOG.stat().st_size if OPENCODE_LOG.is_file() else 0
    seen: dict = {}
    real_run_agent = actors._run_agent

    def recording_run_agent(prompt, **kw):
        seen["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        seen["prompt_chars"] = len(prompt)
        env = kw.get("env") or {}
        seen["env_arm"] = env.get(actors.SEAT_ENV_ARM)
        seen["env_keys"] = sorted(env)
        return real_run_agent(prompt, **kw)

    slots, loop = SlotSampler(), LoopSampler()
    slots.start()
    loop.start()
    started = time.time()
    try:
        with mock.patch.object(actors, "render_context", return_value=control_text()), \
                mock.patch.object(actors, "_with_backoff", one_attempt), \
                mock.patch.object(actors, "_run_agent", recording_run_agent):
            outcome = planner(LANE, arm).propose(CONTEXT)
        if isinstance(outcome, actors.Abstain):
            result["abstain"] = outcome.reason
        else:
            result["hypothesis"] = {k: getattr(outcome, k) for k in (
                "mechanism_id", "statement", "falsifier", "target_surface", "target_symbol")}
    except actors.ProviderTransient as exc:
        result["transient"] = str(exc)[:800]
    finally:
        result["driver_wall_s"] = round(time.time() - started, 1)
        result["slots"] = slots.stop()
        result["loop_during_call"] = loop.stop()
    result.update(seen)
    if arm == "baseline":
        result["prompt_is_control"] = seen.get("prompt_sha256") == fx.CONTROL_SHA256
    result["lane_edited"] = not lane_state()["clean"]
    rows = new_rows(offset)
    metrics = [r for r in rows if r.get("schema") == actor_metrics.METRICS_SCHEMA]
    result["actor_calls"] = len(metrics)
    result["metrics"] = metrics_summary(metrics[0] if metrics else None)
    result["fence"] = fence_columns(metrics[0] if metrics else None)
    result["permission"] = opencode_denies(log_offset, result["metrics"].get("sessions") or [])
    return result


def append(result: dict) -> None:
    with open(RESULTS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    (AB / f"result-p{result['pair']}-{result['arm']}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))


def cmd_run(args) -> int:
    ensure_lane_link()
    plan = None
    if args.loop_control == "pause":
        plan = pause_plan(loop_state())
        if not plan["available"]:
            print(json.dumps(plan, indent=2))
            print("pause mode unavailable for this loop; refusing (see fallback)", flush=True)
            return 3
        print(json.dumps(pause_loop(plan, "pause")), flush=True)
    try:
        lane = lane_state()
        if not (lane["at_anchor"] and lane["clean"]):
            reset_lane()
        order = ["baseline", "trimmed"]
        t0 = time.time()
        durations: list[float] = []
        for pair in range(1, args.pairs + 1):
            if args.budget_h and durations:
                need = 2 * statistics.median(durations)
                if (time.time() - t0) + need > args.budget_h * 3600:
                    print(f"budget: pair {pair} would not fit ({need / 60:.0f} min needed); stopping",
                          flush=True)
                    break
            arms = order if (args.order == "abab" or pair % 2) else order[::-1]
            pair_gen = generation(live=True)
            print(f"pair {pair}: {arms} on :{PORT} pid={pair_gen.get('pid')} "
                  f"kv_unified={pair_gen.get('kv_unified_cmdline')} n_ctx={pair_gen.get('props_n_ctx')}",
                  flush=True)
            for arm in arms:
                result = run_call(pair, arm, pair_gen, args)
                append(result)
                if result.get("generation_changed"):
                    print(f"pair {pair} {arm}: :{PORT} changed generation -- stopping", flush=True)
                    return 2
                if result.get("gate_gave_up"):
                    print(f"pair {pair} {arm}: gate never cleared -- stopping", flush=True)
                    return 4
                durations.append(result.get("driver_wall_s") or 0.0)
                m = result.get("metrics") or {}
                f = result.get("fence") or {}
                print(f"  {arm:8s} wall={result.get('driver_wall_s')}s steps={m.get('steps')} "
                      f"tools={m.get('tool_calls')} decoded={m.get('decoded_tokens')} "
                      f"ctx_first={m.get('context_first_tokens')} ctx_max={m.get('context_max_tokens')} "
                      f"lane/anchor-src={f.get('calls_naming_lane')}/{f.get('calls_naming_anchor_source')} "
                      f"builds={len(f.get('build_like_commands') or [])} "
                      f"denies={(result.get('permission') or {}).get('denies')} "
                      f"valid={m.get('schema_valid')} "
                      f"{'transient' if 'transient' in result else 'ok'}", flush=True)
    finally:
        if plan is not None:
            print(json.dumps(pause_loop(plan, "resume")), flush=True)
    return cmd_summarize(args)


# --------------------------------------------------------------------------- summary

def cmd_summarize(_args) -> int:
    if not RESULTS.is_file():
        print("no results yet")
        return 1
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    keys = ("driver_wall_s", "steps", "tool_calls", "decoded_tokens", "prompt_tokens",
            "cache_read_tokens", "compactions", "context_first_tokens", "context_max_tokens")
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("generation_changed") or row.get("gate_gave_up"):
            continue
        by_arm.setdefault(row["arm"], []).append(row)
    report = {}
    for arm, items in sorted(by_arm.items()):
        stats = {}
        for key in keys:
            values = [(r if key == "driver_wall_s" else r.get("metrics") or {}).get(key)
                      for r in items]
            values = [v for v in values if isinstance(v, (int, float))]
            stats[key] = {"median": statistics.median(values) if values else None,
                          "values": values}
        stats["schema_valid"] = sum(1 for r in items if (r.get("metrics") or {}).get("schema_valid"))
        stats["transients"] = sum(1 for r in items if "transient" in r)
        stats["calls"] = len(items)
        stats["contended_calls"] = sum(1 for r in items
                                       if ((r.get("slots") or {}).get("samples_with_other_busy") or 0))
        stats["calls_overlapping_loop_measurement"] = sum(
            1 for r in items if (r.get("loop_during_call") or {}).get("overlapped_measurement"))
        stats["anchor_source_calls"] = [(r.get("fence") or {}).get("calls_naming_anchor_source")
                                        for r in items]
        stats["lane_calls"] = [(r.get("fence") or {}).get("calls_naming_lane") for r in items]
        stats["build_like_commands"] = [len((r.get("fence") or {}).get("build_like_commands") or [])
                                        for r in items]
        stats["permission_denies"] = [(r.get("permission") or {}).get("denies") for r in items]
        report[arm] = stats
    pairs: dict = {}
    for row in rows:
        pairs.setdefault(row["pair"], {})[row["arm"]] = row
    report["pairs_complete"] = sum(1 for p in pairs.values() if set(ARMS) <= set(p))
    gates = [json.loads(line) for line in GATE_LOG.read_text().splitlines()] \
        if GATE_LOG.is_file() else []
    report["gate_waits"] = {"n": len(gates), "waited_s_total": round(sum(g["waited_s"] for g in gates), 1),
                            "waited_calls": sum(1 for g in gates if g["waited_s"] > 1)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- dry run

def cmd_gate_status(_args) -> int:
    state = loop_state()
    print(json.dumps({**state, "pause": pause_plan(state)}, indent=2))
    return 0


def cmd_dry_run(_args) -> int:
    """Build both arms' prompts and seat configs through the real `propose` with
    `_run_agent` captured: no opencode process, no request to any model port, no lane
    reset, no signal; the gate is evaluated once from /proc and NOT waited on."""
    scratch = AB / "dryrun" / "workers" / "lane0"
    scratch.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"worktree: {WORKTREE} @ {head}")
    text = control_text()
    print(f"control bundle: {len(text):,} chars")
    captured = {}
    for arm in ARMS:
        seen = {}

        def capture(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            return fx.HYPOTHESIS

        with mock.patch.object(actors, "render_context", return_value=text), \
                mock.patch.object(actors, "_with_backoff", one_attempt), \
                mock.patch.object(actors, "_run_agent", side_effect=capture):
            planner(scratch, arm).propose(CONTEXT)
        prompt, env = seen["prompt"], seen["env"] or {}
        captured[arm] = prompt
        sha = hashlib.sha256(prompt.encode()).hexdigest()
        print(f"\n[{arm}] knobs={ARMS[arm] or 'all off'}  arm label={env.get(actors.SEAT_ENV_ARM, 'plain (no env)')}")
        print(f"  prompt: {len(prompt):,} chars ~{est_tokens(len(prompt)):,} tokens (est.)  sha256 {sha[:16]}"
              f"  == OAB-9 control: {sha == fx.CONTROL_SHA256}")
        switches = {k: v for k, v in env.items() if k.startswith("OPENCODE_DISABLE_")}
        print(f"  opencode switches: {switches or 'none'}")
        config = env.get("OPENCODE_CONFIG")
        if config:
            body = json.loads(Path(config).read_text())
            perm = body.get("permission", {})
            bash = perm.get("bash", {})
            print(f"  per-call OPENCODE_CONFIG: {config}")
            print(f"    top-level keys: {sorted(body)}; permission keys: {sorted(perm)}")
            print(f"    bash rules: {len(bash)} (actions: {sorted(set(bash.values()))}); "
                  f"edit: {perm.get('edit')}; skill: {perm.get('skill')}")
            print(f"    external_directory: {perm.get('external_directory')}")
            print(f"    instructions: {body.get('instructions', 'none')}")
        if arm == "trimmed":
            print("  prompt head (lane block):")
            for line in prompt.split("\n\n", 1)[0].splitlines():
                print(f"    | {line}")
    base_n, trim_n = len(captured["baseline"]), len(captured["trimmed"])
    print(f"\ntrimmed - baseline prompt: {trim_n - base_n:+,} chars (the lane block); the "
          "fixed-overhead cut is outside the prompt:")
    if STATIC.is_file():
        static = json.loads(STATIC.read_text())
        proj = static["planner_first_step_projection"]
        per = static["per_role_all_knobs_on"]["planner"]
        print(f"  static (27B tokenizer, {STATIC.name}): fixed overhead "
              f"{proj['fixed_overhead_before']:,} -> {proj['fixed_overhead_after']:,} tokens; "
              f"planner first step {proj['oab9_inline_first_step']:,} -> ~{proj['after_all_knobs']:,} "
              f"(removed {per['removed_tokens']:,}, added {per['added_tokens']:,})")
        for key, row in static["components"].items():
            print(f"    {key:18s} {row['tokens']:>6,} tok  {'exact' if row['exact'] else 'approx'}  [{row['knob']}]")
    state = loop_state()
    print(f"\nDS41 loop gate (from /proc, not waited on): running={state['loop_running']} "
          f"owner={state['owner_pids']} run.py={state['run_pids']} busy={state['busy']} "
          f"measuring={state['measuring']}")
    plan = pause_plan(state)
    print(f"--loop-control pause available: {plan['available']} ({plan.get('reason')})")
    for line in plan.get("commands") or plan.get("fallback_for_main_session") or []:
        print(f"    {line}")
    ensure_lane_link()
    print(f"\nlane: {lane_state()}")
    gen = generation(live=False)
    print(f":{PORT} (from /proc only, no request sent): pid={gen.get('pid')} -np={gen.get('np')} "
          f"-c={gen.get('c')} kv_unified(cmdline)={gen.get('kv_unified_cmdline')}")
    print(f"opencode binary: {actors.OPENCODE} exists={Path(actors.OPENCODE).exists()} (not run)")
    print(f"per-call timeout: {TIMEOUT_S}s; results -> {RESULTS}; gate log -> {GATE_LOG}")
    print("schedule (ABBA): p1 baseline,trimmed; p2 trimmed,baseline; ... 3 pairs = 6 calls "
          "~2.0-3.5 h at 20-35 min/call, plus gate waits.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-run")
    sub.add_parser("gate-status")
    run = sub.add_parser("run")
    run.add_argument("--pairs", type=int, default=3)
    run.add_argument("--order", choices=("abab", "abba"), default="abba",
                     help="abba (default): alternate the lead arm; OAB-9 always led with one arm")
    run.add_argument("--budget-h", type=float, default=None)
    run.add_argument("--loop-control", choices=("gate", "pause", "none"), default="gate")
    run.add_argument("--gate-max-wait-h", type=float, default=None,
                     help="give up (and stop) if one gate wait exceeds this")
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    return {"dry-run": cmd_dry_run, "gate-status": cmd_gate_status, "run": cmd_run,
            "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
