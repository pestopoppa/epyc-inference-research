"""Context-mode A/B for the AutoKernel opencode planner seat (DS41 run-8 prompt).

Successor of `/mnt/raid0/llm/tmp/ak-seat-ab/driver.py` (DS41-C20c, plain vs bounded).
Here the SEAT is fixed (plain: bare `opencode run --auto`, the campaign default) and
the CONTEXT PLACEMENT is the treatment:

  inline   : `ActorSeat(bounded=False, context_mode="inline")` -- the whole rendered
             bundle in the prompt. The control prompt is
             `fixtures/ds41-run8-planner-prompt-node-profile.txt`: run 8's recorded
             planner prompt plus the node_profile section `render_context` now prints
             (the recorded prompt dropped it; see test_actor_context.py).
  variable : `ActorSeat(bounded=False, context_mode="variable")` -- the same bundle
             written to `<AB>/actor-context/<call>/`, the prompt carries INDEX.md.

Both arms go through the REAL `AgentPlanner.propose` of the integration branch
(`lane/ak-planner-integ-20260924`), with exactly two patches: `render_context`
returns the control bundle text (so both arms see run 8's context, not a live
campaign's), and `_with_backoff` makes ONE attempt (an A/B records a transient; it
does not retry it from zero). Everything else -- bundle materialisation, the arm
label, the metrics hook (`epyc.autokernel.actor_call_metrics.v1`), the v1 call
record, the schema repair turn -- is the production code path.

Same :8083 server generation for every call of a pair (OAB-4a): before each call the
driver fingerprints the server (pid + /proc start time + cmdline digest), derives
`kv_unified` from the cmdline (last explicit flag wins; else unified iff `-np` is
absent -- the rule of epyc-orchestrator `stack_commands._live_kv_unified`) and reads
it from the launch log when one is attached, and records `/props` n_ctx. A pair whose
calls straddle a server change is marked `generation_changed` and the run stops.

`/slots` is sampled every 30 s DURING each call: busy slots other than this call's
own are contention (another campaign on :8083), recorded, never hidden.

usage:
  python3 driver.py dry-run                 # no opencode, no model port: build + sizes
  python3 driver.py run --pairs 3           # ABAB: inline, variable, inline, variable, ...
  python3 driver.py run --pairs 2           # the 2-pair variant
  python3 driver.py run --pairs 3 --budget-h 3.5   # start a pair only if it fits
  python3 driver.py summarize               # paired table from results.jsonl
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

WORKTREE = Path("/mnt/raid0/llm/worktrees/research-ak-planner-integ-20260924")
sys.path[:0] = [str(WORKTREE / "scripts/kernel_rnd"), str(WORKTREE / "scripts/lib"),
                str(WORKTREE / "scripts"), str(WORKTREE)]
from autokernel.loop import actor_context, actor_metrics, actors  # noqa: E402
from autokernel.loop import test_actor_context as fx  # noqa: E402  (fixture helpers)

AB = Path("/mnt/raid0/llm/tmp/ak-ctx-ab")
#: A symlink to the C20c lane (a detached llama.cpp worktree at the run-8 anchor).
#: The workspace is the LINK path, so the bundle and actor-replies land beside it in
#: AB (`Path(workspace).parent`), while opencode sees the real lane.
LANE = AB / "lane"
ANCHOR = "ebb68dc55d5f6af4a4a5dccdd2a013fa76c63bee"   # run 8's COR anchor
MODEL = "qwen-gpu/qwen3.8-27b"
EFFORT = "high"
PORT = 8083
TIMEOUT_S = 5400          # the C20c driver's cap; the campaign default is 1800
ARMS = {"inline": "inline", "variable": "variable"}
RESULTS = AB / "results.jsonl"
CALL_LOG = AB / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
#: chars/token measured by the ctxvar lane with `llama-tokenize` on the Qwen3.8-27B
#: vocab for this very prompt (75,978 chars / 26,293 tokens inline; 15,864 / 5,510 as
#: an index). An ESTIMATE for sizes printed here -- the live metrics carry real counts.
CHARS_PER_TOKEN = 2.89
KV_ON = ("-kvu", "--kv-unified")
KV_OFF = ("-no-kvu", "--no-kv-unified")
CONTEXT = {"target": {"recipe": {"backend": "cpu"}}}


def control_text() -> str:
    return fx._real_context_text()


def planner(workspace: Path, mode: str) -> actors.AgentPlanner:
    return actors.AgentPlanner(workspace=workspace, backend=actors.backend_for(MODEL, EFFORT),
                               timeout_s=TIMEOUT_S,
                               seat=actors.ActorSeat(bounded=False, context_mode=mode))


def one_attempt(call, **_kw):
    return call(), 0


def est_tokens(chars: int) -> int:
    return round(chars / CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- server

def server_process() -> dict | None:
    """The :8083 llama-server, read from /proc (inspection only; nothing is signalled)."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if not argv or not argv[0].endswith("llama-server"):
            continue
        if "--port" in argv and argv[argv.index("--port") + 1:argv.index("--port") + 2] == [str(PORT)]:
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
            return {"pid": int(proc.name), "start_ticks": start, "argv": argv, "log": log}
    return None


def kv_unified_from_argv(argv: list[str]) -> bool:
    for token in reversed(argv):
        if token in KV_ON:
            return True
        if token in KV_OFF:
            return False
    return not ("-np" in argv or "--parallel" in argv)


def kv_unified_from_log(log: str | None) -> dict | None:
    """The LAST `load_model: initializing ... kv_unified = '...'` line of the launch log
    (the log is appended across server generations; the last launch is this one)."""
    if not log:
        return None
    pattern = re.compile(r"n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(true|false)'")
    last = None
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "kv_unified = '" in line:
                    found = pattern.search(line)
                    if found:
                        last = found
    except OSError:
        return None
    if last is None:
        return None
    return {"kv_unified": last.group(3) == "true", "n_slots": int(last.group(1)),
            "n_ctx_slot": int(last.group(2))}


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
           "kv_unified_log": kv_unified_from_log(proc["log"]), "launch_log": proc["log"],
           "np": (proc["argv"][proc["argv"].index("-np") + 1] if "-np" in proc["argv"] else None),
           "c": (proc["argv"][proc["argv"].index("-c") + 1] if "-c" in proc["argv"] else None)}
    if live:
        try:
            props = http_json("/props")
            dgs = props.get("default_generation_settings") or {}
            out["props_n_ctx"] = dgs.get("n_ctx", props.get("n_ctx"))
            out["props_total_slots"] = props.get("total_slots")
            out["props_build_info"] = props.get("build_info")
            out["props_model_path"] = props.get("model_path")
        except Exception as exc:  # noqa: BLE001 -- recorded, the call still decides
            out["props_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def same_generation(a: dict, b: dict) -> bool:
    keys = ("pid", "start_ticks", "cmdline_sha256")
    return all(a.get(k) == b.get(k) for k in keys) and a.get("found") and b.get("found")


class SlotSampler(threading.Thread):
    """`/slots` every `period` s while a call runs: the busy-slot count, sampled DURING."""

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


def bundle_access(metrics_row: dict) -> dict:
    """Per bundle path, per tool: how often the actor's tool INPUTS named it, from the
    session exports the metrics hook saved (the row carries only the path set)."""
    access: dict[str, dict[str, int]] = {}
    for session in ((metrics_row.get("opencode") or {}).get("sessions") or []):
        path = (session.get("export") or {}).get("path")
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        for message in data.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") != "tool":
                    continue
                tool = part.get("tool") or "?"
                text = json.dumps((part.get("state") or {}).get("input") or {})
                for rel in set(actor_metrics.BUNDLE_PATH.findall(text)):
                    rel = rel.rstrip(".,")
                    access.setdefault(rel, {}).setdefault(tool, 0)
                    access[rel][tool] += 1
    return access


def required_reading(index_text: str) -> list[str]:
    block = index_text.split("Read these before you propose", 1)
    if len(block) < 2:
        return []
    block = block[1].split("| # |", 1)[0]
    return sorted(set(re.findall(r"^- `([^`]+)`", block, re.M)))


def metrics_summary(row: dict | None) -> dict:
    if not row:
        return {"metrics_row": False}
    oc = row.get("opencode") or {}
    totals = oc.get("totals") or {}
    tools: dict[str, int] = {}
    bundle_tools: dict[str, int] = {}
    for session in oc.get("sessions") or []:
        for name, n in (session.get("tools") or {}).items():
            tools[name] = tools.get(name, 0) + n
        for name, n in (session.get("bundle_tools") or {}).items():
            bundle_tools[name] = bundle_tools.get(name, 0) + n
    return {"metrics_row": True, "seat_arm": row.get("seat_arm"),
            "metrics_error": row.get("metrics_error"),
            "sessions": oc.get("session_ids"), "steps": totals.get("steps"),
            "tool_calls": totals.get("tool_calls"), "tools": tools,
            "decoded_tokens": totals.get("decoded_tokens"),
            "prompt_tokens": totals.get("prompt_tokens"),
            "cache_read_tokens": totals.get("cache_read_tokens"),
            "cache_write_tokens": totals.get("cache_write_tokens"),
            "compactions": totals.get("compactions"),
            "context_first_tokens": oc.get("context_first_tokens"),
            "context_max_tokens": oc.get("context_max_tokens"),
            "tool_output_chars": totals.get("tool_output_chars"),
            "wall_s": row.get("wall_s"), "returncode": row.get("returncode"),
            "timed_out": row.get("timed_out"), "salvaged": row.get("salvaged"),
            "schema_valid": row.get("schema_valid"), "repair_ran": row.get("repair_ran"),
            "bundle_tool_calls": totals.get("bundle_tool_calls"),
            "bundle_tools": bundle_tools}


# --------------------------------------------------------------------------- one call

def run_call(pair: int, arm: str, pair_generation: dict) -> dict:
    gen = generation(live=True)
    result: dict = {"pair": pair, "arm": arm, "context_mode": ARMS[arm],
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "server": gen, "timeout_s": TIMEOUT_S}
    if not same_generation(gen, pair_generation):
        result["generation_changed"] = True
        return result
    reset_lane()
    result["lane"] = lane_state()
    offset = CALL_LOG.stat().st_size if CALL_LOG.is_file() else 0
    bundles_before = set((AB / actor_context.BUNDLE_DIR).glob("*"))
    seen: dict = {}
    real_run_agent = actors._run_agent

    def recording_run_agent(prompt, **kw):
        seen["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        seen["prompt_chars"] = len(prompt)
        seen["env_arm"] = (kw.get("env") or {}).get(actors.SEAT_ENV_ARM)
        return real_run_agent(prompt, **kw)

    sampler = SlotSampler()
    sampler.start()
    started = time.time()
    try:
        with mock.patch.object(actors, "render_context", return_value=control_text()), \
                mock.patch.object(actors, "_with_backoff", one_attempt), \
                mock.patch.object(actors, "_run_agent", recording_run_agent):
            outcome = planner(LANE, ARMS[arm]).propose(CONTEXT)
        if isinstance(outcome, actors.Abstain):
            result["abstain"] = outcome.reason
        else:
            result["hypothesis"] = {k: getattr(outcome, k) for k in (
                "mechanism_id", "statement", "falsifier", "target_surface", "target_symbol")}
    except actors.ProviderTransient as exc:
        result["transient"] = str(exc)[:800]
    finally:
        result["driver_wall_s"] = round(time.time() - started, 1)
        result["slots"] = sampler.stop()
    result.update(seen)
    if arm == "inline":
        result["prompt_is_control"] = seen.get("prompt_sha256") == fx.CONTROL_SHA256
    result["lane_edited"] = not lane_state()["clean"]
    rows = new_rows(offset)
    metrics = [r for r in rows if r.get("schema") == actor_metrics.METRICS_SCHEMA]
    result["actor_calls"] = len(metrics)
    result["metrics"] = metrics_summary(metrics[0] if metrics else None)
    if len(metrics) > 1:
        result["metrics_extra_rows"] = [metrics_summary(r) for r in metrics[1:]]
    new_bundles = sorted(set((AB / actor_context.BUNDLE_DIR).glob("*")) - bundles_before)
    if new_bundles:
        bundle = new_bundles[0]
        manifest = json.loads((bundle / "manifest.json").read_text())
        required = required_reading((bundle / "INDEX.md").read_text())
        access = bundle_access(metrics[0]) if metrics else {}
        result["bundle"] = {
            "dir": str(bundle), "prompt_sha256_matches": manifest["prompt"]["sha256"]
            == seen.get("prompt_sha256"),
            "index_chars": manifest["index_chars"],
            "inline_equivalent_prompt_chars": manifest["inline_equivalent_prompt_chars"],
            "access": access,
            "reads_on_bundle": sum(n.get("read", 0) for n in access.values()),
            "greps_on_bundle": sum(n.get("grep", 0) for n in access.values()),
            "index_rereads": sum(access.get("INDEX.md", {}).values()),
            "required": {rel: rel in access for rel in required},
            "required_read_all": all(rel in access for rel in required) if required else None}
    return result


def append(result: dict) -> None:
    with open(RESULTS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    (AB / f"result-p{result['pair']}-{result['arm']}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))


def cmd_run(args) -> int:
    lane = lane_state()
    if not (lane["at_anchor"] and lane["clean"]):
        reset_lane()
    order = ["inline", "variable"]
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
            result = run_call(pair, arm, pair_gen)
            append(result)
            if result.get("generation_changed"):
                print(f"pair {pair} {arm}: :{PORT} changed generation -- stopping", flush=True)
                return 2
            durations.append(result.get("driver_wall_s") or 0.0)
            m = result.get("metrics") or {}
            print(f"  {arm:8s} wall={result.get('driver_wall_s')}s steps={m.get('steps')} "
                  f"tools={m.get('tool_calls')} decoded={m.get('decoded_tokens')} "
                  f"compactions={m.get('compactions')} ctx_max={m.get('context_max_tokens')} "
                  f"valid={m.get('schema_valid')} "
                  f"{'transient' if 'transient' in result else 'ok'}", flush=True)
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
        if row.get("generation_changed"):
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
        if arm == "variable":
            stats["required_read_all"] = [(r.get("bundle") or {}).get("required_read_all")
                                          for r in items]
            stats["reads_on_bundle"] = [(r.get("bundle") or {}).get("reads_on_bundle") for r in items]
            stats["greps_on_bundle"] = [(r.get("bundle") or {}).get("greps_on_bundle") for r in items]
            stats["index_rereads"] = [(r.get("bundle") or {}).get("index_rereads") for r in items]
        report[arm] = stats
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair"], {})[row["arm"]] = row
    report["pairs_complete"] = sum(1 for p in pairs.values() if {"inline", "variable"} <= set(p))
    report["server_generations"] = sorted({json.dumps({k: (r.get("server") or {}).get(k) for k in (
        "pid", "start_ticks", "kv_unified_cmdline", "kv_unified_log", "props_n_ctx")},
        sort_keys=True) for r in rows})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- dry run

def cmd_dry_run(_args) -> int:
    """Build both arms' prompts through the real `propose` with `_run_agent` captured:
    no opencode process, no request to any model port, no lane reset."""
    scratch = AB / "dryrun" / "workers" / "lane0"
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"integration worktree: {WORKTREE} @ "
          + subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip())
    text = control_text()
    recorded = fx._real_context_text(fx._recorded_prompt())
    print(f"control bundle: {len(text):,} chars (run-8 recorded bundle {len(recorded):,} + "
          f"node_profile section {len(text) - len(recorded):,})")
    captured = {}
    for arm, mode in ARMS.items():
        seen = {}

        def capture(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            return fx.HYPOTHESIS

        before = set((scratch.parent / actor_context.BUNDLE_DIR).glob("*"))
        with mock.patch.object(actors, "render_context", return_value=text), \
                mock.patch.object(actors, "_with_backoff", one_attempt), \
                mock.patch.object(actors, "_run_agent", side_effect=capture):
            planner(scratch, mode).propose(CONTEXT)
        new = sorted(set((scratch.parent / actor_context.BUNDLE_DIR).glob("*")) - before)
        prompt = seen["prompt"]
        captured[arm] = prompt
        print(f"\n[{arm}] context_mode={mode}  arm label={(seen['env'] or {}).get(actors.SEAT_ENV_ARM, 'plain (no env)')}")
        print(f"  prompt: {len(prompt):,} chars  ~{est_tokens(len(prompt)):,} tokens (est. "
              f"{CHARS_PER_TOKEN} chars/token)  sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:16]}")
        if arm == "inline":
            ok = hashlib.sha256(prompt.encode()).hexdigest() == fx.CONTROL_SHA256
            print(f"  == control fixture (run-8 prompt + node_profile): {ok}")
        if new:
            bundle = new[0]
            manifest = json.loads((bundle / "manifest.json").read_text())
            files = [p for p in bundle.rglob("*") if p.is_file()]
            print(f"  bundle: {bundle}")
            print(f"  bundle files: {len(files)}  bytes: {sum(p.stat().st_size for p in files):,}  "
                  f"index: {manifest['index_chars']:,} chars  inline-equivalent prompt: "
                  f"{manifest['inline_equivalent_prompt_chars']:,}")
            print(f"  manifest binds prompt: "
                  f"{manifest['prompt']['sha256'] == hashlib.sha256(prompt.encode()).hexdigest()}")
            for s in manifest["sections"]:
                placement = ("INLINE" if s["inline"] else
                             "file + summary" if s["key"] in actor_context.SUMMARY_SECTIONS else "file")
                print(f"    {s['key']:22s} {s['chars']:>7,}  {placement}")
            print("  required reading named in INDEX.md:")
            for rel in required_reading((bundle / "INDEX.md").read_text()):
                print(f"    {rel}")
    inline_n, var_n = len(captured["inline"]), len(captured["variable"])
    print(f"\nvariable / inline prompt: {var_n:,} / {inline_n:,} chars = {var_n / inline_n:.1%} "
          f"(~{est_tokens(inline_n) - est_tokens(var_n):,} fewer prompt tokens, est.)")
    lane = lane_state()
    print(f"\nlane: {lane}")
    gen = generation(live=False)
    print(f":{PORT} (from /proc only, no request sent): pid={gen.get('pid')} -np={gen.get('np')} "
          f"-c={gen.get('c')} kv_unified(cmdline)={gen.get('kv_unified_cmdline')} "
          f"kv_unified(log)={gen.get('kv_unified_log')} log={gen.get('launch_log')}")
    print(f"opencode binary: {actors.OPENCODE} exists={Path(actors.OPENCODE).exists()} (not run)")
    print(f"per-call timeout: {TIMEOUT_S}s; results -> {RESULTS}")
    print("schedule (ABAB): pair k = inline, variable. 3 pairs = 6 calls ~ 2.0-3.5 h at "
          "20-35 min/call; 2 pairs = 4 calls ~ 1.3-2.3 h.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-run")
    run = sub.add_parser("run")
    run.add_argument("--pairs", type=int, default=3)
    run.add_argument("--order", choices=("abab", "abba"), default="abab",
                     help="abab: every pair inline first (default); abba: alternate the lead arm")
    run.add_argument("--budget-h", type=float, default=None,
                     help="start a pair only if 2x the median call so far still fits")
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    return {"dry-run": cmd_dry_run, "run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
