#!/usr/bin/env python3
"""PB-HOL-1 — slot-order head-of-line collision probe for llama-server's pending-prompt loop.

Hypothesis (server-context.cpp ~3205-3232 @ ffc1bac82): the pending-prompt batch is filled in fixed slot-index order
and `if (!add_ok || batch.size() >= n_batch) return;` stops every LATER slot once the batch is full, so a short
prompt in a HIGHER slot than an in-flight long prefill waits for the whole remaining prefill.

Arms (-np 3, id_slot pinned, cache_prompt=false, slots erased between runs):
  A  short slot BELOW long slot   (long -> slot 2, short -> slot 0)
  B  short slot ABOVE long slot   (long -> slot 0, short -> slot 2)
  C  solo short reference on the idle server (start and end)
Order: C, A,B,A,B,A,B, C.  Predeclared rule: B_median >= 2 x A_median short TTFT => CONFIRMED, else NOT_REPRODUCED.
Budget-on arms (pb512, second server launch, -b/-ub 2048): A-on, B-on with a feeder decoding in slot 1, and B-on-nodec.

Reuses kpf18_gate.py (Server start/stop-by-PID/verify-dead, KFD VRAM sampler, linkage check, mi210_0 device claim).
Never signals anything but its own server child. --dry-run starts nothing, sends nothing, takes no claim.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# kpf18_gate.py (Server start/stop-by-PID, VRAM sampler, linkage check, device claim) is not yet in a repo; point
# PBHOL_KPF_DIR at the directory holding it. The default is the scratch copy the 2026-10-21 runs used.
KPF_DIR = Path(os.environ.get("PBHOL_KPF_DIR", "/mnt/raid0/llm/tmp/kpf-p1-20261004/kpf18"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(KPF_DIR))
import kpf18_gate as K  # noqa: E402

SCHEMA = "epyc.pbhol1.run.v1"
CHAMPION = Path("/mnt/raid0/llm/kernels/builds/gpu-20261020-dd3d9721c/bin/llama-server")
ARGV_FILE = Path("/mnt/raid0/llm/tmp/overnight-20261020/argv.8083-gate.txt")
PROMPT_SET = Path("/mnt/raid0/llm/tmp/overnight-20261020/kpf18-run/prompt_set.jsonl")
LONG_GROUP, SHORT_GROUP, SHORT_SUFFIX = 6, 3, 0   # 72k trunk of group 6 truncated; 512-token suffix of group 3 truncated
LONG_TOKENS, SHORT_TOKENS, COLLIDE_FRAC = 64000, 128, 0.5
N_PRED_LONG, N_PRED_SHORT = 4, 8
NP, PORT = 3, 18090
ARMS = {"A": (0, 2), "B": (2, 0)}          # arm -> (short slot, long slot)
FEED_SLOT = 1                               # third slot, decoding (budget-on arms only)
ORDER_OFF = ["C", "A", "B", "A", "B", "A", "B", "C"]
# budget-on phase: (arm label, placement arm, feeder?) ; one list = one repeat (interleaved A,B,nodec)
ORDER_ON_REP = [("A-on", "A", True), ("B-on", "B", True), ("B-on-nodec", "B", False)]
BUDGET_REPS = 2
# pb512 = KVU-16g's "pb512": --prefill-budget-decoding 512 on the production shape -b 2048 -ub 2048 (KVU-16g compared it
# against -b 512; at -b 512 a 512 budget would be a no-op).  CONFOUND recorded: the on-server also changes -b/-ub.
BUDGET_ARGS = ["--prefill-budget-decoding", "512"]
BUDGET_B, BUDGET_UB = "2048", "2048"
PROG_RE = re.compile(r"id\s+(\d+) \| task (\d+) \| prompt processing, n_tokens =\s*(\d+), progress = ([0-9.]+)")
LOGTS_RE = re.compile(r"^(\d+\.\d+) ")
RULE = "B_median >= 2 * A_median (short TTFT, ms) => CONFIRMED slot-order head-of-line; else NOT_REPRODUCED"
DEFERRED = []   # nothing deferred: the champion build carries --prefill-budget-decoding / --prefill-budget-target-ms (default off)
# estimate model (seconds)
LOAD_S, LINK_S, STOP_S = 90.0, 5.0, 20.0


def parse_deadline(s: str) -> float:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return float(calendar.timegm(time.strptime(s, fmt)))
        except ValueError:
            continue
    if re.fullmatch(r"\d{1,2}:\d\d", s):  # HH:MM today (UTC), rolled to tomorrow if already past
        h, m = map(int, s.split(":"))
        t = time.gmtime()
        d = float(calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, h, m, 0)))
        return d if d > time.time() else d + 86400
    raise SystemExit(f"bad --deadline {s!r}: use 2026-10-21T16:40:00Z or HH:MM (UTC)")


def load_prompts(path: Path) -> tuple[list[int], list[int], dict]:
    long_ids = short_ids = None
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        if r.get("kind") == "trunk" and r["group"] == LONG_GROUP:
            long_ids = r["ids"][:LONG_TOKENS]
        elif r.get("kind") == "pair" and r["group"] == SHORT_GROUP and r["suffix_index"] == SHORT_SUFFIX:
            short_ids = r["suffix_ids"][:SHORT_TOKENS]
    if not long_ids or len(long_ids) < LONG_TOKENS or not short_ids or len(short_ids) < SHORT_TOKENS:
        raise SystemExit(f"prompt set {path} lacks group {LONG_GROUP} trunk >= {LONG_TOKENS} or group {SHORT_GROUP} suffix")
    meta = {"path": str(path), "sha256": K.sha256_file(path), "long_group": LONG_GROUP, "long_tokens": len(long_ids),
            "short_group": SHORT_GROUP, "short_suffix_index": SHORT_SUFFIX, "short_tokens": len(short_ids),
            "first_tokens_distinct": long_ids[0] != short_ids[0]}
    return long_ids, short_ids, meta


def est_long_s() -> float:
    return K.pf_s(LONG_TOKENS, 0) + 3.0


def plan_estimate(budget_reps: int = BUDGET_REPS, phases: tuple[str, ...] = ("off", "on")) -> tuple[float, list[tuple[str, float]]]:
    L = est_long_s()
    A = 0.5 * L + 15       # long aborted once the short has answered
    B = L + 10
    steps = []
    if "off" in phases:
        steps += [("phase 1 (budget off, -b/-ub 512): load + linkage + residency", LOAD_S + LINK_S + 10),
                  ("solo long calibration (64K prefill)", L + 5)]
        for i, a in enumerate(ORDER_OFF):
            steps.append((f"off run {i + 1}: " + ("C solo short" if a == "C" else f"arm {a}-off collision"),
                          3 if a == "C" else (A if a == "A" else B)))
        steps.append(("phase 1 stop by PID + verify dead", STOP_S))
    if "on" not in phases:
        return sum(s for _, s in steps), steps
    steps.append(("phase 2 (budget on pb512, -b/-ub 2048): load + linkage + residency", LOAD_S + LINK_S + 10))
    if "off" not in phases:
        steps.append(("solo long calibration in phase 2 (64K prefill)", L + 5))
    for r in range(budget_reps):
        for lab, arm, fd in ORDER_ON_REP:
            steps.append((f"on rep {r + 1}: {lab}" + (" (+feeder)" if fd else ""),
                          (A if arm == "A" else B + 10) + (12 if fd else 0)))
    steps.append(("phase 2 stop by PID + verify dead + summary", STOP_S))
    return sum(s for _, s in steps), steps


def port_owner_ok(pid: int, port: int) -> tuple[bool, str]:
    """True iff `pid` holds the LISTEN socket for `port` (inode from /proc/net/tcp{,6} matched against /proc/pid/fd)."""
    inodes = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for ln in Path(f).read_text().splitlines()[1:]:
                p = ln.split()
                if len(p) > 9 and p[3] == "0A" and int(p[1].rsplit(":", 1)[1], 16) == port:
                    inodes.add(p[9])
        except (OSError, ValueError):
            pass
    if not inodes:
        return False, f"nothing listening on {port}"
    try:
        mine = {os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")}
    except OSError as exc:
        return False, f"cannot read /proc/{pid}/fd: {exc}"
    hit = [i for i in inodes if f"socket:[{i}]" in mine]
    return (bool(hit), f"listen inodes {sorted(inodes)}; pid {pid} owns {hit}")


class LogTail:
    """Reads the server log (timestamped by the runner's pump) from a byte offset."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def lines_from(self, off: int) -> list[str]:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(off)
                return fh.read().decode(errors="replace").splitlines()
        except OSError:
            return []

    def progress(self, off: int, slot: int) -> tuple[int, float, float] | None:
        """latest (n_tokens, progress_frac, log_epoch) of `slot`'s prompt processing after offset."""
        best = None
        for ln in self.lines_from(off):
            m = PROG_RE.search(ln)
            if m and int(m.group(1)) == slot:
                t = LOGTS_RE.match(ln)
                best = (int(m.group(3)), float(m.group(4)), float(t.group(1)) if t else 0.0)
        return best


class Probe:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out)
        self.rows = K.Jsonl(self.out / "pb_hol_1.jsonl")
        self.srv: K.Server | None = None      # current phase server
        self.servers: list[K.Server] = []     # every server launched (for summary / stop)
        self.client: K.Client | None = None
        self.log: LogTail | None = None
        self.phase = None
        self.cal_s: float | None = None
        self.deadline_hit = threading.Event()
        self.stop_lock = threading.Lock()
        self.long_ids, self.short_ids, self.ps_meta = load_prompts(Path(args.prompt_set))
        self.decoder_ids = load_decoder(Path(args.prompt_set))

    # ---- lifecycle
    def stop_server(self) -> None:
        with self.stop_lock:
            for srv in self.servers:
                if getattr(srv, "_pbhol_stopped", False):
                    continue
                srv._pbhol_stopped = True
                try:
                    srv.stop()
                except Exception as exc:  # noqa: BLE001
                    print(f"[pbhol] stop failed: {exc}", flush=True)
                    self.rows.add({"kind": "stop_error", "label": srv.label, "err": str(exc)})
                if srv.stop_rec:
                    self.rows.add({"kind": "server_stop", "label": srv.label, **srv.stop_rec})

    def watchdog(self) -> None:
        while time.time() < self.args.deadline_epoch:
            if self.deadline_hit.wait(5):
                return
        print("[pbhol] DEADLINE reached: stopping server by PID", flush=True)
        self.deadline_hit.set()
        self.stop_server()

    def check(self) -> None:
        if self.deadline_hit.is_set() or time.time() >= self.args.deadline_epoch:
            raise K.Abort("deadline reached")
        if self.srv and self.srv.sampler and self.srv.sampler.abort.is_set():
            raise K.Abort(f"VRAM: {self.srv.sampler.abort_reason}")
        if self.srv and self.srv.proc and self.srv.proc.poll() is not None:
            raise K.Abort(f"server died rc={self.srv.proc.returncode}")
        w = K.window_status()
        if w.get("preempt_requested_at"):
            raise K.Abort(f"GPU window preempt requested at {w['preempt_requested_at']}")

    def start_phase(self, name: str, budget: bool) -> None:
        self.check()
        ctx, srv = build_server(self.args, self.out, name, budget)
        self.servers.append(srv)
        self.srv, self.phase = srv, name
        srv.start()           # port-free check, no foreign KFD, linkage, residency read-back (kpf18 Server)
        self.client, self.log = ctx.client, LogTail(srv.log)
        ok, why = port_owner_ok(srv.proc.pid, self.args.port)
        self.rows.add({"kind": "port_ownership", "phase": name, "ok": ok, "detail": why, "pid": srv.proc.pid,
                       "argv": srv.argv})
        if not ok:
            raise K.Abort(f"port {self.args.port} not owned by our pid {srv.proc.pid}: {why}")

    def end_phase(self) -> None:
        self.stop_server()
        self.client = None
        self.srv = None   # 2026-10-21 fix: check() at the next start_phase read the stopped server as 'died'

    # ---- helpers
    def wait_idle(self, tmo: float = 30.0) -> None:
        t0 = time.time()
        while time.time() - t0 < tmo:
            if not any(x.get("is_processing") for x in self.client.slots()):
                return
            time.sleep(0.5)

    def erase_all(self) -> None:
        self.wait_idle()
        for s in range(NP):
            self.client.erase(s)
        busy = [x.get("id") for x in self.client.slots() if x.get("is_processing")]
        if busy:
            raise K.Abort(f"slots still busy after erase: {busy}")

    def short_req(self, slot: int) -> dict:
        body = K.body_for(self.short_ids, N_PRED_SHORT, slot, fork=None, cache=False, n_probs=0)
        return self.client.stream(body, timeout=900)

    def run_solo(self, tag: str, rep: int, slot: int = 0) -> dict:
        self.check()
        self.erase_all()
        r = self.short_req(slot)
        rec = {"kind": "run", "phase": self.phase, "budget": False, "arm": "C", "rep": rep, "tag": tag, "short_slot": slot,
               "long_slot": None, "short_ttft_ms": r.get("ttft_ms"), "short_t_send": r["t_send"],
               "short_http": r.get("http"), "short_timings": r.get("timings"), "short_error": r.get("error")}
        self.rows.add(rec)
        print(f"[pbhol] C{rep}: short TTFT {rec['short_ttft_ms']} ms", flush=True)
        return rec

    def calibrate(self) -> None:
        self.check()
        self.erase_all()
        body = K.body_for(self.long_ids, N_PRED_LONG, 0, fork=None, cache=False, n_probs=0)
        r = self.client.stream(body, timeout=1800)
        self.cal_s = (r.get("ttft_ms") or 0) / 1000.0   # prefill duration = TTFT of the long
        self.rows.add({"kind": "calibration", "phase": self.phase, "long_ttft_s": self.cal_s, "http": r.get("http"),
                       "error": r.get("error"), "timings": r.get("timings")})
        if not self.cal_s:
            raise K.Abort(f"calibration failed: {r.get('error')}")
        print(f"[pbhol] calibration: solo 64K prefill {self.cal_s:.1f} s", flush=True)

    def run_collision(self, label: str, arm: str, rep: int, feeder: bool, budget: bool) -> dict:
        short_slot, long_slot = ARMS[arm]
        self.check()
        need = (self.cal_s or est_long_s()) * 1.1 + 40
        if self.args.deadline_epoch - time.time() < need:
            rec = {"kind": "run", "phase": self.phase, "arm": label, "rep": rep,
                   "skipped": f"insufficient time before deadline (< {need:.0f} s left)"}
            self.rows.add(rec)
            print(f"[pbhol] {label}{rep}: SKIPPED ({rec['skipped']})", flush=True)
            return rec
        self.erase_all()
        off = self.log.size()
        stop_long, stop_feed = threading.Event(), threading.Event()
        long_res: dict = {}
        feed_res: dict = {}
        feed_tokens = [0]
        th_feed = None
        if feeder:   # a third slot decoding for the whole collision (arms the decode-gated budget)
            fb = K.body_for(self.decoder_ids, 8192, FEED_SLOT, fork=None, cache=False, n_probs=0)

            def _feed():
                feed_res.update(self.client.stream(fb, stop=stop_feed, on_token=lambda n: feed_tokens.__setitem__(0, n),
                                                   timeout=1800))
            th_feed = threading.Thread(target=_feed, daemon=True)
            th_feed.start()
            t0 = time.time()
            while feed_tokens[0] < 8 and time.time() - t0 < 90 and th_feed.is_alive():
                time.sleep(0.1)
            if feed_tokens[0] < 8:
                stop_feed.set()
                raise K.Abort(f"feeder never started decoding (tokens={feed_tokens[0]}, {feed_res.get('error')})")
        long_body = K.body_for(self.long_ids, N_PRED_LONG, long_slot, fork=None, cache=False, n_probs=0)

        def _long():
            long_res.update(self.client.stream(long_body, stop=stop_long, timeout=1800))

        th = threading.Thread(target=_long, daemon=True)
        t_long = time.time()
        th.start()
        cal = self.cal_s or est_long_s()
        trigger = None
        while True:
            self.check()
            prog = self.log.progress(off, long_slot)
            if prog and prog[1] >= COLLIDE_FRAC:
                trigger = "log_progress"
                break
            if time.time() - t_long > cal * COLLIDE_FRAC * 1.3 + 15:
                trigger = "calibration_time_fallback"
                break
            if not th.is_alive():
                trigger = "long_finished_early"
                break
            time.sleep(0.05)
        try:
            slots_snap = [{k: s.get(k) for k in ("id", "is_processing", "n_ctx", "n_prompt_tokens",
                                                 "n_prompt_tokens_processed", "next_token") if k in s}
                          for s in self.client.slots()]
        except Exception as exc:  # noqa: BLE001
            slots_snap = {"err": str(exc)[:200]}
        deferred_max, stop_poll = [0.0], threading.Event()

        def _poll():
            while not stop_poll.is_set():
                try:
                    deferred_max[0] = max(deferred_max[0], self.client.metrics().get("llamacpp:requests_deferred", 0.0))
                except Exception:  # noqa: BLE001
                    pass
                stop_poll.wait(0.5)

        threading.Thread(target=_poll, daemon=True).start()
        prog_at_send = self.log.progress(off, long_slot)
        feed_at_send = feed_tokens[0]
        short = self.short_req(short_slot)
        t_first = short["t_send"] + (short.get("ttft_ms") or 0) / 1000.0
        prog_at_first = self.log.progress(off, long_slot)
        stop_poll.set()
        long_done_before_abort = not th.is_alive()
        stop_long.set()      # the short has answered: the long prefill is no longer needed
        stop_feed.set()
        th.join(timeout=60)
        if th_feed:
            th_feed.join(timeout=60)
        win = [ln for ln in self.log.lines_from(off)
               if (m := LOGTS_RE.match(ln)) and short["t_send"] - 3 <= float(m.group(1)) <= t_first + 3]
        budget_iters = [ln for ln in win if "prefill-budget" in ln]
        ivals, prev = [], None
        for ln in self.log.lines_from(off):
            m, t = PROG_RE.search(ln), LOGTS_RE.match(ln)
            if m and t and int(m.group(1)) == long_slot:
                cur = (int(m.group(3)), float(t.group(1)))
                if prev and cur[0] > prev[0] and short["t_send"] - 10 <= cur[1] <= short["t_send"]:
                    ivals.append((cur[1] - prev[1]) / ((cur[0] - prev[0]) / 512.0))
                prev = cur
        rec = {"kind": "run", "phase": self.phase, "budget": budget, "arm": label, "placement": arm, "rep": rep,
               "feeder": feeder, "short_slot": short_slot, "long_slot": long_slot, "feeder_slot": FEED_SLOT if feeder else None,
               "feeder_tokens_at_send": feed_at_send, "feeder_tokens_final": feed_tokens[0],
               "trigger": trigger, "short_t_send": short["t_send"], "short_t_first_token": t_first,
               "short_ttft_ms": short.get("ttft_ms"), "short_http": short.get("http"), "short_error": short.get("error"),
               "short_timings": short.get("timings"), "long_t_send": t_long,
               "long_ttft_ms": long_res.get("ttft_ms"), "long_finished_before_abort": long_done_before_abort,
               "long_error": long_res.get("error"),
               "long_progress_at_send": prog_at_send, "long_progress_at_short_first_token": prog_at_first,
               "requests_deferred_max": deferred_max[0], "slots_at_collision": slots_snap,
               "iter_interval_s_per_512": round(statistics.median(ivals), 4) if ivals else None,
               "budget_iter_lines_in_window": len(budget_iters),
               "server_log_window": win[:120], "server_log_window_truncated": len(win) > 120}
        self.rows.add(rec)
        print(f"[pbhol] {label}{rep}: short slot {short_slot} long slot {long_slot} trigger {trigger} "
              f"long@{prog_at_send[1] if prog_at_send else None} -> short TTFT {rec['short_ttft_ms']} ms", flush=True)
        return rec

    # ---- summary
    def summarize(self, status: str) -> dict:
        f = self.out / "pb_hol_1.jsonl"
        rows = [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []
        runs = [r for r in rows if r.get("kind") == "run" and r.get("short_ttft_ms") is not None and not r.get("skipped")]

        def med(arm):
            v = [r["short_ttft_ms"] for r in runs if r["arm"] == arm]
            return (statistics.median(v) if v else None), v
        a, av = med("A-off")
        b, bv = med("B-off")
        c, cv = med("C")
        if a is not None and b is not None and len(av) >= 2 and len(bv) >= 2:
            verdict = "CONFIRMED" if b >= 2 * a else "NOT_REPRODUCED"
        else:
            verdict = "INCOMPLETE"
        on = {}
        for lab in ("A-on", "B-on", "B-on-nodec"):
            m, v = med(lab)
            on[lab] = {"median_ms": m, "ttft_ms": v}
        sec = []
        for r in runs:
            if r["arm"] in ("A-off", "B-off", "A-on", "B-on", "B-on-nodec") and c is not None:
                dq = 0.0 if not r.get("requests_deferred_max") else None  # >0 deferred: wait not separable here
                resid = r["short_ttft_ms"] - c - (dq or 0.0)
                itv = (r.get("iter_interval_s_per_512") or 0) * 1000
                sec.append({"arm": r["arm"], "rep": r["rep"], "deque_wait_ms_assumed": dq, "residual_ms": round(resid, 1),
                            "one_iteration_ms": round(itv, 1) or None,
                            "residual_over_iteration": round(resid / itv, 2) if itv else None,
                            "long_progress_at_send": (r.get("long_progress_at_send") or [None, None])[1]})
        s = {"schema": SCHEMA, "status": status, "verdict": verdict, "rule": RULE,
             "A_off_ttft_ms": av, "B_off_ttft_ms": bv, "C_ttft_ms": cv, "A_off_median": a, "B_off_median": b, "C_median": c,
             "B_over_A_off": round(b / a, 2) if a and b else None, "budget_on_arms": on,
             "budget_args": BUDGET_ARGS, "budget_phase_shape": f"-b {BUDGET_B} -ub {BUDGET_UB}",
             "confound": "budget-on phase also changes -b/-ub (512 -> 2048) so that pb512 can bite; an on-vs-off difference "
                         "is budget+batch-shape together (B-on-nodec, with no decoder, isolates the batch-shape effect).",
             "secondary_residual": sec, "calibration_long_prefill_s": self.cal_s, "deferred_arms": DEFERRED,
             "prompts": self.ps_meta, "binary": str(self.args.champion_bin),
             "binary_sha256": K.sha256_file(Path(self.args.champion_bin)),
             "servers": [{"label": x.label, "readback": x.readback, "linkage": x.linkage,
                          "vram": x.sampler.summary() if x.sampler else None, "stop": x.stop_rec} for x in self.servers],
             "skipped_runs": [r for r in rows if r.get("skipped")], "finished": K.now_iso()}
        K.write_json(self.out / "summary.json", s)
        md = [f"# PB-HOL-1 summary ({s['finished']})", "", f"status: {status}  verdict (off arms): **{verdict}**",
              f"rule: {RULE}", "", "| arm | short TTFT ms (reps) | median |", "|---|---|---|",
              f"| A-off short below long | {av} | {a} |", f"| B-off short above long | {bv} | {b} |",
              f"| C solo | {cv} | {c} |"]
        md += [f"| {k} | {v['ttft_ms']} | {v['median_ms']} |" for k, v in on.items()]
        md += ["", f"B/A (off) = {s['B_over_A_off']}", "", f"Confound: {s['confound']}", "",
               "Secondary (TTFT - solo TTFT - deque wait) vs one iteration:"]
        md += [f"- {x['arm']}{x['rep']}: residual {x['residual_ms']} ms, iteration {x['one_iteration_ms']} ms, "
               f"ratio {x['residual_over_iteration']}, long@{x['long_progress_at_send']}" for x in sec]
        if s["skipped_runs"]:
            md += ["", "SKIPPED runs: " + "; ".join(f"{r['arm']}{r['rep']} ({r['skipped']})" for r in s["skipped_runs"])]
        (self.out / "summary.md").write_text("\n".join(md) + "\n")
        return s


def load_decoder(path: Path) -> list[int]:
    for ln in path.read_text().splitlines():
        if '"kind":"decoder"' in ln:
            return json.loads(ln)["ids"]
    raise SystemExit(f"no decoder prompt in {path}")


def build_server(args, out: Path, name: str, budget: bool) -> tuple[K.Ctx, K.Server]:
    argv, _ = K.parse_argv_file(Path(args.argv_file))
    argv = K.set_flag(argv, ("-np", "--parallel"), str(NP))
    extra: list[str] = []
    if budget:
        argv = K.set_flag(argv, ("-b", "--batch-size"), BUDGET_B)
        argv = K.set_flag(argv, ("-ub", "--ubatch-size"), BUDGET_UB)
        extra = list(BUDGET_ARGS)
    ctx = K.Ctx(args, None, argv, args.launch_prefix.split(), {"source": str(args.argv_file)})
    ctx.out = out
    return ctx, K.Server(ctx, f"pbhol-{name}-{K.utc_tag()}", Path(args.champion_bin), None, extra_args=extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline", required=True, help="hard stop, e.g. 2026-10-21T16:40:00Z or HH:MM (UTC)")
    ap.add_argument("--champion-bin", default=str(CHAMPION))
    ap.add_argument("--argv-file", default=str(ARGV_FILE))
    ap.add_argument("--prompt-set", default=str(PROMPT_SET))
    ap.add_argument("--launch-prefix", default=K.DEFAULT_PREFIX)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-window-id", default=os.environ.get("PBHOL_GPU_WINDOW_ID"))
    ap.add_argument("--vram-ceiling-gib", type=float, default=62.5)
    ap.add_argument("--vram-free-max-gib", type=float, default=2.0)
    ap.add_argument("--load-timeout-s", type=float, default=600)
    ap.add_argument("--card", default=str(K.DEFAULT_CARD))
    ap.add_argument("--no-device-claim", default=None, metavar="REASON")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--phases", default="off,on", help="comma list of off,on (on-only: calibrates in the on phase)")
    args = ap.parse_args()
    args.deadline_epoch = parse_deadline(args.deadline)
    args.out = args.out or str(HERE / "runs" / K.utc_tag())

    # ---- preflight (read-only)
    probs, notes = [], []
    cb = Path(args.champion_bin)
    if not (cb.is_file() and os.access(cb, os.X_OK)):
        probs.append(f"champion binary {cb} not an executable file")
    elif os.path.realpath(cb).startswith(str(K.FROZEN_TREE) + "/"):
        probs.append("champion binary resolves inside the frozen tree")
    else:
        common = cb.parent / "libllama-common.so"
        have_b = K.bytes_contain(common, b"--prefill-budget-decoding")
        notes.append(f"prefill-budget knob compiled into {common.name}: {have_b}")
        if not have_b:
            probs.append("champion build lacks --prefill-budget-decoding: the budget-on phase cannot run")
    if not (1024 <= args.port <= 65535) or 8000 <= args.port <= 8999:
        probs.append(f"--port {args.port} refused (8000-8999 are production ports)")
    if K.port_listening(args.port):
        probs.append(f"port {args.port} already listening")
    for f in (K.LINKAGE, Path(args.argv_file), Path(args.prompt_set)):
        if not f.exists():
            probs.append(f"missing {f}")
    long_ids = short_ids = None
    ps_meta = {}
    if Path(args.prompt_set).exists():
        try:
            long_ids, short_ids, ps_meta = load_prompts(Path(args.prompt_set))
        except SystemExit as exc:
            probs.append(str(exc))
    hard, drift = K.argv_problems(K.parse_argv_file(Path(args.argv_file))[0], True) if Path(args.argv_file).exists() else ([], [])
    probs += hard
    drift = [d for d in drift if not d.startswith("-np")]
    if drift:
        notes.append("argv shape drift vs gate: " + "; ".join(drift))
    if K.port_listening(8083):
        notes.append("production :8083 is LISTENING now (an execute run refuses until its owner stops it)")
    total, steps = plan_estimate(BUDGET_REPS, tuple(x.strip() for x in args.phases.split(",") if x.strip()))
    now = time.time()
    left = args.deadline_epoch - now
    if left < total * 1.15:
        probs.append(f"only {K.fmt_s(left)} until deadline; plan needs ~{K.fmt_s(total)} (x1.15 margin)")

    print("PB-HOL-1 plan")
    print(f"  deadline {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(args.deadline_epoch))} (in {K.fmt_s(left)}), out {args.out}")
    print(f"  binary {cb}\n  argv   {args.argv_file} with -np {NP}; prefix: {args.launch_prefix}; port {args.port}")
    print(f"  prompts {ps_meta}")
    print(f"  long = {LONG_TOKENS} tok (group {LONG_GROUP} trunk), short = {SHORT_TOKENS} tok (group {SHORT_GROUP} suffix {SHORT_SUFFIX}); "
          f"short sent when long prefill >= {COLLIDE_FRAC:.0%} (server-log progress, calibration-time fallback)")
    print(f"  phase 1 (budget off, argv -b 512 -ub 512), order {ORDER_OFF}; A = short slot 0 / long slot 2, "
          f"B = short slot 2 / long slot 0; cache_prompt=false, slots erased between runs; long request aborted once the short has answered")
    print(f"  phase 2 (budget on, second launch: -b {BUDGET_B} -ub {BUDGET_UB} {' '.join(BUDGET_ARGS)} = pb512), "
          f"{BUDGET_REPS} reps each, interleaved {[x[0] for x in ORDER_ON_REP]}; feeder = decoding request in slot {FEED_SLOT} "
          f"(8-token start check, aborted at run end); '-nodec' = budget on, no slot decoding")
    print("  CONFOUND: phase 2 also changes -b/-ub (a 512 budget is a no-op at -b 512); off/on cannot be interleaved because "
          "the budget is a launch flag (each switch = server restart); B-on-nodec isolates the batch-shape effect")
    print(f"  rule: {RULE}")
    print("  deferred arms: none")
    for d, sec in steps:
        print(f"   - {d}: ~{K.fmt_s(sec)}")
    print(f"  ESTIMATE total ~{K.fmt_s(total)} ({total / 60:.1f} min) against a {K.fmt_s(left)} budget; "
          f"long prefill model {est_long_s():.0f} s (replaced by the measured calibration at run time)")
    for n in notes:
        print("  note:", n)
    for p in probs:
        print("  PROBLEM:", p)
    if args.dry_run:
        print("PREFLIGHT:", "FAIL" if probs else "PASS")
        print("DRY RUN complete: no process was started, nothing was sent, no claim taken.")
        return 0
    if probs:
        print("REFUSED (preflight).")
        return 2
    if not args.gpu_window_id:
        print("REFUSED: no --gpu-window-id / PBHOL_GPU_WINDOW_ID")
        return 64

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = Probe(args)
    stack, rc = [], 0

    def on_sig(signum, _f):
        probe.deadline_hit.set()
        probe.stop_server()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    threading.Thread(target=probe.watchdog, daemon=True).start()
    try:
        if args.no_device_claim is None:
            sys.path.insert(0, str(K.CLAIM_MOD_DIR))
            from autokernel.resource import device_claim as dc  # noqa: E402
            journal = dc.ClaimJournal(out / "claim" / "device-claim-journal.jsonl")
            try:
                cm = dc.acquire_device_claim("mi210_0", purpose=f"PB-HOL-1 collision probe on :{args.port}",
                                             campaign_id="PB-HOL-1", journal=journal, holder_label="pbhol1", timeout_s=0,
                                             max_hold_s=max(600, left + 600))
            except dc.DeviceClaimError as exc:
                print(f"[pbhol] REFUSED: mi210_0 device claim not acquired: {exc}", flush=True)
                return 75
            claim = cm.__enter__()
            stack.append(cm)
            K.write_json(out / "claim" / "device-claim-receipt.json", claim.receipt().to_dict())
        K.write_json(out / "run_meta.json", {"args": vars(args), "started": K.now_iso(), "estimate_s": total, "notes": notes,
                                              "gpu_window_id": args.gpu_window_id, "prompts": ps_meta,
                                              "device_claim": args.no_device_claim or "mi210_0 held"})
        phases = [x.strip() for x in args.phases.split(",") if x.strip()]
        if "off" in phases:
            probe.start_phase("off", False)
            probe.calibrate()
            reps = {"A": 0, "B": 0, "C": 0}
            for tag in ORDER_OFF:
                reps[tag] += 1
                if tag == "C":
                    probe.run_solo(f"C{reps[tag]}", reps[tag])
                else:
                    probe.run_collision(f"{tag}-off", tag, reps[tag], False, False)
            probe.end_phase()
        if "on" in phases:
            probe.start_phase("on", True)
            if "off" not in phases:
                probe.calibrate()
            for r in range(1, BUDGET_REPS + 1):
                for lab, arm, fd in ORDER_ON_REP:
                    probe.run_collision(lab, arm, r, fd, True)
        status = "complete"
    except K.Abort as exc:
        print(f"[pbhol] ABORT: {exc}", flush=True)
        status, rc = f"abort: {exc}", 3
    except SystemExit:
        status, rc = "signalled", 5
    except Exception as exc:  # noqa: BLE001
        print(f"[pbhol] ERROR: {type(exc).__name__}: {exc}", flush=True)
        status, rc = f"error: {type(exc).__name__}: {exc}", 4
    finally:
        probe.deadline_hit.set()
        probe.stop_server()
        try:
            s = probe.summarize(locals().get("status", "aborted-before-status"))
            print(f"[pbhol] verdict {s['verdict']} A={s['A_off_median']} B={s['B_off_median']} C={s['C_median']} ms; {out}/summary.md")
        except Exception as exc:  # noqa: BLE001
            print(f"[pbhol] summary failed: {exc}", flush=True)
        for cm in reversed(stack):
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                print(f"[pbhol] claim release failed: {exc}", flush=True)
        dead = all((x.stop_rec.get("verified_dead") if x.stop_rec else x.proc is None) for x in probe.servers)
        print(f"[pbhol] server verified dead: {dead}")
        if not dead:
            rc = rc or 6
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
