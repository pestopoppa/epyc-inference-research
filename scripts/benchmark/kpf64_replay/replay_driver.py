#!/usr/bin/env python3
"""KPF-64 replay driver: replays adapter sessions through an OWN llama-server /completion.

Refuses (exit 3) unless WINDOW_CONFIRMED=1 -- checked FIRST, before any network or heavy import.
Refuses production/dashboard ports (8000-8199) and a port that already has a listener.
One in-flight request per session; sessions advance in order; `--active` sessions run at once
(default = slot count). Results are appended per request (requests.jsonl) and the run is resumable.
--dry-run drives an in-process stub /completion server (NOT inference) on an ephemeral port.
"""
from __future__ import annotations

import os
import sys

if os.environ.get("WINDOW_CONFIRMED") != "1":  # first thing: no httpx/network/imports before this
    sys.stderr.write("replay_driver: refusing to run: WINDOW_CONFIRMED=1 is not set "
                     "(a KPF-64 timing window must be claimed first)\n")
    sys.exit(3)

import argparse  # noqa: E402
import http.server  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapter  # noqa: E402

REFUSED_PORTS = range(8000, 8200)  # production 8070-8099, orchestrator 8000, hub 8100
DEFAULT_PORT = 18964
#: name -> (regex, counts_as_eviction). Strings are the llama.cpp server log lines (see
#: tools/server/server-context.cpp and server-task.cpp); override with --patterns FILE.
DEFAULT_PATTERNS = {
    "ckpt_erase_old": (r"erasing old context checkpoint", True),
    "ckpt_erase_too_close": (r"erasing context checkpoint too close", True),
    "ckpt_erase_invalidated": (r"erased invalidated context checkpoint", True),
    "cache_remove_oldest": (r"removing oldest entry", True),
    "cache_size_limit_reached": (r"cache size limit reached", True),
    "cache_token_limit_reached": (r"cache token limit .* reached", True),
    "cache_skip_too_big": (r"exceeds cache size limit", True),
    "cache_remove_obsolete": (r"removing obsolete cached prompt", False),
    "full_reprocess_forced": (r"forcing full prompt re-processing", True),
    "ckpt_restored": (r"restored context checkpoint", False),
    "idle_slot_saved": (r"saving idle slot to prompt cache", False),
    "cache_load_failed": (r"failed to load prompt from cache", True),
}


def own_pid_serves(pid, port):
    """True iff PID is alive and its argv names --port <port> (the listener is OUR server)."""
    try:
        argv = open(f"/proc/{int(pid)}/cmdline", "rb").read().split(b"\0")
    except (OSError, ValueError):
        return False
    return b"--port" in argv and str(port).encode() in argv[argv.index(b"--port") + 1:argv.index(b"--port") + 2]


def check_port(port, allow_listening=False, own_pid=None):
    if port in REFUSED_PORTS:
        raise SystemExit(f"replay_driver: refusing port {port}: 8000-8199 is production/dashboard range")
    if own_pid is not None:
        if not own_pid_serves(own_pid, port):
            raise SystemExit(f"replay_driver: refusing: pid {own_pid} is not alive or does not serve --port {port}")
        return
    if not allow_listening:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"replay_driver: refusing: port {port} already has a listener (not our server)")


def http_json(url, payload=None, timeout=3600):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def count_log_events(log_path, patterns=None):
    pats = patterns or DEFAULT_PATTERNS
    comp = {k: (re.compile(v[0]), bool(v[1])) for k, v in pats.items()}
    counts = {k: 0 for k in comp}
    if log_path and Path(log_path).exists():
        for line in open(log_path, errors="replace"):
            for k, (rx, _) in comp.items():
                if rx.search(line):
                    counts[k] += 1
    ev = sum(c for k, c in counts.items() if comp[k][1])
    return {"by_pattern": counts, "observed_eviction_events": ev,
            "log_present": bool(log_path and Path(log_path).exists())}


def load_done(path):
    done = {}
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done[(r["session"], r["k"])] = r
    return done


def load_outputs(path):
    outs = {}
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            outs[(r["session"], r["k"])] = r["out"]
    return outs


class Replay:
    def __init__(self, a, base):
        self.a, self.base = a, base
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.req_path, self.outs_path = self.out / "requests.jsonl", self.out / "outputs.jsonl"
        self.lock = threading.Lock()
        self.done = load_done(self.req_path)
        self.outs = load_outputs(self.outs_path)
        self.stop = threading.Event()

    def append(self, path, rec):
        with self.lock, open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def run_session(self, sid):
        sess = json.loads((Path(self.a.sessions) / f"{sid}.json").read_text())
        turns, live = sess["turns"], self.a.history == "live"
        outputs = []
        for k in range(len(turns)):
            outputs.append(self.outs.get((sid, k)) if live else None)
        n = 0
        for k, prompt, prev in adapter.iter_prompts(turns, outputs if live else None):
            if self.stop.is_set():
                return n
            if (sid, k) in self.done:
                continue
            ideal_c = adapter.common_prefix_len(prompt, prev) if prev is not None else 0
            t0 = time.time()
            body = {"prompt": prompt, "n_predict": self.a.n_predict if live else 1, "cache_prompt": True,
                    "temperature": 0.0, "stream": False, "return_tokens": False}
            try:
                r = http_json(f"{self.base}/completion", body, timeout=self.a.timeout)
            except Exception as e:  # recorded, session stops; analysis excludes the session from all arms
                self.append(self.req_path, {"session": sid, "k": k, "error": f"{type(e).__name__}: {e}"[:200]})
                return n
            t1 = time.time()
            tm = r.get("timings") or {}
            pn, cn = int(tm.get("prompt_n", 0)), int(tm.get("cache_n", 0))
            total = pn + cn
            ideal_tok = round(total * ideal_c / len(prompt)) if prompt else 0
            rec = {"session": sid, "store": sess.get("store"), "k": k, "reset": bool(turns[k].get("reset")),
                   "history": self.a.history, "prompt_chars": len(prompt), "prompt_n": pn, "cache_n": cn,
                   "total_tok": total, "ideal_chars": ideal_c, "ideal_tok": ideal_tok,
                   "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
                   "id_slot": r.get("id_slot", tm.get("id_slot")), "t_start": t0, "t_end": t1,
                   "inferred_evict": bool(k > 0 and not turns[k].get("reset") and ideal_tok > 0
                                          and cn < ideal_tok - self.a.infer_tol_tokens)}
            if live:
                out = r.get("content", "")
                outputs[k] = out
                self.append(self.outs_path, {"session": sid, "k": k, "out": out})
            self.append(self.req_path, rec)
            n += 1
        return n

    def snapshots(self):
        path = self.out / "snapshots.jsonl"
        while not self.stop.wait(self.a.snap_interval):
            rec = {"t": time.time()}
            for ep in ("slots", "metrics"):
                try:
                    rec[ep] = http_json(f"{self.base}/slots", timeout=10) if ep == "slots" else http_text(f"{self.base}/metrics")
                except Exception as e:  # snapshots are best effort; absence is recorded, not hidden
                    rec[ep] = f"ERR {type(e).__name__}"
            self.append(path, rec)


# ---- in-process stub (dry run only; NOT inference) ----------------------------------------------
class _Stub(http.server.BaseHTTPRequestHandler):
    slots, lock, nslots, logf = None, threading.Lock(), 4, None
    used, tick = None, 0

    def log_message(self, *a):
        pass

    def _send(self, obj, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/slots"):
            with self.lock:
                self._send([{"id": i, "n_cached": len(s) // 4} for i, s in enumerate(self.slots)])
        elif self.path.startswith("/metrics"):
            self._send(b"llamacpp:prompt_tokens_total 0\n", "text/plain")
        else:
            self._send({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = body["prompt"]
        with self.lock:
            _Stub.tick += 1
            lcps = [adapter.common_prefix_len(p, self.slots[i]) for i in range(self.nslots)]
            lcps = [c if c >= 0.1 * len(p) else 0 for c in lcps]  # slot-similarity threshold, as the real server
            best = max(range(self.nslots), key=lambda i: (lcps[i], not self.slots[i], -self.used[i]))
            lcp = lcps[best]
            self.used[best] = _Stub.tick
            if lcp == 0 and self.slots[best] and self.logf:
                self.logf.write("slot  update_slots: id  0 | task 1 | erasing old context checkpoint (stub)\n")
                self.logf.flush()
            total = max(1, len(p) // 4)
            cache = min(total - 1, lcp // 4)
            out = "ok " * max(1, min(int(body.get("n_predict", 1)), 8))
            self.slots[best] = p + out
        self._send({"content": out, "id_slot": best,
                    "timings": {"prompt_n": total - cache, "cache_n": cache, "predicted_n": 1,
                                "prompt_ms": (total - cache) * 0.05}})


def start_stub(logpath):
    _Stub.slots = [""] * _Stub.nslots
    _Stub.used = [0] * _Stub.nslots
    _Stub.logf = open(logpath, "a")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", required=True, help="adapter output dir (manifest.json + session files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="unnamed")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--active", type=int, default=0, help="concurrent sessions (default = slots)")
    ap.add_argument("--history", choices=("reference", "live"), default="live")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--snap-interval", type=float, default=15.0)
    ap.add_argument("--infer-tol-tokens", type=int, default=32)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--patterns", default=None, help="JSON {name: [regex, counts_as_eviction]}")
    ap.add_argument("--only", default=None, help="comma list of session ids")
    ap.add_argument("--own-server-pid", type=int, default=None,
                    help="PID of the server the launcher started; the listener is then expected")
    ap.add_argument("--dry-run", action="store_true", help="drive an in-process stub (not inference)")
    a = ap.parse_args(argv)
    a.active = a.active or a.slots
    srv = None
    if a.dry_run:
        check_port(0, allow_listening=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        a.server_log = a.server_log or str(Path(a.out) / "stub_server.log")
        _Stub.nslots = a.slots
        srv, _th = start_stub(a.server_log)
        a.port = srv.server_address[1]
    else:
        check_port(a.port, own_pid=a.own_server_pid)
    base = f"http://127.0.0.1:{a.port}"
    man = json.loads((Path(a.sessions) / "manifest.json").read_text())
    ids = [s["id"] for s in man["sessions"] if not a.only or s["id"] in a.only.split(",")]
    rp = Replay(a, base)
    snap = threading.Thread(target=rp.snapshots, daemon=True)
    snap.start()
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=a.active) as ex:
            list(ex.map(rp.run_session, ids))
    except BaseException:
        rp.stop.set()
        raise
    finally:
        rp.stop.set()
        snap.join(timeout=30)
        if srv:
            srv.shutdown()
            srv.server_close()
    pats = json.loads(Path(a.patterns).read_text()) if a.patterns else None
    events = count_log_events(a.server_log, pats)
    (Path(a.out) / "run_meta.json").write_text(json.dumps({
        "arm": a.arm, "history": a.history, "slots": a.slots, "active": a.active, "sessions": ids,
        "wall_s": time.time() - t0, "events": events, "infer_tol_tokens": a.infer_tol_tokens,
        "dry_run": a.dry_run}, indent=1))
    print(json.dumps({"arm": a.arm, "requests": len(load_done(rp.req_path)), "events": events["observed_eviction_events"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
