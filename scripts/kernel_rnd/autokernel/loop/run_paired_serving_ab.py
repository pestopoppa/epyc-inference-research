#!/usr/bin/env python3
"""AKR-SG shakedown driver (queue row ``s5-akrsg-shakedown``): paired A/B serving gate on a REAL endpoint pair.

  WINDOW_CONFIRMED=1 python -m scripts.kernel_rnd.autokernel.loop.run_paired_serving_ab \\
      --a-build /mnt/raid0/llm/kernels/production/cpu --b-build /mnt/raid0/llm/kernels/production/cpu \\
      --model /mnt/raid0/llm/models/<small>.gguf --device cpu --cpu-list-a 0-7 --cpu-list-b 96-103 \\
      --port-a 9511 --port-b 9512 --pairs 8 --aa-pairs 8 --np 2 --out-dir RUN_DIR

A ``--device gpu`` run needs ``--cpu-list-a/-b`` too (the arms' host threads, e.g. 184-191; refused if
unset): every arm is pinned and affinity-enforced by ``ab_server_pair`` (GPS-PIN-2).

Shakedown semantics: A and B may be the SAME build (an A/A shakedown: the record must come out valid with an
effect inside its own floor, ``decisive`` false). What is being shaken down is the RECORD PIPELINE on a live
endpoint: launch, port-ownership and linkage proof, closed-round load, self-check, pairing, floor, /metrics
series, teardown. Output: ``serving_ab.json`` (``epyc.autokernel.serving_ab.v1`` + ``generator.metrics``),
``proof.json``, ``teardown.json``, ``manifest.json``.

Guards: refuses unless WINDOW_CONFIRMED=1 (live); ``--toy`` runs two in-process fake endpoints (no model, no
window) for tests. A ``STOP`` file in --out-dir is honoured between pairs. Never starts anything on a
production port. The server pair is always torn down (try/finally), verified dead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import metrics_map as mm
from . import paired_loadgen as pl
from .ab_server_pair import ArmSpec, ServerPair


class StopRequested(RuntimeError):
    pass


def metered_arm(inner: pl.Arm, url: str, stop_file: Path, sink: list) -> pl.Arm:
    """Wrap an Arm: /metrics scrape before and after each measurement; STOP honoured at the item boundary."""
    def measure(seed: int):
        if stop_file.exists():
            raise StopRequested(f"{stop_file} present")
        before = mm.scrape(url)
        value, check = inner.measure(seed)
        after = mm.scrape(url)
        sink.append({"arm": inner.name, "seed": seed, "metrics": mm.window_report(before, after)})
        return value, check
    return pl.Arm(inner.name, measure, np=inner.np)


def toy_servers(rate_a: float, rate_b: float):
    def mk(rate):
        state = {"pred": 0.0, "secs": 0.0, "n": 0}

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = (f"# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total {state['pred']}\n"
                        f"llamacpp:tokens_predicted_seconds_total {state['secs']}\nllamacpp:n_decode_total {state['n']}\n"
                        "llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n").encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                state["pred"] += 16; state["secs"] += 16 / rate; state["n"] += 16
                out = json.dumps({"timings": {"predicted_per_second": rate}}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

            def log_message(self, *a): pass
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"
    return mk(rate_a), mk(rate_b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--toy-rates", default="100,100")
    ap.add_argument("--a-build"); ap.add_argument("--b-build"); ap.add_argument("--model")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--cpu-list-a"); ap.add_argument("--cpu-list-b")
    ap.add_argument("--port-a", type=int); ap.add_argument("--port-b", type=int)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--aa-pairs", type=int, default=8)
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=20261021)
    ap.add_argument("--belief-store", help="store root for serving-beliefs/ export (default: OUT_DIR/store)")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stop = out / "STOP"
    sink: list = []
    manifest = {"schema": "akr_sg_shakedown/v1", "toy": a.toy, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": {k: v for k, v in vars(a).items()}}
    pair = None
    toys = None
    try:
        if a.toy:
            ra, rb = (float(x) for x in a.toy_rates.split(","))
            toys = toy_servers(ra, rb)
            urls = {"A": toys[0][1], "B": toys[1][1]}
            proof = {"toy": True}
        else:
            if os.environ.get("WINDOW_CONFIRMED") != "1":
                sys.exit("refusing: live mode runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")
            need = [a.a_build, a.b_build, a.model, a.port_a, a.port_b]
            if not all(need):
                sys.exit("live mode needs --a-build --b-build --model --port-a --port-b")
            specs = []
            for name, build, port, cl in (("A", a.a_build, a.port_a, a.cpu_list_a), ("B", a.b_build, a.port_b, a.cpu_list_b)):
                specs.append(ArmSpec(name, Path(build), port, Path(a.model), a.device, cl, a.ctx, a.np, a.threads))
            pair = ServerPair(specs[0], specs[1], out / "server_logs")
            proof = pair.start(a.ready_timeout)
            urls = {"A": pair.url("A"), "B": pair.url("B")}
        (out / "proof.json").write_text(json.dumps(proof, indent=1))
        arm = {k: metered_arm(pl.http_arm(k, urls[k], np=a.np, n_predict=a.n_predict), urls[k], stop, sink) for k in "AB"}
        # Belief write side, at record time. Unit is `session` (rounds against one running server, not
        # independent launches). A toy endpoint or an A/A (same build) shakedown shakes down the PIPELINE
        # only and is marked so; it is still exported so the record grades like any other observation.
        pipeline_only = bool(a.toy or a.a_build == a.b_build)
        rec = pl.paired_record(arm["A"], arm["B"], pairs=a.pairs, base_seed=a.base_seed, aa_pairs=a.aa_pairs,
                               name="akr-sg-shakedown",
                               belief={"model": a.model or "toy://in-process-endpoint",
                                       "anchor_build": a.a_build or "toy://A",
                                       "candidate_build": a.b_build or "toy://B",
                                       "pipeline_only": pipeline_only,
                                       "pipeline_only_reason": ("toy endpoint" if a.toy else "A/A same-build shakedown")
                                       if pipeline_only else ""})
        rec["generator"]["metrics"] = {"mapping": "metrics_map.py (llamacpp:* names of the v10 tree)", "kv": mm.KV_NOT_EXPOSED,
                                       "windows": sink}
        pl.reseal_belief(rec)   # the capture digests the native body, which now includes generator.metrics
        problems = pl.validate_record(rec)
        (out / "serving_ab.json").write_text(json.dumps(rec, indent=1))
        if not problems:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            exported = pl.export_belief(a.belief_store or out / "store", rec, campaign_id="akr-sg-shakedown",
                                        epoch="akr-sg-" + now[:10], recorded_at=now,
                                        mechanism_id="paired-loadgen-shakedown")
            manifest["belief_receipt"] = str(exported)
        manifest.update(verdict="OK" if not problems else "INVALID_RECORD", problems=problems, effect_pct=rec["effect_pct"],
                        noise_floor_pct=rec["noise_floor_pct"], decisive=rec["decisive"], metric_windows=len(sink))
        rc = 0 if not problems else 1
    except StopRequested as e:
        manifest["verdict"] = f"STOPPED: {e}"; rc = 4
    finally:
        td = pair.stop() if pair else {}
        if toys:
            for s, _ in toys:
                s.shutdown()
        (out / "teardown.json").write_text(json.dumps(td, indent=1))
        manifest["teardown_all_dead"] = all(v["dead"] for v in td.values()) if td else None
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: manifest.get(k) for k in ("verdict", "effect_pct", "noise_floor_pct", "decisive", "teardown_all_dead")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
