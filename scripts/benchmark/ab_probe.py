#!/usr/bin/env python3
"""GPU champion A/B probes and summarizer (workspace-8d, 2026-09-29).

Called by ab_arms.sh once per arm, against ONE llama-server on a spare port.

  probe vision --port P --out F --image IMG     Qwen3-VL arm: image+text greedy (n_probs=0) identity
                                                probe, then text-only pp / tg speed through the server.
  probe mtp    --port P --out F                 27B + draft-mtp arm: greedy identity probes (n_probs=0),
                                                SW-9 n_probs>0 probes (raw greedy and post-sampling),
                                                then pp / tg speed through the server.
  summarize --run DIR --kind vision|mtp         Fold every arm-*.json (+ residency and llama-bench
                                                files) into DIR/result.json with verdicts.
  args --pid PID | --argv-file F  ...           Derive the arm server argv from production's argv.

Live-arm checks (called by ab_arms.sh; each prints ONE JSON object with what it SAW, exit 0 = ok,
1 = check failed, 2 = could not look). Each is testable read-only against a production server:
  affinity --pid PID --cpuset CS                Thread placement, classified per thread (see below).
  devlog --pid PID --log F                      GPU-device evidence: startup-log lines (v10 logs almost
                                                nothing at verbosity 3) + the pid's KFD VRAM.
  props --port P --commit C [--model M] [--vision]
                                                /props build_info / model_path / modalities.
  window-watch --baseline F --cpuset CS         Paused-loop window watcher (see ab_arms.sh --watch-mode).

Every request is sequential (one slot busy at a time), cache_prompt=false, fixed seed. The speed
numbers are the SERVER's own timings (prompt_per_second / predicted_per_second), which include MTP
speculation for the 27B -- the serving-relevant figure. llama-bench numbers, when present, come from
separate files written by ab_arms.sh.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HTTP_TIMEOUT_S = 600


def post(port: int, path: str, body: dict, timeout: float = HTTP_TIMEOUT_S) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:  # report what the server SAID, not just the status
        body_txt = exc.read().decode(errors="replace")[:400]
        raise RuntimeError(f"POST {path} -> HTTP {exc.code}: {body_txt}") from exc
    try:
        out = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"POST {path} -> non-JSON reply: {raw[:400]!r}") from exc
    return out, time.monotonic() - t0


def get(port: int, path: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pp_prompt(n_items: int) -> str:
    # Deterministic, tokenizer-agnostic filler; prompt_n in the timings records the real length.
    return " ".join(
        f"Record {i}: sensor {i % 17} reported value {(i * 7919) % 1000} at step {i}." for i in range(n_items)
    )


TG_PROMPT = "Write a detailed technical explanation of how a B-tree index works in a relational database, " \
            "covering node splits, merges, and range scans."

SW9_VISION_NOTE = ("NOT TESTED: the vision model (Qwen3-VL-30B-A3B on :8086) has NO MTP drafter, so SW-9 "
                   "(n_probs>0 under MTP speculation) cannot be exercised; this A/B tests identity, speed and "
                   "residency only. The SW-9 GPU check needs an MTP model: a :8083 window or the 2nd MI210.")

IDENTITY_PROMPTS = [
    "The history of the printing press begins with",
    "def merge_sort(xs):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
]


def timing_fields(t: dict) -> dict:
    keys = (
        "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
        "predicted_per_second", "draft_n", "draft_n_accepted",
    )
    return {k: t.get(k) for k in keys if k in t}


def completion(port: int, prompt: str, **kw) -> dict:
    body = {"prompt": prompt, "cache_prompt": False, "seed": 42, "stream": False}
    body.update(kw)
    out, wall = post(port, "/completion", body)
    return {"content": out.get("content", ""), "timings": out.get("timings", {}), "wall_s": wall,
            "stop_type": out.get("stop_type"), "probs": out.get("completion_probabilities"),
            "tokens_predicted": out.get("tokens_predicted")}


def speed_block(port: int, pp_items: int, pp_reps: int, tg_n: int, tg_reps: int) -> dict:
    pp, tg = [], []
    prompt = pp_prompt(pp_items)
    for _ in range(pp_reps):
        r = completion(port, prompt, n_predict=1, temperature=0.0, top_k=1)
        pp.append(timing_fields(r["timings"]))
    for _ in range(tg_reps):
        r = completion(port, TG_PROMPT, n_predict=tg_n, temperature=0.0, top_k=1, ignore_eos=True)
        tg.append(timing_fields(r["timings"]) | {"content_sha256": sha(r["content"])})
    ppv = [x.get("prompt_per_second") for x in pp]
    tgv = [x.get("predicted_per_second") for x in tg]
    if pp and not all(isinstance(v, (int, float)) for v in ppv):
        raise ValueError(f"server timings lack prompt_per_second: {pp}")
    if tg and not all(isinstance(v, (int, float)) for v in tgv):
        raise ValueError(f"server timings lack predicted_per_second: {tg}")
    return {
        "pp": pp, "tg": tg,
        "pp_tps_median": statistics.median(ppv) if pp else None,
        "tg_tps_median": statistics.median(tgv) if tg else None,
    }


def probs_audit(probs: list | None, post_sampling: bool) -> dict:
    """Count tokens whose probability list is missing/empty (the SW-9 gap)."""
    probs = probs or []
    key = "top_probs" if post_sampling else "top_logprobs"
    empty = sum(1 for tok in probs if not tok.get(key))
    return {"n_tokens": len(probs), "n_empty": empty, "key": key}


def probe_vision(args, res: dict) -> dict:
    port = args.port
    res["_step"] = "read image"
    img_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()
    res.update({"image": args.image, "image_sha256": hashlib.sha256(Path(args.image).read_bytes()).hexdigest()})
    res["_step"] = "warmup"
    completion(port, "Hello", n_predict=4, temperature=0.0, top_k=1)  # warmup
    res["_step"] = "identity_image"
    ident = []
    for rep in range(args.identity_reps):
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this chart precisely: title, axis labels, legend entries, "
                                         "and every number you can read."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]}],
            "temperature": 0.0, "top_k": 1, "seed": 42, "max_tokens": args.identity_tokens,
            "cache_prompt": False, "stream": False,
        }
        out, wall = post(port, "/v1/chat/completions", body)
        try:
            content = out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"chat reply has no choices[0].message: {json.dumps(out)[:400]}") from exc
        if not content:
            raise ValueError(f"chat reply content is EMPTY (vacuous identity): {json.dumps(out)[:400]}")
        ident.append({"rep": rep, "content_sha256": sha(content), "content": content, "wall_s": wall,
                      "timings": timing_fields(out.get("timings", {}))})
    res["identity_image"] = ident
    res["_step"] = "identity_text"
    text_ident = []
    for prompt in IDENTITY_PROMPTS:
        r = completion(port, prompt, n_predict=args.identity_tokens, temperature=0.0, top_k=1, n_probs=0)
        if not r["content"]:
            raise ValueError(f"identity completion returned EMPTY content (stop_type={r['stop_type']})")
        text_ident.append({"prompt_sha256": sha(prompt), "content_sha256": sha(r["content"]),
                           "content": r["content"], "timings": timing_fields(r["timings"])})
    res["identity_text"] = text_ident
    res["_step"] = "speed"
    res["speed"] = speed_block(port, args.pp_items, args.pp_reps, args.tg_n, args.tg_reps)
    res["sw9"] = {"tested": False, "reason": SW9_VISION_NOTE}
    res.pop("_step", None)
    return res


def probe_mtp(args, res: dict) -> dict:
    port = args.port
    res["_step"] = "warmup"
    completion(port, "Hello", n_predict=4, temperature=0.0, top_k=1)  # warmup
    res["_step"] = "identity_text"
    ident = []
    for prompt in IDENTITY_PROMPTS:
        r = completion(port, prompt, n_predict=args.identity_tokens, temperature=0.0, top_k=1, n_probs=0)
        if not r["content"]:
            raise ValueError(f"identity completion returned EMPTY content (stop_type={r['stop_type']})")
        ident.append({"prompt_sha256": sha(prompt), "content_sha256": sha(r["content"]),
                      "content": r["content"], "timings": timing_fields(r["timings"])})
    res["identity_text"] = ident
    res["_step"] = "sw9 n_probs"
    # SW-9: raw (greedy, top_logprobs) and post-sampling (temp 1, top_probs) branches.
    raw = completion(port, "I believe the meaning of life is", n_predict=args.probs_tokens,
                     temperature=0.0, top_k=1, n_probs=5)
    post_s = completion(port, "Today was the day. Today I would finally become a", n_predict=args.probs_tokens,
                        temperature=1.0, n_probs=5, post_sampling_probs=True)
    for key, r, post_flag in (("nprobs_raw", raw, False), ("nprobs_post", post_s, True)):
        audit = probs_audit(r["probs"], post_flag)
        predicted = r["tokens_predicted"] or (r["timings"] or {}).get("predicted_n") or 0
        audit["tokens_predicted"] = predicted
        audit["n_missing"] = max(0, predicted - audit["n_tokens"])   # tokens with no probs entry at all
        audit["n_without_probs"] = audit["n_empty"] + audit["n_missing"]
        audit["timings"] = timing_fields(r["timings"])
        res[key] = audit
    res["_step"] = "speed"
    res["speed"] = speed_block(port, args.pp_items, args.pp_reps, args.tg_n, args.tg_reps)
    res.pop("_step", None)
    return res


# ── argv derivation ─────────────────────────────────────────────────────────
def derive_args(args) -> int:
    """Production argv -> arm argv: drop argv[0], replace --port / --slot-save-path values."""
    if args.pid:
        raw = Path(f"/proc/{args.pid}/cmdline").read_bytes().split(b"\0")
        argv = [a.decode() for a in raw if a]
    else:
        argv = [a for a in Path(args.argv_file).read_text().split("\0") if a]
    argv = argv[1:]  # drop the binary; each arm supplies its own
    out, i, seen_port = [], 0, False
    while i < len(argv):
        a = argv[i]
        if a == "--port":
            out += ["--port", str(args.port)]; i += 2; seen_port = True; continue
        if a == "--slot-save-path":
            out += ["--slot-save-path", args.slot_dir]; i += 2; continue
        out.append(a); i += 1
    if not seen_port:
        print("ERROR: production argv has no --port", file=sys.stderr)
        return 1
    sys.stdout.write("\0".join(out) + "\0")
    return 0


# ── live-arm checks (each prints one JSON object with what it SAW) ────────────
def parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def fmt_cpus(cpus) -> str:
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


# Threads created by the ROCm/HSA runtime. Ground truth (2026-09-29, read-only, all four GPU processes
# on the host: production :8083 and :8086 llama-server, whisper-server, tts-server): EVERY thread is
# named after the executable (comm "llama-server" -- ggml/OMP/httplib/ROCr never rename theirs), and
# exactly ONE thread sits outside the launch cpuset: the ROCr async-event thread, blocked in the kernel
# in kfd_wait_on_events with its affinity RESET by the runtime (0-191 for llama/whisper, 0-31 for tts).
# So a name-only split cannot work on this build; the runtime thread is recognized by its kernel wait
# channel (sampled repeatedly -- a running thread shows wchan 0), with a comm-name rule kept for
# runtimes that do name their threads.
RUNTIME_COMM_RE = re.compile(r"^(hsa|rocr|roc[a-z]*[-_:]|hip|amd|kfd|comgr|hsakmt)", re.I)
RUNTIME_WCHAN_RE = re.compile(r"^(kfd_|amdgpu_|amdkfd)")


def thread_affinity(pid: int, cpuset: str, samples: int = 10, interval_s: float = 0.1) -> dict:
    want = parse_cpus(cpuset)
    base = f"/proc/{pid}/task"
    if not os.path.isdir(base):
        return {"ok": False, "pid": pid, "error": f"{base} does not exist (process gone)"}
    try:
        argv = [a.decode(errors="replace") for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
    except OSError:
        argv = []
    n_threads_arg = None
    for i, a in enumerate(argv[:-1]):
        if a in ("-t", "--threads"):
            try:
                n_threads_arg = int(argv[i + 1])
            except ValueError:
                pass
    threads = {}
    for tdir in sorted(glob.glob(f"{base}/*"), key=lambda x: int(x.rsplit("/", 1)[1])):
        tid = int(tdir.rsplit("/", 1)[1])
        status = _read(f"{tdir}/status")
        if status is None:
            continue  # thread exited while we looked
        cl = next((ln.split(":", 1)[1].strip() for ln in status.splitlines()
                   if ln.startswith("Cpus_allowed_list:")), "")
        threads[tid] = {"tid": tid, "comm": _read(f"{tdir}/comm") or "?", "cpus": parse_cpus(cl),
                        "wchan": set()}
    outside = [t for t in threads.values() if not t["cpus"] <= want]
    for _ in range(max(1, samples)):  # wchan is 0 while a thread runs: sample, keep every value seen
        for t in outside:
            w = _read(f"{base}/{t['tid']}/wchan")
            t["wchan"].add(w if w is not None else "<unreadable>")
        time.sleep(interval_s)

    def row(t):
        return {"tid": t["tid"], "comm": t["comm"], "cpus": fmt_cpus(t["cpus"]),
                "wchan_seen": sorted(t["wchan"])}

    runtime, unexplained = [], []
    for t in outside:
        why = None
        if RUNTIME_COMM_RE.search(t["comm"]):
            why = f"comm {t['comm']!r} is a ROCm/HSA runtime name"
        elif any(RUNTIME_WCHAN_RE.search(w or "") for w in t["wchan"]):
            why = "blocked in " + ",".join(sorted(w for w in t["wchan"] if RUNTIME_WCHAN_RE.search(w or ""))) \
                  + " (ROCr async-event thread)"
        (runtime if why else unexplained).append(row(t) | ({"why": why} if why else {}))
    main = threads.get(pid)
    pinned = sorted((t for t in threads.values() if len(t["cpus"]) == 1), key=lambda t: min(t["cpus"]))
    pinned_inside = [t for t in pinned if t["cpus"] <= want]
    groups: dict[str, int] = {}
    for t in threads.values():
        k = fmt_cpus(t["cpus"])
        groups[k] = groups.get(k, 0) + 1
    problems = []
    if main is None:
        problems.append("main thread not found")
    elif not main["cpus"] <= want:
        problems.append(f"MAIN thread {pid} ({main['comm']}) allowed on {fmt_cpus(main['cpus'])}, outside {cpuset}")
    for u in unexplained:
        if u["tid"] != pid:
            problems.append(f"thread {u['tid']} ({u['comm']}) allowed on {u['cpus']}, outside {cpuset}, "
                            f"wchan {u['wchan_seen']} -- not a recognized ROCm runtime thread")
    res = {
        "ok": not problems, "pid": pid, "cpuset": cpuset, "n_threads": len(threads),
        "affinity_groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
        "main_thread": None if main is None else {"tid": pid, "comm": main["comm"], "cpus": fmt_cpus(main["cpus"])},
        "threads_arg": n_threads_arg,
        # OMP/ggml compute threads are pinned one per CPU once the pool exists. The pool is created at
        # the FIRST compute (production :8083: 60 s after start, at its first request), so right after
        # /health it may not exist yet -- which is why ab_arms.sh re-runs this check after the probe.
        "pinned_single_cpu_threads": [{"tid": t["tid"], "cpu": min(t["cpus"])} for t in pinned],
        "pinned_inside_cpuset": len(pinned_inside),
        "compute_pool_seen": n_threads_arg is not None and len(pinned_inside) >= n_threads_arg,
        "runtime_threads_outside": runtime,
        "unexplained_threads_outside": unexplained,
        "problems": problems,
    }
    return res


def cmd_affinity(args) -> int:
    res = thread_affinity(args.pid, args.cpuset, samples=args.samples)
    if "error" in res:
        print(json.dumps(res, sort_keys=True))
        return 2
    rt = "; ".join(f"{r['tid']} {r['comm']} cpus={r['cpus']} ({r['why']})" for r in res["runtime_threads_outside"])
    res["summary"] = (f"{'OK' if res['ok'] else 'FAIL'}: {res['n_threads']} threads, main on "
                      f"{(res['main_thread'] or {}).get('cpus')}, {res['pinned_inside_cpuset']} pinned inside "
                      f"{args.cpuset} (-t {res['threads_arg']}), runtime outside: [{rt or 'none'}]"
                      + (f"; PROBLEMS: {' | '.join(res['problems'])}" if res["problems"] else ""))
    print(json.dumps(res, sort_keys=True))
    return 0 if res["ok"] else 1


DEVLOG_RE = re.compile(
    r"using device ROCm\d|offloaded \d+/\d+ layers to GPU|ROCm\d+ (model|KV|compute) buffer size|"
    r"device 'ROCm\d+'|found \d+ ROCm devices|ggml_cuda_init|ROCm\d+ \(AMD Instinct")


def kfd_vram(pid: int) -> tuple[bool, int]:
    d = f"/sys/class/kfd/kfd/proc/{pid}"
    tot = 0
    for f in glob.glob(f"{d}/vram_*"):
        v = _read(f)
        if v and v.isdigit():
            tot += int(v)
    return os.path.isdir(d), tot


def cmd_devlog(args) -> int:
    lines = []
    try:
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if DEVLOG_RE.search(ln):
                    lines.append(ln.rstrip("\n"))
    except OSError as exc:
        lines = [f"<cannot read {args.log}: {exc}>"]
    in_kfd, vram = kfd_vram(args.pid) if args.pid else (False, 0)
    argv = []
    if args.pid:
        try:
            argv = [a.decode() for a in Path(f"/proc/{args.pid}/cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            pass
    dev_arg = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("--device", "-dev")]
    ngl_arg = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("-ngl", "--n-gpu-layers", "--gpu-layers")]
    # Verdict = the pid holds real VRAM (KFD, the kernel's own accounting). The startup log is recorded
    # but NOT required: the v10 server at verbosity 3 prints no "using device"/"offloaded" lines at all
    # -- its only ROCm mention is a sampler-support warning (runs/w8083-20260929T070737Z/arm-1P.server.log).
    ok = in_kfd and vram >= args.min_vram_bytes
    res = {"ok": ok, "pid": args.pid, "log": args.log, "startup_log_device_lines": lines[:8],
           "n_startup_log_device_lines": len(lines), "pid_in_kfd": in_kfd, "pid_vram_bytes": vram,
           "min_vram_bytes": args.min_vram_bytes, "argv_device": dev_arg, "argv_ngl": ngl_arg,
           "summary": f"{'OK' if ok else 'FAIL'}: pid {args.pid} in KFD={in_kfd} VRAM={vram / 1e9:.2f} GB "
                      f"(need >= {args.min_vram_bytes / 1e9:.2f}); {len(lines)} device line(s) in the log; "
                      f"argv --device {dev_arg or '-'} -ngl {ngl_arg or '-'}"}
    print(json.dumps(res, sort_keys=True))
    return 0 if ok else 1


def cmd_props(args) -> int:
    try:
        d = get(args.port, "/props")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"ok": False, "error": f"GET /props on :{args.port}: {exc!r}",
                          "summary": f"FAIL: GET /props on :{args.port}: {exc!r}"}))
        return 2
    bi = d.get("build_info")
    mp = d.get("model_path")
    mods = d.get("modalities") or {}
    params = (d.get("default_generation_settings") or {}).get("params") or {}
    problems = []
    if not isinstance(bi, str) or args.commit not in bi:
        problems.append(f"build_info {bi!r} lacks {args.commit}")
    if args.model and mp != args.model:
        problems.append(f"model_path {mp!r} != {args.model!r}")
    if args.vision and mods.get("vision") is not True:
        problems.append(f"modalities.vision is {mods.get('vision')!r} (mmproj not loaded?)")
    res = {"ok": not problems, "port": args.port, "build_info": bi, "model_path": mp, "modalities": mods,
           "total_slots": d.get("total_slots"),
           # Recorded, never gated: production :8083 (v10, launched with --spec-type draft-mtp and
           # --spec-draft-n-max 4, MTP demonstrably accepting drafts in its log) reports
           # speculative.types='none' and speculative.n_max=3 here -- /props shows per-request DEFAULTS,
           # not the server's speculative config. MTP activity is proven by timings.draft_n_accepted.
           "speculative_types_props": params.get("speculative.types"),
           "speculative_n_max_props": params.get("speculative.n_max"),
           "problems": problems,
           "summary": (f"{'OK' if not problems else 'FAIL'}: build_info={bi} model_path={mp} "
                       f"vision={mods.get('vision')}" + (f"; PROBLEMS: {' | '.join(problems)}" if problems else ""))}
    print(json.dumps(res, sort_keys=True))
    return 0 if not problems else 1


def cmd_window_watch(args) -> int:
    """Paused-loop watcher: the operator/peer paused the AutoKernel loop, so the window reads
    closed/exited for the WHOLE run and window_gate.py --mid-run would abort the arm within 5 s (the
    gate refused with exactly that on 2026-09-29 07:07Z). Instead: abort the moment the loop comes back
    -- (state, phase) differs from the baseline taken at start, loop_holds_claim turns true, or the
    loop reserves cpus inside our cpuset. A baseline of state=open falls back to the strict rules."""
    try:
        base = json.load(open(args.baseline))
        win = json.load(open(args.window))
    except (OSError, ValueError) as exc:
        print(f"window-watch: cannot read baseline/window: {exc}")
        return 2
    b_state, b_phase = base.get("state"), base.get("phase")
    reasons = []
    now = dt.datetime.now(dt.timezone.utc)

    def stale(w):
        exp = w.get("expires_at")
        try:
            return not exp or now >= dt.datetime.fromisoformat(exp.replace("Z", "+00:00"))
        except ValueError:
            return True
    # A paused baseline must be a STOPPED loop: the writer's exit state (cpu_window.py: closed,
    # loop_holds_claim false, phase "exited", heartbeat expired). "closing"/"closed" with a live phase
    # means the loop is running and about to measure -- paused mode must never accept that.
    if b_state != "open" and not (b_state == "closed" and base.get("loop_holds_claim") is False
                                   and (b_phase == "exited" or stale(base))):
        print(f"window-watch REFUSED: paused mode needs a STOPPED loop (closed/exited, no claim); the "
              f"baseline is {b_state}/{b_phase} loop_holds_claim={base.get('loop_holds_claim')} -- the loop is live")
        return 3
    if win.get("loop_holds_claim") is True:
        reasons.append("loop_holds_claim=true")
    overlap = sorted(parse_cpus(win.get("cpus_reserved_by_loop", "")) & parse_cpus(args.cpuset))
    if overlap:
        reasons.append(f"loop-reserved cpus {fmt_cpus(overlap)} overlap {args.cpuset}")
    if b_state == "open":
        if win.get("state") != "open":
            reasons.append(f"state {win.get('state')!r} (baseline was open)")
        if stale(win):
            reasons.append(f"stale window (expires_at {win.get('expires_at')})")
    elif (win.get("state"), win.get("phase")) != (b_state, b_phase):
        reasons.append(f"loop came back: state/phase {win.get('state')}/{win.get('phase')} "
                       f"!= paused baseline {b_state}/{b_phase}")
    print(("window-watch REFUSED: " + "; ".join(reasons)) if reasons else
          f"window-watch OK: state/phase {win.get('state')}/{win.get('phase')} (baseline {b_state}/{b_phase})")
    return 3 if reasons else 0


# ── summarize ───────────────────────────────────────────────────────────────
def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def residency_summary(path: Path, pid: int | None) -> dict:
    samples = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except ValueError:
                pass
    arm = [s for s in samples if pid is not None and str(pid) in (s.get("kfd_pids") or [])]
    vram = [s.get("pid_vram_bytes", 0) or 0 for s in arm]
    return {
        "n_samples": len(samples),
        "n_samples_pid_in_kfd": len(arm),
        "n_samples_pid_vram_nonzero": sum(1 for v in vram if v > 0),
        "pid_vram_bytes_max": max(vram) if vram else 0,
        "total_vram_used_bytes_max": max((s.get("vram_used_bytes") or 0 for s in samples), default=0),
        "gpu_busy_pct_max": max((s.get("gpu_busy_pct") or 0 for s in samples), default=0),
        "resident": len([v for v in vram if v > 0]) >= 2,  # two-sample persistence
    }


def summarize(args) -> int:
    run = Path(args.run)
    arms = []
    for meta_p in sorted(run.glob("arm-*.meta.json")):
        meta = load_json(meta_p) or {}
        tag = meta.get("tag")
        probe = load_json(run / f"arm-{tag}.probe.json")
        bench = load_json(run / f"arm-{tag}.llama-bench.json")
        res = residency_summary(run / f"arm-{tag}.residency.jsonl", meta.get("pid"))
        bench_res = residency_summary(run / f"arm-{tag}.llama-bench.residency.jsonl", meta.get("bench_pid"))
        arms.append({"tag": tag, "arm": meta.get("arm"), "meta": meta, "probe": probe, "llama_bench": bench,
                     "residency": res, "llama_bench_residency": bench_res})

    def by_arm(letter):
        # a probe that failed (ok=false) carries no comparable data; counting it would make the
        # identity comparison vacuous (two empty digest lists compare equal)
        return [a for a in arms if a["arm"] == letter and a["probe"] and a["probe"].get("ok") is True]

    P, C = by_arm("P"), by_arm("C")
    verdict: dict = {}
    verdict["probe_failed"] = {a["tag"]: (a["probe"] or {}).get("error", "no probe file")
                               for a in arms if not (a["probe"] and a["probe"].get("ok") is True)}

    # 1. GPU-device evidence (KFD VRAM of the pid at verify time; the startup-log lines are recorded,
    #    not required -- v10 logs none) + residency during the run + placement
    def dev_ok(a):
        ev = a["meta"].get("device_evidence")
        if isinstance(ev, dict):
            return ev.get("ok") is True
        return bool(a["meta"].get("startup_log_device_lines"))  # pre-2026-09-29T07:15Z meta files
    verdict["gpu_device_evidence"] = {a["tag"]: dev_ok(a) for a in arms}
    verdict["gpu_resident_during_run"] = {a["tag"]: a["residency"]["resident"] for a in arms}
    verdict["own_ggml_mapped"] = {a["tag"]: a["meta"].get("runtime_maps_ok") for a in arms}

    def aff_ok(a):
        post = a["meta"].get("affinity_post")
        pre = a["meta"].get("affinity_pre")
        if post is None and pre is None:
            return True  # meta written before the affinity split (verify_arm then died on any outside thread)
        return all(x is None or x.get("ok") is True for x in (pre, post))
    verdict["affinity_ok"] = {a["tag"]: aff_ok(a) for a in arms}
    verdict["affinity_detail"] = {a["tag"]: {"pre": (a["meta"].get("affinity_pre") or {}).get("summary"),
                                             "post": (a["meta"].get("affinity_post") or {}).get("summary")}
                                  for a in arms}

    # 2. byte identity (n_probs=0, greedy)
    def ident_keys(a):
        p = a["probe"]
        keys = [x["content_sha256"] for x in p.get("identity_text", [])]
        keys += [x["content_sha256"] for x in p.get("identity_image", [])]
        return keys

    ident = {a["tag"]: ident_keys(a) for a in P + C}
    ref = ident[P[0]["tag"]] if P else None
    nonvacuous = bool(ref) and all(len(ident[a["tag"]]) == len(ref) for a in P + C)
    verdict["identity"] = {
        "reference": P[0]["tag"] if P else None,
        "n_digests_per_arm": len(ref) if ref else 0,
        "P_self_deterministic": (nonvacuous and all(ident[a["tag"]] == ref for a in P)) if P else None,
        "C_equals_P": (nonvacuous and all(ident[a["tag"]] == ref for a in C)) if (P and C) else None,
        "per_arm_digests": ident,
    }

    # 3. speed: C vs mean(P), noise floor = the P-vs-P spread (ABA drift) with a floor of NOISE_MIN_PCT
    def med(arms_, key):
        vals = [a["probe"]["speed"][key] for a in arms_ if a["probe"].get("speed", {}).get(key)]
        return vals

    speed = {}
    for key in ("pp_tps_median", "tg_tps_median"):
        pv, cv = med(P, key), med(C, key)
        if pv and cv:
            pm, cm = statistics.mean(pv), statistics.mean(cv)
            spread = (max(pv) - min(pv)) / pm * 100 if len(pv) > 1 else None
            floor = max(spread or 0.0, args.noise_min_pct)
            delta = (cm / pm - 1) * 100
            speed[key] = {"P": pv, "C": cv, "P_mean": pm, "C_mean": cm, "C_over_P_pct": round(delta, 3),
                          "P_spread_pct": None if spread is None else round(spread, 3),
                          "noise_floor_pct": round(floor, 3), "unit": "tok/s (server timings)",
                          "within_noise": abs(delta) <= floor, "regression": delta < -floor}
    for key in ("pp", "tg"):
        pv = [a["llama_bench"][key] for a in P if a["llama_bench"] and a["llama_bench"].get(key)]
        cv = [a["llama_bench"][key] for a in C if a["llama_bench"] and a["llama_bench"].get(key)]
        if pv and cv:
            pm, cm = statistics.mean(pv), statistics.mean(cv)
            spread = (max(pv) - min(pv)) / pm * 100 if len(pv) > 1 else None
            floor = max(spread or 0.0, args.noise_min_pct)
            delta = (cm / pm - 1) * 100
            speed[f"llama_bench_{key}"] = {"P": pv, "C": cv, "C_over_P_pct": round(delta, 3),
                                           "P_spread_pct": None if spread is None else round(spread, 3),
                                           "noise_floor_pct": round(floor, 3), "unit": "tok/s (llama-bench avg_ts)",
                                           "within_noise": abs(delta) <= floor, "regression": delta < -floor}
    verdict["speed"] = speed

    # 4. SW-9 probs (mtp only)
    if args.kind == "mtp":
        def nprob(a, k):
            return a["probe"].get(k) or {}
        verdict["sw9"] = {
            a["tag"]: {
                "raw": {k: nprob(a, "nprobs_raw").get(k) for k in ("tokens_predicted", "n_tokens", "n_empty", "n_without_probs")}
                       | {"draft_n_accepted": (nprob(a, "nprobs_raw").get("timings") or {}).get("draft_n_accepted")},
                "post": {k: nprob(a, "nprobs_post").get(k) for k in ("tokens_predicted", "n_tokens", "n_empty", "n_without_probs")}
                        | {"draft_n_accepted": (nprob(a, "nprobs_post").get("timings") or {}).get("draft_n_accepted")},
            } for a in P + C
        }
        verdict["sw9_champion_probs_on_every_token"] = bool(C) and all(
            nprob(a, "nprobs_raw").get("n_tokens", 0) > 0 and nprob(a, "nprobs_raw").get("n_without_probs") == 0
            and nprob(a, "nprobs_post").get("n_tokens", 0) > 0 and nprob(a, "nprobs_post").get("n_without_probs") == 0
            and ((nprob(a, "nprobs_raw").get("timings") or {}).get("draft_n_accepted") or 0) > 0
            for a in C)
        verdict["sw9_v10_shows_gap"] = bool(P) and all(
            (nprob(a, "nprobs_raw").get("n_without_probs") or 0) > 0 for a in P)
    else:
        verdict["sw9"] = {"tested": False, "reason": SW9_VISION_NOTE}

    checks = {
        "arms_P_and_C_with_ok_probe": bool(P) and bool(C),
        "no_failed_probe": not verdict["probe_failed"],
        "gpu_device_evidence": all(verdict["gpu_device_evidence"].values()) if arms else False,
        "gpu_resident_during_run": all(verdict["gpu_resident_during_run"].values()) if arms else False,
        "own_ggml_mapped": all(v is True for v in verdict["own_ggml_mapped"].values()) if arms else False,
        "affinity_ok": all(verdict["affinity_ok"].values()) if arms else False,
        "identity_C_equals_P": verdict["identity"]["C_equals_P"] is True,
        "speed_no_regression": all(not s["regression"] for s in speed.values()) and bool(speed),
    }
    if args.kind == "mtp":
        checks["sw9_champion_probs_on_every_token"] = verdict["sw9_champion_probs_on_every_token"]
    verdict["checks"] = checks
    verdict["failed_checks"] = [k for k, v in checks.items() if not v]
    verdict["PASS"] = all(checks.values())

    result = {
        "schema": "epyc.gpu_champion_ab.result.v1",
        "kind": args.kind,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run),
        "context": load_json(run / "context.json"),
        "verdict": verdict,
        "arms": [{k: v for k, v in a.items() if k != "probe"} | {"probe_file": f"arm-{a['tag']}.probe.json"}
                 for a in arms],
    }
    (run / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    v = verdict
    print(f"RESULT {args.kind}: PASS={v['PASS']} arms P={len(P)} C={len(C)} identity C==P={v['identity']['C_equals_P']} "
          f"P-self={v['identity']['P_self_deterministic']} "
          + " ".join(f"{k}:{s['C_over_P_pct']:+.2f}%(floor {s['noise_floor_pct']:.2f}%)" for k, s in speed.items())
          + (f" sw9_C_all_probs={v.get('sw9_champion_probs_on_every_token')} sw9_P_gap={v.get('sw9_v10_shows_gap')}"
             if args.kind == "mtp" else " SW-9=NOT-TESTED(no MTP drafter on the vision model)")
          + (f" FAILED={','.join(v['failed_checks'])}" if v["failed_checks"] else ""))
    print(f"result -> {run / 'result.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("kind", choices=("vision", "mtp"))
    pr.add_argument("--port", type=int, required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--image")
    pr.add_argument("--identity-reps", type=int, default=1)
    pr.add_argument("--identity-tokens", type=int, default=192)
    pr.add_argument("--probs-tokens", type=int, default=64)
    pr.add_argument("--pp-items", type=int, default=150)
    pr.add_argument("--pp-reps", type=int, default=2)
    pr.add_argument("--tg-n", type=int, default=256)
    pr.add_argument("--tg-reps", type=int, default=3)
    sm = sub.add_parser("summarize")
    sm.add_argument("--run", required=True)
    sm.add_argument("--kind", choices=("vision", "mtp"), required=True)
    sm.add_argument("--noise-min-pct", type=float, default=2.0,
                    help="minimum noise floor, percent of tok/s; the P-vs-P spread widens it")
    ag = sub.add_parser("args")
    ag.add_argument("--pid", type=int)
    ag.add_argument("--argv-file")
    ag.add_argument("--port", type=int, required=True)
    ag.add_argument("--slot-dir", required=True)
    af = sub.add_parser("affinity")
    af.add_argument("--pid", type=int, required=True)
    af.add_argument("--cpuset", required=True)
    af.add_argument("--samples", type=int, default=10, help="wchan samples, 0.1 s apart")
    dl = sub.add_parser("devlog")
    dl.add_argument("--pid", type=int, default=0)
    dl.add_argument("--log", required=True)
    dl.add_argument("--min-vram-bytes", type=int, default=1_000_000_000)
    pp_ = sub.add_parser("props")
    pp_.add_argument("--port", type=int, required=True)
    pp_.add_argument("--commit", required=True)
    pp_.add_argument("--model")
    pp_.add_argument("--vision", action="store_true")
    ww = sub.add_parser("window-watch")
    ww.add_argument("--baseline", required=True)
    ww.add_argument("--cpuset", required=True)
    ww.add_argument("--window", default="/mnt/raid0/llm/autokernel/cpu-window.json")
    a = ap.parse_args()
    if a.cmd == "args":
        return derive_args(a)
    if a.cmd == "summarize":
        return summarize(a)
    if a.cmd == "affinity":
        return cmd_affinity(a)
    if a.cmd == "devlog":
        return cmd_devlog(a)
    if a.cmd == "props":
        return cmd_props(a)
    if a.cmd == "window-watch":
        return cmd_window_watch(a)
    res: dict = {"kind": a.kind, "port": a.port}
    try:
        (probe_vision if a.kind == "vision" else probe_mtp)(a, res)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- ANY failure must still leave a probe file saying where
        res["ok"] = False
        res["error"] = repr(exc)
        res["failed_at_step"] = res.pop("_step", None)
        res["traceback"] = traceback.format_exc(limit=6)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    sp = res.get("speed") or {}

    def f(x, nd):
        return "None" if not isinstance(x, (int, float)) else f"{x:.{nd}f}"
    print(f"probe {a.kind} port {a.port}: ok={res['ok']}"
          + (f" tg={f(sp.get('tg_tps_median'), 2)} pp={f(sp.get('pp_tps_median'), 1)} tok/s"
             if res.get("ok") else f" FAILED at step {res.get('failed_at_step')!r}: {res.get('error')}"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
