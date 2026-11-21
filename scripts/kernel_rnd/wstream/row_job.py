#!/usr/bin/python3
"""ak-wstream-decode-ubench-cpu-20261121 (autokernel; autokernel-akx-program.md:WSTREAM-DECODE-UBENCH-CPU).

QUESTION  can any lossless scheme decode Q8_0 int8 weight codes faster than DRAM delivers them on one Zen 5 core?
MODEL     none. Synthetic i.i.d. code streams drawn from measured 256-bin histograms of real Q8_0 tensors
          (inputs/hist_q27b.txt = Qwen3.8-27B Q8_0 ffn_*, inputs/hist_a3b.txt = Qwen3.6-35B-A3B Q8_0 ffn_*_exps).
HARNESS   inputs/wstream_ubench (AVX-512 C, source inputs/wstream_ubench.c; hashes pinned in SOURCE.sha256).
          Per invocation: 1 GiB raw codes, 4 schemes (static scalar Huffman | split raw-low-nibble + 32-lane Huffman hi
          nibble | tANS 4-state | 32-lane interleaved Huffman | scales_only = raw int8 codes + 32-lane-Huffman 8-bit scale index into
          a per-tensor fp16 table; its decoded bytes count codes + 2-byte scales, its saving is taken from the census), every chunk round-trip checked (lossless), decode timed on
          ONE core (physical cpu 0) into an L2-resident chunk; Q8_0 VNNI dot cycles/byte; 24-thread DRAM read GB/s on
          physical 0-23 (per-core share = that / 24, the decode threads of a q0 instance). Scheme order rotates per round.
ROUNDS    6 interleaved rounds x {q27b, a3b}.
RULE      need = 1.25 x share / ratio (decoded bytes/s a core must produce to consume its compressed share).
          scheme PASS iff lossless AND median decoded GB/s >= need AND fused (1/(1/dec + 1/dot)) GB/s >= need
          (decode + existing dot compute fit in the time the core has per streamed byte). LOSSLESS_FAIL iff any chunk
          mismatches. Row outcome PASS iff any scheme passes, else FAIL.
CHECKPOINTS after rounds 3 and 6. STOP-ON-ANSWER: round 6, or after round 3 every scheme < 0.5 x share (outcome
          FAIL_BELOW_HALF). STOP-ON-FUTILITY: missing input / hash mismatch / wrong grant (FUTILITY_*, never rc 75).
rc: 0 verdict written | 75 STOP/FILLER_DEADLINE (resumable per invocation) | 3 smoke not ok
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

D = Path("/mnt/raid0/llm/tmp/ak-wstream-decode-ubench-cpu-20261121")
ID = "ak-wstream-decode-ubench-cpu-20261121"
TASK = "autokernel-akx-program.md:WSTREAM-DECODE-UBENCH-CPU"
OUT = D / "out"
BIN = D / "inputs/wstream_ubench"
HISTS = {"q27b": D / "inputs/hist_q27b.txt", "a3b": D / "inputs/hist_a3b.txt"}
PIN = list(range(0, 24))
ROUNDS, INTERIM, RAW_MIB, MARGIN = 6, 3, 1024, 1.25
SCHEMES = ["static", "split", "tans", "lanes32", "scales_only"]
SMOKE = os.environ.get("FILLER_SMOKE") == "1" or os.environ.get("FILLER_DRYRUN") == "1"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsave(p, o):
    p = Path(p); t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(o, indent=1)); os.replace(t, p)


def stop_now():
    dl = os.environ.get("FILLER_DEADLINE")
    return (D / "STOP").exists() or bool(dl and time.time() > int(dl))


def run(hist, rnd, mib, bwn, cpus):
    cmd = ["/usr/bin/taskset", "-c", cpus, str(BIN), str(HISTS[hist]), str(mib), str(rnd), str(bwn)]
    env = {"PATH": "/usr/bin:/bin", "HIP_VISIBLE_DEVICES": "-1", "FILLER_ID": os.environ.get("FILLER_ID", ID)}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if p.returncode:
        raise RuntimeError(f"rc {p.returncode}: {p.stderr[-400:]}")
    r = json.loads(p.stdout.strip().splitlines()[-1]); r["hist"] = hist; r["utc"] = now(); r["argv"] = cmd
    return r


def decide(recs, final):
    share = statistics.median(r["per_core_share_gbs"] for r in recs)
    dot = statistics.median(r["dot_q8_gbps"] for r in recs)
    tab = {}
    for s in SCHEMES:
        dec = statistics.median(r["schemes"][s]["decoded_gbps"] for r in recs)
        ratio = statistics.median(r["schemes"][s]["ratio"] for r in recs)
        cyc = statistics.median(r["schemes"][s]["tsc_cyc_per_byte"] for r in recs)
        lossless = all(r["schemes"][s]["ok"] for r in recs)
        need = MARGIN * share / ratio
        fused = 1.0 / (1.0 / dec + 1.0 / dot)
        v = "LOSSLESS_FAIL" if not lossless else ("PASS" if dec >= need and fused >= need else "FAIL")
        tab[s] = {"decoded_gbps_median": round(dec, 3), "ratio": round(ratio, 5), "saving_pct": round(100 * (1 - ratio), 3),
                  "tsc_cyc_per_output_byte": round(cyc, 4), "need_gbps": round(need, 3), "fused_gbps": round(fused, 3),
                  "frac_of_share": round(dec / share, 3), "lossless": lossless, "verdict": v}
    if any(t["verdict"] == "PASS" for t in tab.values()) and final:
        out = "PASS"
    elif not final and all(t["decoded_gbps_median"] < 0.5 * share for t in tab.values()):
        out = "FAIL_BELOW_HALF"
    elif final:
        out = "FAIL"
    else:
        out = None
    return out, {"per_core_share_gbs": round(share, 3), "dot_q8_gbps": round(dot, 3), "schemes": tab}


def finish(verdict, **kw):
    jsave(OUT / "RESULT.json", {"id": ID, "status": "complete", "verdict": verdict, "updated_utc": now(), "handoff_task": TASK, **kw})
    return 0


def missing():
    return [str(p) for p in [BIN, *HISTS.values()] if not p.exists() or p.stat().st_size == 0]


def smoke():
    (OUT / "smoke").mkdir(parents=True, exist_ok=True)
    cpus = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    res = {"id": ID, "utc": now(), "missing": missing(), "cpus": cpus}
    try:
        r = run("q27b", 0, 32, 1, cpus)  # small: smoke budget ~1 GiB
        res["harness"] = r; res["lossless_all"] = all(r["schemes"][s]["ok"] for s in SCHEMES)
        res["decide_selftest"] = decide([r], True)[0] in ("PASS", "FAIL")
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e); res["lossless_all"] = False
    res["ok"] = not res["missing"] and res.get("lossless_all") and res.get("decide_selftest", False)
    jsave(OUT / "smoke" / f"{ID}.json", res); jsave(OUT / "TERMINAL-smoke.json", res)
    return 0 if res["ok"] else 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if SMOKE:
        return smoke()
    if missing():
        return finish("FUTILITY_MISSING_INPUT", missing=missing())
    phys = {c % 96 for c in os.sched_getaffinity(0)}
    if not set(PIN) <= phys:
        return finish("FUTILITY_WRONG_GRANT", granted_physical=sorted(phys), need="0-23 (q0)")
    led = OUT / "results.jsonl"
    recs = [json.loads(l) for l in led.read_text().splitlines()] if led.exists() else []
    done = {(r["round"], r["hist"]) for r in recs}
    for rnd in range(ROUNDS):
        for h in HISTS:
            if (rnd, h) in done:
                continue
            if stop_now():
                return 75
            try:
                r = run(h, rnd, RAW_MIB, 24, "0-23")
            except Exception as e:  # noqa: BLE001
                return finish("FUTILITY_HARNESS", error=str(e))
            recs.append(r)
            with led.open("a") as f:
                f.write(json.dumps(r) + "\n")
        n = rnd + 1
        if n == INTERIM:
            out, tab = decide(recs, False)
            jsave(OUT / "interim_r3.json", {"outcome": out, **tab})
            if out:
                return finish("ANSWERED", outcome=out, rounds=n, table=tab)
    out, tab = decide(recs, True)
    return finish("ANSWERED", outcome=out, rounds=ROUNDS, table=tab)


if __name__ == "__main__":
    sys.exit(main())
