#!/usr/bin/env python3
"""24-prompt production-mix client for the R23-64 thread sweep.

ADAPTED FROM the INF-70 champion-3 / harness1 client (sha256 ebc74836b000...ddb, see PROVENANCE.txt):
chat-completions, enable_thinking=false, greedy, max_tokens 200, cache_prompt false, coherence
classified BY REASON. Token-weighted rates are computed in analyze_sweep.py (pred_n >= 16 floor).

Changes, each a correction (the request body and the row schema are UNCHANGED, so a row is comparable
with every INF-70 row):
  * prompts.json and classify.py are the vendored copies next to this file, not paths into scratch;
  * rows are written to <label>.rows.jsonl.partial after every prompt and RENAMED to
    <label>.rows.jsonl only when the launch is complete with zero HTTP errors. The old client rewrote
    <label>.rows.jsonl directly, so a launch killed mid-run left a partial file that the
    "existing rows.jsonl is skipped" resume rule then counted as DONE;
  * the per-request timeout is a parameter (default 300 s, was 1800) so one hung request cannot eat a window.

  usage: client.py <label> <port> [--out DIR] [--timeout S]      (rc 0 complete, 3 incomplete)
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from classify import classify  # noqa: E402


def load_items(path=None):
    prompts = json.load(open(path or os.path.join(HERE, "prompts.json")))
    return [dict(id=p["id"], cls=p["class"], text=p["prompt"]) for p in prompts]


def mkbody(text, max_tokens=200):
    return {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.0}


def post(port, body, timeout):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def make_row(k, it, j, wall):
    ch = (j.get("choices") or [{}])[0]
    content = (ch.get("message") or {}).get("content", "") or ""
    u = j.get("usage") or {}
    tim = j.get("timings") or {}
    npred = u.get("completion_tokens") or tim.get("predicted_n") or 0
    c = classify(dict(tokens=list(range(npred)), stop_type=ch.get("finish_reason"), content=content))
    tps = tim.get("predicted_per_second") or (npred / wall if wall > 0 and npred else None)
    return dict(seq=k, id=it["id"], cls=it["cls"],
                prompt_n=u.get("prompt_tokens") or tim.get("prompt_n"),
                pred_n=npred, tps=round(tps, 4) if tps else None,
                pp=round(tim.get("prompt_per_second"), 3) if tim.get("prompt_per_second") else None,
                prompt_ms=tim.get("prompt_ms"), pred_ms=tim.get("predicted_ms"),
                wall=round(wall, 3), draft_n=tim.get("draft_n"), draft_acc=tim.get("draft_n_accepted"),
                finish=ch.get("finish_reason"), verdict=c.get("cls"),
                sha=hashlib.sha256(content.encode()).hexdigest()[:16],
                n_uniq=c.get("uniq"), words=c.get("words"), text=content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("port")
    ap.add_argument("--out", default=os.environ.get("CLIENT_OUT", "."))
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    items = load_items()
    rows, t0a = [], time.time()
    try:
        post(a.port, mkbody("hi"), a.timeout)
        print("warmup ok", flush=True)
    except Exception as e:  # noqa: BLE001 -- the warmup is advisory; a dead server fails every prompt below
        print("warmup failed:", e, flush=True)
    part = os.path.join(a.out, f"{a.label}.rows.jsonl.partial")
    for k, it in enumerate(items):
        t0 = time.time()
        try:
            j = post(a.port, mkbody(it["text"]), a.timeout)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(seq=k, id=it["id"], cls=it["cls"], verdict="HTTP-ERROR", note=str(e)[:120]))
            print(json.dumps(rows[-1]), flush=True)
            continue
        rows.append(make_row(k, it, j, time.time() - t0))
        r = rows[-1]
        print(json.dumps({kk: vv for kk, vv in r.items() if kk != "text"} | {"head": r["text"][:70].replace("\n", " ")}), flush=True)
        with open(part, "w") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
    errs = sum(1 for r in rows if r.get("verdict") == "HTTP-ERROR")
    with open(part, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    if errs or len(rows) != len(items):
        os.replace(part, os.path.join(a.out, f"{a.label}.rows.failed.jsonl"))
        print(f"INCOMPLETE {a.label}: {errs} HTTP errors, {len(rows)}/{len(items)} rows", flush=True)
        return 3
    os.replace(part, os.path.join(a.out, f"{a.label}.rows.jsonl"))
    print(f"ARM_DONE {a.label} wall={time.time()-t0a:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
