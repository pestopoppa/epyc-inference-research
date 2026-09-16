#!/usr/bin/env python3
"""OCC-1 runner: billed-token cost and QA recall, bitmap frames vs raw text, on a served reader.

    plan    render every frame, predict costs, fingerprint the request suite   (NO inference)
    run     send the suite to a live llama-server, one request at a time        (inference)
    report  paired analysis + pre-registered verdict from records.jsonl         (NO inference)

Needs Pillow for plan/run (`uv run --with pillow ...`). See README.md for the GPU recipe.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.benchmark.occ1"

from . import costs, fixture, prompts, render, stats  # noqa: E402

DEFAULT_ARMS = "text,img-6x10-bw,img-6x10-color,img-8x13-bw,img-8x8u-bw,img-12x12u-bw"
EXPECTED_MODEL = "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
EXPECTED_BUILD = "ef81196d5"
# Pre-registered decision parameters (do not change after the first `run`).
PREREG = {
    "primary_metric": "SQuAD F1 per question, paired arm-vs-text (higher is better)",
    "cost_metric": "server usage.prompt_tokens per request (lower is better)",
    "max_token_ratio": 0.5,
    "ni_margin_f1": 0.05,
    "ci": "95% percentile bootstrap of mean paired F1 delta, resampling whole chunks, 10000 iters, seed 0",
    "secondary": "EM, exact two-sided McNemar; UNREADABLE rate; F1 by position quartile",
    "decision": {
        "POSITIVE": "ratio <= 0.5 AND F1-delta CI lower bound > -0.05",
        "NOT_NONINFERIOR": "ratio <= 0.5 AND CI lower bound <= -0.05",
        "NEGATIVE_COST": "ratio > 0.5 (no worthwhile saving regardless of recall)",
        "OCC-1 overall": "POSITIVE if ANY image arm is POSITIVE (that arm seeds OCC-3), else NEGATIVE",
        "VOID": "text-arm F1 < 0.60 (reader/instrument broken), any transport error left unrepaired, "
                "server identity mismatch, or suite fingerprint drift between plan and run",
    },
}


def parse_arm(name: str) -> dict:
    if name == "text":
        return {"name": name, "kind": "text"}
    parts = name.split("-")
    if len(parts) != 3 or parts[0] != "img" or parts[1] not in render.FONTS or parts[2] not in render.VARIANTS:
        raise SystemExit(f"bad arm {name!r}: expected text | img-<{'|'.join(render.FONTS)}>-<bw|color>")
    return {"name": name, "kind": "image", "font": render.FONTS[parts[1]], "variant": parts[2]}


def load_fixture(args) -> tuple[list[fixture.Chunk], dict[int, list[dict]]]:
    path = fixture.ensure_squad(Path(args.cache), download=args.download)
    paras = fixture.load_paragraphs(path)
    flow, offsets = fixture.build_flow(paras)
    chunks = fixture.chunk_flow(flow, args.chunk_chars)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
    qs = {c.index: fixture.sample_chunk_questions(paras, offsets, c, args.qpc, args.seed) for c in chunks}
    return chunks, qs


def frame_path(out: Path, arm: str, chunk: int, k: int) -> Path:
    return out / "frames" / arm / f"c{chunk:03d}_f{k}.png"


def text_token_estimator(path: str | None):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("tokenizers not installed; skipping text-token estimate", file=sys.stderr)
        return None
    tok = Tokenizer.from_file(path)
    return lambda s: len(tok.encode(s, add_special_tokens=False).ids)


def build_requests(args, out: Path, write_frames: bool) -> dict:
    """Deterministic request suite. Renders frames (and writes them when write_frames)."""
    chunks, qs = load_fixture(args)
    arms = [parse_arm(a.strip()) for a in args.arms.split(",") if a.strip()]
    if not any(a["kind"] == "text" for a in arms):
        raise SystemExit("the text arm is the paired baseline and is required")
    renderers = {
        a["name"]: render.Renderer(a["font"], a["variant"], Path(args.cache), args.download)
        for a in arms if a["kind"] == "image"
    }
    est = text_token_estimator(args.tokenizer)
    requests = []
    for c in chunks:
        context = render.normalize_text(c.text)
        questions = qs[c.index]
        if not questions:
            continue
        # rotate arm order per chunk so slow host/GPU drift does not alias onto one arm
        rot = c.index % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            req = {"key": f"{arm['name']}|c{c.index:03d}", "arm": arm["name"], "chunk": c.index,
                   "n_chars": len(context), "questions": questions}
            if arm["kind"] == "text":
                req["messages_fp"] = prompts.fingerprint(prompts.text_messages(context, questions))
                req["pred_text_tokens"] = est(context) if est else None
                req["frames"] = []
            else:
                frames = renderers[arm["name"]].render(context)
                cols, rows = render.frame_capacity(arm["font"])
                meta = []
                for k, (img, s, e) in enumerate(frames):
                    p = frame_path(out, arm["name"], c.index, k)
                    if write_frames:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        img.save(p, format="PNG", compress_level=6)
                    meta.append({"path": str(p.relative_to(out)), "w": img.width, "h": img.height,
                                 "chars": [s, e], "pixel_sha256": render.pixel_digest(img),
                                 "pred_tokens": costs.image_tokens(img.width, img.height),
                                 "resample_free": costs.is_resample_free(img.width, img.height)})
                req.update(frames=meta, cols=cols, rows=rows)
                req["messages_fp"] = prompts.fingerprint(
                    {"pixels": [m["pixel_sha256"] for m in meta], "cols": cols, "rows": rows,
                     "q": prompts.question_block(questions), "head": prompts.IMAGE_HEAD, "tail": prompts.IMAGE_TAIL})
                req["pred_image_tokens"] = sum(m["pred_tokens"] for m in meta)
            requests.append(req)
    suite_fp = prompts.fingerprint([[r["key"], r["messages_fp"]] for r in requests])
    return {"arms": [a["name"] for a in arms], "n_chunks": len(chunks),
            "n_questions_per_arm": sum(len(v) for v in qs.values()),
            "chunk_chars": args.chunk_chars, "qpc": args.qpc, "seed": args.seed,
            "squad_sha256": fixture.SQUAD_SHA256, "font_sha256": render.FONT_SHA256,
            "frame": {"w": render.FRAME_W, "h_max": render.FRAME_H_MAX, "h_min": render.FRAME_H_MIN,
                      "align": render.ALIGN},
            "reader_grid": vars(costs.QWEN3VL_PROD), "prereg": PREREG,
            "suite_fingerprint": suite_fp, "requests": requests}


def cmd_plan(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plan = build_requests(args, out, write_frames=True)
    plan["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plan["render_s"] = round(time.time() - t0, 1)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"plan: {len(plan['requests'])} requests, {plan['n_chunks']} chunks, "
          f"{plan['n_questions_per_arm']} questions/arm, suite {plan['suite_fingerprint'][:12]}, "
          f"render {plan['render_s']}s")
    for arm in plan["arms"]:
        rs = [r for r in plan["requests"] if r["arm"] == arm]
        if arm == "text":
            t = [r["pred_text_tokens"] for r in rs if r["pred_text_tokens"]]
            print(f"  {arm:<16} context tokens/chunk ~{statistics.mean(t):.0f} (tokenizer estimate)" if t
                  else f"  {arm:<16} (no tokenizer estimate)")
        else:
            nf = [len(r["frames"]) for r in rs]
            it = [r["pred_image_tokens"] for r in rs]
            rf = all(m["resample_free"] for r in rs for m in r["frames"])
            print(f"  {arm:<16} frames/chunk {statistics.mean(nf):.2f}  image tokens/chunk "
                  f"{statistics.mean(it):.0f}  resample_free={rf}")
    return 0


def frame_digest(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as im:
        return render.pixel_digest(im.convert("RGB"))


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def server_identity(url: str) -> dict:
    props = http_json(f"{url}/props", timeout=30)
    return {"build_info": props.get("build_info"), "model_path": props.get("model_path"),
            "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "modalities": props.get("modalities"), "total_slots": props.get("total_slots")}


def cmd_run(args, post=http_json, ident=server_identity) -> int:
    out = Path(args.out)
    plan_path = out / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"{plan_path} missing: run `plan` first")
    plan = json.loads(plan_path.read_text())
    # Re-derive the suite from the fixture; any drift voids the run before a token is spent.
    fresh = build_requests(args, out, write_frames=False)
    if fresh["suite_fingerprint"] != plan["suite_fingerprint"]:
        raise SystemExit("suite fingerprint drift between plan and run (fixture/font/prompt/args changed)")
    identity = ident(args.url)
    problems = []
    if EXPECTED_MODEL not in str(identity.get("model_path")):
        problems.append(f"model_path {identity.get('model_path')!r} is not {EXPECTED_MODEL}")
    if args.expect_build and args.expect_build not in str(identity.get("build_info")):
        problems.append(f"build_info {identity.get('build_info')!r} lacks {args.expect_build}")
    if not (identity.get("modalities") or {}).get("vision"):
        problems.append("server has no vision modality (was --mmproj passed?)")
    if problems and not args.force:
        raise SystemExit("server identity check failed: " + "; ".join(problems))
    (out / "server_identity.json").write_text(json.dumps({**identity, "problems": problems}, indent=1))

    chunk_text = {c.index: c.text for c in load_fixture(args)[0]}
    rec_path = out / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                done.add(r["key"])
    todo = [r for r in plan["requests"] if r["key"] not in done]
    if args.max_requests:
        todo = todo[: args.max_requests]
    print(f"run: {len(todo)} to send ({len(done)} already done) -> {args.url}")
    with rec_path.open("a") as fh:
        for i, req in enumerate(todo):
            questions = req["questions"]
            if req["frames"]:
                pngs = []
                for m in req["frames"]:
                    pngs.append((out / m["path"]).read_bytes())
                    if frame_digest(out / m["path"]) != m["pixel_sha256"]:
                        raise SystemExit(f"frame pixel drift: {m['path']} (re-run plan)")
                messages = prompts.image_messages(pngs, req["cols"], req["rows"], questions)
            else:
                messages = prompts.text_messages(render.normalize_text(chunk_text[req["chunk"]]), questions)
            payload = {"messages": messages, "temperature": 0.0, "seed": args.seed,
                       "max_tokens": args.max_tokens, "cache_prompt": False, "stream": False}
            t0 = time.time()
            err = None
            resp: dict = {}
            for attempt in range(2):
                try:
                    resp = post(f"{args.url}/v1/chat/completions", payload)
                    err = None
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    err = f"{type(e).__name__}: {e}"
            wall = time.time() - t0
            choice = (resp.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content")) or ""
            answers = fixture.parse_numbered(text, len(questions))
            usage = resp.get("usage") or {}
            timings = resp.get("timings") or {}
            rec = {"key": req["key"], "arm": req["arm"], "chunk": req["chunk"], "error": err,
                   "finish_reason": choice.get("finish_reason"), "raw": text,
                   "prompt_tokens": usage.get("prompt_tokens"),
                   "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "completion_tokens": usage.get("completion_tokens"),
                   "prompt_ms": timings.get("prompt_ms"), "predicted_ms": timings.get("predicted_ms"),
                   "wall_s": round(wall, 3), "n_frames": len(req["frames"]),
                   "pred_image_tokens": req.get("pred_image_tokens"),
                   "items": [{"qid": q["qid"], "pos_rel": q["pos_rel"], "answer": a,
                              "em": fixture.exact_match(a, q["golds"]), "f1": fixture.f1(a, q["golds"]),
                              "unreadable": "unreadable" in a.lower(), "missing": a == ""}
                             for q, a in zip(questions, answers)]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{i + 1}/{len(todo)}] {req['key']} pt={rec['prompt_tokens']} "
                  f"f1={stats.mean([x['f1'] for x in rec['items']]):.3f} {wall:.1f}s"
                  + (f" ERROR {err}" if err else ""), flush=True)
    return 0


def aggregate(plan: dict, records: list[dict]) -> dict:
    latest: dict[str, dict] = {}
    for r in records:
        if not r.get("error"):
            latest[r["key"]] = r
    errors = [r["key"] for r in records if r.get("error") and r["key"] not in latest]
    by_arm: dict[str, list[dict]] = {}
    for r in latest.values():
        by_arm.setdefault(r["arm"], []).append(r)
    chars = {r["key"]: r["n_chars"] for r in plan["requests"]}
    text_items = {(r["chunk"], it["qid"]): it for r in by_arm.get("text", []) for it in r["items"]}
    text_chunks = {r["chunk"]: r for r in by_arm.get("text", [])}
    rows = []
    for arm in plan["arms"]:
        rs = sorted(by_arm.get(arm, []), key=lambda r: r["chunk"])
        items = [it for r in rs for it in r["items"]]
        if not items:
            continue
        pts = [r["prompt_tokens"] for r in rs if r["prompt_tokens"] is not None]
        row = {
            "arm": arm, "requests": len(rs), "n": len(items),
            "em": stats.mean([i["em"] for i in items]),
            "f1": stats.mean([i["f1"] for i in items]), "f1_se": stats.se_mean([i["f1"] for i in items]),
            "unreadable": sum(i["unreadable"] for i in items), "missing": sum(i["missing"] for i in items),
            "truncated": sum(r["finish_reason"] == "length" for r in rs),
            "prompt_tokens_mean": stats.mean(pts) if pts else None,
            "prompt_tokens_per_1k_chars": (1000 * sum(pts) / sum(chars[r["key"]] for r in rs)) if pts else None,
            "cached_tokens_max": max((r.get("cached_tokens") or 0) for r in rs),
            "prompt_ms_median": statistics.median([r["prompt_ms"] for r in rs if r.get("prompt_ms")] or [0]),
            "wall_s_median": statistics.median([r["wall_s"] for r in rs]),
            "f1_by_quartile": [
                stats.mean([i["f1"] for i in items if lo <= i["pos_rel"] < hi]) if any(
                    lo <= i["pos_rel"] < hi for i in items) else None
                for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))],
        }
        if arm != "text" and text_items:
            pairs = [(i, text_items[(r["chunk"], i["qid"])], r["chunk"])
                     for r in rs for i in r["items"] if (r["chunk"], i["qid"]) in text_items]
            deltas = [a["f1"] - t["f1"] for a, t, _ in pairs]
            lo, hi = stats.paired_bootstrap_ci(deltas, cluster=[c for *_, c in pairs])
            b = sum(1 for a, t, _ in pairs if a["em"] and not t["em"])
            c = sum(1 for a, t, _ in pairs if t["em"] and not a["em"])
            paired_chunks = [r for r in rs if r["chunk"] in text_chunks
                             and r["prompt_tokens"] and text_chunks[r["chunk"]]["prompt_tokens"]]
            ratio = (sum(r["prompt_tokens"] for r in paired_chunks)
                     / sum(text_chunks[r["chunk"]]["prompt_tokens"] for r in paired_chunks)) if paired_chunks else None
            overhead = [r["prompt_tokens"] - r["pred_image_tokens"] for r in rs
                        if r["prompt_tokens"] is not None and r.get("pred_image_tokens") is not None]
            row.update({
                "n_paired": len(pairs), "f1_delta": stats.mean(deltas), "f1_delta_ci95": [lo, hi],
                "em_discordant_arm_only": b, "em_discordant_text_only": c,
                "em_mcnemar_p": stats.exact_mcnemar(b, c),
                "token_ratio_vs_text": ratio,
                "non_image_prompt_tokens_median": statistics.median(overhead) if overhead else None,
                "non_image_prompt_tokens_range": [min(overhead), max(overhead)] if overhead else None,
                "verdict": stats.verdict(ratio, lo, PREREG["max_token_ratio"], PREREG["ni_margin_f1"])
                if ratio is not None else "NO_PAIRS",
            })
        rows.append(row)
    text_row = next((r for r in rows if r["arm"] == "text"), None)
    void = []
    if text_row is None:
        void.append("no text arm")
    elif text_row["f1"] < 0.60:
        void.append(f"text-arm F1 {text_row['f1']:.3f} < 0.60")
    if errors:
        void.append(f"{len(errors)} unrepaired transport errors")
    if any((r.get("cached_tokens_max") or 0) > 0 for r in rows):
        void.append("prompt cache hit observed (cache_prompt must be off)")
    expected = len(plan["requests"])
    if len(latest) < expected:
        void.append(f"incomplete: {len(latest)}/{expected} requests")
    overall = "VOID" if void else (
        "POSITIVE" if any(r.get("verdict") == "POSITIVE" for r in rows) else "NEGATIVE")
    return {"overall": overall, "void_reasons": void, "rows": rows, "prereg": PREREG,
            "suite_fingerprint": plan["suite_fingerprint"]}


def cmd_report(args) -> int:
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    rec_path = out / "records.jsonl"
    records = [json.loads(x) for x in rec_path.read_text().splitlines()] if rec_path.exists() else []
    summary = aggregate(plan, records)
    ident = out / "server_identity.json"
    summary["server_identity"] = json.loads(ident.read_text()) if ident.exists() else None
    if summary["server_identity"] and summary["server_identity"].get("problems"):
        summary["void_reasons"].append("server identity problems: " + "; ".join(summary["server_identity"]["problems"]))
        summary["overall"] = "VOID"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# OCC-1 summary — overall **{summary['overall']}**", ""]
    if summary["void_reasons"]:
        lines += ["VOID/incomplete: " + "; ".join(summary["void_reasons"]), ""]
    lines += ["| arm | n | EM | F1 ±se | ΔF1 vs text [95% CI] | McNemar p (EM) | prompt tok/req | tok/1k chars | ratio vs text | UNREAD | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        ci = r.get("f1_delta_ci95")
        lines.append(
            f"| {r['arm']} | {r['n']} | {r['em']:.3f} | {r['f1']:.3f} ±{r['f1_se']:.3f} | "
            + (f"{r['f1_delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—") + " | "
            + (f"{r['em_mcnemar_p']:.4f}" if "em_mcnemar_p" in r else "—") + " | "
            + (f"{r['prompt_tokens_mean']:.0f}" if r["prompt_tokens_mean"] else "—") + " | "
            + (f"{r['prompt_tokens_per_1k_chars']:.1f}" if r["prompt_tokens_per_1k_chars"] else "—") + " | "
            + (f"{r['token_ratio_vs_text']:.3f}" if r.get("token_ratio_vs_text") else "—")
            + f" | {r['unreadable']} | {r.get('verdict', 'baseline')} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "report"])
    ap.add_argument("--out", required=True, help="run directory (plan.json, frames/, records.jsonl)")
    ap.add_argument("--cache", default=str(fixture.DEFAULT_CACHE))
    ap.add_argument("--download", action="store_true", help="fetch SQuAD/fonts if absent (hash-checked)")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--chunk-chars", type=int, default=fixture.CHUNK_CHARS)
    ap.add_argument("--qpc", type=int, default=30, help="questions per chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-chunks", type=int, default=0, help="0 = every full chunk of SQuAD dev")
    ap.add_argument("--tokenizer", default="/mnt/raid0/llm/hf-models/Qwen3-4B-Instruct-2507/tokenizer.json",
                    help="plan-time text-token ESTIMATE only (Qwen3 BPE); '' to skip")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--expect-build", default=EXPECTED_BUILD)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="run despite identity problems (run is VOID)")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
