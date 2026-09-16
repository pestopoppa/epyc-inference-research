#!/usr/bin/env python3
"""M-12 B5: prove prefix reuse (context-checkpoint restore) works on a RUNNING long-context server.

This script sends a handful of requests to a server the operator has already
launched. It starts, stops and kills nothing. Read the verdict from the last
line: ``SMOKE_PREFIX_REUSE: PASS`` or ``SMOKE_PREFIX_REUSE: FAIL (...)``.

Why this test exists
--------------------
Qwen3.6/3.8 are hybrid models: their recurrent (GDN) state cannot be rolled back
to an arbitrary prefix, so the server reuses a prefix only by restoring a context
checkpoint. The champion (``ef81196d5``, ``tools/server/server-context.cpp``) makes
three kinds of checkpoint:

* at user-message starts;
* ``4 + n_ubatch`` tokens before the prompt end;
* 4 tokens before the prompt end.

An M-12 prompt is one user message: a ~105K-token book (or a ~192K-token BEAM
transcript) followed by a short question. The ``end - (4 + n_ubatch)``
checkpoint therefore falls inside the shared prefix. The next question should
restore it and re-decode about 2.1K tokens, not the whole ~105K. Without that,
the Tulving full arm costs about 18 h instead of about 1 h. That behaviour was
read from the source code, never observed; this script observes it.

The verdict is decided from the server's own accounting, not from wall-clock
time:

* ``timings.cache_n`` is the number of prompt tokens reused
  (``slot.n_prompt_tokens_cache = n_past`` after the restore);
* ``timings.prompt_n`` is the number of tokens actually processed.

Wall-clock time is reported but only corroborates the verdict.

Legs (``--legs``)
-----------------
* ``tulving`` (default): cold book+Q1, then book+Q2, which must reuse. Then
  book+Q2 again, whose answer must match (a mismatch is a WARN: batch-split
  numerics can differ, but it is worth a look).
* ``beam`` (np >= 2 recommended): cold transcript+Q1, then transcript+Q2,
  which must reuse. Then book+Q3 must still reuse, which proves a second
  prefix did not evict the first. With np=1 that last check passes only if
  ``--cache-ram`` restored the book.

Timing at the champion's estimated prefill rates (bench-class planning numbers,
not measurements):

* 35B: tulving leg ~1.7 min, beam leg ~4.2 min.
* 27B: tulving leg ~4.5 min; do not run the beam leg (it needs ~10 min cold).

Requests go to ``/v1/chat/completions`` with ``temperature 0``, ``max_tokens``
16, ``chat_template_kwargs.enable_thinking=false`` and ``cache_prompt=true``.
These are the M-12 run parameters (B3).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: A prompt counts as "reused" when at least this share of its tokens came from cache...
REUSE_MIN_SHARE = 0.95
#: ...and the tokens actually processed stay within one checkpoint stride plus the suffix.
REUSE_SLACK_TOKENS = 2048
#: The warm prefill must be at most this fraction of the cold one (corroboration only).
WARM_TIME_MAX_RATIO = 0.25
#: A cold request must process at least this share of its prompt (otherwise it was warm).
COLD_MIN_SHARE = 0.90
MI210_VRAM_MIN_TOTAL = 60 * 1024**3

TULVING_QUESTIONS = (
    "List all locations where events took place in chapter order.",
    "List all dates on which events happened.",
    "List all entity/person names that appear in the book.",
)
INSTRUCTION = "Answer the question precisely. List items one per line starting with '- '. If none, say 'None'."


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get(url: str, timeout: float = 10) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _vram_card() -> Path | None:
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        try:
            if int((card / "mem_info_vram_total").read_text()) >= MI210_VRAM_MIN_TOTAL:
                return card
        except (OSError, ValueError):
            continue
    return None


class VramSampler:
    """Samples the MI210's used VRAM from sysfs DURING a request (read-only)."""

    def __init__(self, interval: float = 0.5):
        self.card = _vram_card()
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.interval = interval

    def _run(self):
        while not self._stop.is_set() and self.card is not None:
            try:
                self.samples.append(int((self.card / "mem_info_vram_used").read_text()))
            except (OSError, ValueError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def tulving_prompts(chapters: int = 200) -> tuple[str, list[str]]:
    """The real M-12a full-arm prompt shape over the real book, without pandas."""
    from tulving_episodic_adapter import (
        _DEFAULT_DATA_DIR, _DEFAULT_VARIANT, TulvingEpisodicAdapter, full_book_prompt)

    adapter = TulvingEpisodicAdapter(chapters=chapters, context_mode="none")
    variant_dir = _DEFAULT_DATA_DIR / _DEFAULT_VARIANT
    files = adapter._select_target_qa_files(sorted(variant_dir.rglob("df_qa.parquet")))
    if not files:
        raise SystemExit(f"no {chapters}ch book under {variant_dir}")
    book = adapter._load_book_text(files[0].parent)
    if not book:
        raise SystemExit(f"book text missing next to {files[0]}")
    tag = f"{chapters}ch ({adapter._chapter_count(files[0])} chapters on disk)"
    return tag, [full_book_prompt(book, f"{q}\n\n{INSTRUCTION}") for q in TULVING_QUESTIONS]


def beam_prompts() -> tuple[str, list[str]]:
    """Two questions over the LONGEST BEAM 100K conversation (the vanilla arm's prompt)."""
    from long_context_adapters import BEAMAdapter

    adapter = BEAMAdapter(split="100K")
    items = adapter.extract_all()
    if not items:
        raise SystemExit("no BEAM 100K data loaded")
    longest = max(items, key=lambda p: p["metadata"]["context_length_chars"])
    conv = longest["metadata"]["conversation_id"]
    same = [p["prompt"] for p in items if p["metadata"]["conversation_id"] == conv][:2]
    return f"conversation {conv}", same


def ask(base: str, prompt: str, *, max_tokens: int, timeout: float) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    t0 = time.monotonic()
    with VramSampler() as vram:
        data = _post(f"{base}/v1/chat/completions", payload, timeout)
    wall = time.monotonic() - t0
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "cache_n": timings.get("cache_n"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "finish_reason": choice.get("finish_reason"),
        "content": (choice.get("message") or {}).get("content", ""),
        "vram_max_bytes": max(vram.samples) if vram.samples else None,
    }


def judge_cold(name: str, r: dict, fails: list, warns: list) -> None:
    total, processed = r["prompt_tokens"] or 0, r["prompt_n"] or 0
    if total and processed < COLD_MIN_SHARE * total:
        warns.append(f"{name}: was not cold (processed {processed}/{total}); "
                     "the slot already held this prefix, so the warm check is weaker")


def judge_warm(name: str, r: dict, cold: dict | None, fails: list, warns: list) -> None:
    total, cached, processed = r["prompt_tokens"], r["cache_n"], r["prompt_n"]
    if total is None or cached is None or processed is None:
        fails.append(f"{name}: server returned no timings/usage; cannot verify reuse")
        return
    if cached < REUSE_MIN_SHARE * total:
        fails.append(f"{name}: reused {cached}/{total} tokens (< {REUSE_MIN_SHARE:.0%}); "
                     "checkpoint NOT restored")
    if processed > r["_stride"] + REUSE_SLACK_TOKENS:
        fails.append(f"{name}: processed {processed} tokens (> {r['_stride']} + "
                     f"{REUSE_SLACK_TOKENS}); prefix was re-decoded")
    if cold and cold.get("prompt_ms") and r.get("prompt_ms") is not None:
        if r["prompt_ms"] > WARM_TIME_MAX_RATIO * cold["prompt_ms"]:
            warns.append(f"{name}: warm prefill {r['prompt_ms']:.0f} ms is > "
                         f"{WARM_TIME_MAX_RATIO:.0%} of cold {cold['prompt_ms']:.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--expect-model", default=None,
                        help="Substring the served model path must contain")
    parser.add_argument("--expect-slot-ctx", type=int, default=196608)
    parser.add_argument("--ubatch", type=int, default=2048,
                        help="The server's -ub; the checkpoint stride is 4 + ubatch")
    parser.add_argument("--legs", default="tulving", help="Comma list: tulving,beam")
    parser.add_argument("--chapters", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1500)
    parser.add_argument("--out", type=Path, default=None, help="Receipt JSON path")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    stride = 4 + args.ubatch
    fails: list[str] = []
    warns: list[str] = []
    receipt: dict = {"schema": "epyc.m12.smoke_prefix_reuse.v1",
                     "started_at": datetime.now(timezone.utc).isoformat(),
                     "base": base, "thresholds": {
                         "reuse_min_share": REUSE_MIN_SHARE, "stride": stride,
                         "slack_tokens": REUSE_SLACK_TOKENS,
                         "warm_time_max_ratio": WARM_TIME_MAX_RATIO}}

    def finish() -> int:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        receipt["fails"], receipt["warns"] = fails, warns
        verdict = "PASS" if not fails else "FAIL"
        receipt["verdict"] = verdict
        out = args.out or Path("/mnt/raid0/llm/tmp/m12-smoke") / (
            f"smoke_prefix_reuse_{args.port}_{datetime.now():%Y%m%dT%H%M%S}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        for w in warns:
            print(f"WARN: {w}")
        print(f"receipt: {out}")
        line = f"SMOKE_PREFIX_REUSE: {verdict}"
        if fails:
            line += " (" + "; ".join(fails) + ")"
        print(line)
        return 0 if not fails else 1

    # ── preflight: the server is up, is the expected model, and has the expected slots ──
    try:
        _get(f"{base}/health")
        props = _get(f"{base}/props")
        slots = _get(f"{base}/slots")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        fails.append(f"server at {base} not reachable or /slots disabled: {exc}")
        return finish()
    model_path = str((props or {}).get("model_path", ""))
    receipt["model_path"] = model_path
    receipt["build_info"] = (props or {}).get("build_info")
    receipt["slots"] = [{"id": s.get("id"), "n_ctx": s.get("n_ctx"),
                         "speculative": s.get("speculative")} for s in slots]
    n_slots = len(slots)
    if args.expect_model and args.expect_model not in model_path:
        fails.append(f"served model {model_path!r} does not contain {args.expect_model!r}")
    short = [s["n_ctx"] for s in slots if (s.get("n_ctx") or 0) < args.expect_slot_ctx]
    if short:
        fails.append(f"slot n_ctx {short} < {args.expect_slot_ctx}; the recipe did not "
                     "give each slot the long context")
    if fails:
        return finish()
    print(f"server {base}: {model_path} | {n_slots} slot(s) | n_ctx/slot "
          f"{sorted({s.get('n_ctx') for s in slots})}")

    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    results: dict = {}
    book_q3: str | None = None
    receipt["legs"] = results
    try:
        if "tulving" in legs:
            tag, prompts = tulving_prompts(args.chapters)
            print(f"[tulving] book {tag}: A cold ...", flush=True)
            a = ask(base, prompts[0], max_tokens=args.max_tokens, timeout=args.timeout)
            print(f"  A {a['prompt_tokens']} tok, processed {a['prompt_n']}, "
                  f"{a['prompt_ms'] or 0:.0f} ms, vram_max {a['vram_max_bytes']}", flush=True)
            b = ask(base, prompts[1], max_tokens=args.max_tokens, timeout=args.timeout)
            b["_stride"] = stride
            print(f"  B reused {b['cache_n']}/{b['prompt_tokens']}, processed {b['prompt_n']}, "
                  f"{b['prompt_ms'] or 0:.0f} ms", flush=True)
            c = ask(base, prompts[1], max_tokens=args.max_tokens, timeout=args.timeout)
            c["_stride"] = stride
            print(f"  C reused {c['cache_n']}/{c['prompt_tokens']}, processed {c['prompt_n']}",
                  flush=True)
            judge_cold("tulving A", a, fails, warns)
            judge_warm("tulving B", b, a, fails, warns)
            judge_warm("tulving C", c, a, fails, warns)
            if b["content"] != c["content"]:
                warns.append("tulving: repeated prompt gave a different answer after restore")
            if a["vram_max_bytes"] is None:
                warns.append("could not sample MI210 VRAM (no sysfs card with >=60 GiB)")
            results["tulving"] = {"book": tag, "A": a, "B": b, "C": c}
            book_q3 = prompts[2]

        if "beam" in legs:
            if n_slots < 2:
                warns.append("beam leg with 1 slot: the book check relies on --cache-ram")
            tag, prompts = beam_prompts()
            print(f"[beam] {tag}: D cold ...", flush=True)
            d = ask(base, prompts[0], max_tokens=args.max_tokens, timeout=args.timeout)
            print(f"  D {d['prompt_tokens']} tok, processed {d['prompt_n']}, "
                  f"{d['prompt_ms'] or 0:.0f} ms", flush=True)
            e = ask(base, prompts[1], max_tokens=args.max_tokens, timeout=args.timeout)
            e["_stride"] = stride
            print(f"  E reused {e['cache_n']}/{e['prompt_tokens']}, processed {e['prompt_n']}",
                  flush=True)
            judge_cold("beam D", d, fails, warns)
            judge_warm("beam E", e, d, fails, warns)
            results["beam"] = {"conversation": tag, "D": d, "E": e}
            if book_q3 is not None:
                f = ask(base, book_q3, max_tokens=args.max_tokens, timeout=args.timeout)
                f["_stride"] = stride
                print(f"  F (book again) reused {f['cache_n']}/{f['prompt_tokens']}, "
                      f"processed {f['prompt_n']}", flush=True)
                judge_warm("beam F (book after transcript)", f, results["tulving"]["A"],
                           fails, warns)
                results["beam"]["F"] = f
    except urllib.error.HTTPError as exc:
        fails.append(f"HTTP {exc.code}: {exc.read()[:300]!r}")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        fails.append(f"request failed: {exc}")
    if not results and not fails:
        fails.append(f"no legs ran (--legs {args.legs!r})")
    try:
        receipt["slots_after"] = _get(f"{base}/slots")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        receipt["slots_after_error"] = str(exc)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
