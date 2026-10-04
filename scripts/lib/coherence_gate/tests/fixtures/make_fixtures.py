"""Regenerate the vendored test fixtures (one-off; provenance record). No inference.

The degeneracy tests were written (2026-10-04, q38t7-rescore) against files under /mnt/raid0/llm/tmp
and /workspace/wiki, which are not stable test inputs. This script copies the minimum into
fixtures/ so the tests run from the repo alone (the tokenizer is the only external input):

  reasoning_traces.json   8 Qwen-27B reasoning traces (>= 1500 tokens), cut to their first 1700 tokens.
  prose_excerpt.md        the head of wiki/benchmark-methodology.md (>= 1600 tokens, >= 1300 words).
  q38t7_stored_stats.json the stored v1 `coherence` stats of Q38-T7 run 20261004T025247Z
                          (phase B/C long generations + phase A gsm8k rows).
  v1_golden.json          lib_gpublock.classify (the frozen v1) on the v1 continuity samples.

  python3 make_fixtures.py   (needs the `tokenizers` package and the source paths below)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json"
RAW = Path("/mnt/raid0/llm/tmp/standalone-v3-probe.UhfAMX/artifacts/architect-27b-finetunes-v8-20260726/"
           "expanded-six-arm-v4-tail-replay-20260727/A3-tc/raw_capture.sealed.jsonl")
WIKI = Path("/workspace/wiki/benchmark-methodology.md")
GPUBLOCK = Path("/mnt/raid0/llm/tmp/gpu-block-27b-20261003")
CALLS = GPUBLOCK / "results/q38_t7/20261004T025247Z/calls.jsonl"


def main() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        sys.path.append("/mnt/raid0/llm/epyc-inference-research/.venv/lib/python3.13/site-packages")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids  # noqa: E731

    traces = []
    for line in RAW.open():
        r = json.loads(line).get("reasoning") or ""
        ids = enc(r)
        if len(ids) >= 1500:
            traces.append(tok.decode(ids[:1700]))
        if len(traces) >= 8:
            break
    (HERE / "reasoning_traces.json").write_text(json.dumps({"source": str(RAW), "cut_tokens": 1700,
                                                            "traces": traces}, indent=0) + "\n")

    wiki = WIKI.read_text()
    cut = 6000
    while len(enc(wiki[:cut])) < 1600 or len(wiki[:cut].split()) < 1300:
        cut += 1000
    cut = wiki.find("\n", cut) + 1 or cut
    (HERE / "prose_excerpt.md").write_text(wiki[:cut])

    rows = [json.loads(x) for x in CALLS.open()]
    keep = [{"phase": r["phase"], "id": r.get("id"), "arm": r.get("arm"), "coherence": r["coherence"]}
            for r in rows if "coherence" in r and (r["phase"] in ("B", "C") or
                                                   (r["phase"] == "A" and str(r.get("id")).startswith("gsm8k_00")))]
    (HERE / "q38t7_stored_stats.json").write_text(json.dumps({"source": str(CALLS), "rows": keep}, indent=0) + "\n")

    sys.path.insert(0, str(GPUBLOCK))
    import lib_gpublock as L
    prose = (HERE / "prose_excerpt.md").read_text()

    def cut_ids(text: str, n: int):
        ids = enc(text)[:n]
        return tok.decode(ids), ids
    samples = {"prose1500": cut_ids(prose, 1500), "trace1000": cut_ids(traces[0], 1000),
               "ok30": cut_ids("ok " * 30, 30), "E": ("E", enc("E"))}
    golden = {name: {fin: L.classify(text, ids, fin) for fin in ("stop", "length")}
              for name, (text, ids) in samples.items()}
    (HERE / "v1_golden.json").write_text(json.dumps({"source": str(GPUBLOCK / "lib_gpublock.py"),
                                                     "golden": golden}, indent=1) + "\n")


if __name__ == "__main__":
    main()
