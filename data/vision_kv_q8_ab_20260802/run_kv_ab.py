#!/usr/bin/env python3
"""Paired A/B: does q8_0 KV-cache quantisation degrade Qwen3-VL-30B-A3B vision quality?

  ONE QUESTION.  Same weights, same mmproj, same 250 MMMU-val questions, same
  order, same sampler, same prompt bytes.  The only difference between the two
  arms is `-ctk q8_0 -ctv q8_0` on the SERVER.  Everything else is held fixed
  and most of it is verified programmatically rather than asserted in prose.

  THIS SCRIPT NEVER LAUNCHES ANYTHING.  It has no process-spawning code path.
  It talks HTTP to a server the OPERATOR has already started, and it refuses to
  touch the production port (8086) unless explicitly overridden.  The two arms
  MUST be run sequentially: the MI210 cannot hold two instances of this model.

  `run_kv_ab.py runbook` prints the operator's exact command lines, a
  programmatic diff proving the only argv delta is the KV flags, and the VRAM
  predictions.  `run_kv_ab.py selftest` validates every non-inference path
  offline (question loading, prompt construction, request-body identity across
  arms, scoring, McNemar, the decision rule) and performs zero HTTP.

WHAT IS HELD FIXED, AND WHY EACH ONE MATTERS
  weights + mmproj      A different GGUF is a different model; the question is
                        about the cache, not the model.
  the 250 questions,    Reused verbatim from the cutover instrument
  their order, and      (vision_final.py + mmmu_manifest.json, both hash-pinned
  the prompt bytes      below).  A different instrument is not comparable to the
                        159/250 in the registry, and comparability is the point.
                        Order is fixed because the server keeps one slot; a
                        different order is a different sequence of slot states.
  sampler               temperature 0 (greedy) + seed 42 + a single generous
                        max_tokens for BOTH arms.  See SAMPLING below.
  max_tokens = 2048     One flat, generous budget instead of the precedent's
                        512-then-rescue-at-2048 two-pass.  Truncation is a
                        confound: if one arm rambles slightly longer it loses
                        answers to the cap, which is a budget artifact, not a
                        vision result.  A flat budget makes the rule symmetric
                        by construction and removes the second server boot.
  --image-min-tokens    Upstream warns Qwen-VL needs >=1024 image tokens.  It is
  1024                  the production flag and it pins the image-token floor so
                        both arms see the identical visual input.
  -c                    Same in both arms.  The KV *allocation* differs in size
                        between arms by design; the number of cells must not.
  --cache-ram 0         Kills the server's host-side prompt-cache store.  This
  + cache_prompt:false  project has already been burned by a prompt cache: a
                        vision speed measurement was invalidated because one
                        prompt was reused and the drafter copied its own prior
                        answer.  Here the risk is subtler -- with a cache alive,
                        question k's prefill can be served from question k-1's
                        KV, so a *cached* (and under q8_0, already-quantised)
                        prefix silently substitutes for a fresh one, and the
                        substitution rate can differ between arms.  Belt and
                        braces: `--cache-ram 0` removes the external store,
                        `"cache_prompt": false` in every request forces a full
                        re-prefill per question.  Cross-ARM sharing is
                        structurally impossible anyway -- the arms are separate
                        processes with separate KV allocations, run minutes
                        apart -- but per-QUESTION sharing is not, so we kill it.
  one request at a time No concurrency: two in-flight requests share a slot and
                        a batch, and batch composition changes numerics.

SAMPLING -- a deliberate, documented deviation from the precedent
  The cutover ran temperature 0.2 / seed 42 (the production serving temp).  This
  A/B runs temperature 0.0 (greedy) by default.  Reason: with seed pinned the
  sampler is a deterministic function of the logits either way, but a nonzero
  temperature AMPLIFIES tiny logit perturbations into answer flips.  q8_0 KV
  perturbs logits slightly by construction, so temp 0.2 would manufacture
  discordant pairs that are sampler amplification rather than degraded vision.
  Discordant pairs are the entire denominator of the paired test: inflating them
  destroys power (see `selftest`'s power table -- at 16 pp of symmetric noise the
  test cannot see a 3 pp true effect at all).  Greedy answers the sharper
  question "does the model's preferred answer change".  The deviation is applied
  to BOTH arms, so it cancels in the paired contrast; it only means the f16
  arm's ABSOLUTE score need not land exactly on 63.6%.  Pass
  `--temperature 0.2` to mirror the precedent exactly if that is preferred --
  but then expect a wider CI and read the power note before believing a null.

Instrument provenance (hash-pinned, see INSTRUMENT_SHA256):
  questions  /mnt/raid0/llm/tmp/mmmu_manifest.json
  scorer     /mnt/raid0/llm/tmp/vision_final.py  (build_prompt + extract_letter
             imported directly, so the scoring code is the SAME code, not a
             re-implementation of it)
  precedent  /mnt/raid0/llm/tmp/vision_final_results.json -> 159/250 (63.6%)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Instrument provenance.  Import the SCORER ITSELF from the cutover harness so
# the two measurements cannot silently diverge, and hash-pin it so an edit to
# that file is a loud failure rather than a quiet change of instrument.
# --------------------------------------------------------------------------- #
PRECEDENT_DIR = "/mnt/raid0/llm/tmp"
PRECEDENT_HARNESS = f"{PRECEDENT_DIR}/vision_final.py"
MANIFEST = f"{PRECEDENT_DIR}/mmmu_manifest.json"
PRECEDENT_RESULTS = f"{PRECEDENT_DIR}/vision_final_results.json"

INSTRUMENT_SHA256 = {
    PRECEDENT_HARNESS: "1334461599c6fa2a24a1741f58b43b396239df66b358788d128b10056d1017cd",
    MANIFEST: "a1ccf9ab97dbe4c0ce24c47c78e4df07f9ed98c3f68767345bf7424c4fe07b32",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_instrument(allow_drift: bool = False) -> dict:
    """Verify the pinned instrument files, byte for byte.

    Integrity, not presence: it is not enough that vision_final.py EXISTS, the
    scoring code has to be the code that produced 159/250.  Drift is fatal by
    default -- an instrument that changed underneath us produces numbers that
    look comparable and are not.
    """
    seen = {}
    drift = []
    for path, want in INSTRUMENT_SHA256.items():
        if not os.path.exists(path):
            drift.append(f"{path}: MISSING")
            continue
        got = sha256_file(path)
        seen[path] = got
        if got != want:
            drift.append(f"{path}: sha256 {got} != pinned {want}")
    if drift and not allow_drift:
        raise SystemExit(
            "INSTRUMENT DRIFT -- refusing to run.\n  "
            + "\n  ".join(drift)
            + "\nThe cutover number 159/250 was produced by the pinned bytes. If the\n"
            "change is intended, re-pin INSTRUMENT_SHA256 and say so in the report;\n"
            "or pass --allow-instrument-drift to proceed with `instrument_verified:false`."
        )
    return {"sha256": seen, "drift": drift, "verified": not drift}


sys.path.insert(0, PRECEDENT_DIR)
from vision_final import INSTR, LETTERS, build_prompt, extract_letter, norm  # noqa: E402

# vision_final imports subprocess/signal for its OWN launcher.  We take only its
# pure functions.  Nothing with a process in it may leak into this namespace.
for _forbidden in ("kill_and_verify", "wait_ready", "vram_mb", "main", "ask", "boot"):
    assert _forbidden not in globals(), f"process-touching symbol {_forbidden} leaked in"

# --------------------------------------------------------------------------- #
# The system under test.  Mirrors the LIVE worker_vision launch (read from
# /proc/<pid>/cmdline on 2026-08-02) and stack_priors.yaml
# roles.worker_vision.serving.launch.runtime.
# --------------------------------------------------------------------------- #
MODEL_DIR = "/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF"
MODEL = f"{MODEL_DIR}/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
MMPROJ = f"{MODEL_DIR}/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"

BIN_DIR = "/mnt/raid0/llm/llama.cpp/build-hip/bin"          # production-consolidated-v8, FROZEN
LLAMA_SERVER = f"{BIN_DIR}/llama-server"
LD_LIBRARY_PATH = f"{BIN_DIR}:/opt/rocm/lib"                 # build-hip MUST lead: three ggml
                                                             # generations live on this host.
HOST_CPUS = "184-191"                                        # GPU host threads = SMT siblings
SERVER_ENV = {                                               # exactly the live process's env
    "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
    "GGML_IQK": "1",
    "OMP_DYNAMIC": "false",
    "OMP_PLACES": "cores",
    "OMP_PROC_BIND": "spread",
    "OMP_WAIT_POLICY": "active",
    "ROCM_PATH": "/opt/rocm",
    "HIP_PATH": "/opt/rocm",
}

PRODUCTION_PORT = 8086          # the LIVE worker_vision / vision_escalation server
BENCH_PORT = 8087               # the A/B port -- deliberately NOT 8086
DEFAULT_CTX = 65536             # the context the q8_0 decision is actually about
ARMS = ("f16", "q8_0")

# ---- sampler / budget (identical in both arms) ---------------------------- #
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 2048
REQUEST_TIMEOUT_S = 1200

# ---- model geometry, read from the GGUF header 2026-08-02 ----------------- #
N_LAYER = 48            # qwen3vlmoe.block_count
N_HEAD_KV = 4           # qwen3vlmoe.attention.head_count_kv
HEAD_DIM_K = 128        # qwen3vlmoe.attention.key_length
HEAD_DIM_V = 128        # qwen3vlmoe.attention.value_length
KV_PAD = 256            # FATTN_KQ_STRIDE; llama.cpp pads the cache to this with FA on

# ggml on-disk element cost.  q8_0 is NOT 1 byte/element: a block is 32 values +
# one f16 scale = 34 bytes, i.e. 1.0625 B/elem.  Ignoring the scale under-counts
# the q8_0 cache by 6.25%.
BYTES_PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34.0 / 32.0,
                  "q5_1": 24.0 / 32.0, "q5_0": 22.0 / 32.0, "q4_1": 20.0 / 32.0,
                  "q4_0": 18.0 / 32.0}

# Calibration anchor: the 2026-07-31 cutover measured 21049 MiB for this model at
# -c 16384 with f16 KV.  weights + mmproj + KV accounts for 20266 MiB of that, so
# the fixed backend/graph overhead is the remainder.
MEASURED_TOTAL_MIB_AT_16384_F16 = 21049
MIB = 1024.0 * 1024.0

# --------------------------------------------------------------------------- #
# PRE-REGISTERED decision rule.  Fixed here, in code, before any data exists.
# `compare` mechanically applies it; it does not get re-negotiated afterwards.
# --------------------------------------------------------------------------- #
PREREG = {
    "primary_test": "two-sided exact McNemar on the 250 paired correctness outcomes",
    "alpha": 0.05,
    # Non-inferiority margin on delta = (q8_0 - f16) accuracy, in percentage
    # points.  3.0 pp because (a) the model was adopted for +11.2 pp over the
    # incumbent and giving back a quarter of that for 2.8 GiB is not "free", and
    # (b) 3 pp is honestly near the resolution limit of n=250 -- see `selftest`.
    "ni_margin_pp": 3.0,
    # Instrument sanity band for the f16 arm: the Wilson 95% CI of the registry's
    # 159/250.  Outside it, something about the rig changed and the run is void,
    # not "interesting".
    "instrument_band_pct": (57.5, 69.3),
    "precedent_raw": "159/250",
    "decision": {
        "ADOPT_Q8_0": "McNemar p >= alpha AND exact 95% CI lower bound on delta > -3.0 pp "
                      "AND no scorer/truncation asymmetry AND run not void",
        "KEEP_F16": "McNemar p < alpha with f16 ahead, OR point estimate delta <= -3.0 pp",
        "INCONCLUSIVE": "anything else -- notably p >= alpha with a CI that still "
                        "admits a worse-than-3-pp drop. This is NOT evidence of no harm.",
        "VOID": "instrument drift, unverified arm identity, transport errors, "
                "prompt-bytes mismatch between arms, or f16 arm outside the sanity band",
    },
}

_HTTP_CALLS = 0     # selftest asserts this stays 0


# ========================================================================== #
# Server command lines.  BUILT AND PRINTED ONLY.  Never executed from here.
# ========================================================================== #
def server_argv(arm: str, ctx: int = DEFAULT_CTX, port: int = BENCH_PORT,
                jinja: bool = False) -> list[str]:
    """The exact argv the OPERATOR runs for one arm.

    Built from one template so the two arms cannot drift by accident; `runbook`
    diffs the two lists and asserts the only delta is the KV flags.

    `jinja` defaults OFF because the LIVE worker_vision server does not pass
    --jinja and this decision is about production. The cutover harness DID pass
    it, so `--jinja` is offered for exact instrument replication instead; either
    way it is applied to both arms and cancels in the paired contrast.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    argv = [
        "taskset", "-c", HOST_CPUS, LLAMA_SERVER,
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "--host", "127.0.0.1", "--port", str(port),
        "-np", "1",
        "-c", str(ctx),
        "-t", "8",
        "--flash-attn", "on",
        "--device", "ROCm0",
        "-ngl", "999",
        "--image-min-tokens", "1024",
        "--cache-ram", "0",
        "--log-colors", "off",
    ]
    if jinja:
        argv.append("--jinja")
    if arm == "q8_0":
        argv += ["-ctk", "q8_0", "-ctv", "q8_0"]
    return argv


def flag_value(argv: list[str], flag: str) -> str | None:
    """Value of `flag` as passed TO LLAMA-SERVER.

    Must skip the taskset prefix: `taskset -c 184-191` and `llama-server -c 65536`
    both spell `-c`, and a naive argv.index("-c") silently reads the CPU list --
    a check that passes for the wrong reason is worse than no check.
    """
    start = argv.index(LLAMA_SERVER) if LLAMA_SERVER in argv else 0
    for i in range(start, len(argv) - 1):
        if argv[i] == flag:
            return argv[i + 1]
    return None


def argv_diff(a: list[str], b: list[str]) -> dict:
    """Longest-common-prefix/suffix diff of two argv lists."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return {
        "common_prefix_len": i,
        "common_suffix_len": j,
        "only_in_a": a[i:len(a) - j],
        "only_in_b": b[i:len(b) - j],
    }


# ========================================================================== #
# VRAM model
# ========================================================================== #
def kv_bytes(ctx: int, kv_type: str) -> int:
    cells = int(math.ceil(ctx / KV_PAD) * KV_PAD)
    per_layer = (HEAD_DIM_K + HEAD_DIM_V) * N_HEAD_KV * BYTES_PER_ELEM[kv_type]
    return int(round(per_layer * N_LAYER * cells))


def fa_dequant_scratch_bytes(ctx: int, kv_type: str) -> int:
    """f16 scratch the HIP FlashAttention path needs when the cache is quantised.

    On gfx90a (CDNA2, MFMA) with head_dim 128 and gqa_ratio 8, prefill lands on
    BEST_FATTN_KERNEL_MMA_F16, which requires f16 K and V; ggml therefore
    dequantises the K and V views into scratch appended to the FA node's
    allocation (ggml_cuda_flash_attn_ext_get_alloc_size -> need_f16_K/V).  The
    worst-case graph reserve sizes those views at the FULL cache, so a quantised
    cache buys back somewhat less VRAM than the raw cache-size delta suggests.
    Returns the size of ONE live K+V scratch pair; the graph allocator may keep
    1-2 of them alive, hence the range reported by `predict_vram`.
    """
    if kv_type == "f16":
        return 0
    cells = int(math.ceil(ctx / KV_PAD) * KV_PAD)
    return int(2 * cells * N_HEAD_KV * HEAD_DIM_K * 2)      # K and V, 2 B/elem


def predict_vram(ctx: int, kv_type: str) -> dict:
    weights = os.path.getsize(MODEL) if os.path.exists(MODEL) else 18_556_687_200
    mmproj = os.path.getsize(MMPROJ) if os.path.exists(MMPROJ) else 1_083_499_584
    kv = kv_bytes(ctx, kv_type)
    base_overhead_mib = (MEASURED_TOTAL_MIB_AT_16384_F16
                         - (weights + mmproj + kv_bytes(16384, "f16")) / MIB)
    scratch = fa_dequant_scratch_bytes(ctx, kv_type)
    lo = (weights + mmproj + kv) / MIB + base_overhead_mib + scratch / MIB
    hi = (weights + mmproj + kv) / MIB + base_overhead_mib + 2 * scratch / MIB
    return {
        "ctx": ctx, "kv_type": kv_type,
        "weights_mib": round(weights / MIB, 1),
        "mmproj_mib": round(mmproj / MIB, 1),
        "kv_mib": round(kv / MIB, 1),
        "kv_bytes_per_token": round(kv / ctx, 1),
        "fixed_overhead_mib": round(base_overhead_mib, 1),
        "fa_dequant_scratch_mib": round(scratch / MIB, 1),
        "predicted_at_load_mib": (round(lo, 0), round(hi, 0)),
        # The registry records that VRAM grows on first EXECUTION, not at load:
        # the four-model steady state ran ~1 GiB above the load-time figure.
        "predicted_after_first_request_mib": (round(lo + 1024, 0), round(hi + 1024, 0)),
    }


# ========================================================================== #
# Request construction -- identical bytes in both arms, by construction
# ========================================================================== #
@lru_cache(maxsize=1)
def load_questions() -> tuple:
    with open(MANIFEST) as fh:
        qs = json.load(fh)
    assert isinstance(qs, list) and len(qs) == 250, f"expected 250 questions, got {len(qs)}"
    idxs = [q["idx"] for q in qs]
    assert idxs == sorted(idxs), "manifest is not in ascending idx order"
    assert len(set(idxs)) == len(idxs), "duplicate idx in manifest"
    assert len(set(q["image_path"] for q in qs)) == len(qs), (
        "manifest reuses an image across questions -- a repeated image is exactly the "
        "cache-sharing failure mode this protocol has to exclude")
    return tuple(qs)


@lru_cache(maxsize=512)
def _image_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def build_request(q: dict, temperature: float, seed: int, max_tokens: int) -> dict:
    """The JSON body for one question.

    Contains NOTHING arm-specific: `selftest` proves the bodies are byte-identical
    across arms, so any measured difference has to come from the server's KV type.
    """
    return {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + _image_b64(q["image_path"])}},
            {"type": "text", "text": build_prompt(q)},
        ]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        # Inert at temperature 0 (llama.cpp takes the argmax), pinned anyway so a
        # future change to a server-side sampler default cannot differ between two
        # arms that may be run hours apart.
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        # Force a full re-prefill per question: no slot-prefix reuse, so no
        # question is ever answered off another question's (arm-dependent) KV.
        "cache_prompt": False,
        "stream": False,
    }


def request_fingerprint(body: dict) -> str:
    """Hash of the request MINUS the image blob (keeps the digest cheap) plus the
    image's own content hash."""
    shallow = json.loads(json.dumps(body))
    url = shallow["messages"][0]["content"][0]["image_url"]["url"]
    shallow["messages"][0]["content"][0]["image_url"]["url"] = hashlib.sha256(
        url.encode()).hexdigest()
    return hashlib.sha256(json.dumps(shallow, sort_keys=True).encode()).hexdigest()


# ========================================================================== #
# Arm identity verification (read-only; no process control)
# ========================================================================== #
KV_LOG_RE = (r"size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells,\s*(\d+)\s*layers"
             r".*?K\s*\(([a-z0-9_]+)\)\s*:\s*([\d.]+)\s*MiB,\s*V\s*\(([a-z0-9_]+)\)\s*:\s*([\d.]+)\s*MiB")


def parse_server_log(path: str) -> dict:
    """Pull the arm's ground truth out of the server's own startup log.

    llama.cpp prints, from llama_kv_cache:
      size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): ... , V (%s): ...
    That line is the server telling us what it actually allocated, which is a
    far better arm check than trusting the operator pasted the right flags.
    """
    import re
    out = {"log_path": path, "kv": None, "compute_buffer_mib": None, "n_ctx": None}
    with open(path, errors="ignore") as fh:
        text = fh.read()
    m = None
    for m in re.finditer(KV_LOG_RE, text, re.S):
        pass                                    # keep the LAST occurrence
    if m:
        out["kv"] = {
            "total_mib": float(m.group(1)), "cells": int(m.group(2)),
            "layers": int(m.group(3)),
            "k_type": m.group(4), "k_mib": float(m.group(5)),
            "v_type": m.group(6), "v_mib": float(m.group(7)),
        }
    mc = None
    for mc in re.finditer(r"([A-Za-z0-9]+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB", text):
        pass
    if mc:
        out["compute_buffer_mib"] = float(mc.group(2))
    mn = None
    for mn in re.finditer(r"n_ctx\s*[:=]\s*(\d+)", text):
        pass
    if mn:
        out["n_ctx"] = int(mn.group(1))
    return out


def verify_arm(arm: str, log_facts: dict, ctx: int) -> tuple[bool, list[str]]:
    want = "f16" if arm == "f16" else "q8_0"
    problems = []
    kv = log_facts.get("kv")
    if not kv:
        return False, ["server log does not contain the llama_kv_cache size line"]
    if kv["k_type"] != want or kv["v_type"] != want:
        problems.append(f"log says K={kv['k_type']} V={kv['v_type']}, arm claims {want}")
    if kv["layers"] != N_LAYER:
        problems.append(f"log says {kv['layers']} layers, model has {N_LAYER}")
    if kv["cells"] != int(math.ceil(ctx / KV_PAD) * KV_PAD):
        problems.append(f"log says {kv['cells']} cells, --ctx {ctx} implies "
                        f"{int(math.ceil(ctx / KV_PAD) * KV_PAD)}")
    predicted = kv_bytes(ctx, want) / MIB
    if abs(kv["total_mib"] - predicted) / max(predicted, 1) > 0.02:
        problems.append(f"log KV {kv['total_mib']:.1f} MiB vs predicted {predicted:.1f} MiB "
                        f"(>2% off -- the VRAM model or the geometry is wrong)")
    return (not problems), problems


# ========================================================================== #
# HTTP (only ever against a server the operator already started)
# ========================================================================== #
def _post(url: str, body: dict, timeout: int = REQUEST_TIMEOUT_S) -> dict:
    global _HTTP_CALLS
    _HTTP_CALLS += 1
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str, timeout: int = 30) -> dict:
    global _HTTP_CALLS
    _HTTP_CALLS += 1
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


# ========================================================================== #
# Statistics
# ========================================================================== #
@lru_cache(maxsize=None)
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs -- byte-for-byte the
    test vision_final_analyze.py used for the cutover."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)

    def p_ge(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def p_le(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            a, b = (m, b) if p_ge(m) < alpha / 2 else (a, m)
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            a, b = (m, b) if p_le(m) > alpha / 2 else (a, m)
        hi = (a + b) / 2
    return lo, hi


def delta_ci_pp(b: int, c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact conditional 95% CI on delta = (b - c)/n, in percentage points.

    Conditional on d = b + c, b ~ Bin(d, p) and delta = (2p - 1) * d/n, so a
    Clopper-Pearson interval on p maps straight onto delta.
    """
    d = b + c
    if d == 0:
        return (0.0, 0.0)
    lo_p, hi_p = clopper_pearson(b, d, alpha)
    return (100 * (2 * lo_p - 1) * d / n, 100 * (2 * hi_p - 1) * d / n)


def wilson_pct(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def min_detectable_gap(d: int, n: int, alpha: float = 0.05) -> float | None:
    """Smallest |b-c| at this discordance that would have reached p < alpha,
    expressed in pp of n.  None if no split at this d can ever be significant."""
    best = None
    for k in range(0, d // 2 + 1):
        if mcnemar_exact(d - k, k) < alpha:
            best = k
    return None if best is None else 100 * (d - 2 * best) / n


def mcnemar_power(beta: float, gamma: float, n: int = 250, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test.

    Each item is independently: arm-B-only-correct w.p. beta, arm-A-only-correct
    w.p. gamma, concordant otherwise.  True effect = beta - gamma.
    """
    conc = 1 - beta - gamma
    assert conc >= 0
    lf = [0.0] * (n + 1)
    for i in range(1, n + 1):
        lf[i] = lf[i - 1] + math.log(i)
    lb = math.log(beta) if beta > 0 else float("-inf")
    lg = math.log(gamma) if gamma > 0 else float("-inf")
    lc = math.log(conc) if conc > 0 else float("-inf")
    tot = 0.0
    for b in range(n + 1):
        if beta == 0 and b > 0:
            break
        for c in range(n - b + 1):
            if gamma == 0 and c > 0:
                break
            if mcnemar_exact(b, c) >= alpha:
                continue
            lp = lf[n] - lf[b] - lf[c] - lf[n - b - c]
            lp += (b * lb if b else 0) + (c * lg if c else 0) + ((n - b - c) * lc if n - b - c else 0)
            if lp > -700:
                tot += math.exp(lp)
    return tot


# ========================================================================== #
# score -- one arm against an ALREADY-RUNNING server
# ========================================================================== #
def cmd_score(args) -> int:
    port = int(args.url.rstrip("/").rsplit(":", 1)[-1].split("/")[0])
    if port == PRODUCTION_PORT and not args.allow_production_port:
        raise SystemExit(
            f"REFUSING: {args.url} is the LIVE worker_vision / vision_escalation port "
            f"({PRODUCTION_PORT}). Scoring it would (a) load the production server with "
            f"250 vision requests and (b) measure whatever KV type production happens to "
            f"be running, not the arm you asked for. Start the A/B server on "
            f"{BENCH_PORT} instead, or pass --allow-production-port if you truly mean it.")

    instrument = check_instrument(args.allow_instrument_drift)
    qs = list(load_questions())
    if args.limit:
        qs = qs[:args.limit]

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, f"arm_{args.arm}")
    jsonl_path, json_path = stem + ".rows.jsonl", stem + ".json"

    # --- arm identity ------------------------------------------------------ #
    # A dry run happens BEFORE the operator starts a server, so it neither
    # demands a log nor touches the network -- that is the whole point of it.
    log_facts, arm_verified, arm_problems, props = {}, False, ["not checked"], {}
    if not args.dry_run:
        if args.server_log:
            log_facts = parse_server_log(args.server_log)
            arm_verified, arm_problems = verify_arm(args.arm, log_facts, args.ctx)
            if not arm_verified and not args.no_log_check:
                raise SystemExit(
                    "ARM IDENTITY UNVERIFIED -- refusing to run:\n  "
                    + "\n  ".join(arm_problems)
                    + "\nThe server's own llama_kv_cache line is the ground truth for "
                      "which arm is loaded. Fix the launch, or pass --no-log-check "
                      "(the result is then stamped arm_verified:false and `compare` "
                      "will void it).")
        elif not args.no_log_check:
            raise SystemExit(
                "--server-log is required: without the server's startup log there is "
                "no independent proof that this process is the arm you say it is. "
                "Pass --no-log-check to proceed unverified (compare will void it).")
        try:
            props = _get(args.url.rstrip("/") + "/props")
        except Exception as exc:                            # noqa: BLE001
            props = {"_error": f"{type(exc).__name__}: {exc}"}

    # --- request bodies (built once; fingerprints prove cross-arm identity) - #
    bodies = [build_request(q, args.temperature, args.seed, args.max_tokens) for q in qs]
    fps = [request_fingerprint(b) for b in bodies]
    suite_fp = hashlib.sha256("".join(fps).encode()).hexdigest()

    meta = {
        "arm": args.arm, "url": args.url, "ctx": args.ctx,
        "temperature": args.temperature, "seed": args.seed, "max_tokens": args.max_tokens,
        "n": len(qs), "manifest": MANIFEST,
        "instrument": instrument, "prereg": PREREG,
        "server_log_facts": log_facts, "arm_verified": arm_verified,
        "arm_problems": arm_problems if not arm_verified else [],
        "props": props,
        "request_suite_fingerprint": suite_fp,
        "expected_server_argv": server_argv(args.arm, args.ctx, port, args.jinja),
        "predicted_vram": predict_vram(args.ctx, args.arm),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        # Every non-inference path, exercised, zero packets sent.
        meta["request_fingerprints"] = dict(zip((q["idx"] for q in qs), fps))
        meta["prompt_sample"] = build_prompt(qs[0])
        with open(stem + ".dryrun.json", "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"DRY RUN: built {len(bodies)} request bodies, sent 0.")
        print(f"suite fingerprint {suite_fp}")
        print(f"wrote {stem}.dryrun.json")
        return 0

    done = {}
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done[r["idx"]] = r
        print(f"resuming: {len(done)} rows already persisted")
    elif os.path.exists(jsonl_path):
        raise SystemExit(f"{jsonl_path} exists. Move it aside, or pass --resume.")

    rows, t0 = [], time.time()
    with open(jsonl_path, "a") as sink:
        for i, (q, body, fp) in enumerate(zip(qs, bodies, fps)):
            if q["idx"] in done:
                rows.append(done[q["idx"]])
                continue
            err, attempts, got, usage, t_q = None, 0, "", {}, time.time()
            for attempt in range(3):
                attempts = attempt + 1
                try:
                    d = _post(args.url.rstrip("/") + "/v1/chat/completions", body)
                    msg = d["choices"][0]["message"]
                    got = (msg.get("content") or "").strip()
                    think = (msg.get("reasoning_content") or "").strip()
                    usage = d.get("usage", {}) or {}
                    err = None
                    break
                except Exception as exc:                    # noqa: BLE001
                    got, think, usage = "", "", {}
                    err = f"{type(exc).__name__}: {exc}"
                    # NO restart, NO process control: this harness only ever waits.
                    time.sleep(2 + 3 * attempt)
            letter, method = extract_letter(got, q["options"])
            comp = usage.get("completion_tokens")
            row = {
                "idx": q["idx"], "id": q["id"], "subject": q["subject"],
                "expected": q["answer"], "raw": got,
                "reasoning": (think[:400] if think else None),
                "pred": letter, "parse": method,
                "correct": bool(letter == q["answer"]),
                "capped": bool((comp or 0) >= args.max_tokens),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": comp,
                "latency_s": round(time.time() - t_q, 2),
                "attempts": attempts, "error": err,
                "request_fp": fp,
            }
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()                                    # every question is a drain point
            if (i + 1) % 25 == 0:
                ok = sum(r["correct"] for r in rows)
                print(f"  {args.arm} {i+1}/{len(qs)}  running {ok}/{len(rows)}  "
                      f"{time.time()-t0:.0f}s", flush=True)

    summary = summarise(rows, args.max_tokens)
    meta["elapsed_s"] = round(time.time() - t0, 1)
    out = {"_meta": meta, **summary, "rows": rows}
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n== arm {args.arm}: {summary['correct']}/{summary['n']} "
          f"({summary['pct']}%)  parse_fail={summary['parse_fail']} empty={summary['empty']} "
          f"capped={summary['capped']} errors={summary['request_errors']} "
          f"{meta['elapsed_s']:.0f}s")
    print(f"wrote {json_path}")
    if summary["request_errors"]:
        print("WARNING: transport errors present -- `compare` will void this run.")
    return 0


def summarise(rows: list[dict], max_tokens: int) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "n": n, "correct": correct, "pct": round(100 * correct / n, 1) if n else 0.0,
        "wilson95": tuple(round(x, 1) for x in wilson_pct(correct, n)),
        "parse_fail": sum(r["parse"] == "parse_fail" for r in rows),
        "empty": sum(r["parse"] == "empty" for r in rows),
        "capped": sum(bool(r.get("capped")) for r in rows),
        "request_errors": sum(bool(r.get("error")) for r in rows),
        "retries": sum(max(0, (r.get("attempts") or 1) - 1) for r in rows),
        "parse_methods": dict(Counter(r["parse"] for r in rows)),
        "median_prompt_tokens": (sorted(r.get("prompt_tokens") or 0 for r in rows)[n // 2]
                                 if n else None),
        "total_completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
    }


# ========================================================================== #
# compare -- paired statistics + the pre-registered rule
# ========================================================================== #
def paired_counts(a_rows: dict, b_rows: dict, key) -> tuple[int, int, int, int]:
    """(b, c, both, neither) where b = B-only-true, c = A-only-true, on `key`."""
    bb = cc = both = neither = 0
    for idx, ra in a_rows.items():
        rb = b_rows[idx]
        ka, kb = bool(key(ra)), bool(key(rb))
        if kb and not ka:
            bb += 1
        elif ka and not kb:
            cc += 1
        elif ka and kb:
            both += 1
        else:
            neither += 1
    return bb, cc, both, neither


def evaluate(f16: dict, q8: dict, prereg: dict = PREREG) -> dict:
    """Apply the pre-registered rule.  No judgement calls live here."""
    a_rows = {r["idx"]: r for r in f16["rows"]}
    b_rows = {r["idx"]: r for r in q8["rows"]}
    n = len(a_rows)
    void = []

    if set(a_rows) != set(b_rows):
        void.append("arms answered different question sets")
    if not f16["_meta"].get("arm_verified") or not q8["_meta"].get("arm_verified"):
        void.append("arm identity unverified for at least one arm")
    if not f16["_meta"]["instrument"]["verified"] or not q8["_meta"]["instrument"]["verified"]:
        void.append("instrument (question set / scorer) drifted from the pinned bytes")
    if f16["_meta"]["request_suite_fingerprint"] != q8["_meta"]["request_suite_fingerprint"]:
        void.append("the two arms did not send identical request bodies")
    for name, arm in (("f16", f16), ("q8_0", q8)):
        if arm["request_errors"]:
            void.append(f"{name} arm had {arm['request_errors']} transport errors")
    for fixed in ("temperature", "seed", "max_tokens", "ctx"):
        if f16["_meta"][fixed] != q8["_meta"][fixed]:
            void.append(f"arms differ in {fixed}: "
                        f"{f16['_meta'][fixed]} vs {q8['_meta'][fixed]}")
    lo_band, hi_band = prereg["instrument_band_pct"]
    if not (lo_band <= f16["pct"] <= hi_band):
        void.append(f"f16 arm {f16['pct']}% is outside the instrument sanity band "
                    f"{lo_band}-{hi_band}% (Wilson CI of the registry's "
                    f"{prereg['precedent_raw']}) -- something about the rig changed")

    b, c, both, neither = paired_counts(a_rows, b_rows, lambda r: r["correct"])
    d = b + c
    p = mcnemar_exact(b, c)
    delta_pp = 100 * (b - c) / n
    ci_lo, ci_hi = delta_ci_pp(b, c, n)

    # Failure-reason asymmetries.  A cross-arm parse-failure gap is a SCORER
    # artifact, not a vision result -- this project has been burned by exactly
    # that -- and a truncation gap is a token-budget artifact.
    pf_b, pf_c, _, _ = paired_counts(a_rows, b_rows, lambda r: r["parse"] == "parse_fail")
    cap_b, cap_c, _, _ = paired_counts(a_rows, b_rows, lambda r: bool(r.get("capped")))
    p_parse = mcnemar_exact(pf_b, pf_c)
    p_cap = mcnemar_exact(cap_b, cap_c)
    artifacts = []
    if p_parse < prereg["alpha"]:
        artifacts.append(f"parse-failure asymmetry (q8_0-only {pf_b} vs f16-only {pf_c}, "
                         f"p={p_parse:.4f}) -- rescore offline before reading the quality result")
    if p_cap < prereg["alpha"]:
        artifacts.append(f"truncation asymmetry (q8_0-only {cap_b} vs f16-only {cap_c}, "
                         f"p={p_cap:.4f}) -- the token budget is deciding, not the cache")

    agree = sum(1 for i in a_rows if a_rows[i]["pred"] == b_rows[i]["pred"])

    if void:
        verdict = "VOID"
    elif artifacts:
        verdict = "INCONCLUSIVE (scorer/budget artifact)"
    elif (p < prereg["alpha"] and c > b) or delta_pp <= -prereg["ni_margin_pp"]:
        verdict = "KEEP f16"
    elif p >= prereg["alpha"] and ci_lo > -prereg["ni_margin_pp"]:
        verdict = "ADOPT q8_0"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n": n, "f16_correct": f16["correct"], "q8_correct": q8["correct"],
        "f16_pct": f16["pct"], "q8_pct": q8["pct"],
        "b_q8_only_correct": b, "c_f16_only_correct": c,
        "both_correct": both, "neither_correct": neither,
        "discordant": d, "delta_pp": round(delta_pp, 2),
        "mcnemar_p": p, "delta_ci95_pp": (round(ci_lo, 2), round(ci_hi, 2)),
        "achieved_mde_pp": min_detectable_gap(d, n),
        "answer_agreement": agree, "answer_agreement_pct": round(100 * agree / n, 1),
        "parse_fail_f16": f16["parse_fail"], "parse_fail_q8": q8["parse_fail"],
        "parse_asymmetry_p": p_parse, "capped_asymmetry_p": p_cap,
        "artifacts": artifacts, "void_reasons": void, "verdict": verdict,
        "prereg": prereg,
    }


def cmd_compare(args) -> int:
    with open(args.f16) as fh:
        f16 = json.load(fh)
    with open(args.q8) as fh:
        q8 = json.load(fh)
    res = evaluate(f16, q8)
    print_comparison(res, f16, q8)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


def print_comparison(res: dict, f16: dict | None = None, q8: dict | None = None) -> None:
    print("=" * 78)
    print("KV cache f16 vs q8_0 -- Qwen3-VL-30B-A3B-Instruct Q4_K_M, MMMU-val 250 MC")
    print("=" * 78)
    print(f"  f16   {res['f16_correct']}/{res['n']}  ({res['f16_pct']}%)")
    print(f"  q8_0  {res['q8_correct']}/{res['n']}  ({res['q8_pct']}%)")
    print(f"  delta {res['delta_pp']:+.2f} pp   exact 95% CI "
          f"[{res['delta_ci95_pp'][0]:+.2f}, {res['delta_ci95_pp'][1]:+.2f}] pp")
    print()
    print("  paired 2x2 (rows f16, cols q8_0)")
    print(f"    both correct        {res['both_correct']:>4}")
    print(f"    q8_0 only correct b {res['b_q8_only_correct']:>4}")
    print(f"    f16  only correct c {res['c_f16_only_correct']:>4}")
    print(f"    neither             {res['neither_correct']:>4}")
    print(f"    discordant d = b+c  {res['discordant']:>4}")
    print(f"  exact McNemar two-sided p = {res['mcnemar_p']:.4f}")
    mde = res["achieved_mde_pp"]
    print(f"  achieved sensitivity: at d={res['discordant']} the smallest detectable "
          f"|b-c| is " + (f"{mde:.2f} pp" if mde is not None else
                          "unreachable (no split at this d can reach p<0.05)"))
    print(f"  identical predicted letter on {res['answer_agreement']}/{res['n']} "
          f"({res['answer_agreement_pct']}%)")
    print()
    print(f"  parse failures  f16 {res['parse_fail_f16']}  q8_0 {res['parse_fail_q8']}  "
          f"(asymmetry p={res['parse_asymmetry_p']:.4f})")
    print(f"  truncation asymmetry p={res['capped_asymmetry_p']:.4f}")
    for a in res["artifacts"]:
        print(f"  !! ARTIFACT: {a}")
    for v in res["void_reasons"]:
        print(f"  !! VOID: {v}")
    print()
    print(f"  PRE-REGISTERED VERDICT: {res['verdict']}")
    print(f"    (alpha={res['prereg']['alpha']}, non-inferiority margin "
          f"{res['prereg']['ni_margin_pp']} pp)")
    print("=" * 78)


# ========================================================================== #
# runbook
# ========================================================================== #
def cmd_runbook(args) -> int:
    ctx, port = args.ctx, args.port
    a = server_argv("f16", ctx, port, args.jinja)
    b = server_argv("q8_0", ctx, port, args.jinja)
    diff = argv_diff(a, b)
    env_line = " ".join(f"{k}={v}" for k, v in SERVER_ENV.items())

    print("#" * 78)
    print("# OPERATOR RUNBOOK -- KV f16 vs q8_0, Qwen3-VL-30B-A3B")
    print("#   The two arms MUST run SEQUENTIALLY. The MI210 cannot hold two")
    print("#   instances of this model, and the live worker_vision server on port")
    print(f"#   {PRODUCTION_PORT} must be stopped first -- that is the OPERATOR's call, not the")
    print("#   harness's. Stop it with orchestrator_stack.py, never with a name-pattern kill.")
    print("#" * 78)
    for name, arm, argv in (("ARM 1 - f16 KV (baseline)", "f16", a),
                            ("ARM 2 - q8_0 KV (candidate)", "q8_0", b)):
        print(f"\n## {name}")
        print(f"env {env_line} \\\n  " + " \\\n  ".join(_wrap_argv(argv))
              + f" \\\n  > /mnt/raid0/llm/tmp/vision-kv-q8/server_{arm}.log 2>&1 &")
    print("\n## programmatic argv diff (f16 -> q8_0)")
    print(json.dumps(diff, indent=1))
    only_b = diff["only_in_b"]
    ok = (diff["only_in_a"] == [] and only_b == ["-ctk", "q8_0", "-ctv", "q8_0"])
    print(f"ONLY-DELTA CHECK: {'PASS' if ok else 'FAIL'} -- "
          f"f16-only={diff['only_in_a']} q8_0-only={only_b}")

    print("\n## scoring (server already up; this script never launches anything)")
    for arm in ARMS:
        print(f"python3 {os.path.abspath(__file__)} score \\\n"
              f"  --url http://127.0.0.1:{port} --arm {arm} --ctx {ctx} \\\n"
              f"  --server-log /mnt/raid0/llm/tmp/vision-kv-q8/server_{arm}.log \\\n"
              f"  --outdir /mnt/raid0/llm/tmp/vision-kv-q8/results")
    print(f"\npython3 {os.path.abspath(__file__)} compare \\\n"
          f"  --f16 /mnt/raid0/llm/tmp/vision-kv-q8/results/arm_f16.json \\\n"
          f"  --q8  /mnt/raid0/llm/tmp/vision-kv-q8/results/arm_q8_0.json \\\n"
          f"  --out /mnt/raid0/llm/tmp/vision-kv-q8/results/verdict.json")

    print("\n## VRAM prediction -- check against `rocm-smi --showmeminfo vram` at launch")
    for arm in ARMS:
        v = predict_vram(ctx, arm)
        print(f"  {arm:<5} @ -c {ctx}: weights {v['weights_mib']:.0f} + mmproj "
              f"{v['mmproj_mib']:.0f} + KV {v['kv_mib']:.0f} + fixed "
              f"{v['fixed_overhead_mib']:.0f}" +
              (f" + FA dequant scratch {v['fa_dequant_scratch_mib']:.0f}"
               if v["fa_dequant_scratch_mib"] else "") +
              f"  =>  {v['predicted_at_load_mib'][0]:.0f}-{v['predicted_at_load_mib'][1]:.0f} MiB "
              f"at load, {v['predicted_after_first_request_mib'][0]:.0f}-"
              f"{v['predicted_after_first_request_mib'][1]:.0f} MiB after the first request")
        print(f"        KV per token: {v['kv_bytes_per_token']:.0f} B "
              f"({v['kv_bytes_per_token']/1024:.1f} KiB)")
    dv = (predict_vram(ctx, "f16")["kv_mib"] - predict_vram(ctx, "q8_0")["kv_mib"])
    print(f"  raw KV saving at -c {ctx}: {dv:.0f} MiB ({dv/1024:.2f} GiB); net saving is "
          f"smaller by the FA dequant scratch")
    print("\n  Verify from the server's own log, not just rocm-smi:")
    print('    grep -E "llama_kv_cache: *size|compute buffer size" server_<arm>.log')
    return 0


def _wrap_argv(argv: list[str]) -> list[str]:
    """Group argv into flag+value pairs for readable printing."""
    out, i = [], 0
    while i < len(argv):
        if argv[i].startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            out.append(f"{argv[i]} {argv[i+1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


# ========================================================================== #
# selftest -- every non-inference path, zero packets
# ========================================================================== #
def cmd_selftest(args) -> int:
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    print("\n--- 1. instrument integrity (hash-pinned) ---")
    inst = check_instrument(allow_drift=True)
    for path, want in INSTRUMENT_SHA256.items():
        check(os.path.basename(path), inst["sha256"].get(path) == want,
              inst["sha256"].get(path, "MISSING")[:16])

    print("\n--- 2. question set ---")
    qs = list(load_questions())
    check("250 questions", len(qs) == 250)
    check("250 distinct images", len(set(q["image_path"] for q in qs)) == 250)
    check("all images present on disk",
          all(os.path.exists(q["image_path"]) for q in qs))
    check("every answer is a valid option letter",
          all(q["answer"] in LETTERS[:len(q["options"])] for q in qs))
    check("option counts", True, str(dict(Counter(len(q["options"]) for q in qs))))

    print("\n--- 3. prompt construction ---")
    p0 = build_prompt(qs[0])
    check("deterministic", build_prompt(qs[0]) == p0)
    check("<image 1> placeholder stripped", "<image 1>" not in p0)
    check("options lettered A..", "\nA. " in p0 and "\nB. " in p0)
    check("instruction appended", p0.endswith(INSTR))
    all_prompts = [build_prompt(q) for q in qs]
    check("250 prompts built", len(all_prompts) == 250)
    check("prompt set hash stable",
          hashlib.sha256("".join(all_prompts).encode()).hexdigest()
          == hashlib.sha256("".join(build_prompt(q) for q in qs).encode()).hexdigest())
    print("      first prompt >>> " + p0.replace("\n", " | ")[:150])

    print("\n--- 4. request bodies are IDENTICAL across arms ---")
    # There is no arm parameter in build_request at all; this proves it end to end
    # on the real manifest, so the only possible difference is server-side.
    fp_a = [request_fingerprint(build_request(q, 0.0, 42, 2048)) for q in qs[:25]]
    fp_b = [request_fingerprint(build_request(q, 0.0, 42, 2048)) for q in qs[:25]]
    check("same inputs -> same bytes", fp_a == fp_b)
    check("different question -> different bytes", len(set(fp_a)) == 25)
    body = build_request(qs[0], 0.0, 42, 2048)
    check("cache_prompt disabled", body["cache_prompt"] is False)
    check("stream disabled", body["stream"] is False)
    check("temperature/seed/max_tokens pinned",
          (body["temperature"], body["seed"], body["max_tokens"]) == (0.0, 42, 2048))
    check("image is inline base64 data URI",
          body["messages"][0]["content"][0]["image_url"]["url"].startswith(
              "data:image/png;base64,"))

    print("\n--- 5. server argv: only delta is the KV flags ---")
    a, b = server_argv("f16"), server_argv("q8_0")
    d = argv_diff(a, b)
    check("f16 has nothing q8_0 lacks", d["only_in_a"] == [], str(d["only_in_a"]))
    check("q8_0 adds exactly -ctk q8_0 -ctv q8_0",
          d["only_in_b"] == ["-ctk", "q8_0", "-ctv", "q8_0"], str(d["only_in_b"]))
    check("no --cache-ram drift", a.count("--cache-ram") == b.count("--cache-ram") == 1)
    check("--cache-ram 0 in both",
          flag_value(a, "--cache-ram") == flag_value(b, "--cache-ram") == "0")
    check("same llama-server -c in both (not taskset's -c)",
          flag_value(a, "-c") == flag_value(b, "-c") == str(DEFAULT_CTX),
          f"{flag_value(a, '-c')} / {flag_value(b, '-c')}")
    check("taskset -c is the CPU list, distinct from the context flag",
          a[a.index("taskset") + 2] == HOST_CPUS)
    check("same -np, -t, -ngl, --image-min-tokens, --flash-attn in both",
          all(flag_value(a, f) == flag_value(b, f)
              for f in ("-np", "-t", "-ngl", "--image-min-tokens", "--flash-attn", "--device",
                        "-m", "--mmproj", "--port")))
    check("neither arm uses the production port",
          str(PRODUCTION_PORT) not in a and str(PRODUCTION_PORT) not in b)
    aj, bj = server_argv("f16", jinja=True), server_argv("q8_0", jinja=True)
    check("--jinja variant still has exactly the KV-flag delta",
          argv_diff(aj, bj)["only_in_b"] == ["-ctk", "q8_0", "-ctv", "q8_0"]
          and argv_diff(aj, bj)["only_in_a"] == [])
    check("--jinja is applied to both arms or neither",
          aj.count("--jinja") == bj.count("--jinja") == 1
          and a.count("--jinja") == b.count("--jinja") == 0)
    alt = server_argv("f16", 16384, 9001)
    check("ctx/port propagate into argv",
          flag_value(alt, "-c") == "16384" and flag_value(alt, "--port") == "9001",
          f"-c {flag_value(alt, '-c')} --port {flag_value(alt, '--port')}")

    print("\n--- 6. scorer: classifies by reason ---")
    opts4 = ["alpha", "beta", "gamma", "delta"]
    cases = [
        ("B", opts4, "B", "bare"),
        ("**C**", opts4, "C", "bare"),
        ("(A)", opts4, "A", "bare"),
        ("D.", opts4, "D", "bare"),
        ("Answer: C", opts4, "C", None),
        ("The answer is D", opts4, "D", None),
        ("A) alpha", opts4, "A", "leading"),
        ("...therefore \\boxed{B}.", opts4, "B", None),
        ("gamma", opts4, "C", "option_text"),
        ("Let me think.\nSeveral steps.\nB", opts4, "B", None),
        ("", opts4, None, "empty"),
        ("   ", opts4, None, "empty"),
        ("I cannot determine this from the image.", opts4, None, "parse_fail"),
        ("Could be A or C, hard to say.", opts4, None, "parse_fail"),
        ("E", opts4, None, "parse_fail"),          # out of range for a 4-option item
    ]
    for text, opts, want_letter, want_method in cases:
        got, method = extract_letter(text, opts)
        ok = (got == want_letter) and (want_method is None or method.startswith(want_method))
        check(f"scorer {text[:34]!r}", ok, f"-> {got} via {method}")
    check("norm() is importable and idempotent", norm(norm("A, b. ")) == norm("A, b. "))

    print("\n--- 7. McNemar: exact values ---")
    check("b=c=0 -> p=1", mcnemar_exact(0, 0) == 1.0)
    check("b=6,c=0 -> p=2/64", abs(mcnemar_exact(6, 0) - 2 / 64) < 1e-12,
          f"{mcnemar_exact(6,0):.6f}")
    check("b=0,c=6 symmetric", mcnemar_exact(0, 6) == mcnemar_exact(6, 0))
    check("b=5,c=0 -> p=2/32 (>0.05, not significant)",
          abs(mcnemar_exact(5, 0) - 0.0625) < 1e-12)
    check("b=9,c=1 -> p=2*(1+10)/1024",
          abs(mcnemar_exact(9, 1) - 2 * 11 / 1024) < 1e-12, f"{mcnemar_exact(9,1):.6f}")
    check("cutover reproduces p=0.0011 for 49 vs 21",
          abs(mcnemar_exact(49, 21) - 0.0011) < 5e-5, f"{mcnemar_exact(49,21):.5f}")
    check("Wilson CI for 159/250 matches the pre-registered band",
          tuple(round(x, 1) for x in wilson_pct(159, 250)) == PREREG["instrument_band_pct"],
          str(tuple(round(x, 1) for x in wilson_pct(159, 250))))
    lo, hi = delta_ci_pp(5, 5, 250)
    check("delta CI is symmetric when b=c", abs(lo + hi) < 1e-9, f"[{lo:.2f},{hi:.2f}]")
    check("delta CI narrows as d falls",
          abs(delta_ci_pp(3, 3, 250)[0]) < abs(delta_ci_pp(10, 10, 250)[0]))

    print("\n--- 8. end-to-end on SYNTHETIC paired data with a KNOWN answer ---")
    synth_ok = _selftest_synthetic(check)

    print("\n--- 9. power / sensitivity at n=250 (exact) ---")
    print("      true degradation D (pp) x symmetric noise nu (pp) -> power")
    nus = [0, 4, 16]
    print("        D\\nu " + "".join(f"{v:>8}" for v in nus))
    for D in (1, 2, 3, 5, 8):
        row = f"        {D:>4} "
        for nu in nus:
            row += f"{mcnemar_power(nu/200, nu/200 + D/100):>8.3f}"
        print(row)
    check("power to detect a 1 pp drop is negligible",
          mcnemar_power(0.0, 0.01) < 0.10, f"{mcnemar_power(0.0, 0.01):.3f}")
    check("power to detect a 5 pp drop under low noise is high",
          mcnemar_power(0.0, 0.05) > 0.95, f"{mcnemar_power(0.0, 0.05):.3f}")

    print("\n--- 10. VRAM model ---")
    v16 = predict_vram(16384, "f16")
    check("f16 KV = 96 KiB/token", abs(v16["kv_bytes_per_token"] - 98304) < 1,
          f"{v16['kv_bytes_per_token']:.0f} B")
    v8 = predict_vram(16384, "q8_0")
    check("q8_0 KV = 51 KiB/token (34 B per 32-value block, NOT 32)",
          abs(v8["kv_bytes_per_token"] - 52224) < 1, f"{v8['kv_bytes_per_token']:.0f} B")
    check("model reproduces the measured 21049 MiB at -c 16384 f16",
          abs(v16["predicted_at_load_mib"][0] - MEASURED_TOTAL_MIB_AT_16384_F16) < 2,
          f"{v16['predicted_at_load_mib'][0]:.0f} MiB")
    check("q8_0 saves ~2.8 GiB of KV at -c 65536",
          abs((predict_vram(65536, 'f16')['kv_mib']
               - predict_vram(65536, 'q8_0')['kv_mib']) / 1024 - 2.81) < 0.02)

    print("\n--- 11. no packets were sent ---")
    check("_HTTP_CALLS == 0", _HTTP_CALLS == 0, str(_HTTP_CALLS))

    print("\n" + "=" * 60)
    if fails:
        print(f"SELFTEST FAILED: {len(fails)} check(s)")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"SELFTEST PASSED (synthetic end-to-end: {synth_ok})")
    return 0


def _fake_arm(correct_idx: set, parse_fail_idx: set = frozenset(),
              capped_idx: set = frozenset(), *, arm: str, n: int = 250,
              errors: int = 0, verified: bool = True, fp: str = "FP") -> dict:
    qs = list(load_questions())[:n]
    rows = []
    for q in qs:
        i = q["idx"]
        pf = i in parse_fail_idx
        rows.append({
            "idx": i, "id": q["id"], "subject": q["subject"], "expected": q["answer"],
            "raw": "" if pf else q["answer"],
            "pred": None if pf else (q["answer"] if i in correct_idx else
                                     ("Z" if q["answer"] != "Z" else "Y")),
            "parse": "parse_fail" if pf else "bare",
            "correct": (i in correct_idx) and not pf,
            "capped": i in capped_idx,
            "prompt_tokens": 1159, "completion_tokens": 2048 if i in capped_idx else 8,
            "attempts": 1, "error": ("boom" if i < errors else None),
        })
    s = summarise(rows, 2048)
    return {"_meta": {"arm": arm, "arm_verified": verified,
                      "instrument": {"verified": True},
                      "request_suite_fingerprint": fp,
                      "temperature": 0.0, "seed": 42, "max_tokens": 2048, "ctx": 65536},
            **s, "rows": rows}


def _selftest_synthetic(check) -> str:
    """Inject KNOWN paired outcomes and confirm the pipeline recovers them.

    Construction: f16 gets a fixed 159 right (the cutover score, so the
    instrument band passes). q8_0 keeps those minus `c` of them and additionally
    wins `b` that f16 missed. Then b, c, delta and the verdict are known in
    advance and the code has to reproduce them exactly.
    """
    qs = list(load_questions())
    all_idx = [q["idx"] for q in qs]
    f16_correct = set(all_idx[:159])
    f16_wrong = [i for i in all_idx if i not in f16_correct]

    def build(b: int, c: int):
        q8 = set(f16_correct)
        for i in sorted(f16_correct)[:c]:
            q8.discard(i)
        for i in f16_wrong[:b]:
            q8.add(i)
        return (_fake_arm(f16_correct, arm="f16"), _fake_arm(q8, arm="q8_0"))

    notes = []

    # (a) a real, decision-relevant degradation: q8_0 loses 12, wins 2 -> -4.0 pp
    A, B = build(b=2, c=12)
    r = evaluate(A, B)
    ok = (r["b_q8_only_correct"] == 2 and r["c_f16_only_correct"] == 12
          and abs(r["delta_pp"] + 4.0) < 1e-9
          and abs(r["mcnemar_p"] - mcnemar_exact(2, 12)) < 1e-12
          and r["verdict"] == "KEEP f16")
    check("synthetic b=2,c=12 -> delta -4.00 pp, verdict KEEP f16", ok,
          f"b={r['b_q8_only_correct']} c={r['c_f16_only_correct']} "
          f"delta={r['delta_pp']} p={r['mcnemar_p']:.4f} -> {r['verdict']}")
    notes.append(f"b=2,c=12 -> {r['verdict']}")

    # (b) genuinely lossless: 4 flips each way -> ADOPT
    A, B = build(b=4, c=4)
    r = evaluate(A, B)
    ok = (r["delta_pp"] == 0.0 and r["mcnemar_p"] == 1.0
          and r["delta_ci95_pp"][0] > -PREREG["ni_margin_pp"]
          and r["verdict"] == "ADOPT q8_0")
    check("synthetic b=c=4 -> CI inside the 3 pp margin, verdict ADOPT q8_0", ok,
          f"CI={r['delta_ci95_pp']} -> {r['verdict']}")
    notes.append(f"b=c=4 -> {r['verdict']}")

    # (c) the trap: p is not significant but the CI still admits real harm
    A, B = build(b=8, c=15)
    r = evaluate(A, B)
    ok = (r["mcnemar_p"] >= PREREG["alpha"]
          and r["delta_ci95_pp"][0] <= -PREREG["ni_margin_pp"]
          and r["verdict"] == "INCONCLUSIVE")
    check("synthetic b=8,c=15 -> p>0.05 but CI admits >3 pp harm, verdict INCONCLUSIVE", ok,
          f"p={r['mcnemar_p']:.4f} CI={r['delta_ci95_pp']} -> {r['verdict']}")
    notes.append(f"b=8,c=15 -> {r['verdict']}")

    # (d) noisy null: b=c=25 -> p=1 but the CI is too wide to certify anything
    A, B = build(b=25, c=25)
    r = evaluate(A, B)
    ok = (r["mcnemar_p"] == 1.0 and r["delta_ci95_pp"][0] <= -PREREG["ni_margin_pp"]
          and r["verdict"] == "INCONCLUSIVE")
    check("synthetic b=c=25 -> p=1.0 yet INCONCLUSIVE (a null is not equivalence)", ok,
          f"p={r['mcnemar_p']:.4f} CI={r['delta_ci95_pp']} -> {r['verdict']}")
    notes.append(f"b=c=25 -> {r['verdict']}")

    # (e) scorer artifact: identical correctness, but q8_0 parse-fails 14 more
    A = _fake_arm(f16_correct, arm="f16")
    pf = set(sorted(f16_correct)[:14])
    B = _fake_arm(f16_correct - pf, parse_fail_idx=pf, arm="q8_0")
    r = evaluate(A, B)
    ok = (r["artifacts"] and "parse-failure asymmetry" in r["artifacts"][0]
          and r["verdict"].startswith("INCONCLUSIVE"))
    check("synthetic parse-fail asymmetry is caught as a SCORER artifact", ok,
          f"{r['artifacts']} -> {r['verdict']}")
    notes.append("parse-artifact caught")

    # (f) void paths
    A, B = build(b=4, c=4)
    B["_meta"]["arm_verified"] = False
    check("unverified arm identity voids the run",
          evaluate(A, B)["verdict"] == "VOID")
    A, B = build(b=4, c=4)
    B["_meta"]["request_suite_fingerprint"] = "DIFFERENT"
    check("differing request bytes void the run",
          evaluate(A, B)["verdict"] == "VOID")
    A, B = build(b=4, c=4)
    B["_meta"]["max_tokens"] = 512
    check("differing max_tokens voids the run", evaluate(A, B)["verdict"] == "VOID")
    A2 = _fake_arm(set(list(f16_correct)[:100]), arm="f16")     # 40% -> outside the band
    _, B2 = build(b=4, c=4)
    check("f16 arm outside the instrument sanity band voids the run",
          evaluate(A2, B2)["verdict"] == "VOID")
    A, B = build(b=4, c=4)
    B["request_errors"] = 3
    check("transport errors void the run", evaluate(A, B)["verdict"] == "VOID")

    return "; ".join(notes)


# ========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score ONE arm against an already-running server")
    s.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8087")
    s.add_argument("--arm", required=True, choices=ARMS)
    s.add_argument("--outdir", default="/mnt/raid0/llm/tmp/vision-kv-q8/results")
    s.add_argument("--server-log", help="the arm's llama-server log (arm identity proof)")
    s.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    s.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    s.add_argument("--limit", type=int, help="debug only: score the first N questions")
    s.add_argument("--dry-run", action="store_true",
                   help="build every request and send NOTHING")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--no-log-check", action="store_true",
                   help="skip arm verification; stamps arm_verified:false (compare will void)")
    s.add_argument("--jinja", action="store_true",
                   help="record that the server was launched with --jinja (metadata only)")
    s.add_argument("--allow-production-port", action="store_true")
    s.add_argument("--allow-instrument-drift", action="store_true")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="paired stats + the pre-registered verdict")
    c.add_argument("--f16", required=True)
    c.add_argument("--q8", required=True)
    c.add_argument("--out")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("runbook", help="print the operator's exact commands + argv diff")
    r.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    r.add_argument("--port", type=int, default=BENCH_PORT)
    r.add_argument("--jinja", action="store_true",
                   help="add --jinja to BOTH arms (the cutover harness used it; "
                        "live production does not)")
    r.set_defaults(func=cmd_runbook)

    t = sub.add_parser("selftest", help="validate every non-inference path, send nothing")
    t.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
