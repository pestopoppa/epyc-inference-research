"""AK-HR-2 frozen mechanism taxonomy v1 and its deterministic rule parser.

The classes and their definitions live in ``data/ak-history-replay-20261020/ak-hr-2/taxonomy_v1.md``.
This module is the rule parser calibrated ONLY on the 60-hypothesis ``loop-memory`` pilot. Both files
are sha256-hashed into ``taxonomy_freeze.json`` before any window analysis runs.
"""
from __future__ import annotations

import re
from typing import Iterable

CLASSES = (
    "tiling_launch_geometry", "vector_isa_path", "reduction_sync", "kernel_fusion",
    "reuse_redundancy", "memory_placement_prefetch", "quant_layout_dequant",
    "dispatch_algorithm_selection", "threading_runtime_recipe", "other",
)
HARNESS_SENTINEL_MECHANISMS = frozenset({"anchor-aa-guard", "champion-vs-production"})
MIN_SCORE = 1.5

# (regex, weight). Applied to lower-cased "mechanism_id | symbol | surface | statement".
RULES: dict[str, tuple[tuple[str, float], ...]] = {
    "tiling_launch_geometry": (
        (r"\btil(e|es|ed|ing)\b", 2.0), (r"\bblocking\b", 2.0), (r"\bcta\b", 2.0),
        (r"work-?group", 2.0), (r"\bwaves?\b|\bwavefronts?\b|\bsubwaves?\b|sub-wave", 1.5),
        (r"nwarps|\bwarps?\b", 1.5), (r"threads? per (block|cta|workgroup|row)", 2.0),
        (r"rows? per (cta|block|thread|workgroup|wave)", 2.5), (r"occupancy", 2.0),
        (r"launch (geometry|config|bounds|shape)|grid\b", 1.5), (r"partition", 1.5),
        (r"stream-?k|split-?k", 1.5), (r"lane[- ]to[- ]|lane mapping|one thread per", 2.0),
        (r"\d+-thread (cta|block|workgroup)|\d+ threads", 2.0), (r"block size|chunk size|tile size", 2.0),
        (r"cache[- ]block|loop (tiling|blocking|interchange)", 2.5),
        (r"\bncols?\b|\bquery (tile|columns?|rows?)\b", 1.0),
        (r"row[- ]?pack|row groups?", 2.0),
    ),
    "vector_isa_path": (
        (r"float4|half2|float2|uint32_t store|vectori[sz]", 2.0),
        (r"\bsimd\b|\bavx-?(2|512)?\b|vnni|\bamx\b|\bneon\b|\bsve\b", 2.5),
        (r"_mm\d+_|vpdpbusd|vpermb|vpshufb|intrinsic", 2.5), (r"dp4a|v_dot|\bmfma\b|\bmma\b", 1.5),
        (r"packed", 1.5), (r"v_perm|__builtin_amdgcn_perm|\bperm\b", 1.5), (r"swar", 2.5),
        (r"\bilp\b|accumulator chain|independent .*accumulator|dependency chain|serial dp4a", 2.5),
        (r"unroll", 1.5), (r"wide[r]? (load|store)|aligned (load|store)|vector load", 2.0),
        (r"round-to-nearest f32|cvt\b|convert pairs", 1.5), (r"\bfma\b|fused multiply", 1.5),
        (r"gather|scatter|shuffle bytes|bf16 dot", 1.0),
    ),
    "reduction_sync": (
        (r"reduc(e|es|ed|tion|tions)\b", 1.5), (r"shfl|shuffle", 1.5), (r"\bdpp\b|swizzle|bpermute", 2.0),
        (r"barrier", 2.0), (r"__syncthreads|syncwarp|\bsync(hroni[sz]\w*)?\b", 1.5), (r"atomic", 2.0),
        (r"fence|release ordering|acquire", 2.0), (r"all-?reduce|horizontal (sum|add)|block_reduce", 2.5),
        (r"\bxor\b", 1.0), (r"fixup", 2.0), (r"tree (shape|reduce|reduction)|reduction order", 2.0),
        (r"last-arriv|completion atomic", 2.0),
    ),
    "kernel_fusion": (
        (r"\bfus(e|ed|es|ing|ion)\b|-fused|fusion", 4.0), (r"inline[- ]?(q8|quant|act)|inline-", 2.5),
        (r"epilogue", 1.5), (r"single[- ]pass|one launch|single launch|merge[sd]? .*(kernel|launch|pass)", 2.5),
        (r"fold(s|ed)? .* into", 2.0), (r"intermediate (global-?mem|buffer|tensor)", 1.5),
        (r"removes? (the )?separate .*launch", 2.5),
    ),
    "reuse_redundancy": (
        (r"redundan", 3.0), (r"\breus(e|ed|es|ing)\b|-reuse|re-use", 3.0), (r"hoist", 3.0),
        (r"\bcache[sd]?\b.*(result|conversion|across|per)", 2.0), (r"\bonce\b", 1.5),
        (r"broadcast", 1.5), (r"precompute|pre-compute|angle table|lookup table|\bscratch table", 2.5),
        (r"skip(s|ping|ped)?\b|eliminat", 1.5), (r"recomput|re-?load", 2.0),
        (r"shared across|across (sibling|consumers)|common subexpression|\bmemoi[sz]", 2.0),
        (r"index arithmetic|fast_div|address (arithmetic|calculation)", 2.0),
        (r"sidecar", 1.0), (r"transcendental|sincos", 1.5),
    ),
    "memory_placement_prefetch": (
        (r"\blds\b|shared memory|shared-mem|local data share", 2.5),
        (r"registers?\b|\bvgprs?\b|register-resident", 2.0),
        (r"live range|spill", 2.5), (r"bank( |-)?conflict|\bbank", 2.0), (r"scratch\b", 1.0),
        (r"prefetch", 3.0), (r"cache[- ]?(line|locality|resident|miss)|\bl[123] cache|\bl[123]\b", 2.0),
        (r"non-temporal|streaming store|\bnt store", 3.0), (r"hugepage|huge page|\bthp\b|madvise", 3.0),
        (r"alignment|aligned alloc|padding|allocation|arena|mmap|mlock", 1.5),
        (r"hbm|global memory|coalesc", 1.5),
        (r"fragment", 1.0), (r"late[- ]materiali[sz]e|round-?trip", 1.5),
        (r"memory placement|first-?touch", 3.0),
    ),
    "quant_layout_dequant": (
        (r"dequant\w*", 2.5), (r"unpack|nibble|\bql\b|\bqh\b|ql/qh", 2.0),
        (r"scales?\b.*\bmins?\b|\bmins?\b.*scale|scale/min|scale[- ]min|metadata", 1.5),
        (r"block_q\w+|superblock|q8_1 (block|scale|sum|activation)|q8_1", 0.8),
        (r"sidecar|block[- ]sum|post-rounding correction|min-term|minimum term", 1.5),
        (r"signed|unsigned|offset", 1.0),
        (r"repack|interleav|re-?layout|\blayout\b|x4t?\b|r4\b|\bq4_?k_?r4", 2.5),
        (r"quantiz(e|es|ation)|requantiz", 0.8),
        (r"\biq[1-4]_\w+|\bmxfp|\bnvfp|\bq[2-8]_[0-9k]|\bq[2-8]k\b|\bq[2-8]_0", 0.4),
        (r"bf16|fp8|int8", 0.3), (r"activation quant", 1.5),
    ),
    "dispatch_algorithm_selection": (
        (r"dispatch", 2.5), (r"crossover", 3.0), (r"\broute\b|\broutes\b|routing|route selection", 2.0),
        (r"should_use|heuristic|threshold", 2.5),
        (r"\bselect(s|ion)?\b.*(kernel|path|schedule|algorithm|mmvq|mmq)", 2.0),
        (r"\bdisabl\w+|\benabl\w+", 1.5), (r"\bgate[sd]? (on|by|to)\b|\brestrict", 1.5),
        (r"fallback", 1.5),
        (r"instead of (the )?(current|stream-k|mmq|mmvq|existing)", 1.5), (r"algorithm", 1.5),
        (r"switch(es)? (to|from)|prefer(s)?\b", 1.5),
    ),
    "threading_runtime_recipe": (
        (r"\b(cpu|host|omp|openmp|worker|team|pool) threads?\b|\bnthreads?\b|n_threads|"
         r"thread (pool|team|count|affinity|binding)", 3.0),
        (r"affinity|\bpin(ned|ning)?\b|\bsmt\b|hyper-?thread|\bsibling", 3.0),
        (r"\bnuma\b|numactl|interleave[: ]|first-touch", 3.0),
        (r"\bomp_\w+|openmp|\bomp\b|gomp|kmp_", 3.0),
        (r"wait[_ ]policy|spin-?wait|busy-?wait|futex|spin policy", 3.0),
        (r"\brecipe\b|runtime[- ]arm|load_threads|environment variable|\benv\b|launch flag|runtime knob", 2.5),
        (r"\bscheduler\b|work[- ]?steal|dynamic schedul|static schedul|chunk(ing)? .*thread", 2.5),
        (r"\bcores?\b|\bccd\b|\bccx\b|\bl3 slice", 1.5),
        (r"batch(ed)? (width|size)|ubatch|np=|parallel slots", 1.0),
    ),
}
_COMPILED = {c: tuple((re.compile(p), w) for p, w in rules) for c, rules in RULES.items()}


def hypothesis_text(mechanism_id: str | None, symbol: str | None, surface: str | None,
                    statement: str | None) -> str:
    # Symbol and surface name the TARGET (``dequantize_block_q4_K``, ``mmvq.cu``), not the mechanism, and
    # leaked class words into the score in the pilot; the hypothesis text is the mechanism id + statement.
    return " | ".join(str(x or "") for x in (mechanism_id, statement)).lower()


def has_intent(mechanism_id: str | None, statement: str | None) -> bool:
    """Intent text = a statement, or a mechanism id that is not a harness sentinel."""
    if statement and statement.strip():
        return True
    return bool(mechanism_id) and mechanism_id not in HARNESS_SENTINEL_MECHANISMS


WEAK_WEIGHT = 1.0       # patterns at or below this weight are naming evidence, not mechanism evidence
WEAK_CAP = 1.0          # ...and together contribute at most this much to one class


def scores(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls, rules in _COMPILED.items():
        strong = weak = 0.0
        for rx, weight in rules:
            hits = len(rx.findall(text))
            if not hits:
                continue
            value = weight * (1.0 + 0.15 * min(hits - 1, 2))
            if weight <= WEAK_WEIGHT:
                weak += value
            else:
                strong += value
        out[cls] = round(strong + min(weak, WEAK_CAP), 6)
    return out


def classify(mechanism_id: str | None, symbol: str | None, surface: str | None,
             statement: str | None) -> tuple[str, dict[str, float]]:
    sc = scores(hypothesis_text(mechanism_id, symbol, surface, statement))
    best = max(CLASSES[:-1], key=lambda c: (sc[c], -CLASSES.index(c)))
    return (best if sc[best] >= MIN_SCORE else "other"), sc


def cohen_kappa(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = list(a), list(b)
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
