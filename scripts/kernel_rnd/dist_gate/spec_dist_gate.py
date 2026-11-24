#!/usr/bin/env python3
"""Distribution and identity gates for speculative decoding (SPEC-PIT.1, SPEC-ID-GATE-1).

Pure Python (stdlib only), deterministic for fixed seeds, no inference.

SPEC-PIT.1 -- is an arm's T>0 output distributed as the target sampler says it should be?
  * ``post_sampler_probs``   -- the target distribution after the llama.cpp sampler chain
                                (top_k -> top_p -> min_p -> temperature, the default chain order).
  * ``randomized_pit``       -- PIT value F(x-) + U * P(tie group of x). Uniform(0,1) iff x ~ p.
  * ``ks_uniform``           -- one-sample Kolmogorov-Smirnov test against Uniform(0,1).
  * ``rank_hist`` / ``chi2_homog`` -- rank-of-sampled-token histogram and a homogeneity chi-square.
  * ``match_by_key`` / ``match_by_rule`` -- SELECTION MATCHING. An arm that only emits tokens at some
                                positions (e.g. p-min skips drafting at low-confidence positions, so its
                                captured rows over-represent confident positions) must be compared with
                                the reference restricted to the same positions, either the co-captured
                                keys or the arm's own selection rule applied to the reference.
                                Without this the rank chi-square fails on a sampler that is exact.
  * ``perm_test_requests``   -- request-level permutation test of the chi-square statistic. Whole
                                requests are shuffled between arms, so within-sequence dependence is
                                respected (tokens of one request are not independent draws).
  * ``gate``                 -- PASS / FAIL / INCONCLUSIVE with the thresholds in ``THRESHOLDS``.

  Note: the KS on PIT is the principled exactness test; it needs only the arm's own per-token target
  probabilities and no reference arm. The rank chi-square is a secondary, reference-relative check and
  is valid only when selection-matched and tested at request level (CAFE-1 investigation 2026-11-23).

SPEC-ID-GATE-1 -- token-identity comparisons:
  * ``margin_class``         -- exact0 / le0.1 / le0.5 / gt0.5 (top-2 margin in nats at first divergence).
  * ``classify_divergence``  -- NOISE / DEGEN_AFTER_TIE / DEFECT_CANDIDATE / DEFECT for one first divergence.
  * ``paired_count_test``    -- paired sign-flip test of per-block divergence counts (arm pair minus
                                same-run A/A pair), for configs where the identity floor is not zero.
SPEC-ID-GATE-2 -- replica floor:
  * ``replica_floor``        -- largest-pair A/A divergence count over all pairs of reference launches.
  * ``identity_replica_gate`` -- candidate vs >= 3 reference launches (>= 5 for a promotion gate);
                                fewer launches -> INCONCLUSIVE.
Protocol: /workspace/docs/guides/agent-workflows/spec-identity-gate-protocol.md (epyc-root).

Record format used by the gate (one per emitted token), either
  {"probs": {token_id: target_prob, ...}, "token": token_id, "key": hashable (optional)}
``probs`` must already be the post-sampler target distribution (use ``post_sampler_probs`` on raw
logits or pre-sampler probabilities), or, from a server that computed the masses itself (llama-server
``pit_probs``, fork eea10be26):
  {"pit_less": float, "pit_tie": float, "rank": int, "key": ... (optional)}
where pit_less / pit_tie are the post-chain mass strictly below / tied with the sampled token over the
full candidate set (-1 or absent = out of support) and rank is its 0-based rank (rank_cap = other).
A request is a list of records.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

Record = dict
Request = List[Record]

# Documented thresholds. Changing these changes gate verdicts; record the change in the protocol doc.
THRESHOLDS = {
    "ks_fail_p": 1e-5,          # SPEC-PIT.1 box: fail an arm on KS p <= 1e-5 (DeepInfer test_sampler.py recipe)
    "chi2_fail_p": 1e-3,        # rank chi-square (request-level permutation p) fails below this
    "min_tokens": 2000,         # fewer PIT values -> INCONCLUSIVE (KS power too low for small biases)
    "min_requests": 8,          # fewer requests per arm -> chi-square INCONCLUSIVE (permutation too coarse)
    "n_perm": 2000,             # permutations; p resolution 1/(n+1) < chi2_fail_p
    "rank_cap": 20,             # ranks >= cap pool into one bin (top-20 + other)
    "chi2_minpool": 10,         # merge tail rank bins until each pooled bin has >= this many tokens
    "support_eps": 0.0,         # a sampled token with target prob <= eps is out of support -> FAIL
}

MARGIN_CLASSES = ("exact0", "le0.1", "le0.5", "gt0.5")


# ----------------------------------------------------------------------------------------- sampler chain
def _softmax(logits: Mapping[int, float]) -> Dict[int, float]:
    m = max(logits.values())
    e = {t: math.exp(v - m) for t, v in logits.items()}
    z = sum(e.values())
    return {t: v / z for t, v in e.items()}


def post_sampler_probs(logits: Mapping[int, float], temperature: float = 1.0, top_k: int = 0,
                       top_p: float = 1.0, min_p: float = 0.0, *, are_logprobs: bool = False) -> Dict[int, float]:
    """Target distribution after the llama.cpp default chain top_k -> top_p -> min_p -> temperature.

    ``logits`` maps token -> logit (or log-prob when ``are_logprobs``; identical after softmax).
    Truncation samplers act on the T=1 softmax, as in llama.cpp; temperature rescales the survivors.
    temperature <= 0 means greedy: all mass on the argmax (ties broken by lowest token id).
    """
    if not logits:
        raise ValueError("empty logits")
    items = sorted(logits.items(), key=lambda kv: (-kv[1], kv[0]))
    if temperature <= 0:
        return {items[0][0]: 1.0}
    if top_k and top_k > 0:
        items = items[:top_k]
    p = _softmax(dict(items))
    order = [t for t, _ in items]
    if top_p < 1.0:
        keep, c = [], 0.0
        for t in order:
            keep.append(t)
            c += p[t]
            if c >= top_p:
                break
        order = keep
        p = _softmax({t: logits[t] for t in order})
    if min_p > 0.0:
        pmax = p[order[0]]
        order = [t for t in order if p[t] >= min_p * pmax]
    return _softmax({t: logits[t] / temperature for t in order})


# ----------------------------------------------------------------------------------------- PIT and KS
def randomized_pit(probs: Mapping[int, float], token: int, rng: random.Random) -> Optional[float]:
    """F(x-) + U * m(x), where F(x-) is the mass of tokens strictly less probable than x and m(x) the mass of
    x's tie group (tokens with exactly p_x). Uniform(0,1) when x ~ probs. None if x is out of support."""
    px = probs.get(token, 0.0)
    if px <= THRESHOLDS["support_eps"]:
        return None
    less = sum(p for p in probs.values() if p < px)
    tie = sum(p for p in probs.values() if p == px)
    z = sum(probs.values())
    return (less + rng.random() * tie) / z


def ks_uniform(u: Sequence[float]) -> Tuple[float, float]:
    """One-sample KS against Uniform(0,1). Returns (p, D). Asymptotic Kolmogorov law with the
    Stephens small-sample correction; adequate for n >= ~35."""
    n = len(u)
    if n == 0:
        return 1.0, 0.0
    s = sorted(u)
    d = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(s))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam < 1e-3:
        return 1.0, d
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return max(0.0, min(1.0, p)), d


# ----------------------------------------------------------------------------------------- rank chi-square
def record_pit(r: Record, rng: random.Random) -> Optional[float]:
    """PIT of one record: server-computed masses when present, else from the probs dict."""
    if "pit_less" in r:
        less, tie = r["pit_less"], r["pit_tie"]
        if less is None or tie is None or less < 0 or not tie > 0:
            return None
        return min(1.0, less + rng.random() * tie)
    return randomized_pit(r["probs"], r["token"], rng)


def record_rank(r: Record, cap: int = THRESHOLDS["rank_cap"]) -> int:
    if "rank" in r and "probs" not in r:
        return min(int(r["rank"]), cap)
    return token_rank(r["probs"], r["token"], cap)


def token_rank(probs: Mapping[int, float], token: int, cap: int = THRESHOLDS["rank_cap"]) -> int:
    """0-based rank of token by descending prob (ties by token id); out of support or >= cap -> cap."""
    if token not in probs:
        return cap
    order = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    for i, (t, _) in enumerate(order):
        if t == token:
            return min(i, cap)
    return cap


def rank_hist(requests: Iterable[Request], cap: int = THRESHOLDS["rank_cap"]) -> List[int]:
    h = [0] * (cap + 1)
    for req in requests:
        for r in req:
            h[record_rank(r, cap)] += 1
    return h


def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of chi-square(k) via the regularized incomplete gamma function."""
    if k <= 0:
        return 1.0
    a, x2 = k / 2.0, x / 2.0
    if x2 <= 0:
        return 1.0
    if x2 < a + 1:
        s = t = 1.0 / a
        n = a
        for _ in range(2000):
            n += 1
            t *= x2 / n
            s += t
            if abs(t) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x2 + a * math.log(x2) - math.lgamma(a)))
    b = x2 + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1.0 / d if d else 1e300
        c = b + an / c
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x2 + a * math.log(x2) - math.lgamma(a)) * h)


def chi2_homog(a: Sequence[int], b: Sequence[int], minpool: int = THRESHOLDS["chi2_minpool"]) -> Tuple[float, int, float]:
    """Two-sample homogeneity chi-square over rank bins, tail bins merged to >= minpool. (chi2, dof, p)."""
    bins, ca, cb = [], 0, 0
    for x, y in zip(a, b):
        ca += x
        cb += y
        if ca + cb >= minpool:
            bins.append((ca, cb))
            ca = cb = 0
    if (ca or cb) and bins:
        x, y = bins[-1]
        bins[-1] = (x + ca, y + cb)
    na, nb = sum(x for x, _ in bins), sum(y for _, y in bins)
    if len(bins) < 2 or not na or not nb:
        return 0.0, 0, 1.0
    chi = 0.0
    for x, y in bins:
        t = x + y
        ea, eb = t * na / (na + nb), t * nb / (na + nb)
        chi += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    return chi, len(bins) - 1, _chi2_sf(chi, len(bins) - 1)


# ----------------------------------------------------------------------------------------- selection matching
def match_by_key(ref: Sequence[Request], arm: Sequence[Request]) -> Tuple[List[Request], List[Request]]:
    """Keep only records whose ``key`` is captured in BOTH arms (co-captured positions). Requests are
    paired by index; both inputs must list the same prompts/seeds in the same order."""
    if len(ref) != len(arm):
        raise ValueError("match_by_key needs request-paired arms (same prompts, same order)")
    out_r, out_a = [], []
    for qr, qa in zip(ref, arm):
        kr = {r["key"] for r in qr}
        ka = {r["key"] for r in qa}
        both = kr & ka
        out_r.append([r for r in qr if r["key"] in both])
        out_a.append([r for r in qa if r["key"] in both])
    return out_r, out_a


def match_by_rule(ref: Sequence[Request], select: Callable[[Record], bool]) -> List[Request]:
    """Apply the arm's selection rule (e.g. the p-min draft-confidence threshold) to the reference arm."""
    return [[r for r in q if select(r)] for q in ref]


# ----------------------------------------------------------------------------------------- request-level permutation
def perm_test_requests(ra: Sequence[Request], rb: Sequence[Request], n_perm: int = THRESHOLDS["n_perm"],
                       seed: int = 1, cap: int = THRESHOLDS["rank_cap"]) -> Tuple[float, float]:
    """Permutation p of the rank chi-square, shuffling whole requests between arms. Returns (chi2_obs, p)."""
    ha = [rank_hist([q], cap) for q in ra]
    hb = [rank_hist([q], cap) for q in rb]

    def total(hs):
        t = [0] * (cap + 1)
        for h in hs:
            for i, v in enumerate(h):
                t[i] += v
        return t

    obs = chi2_homog(total(ha), total(hb))[0]
    pool = ha + hb
    na = len(ha)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if chi2_homog(total(pool[:na]), total(pool[na:]))[0] >= obs - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


# ----------------------------------------------------------------------------------------- the gate
def pit_values(requests: Iterable[Request], seed: int = 12345) -> Tuple[List[float], int]:
    """PIT per record; returns (values, n_out_of_support)."""
    rng = random.Random(seed)
    u, oos = [], 0
    for q in requests:
        for r in q:
            v = record_pit(r, rng)
            if v is None:
                oos += 1
            else:
                u.append(v)
    return u, oos


def gate(arm: Sequence[Request], ref: Optional[Sequence[Request]] = None, *,
         select: Optional[Callable[[Record], bool]] = None, match_keys: bool = False,
         thresholds: Optional[Mapping] = None, seed: int = 12345) -> dict:
    """SPEC-PIT.1 verdict for one arm.

    FAIL         any sampled token out of target support; or arm KS p <= ks_fail_p; or (with ``ref``)
                 selection-matched request-level permutation chi-square p < chi2_fail_p.
    INCONCLUSIVE arm has < min_tokens PIT values; or the reference arm fails its own KS (instrument or
                 capture broken: the calibration arm must pass); or chi-square requested but either arm
                 has < min_requests requests after matching.
    PASS         otherwise.

    ``ref`` is the speculation-off / p-min-0 reference arm, sampled under the same sampler config.
    Matching: ``select`` applies the arm's selection rule to ``ref``; ``match_keys`` keeps only
    co-captured ``key`` values. Pass neither only if the arm captures every position the reference does.
    """
    th = dict(THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    out: dict = {"thresholds": th, "reasons": []}
    u, oos = pit_values(arm, seed)
    out["n_tokens"], out["out_of_support"] = len(u), oos
    out["ks_p"], out["ks_d"] = ks_uniform(u)
    fail = inconclusive = False
    if oos:
        fail = True
        out["reasons"].append(f"{oos} sampled token(s) outside target support")
    if len(u) < th["min_tokens"]:
        inconclusive = True
        out["reasons"].append(f"n_tokens {len(u)} < min_tokens {th['min_tokens']}")
    elif out["ks_p"] <= th["ks_fail_p"]:
        fail = True
        out["reasons"].append(f"KS p {out['ks_p']:.3g} <= {th['ks_fail_p']}")
    if ref is not None:
        ur, _ = pit_values(ref, seed + 1)
        out["ref_ks_p"] = ks_uniform(ur)[0]
        if len(ur) >= th["min_tokens"] and out["ref_ks_p"] <= th["ks_fail_p"]:
            inconclusive = True
            out["reasons"].append(f"reference arm fails its own KS (p {out['ref_ks_p']:.3g}): instrument not calibrated")
        r_m, a_m = ref, arm
        if match_keys:
            r_m, a_m = match_by_key(ref, arm)
        if select is not None:
            r_m = match_by_rule(r_m, select)
        out["matched"] = bool(match_keys or select is not None)
        r_m = [q for q in r_m if q]
        a_m = [q for q in a_m if q]
        out["n_req_ref"], out["n_req_arm"] = len(r_m), len(a_m)
        if len(r_m) < th["min_requests"] or len(a_m) < th["min_requests"]:
            inconclusive = True
            out["reasons"].append("too few requests for the request-level chi-square")
        else:
            out["chi2_naive"] = chi2_homog(rank_hist(r_m, th["rank_cap"]), rank_hist(a_m, th["rank_cap"]),
                                           th["chi2_minpool"])
            out["chi2"], out["chi2_perm_p"] = perm_test_requests(r_m, a_m, th["n_perm"], seed, th["rank_cap"])
            if out["chi2_perm_p"] < th["chi2_fail_p"]:
                fail = True
                out["reasons"].append(f"rank chi2 permutation p {out['chi2_perm_p']:.3g} < {th['chi2_fail_p']}")
    out["verdict"] = "FAIL" if fail else ("INCONCLUSIVE" if inconclusive else "PASS")
    return out


# ----------------------------------------------------------------------------------------- SPEC-ID-GATE-1 helpers
def margin_class(margin_nats: float) -> str:
    """Top-2 logit/log-prob margin (nats) at a first divergence -> exact0 / le0.1 / le0.5 / gt0.5."""
    if margin_nats < 0:
        raise ValueError("margin must be >= 0")
    if margin_nats == 0.0:
        return "exact0"
    if margin_nats <= 0.1:
        return "le0.1"
    if margin_nats <= 0.5:
        return "le0.5"
    return "gt0.5"


def classify_divergence(margin_nats: Optional[float], *, identity_floor_zero: bool, tau: float = 0.5,
                        degen_after: bool = False, control_degen: bool = False) -> dict:
    """Classify one first divergence between two arms (SPEC-ID-GATE-1 (c)).

    identity_floor_zero: True when the comparison is inside the exact-identity regime (same server, single
        request, deterministic config, e.g. CPU spec-off or CPU MTP/DFlash at concurrency 1). There the
        same-run A/A floor is 0, so ANY divergence is a real numerics change -> at least DEFECT_CANDIDATE.
    tau: the class tolerance in nats (measured per width class by SPEC-ID-NOISE-1; 0.5 is the default upper
        bound -- no measured A/A first divergence exceeded 0.5 nats on CPU or GPU).
    degen_after / control_degen: the arm falls into a repeat loop after the divergence / the speculation-off
        np4 control shows the same loop.
    margin_nats None (logprobs not retained) -> UNCLASSIFIABLE.
    """
    if margin_nats is None:
        return {"margin_class": None, "verdict": "UNCLASSIFIABLE",
                "reason": "no top-2 margin retained; re-run with logprobs"}
    mc = margin_class(margin_nats)
    if degen_after and not control_degen:
        v, why = "DEGEN_AFTER_TIE", "repeat loop after the divergence that the spec-off control does not show (defect)"
    elif identity_floor_zero:
        v, why = "DEFECT_CANDIDATE", "exact-identity regime (A/A floor 0): any divergence is a real numerics change"
    elif margin_nats <= tau:
        v, why = "NOISE", f"near-tie flip: margin {margin_nats:.4g} <= class tau {tau}"
    else:
        v, why = "DEFECT_CANDIDATE", f"margin {margin_nats:.4g} > class tau {tau}"
    return {"margin_class": mc, "verdict": v, "reason": why}


def paired_count_test(arm_counts: Sequence[float], aa_counts: Sequence[float], n_perm: int = 20000,
                      seed: int = 7) -> dict:
    """Paired d test: per block, d = (divergences in the arm pair) - (divergences in the same-run A/A pair).
    One-sided exact/Monte-Carlo sign-flip p for mean d > 0, plus a 95% t interval of mean d."""
    if len(arm_counts) != len(aa_counts) or not arm_counts:
        raise ValueError("need equal-length, non-empty per-block counts")
    d = [a - b for a, b in zip(arm_counts, aa_counts)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    tcrit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
             15: 2.13, 20: 2.09, 30: 2.04}
    df = n - 1
    tc = tcrit.get(df) or (1.96 if df > 30 else tcrit[min(k for k in tcrit if k >= df)])
    half = tc * sd / math.sqrt(n) if n > 1 else float("inf")
    if n <= 16:
        ge, tot = 0, 0
        for mask in range(1 << n):
            s = sum(x if not (mask >> i) & 1 else -x for i, x in enumerate(d))
            tot += 1
            ge += s >= sum(d) - 1e-12
        p = ge / tot
    else:
        rng = random.Random(seed)
        obs = sum(d)
        ge = sum(1 for _ in range(n_perm) if sum(x if rng.random() < 0.5 else -x for x in d) >= obs - 1e-12)
        p = (ge + 1) / (n_perm + 1)
    return {"n_blocks": n, "mean_d": mean, "ci95": (mean - half, mean + half), "p_one_sided": p}


# ----------------------------------------------------------------------------------------- SPEC-ID-GATE-2 replica floor
REPLICA_MIN = {"default": 3, "promotion": 5}  # SPEC-ID-GATE-2 (a): reference launches needed for a floor


def _divergent_prompts(a: Mapping[Hashable, Sequence[int]], b: Mapping[Hashable, Sequence[int]]) -> List[Hashable]:
    """Prompt ids whose token sequences differ between two launches (a missing prompt counts as divergent)."""
    return sorted((k for k in set(a) | set(b) if list(a.get(k, ())) != list(b.get(k, ())) or k not in a or k not in b),
                  key=str)


def replica_floor(launches: Sequence[Mapping[Hashable, Sequence[int]]]) -> dict:
    """Largest-pair A/A floor over all pairs of reference launches (SPEC-ID-GATE-2 (a)).

    launches: one {prompt_id: [token ids]} per reference launch of the unchanged config."""
    pairs = []
    for i in range(len(launches)):
        for j in range(i + 1, len(launches)):
            pairs.append({"pair": (i, j), "divergent": len(_divergent_prompts(launches[i], launches[j]))})
    floor = max((p["divergent"] for p in pairs), default=None)
    return {"n_launches": len(launches), "pairs": pairs, "largest_pair": floor}


def identity_replica_gate(candidate: Mapping[Hashable, Sequence[int]],
                          references: Sequence[Mapping[Hashable, Sequence[int]]], *,
                          promotion: bool = False, margins: Optional[Mapping[Hashable, Optional[float]]] = None,
                          tau: float = 0.5) -> dict:
    """T=0 identity gate against a replica floor (SPEC-ID-GATE-2).

    INCONCLUSIVE when fewer than 3 reference launches are supplied (5 when ``promotion``): one or two
    launches cannot bound launch-to-launch drift. Otherwise the candidate's divergence count is the
    largest count against any reference launch; FAIL if it exceeds the largest reference pair, or if any
    first divergence (``margins``: prompt id -> top-2 margin at the first divergent position) classifies
    as anything but NOISE. Missing margins for a divergent prompt -> UNCLASSIFIABLE -> INCONCLUSIVE."""
    need = REPLICA_MIN["promotion" if promotion else "default"]
    if len(references) < need:
        return {"verdict": "INCONCLUSIVE", "n_references": len(references), "needed": need,
                "reason": f"{len(references)} reference launches < {need}; the A/A floor is unbounded"}
    floor = replica_floor(references)
    per_ref = [_divergent_prompts(candidate, r) for r in references]
    cand = max(len(d) for d in per_ref)
    divergent = sorted({k for d in per_ref for k in d}, key=str)
    classes = {}
    if margins is not None or floor["largest_pair"] == 0:
        for k in divergent:
            m = None if margins is None else margins.get(k)
            classes[k] = classify_divergence(m, identity_floor_zero=floor["largest_pair"] == 0, tau=tau)
    verdicts = {c["verdict"] for c in classes.values()}
    if cand > floor["largest_pair"] or verdicts - {"NOISE", "UNCLASSIFIABLE"}:
        v = "FAIL"
    elif "UNCLASSIFIABLE" in verdicts or (margins is None and divergent):
        v = "INCONCLUSIVE"
    else:
        v = "PASS"
    return {"verdict": v, "n_references": len(references), "needed": need, "floor": floor,
            "candidate_divergent": cand, "per_reference": [len(d) for d in per_ref], "classes": classes}
