"""AK-HR-2 taxonomy-free parts, run after the held-out agreement gate FAILED (kappa < 0.6).

Nothing here uses the frozen mechanism taxonomy, so none of it can produce the predeclared
``collapse seen`` / ``not detected`` verdict. It produces:
  * no_intent accounting by status, per campaign;
  * (ii) the with-lessons repeat-null rate (structural agreement via the real ``do_not_repeat``
    ``structural_target`` / ``agreement`` used by ``dispatch_guard.characterised_reason``) with
    ``refused_at_formation`` as lower bound;
  * (i) the signed faithfulness ratio rho per campaign (mechanism class is NOT available: no class labels);
  * surface-diff-distance availability (retained patch counts);
  * DESCRIPTIVE window statistics (W=20) over mechanism_id and target-file, with the 1000-permutation
    slope comparison, labelled non-predeclared and verdict-free.
Reads only ``history_records.jsonl``. Deterministic (seeded); outputs carry a config echo.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))   # scripts/kernel_rnd
import ak_hr2_taxonomy as T  # noqa: E402  (only has_intent / sentinel list is used)
from autokernel.controller import do_not_repeat  # noqa: E402
from autokernel.loop import dispatch_guard  # noqa: E402

W = 20
PERMUTATIONS = 1000
SEED = 20261020
ANSWER = dispatch_guard.ANSWER_STATUSES
SCREEN_TIER = ("keep_candidate", "measured_null", "regression")
CONFIRM_TIER = ("kept", "measured_serving_gate")


def norm_ts(s):
    """Fixed-width ISO so whole-second and fractional stamps order correctly."""
    if not s:
        return ""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
    else:
        head, frac = s, ""
    return f"{head}.{frac.ljust(6, '0')[:6]}Z"


def sort_key(r):
    return (norm_ts(r["recorded_at"]), r["store_label"], r["rowid"] if r["rowid"] is not None else -1)


def entropy(counts):
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c) if n else 0.0


def window_series(seq):
    """H_t (nats) and distinct count for each sliding window of W over a label sequence."""
    out = []
    cnt = collections.Counter(seq[:W])
    for i in range(len(seq) - W + 1):
        if i:
            cnt[seq[i - 1]] -= 1
            if not cnt[seq[i - 1]]:
                del cnt[seq[i - 1]]
            cnt[seq[i + W - 1]] += 1
        out.append((entropy(cnt), len(cnt)))
    return out


def slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2
    my = sum(ys) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den


def percentile(vals, p):
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def descriptive_windows(per_epoch, rng):
    """per_epoch: {epoch: [labels in order]}. Campaign statistic = window-weighted mean slope."""
    eligible = {e: s for e, s in per_epoch.items() if len(s) >= W}
    epochs = []
    obs_num = obs_den = 0.0
    for e, s in sorted(eligible.items()):
        ser = window_series(s)
        sl = slope([h for h, _ in ser])
        epochs.append({"epoch": e[:8], "n": len(s), "sliding_windows": len(ser), "full_windows": len(s) // W,
                       "mean_H": sum(h for h, _ in ser) / len(ser), "mean_distinct": sum(d for _, d in ser) / len(ser),
                       "first_window_H": ser[0][0], "last_window_H": ser[-1][0], "H_slope_per_window": sl})
        obs_num += sl * len(ser)
        obs_den += len(ser)
    obs = obs_num / obs_den if obs_den else None
    null = []
    if eligible:
        for _ in range(PERMUTATIONS):
            num = 0.0
            for e, s in eligible.items():
                p = list(s)
                rng.shuffle(p)
                ser = window_series(p)
                num += slope([h for h, _ in ser]) * len(ser)
            null.append(num / obs_den)
    p5 = percentile(null, 0.05) if null else None
    return {"epochs_eligible": epochs, "total_full_windows": sum(len(s) // W for s in per_epoch.values()),
            "total_sliding_windows": int(obs_den), "observed_weighted_slope": obs,
            "null_p5_slope": p5, "below_null_p5": (obs < p5) if obs is not None else None,
            "permutation_p_lower": (sum(1 for v in null if v <= obs) / len(null)) if null else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    raw = a.history.read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines()]
    ex = sorted((r for r in rows if r["record_kind"] == "experiment"), key=sort_key)
    camps = sorted({r["campaign_dir"] for r in ex})
    out = {"schema": "ak-hr-2.taxfree.v1",
           "status": "TAXONOMY GATE FAILED (see heldout_agreement.json); this file carries no collapse verdict",
           "config_echo": {"argv": sys.argv[1:], "python": platform.python_version(), "W": W,
                           "permutations": PERMUTATIONS, "seed": SEED, "answer_statuses": sorted(ANSWER),
                           "history_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
                           "experiment_rows": len(ex)}}

    # ---- no_intent accounting -------------------------------------------------------------------------
    no_intent = {}
    per_row = []
    for c in camps:
        crows = [r for r in ex if r["campaign_dir"] == c]
        ni = collections.Counter(r["status"] for r in crows if not T.has_intent(r["mechanism_id"], r["statement"]))
        wi = collections.Counter(r["status"] for r in crows if T.has_intent(r["mechanism_id"], r["statement"]))
        no_intent[c] = {"rows": len(crows), "with_intent": sum(wi.values()), "no_intent": sum(ni.values()),
                        "no_intent_by_status": dict(sorted(ni.items())), "with_intent_by_status": dict(sorted(wi.items()))}
    out["no_intent_accounting"] = no_intent

    # ---- structural targets (real do_not_repeat) ------------------------------------------------------
    def tgt(r):
        t, _ = do_not_repeat.structural_target({"mechanism": r["mechanism_id"], "symbol": r["task_key"]["target_symbol"],
                                                "file": r["task_key"]["target_surface"]})
        return t

    def same_epoch(a_, b_):
        if a_["task_key"]["epoch_sha256"] == b_["task_key"]["epoch_sha256"]:
            return True
        me = a_["task_key"]["measurement_epoch"]
        return me is not None and me == b_["task_key"]["measurement_epoch"]

    def repeat_stats(crows, tg, formation_idx, exclude_same_statement):
        n_f = len(formation_idx)
        fset = set(formation_idx)
        ans_idx = collections.defaultdict(list)
        agreeA = agreeB = guard3 = refused = refused_agree = refused_guard = text_cites = 0
        by_status_agree = collections.Counter()
        for i, r in enumerate(crows):
            if i in fset:
                t = tg[i]
                prior = ans_idx.get(t.mechanism, []) if t.is_identified else []
                if exclude_same_statement:
                    prior = [j for j in prior if not (r["statement"] and crows[j]["statement"] == r["statement"])]
                matchB = [j for j in prior if tg[j].agreement(t)[0]]
                matchA = [j for j in matchB if same_epoch(crows[j], r)]
                a_, b_, g_ = bool(matchA), bool(matchB), len(matchA) >= 3
                agreeA += a_
                agreeB += b_
                guard3 += g_
                if r["status"] == "refused_at_formation":
                    refused += 1
                    refused_agree += a_
                    refused_guard += g_
                    text_cites += "do_not_repeat" in (r["outcome"]["refusal_reason"] or "")
                if a_:
                    by_status_agree[r["status"]] += 1
            if r["status"] in ANSWER and tg[i].is_identified:
                ans_idx[tg[i].mechanism].append(i)
        rate = lambda k: (k / n_f if n_f else None)
        return {"formations": n_f, "A_agree_same_epoch": agreeA, "A_rate": rate(agreeA),
                "B_agree_any_epoch": agreeB, "B_rate": rate(agreeB),
                "guard_rule_three_same_epoch_answers": guard3, "guard_rate": rate(guard3),
                "refused_at_formation_formations": refused, "refused_at_formation_rate": rate(refused),
                "refused_rows_whose_text_cites_do_not_repeat": text_cites,
                "refused_rows_structurally_agreeing_A": refused_agree,
                "refused_rows_meeting_guard_rule": refused_guard,
                "agreeing_formations_by_status_A": dict(sorted(by_status_agree.items())),
                "refused_rate_le_A_rate": (refused <= agreeA) if n_f else None}

    rep = {}
    for c in camps:
        crows = [r for r in ex if r["campaign_dir"] == c]
        tg = [tgt(r) for r in crows]
        allf, dedf, seen_stmt = [], [], set()
        for i, r in enumerate(crows):
            mid = r["mechanism_id"]
            if not mid or mid in T.HARNESS_SENTINEL_MECHANISMS or not T.has_intent(mid, r["statement"]):
                continue
            allf.append(i)
            key = r["statement"] if r["statement"] else ("mid", mid)
            if key not in seen_stmt:
                seen_stmt.add(key)
                dedf.append(i)
        rep[c] = {"V1_all_intent_rows_incl_same_statement_matches": repeat_stats(crows, tg, allf, False),
                  "V2_all_intent_rows_excluding_same_statement_matches": repeat_stats(crows, tg, allf, True),
                  "V3_first_occurrence_per_statement": repeat_stats(crows, tg, dedf, False)}
    out["repeat_null_ii"] = {
        "definition": "formation = intent-bearing row with a non-sentinel mechanism_id (V1 every such row; V2 same but an earlier answer row with the "
                      "identical statement text is not allowed to match, to remove a hypothesis agreeing with its own lifecycle rows; V3 first "
                      "occurrence per statement); agreement = do_not_repeat.structural_target(mechanism, symbol, file).agreement against an "
                      "EARLIER row of the same campaign with status in ANSWER_STATUSES; A restricts to same epoch (full epoch or equal "
                      "measurement_epoch, the dispatch_guard rule), B is any epoch; guard_rule needs >= 3 such answers. "
                      "Candidate-diff and research_scope escape clauses of dispatch_guard are not replayable from the export.",
        "per_campaign": rep}

    # ---- faithfulness rho (campaign level; class labels unavailable) ----------------------------------
    def eff(r):
        e = r["outcome"]["effect_fraction"]
        if e is None and r["outcome"]["comparison"]:
            e = r["outcome"]["comparison"].get("effect")
        return e
    pairs = []
    conf_rows = collections.Counter()
    for c in camps:
        crows = [r for r in ex if r["campaign_dir"] == c and r["mechanism_id"]]
        for i, r in enumerate(crows):
            if r["status"] not in CONFIRM_TIER:
                continue
            conf_rows[(c, "confirm_rows")] += 1
            ce = eff(r)
            sha = r["candidate"]["candidate_diff_sha256"]
            cands = [s for s in crows[:i] if s["mechanism_id"] == r["mechanism_id"] and s["status"] in SCREEN_TIER
                     and eff(s) is not None]
            strict = [s for s in cands if sha and s["candidate"]["candidate_diff_sha256"] == sha]
            rec = {"campaign_dir": c, "confirm_status": r["status"], "mechanism_id": r["mechanism_id"],
                   "confirm_row": [r["store_label"], r["rowid"]], "confirm_effect": ce, "candidate_diff_sha256": sha}
            if ce is None:
                rec.update(judgeable=False, reason="confirmed tier has no numeric effect")
            elif not cands:
                rec.update(judgeable=False, reason="no earlier screen-tier row for this mechanism_id in the campaign")
            else:
                s = (strict or cands)[-1]
                se = eff(s)
                rec.update(screen_row=[s["store_label"], s["rowid"]], screen_status=s["status"], screen_effect=se,
                           pairing="strict_same_candidate_diff_sha256" if strict else "mechanism_id_only_candidate_identity_not_established",
                           same_epoch=same_epoch(s, r))
                if se is None or abs(se) < 1e-9:
                    rec.update(judgeable=False, reason="screen effect is zero")
                else:
                    rec.update(judgeable=True, rho=ce / se, unstable_small_screen=abs(se) < 1e-4)
            pairs.append(rec)
    fa = {}
    for c in camps:
        ps = [p for p in pairs if p["campaign_dir"] == c]
        j = [p for p in ps if p["judgeable"]]
        strict = [p for p in j if p["pairing"].startswith("strict")]
        fa[c] = {"confirm_tier_rows": len(ps), "judgeable": len(j), "judgeable_strict_same_candidate": len(strict),
                 "not_judgeable": len(ps) - len(j),
                 "not_judgeable_reasons": dict(collections.Counter(p["reason"] for p in ps if not p["judgeable"])),
                 "rho_all_pairs": [round(p["rho"], 4) for p in j],
                 "rho_sum_ratio": (sum(p["confirm_effect"] for p in j) / sum(p["screen_effect"] for p in j)) if j else None,
                 "rho_sign_flips": sum(1 for p in j if p["rho"] < 0)}
    out["faithfulness_i"] = {
        "definition": "rho = confirm-tier effect / screen-tier effect. Confirm tier = status kept or measured_serving_gate; screen tier = the "
                      "latest EARLIER row of the same campaign and mechanism_id with status keep_candidate, measured_null or regression and a "
                      "numeric effect; strict pairing when candidate_diff_sha256 matches. The export holds one effect per row, not both "
                      "tiers per attempt, so this is a cross-row join. Mechanism class breakdown: not available (taxonomy gate failed).",
        "per_campaign": fa, "pairs": pairs}

    # ---- surface diff distance availability -----------------------------------------------------------
    sd = {}
    for c in camps:
        by_e = collections.Counter(r["task_key"]["epoch_sha256"][:8] for r in ex if r["campaign_dir"] == c and r["payload_flags"]["has_retained_patch"])
        sd[c] = {"rows_with_retained_patch": sum(by_e.values()), "max_in_one_epoch": max(by_e.values()) if by_e else 0,
                 "judgeable": (max(by_e.values()) if by_e else 0) >= W,
                 "note": "patch bytes are not in the export; judgeable would additionally need a store read"}
    out["surface_diff_distance"] = sd

    # ---- descriptive window statistics (non-predeclared, taxonomy-free) --------------------------------
    desc = {}
    for c in camps:
        rng = random.Random(f"{SEED}:{c}")
        eps = collections.defaultdict(lambda: {"mid": [], "file": []})
        seen = set()
        for r in (x for x in ex if x["campaign_dir"] == c):
            if not T.has_intent(r["mechanism_id"], r["statement"]) or r["mechanism_id"] in T.HARNESS_SENTINEL_MECHANISMS:
                continue
            e = r["task_key"]["epoch_sha256"]
            key = (e, r["statement"] if r["statement"] else ("mid", r["mechanism_id"]))
            if key in seen:
                continue
            seen.add(key)
            eps[e]["mid"].append(r["mechanism_id"])
            eps[e]["file"].append(r["task_key"]["target_surface"] or "(none)")
        desc[c] = {"hypotheses_deduped": sum(len(v["mid"]) for v in eps.values()), "epochs": len(eps),
                   "epoch_sizes": sorted((len(v["mid"]) for v in eps.values()), reverse=True),
                   "by_mechanism_id": descriptive_windows({e: v["mid"] for e, v in eps.items()}, rng),
                   "by_target_file": descriptive_windows({e: v["file"] for e, v in eps.items()}, rng)}
    out["descriptive_windows_not_predeclared"] = {
        "note": "Entropy / distinct counts over mechanism_id and over target file, W=20 sliding, deduplicated statements, order = recorded_at; "
                "campaign statistic = window-weighted mean OLS slope of H_t; null = 1000 within-epoch order permutations; below_null_p5 is "
                "reported but is NOT the predeclared verdict (that needs the mechanism taxonomy, which failed its held-out gate).",
        "per_campaign": desc}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "taxfree_results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", a.out_dir / "taxfree_results.json")


if __name__ == "__main__":
    main()
