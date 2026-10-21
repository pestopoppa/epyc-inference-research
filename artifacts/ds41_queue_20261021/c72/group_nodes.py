"""DS41-C7.2 grouping: break GGML_OP_TOP_K and the lightning indexer out of a `ggml-cpu-prof/1` node dump.

usage: group_nodes.py --ctx 8192=<dir> --ctx 32768=<dir> ... --out <result.json> [--md <table.md>]
  <dir> holds ggml-cpu-prof.json (GGML_CPU_PROF_JSON_FILE of the profiled sibling) and meta.json
  {n_tokens, n_decode, skip, n_ubatch, threads, prefill_s, binary}.

Groups (node wall is per accumulated graph eval, so share = node.wall_us / dump.total_wall_us, the DECODE step):
  topk_index_source : op TOP_K and name/label `lid_top_k*`          (one per index-source layer, expected 8)
  topk_candidate    : every other op TOP_K                           (the 2048-block candidate stage)
  lid_kq            : name/label `lid_kq*`                           (node COUNT must equal 8: reuse layers must not recompute)
  indexer_other     : name/label `lid_*` that is neither lid_kq nor lid_top_k, plus any LIGHTNING_INDEXER op
  indexer_total     : lid_kq + indexer_other + topk_index_source     (everything the indexer owns)
Thread occupancy (needs GGML_CPU_PROF_THREADS=1, per_thread_compute true): effective_threads = thr_mean*n_threads/thr_max
(= sum of per-thread compute / the slowest thread). A single-row CPU top-k (std::partial_sort, nrows = 1) reads ~1.0.
Gates reported, never silently passed: single accumulated shape, accumulated evals <= n_decode, lid_kq count == 8.
Shares transfer from the instrumented sibling; absolute times do not (profiling perturbs them).
"""
import argparse, json, re, sys

LID_KQ = re.compile(r"^lid_kq(?![A-Za-z0-9])")
LID_TOPK = re.compile(r"^lid_top_k(?![A-Za-z0-9])")
LID_ANY = re.compile(r"^lid_")


def ident(n):
    """The node's cb() name; the dump's `name` is the tensor name (`lid_kq-12`), `label` the cb label. Match either."""
    return [s for s in (n.get("name") or "", n.get("label") or "") if s]


def match(rx, n):
    return any(rx.match(s) for s in ident(n))


def classify(n):
    op = n.get("op", "")
    if op == "TOP_K":
        return "topk_index_source" if match(LID_TOPK, n) else "topk_candidate"
    if match(LID_KQ, n):
        return "lid_kq"
    if match(LID_ANY, n) or op == "LIGHTNING_INDEXER":
        return "indexer_other"
    return None


def one(label, d):
    prof = json.load(open(f"{d}/ggml-cpu-prof.json"))
    meta = json.load(open(f"{d}/meta.json"))
    assert prof.get("schema") == "ggml-cpu-prof/1", f"{d}: schema {prof.get('schema')}"
    total = prof["total_wall_us"]
    nth = prof.get("n_threads", meta.get("threads", 0))
    per_thread = bool(prof.get("per_thread_compute"))
    groups = {k: {"nodes": 0, "wall_us": 0.0, "compute_us": 0.0, "dead_us": 0.0, "_eff": [], "_names": []}
              for k in ("topk_index_source", "topk_candidate", "lid_kq", "indexer_other")}
    for n in prof.get("nodes", []):
        g = classify(n)
        if g is None:
            continue
        r = groups[g]
        r["nodes"] += 1
        r["wall_us"] += n["wall_us"]; r["compute_us"] += n["compute_us"]; r["dead_us"] += n.get("dead_us", 0.0)
        if per_thread and n.get("thr_max_us", 0) > 0:
            r["_eff"].append(n["thr_mean_us"] * nth / n["thr_max_us"])
        if len(r["_names"]) < 40:
            r["_names"].append(ident(n)[0] if ident(n) else f"idx{n['idx']}")
    tk = {k: groups[k] for k in groups}
    tk["indexer_total"] = {"nodes": groups["lid_kq"]["nodes"] + groups["indexer_other"]["nodes"] + groups["topk_index_source"]["nodes"],
                           "wall_us": sum(groups[k]["wall_us"] for k in ("lid_kq", "indexer_other", "topk_index_source")),
                           "compute_us": sum(groups[k]["compute_us"] for k in ("lid_kq", "indexer_other", "topk_index_source")),
                           "dead_us": 0.0, "_eff": [], "_names": []}
    tk["topk_all"] = {"nodes": groups["topk_index_source"]["nodes"] + groups["topk_candidate"]["nodes"],
                      "wall_us": groups["topk_index_source"]["wall_us"] + groups["topk_candidate"]["wall_us"],
                      "compute_us": groups["topk_index_source"]["compute_us"] + groups["topk_candidate"]["compute_us"],
                      "dead_us": groups["topk_index_source"]["dead_us"] + groups["topk_candidate"]["dead_us"],
                      "_eff": groups["topk_index_source"]["_eff"] + groups["topk_candidate"]["_eff"], "_names": []}
    out = {}
    for k, r in tk.items():
        eff = r.pop("_eff"); names = r.pop("_names")
        r["share_of_decode_wall"] = r["wall_us"] / total
        r["share_of_decode_compute"] = r["compute_us"] / prof["total_compute_us"] if prof.get("total_compute_us") else None
        r["effective_threads_mean"] = (sum(eff) / len(eff)) if eff else None
        r["effective_threads_max"] = max(eff) if eff else None
        r["thread_occupancy_measured"] = bool(eff)
        r["example_names"] = names[:6]
        out[k] = r
    shapes = prof.get("shapes", [])
    acc_shapes = [s for s in shapes if s.get("accumulated", 0) > 0]
    G = prof["graph_evals_accumulated"]
    gates = {
        "single_accumulated_shape": len(acc_shapes) == 1,
        "accumulated_evals_le_n_decode": G <= meta["n_decode"],
        "accumulated_evals": G, "n_decode_requested": meta["n_decode"], "skip": meta["skip"],
        "lid_kq_count": groups["lid_kq"]["nodes"], "lid_kq_count_must_equal": 8,
        "lid_kq_count_ok": groups["lid_kq"]["nodes"] == 8,
        "node_names_matched": groups["lid_kq"]["nodes"] + groups["topk_index_source"]["nodes"] > 0,
        "per_thread_compute": per_thread,
    }
    gates["all_ok"] = all(gates[k] for k in ("single_accumulated_shape", "accumulated_evals_le_n_decode", "lid_kq_count_ok", "node_names_matched"))
    return {"label": label, "n_tokens": meta["n_tokens"], "threads": nth, "prefill_s": meta.get("prefill_s"),
            "binary": meta.get("binary"), "decode_wall_us_per_eval": total, "n_nodes": prof.get("n_nodes"),
            "groups": out, "gates": gates,
            "limitations": ["shares from an INSTRUMENTED sibling build; absolute times are perturbed and not a baseline",
                            "prefill graphs excluded by GGML_CPU_PROF_SKIP; profiled prefill timing is inadmissible",
                            "indicative host, not a quiet window"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", action="append", required=True, help="<label>=<dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    res = [one(*c.split("=", 1)) for c in a.ctx]
    json.dump({"schema": "ds41.c72.v1", "results": res}, open(a.out, "w"), indent=1)
    lines = ["| ctx | top-k share (all) | top-k index-source | top-k candidate | indexer total | lid_kq nodes (=8) | top-k eff. threads | gates |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        g = r["groups"]
        eff = g["topk_all"]["effective_threads_mean"]
        lines.append("| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {} | {} | {} |".format(
            r["n_tokens"], g["topk_all"]["share_of_decode_wall"], g["topk_index_source"]["share_of_decode_wall"],
            g["topk_candidate"]["share_of_decode_wall"], g["indexer_total"]["share_of_decode_wall"], r["gates"]["lid_kq_count"],
            "n/a" if eff is None else f"{eff:.2f}", "ok" if r["gates"]["all_ok"] else "FAIL " + json.dumps({k: v for k, v in r["gates"].items() if v is False})))
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
