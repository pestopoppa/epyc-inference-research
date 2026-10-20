"""AK-HR-2 stage 1: rule parser vs independent labeller on the 40 held-out DS41/Q38FN hypotheses.

The labeller (an LLM agent, not a human) labelled ``heldout40_for_labeller.jsonl`` before this script ran
the parser on them; the labels file is read here for the first time together with the parser output.
Gate (operator-confirmed): stop if Cohen's kappa < 0.6 or the parser's ``other`` share > 50%.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ak_hr2_taxonomy as T  # noqa: E402

KAPPA_MIN = 0.6
OTHER_MAX = 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", type=Path, required=True)   # heldout40_for_labeller.jsonl
    ap.add_argument("--index", type=Path, required=True)     # heldout40_index.json (campaign/rowid per hid)
    ap.add_argument("--labels", type=Path, required=True)    # heldout40_labels_independent.json
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    held = [json.loads(l) for l in a.heldout.read_text().splitlines()]
    index = {r["hid"]: r for r in json.loads(a.index.read_text())}
    lab = json.loads(a.labels.read_text())
    labels = lab["labels"]
    per_row, ind, par = [], [], []
    for h in held:
        hid = h["hid"]
        cls, sc = T.classify(h["mechanism_id"], h["target_symbol"], h["target_surface"], h["statement"])
        g = labels[str(hid)]
        ind.append(g)
        par.append(cls)
        per_row.append({"hid": hid, **index[hid], "mechanism_id": h["mechanism_id"],
                        "independent_label": g, "rule_label": cls, "agree": g == cls,
                        "rule_top_score": sc.get(cls) if cls != "other" else max(sc.values()),
                        "note": (lab.get("notes") or {}).get(str(hid))})
    kappa = T.cohen_kappa(par, ind)
    agree = sum(x == y for x, y in zip(par, ind))
    other_rule = sum(c == "other" for c in par) / len(par)
    other_ind = sum(c == "other" for c in ind) / len(ind)
    conf = collections.Counter((i, p) for i, p in zip(ind, par))
    by_campaign = {}
    for r in per_row:
        c = by_campaign.setdefault(r["campaign_dir"], [0, 0])
        c[0] += 1
        c[1] += r["agree"]
    gate_pass = kappa >= KAPPA_MIN and other_rule <= OTHER_MAX
    out = {
        "schema": "ak-hr-2.heldout_agreement.v1",
        "labeller": lab.get("labeller"),
        "caveat": "The independent labeller is an LLM agent, not a human; kappa is rule-parser vs independent-LLM-labeller agreement.",
        "n": len(held), "observed_agreement": agree / len(held), "cohen_kappa": kappa,
        "other_share_rule": other_rule, "other_share_independent": other_ind,
        "gate": {"kappa_min": KAPPA_MIN, "other_max": OTHER_MAX, "pass": gate_pass},
        "agreement_by_campaign": {k: {"n": v[0], "agree": v[1]} for k, v in by_campaign.items()},
        "label_counts": {"independent": dict(collections.Counter(ind)), "rule": dict(collections.Counter(par))},
        "confusion_independent_rule": {f"{i}|{p}": n for (i, p), n in sorted(conf.items())},
        "input_sha256": {"heldout": sha(a.heldout), "index": sha(a.index), "labels": sha(a.labels),
                         "parser": sha(Path(T.__file__))},
        "per_row": per_row,
    }
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("n", "observed_agreement", "cohen_kappa", "other_share_rule",
                                          "other_share_independent", "gate")}, indent=1))
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
