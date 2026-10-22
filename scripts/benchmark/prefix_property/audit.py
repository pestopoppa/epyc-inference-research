#!/usr/bin/env python3
"""Run P1/P2/P3 for every registry role's template over every fixture (template + vocab only).

    /usr/bin/python3 scripts/benchmark/prefix_property/audit.py [--registry PATH] [--json OUT] [--thinking]

Always `enable_thinking=false` (UFH14-LB5). Thinking-ON is not audited: with the thinking opener the model's
reply carries reasoning text no fixture here can honestly supply. Exit 0 always: the audit REPORTS;
tests/test_live_engine.py pins the expected outcomes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

REGISTRY = "/workspace/repos/epyc-orchestrator/orchestration/model_registry.yaml"


def audit_entry(entry, fixtures, enable_thinking=False):
    eng, tok, eog = h.engine_for(entry)
    wire = h.tool_wire(eng.template_text)
    rows = {"id": entry["id"], "roles": entry["roles"], "served": entry.get("served", True),
            "enable_thinking": enable_thinking, "eog": eog, "tool_wire": wire, "template_bytes": len(eng.template_text), "fixtures": {}}
    for fx in fixtures:
        if len(fx["messages"]) < 3:
            continue
        p1 = h.check_reply_prefix(eng, fx, eog=eog, enable_thinking=enable_thinking, wire=wire)
        p2 = h.check_token_prefix(eng, tok, fx, eog=eog, enable_thinking=enable_thinking, wire=wire)
        rows["fixtures"][fx["id"]] = {
            "P1_reply_prefix": [{"turn": f.turn, "ok": f.ok, "kind": f.kind, "at": f.first_diff, "brief": f.brief()} for f in p1],
            "P2_token_prefix": [{"turn": f.turn, "ok": f.ok, "kind": f.kind, "at": f.first_diff} for f in p2]}
    a, b = h.load_fixture("agent_tool_loop"), h.load_fixture("agent_tool_loop_b")
    t = h.check_shared_trunk(eng, tok, a, b, enable_thinking=enable_thinking)
    rows["P3_shared_trunk"] = {"ok": t.ok, "kind": t.kind, **t.detail}
    rows["verdict"] = {"P1": all(f["ok"] for fx in rows["fixtures"].values() for f in fx["P1_reply_prefix"]),
                       "P2": all(f["ok"] for fx in rows["fixtures"].values() for f in fx["P2_token_prefix"]),
                       "P3": t.ok}
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    fixtures = h.all_fixtures()
    out = []
    for entry in h.role_templates(a.registry):
        r = audit_entry(entry, fixtures, False)
        out.append(r)
        v = r["verdict"]
        print(f"{r['id']:78} served={int(r['served'])} wire={r['tool_wire']} P1={v['P1']} P2={v['P2']} P3={v['P3']}  roles={r['roles']}")
        for fid, fx in r["fixtures"].items():
            bad = [f for f in fx["P1_reply_prefix"] if not f["ok"]]
            if bad:
                print(f"      {fid}: {bad[0]['brief']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
