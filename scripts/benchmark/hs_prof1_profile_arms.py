#!/usr/bin/env python3
"""HS-PROF-1: build the profile snapshots for the BEAM extra arm (admission schema vs PREF format).

Two subcommands.

``derive``  (INFERENCE: one deriver LLM call per conversation; refuses unless WINDOW_CONFIRMED=1)
    Runs the orchestrator deriver prompt over each BEAM conversation transcript against a local
    OpenAI-compatible endpoint and stores the raw ``PREF [category] text`` output per conversation.
    The deriver truncates the transcript to its first 8,000 characters (its own behaviour,
    identical for both arms).

``build``   (offline, no model)
    Replays the stored deriver output through the orchestrator ``ProfileStore`` twice:
      arm ``pref_current``  no admission policy (today's behaviour)
      arm ``admission``     ``AdmissionPolicy()`` (kinds, stability, no behavioural/inferred, cap)
    and writes one ``{conversation_id: frozen_snapshot}`` JSON per arm. Feed each to the BEAM adapter:
      BEAM_CONTEXT_MODE=profile BEAM_PROFILE_SNAPSHOTS=<json> BEAM_PROFILE_ARM=<arm> \
      BEAM_ABILITIES=preference_following,instruction_following

Needs the orchestrator checkout on ``EPYC_ORCH_ROOT`` (default /workspace/repos/epyc-orchestrator).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ARMS = ("pref_current", "admission")


def _orch():
    root = os.environ.get("EPYC_ORCH_ROOT", "/workspace/repos/epyc-orchestrator")
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.user_modeling.admission import AdmissionPolicy
    from src.user_modeling.deriver import _DERIVE_PROMPT, extract_preferences_from_text
    from src.user_modeling.profile_store import ProfileStore
    return AdmissionPolicy, _DERIVE_PROMPT, extract_preferences_from_text, ProfileStore


def build_snapshots(derived: dict[str, str]) -> dict[str, dict[str, str]]:
    """{arm: {conversation_id: snapshot}} from {conversation_id: raw deriver output}."""
    AdmissionPolicy, _, extract, ProfileStore = _orch()
    out: dict[str, dict[str, str]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for cid, raw in derived.items():
            with tempfile.TemporaryDirectory() as td:
                store = ProfileStore(Path(td) / "p.db",
                                     admission=AdmissionPolicy() if arm == "admission" else None)
                for fact in extract(raw):
                    store.add_fact(cid, fact)
                out[arm][cid] = store.frozen_snapshot(cid)
    return out


def derive(base_url: str, model: str, data_dir: Path | None, split: str, timeout: float = 1800.0) -> dict[str, str]:
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        raise SystemExit("refusing: derive calls a model; set WINDOW_CONFIRMED=1 in a granted inference window")
    _, prompt_t, _, _ = _orch()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from long_context_adapters import BEAMAdapter

    ad = BEAMAdapter(data_dir=data_dir, split=split, context_mode="full")
    ad.extract_all()
    out = {}
    for cid, transcript in ad._transcripts.items():
        body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 1024, "stream": False,
                           "messages": [{"role": "user", "content": prompt_t.format(transcript=transcript[:8000])}]}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - operator-set local URL
            out[cid] = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    return out


ARMS = ("pref_current", "admission")


def table(out_dir: Path) -> dict:
    """Judge-scored arm delta: per ability, the two arm scores, the delta (admission minus pref_current) and the
    PAIRED per-question win/loss/tie counts over the question ids both arms scored. Report-only: no verdict is
    derived here (HS-PROF-1 states no threshold); n is always shown. Missing arms or an empty overlap raise."""
    scored = {}
    for arm in ARMS:
        f = out_dir / f"scored_{arm}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}: run the judge phase for both arms")
        scored[arm] = json.loads(f.read_text())
    a, b = (scored[x]["per_question"] for x in ARMS)
    qa = {r["question_id"]: r for r in a}
    qb = {r["question_id"]: r for r in b}
    shared = sorted(set(qa) & set(qb))
    if not shared:
        raise SystemExit("the two arms share no question ids")
    out = {"schema": "epyc.hs_prof1_table.v1", "arms": list(ARMS), "delta": "admission minus pref_current",
           "n_shared_questions": len(shared), "only_in_one_arm": len(set(qa) ^ set(qb)), "per_ability": {}}
    for ability in sorted({qa[q]["ability"] for q in shared}):
        ids = [q for q in shared if qa[q]["ability"] == ability]
        d = [qb[q]["score"] - qa[q]["score"] for q in ids]
        out["per_ability"][ability] = {
            "n": len(ids),
            "pref_current": sum(qa[q]["score"] for q in ids) / len(ids),
            "admission": sum(qb[q]["score"] for q in ids) / len(ids),
            "mean_delta": sum(d) / len(d),
            "admission_wins": sum(1 for x in d if x > 0), "pref_wins": sum(1 for x in d if x < 0), "ties": sum(1 for x in d if x == 0)}
    out["judge_models"] = sorted({str(scored[x]["summary"].get("judge_model")) for x in ARMS})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--base-url", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--split", default="100K")
    d.add_argument("--data-dir", type=Path)
    d.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--derived", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    t = sub.add_parser("table")
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "table":
        res = table(a.out_dir)
        a.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["per_ability"], indent=1))
        return 0
    if a.cmd == "derive":
        a.out.write_text(json.dumps(derive(a.base_url, a.model, a.data_dir, a.split), indent=1))
        return 0
    snaps = build_snapshots(json.loads(a.derived.read_text()))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for arm, m in snaps.items():
        (a.out_dir / f"snapshots_{arm}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({arm: {"conversations": len(m), "empty": sum(1 for v in m.values() if not v),
                            "chars": sum(len(v) for v in m.values())} for arm, m in snaps.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
