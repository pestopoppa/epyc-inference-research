#!/usr/bin/env python3
"""Run the R23-44 serving gate ONCE on the seeded bundle, BY HAND.

    python3 -m autokernel.loop.serving_gate \
        --cor-build /mnt/raid0/llm/tmp/build-cor-445e93a8 \
        --tip-build /mnt/raid0/llm/autokernel/loop-memory/anchor-gen-021 \
        [--pairs 5] [--execute] [--apply]

Run ONLY with the loop dead and the GPU claim free, AFTER `seed_bundle --apply` wrote the
bundle with a MEASURED compounded tip-vs-cor. Uses the loop's own `serving.compare`
(llama-server under the canonical recipe, paired alternating, per-request aggregate) and
`accumulate.resolve`, so the record is the same schema the loop would have written
(`epyc.autokernel.serving_ab.v1`) -- a hand-run gate that wrote a different shape would be
evidence nothing downstream could read.

WHAT CHANGED WHEN THIS CAME INTO THE PACKAGE. The scratch original built the floor path
itself (`STORE / f"serving-floor.{recipe.name}.json"`) and read `["floor_pct"]` off it.
That is the READER half of the defect `run.py` was fixed for: a floor is a property of the
measured CONDITION, not of the recipe's NAME, so a hand-run gate could be -- and under
R23-49's cpu_list pin nearly was -- judged against a floor calibrated under another
condition, with nothing raised. It now goes through `serving.load_floor`, which refuses a
mismatch and stamps the provenance onto the record.

Dry by default (`instruments.Posture`). `--execute` spends the gate and writes the
evidence record; `--apply` additionally advances the champion of record on a PROMOTE. The
headline is left to the relaunch's publish (the loop re-publishes on its first keep or
guard).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from . import accumulate, instruments, keep_screen, pool, serving, status


def measure(recipe: serving.Recipe, cor_build: Path, tip_build: Path, *,
            pairs: int, floor_pct: float, floor_unit: str) -> dict:
    """THE MEASUREMENT SEAM -- the only thing here that touches the GPU.

    A thin wrapper on purpose: everything above it (floor identity, the scheduling
    decision, the resolve/record/apply chain) is then testable without hardware, and a
    test that stubs this cannot accidentally launch a server.

    `floor_unit` is the loaded floor's own unit and is passed straight through, so the
    unit check happens where the comparison does and cannot be skipped by a caller that
    forgot (R23-55). This gate's effect is `process`-unit: `serving.compare` relaunches
    the server for every sample of every arm.
    """
    return serving.compare(recipe, cor_build, tip_build, pairs=pairs, floor_pct=floor_pct,
                           floor_unit=floor_unit)


#: e2e designs `classify_e2e_design` can return. Only `ABBA` may confirm an op-level win.
E2E_DESIGNS = ("ABBA", "fixed_order_pairs", "separate_runs", "unpaired", "unknown_design")
_ARMS = ("anchor", "candidate")


def _min_pairs() -> int:
    from . import bench
    return bench.MIN_PAIRS


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def classify_e2e_design(row, declared: dict | None = None) -> tuple[str, list[str]]:
    """Verify, FROM THE RECORD, how an e2e row was measured. Returns (design, reasons);
    `design == "ABBA"` only when every check passes (reasons is then empty).

    ABBA = one invocation (a single measurement-plan seed), both arms sampled pairs times
    with launches interleaved by the plan's per-pair order (anchor-first and
    candidate-first both present), every launch accounted for in `launch_membership`
    (ordinal = 2*pair + position), pairs >= bench.MIN_PAIRS, and any digest in `declared`
    (request_digest, recipe_hash, anchor_build_digest, candidate_build_digest) equal to
    the row's. Missing field => not ABBA (fail closed)."""
    if not isinstance(row, dict):
        return "unknown_design", ["row is not a mapping"]
    # Explicit separate-run markers win: two run/session ids is the definition.
    for kind in ("run_id", "session_id"):
        a, c = row.get(f"anchor_{kind}"), row.get(f"candidate_{kind}")
        if a is not None and c is not None and a != c:
            return "separate_runs", [f"anchor_{kind} != candidate_{kind}"]
    a_s, c_s = row.get("anchor_samples"), row.get("candidate_samples")
    if not (isinstance(a_s, list) and isinstance(c_s, list)) or not a_s or not c_s:
        return "unpaired", ["anchor/candidate samples missing or empty"]
    if len(a_s) != len(c_s):
        return "unpaired", ["arm sample counts differ"]
    n, pairs = len(a_s), row.get("pairs")
    why: list[str] = []
    if not _is_int(pairs) or pairs != n:
        why.append("pairs missing or != samples per arm")
    elif pairs < _min_pairs():
        why.append(f"pairs {pairs} < minimum {_min_pairs()}")
    plan, members = row.get("measurement_plan"), row.get("launch_membership")
    if not isinstance(plan, dict) or not isinstance(members, list):
        # legacy v1 rows: fixed anchor-then-candidate per pair, not interleaved.
        return "fixed_order_pairs", why + ["no measurement_plan/launch_membership: fixed AB order"]
    seed, orders = plan.get("seed"), plan.get("orders")
    if not isinstance(seed, str) or not seed:
        why.append("measurement_plan.seed missing")
    ok_orders = (isinstance(orders, list) and len(orders) == n and all(
        isinstance(o, (list, tuple)) and sorted(o) == sorted(_ARMS) for o in orders))
    if not ok_orders:
        return "unknown_design", why + ["measurement_plan.orders invalid"]
    firsts = {o[0] for o in orders}
    if firsts != set(_ARMS):
        return "fixed_order_pairs", why + ["orders not counterbalanced (one arm always first)"]
    seen = {}
    for m in members:
        if not isinstance(m, dict) or not _is_int(m.get("ordinal")) or not _is_int(m.get("pair_index")):
            return "unknown_design", why + ["launch_membership entry malformed"]
        seen[m["ordinal"]] = (m["pair_index"], m.get("arm"))
    expected = {2 * i + pos: (i, arm) for i, o in enumerate(orders) for pos, arm in enumerate(o)}
    if len(members) != len(expected) or seen != expected:
        why.append("launch_membership does not match the plan's interleaved order")
    for key, val in (declared or {}).items():
        if val is None or row.get(key) != val:
            why.append(f"declared {key} not matched by row")
    return ("ABBA", []) if not why else ("unknown_design", why)


def e2e_evidence(row: dict, declared: dict | None = None) -> dict:
    """Project a `serving.compare` row into e2e evidence for `keep_screen.op_keep_status`.
    The design is VERIFIED from the row (`classify_e2e_design`), never assumed; only a
    verified ABBA carries `same_run=True`. Anything else keeps its true design label and
    the op win stays pending."""
    design, reasons = classify_e2e_design(row, declared)
    row = row if isinstance(row, dict) else {}
    return {"design": design, "same_run": design == "ABBA", "design_reasons": reasons,
            "effect_pct": row.get("effect_pct"),
            "floor_pct": row.get("noise_floor_pct", row.get("floor_pct")),
            "decisive": row.get("decisive")}


def op_win_verdict(op: dict, row: dict | None, declared: dict | None = None) -> dict:
    """Status of an op-level win given an (optional) serving-gate row."""
    return keep_screen.op_keep_status(
        op, e2e_evidence(row, declared) if row is not None else None)


def _ppl_contract_fold_blocked(args, bundle) -> bool:
    """True (fail closed) unless the bundle provably carries no ppl_contract change."""
    from . import gates
    try:
        changed = subprocess.run(
            ["git", "-C", str(args.champion_worktree), "diff", "--name-only",
             bundle.champion_of_record, bundle.tip],
            capture_output=True, text=True, check=True).stdout.splitlines()
        return gates.ppl_contract_fold_required(
            bundle.keeps, changed, gates.ppl_contract_ledger_read(Path(args.store)),
            trailer=gates.ppl_contract_range_requires_gate(
                Path(args.champion_worktree), bundle.champion_of_record, bundle.tip,
                store=Path(args.store)))
    except Exception:  # noqa: BLE001 -- undecidable is blocked
        return True


def _is_ancestor(worktree: Path, ancestor: str, descendant: str) -> bool:
    """Read-only lineage check for the explicitly pinned serving-gate tip."""
    done = subprocess.run(
        ["git", "-C", str(worktree), "merge-base", "--is-ancestor",
         ancestor, descendant], capture_output=True, text=True)
    if done.returncode not in (0, 1):
        raise RuntimeError(
            f"git ancestry check failed in {worktree}: {done.stderr.strip()}"
        )
    return done.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autokernel.loop.serving_gate", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    instruments.add_recipe_args(parser)
    parser.add_argument("--cor-build", type=Path, required=True,
                        help="build of the champion OF RECORD (the anchor arm)")
    parser.add_argument("--tip-build", type=Path, required=True,
                        help="build of the accumulator TIP (the candidate arm)")
    parser.add_argument("--tip", required=True,
                        help="source commit exactly represented by --tip-build")
    parser.add_argument("--surface", choices=accumulate.SURFACES,
                        help="explicit CPU/GPU bundle surface; omitted preserves the legacy store")
    parser.add_argument("--champion-worktree", type=Path, default=pool.CHAMPION_TREE,
                        help="read-only source topology for Bundle ancestry checks")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--fire-multiple", type=float, default=2.5)
    instruments.add_posture_args(
        parser, apply_help="on PROMOTE, advance the bundle's champion of record and save "
                           "a fresh bundle.")
    parser.add_argument("--force", action="store_true",
                        help="ONE-OFF operator permission (2026-09-08 fold window): spend "
                             "the serving gate even if the bundle is below fire_multiple x "
                             "floor. The loop's own scheduling rule is unchanged; promotion "
                             "still requires the serving A/B to clear the calibrated floor.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    posture = instruments.resolve_posture(args)
    try:
        recipe = serving.Recipe.load(args.recipe)
        print(f"recipe    {recipe.describe()}")
        # The effect this gate produces is between-PROCESS (a fresh server per sample), so
        # only a process-unit floor can be its bar; a legacy unit-less floor refuses here,
        # before any GPU time is spent (R23-55).
        reading = instruments.read_floor(args.store, recipe,
                                         effect_unit=serving.COMPARE_EFFECT_UNIT)
        floor = reading.floor_pct
        bundle, _ = accumulate.load_bundle(
            Path(args.store), anchor_commit=args.tip,
            is_ancestor=lambda older, newer: _is_ancestor(
                args.champion_worktree, older, newer),
            read_only=posture.dry_run, surface=args.surface)
        policy = accumulate.AccumulatorPolicy(fire_multiple=args.fire_multiple)
        thr = policy.fire_threshold_pct(floor)
        print(f"floor     {floor:.3f}% [{reading.provenance}] unit={reading.unit} "
              f"n={reading.n} from {reading.path.name} | fire threshold {thr:.2f}%")
        validity = accumulate.effective_measurement_validity(bundle, Path(args.store))
        if validity == accumulate.MEASUREMENT_STALE_BASELINE:
            raise accumulate.BundleRecoveryRequired("stale_baseline: compounded gain cannot be quoted")
        magnitude_label = ("current combined measurement" if validity == accumulate.MEASUREMENT_CURRENT
                           else "historical-only magnitude; threshold disabled")
        print(f"bundle    cor {bundle.champion_of_record[:12]} tip {bundle.tip[:12]} "
              f"keeps {len(bundle.keeps)} compounded {bundle.compounded_bench_pct:+.3f}% "
              f"validity={validity} ({magnitude_label})")
        decision = accumulate.decide_after_keep(bundle, floor, policy, store=Path(args.store))
        print(f"schedule  decide_after_keep -> {decision}  (fire_multiple x floor = "
              f"{thr:.2f}% is only the loop's SCHEDULING heuristic)")
        if decision is not accumulate.Decision.FIRE_SERVING and not args.force:
            print("bundle does not clear the scheduling threshold and --force not given. "
                  "Nothing measured.")
            return instruments.REFUSED
        # Operator 2026-09-08: "just make sure it passes the noise floor". In the fold
        # window the gate is spent regardless of the scheduling threshold; PROMOTE still
        # requires the serving A/B to be DECISIVE, i.e. |effect| >= the calibrated serving
        # floor AND positive -- that check lives in `accumulate.resolve()` and is not
        # bypassable here.
        cor_binary = instruments.require_binary(args.cor_build, "llama-server")
        tip_binary = instruments.require_binary(args.tip_build, "llama-server")
        print(f"builds    anchor {cor_binary}\n          candidate {tip_binary}")
        print(f"posture   {posture.describe()}")
        if posture.dry_run:
            print(f"\nDRY RUN -- would run {args.pairs} alternating serving pairs against "
                  f"floor {floor:.3f}%. Pass --execute to spend the gate.")
            return 0
        started = time.time()
        row = measure(recipe, args.cor_build, args.tip_build,
                      pairs=args.pairs, floor_pct=floor, floor_unit=reading.unit)
    except (instruments.InstrumentRefusal, serving.ServingFloorMismatch,
            accumulate.BundleRecoveryRequired) as refusal:
        # A floor calibrated under a DIFFERENT recipe is a refusal, not a verdict: it
        # exits REFUSED with the message rather than a traceback, so the caller can tell
        # "no reading" from "a bad reading".
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    try:
        plan = accumulate.resolve(bundle, row, policy, store=Path(args.store))
    except accumulate.BundleRecoveryRequired as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    record = {"outcome": plan["outcome"].value, "reason": plan["reason"],
              "bundled_keeps": list(bundle.keeps),
              "planner_evidence": plan.get("planner_evidence"),
              "hand_run": "autokernel.loop.serving_gate",
              # The provenance of the bar travels with the verdict: a reader cannot
              # otherwise tell a checked floor from an assumed one (run.py does the same).
              "floor_provenance": reading.provenance,
              # The bar's unit and n travel with the verdict for the same reason its
              # provenance does: a reader cannot otherwise tell which dispersion this
              # effect was judged against (R23-55 / R23-61).
              "floor_unit": reading.unit, "floor_n": reading.n, **row}
    written = status.write_json(Path(args.store) / "serving",
                                f"bundle-{bundle.tip[:12]}.json", record, prefix=".sv-")
    print(f"serving   {plan['reason']}  [{time.time() - started:.0f}s]  record: {written}")
    print(f"effect    anchor {row['anchor_tok_s']:.2f} tok/s vs tip "
          f"{row['candidate_tok_s']:.2f} tok/s -> {row['effect_pct']:+.2f}% "
          f"(floor {floor:.3f}%, decisive={row['decisive']})")
    if plan["outcome"] is accumulate.Outcome.PROMOTE:
        if posture.apply and _ppl_contract_fold_blocked(args, bundle):
            # Review 2026-10-06 (layer e): this hand-run gate measures throughput only. A
            # bundle carrying a ppl_contract keep may fold only through the loop, whose
            # fold re-runs the quality layers on the whole bundle.
            print("REFUSED: bundle carries a ppl_contract change; the hand-run gate has no "
                  "quality eval -- champion of record NOT advanced", file=sys.stderr)
            return instruments.REFUSED
        if posture.apply:
            fresh = accumulate.Bundle(champion_of_record=bundle.tip, tip=bundle.tip,
                                      surface=bundle.surface)
            try:
                if fresh.surface is None:
                    fresh.save(Path(args.store))
                else:
                    fresh.promote_shared_cor(
                        Path(args.store), expected_cor=bundle.champion_of_record)
            except accumulate.BundleRecoveryRequired as refusal:
                print(f"REFUSED: shared COR promotion failed: {refusal}", file=sys.stderr)
                return instruments.REFUSED
            print("bundle reset and saved:", json.dumps(fresh.to_dict()))
            print("PROMOTE: champion of record advances to", bundle.tip[:12])
        else:
            print("PROMOTE (dry): champion of record would advance to", bundle.tip[:12])
            print("(dry: pass --apply to advance the champion of record)")
    else:
        print("HOLD: champion of record unchanged; bundle kept; divergence is planner "
              "evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
