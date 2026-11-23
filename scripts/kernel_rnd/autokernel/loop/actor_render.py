"""Prompt/context rendering of the actor seats, moved verbatim from actors.py (AK-SPLIT-ACTORS)."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from . import gpu_counter_signals
from . import sampling_axis

def _cpu_target(context: Mapping[str, Any]) -> bool:
    target = context.get("target")
    recipe = target.get("recipe") if isinstance(target, Mapping) else None
    return isinstance(recipe, Mapping) and recipe.get("backend") == "cpu"


#: Section header of the rendered node profile; `actor_context.SECTION_HEADERS` names
#: it too, so the variable-mode splitter files it as its own section.
NODE_PROFILE_HEADER = ("## Node profile — per-op wall SHARES on an instrumented sibling "
                       "of the same anchor")


def _pct(value: Any) -> str:
    return (f"{value * 100:.2f}%" if isinstance(value, (int, float))
            and not isinstance(value, bool) else "n/a")


def _num(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:g}" if float(value).is_integer() else f"{value:.3g}"


def _render_gpu_serving_profile(observation: Mapping[str, Any], *, limit: int = 12) -> list[str]:
    """G2 (2026-10-04): the selected GPU serving target's own rocprofv3 windows.

    One whole-lifetime trace of the target's server, windows cut by timestamp; per kernel
    the device-time share plus registers, spills, scratch, LDS and a gfx90a occupancy
    estimate from the build's own code objects. Skipped windows are printed with their
    reason so the planner never reads absence as "no time there"."""
    lines = ["", "## GPU serving profile — the target's own server under rocprofv3, by window"]
    if observation.get("status") != "observed":
        lines.append(f"GPU serving profile {observation.get('status')}: "
                     f"{observation.get('reason', 'not collected')}")
        return lines
    lines.append(f"Record: {observation.get('record')}; clock {observation.get('clock')}; "
                 f"HIP proof {observation.get('hip_proof')}; primary window "
                 f"`{observation.get('primary_window')}`. Occupancy is an ESTIMATE "
                 "(waves/SIMD from VGPR+AGPR, SGPR and LDS), not a counter.")
    weighted = observation.get("launch_weighted") or {}
    if weighted.get("status") == "observed":
        # AKX-ALL-29.2a: what to attack first, summed over windows; ranking only.
        lines.append(f"### Launch-weighted kernel order — launches/token x mean time at "
                     f"shape, summed over {', '.join(weighted.get('windows') or [])}")
        lines.append("Ranks which kernel to attack and which candidate is timed first; a "
                     "keep still needs a measured end-to-end win.")
        lines.append("| ns/token | kernel | per window (launches/token x mean µs) |")
        lines.append("|---|---|---|")
        for row in (weighted.get("kernels") or [])[:limit]:
            cells = "; ".join(
                f"{w}: {c.get('launches_per_token')} x {c.get('mean_us')}"
                for w, c in (row.get("by_window") or {}).items())
            lines.append(f"| {row['weighted_ns_per_token']} | `{row['kernel']}` | {cells} |")
    elif weighted:
        lines.append(f"Launch-weighted kernel order unavailable ({weighted.get('reason')}); "
                     "the primary window's device-time order is used.")
    for name, window in (observation.get("windows") or {}).items():
        status = window.get("status")
        if status != "observed":
            lines.append(f"- `{name}`: {status} — {window.get('reason', '')}")
            continue
        rows = (window.get("kernels") or [])[:limit if name == observation.get("primary_window")
                                             else max(4, limit // 2)]
        lines.append(f"### `{name}` — {window.get('window_s')} s, busy "
                     f"{(window.get('busy_fraction') or 0) * 100:.1f}%, "
                     f"{window.get('dispatches')} dispatches")
        lines.append("| share | calls | mean µs | VGPR/AGPR | SGPR | spills v/s | scratch B | "
                     "LDS B | waves/SIMD (limiter) | kernel |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            waves = row.get("waves_per_simd_est")
            lines.append(
                f"| {row['share'] * 100:.2f}% | {row['calls']} | {row['mean_us']} | "
                f"{row.get('vgpr')}/{row.get('agpr')} | {row.get('sgpr')} | "
                f"{row.get('vgpr_spill')}/{row.get('sgpr_spill')} | {row.get('scratch_bytes')} | "
                f"{row.get('lds_bytes')} | "
                f"{'' if waves is None else str(waves) + ' (' + str(row.get('occupancy_limiter')) + ')'} | "
                f"`{row['kernel']}` |")
    return lines


#: 2026-10-06 autonomy self-check (`roofline_coverage.py`): where the measured
#: bandwidth headroom actually is, per NODE (not the prefill-pooled path aggregate
#: above), and whether `gates.py` admits ANY route that could touch it today.
ROOFLINE_GAP_HEADER = ("### ROOFLINE GAP (where headroom is) -- per-node achieved "
                       "bandwidth vs the host ceiling, cross-checked against admitted "
                       "routes")


def _render_roofline_gaps(roofline_gaps: Mapping[str, Any] | None, *,
                          limit: int = 12) -> list[str]:
    """The planner's replacement for raw path-level share on bandwidth-bound nodes.

    `roofline_gaps` is `roofline_coverage.from_node_dump`'s output, attached by
    `node_profile.section()` only when a host ceiling was configured and the dump
    carried per-node rows. Absent (not collected, or the dump predates per-node
    rows) prints nothing -- this is an addition to the node-profile section, never a
    required one.
    """
    if not isinstance(roofline_gaps, Mapping) or not roofline_gaps.get("gap_table"):
        return []
    lines = ["", ROOFLINE_GAP_HEADER,
             f"Ceiling: {_num(roofline_gaps.get('ceiling_gb_s'))} GB/s (host memory-bandwidth "
             "ceiling, configured not measured here). gap share = time share x "
             "(1 - achieved/ceiling); `route` is `covered` only when `gates.py`'s OWN "
             "admitted-route tables name the resolved (file, symbol) -- never this "
             "module's own guess.",
             "| gap share | achieved GB/s | op | type | route | covered |",
             "|---|---|---|---|---|---|"]
    for row in roofline_gaps["gap_table"][:limit]:
        symbol = row.get("symbol")
        route = f"`{row.get('file')}` ({symbol})" if row.get("file") else "unresolved"
        covered = row.get("covered")
        covered_text = "yes" if covered is True else ("NO" if covered is False else "unknown")
        lines.append(f"| {_pct(row.get('gap_share'))} | {_num(row.get('achieved_gb_s'))} | "
                     f"`{row.get('op')}` | `{row.get('src0_type')}` | {route} | {covered_text} |")
    uncovered = roofline_gaps.get("uncovered_gap") or {}
    if uncovered.get("triggered"):
        lines.append("")
        lines.append(f"**UNCOVERED GAP: {_pct(uncovered.get('uncovered_fraction'))} of the "
                     "top-K gap share above sits in nodes with NO admitted route.** Widening "
                     "a route is an explicit operator decision (CLAUDE.md, Experimental "
                     "Kernel Workflow) -- do not author a patch against these files/symbols "
                     "until one is admitted: "
                     + "; ".join(f"`{item.get('file')}` ({item.get('symbol')})"
                                 for item in uncovered.get("uncovered_nodes", [])[:6]
                                 if item.get("file")) + ".")
    return lines


def _render_node_profile(observation: Mapping[str, Any], *, limit: int = 12) -> list[str]:
    """`node_profile.section()` as the planner reads it: SHARES and ratios only.

    Absolute microseconds are deliberately not printed -- the instrument's own first
    limitation is that absolutes do not transfer from the instrumented sibling to the
    measured binary. An absent part prints its reason, never a zero."""
    lines = ["", NODE_PROFILE_HEADER]
    status = observation.get("status")
    lines.append("Per-op, host-phase and engram counters from an INSTRUMENTED SIBLING build "
                 "of the current anchor (same commit, same frozen requests), never from the "
                 "measured binary. Shares transfer to the measured binary; absolute times do "
                 "not. Never a baseline, never an arm, never comparable to a measured number. "
                 "An absent part is a missing instrument, never a zero.")
    build = observation.get("build") if isinstance(observation.get("build"), Mapping) else {}
    if build:
        lines.append(f"Sibling build: `{build.get('dir')}` (recipe `{build.get('recipe')}`, "
                     f"anchor `{build.get('anchor_commit')}`); teardown: "
                     f"{observation.get('teardown')}.")
    if status != "observed":
        lines.append(f"Node profile {status or 'absent'}: "
                     f"{observation.get('reason', 'no instrumented observation collected')}")
    families = observation.get("mechanism_shares") or []
    if families:
        lines.append("")
        lines.append("### Mechanism families by wall share (the op-level view of the "
                     "sampled CPU-profile families above)")
        lines.append("| rank | wall share | mechanism family | ops |")
        lines.append("|---|---|---|---|")
        for rank, row in enumerate(families[:limit], 1):
            ops = ", ".join(str(op.get("op")) for op in row.get("ops") or [])
            lines.append(f"| {rank} | {_pct(row.get('wall_fraction'))} | "
                         f"`{row.get('family')}` | {ops} |")
    ops = observation.get("ranked_op_shares") or []
    if ops:
        node = observation.get("node") if isinstance(observation.get("node"), Mapping) else {}
        lines.append("")
        lines.append(f"### Ops by wall share ({_num(node.get('graph_evals_accumulated'))} "
                     f"accumulated graph evals, {_num(node.get('n_threads'))} threads; wall "
                     "includes barrier wait and straggler imbalance, so compute/wall < 1 is "
                     "time spent waiting)")
        lines.append("| wall share | compute/wall | op | calls per eval |")
        lines.append("|---|---|---|---|")
        for row in ops[:limit]:
            wall, compute = row.get("wall_us"), row.get("compute_us")
            ratio = (compute / wall if isinstance(wall, (int, float)) and wall
                     and isinstance(compute, (int, float)) else None)
            lines.append(f"| {_pct(row.get('wall_fraction'))} | {_num(ratio)} | "
                         f"`{row.get('op')}` | {_num(row.get('count'))} |")
    paths = observation.get("weight_path_shares") or []
    if paths:
        lines.append("")
        lines.append("### Weight paths by wall share")
        lines.append("| wall share | weight path | calls per eval |")
        lines.append("|---|---|---|")
        for row in paths[:limit]:
            lines.append(f"| {_pct(row.get('wall_fraction'))} | `{row.get('path')}` | "
                         f"{_num(row.get('calls'))} |")
        lines.append("Per-call PATH aggregates above pool every node shape and phase "
                     "that hits one weight path, prefill included -- their bandwidth is "
                     "NOT a roofline reading. The ROOFLINE GAP table below is the "
                     "per-node replacement for that framing.")
    lines.extend(_render_roofline_gaps(observation.get("roofline_gaps"), limit=limit))
    phases = observation.get("host_phase_shares") or []
    if phases:
        lines.append("")
        lines.append("### Host phases, share of decode (the `ctx.*` rows already contain the "
                     "per-input-class rows; never sum the two families)")
        lines.append("| decode share | phase | family |")
        lines.append("|---|---|---|")
        for row in phases[:limit]:
            lines.append(f"| {_pct(row.get('decode_fraction'))} | `{row.get('phase')}` | "
                         f"{row.get('family')} |")
    engram = observation.get("engram_fault_mix")
    if isinstance(engram, Mapping):
        faults = engram.get("fault_counts_are_measured") is True
        lines.append("")
        lines.append(f"### Engram counters (level {engram.get('level')}, op level "
                     f"{engram.get('op_level')}, fault source {engram.get('fault_source')})")
        gather = engram.get("gather_share_of_decode")
        lines.append(f"- row-gather share of decode: "
                     f"{_pct(gather) if gather is not None else 'not measured'}")
        lines.append("- faults per decode token (minor / major): "
                     + (f"{_num(engram.get('minflt_per_decode_token'))} / "
                        f"{_num(engram.get('majflt_per_decode_token'))}" if faults else
                        "UNMEASURED at this level (a zero here would not be an absence of "
                        "faults)"))
    limitations = observation.get("limitations") or []
    if limitations:
        lines.append("")
        lines.append("Limitations: " + "; ".join(str(item) for item in limitations) + ".")
    return lines


LONG_CONTEXT_HEADER = "## Long-context regime — the target is ALSO judged at depth"


def _render_long_context(section: Mapping[str, Any], *, limit: int = 12) -> list[str]:
    """`longctx.planner_context` as the planner reads it (audit C4): the target depth, how
    keeps are judged there, the production context histogram as the workload weighting,
    and the perf/node profiles taken AT DEPTH with the slot restored."""
    lines = ["", LONG_CONTEXT_HEADER,
             f"Target depth: {section.get('target_depth_tokens')} tokens of KV (ctx "
             f"{section.get('ctx')}; prefill at depth adds a {section.get('tail_tokens')}-token "
             f"tail). {section.get('how_judged', '')}. The short-prompt profile above "
             "UNDERSTATES everything that grows with context (attention, KV traffic).",
             "", "### Production workload weighting (decode wall by context at completion)"]
    servers = ((section.get("workload_weighting") or {}).get("servers") or {})
    if not servers:
        lines.append("No production context histogram for this target "
                     f"({(section.get('workload_weighting') or {}).get('status', 'absent')}).")
    for server, body in sorted(servers.items()):
        lines.append(f"Server {server}: {body.get('requests')} requests, context p50 "
                     f"{body.get('ctx_p50')}, p90 {body.get('ctx_p90')}, max {body.get('ctx_max')}.")
        lines.append("| context bucket | requests | decode wall share | decode tok/s | "
                     "prefill wall share |")
        lines.append("|---|---|---|---|---|")
        for row in body.get("buckets") or []:
            lines.append(f"| {row.get('bucket')} | {row.get('requests')} | "
                         f"{_pct(row.get('decode_wall_share'))} | {_num(row.get('decode_tok_s'))} | "
                         f"{_pct(row.get('prefill_wall_share'))} |")
    cpu = section.get("cpu_profile_at_depth") or {}
    lines.extend(["", "### Sampled CPU profile at depth (slot restored; decode at depth)"])
    if cpu.get("status") == "observed":
        lines.append("| rank | sampled-period fraction | mechanism family |")
        lines.append("|---|---|---|")
        for rank, row in enumerate((cpu.get("ranked_levers") or [])[:limit], 1):
            lines.append(f"| {rank} | {_pct(row.get('sampled_period_fraction'))} | "
                         f"`{row.get('family')}` |")
    else:
        lines.append(f"CPU profile at depth {cpu.get('status', 'absent')}: "
                     f"{cpu.get('reason') or 'not collected'}")
    node = section.get("node_profile_at_depth") or {}
    lines.extend(["", "### Node profile at depth (instrumented sibling; prefill + decode at depth)"])
    if node.get("status") == "observed":
        lines.append("| rank | wall share | op |")
        lines.append("|---|---|---|")
        for rank, row in enumerate((node.get("ranked_op_shares") or [])[:limit], 1):
            lines.append(f"| {rank} | {_pct(row.get('wall_fraction'))} | `{row.get('op')}` |")
    else:
        lines.append(f"Node profile at depth {node.get('status', 'absent')}: "
                     f"{node.get('reason') or 'not collected'}")
    return lines


def render_context(context: Mapping[str, Any], *, limit: int = 12,
                   belief_section: str = "") -> str:
    """The bundle, as the actor sees it. Everything here was previously discarded."""
    lines: list[str] = []
    cpu = _cpu_target(context)
    if context.get("target"):
        lines.extend(["## Selected target (original launch, model, requests and build)",
                      "Repeated subtrees are printed once; later copies read `<same as $.path>`.",
                      "```json", json.dumps(_dedupe_subtrees(context["target"]), indent=2,
                                            sort_keys=True),
                      "```", ""])

    # First, because it is the cheapest rejection: standing constraints and the
    # settled list. `program.md` carried "Already in v9: GGML_IQK, MMQ, HIP graphs"
    # for the whole of run 6 while the planner proposed exactly those and the critic
    # rejected all nine iterations for it -- a document nobody was wired to read.
    program = (context.get("program") or "").strip()
    if program:
        lines.append("## Standing constraints and settled questions (read this first)")
        lines.append(program)
        lines.append("")

    # R24-8: make the live pool alarm visible to the next planner call as evidence,
    # not as a new objective or a hidden selection rule.
    loop_health = context.get("loop_health_alarm")
    if isinstance(loop_health, Mapping) and loop_health.get("operator_attention_required"):
        evidence_line = loop_health.get("planner_evidence_line")
        lines.append("## Pool health attention (observe-only)")
        lines.append(str(evidence_line or "R24-8 alarm active; details unavailable."))
        lines.append("This is diagnostic context only; it does not change grading, keeps, or stopping.")
        lines.append("")

    # DS41 audit (2026-10): `program.md`'s settled-mechanisms list above is maintained
    # BY HAND and goes stale (the exact failure the comment above just described, for
    # the same reason: a document nobody is wired to keep current). This block is
    # GENERATED, every turn, from the store itself: the accumulator's own keeps
    # (`accumulator-bundle.json`) plus every kept/keep_candidate row recall returned --
    # never edited, never hand-maintained, and never stale by more than one turn.
    accumulator_keeps = [m for m in (context.get("accumulator_keeps") or [])
                        if isinstance(m, str) and m]
    kept_rows = [row for row in (context.get("prior_experiments") or [])
                if isinstance(row, Mapping) and row.get("status") in ("kept", "keep_candidate")
                and row.get("mechanism_id")]
    already_at_head = list(dict.fromkeys(accumulator_keeps))
    for row in kept_rows:
        mechanism = str(row["mechanism_id"])
        if mechanism not in already_at_head:
            already_at_head.append(mechanism)
    if already_at_head:
        lines.append("## ALREADY IMPLEMENTED AT HEAD: do not re-propose")
        lines.append("Generated from the accumulator's own keeps and the kept/keep_candidate "
                     "rows recall returned, not a hand-maintained list.")
        for mechanism in already_at_head:
            lines.append(f"- `{mechanism}`")
        lines.append("")

    # Before anything else the planner is asked to look at candidates that were
    # FORMED AND NEVER MEASURED. A lane authors against champion C0; another lane's
    # keep advances it to C1; the first candidate is refused as superseded. That work
    # is only wasted if nobody looks at it again -- the hypothesis, its falsifier and
    # the champion it was formed against are all recorded, so it can be re-proposed
    # against the champion that displaced it. Cheaper than deriving a new one, and it
    # is a QUEUE, not a graveyard.
    # Critic-ACCEPTED hypotheses whose patches failed and which are queued to be
    # re-authored (loop.PATCH_ROUNDS_EXHAUSTED / SCOPE_BLOCKED): the next draws resume
    # them BEFORE the planner is asked, so proposing one again duplicates queued work.
    pending = [row for row in (context.get("pending_accepted_hypotheses") or [])
               if isinstance(row, Mapping)][:limit]
    if pending:
        lines.append("## Accepted hypotheses pending authoring — do NOT re-propose these")
        lines.append("Each was accepted by the critic; its patches failed on authoring or "
                     "scope, not on the idea. The loop re-authors them itself, with the "
                     "rejections as feedback. Propose something different.")
        for row in pending:
            used, budget = row.get("author_attempts_used"), row.get("author_attempts_budget")
            state = row.get("state") or "pending"
            lines.append(f"- `{row.get('mechanism_id')}` ({row.get('target')}; {state}"
                         + (f", {used}/{budget} authoring attempts spent"
                            if budget is not None else "") + ")"
                         + (f" — {row['statement']}" if row.get("statement") else "")
                         + (f"\n    blocked: {row['blocked_reason']}"
                            if row.get("blocked_reason") else ""))
        lines.append("")

    superseded = [row for row in (context.get("prior_experiments") or [])
                  if row.get("status") == "superseded"][:limit]
    if superseded:
        lines.append("## Formed but never measured — consider these FIRST")
        lines.append("Each was authored against a champion that moved before it could "
                     "be measured. None was refuted. Re-propose one if it still "
                     "addresses the current profile; say why if it does not.")
        for row in superseded:
            lines.append(f"- `{row.get('mechanism_id')}` — {row.get('statement') or ''}"
                         + (f"\n    falsifier: {row['falsifier']}"
                            if row.get("falsifier") else "")
                         + (f"\n    {row['refusal_reason']}"
                            if row.get("refusal_reason") else ""))
        lines.append("")

    hotspots = context.get("kernel_hotspots") or []
    lines.append("## CPU profile for the selected experimental target" if cpu else
                 "## Where the device time actually goes (rocprofv3, current champion)")
    if cpu and context.get("cpu_profile"):
        observation = context["cpu_profile"]
        lines.append("Sampled user-cycle attribution for the original request, not exact CPU "
                     "cost, wall-time share, a speedup estimate or an acceptance A/B.")
        if observation.get("status") == "observed":
            lines.append(f"Original record: {observation.get('record')} "
                         f"(SHA-256 {observation.get('record_sha256')})")
            lines.append(f"Execution: {observation.get('execution_digest')}; "
                         f"frozen prompts: {observation.get('prompt_manifest_digest')}")
            lines.append("| sampled-period fraction | observed periods | DSO | symbol |")
            lines.append("|---|---|---|---|")
            for row in observation.get("hotspots", [])[:limit]:
                lines.append(f"| {row['sampled_period_fraction'] * 100:.2f}% | {row['period']} | "
                             f"`{row.get('dso')}` | `{row['symbol']}` |")
            ranked = observation.get("ranked_levers") or []
            if ranked:
                lines.append("")
                lines.append("### Ranked mechanism families from that same profile")
                lines.append("This is a lossless grouping of the sampled symbols above, not a "
                             "speedup estimate. Start with the highest-share unresolved causal "
                             "mechanism; do not spend the iteration on a lower-share cosmetic "
                             "variant without explaining why.")
                lines.append("| rank | sampled-period fraction | mechanism family | evidence |")
                lines.append("|---|---|---|---|")
                for rank, row in enumerate(ranked[:limit], 1):
                    lines.append(f"| {rank} | {row['sampled_period_fraction'] * 100:.2f}% | "
                                 f"`{row['family']}` | {row['evidence_kind']} |")
            locations = observation.get("location_attribution")
            if locations:
                lines.append("")
                lines.append("### Where sampled threads executed")
                lines.append("These are user-cycle sample periods on sampled execution CPUs. They do "
                             "not measure remote-memory traffic, completed work per thread, "
                             "wall-time imbalance, or a causal NUMA penalty.")
                lines.append(f"Active TIDs: {locations['active_tid_count']} of "
                             f"{locations['sampled_tid_count']} sampled (activity cutoff "
                             f"{locations['active_period_cutoff']:.0f} periods).")
                lines.append("| execution NUMA node | sampled-period share | sync fraction within node |")
                lines.append("|---|---|---|")
                for row in locations["execution_nodes"][:limit]:
                    lines.append(f"| {row['numa_node']} | "
                                 f"{row['sampled_period_fraction'] * 100:.2f}% | "
                                 f"{row['sync_fraction_within_node'] * 100:.2f}% |")
                lines.append("Low/high synchronization-fraction active TIDs (descriptive extremes):")
                lines.append("| TID | sampled CPUs | execution nodes | sync fraction |")
                lines.append("|---|---|---|---|")
                for row in locations["low_high_sync_threads"][:limit]:
                    lines.append(f"| {row['tid']} | {row['sampled_cpus']} | "
                                 f"{row['execution_nodes']} | "
                                 f"{row['sync_fraction_within_tid'] * 100:.2f}% |")
            lines.extend(observation.get("limitations", []))
        else:
            lines.append(f"CPU profile {observation.get('status')}: "
                         f"{observation.get('reason', 'no original observation collected')}")
    elif hotspots:
        lines.append("| share | ns | calls | kernel |")
        lines.append("|---|---|---|---|")
        for row in list(hotspots)[:limit]:
            share = row.get("share_of_device_time") or row.get(
                "anchor_share_of_device_time") or 0.0
            lines.append(f"| {share * 100:.2f}% | {row.get('total_duration_ns')} | "
                         f"{row.get('calls')} | `{row.get('signature')}` |")
        lines.append("\nA mechanism aimed at a route with negligible share cannot move "
                     "the target runtime no matter how correct it is.")
    else:
        lines.append("(no profile yet — say so rather than guessing a target)")

    # run.py has put `node_profile` in the CPU context since the instrumented-sibling
    # capture landed, and its CPU directive tells the planner to "Read node_profile" --
    # but nothing here printed it, so every CPU prompt through DS41 run 9 carried the
    # instruction and not the data (run 8's retained observation was 21.4k chars of
    # JSON the planner never saw). Rendered as tables, like `cpu_profile`, in BOTH
    # context modes: the A/B arms differ only in where they read it.
    if cpu and context.get("node_profile"):
        lines.extend(_render_node_profile(context["node_profile"], limit=limit))
    if not cpu and context.get("gpu_serving_profile"):
        lines.extend(_render_gpu_serving_profile(context["gpu_serving_profile"], limit=limit))
        if context.get("two_denominator"):   # AK-PROF-TWO-DENOM: wall share + launch gap
            lines.extend(["", "## GPU share by two denominators",
                          hotspots.render_two_denominator(context["two_denominator"])])
    if not cpu and context.get("amdahl_ceilings"):   # AK-AMDAHL-PRUNE: e2e ceilings
        from . import amdahl
        lines.extend(amdahl.render(context["amdahl_ceilings"]))
    if not cpu and context.get("gpu_counters"):   # RVP-CTR-1: counter-grounded planner signals
        lines.extend(gpu_counter_signals.render(context["gpu_counters"], limit=limit))
    if cpu and context.get("long_context"):
        lines.extend(_render_long_context(context["long_context"], limit=limit))

    prior = context.get("prior_experiments") or []

    # A flat DNR list stops exact repeats, but it does not stop the planner from
    # spending a campaign on cosmetic variants of the same failed idea. Treat three
    # resolved failures in one mechanism family as a signal to change the QUESTION,
    # not merely the implementation. This is formation guidance only: it neither
    # changes a recorded result nor assigns a magnitude to a historical observation.
    #
    # DS41-C76: a keep resets the family, as it does for the family-level block below.
    # Recall is newest-first, so failures older than the family's latest keep were
    # formed against source that keep has since changed. Without the reset, the
    # "local quant/dot kernel" family (any `q8_0`/`q8-` text) stayed exhausted
    # straight through the dense Q8_0 keeps it had just produced.
    exhausted, stagnant = diminishing_returns_triggers(prior)
    if exhausted:
        lines.append("\n## DIMINISHING-RETURNS ESCAPE — mandatory for this turn")
        # DS41-C76: this block used to order that the next hypothesis "MUST target one
        # of graph scheduling, row/work partitioning, NUMA/memory placement, or
        # expert/load balance" whenever ANY family reached three failures. Once those
        # four categories were themselves tried, both DS41 lanes abstained on every
        # iteration (rows 134-143, 2026-09-29) -- including a planner that named the
        # dense MUL_MAT per-core efficiency gap, the route every kept win came from.
        # The families are keyword buckets (`_mechanism_family`), and "local quant/dot
        # kernel" alone covers most of a quantized model's hot path, so the rule bans
        # cosmetic variants of the listed mechanisms, not the whole bucket.
        lines.append(
            "Repeated nulls/refusals show that the mechanisms below are exhausted. "
            "Do NOT propose a cosmetic variant of them (same bottleneck, same code "
            "region, reworded or re-parameterised). Choose EITHER a materially "
            "different mechanism -- a different bottleneck or code region the current "
            "profile supports, which may sit in a listed family or in a route that "
            "already produced keeps -- stating why the listed failures do not predict "
            "it, OR escalate the causal question: why the hot work is waiting, "
            "imbalanced, poorly partitioned, remotely placed, or serialised (graph "
            "scheduling, row/work partitioning, NUMA/memory placement, expert/load "
            "balance). Every escalation category being closed is NOT grounds to "
            "abstain while a materially different mechanism remains.")
        for family, rows in sorted(exhausted.items()):
            mechanisms = list(dict.fromkeys(
                str(row.get("mechanism_id")) for row in rows
                if row.get("mechanism_id")))
            lines.append(f"- `{family}`: {len(rows)} resolved failures across "
                         f"{', '.join(mechanisms[:6])}")
        lines.append("")

    # A mechanism ID is actor prose; the durable source path and symbol are the
    # host-owned family identity.  Detect a run of distinct null/refused ideas in
    # that family without reading effect magnitudes (especially stale ones).  A
    # keep resets the run because it changed the source the later ideas see.
    if stagnant:
        lines.append("## Family-level diminishing returns — abstraction escape required")
        lines.append("These are attempt/outcome facts only; no stale or cross-epoch magnitude "
                     "is aggregated. Rewording the same leaf mechanism is not exploration.")
        for (surface, symbol), rows in stagnant.items():
            summary = ", ".join(
                f"{row['mechanism_id']} ({row['status']})" for row in rows[:limit])
            lines.append(f"- `{surface}::{symbol}`: {len(rows)} distinct recent ideas — {summary}")
        lines.append("Planner: move one abstraction level up to a caller/operator/dispatch or "
                     "data-movement mechanism grounded in the current profile and source route.")
        lines.append("Critic: reject a same-family synonym unless it supplies a materially "
                     "distinct causal model and new profile/source/history evidence that the "
                     "listed attempts did not test.")
        lines.append("")

    # Mechanisms already CHARACTERISED by repeated measurement. Run 15 spent 9 of its
    # 10 measurements re-sampling two unchanged patches: a near-floor result reads as
    # "almost", so the planner re-proposed it. But re-measuring unchanged code adds no
    # information -- it redraws the same noise. Pooling says what is actually known,
    # and a characterised mechanism is FINISHED unless the code changes.
    #
    # `comparable_measurement` is load bearing here and was MISSING. This pooled every
    # row carrying a number -- cross-epoch ones included -- into one median printed
    # under a heading that tells the planner not to re-measure. Against the live store,
    # through the first ~20 rows of epoch `6a4dccec`, it read "`akm-q4k-q8-sum-sidecar`:
    # measured 4x, median -8.814%" with all four magnitudes taken against a DIFFERENT
    # anchor and build: a cross-epoch magnitude deciding a mechanism was finished. That
    # is what `P-AK-SEARCH-1` denial 4 forbade and what `-A3` clause 2 still forbids --
    # A3 moved the ORDERING question only, never the comparability one. It bit at every
    # epoch transition, because a new epoch's first recall window is the old epoch's
    # tail. The marker was right there on the row and the loop read the number instead,
    # which is why `experiments.rank()` now deletes the number as well as marking it.
    #
    # BOTH markers, and a row carrying NEITHER is still pooled. `recall()` always writes
    # both -- `test_ranking.py` asserts that, which is what makes this the whole real
    # path -- so the two spellings are complements there and the only rows this default
    # reaches are hand-built ones with no provenance to judge. Requiring a positive
    # `comparable_measurement` instead was the first version and it silently switched
    # the block off for every synthetic context, `test_seed.py`'s five-sample run-15
    # regression included: a conformance fix that disables the feature it is protecting.
    from . import claims as claim_contract
    repeats: dict[str, list[float]] = {}
    for row in prior:
        effect = row.get("effect_fraction")
        if (row.get("mechanism_id") and isinstance(effect, (int, float))
                and not row.get("stale_epoch")
                and row.get("comparable_measurement", True)
                and claim_contract.mechanism_status(row) == claim_contract.VERIFIED):
            repeats.setdefault(row["mechanism_id"], []).append(effect * 100.0)
    characterised = {k: v for k, v in repeats.items() if len(v) >= 3}
    if characterised:
        lines.append("## Characterised — do NOT re-measure these")
        lines.append("Each was measured repeatedly on UNCHANGED code. Re-running one "
                     "redraws the same noise and tells you nothing new. Change the "
                     "mechanism or pick a different target.")
        for mechanism, values in characterised.items():
            values = sorted(values)
            median = values[len(values) // 2]
            positive = sum(1 for v in values if v > 0)
            lines.append(
                f"- `{mechanism}`: measured {len(values)}x, median {median:+.3f}%, "
                f"{positive}/{len(values)} positive "
                f"[{', '.join(f'{v:+.2f}' for v in values)}]")
        lines.append("")

    # DS41-C97: runtime treatments already measured on THIS original launch, keyed by
    # what they change rather than by mechanism id. Run 10w re-proposed and re-measured
    # OMP_WAIT_POLICY=passive under a second id; the loop now refuses that before
    # measurement (`runtime_identity.duplicate`), and this says so up front.
    runtime_settled = [row for row in (context.get("runtime_treatments_observed") or [])
                       if isinstance(row, Mapping) and row.get("describe")]
    if runtime_settled:
        lines.append("## Runtime treatments already measured on this recipe — do NOT re-propose")
        lines.append("Same original launch, recipe and frozen requests. A runtime treatment is "
                     "identified by the launch change it makes, not by its mechanism id or "
                     "wording: re-proposing one of these changes under a new name is refused "
                     "before measurement. Propose a different change or a source mechanism.")
        for row in runtime_settled[-limit:]:
            effect = row.get("effect")
            measured = f"{effect * 100:+.3f}%" if isinstance(effect, (int, float)) else "—"
            lines.append(f"- runtime treatment `{row['describe']}` measured {measured} "
                         f"({row.get('status')}) on this recipe as `{row.get('mechanism_id')}` "
                         f"(row {str(row.get('attempt_id'))[:12]}, {row.get('recorded_at')}) "
                         "— settled, do not re-propose")
        lines.append("")

    # AK-PM-17-S2: every worker sees the same reflexion broadcast digest (de-duplicated
    # lessons / avoid / try patterns from the latest rounds) in its next-round prompt.
    from . import reflexion as _reflexion
    digest = _reflexion.broadcast_view(context.get("prior_experiments") or [])
    if any(digest.values()):
        lines.append("\n## Lessons from recent rounds (reflexion broadcast, observe-only)")
        for key, title in (("lessons", "Lessons"), ("avoid_patterns", "Avoid"),
                           ("try_patterns", "Try")):
            for item in digest[key]:
                lines.append(f"- {title}: {item}")
    lines.append("\n## Already tried")
    if prior:
        supersessions = abstention_supersessions(list(prior))
        newest = list(prior)[:limit]
        newest_ids = {id(row) for row in newest}
        for row in newest:
            stale = " [STALE EPOCH — the fact it was tried is usable, the NUMBER is not]" \
                if row.get("stale_epoch") else ""
            effect = row.get("effect_fraction")
            measured = f"{effect * 100:+.3f}%" if isinstance(effect, (int, float)) else "—"
            mechanism_claim = claim_contract.mechanism_status(row)
            lines.append(f"- `{row.get('mechanism_id')}` → {row.get('status')} "
                         f"{measured}{stale} [mechanism claim: {mechanism_claim}]"
                         + _supersession_note(row, supersessions) + sampling_axis.scope_note(row)
                         + (f"\n    refused: {row['refusal_reason']}"
                            if row.get("refusal_reason") else ""))

        # DS41 audit (2026-10): the window above is recency-only and 12 rows wide, so
        # a kept/keep_candidate outside it -- or a run of 29+ "already implemented"
        # abstentions whose distinct mechanism families never fit -- silently fell out
        # of the prompt. These are ALWAYS shown (compact, one line each), regardless of
        # recency: every kept/keep_candidate row of the CURRENT epoch (same_epoch, set
        # by `recall()`), plus the newest row of every distinct abstained/null-mechanism
        # family not already in the verbose window above. Bounded, so a long-running
        # epoch cannot explode the prompt.
        pinned_kept = []
        seen_kept_ids: set[str] = set()
        for row in prior:
            if id(row) in newest_ids or not row.get("same_epoch"):
                continue
            if row.get("status") not in ("kept", "keep_candidate"):
                continue
            key = row.get("attempt_id") or id(row)
            if key in seen_kept_ids:
                continue
            seen_kept_ids.add(key)
            pinned_kept.append(row)

        pinned_abstained = []
        superseded_pinned: list = []
        seen_families: set[str] = set()
        for row in prior:
            if id(row) in newest_ids or not row.get("same_epoch"):
                continue
            if row.get("status") not in ("abstained", "measured_null"):
                continue
            if row.get("attempt_id") in supersessions:
                # A superseded abstention must not hold the family slot (and so hide the
                # route behind the old record); it is listed below with its link.
                superseded_pinned.append(row)
                continue
            family = (_mechanism_family(row) or str(row.get("mechanism_id") or "")
                      or "(unlabeled)")
            if family in seen_families:
                continue
            seen_families.add(family)
            pinned_abstained.append(row)

        pinned = pinned_kept + pinned_abstained + superseded_pinned
        omit_cap = max(limit * 4, 40)
        omitted = max(0, len(pinned) - omit_cap)
        if omitted:
            pinned = pinned[:omit_cap]
        if pinned:
            lines.append("")
            lines.append("### Always shown — every current-epoch keep/keep_candidate, "
                         "plus every distinct abstained/null-mechanism family")
            lines.append("Compact: status, mechanism, target, effect, short reason. The "
                         "full record for the newest rows is above.")
            for row in pinned:
                effect = row.get("effect_fraction")
                measured = f"{effect * 100:+.3f}%" if isinstance(effect, (int, float)) else "—"
                target = f"{row.get('target_surface')}::{row.get('target_symbol')}" \
                    if row.get("target_surface") or row.get("target_symbol") else "—"
                reason = (row.get("refusal_reason") or row.get("statement") or "")[:160]
                lines.append(f"- {row.get('status')} `{row.get('mechanism_id')}` "
                             f"({target}) {measured}" + (f" — {reason}" if reason else "")
                             + _supersession_note(row, supersessions) + sampling_axis.scope_note(row))
            if omitted:
                lines.append(f"- … {omitted} more omitted")
    else:
        lines.append("(nothing yet)")

    # The belief-kernel section sits directly after the in-store memory it extends
    # (DESIGN §7.5); it used to be appended to the prompt tail by `propose()`.
    if belief_section:
        lines.append("\n" + belief_section)

    shared = context.get("shared_prior_experiments")
    if shared:
        lines.append("\n## Shared historical mechanisms — transfer NOT established")
        lines.append("These are original outcomes on other recorded scopes, not applicable gains, "
                     "local refutations or reasons to skip validation. Preserve their model, quant, "
                     "recipe, surface and caveats; unknown means not captured. Use ideas as suggestions "
                     "only. These rows are excluded from the characterised-mechanism pooling above.")
        lines.extend(["```json", json.dumps(_slim_shared_history(shared), sort_keys=True, indent=2),
                      "```"])

    feedback = context.get("serving_observations")
    if feedback:
        lines.append("\n## Original serving observations — recall, not qualified gains")
        lines.append("Same model/recipe/request/epoch and original anchor only. A null or "
                     "uncalibrated result is worth remembering; belief status does not "
                     "promote it to a gain, and CPU allowed lists do not prove placement "
                     "or absence of contention. Prior experiment history above is independent.")
        lines.append("```json")
        lines.append(json.dumps(_dedupe_subtrees(feedback), sort_keys=True, indent=2))
        lines.append("```")

    for label, key in (("Your hypothesis was rejected", "prior_hypothesis_rejections"),
                       ("Your patch was rejected", "prior_patch_rejections")):
        reasons = context.get(key) or []
        if reasons:
            lines.append(f"\n## {label} — answer these, do not re-derive")
            lines.extend(f"- {reason}" for reason in reasons)

    inbox = context.get("inbox") or []
    if inbox:
        lines.append("\n## Operator suggestions (async; use if relevant)")
        lines.extend(f"- {item}" for item in inbox)
    return "\n".join(lines)


#: The actor context is the first ~46k tokens of every planner step on a 98k slot
#: (DS41 2026-09-24: 95.7k chars). Two blocks were 55% of it: the target JSON, whose
#: `recipe` and `common_cpu_scope.full_transfer_target` repeat the same launch
#: subtrees (dsos, capability, template, ...), and the shared-history rows, whose
#: bulk was refusal prose and content hashes. Neither trim drops a fact a reader
#: could act on: a duplicate points at its first copy, a hash is not evidence to a
#: planner, and clipped prose keeps its head and says it was clipped.
DEDUPE_MIN_CHARS = 200
SHARED_ROW_ID_FIELDS = frozenset({"attempt_id", "original_epoch", "result_sha256", "source_store",
                                  "hypothesis_id", "recorded_at", "campaign_id"})
SHARED_ROW_PROSE_CHARS = 500


def _dedupe_subtrees(value: Any, *, _seen: dict[str, str] | None = None, _path: str = "$") -> Any:
    """Replace every repeat of a large subtree with a pointer to its first copy."""
    seen = {} if _seen is None else _seen
    if isinstance(value, (dict, list)):
        key = json.dumps(value, sort_keys=True)
        if len(key) >= DEDUPE_MIN_CHARS:
            if key in seen:
                return f"<same as {seen[key]}>"
            seen[key] = _path
    if isinstance(value, dict):
        return {k: _dedupe_subtrees(value[k], _seen=seen, _path=f"{_path}.{k}")
                for k in sorted(value)}
    if isinstance(value, list):
        return [_dedupe_subtrees(item, _seen=seen, _path=f"{_path}[{i}]")
                for i, item in enumerate(value)]
    return value


def _clip(value: Any, limit: int = SHARED_ROW_PROSE_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f" …[clipped {len(value) - limit} chars]"
    return value


def _slim_shared_history(shared: Any) -> Any:
    """Shared-history rows without content-hash/id fields, prose clipped."""
    if not isinstance(shared, Mapping) or not isinstance(shared.get("rows"), list):
        return shared
    rows = [{k: _clip(v) for k, v in row.items() if k not in SHARED_ROW_ID_FIELDS}
            if isinstance(row, Mapping) else row for row in shared["rows"]]
    return {**shared, "rows": rows}


#: AKX-CHK-2: statuses of a route that was DECLINED or FAILED without a scientific verdict.
#: A later measured finding on the same mechanism supersedes the record (the original row is
#: kept; the planner is shown the link) instead of the old record suppressing the route.
_SUPERSEDABLE_STATUSES = frozenset({
    "abstained", "bench_failed", "lane_error", "authoring_failed", "authoring_harness_failure",
    "patch_rounds_exhausted", "scope_blocked", "refused_at_formation"})
_SUPERSEDING_STATUSES = frozenset({"kept", "keep_candidate", "regression", "measured_null"})


def abstention_supersessions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """`{attempt_id: newer row}` for abstained/failed rows a later finding supersedes.

    Explicit link first: a row carrying `superseded_by` (an attempt_id present in `rows`)
    is superseded by that row. Otherwise derived: a same-epoch row with a measured status
    (`_SUPERSEDING_STATUSES`) on the SAME non-empty `mechanism_id` and a later `recorded_at`.
    Pure and read-only: no row is mutated or dropped, so the original record stays."""
    by_id = {r.get("attempt_id"): r for r in rows if r.get("attempt_id")}
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        aid = row.get("attempt_id")
        if not aid or row.get("status") not in _SUPERSEDABLE_STATUSES:
            continue
        explicit = by_id.get(row.get("superseded_by"))
        if explicit is not None and explicit is not row:
            out[aid] = explicit
            continue
        mech = row.get("mechanism_id")
        if not mech:
            continue
        later = [r for r in rows
                 if r.get("mechanism_id") == mech and r.get("status") in _SUPERSEDING_STATUSES
                 and r.get("same_epoch", True) and row.get("same_epoch", True)
                 and str(r.get("recorded_at") or "") > str(row.get("recorded_at") or "")]
        if later:
            out[aid] = min(later, key=lambda r: str(r.get("recorded_at") or ""))
    return out


def _supersession_note(row: Mapping[str, Any], sup: Mapping[str, Mapping[str, Any]]) -> str:
    newer = sup.get(row.get("attempt_id"))
    if newer is None:
        return ""
    return (f" [SUPERSEDED by {str(newer.get('attempt_id'))[:12]}: {newer.get('status')} "
            f"{newer.get('recorded_at') or ''} — the route is open; the original record is kept]")


#: The statuses each diminishing-returns trigger in `render_context` counts as a resolved
#: failure of a family. DECISION 2026-10-24 (AK-HR-2 follow-up; review
#: /mnt/raid0/llm/tmp/ak-trigger-review-20261024/MEMO.md):
#:
#: * `abstained` is NOT counted. An abstention forms no hypothesis (`Outcome.hypothesis`
#:   is None): the planner only names what it looked at and declined, and AK-HR-2 classes
#:   it as a harness-class event, as `_SUPERSEDABLE_STATUSES` already does ("declined ...
#:   without a scientific verdict"). Search-stopping practice agrees: an attempt that
#:   yields no observation does not count toward patience. Note the abstentions are
#:   reasoned planner verdicts (about 99% of 356), not harness noise, and 95% name no
#:   target, so this is a semantic ruling, not a cleanup. Replayed over the closed history
#:   (recency recall, 40-row window): Q38FN, one family and one give-up episode move (11
#:   near-identical "already implemented" abstentions counted as one distinct failed
#:   idea; the family later produced a keep anyway), about 1.5 extra same-family attempts
#:   per campaign of that kind; DS41, no effect.
#: * `refused_at_formation` IS still counted. It is a pre-build verdict on a FORMED idea:
#:   a structural repeat of an answered row (`do_not_repeat`) or a refused premise, and
#:   AK-HR-2 uses it as the lower bound of the repeat-null rate, not as a harness event.
#:   Re-proposing an answered region under new names is exactly what both blocks exist
#:   to break. Replay: it supplied the third row of 20 of DS41's 88 exhaustion firings
#:   and of 468/2167 exhaustion and 1224/3039 stagnation firings in loop-memory.
FAMILY_EXHAUSTION_STATUSES = frozenset({
    "measured_null", "regression", "refused_at_formation", "authoring_refused",
    "screened_out", "hypothesis_retired"})
FAMILY_STAGNATION_STATUSES = frozenset({
    "measured_null", "regression", "refused_at_formation", "runtime_refused"})
FAMILY_RESET_STATUSES = frozenset({"kept", "keep_candidate"})


def _newest_first(prior: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """`prior` in recorded_at order, newest first (stable; undated rows last).

    Both triggers reset a family at its newest keep, which needs time order. Recency
    recall already is; RANKED recall (`--rank-prior-experiments`, the default since
    2026-10-20) is an order of merit that puts every `kept` row (merit 5.0) ahead of the
    newer failures of its family. Walking in that order breaks "a keep resets the
    family" in BOTH directions: failures OLDER than the keep that rank above it are
    counted (replay: 116 of 189 DS41 exhaustion family-firings, 591 of 2,026 in
    loop-memory), and a family with three or more failures NEWER than its keep is hidden
    (191 Q38FN stagnation family-firings). Walking by recorded_at fixes both."""
    dated = [row for row in prior if isinstance(row.get("recorded_at"), str)
             and row.get("recorded_at")]
    undated = [row for row in prior if not (isinstance(row.get("recorded_at"), str)
                                            and row.get("recorded_at"))]
    return sorted(dated, key=lambda row: row["recorded_at"], reverse=True) + undated


def diminishing_returns_triggers(
        prior: Sequence[Mapping[str, Any]], *,
        exhaustion_statuses: frozenset[str] = FAMILY_EXHAUSTION_STATUSES,
        stagnation_statuses: frozenset[str] = FAMILY_STAGNATION_STATUSES,
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[tuple[str, str], list[Mapping[str, Any]]]]:
    """(`exhausted`, `stagnant`): the families each `render_context` escape block names.

    `exhausted`: keyword family (`_mechanism_family`) -> its >= 3 resolved failures newer
    than the family's newest keep. `stagnant`: (target_surface, target_symbol) -> >= 3
    rows with DISTINCT mechanism ids newer than that family's newest keep. Attempt and
    outcome facts only; no magnitude is read. Pure."""
    rows = _newest_first(prior)
    exhausted: dict[str, list[Mapping[str, Any]]] = {}
    kept_families: set[str] = set()
    for row in rows:
        family = _mechanism_family(row)
        if not family or family in kept_families:
            continue
        status = row.get("status")
        if status in FAMILY_RESET_STATUSES:
            kept_families.add(family)
        elif status in exhaustion_statuses:
            exhausted.setdefault(family, []).append(row)
    exhausted = {family: found for family, found in exhausted.items() if len(found) >= 3}

    family_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    closed_families: set[tuple[str, str]] = set()
    for row in rows:
        surface, symbol = row.get("target_surface"), row.get("target_symbol")
        if not isinstance(surface, str) or not surface \
                or not isinstance(symbol, str) or not symbol:
            continue
        family = (surface, symbol)
        if family in closed_families:
            continue
        status = row.get("status")
        if status in FAMILY_RESET_STATUSES:
            # Older outcomes precede the source-changing keep and cannot establish
            # stagnation against its successor.
            closed_families.add(family)
        elif status in stagnation_statuses:
            family_rows.setdefault(family, []).append(row)
    stagnant: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for family, found in family_rows.items():
        distinct, seen = [], set()
        for row in found:
            mechanism = row.get("mechanism_id")
            if isinstance(mechanism, str) and mechanism and mechanism not in seen:
                seen.add(mechanism)
                distinct.append(row)
        if len(distinct) >= 3:
            stagnant[family] = distinct
    return exhausted, stagnant


def _mechanism_family(row: Mapping[str, Any]) -> str | None:
    """Return a coarse causal family used only to detect search stagnation."""
    text = " ".join(str(row.get(key) or "").lower() for key in (
        "mechanism_id", "statement", "target_symbol", "target_surface",
        "mechanism_family"))
    families = (
        ("synchronization/barrier", ("barrier", "spin-wait", "spin_wait",
                                     "omp wait", "openmp wait", "futex")),
        ("local quant/dot kernel", ("q4_k", "q4k", "q5_k", "q5k", "q8_0",
                                    "q8-", "vec_dot", "dot-product", "dot_product")),
        ("local fusion", ("fusion", "fuse-", "fused", "up-gate", "up_gate")),
        ("prefetch/cache", ("prefetch", "cacheline", "cache-line", "l1", "l2")),
    )
    for family, needles in families:
        if any(needle in text for needle in needles):
            return family
    return None
