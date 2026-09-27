"""NUMA CPU pinning topology + per-role pinning configuration.

Source of truth for "which CPUs and how many threads each role gets". Encodes
the validated EPYC 9655 topology + per-role NUMA wiring (Probe B 2026-05-04
canonical results, 2026-05-08 gemma4 MTP swap). `orchestrator_stack.py`
re-imports the constants and `_numa_prefix()` so the launcher and server-build
code keeps working unchanged.

2026-08-01 — TWO KINDS OF FACT, TWO HOMES. The per-role WIRING (which role runs
on which shape, on which port, under which memory policy) is CONFIGURATION and
now lives in `orchestration/stack_topology.yaml` under `numa_config:`; this
module loads it. The SHAPES that wiring names — NUMA_Q*, NUMA_FULL,
NUMA_HALF_*, GPU_HOST_LANE, _NPS4_NODES — stay here, because they are HOST FACTS
rather than configuration: they describe how this machine is physically wired,
and a wrong one is a hardware MISDESCRIPTION, not a config choice. There is
nothing an operator may legitimately decide about them, so they get no knob.
NUMA_CONFIG keeps exactly the shape it had as a Python literal, so no consumer
changed.

Key findings (2026-03-18 benchmarks, refined through 2026-05-08):
- ⚠ SUPERSEDED 2026-07-30 — "Models ≤65 GB: 4×48t NUMA-quarter instances give
  6-7× aggregate throughput". That 6-7× was measured against a 1×192t
  all-SMT-threads baseline, NOT against a correctly-placed full-machine
  instance, and it was never matched on total concurrency. Directly measured
  at matched total concurrency T (llama-bench tg128, spec-dec off, drop_caches
  before every arm, kernel production-consolidated-v8 / binary 10107), one
  full-machine instance (taskset -c 0-95 + numactl --interleave=all) beats four
  quarters at EVERY rung: T=4 79.7 vs 52.9, T=8 105.1 vs 81.0, T=16 131.0 vs
  108.4, T=32 145.9 vs 143.8 aggregate tok/s.
- Models 130-250 GB: 1×96t pinning gives 1.2-1.5×. NB the historical
  "NUMA-node" wording here is a misnomer: a node on this NPS4 host is 24
  physical cores, i.e. one quarter. (The NUMA_NODE0/NUMA_NODE1 constants that
  carried the misnomer were deleted 2026-08-11; see the note by NUMA_HALF_*.)
- Using all 192t is ANTI-OPTIMAL (46-60% penalty vs 96 physical cores)
- ⚠ CORRECTED 2026-07-30 — "taskset alone is sufficient for the S4
  no-mmap/single-owner regime; shared mmap multi-instance roles need explicit
  evidence before changing memory policy". The first clause stands. The second
  understated the problem: shared-mmap fleets do not merely lack evidence for a
  memory policy, they have NO usable memory policy at all. GGUF pages are
  placed ONCE, by whichever instance faults them in first; every later instance
  inherits that placement regardless of its own --membind/--interleave.
  Measured on a quad-quarter fleet via live numa_maps: 25.6 / 25.6 / 24.2 /
  26.9 % node-local under shared mmap vs 100 % each under --no-mmap, with fleet
  decode 40.91 -> 52.13 tok/s. Consequence: fleet placement — and therefore
  fleet throughput — depends on instance START ORDER and is nondeterministic
  across reboots.
- mlock gives 30× latency improvement under memory pressure (S2)
- Total mlock budget: ~701 GB of 1.13 TB (62%), leaving ~429 GB for KV + OS

WIRING CHANGE APPLIED 2026-07-30 (operator-ratified). This block previously read
"NO WIRING CHANGE IS AUTHORISED … this module is deliberately behaviourally
unchanged … P-BENCH-PLACEMENT-1 … is STAGED, not applied". Both clauses are now
false and are corrected rather than deleted, so the transition stays auditable:

  * P-BENCH-PLACEMENT-1 was RATIFIED 2026-07-30 (epyc-root commit 07b7dcab) into
    MEASUREMENT.md §2 and the CPU annex measurement/protocols/bench-cpu.md.
    Figures taken under it are decision-grade within its scope — subject to its
    own gates, which ratification does not waive.
  * The topology was changed: quarters retired, every quarterable role now runs
    1 full + 2 halves. See the HALF FLEET block below and the per-role entries.

Two invariants are now ASSERTED AT IMPORT (see _assert_instance_invariants at
the end of this file), because both were derivable from the cpuset all along and
a mismatch is therefore always a bug: -t must equal the cpuset's PHYSICAL core
count, and any cpuset spanning more than one NPS4 node must declare a numactl
policy. Before today 15/19 instances violated the first and 17/19 the second,
including exactly the two later measured at 2.16x and 1.85x below canonical.

Analysis: epyc-root handoffs/active/numa-placement-defect-20260730.md.
Registration gate for any new model: epyc-inference-research
docs/protocols/model-registration-runbook.md (MRG-1).
"""

from __future__ import annotations

from pathlib import Path

import yaml


# =============================================================================
# HOST FACTS — NUMA CPU Pinning, validated via benchmarks (2026-03-18)
# =============================================================================
# Everything in this section describes the MACHINE, not a decision. It is
# deliberately NOT in stack_topology.yaml: these are not knobs, and a wrong value
# here is a misdescription of hardware that would silently invalidate every
# figure taken under the shape it names.
# EPYC 9655: 192 logical cores (96 physical + SMT), 4 NUMA nodes (NPS4), ~290 GB each.
# Node 0: cores  0-23, HT  96-119      Node 2: cores 48-71, HT 144-167
# Node 1: cores 24-47, HT 120-143      Node 3: cores 72-95, HT 168-191
#
# CORRECTED 2026-07-29 (`auditor`, P2-5l). This block previously read "2 NUMA nodes
# (~566 GB each) / Node 0: cores 0-47, HT 96-143 / Node 1: cores 48-95, HT 144-191",
# which was wrong on node count, per-node memory AND cpu ranges — and it contradicted
# the NUMA_Q* quarter definitions immediately below it, which were and are correct
# (NUMA_Q0A "0-23,96-119" is exactly node 0). Re-derived from `numactl -H` on this
# host: "available: 4 nodes (0-3)", node 0 cpus 0-23 + 96-119, node size 289860 MB.
# Nothing behavioural changed; the code was already right and only its own header
# misdescribed the machine. A misdocumented topology invariant is how the next
# placement defect gets built, which is why this was worth correcting rather than
# leaving as harmless prose.

# NUMA quarter definitions: (cpu_list, thread_count)
# ⚠ The digit in these names is the old NPS2 HALF, not the node — `NUMA_Q1A` is NOT
# node 1. Read the cpu list, never the name. Full statement, and why they are
# deliberately NOT renamed, in the NPS2-ERA ARTEFACT block below.
NUMA_Q0A = ("0-23,96-119", 48)
NUMA_Q0B = ("24-47,120-143", 48)
NUMA_Q1A = ("48-71,144-167", 48)
NUMA_Q1B = ("72-95,168-191", 48)

# ── NPS2-ERA NAMING ARTEFACT — about the QUARTER constants above. ────────────
# (Until 2026-08-11 this block also covered NUMA_NODE0/NUMA_NODE1, which were defined
# below it and are now deleted — see the deletion note at the end of the block. That
# split subject is why the artefact was once mis-filed against the quarters.)
# Live topology 2026-07-30 (`numactl --hardware`, NPS4):
#     node0 = 0-23,96-119   node1 = 24-47,120-143
#     node2 = 48-71,144-167 node3 = 72-95,168-191   (distances: local 10, remote 12)
# The now-deleted NUMA_NODE0 spanned node0+node1 and NUMA_NODE1 spanned node2+node3;
# those names were correct under NPS2, before the 2026-04-24 NPS4 reboot.
# The quarter constants above ARE exactly the four NPS4 nodes, but their NAMES carry the
# same artefact: the digit is the old HALF, not the node. Q0A=node0 and Q0B=node1 (both
# inside old half 0); Q1A=node2 and Q1B=node3 — so `NUMA_Q1A` is NOT node 1. Read the cpu
# list, never the digit.
#
# NOT renamed, deliberately — but note the rename is CHEAP, not blocked. Since the
# 2026-08-01 W1 cutover no role in stack_topology.yaml declares a quarter (`cpu_shape` is
# only ever NUMA_FULL / NUMA_HALF_A / NUMA_HALF_B / GPU_HOST_LANE), so NUMA_Q* survives
# only in _CPU_SHAPES / _SHAPE_CLASSES here plus test prose. The reason to leave it is that
# a rename fixes a name nothing live reads; it is not that a rename is expensive.
# (2026-08-11 `mainC`, closing the P2-5l naming residual raised by `auditor`. TWO prior
# statements at this spot were themselves wrong and are corrected rather than deleted:
# (i) it once claimed the quarters "are correctly named" — the same defect one level down;
# (ii) my own 2026-08-11 replacement justified the non-rename by claiming `cpu_shape` NAMES
# are a persisted join key in stack_priors entries and stack manifests. FALSE, caught by
# `mainA` and re-verified here: those persist `numa_instance` (an int) and
# `cpu_shape_class` (quarter|half|full) — never the shape name. The conclusion survives on
# the ground above; the evidence for it did not.)
#
# ⚠ NUMA_NODE0 / NUMA_NODE1 were DELETED 2026-08-11 (`mainB`, file owner), closing the
# sharper half of the P2-5l naming residual. `mainA` established the case and left the
# call to the session holding the file; the annotate-don't-rename reasoning above does
# NOT extend to these two, because they were not merely misnamed:
#   * declarable from the YAML via _CPU_SHAPES, and named by ZERO instances;
#   * they paired a 48-PHYSICAL-core cpuset with 96 threads — precisely what
#     _assert_instance_invariants fatals on (`-t` must equal physical core count).
# So the only reachable effect of a role ever declaring one was an import-time
# AssertionError: the shape table offered them and they could not be used. A name that
# cannot be used correctly is deleted, not annotated — an annotation still leaves the
# trap in the table for the next author who greps for a "node" shape.
# Safe: nothing imported them (all surviving mentions were prose), and
# tests/unit/test_stack_numa.py had already declared them "deletion candidates once the
# remaining test fixtures stop naming them" — which was true, those fixtures name them
# only in comments. Their cpusets live on VERBATIM as NUMA_HALF_A / NUMA_HALF_B below,
# with the CORRECT thread count, so no shape was lost.
#
# The measurement history that hung on them is preserved and re-anchored to the half
# fleet (same cpusets) in the HALF FLEET block below — deleting a constant must not
# delete the evidence taken under its shape.
# Full-machine physical-cores-only (no SMT) — for canonical-recipe wiring
# (single-instance latency-optimal). 96 physical cores spanning all 4 NPS4 nodes.
# Pair with numactl_policy="interleave=all" so memory distributes across all 4 nodes
# (matches the canonical bench recipe used by Probe B 2026-05-04).
NUMA_FULL = ("0-95", 96)

# ── GPU HOST LANE (operator-ratified 2026-08-01) ─────────────────────────────
# The 8 host threads a VRAM-resident role needs for tokenising, sampling and
# request marshalling. NOT a decode instance — see _assert_instance_invariants.
#
# 184-191 are the SMT siblings of physical 88-95, i.e. NPS4 node 3. Choosing
# siblings is deliberate: the physical cores stay available to the CPU fleet,
# and the lane still lands node-local to node 3 so `membind=3` is meaningful.
#
# CANONICAL BECAUSE IT IS WHAT WAS MEASURED. Every GPU figure this stack relies
# on was taken under exactly this placement:
#   * the Qwen3.6-27B SWE-bench arm A3 (23/40, the frozen 4-arm authority)
#   * the Qwen3-VL-30B MMMU-250 cutover (159/250, +11.2 pp)
# Both ran `taskset -c 184-191 ... -t 8`. Deploying a different host shape than
# the one the evidence was gathered under would silently invalidate the transfer.
#
# SHARED, not per-role: GPU roles co-tenant this lane rather than each reserving
# an NPS4 quarter. Two roles needing ~8 host threads each do not justify fencing
# 48 physical cores away from the CPU fleet, and the four-model GPU steady state
# was itself measured with the tenants sharing host threads.
GPU_HOST_LANE = ("184-191", 8)

# ── 2026-07-30 HALF FLEET (operator-ratified) ────────────────────────────────
# Each half holds 48 PHYSICAL cores (0-47 or 48-95) plus their 48 SMT siblings,
# so -t is 48, NOT 96. Declaring 96 on these cpusets re-introduces 2x SMT
# oversubscription — measured cost -13% per-stream and -8.5% aggregate at np=4.
# (Until 2026-08-11 these same two cpusets were ALSO exported as NUMA_NODE0 /
# NUMA_NODE1 carrying exactly that wrong 96, which is why the warning existed;
# those constants are now deleted — see the deletion note above.)
#
# ── Measurement history for these cpusets, re-anchored here 2026-08-11 when the
#    NUMA_NODE* constants it hung on were deleted. The evidence is about the
#    SHAPE, which survives; only the name it was written under went away. ──
# CONSEQUENCE, measured: with no numactl policy, weights first-touch onto whichever
# node loads them, so ~half a 96-thread team reads every weight cross-node. The E5
# affinity artifact for this shape recorded pages_by_node {N0: 9226101} / total
# 9226101 — i.e. all 35.2 GiB on node0 while the threads spanned node0+node1.
#
# ⚠ The 2026-04-17 head-to-head once quoted to justify the old 0-47 full wiring is
# NOT valid evidence for it: that run predates the NPS4 reboot (so the cpuset
# genuinely WAS one node then), and its source CSV records spec == "baseline", i.e.
# SPECULATIVE DECODING OFF. Do not cite 26.60/27.06 t/s as a current figure here.
#
# The correct cpuset was under measurement (E5 re-run, 2026-07-30) when that note
# was written; it is carried forward verbatim rather than re-judged, since
# certifying it closed would need the E5 result and that is not this session's to run.
#
#   HALF_A = NPS4 nodes 0+1. GPU-DISJOINT. The GPU shadow lane pins host threads
#            to logical 184-191, which fold to physical 88-95 = region q3;
#            fold_smt_to_physical overlap with "0-47,96-143" is EMPTY. Measured
#            under a bandwidth-generating lane proxy, a 0-95 full instance lost
#            34% while this half lost nothing.
#   HALF_B = NPS4 nodes 2+3. NOT GPU-disjoint — it literally contains 72-95 and
#            168-191, so it IS a physical co-tenant of the lane.
#
# Pair each with its matching interleave list (interleave=0,1 / interleave=2,3).
# --interleave=all on a half would place weights on nodes its threads cannot
# reach locally, which is the same defect class this fleet exists to fix.
NUMA_HALF_A = ("0-47,96-143", 48)
NUMA_HALF_B = ("48-95,144-191", 48)

# ── Import-time invariants (2026-07-30) ──────────────────────────────────────
# Two of the three per-instance fields are DERIVABLE from the cpuset, so a
# mismatch is always a bug rather than a choice. Asserting here — a leaf module,
# so no import cycle — makes the 2026-07-30 defect class structurally
# unreachable rather than merely fixed.
#
# Before today 15/19 instances violated the thread rule and 17/19 the policy
# rule, including exactly the two measured at 2.16x and 1.85x below canonical.
# All 13 current instances satisfy both.
_NPS4_NODES = {0: "0-23,96-119", 1: "24-47,120-143",
               2: "48-71,144-167", 3: "72-95,168-191"}


def _parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _nodes_touched(spec: str) -> list[int]:
    cpus = _parse_cpus(spec)
    return sorted(n for n, s in _NPS4_NODES.items() if cpus & _parse_cpus(s))


def _assert_instance_invariants() -> None:
    """Fail loudly at import if any instance contradicts its own cpuset."""
    problems: list[str] = []
    for role, cfg in NUMA_CONFIG.items():
        per = cfg.get("numactl_policy_instances") or {}
        one = cfg.get("numactl_policy")
        gpu_lane = bool(cfg.get("gpu_host_lane"))
        for idx, (cpus, port, threads) in enumerate(cfg.get("instances", [])):
            phys = len([c for c in _parse_cpus(cpus) if c < 96])
            if gpu_lane:
                # A GPU HOST LANE IS NOT A DECODE INSTANCE, so the physical-core
                # rule below does not apply to it.
                #
                # The rule exists because for a CPU decode instance, threads beyond
                # the cpuset's physical core count mean SMT oversubscription
                # (measured -13% per-stream, -8.5% aggregate at np=4). A GPU role's
                # weights are VRAM-resident under -ngl; its host threads only
                # tokenise, sample and marshal requests. Pinning those to SMT
                # SIBLINGS is the POINT — it leaves the physical cores free for the
                # CPU fleet — so the canonical lane "184-191" has ZERO physical
                # cores by that formula and would be flagged as an infinite
                # oversubscription. It is the opposite: the most considerate
                # placement available.
                #
                # This is the same missing `device` dimension that shows up in the
                # contention model (regions derived from cpusets with no notion of
                # where the weights live). Recorded rather than silently special-
                # cased, because the real fix is to derive instance SHAPE from the
                # role's declared device instead of restating it here.
                if threads > len(_parse_cpus(cpus)):
                    problems.append(
                        f"{role}[{idx}] :{port} -t {threads} exceeds the {len(_parse_cpus(cpus))} "
                        f"logical cores in GPU host lane {cpus!r}"
                    )
            elif threads != phys:
                problems.append(
                    f"{role}[{idx}] :{port} -t {threads} but cpuset {cpus!r} holds "
                    f"{phys} PHYSICAL cores — SMT oversubscription"
                )
            nodes = _nodes_touched(cpus)
            if len(nodes) > 1 and not (per.get(idx) or one):
                problems.append(
                    f"{role}[{idx}] :{port} cpuset {cpus!r} spans NPS4 nodes {nodes} "
                    f"with NO numactl policy — weights land wherever they first touch"
                )
    if problems:
        raise AssertionError(
            "stack_numa NUMA_CONFIG invariant violation:\n  " + "\n  ".join(problems)
        )


# =============================================================================
# Per-role NUMA wiring — DECLARED DATA
# =============================================================================
# Loaded from orchestration/stack_topology.yaml `numa_config:`. Until 2026-08-01
# this was a Python literal here, which is why a ring of local fallback tables
# grew up around it: a Python table invites a second Python table beside it.
#
# NUMA_CONFIG keeps the identical runtime shape it always had —
#   {role: {"instances": [(cpu_list, port, threads), ...], ...}}
# — so orchestrator_stack, stack_manifest, the placement state machine, the
# contention model and src/runtime/instance_topology are all untouched.
#
# Roles with multiple instances get round-robin routing (requires orchestrator
# support). Every load-bearing comment about WHY a role is wired the way it is
# moved with the data; read stack_topology.yaml for the measurement history.

_TOPOLOGY_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "stack_topology.yaml"

# The host shapes a declared instance may name. Adding one is a change to the
# description of this machine and belongs HERE, not in the YAML.
_CPU_SHAPES: dict[str, tuple[str, int]] = {
    "NUMA_Q0A": NUMA_Q0A,
    "NUMA_Q0B": NUMA_Q0B,
    "NUMA_Q1A": NUMA_Q1A,
    "NUMA_Q1B": NUMA_Q1B,
    "NUMA_FULL": NUMA_FULL,
    "NUMA_HALF_A": NUMA_HALF_A,
    "NUMA_HALF_B": NUMA_HALF_B,
    "GPU_HOST_LANE": GPU_HOST_LANE,
}

# ── SHAPE CLASSES (2026-08-02) ───────────────────────────────────────────────
# The equivalence classes over the host shapes above. A CLASS is what a MODEL
# can meaningfully have an opinion about ("how many concurrent slots do I want
# on a half-size instance"); a SHAPE is the machine fact underneath it ("which
# 48 physical cores, which two NPS4 nodes").
#
# This table is the whole reason the master registry can declare
# `slots_by_shape: {full: 16, half: 4}` without learning anything about cpusets.
# It lives HERE, with the shapes, because it is derived from them and from
# nothing else — mapping NUMA_HALF_B to "half" is not a decision anyone may
# make differently, it is a restatement of the shape's size. There is
# deliberately NO role -> class table anywhere: the class of an instance is
# looked up from the `cpu_shape` its stack_topology.yaml entry already declares.
#
_SHAPE_CLASSES: dict[str, str] = {
    "NUMA_Q0A": "quarter",
    "NUMA_Q0B": "quarter",
    "NUMA_Q1A": "quarter",
    "NUMA_Q1B": "quarter",
    "NUMA_FULL": "full",
    "NUMA_HALF_A": "half",
    "NUMA_HALF_B": "half",
    "GPU_HOST_LANE": "gpu_host_lane",
}

# The classes a declaration may name. Exported so the registry-side guard can
# reject a typo (`halve: 4`) instead of silently falling back to the flat value.
CPU_SHAPE_CLASSES: frozenset[str] = frozenset(_SHAPE_CLASSES.values())

_INSTANCE_FIELDS = frozenset({"cpu_shape", "port"})
_ROLE_FIELDS = frozenset(
    {
        "instances",
        "full_instance_idx",
        "placement_policy",
        "numactl_policy",
        "numactl_policy_instances",
        "mlock",
        "gpu_host_lane",
        "spec_overrides",
        # INF-70/C7, opt-in, default absent (= 0 = off). GiB of free memory to
        # force onto EVERY NUMA node before this role's server is launched.
        # `numactl --interleave=all` is a per-allocation HINT the kernel
        # abandons for any node with no free pages, so a long-lived box can
        # skew a large model onto one node with no warning (measured
        # 2026-09-02: 57.7/10.7/8.0/17.7 GB of a 98 GB model, decode -25%).
        # Consumed by scripts/server/stack_numa_evict.py.
        "numa_pre_evict_gib",
    }
)


def _load_numa_config(path: Path | None = None) -> tuple[dict[str, dict], dict[str, tuple[str, ...]]]:
    """Load the declared per-role NUMA wiring from stack_topology.yaml.

    Returns ``(NUMA_CONFIG, shape_names_by_role)``. The second element preserves
    the per-instance ``cpu_shape`` NAME, which the tuple form of ``instances``
    throws away — ``(cpu_list, port, threads)`` is unpacked positionally by the
    launcher, the placement state machine, the contention model and
    instance_topology, so it must stay exactly three wide. The shape name is
    carried alongside instead of being wedged into the tuple.

    No fallback table, no defaults, no `except: return {}`. A launcher that
    cannot read its own wiring must fail at IMPORT — a fail-open default here is
    indistinguishable from a correct load right up until it puts a 122B on the
    wrong cores, and the resulting run still produces plausible tokens.

    Unknown keys are rejected rather than ignored: a misspelled `numactl_polic`
    that silently vanishes is exactly how an instance ends up with no memory
    policy, which is the 2026-07-30 defect class this module now asserts against.
    """
    topology_path = _TOPOLOGY_PATH if path is None else path
    try:
        document = yaml.safe_load(topology_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"stack_numa: declared NUMA wiring missing at {topology_path}. "
            f"The launcher has no fallback wiring by design."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"stack_numa: {topology_path} did not parse to a mapping")
    declared = document.get("numa_config")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"stack_numa: {topology_path} declares no non-empty 'numa_config'")

    config: dict[str, dict] = {}
    shape_names: dict[str, tuple[str, ...]] = {}
    for role, raw in declared.items():
        if not isinstance(raw, dict):
            raise ValueError(f"stack_numa: numa_config['{role}'] must be a mapping")
        unknown = set(raw) - _ROLE_FIELDS
        if unknown:
            raise ValueError(
                f"stack_numa: numa_config['{role}'] has unknown field(s) {sorted(unknown)}; "
                f"known fields are {sorted(_ROLE_FIELDS)}"
            )
        raw_instances = raw.get("instances")
        if not isinstance(raw_instances, list) or not raw_instances:
            raise ValueError(f"stack_numa: numa_config['{role}'] declares no instances")

        instances: list[tuple[str, int, int]] = []
        role_shapes: list[str] = []
        for idx, entry in enumerate(raw_instances):
            if not isinstance(entry, dict):
                raise ValueError(f"stack_numa: numa_config['{role}'].instances[{idx}] must be a mapping")
            unknown = set(entry) - _INSTANCE_FIELDS
            if unknown:
                raise ValueError(
                    f"stack_numa: numa_config['{role}'].instances[{idx}] has unknown "
                    f"field(s) {sorted(unknown)}; known fields are {sorted(_INSTANCE_FIELDS)}"
                )
            shape_name = entry.get("cpu_shape")
            if shape_name not in _CPU_SHAPES:
                raise ValueError(
                    f"stack_numa: numa_config['{role}'].instances[{idx}] names unknown "
                    f"cpu_shape {shape_name!r}; known shapes are {sorted(_CPU_SHAPES)}"
                )
            port = entry.get("port")
            if not isinstance(port, int) or isinstance(port, bool):
                raise ValueError(
                    f"stack_numa: numa_config['{role}'].instances[{idx}] port must be an int"
                )
            cpu_list, threads = _CPU_SHAPES[shape_name]
            instances.append((cpu_list, port, threads))
            role_shapes.append(str(shape_name))

        cfg: dict = {"instances": instances}
        for field in raw:
            if field != "instances":
                cfg[field] = raw[field]
        config[role] = cfg
        shape_names[role] = tuple(role_shapes)
    return config, shape_names


NUMA_CONFIG, NUMA_INSTANCE_SHAPES = _load_numa_config()

# role -> per-instance SHAPE CLASS, index-aligned with NUMA_CONFIG[role]["instances"].
# This is the launcher-side half of the per-instance `-np` join: the master
# registry declares slots per CLASS, this says which class each instance is, and
# src/registry/stack_priors.py multiplies the two together.
NUMA_INSTANCE_SHAPE_CLASSES: dict[str, tuple[str, ...]] = {
    role: tuple(_SHAPE_CLASSES[name] for name in names)
    for role, names in NUMA_INSTANCE_SHAPES.items()
}


def instance_shape_class(role: str, instance_idx: int = 0) -> str | None:
    """Shape class of one declared instance, or None when the role/index is unknown.

    None is NOT a default — callers must treat it as "no per-shape declaration
    applies" and fall back to the role's flat slot count, never to a guess.
    """
    classes = NUMA_INSTANCE_SHAPE_CLASSES.get(role)
    if not classes or not (0 <= instance_idx < len(classes)):
        return None
    return classes[instance_idx]

# Roles that should use --mlock (requires ulimit -l unlimited in launch env)
MLOCK_ROLES = {role for role, cfg in NUMA_CONFIG.items() if cfg.get("mlock")}


def _numa_prefix(role: str, instance_idx: int = 0) -> list[str]:
    """Return CPU-pinning + memory-policy prefix for a role instance.

    Default: taskset -c <cpu_list>. The S4 benchmark found no benefit from
    adding numactl --membind for no-mmap/single-owner runs where first-touch owns
    the private copy. That finding does not generalize to shared-mmap quarter
    fleets.

    ⚠ CORRECTED 2026-07-30 (comment only — no behaviour change here). This
    docstring used to continue: "those roles rely on interleaved shared pages
    unless a role-specific A/B plus live numa_maps proof justifies changing
    memory policy". That premise is DISPROVED. Shared-mmap quarter fleets do
    NOT reliably get interleaved pages — they get whatever the FIRST instance
    to fault a page in chose. The GGUF is placed once; later instances inherit
    that placement regardless of their own --membind/--interleave, so at most
    one instance can be node-local and fleet placement depends on instance
    START ORDER.

    The evidence this docstring demanded now exists, and it is exactly a
    role-specific A/B plus live numa_maps: a quad-quarter fleet measured
    25.6 / 25.6 / 24.2 / 26.9 % node-local pages under shared mmap versus
    100 % each under --no-mmap, with fleet decode 40.91 -> 52.13 tok/s (at a
    cost of ~+141 GB RAM, since --no-mmap gives each instance a private copy).
    Instrument: llama-bench tg128, spec-dec off, drop_caches before every arm,
    kernel production-consolidated-v8 / binary 10107.

    NOT AUTHORISED AS A CHANGE. This is OBSERVATION-GRADE — P-BENCH-PLACEMENT-1
    has a MEASUREMENT.md registry entry that is STAGED, not applied, so it may
    not gate a keep / revert / deploy / promote decision. Any memory-policy or
    mmap change to a serving role is operator-gated and must be executed by the
    session that owns the inference. See the module docstring.

    If the role's NUMA_CONFIG entry has a "numactl_policy" key (e.g. "interleave=all"),
    wraps the launch with `numactl --<policy> --` ahead of taskset. Roles can
    instead provide "numactl_policy_instances" for instance-specific policy,
    such as worker_general idx0 requiring interleave while its --no-mmap
    quarter instances should rely on taskset + first-touch locality.

    The canonical full-machine example is architect_CRITIC: numactl
    --interleave=all + taskset -c 0-95, the placement Probe B measured on
    2026-05-04 (12.19 t/s single-instance vs 4.3 t/s under legacy 2x cross-NUMA +
    first-touch). architect_GENERAL held that placement when this docstring was
    written and no longer does: the 2026-07-31 cutover moved the 122B to
    architect_critic and re-placed architect_general as a GPU role whose HOST
    threads sit on the 8-thread GPU host lane (taskset -c 184-191,
    numactl_policy membind=3 — the SMT siblings of 88-95, per
    stack_topology.yaml's GPU_HOST_LANE shape class). Citing it as the
    interleave=all exemplar sent readers to a membind role.
    2026-09-27 ARCHITECT SWAP (operator-decided): the labels swapped back over
    unchanged processes — the full-machine interleave=all instance (:8074,
    Flash-Next) is architect_GENERAL again, and architect_CRITIC is now the GPU
    host-lane membind=3 role (:8083, 27B). Cite the :8074 process, not a role name.
    """
    cfg = NUMA_CONFIG.get(role)
    if cfg and instance_idx < len(cfg["instances"]):
        cpu_list = cfg["instances"][instance_idx][0]
        prefix: list[str] = []
        policy = None
        instance_policies = cfg.get("numactl_policy_instances")
        if isinstance(instance_policies, dict):
            policy = instance_policies.get(instance_idx)
            if policy is None:
                policy = instance_policies.get(str(instance_idx))
        if policy is None:
            policy = cfg.get("numactl_policy")
        if policy:
            prefix.extend(["numactl", f"--{policy}", "--"])
        prefix.extend(["taskset", "-c", cpu_list])
        return prefix
    # Fallback: no pinning (embedders, fast workers, dev mode)
    return []


_assert_instance_invariants()
