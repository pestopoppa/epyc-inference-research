#!/usr/bin/env python3
"""test_packager.py — the AK6 release packager (design §11.2, §11.3, §11.5, AK7).

Run standalone (no pytest needed):

    python3 -m unittest scripts.kernel_rnd.autokernel.release.test_packager -v
    python3 -W error::ResourceWarning -m unittest \
        scripts.kernel_rnd.autokernel.release.test_packager

WHAT THIS SUITE IS FOR
----------------------
Coverage is the least of it. Four things:

  * **The cardinal rule, as a test rather than a docstring.** `TestTheBoundary`
    asserts the four self-audits PASS, that every §11.2 "may not" raises, that the
    package's own record carries no authority-flavoured key, and — the part that
    matters — that each audit BITES: doctored source with `import os`, a
    `datetime.now()`, a direct `run_t3()` call, or a softened refusal door all FAIL.
    An audit nobody has seen fail is a parser with a docstring attached.
  * **The negative space.** A seal with a drifted evaluator, a reused version name,
    an archive holding a rebuild, an unvalidated command, a watch window whose
    bands moved after the data arrived, a package whose state was stamped — each has
    a test asserting the REFUSAL, because every one is a way a release could look
    ready while being wrong.
  * **AK7's self-trigger denial.** The loop cannot mint a freeze request, cannot
    read a clock, and cannot supply operator authority. Three tests, one per route.
  * **End to end with zero production writes.** `TestEndToEnd` drives a real
    `t3.T3Request` through the real `t3.T3Runner` into a real package, asserts it
    satisfies `schemas.validate_release_package`, and asserts the whole thing while
    touching no file, no process and no tree.

Everything is hand-built fixtures: no inference, no benchmark, no build, no read of
any production tree.
"""
from __future__ import annotations

import dataclasses
import unittest

from .. import schemas, storage
from . import preflight as guards
from ..evaluator import integrity
from . import packager, t3

# =============================================================================
# Fixtures — the same shapes `test_t3.py` uses, so a package is assembled from a
# request that suite already proves is well-formed.
# =============================================================================

NOW = "2026-08-03T12:00:00Z"
CAMPAIGN_START = "2026-08-01T00:00:00Z"
LLAMA_BACKENDS = ("llama_cpu", "llama_gpu")
#: Real-shaped commits. NOT `"a" * 40`: `schemas.is_placeholder_digest` correctly
#: reads a single repeated hex character as a fabricated identity, and the packager
#: refuses one — a fabricated anchor is silent and wrong where an absent one is loud.
V8_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
BASE_COMMIT = V8_HEAD
CANDIDATE_COMMIT = "9f2c81ab53d7e604c1b8ae37f05d9142cc6b0e78"
BUILD_ROOT = "/mnt/raid0/llm/llama.cpp-experimental/build"
ARCHIVE_ROOT = "/mnt/raid0/llm/kernels/archive/v8"
INSTALL_PATH = "/mnt/raid0/llm/kernels/production"
ERA_REGISTRY = "orchestration/instrument_eras.yaml"
AUTOPILOT_BASELINE = "orchestration/autopilot_baseline.yaml"
CPU_LINK = "/mnt/raid0/llm/kernels/production/cpu"
GPU_LINK = "/mnt/raid0/llm/kernels/production/gpu"
AFFECTED_ROLES = ("worker_general", "frontdoor")


def digest(label: str) -> str:
    """A well-formed, non-placeholder digest derived from a label."""
    return schemas.content_hash({"test_fixture": label})


def ratified_protocol(protocol_id: str, **overrides) -> t3.ProtocolBinding:
    """A per-phase binding for a protocol that IS ratified (Annex B)."""
    fields = {"protocol_id": protocol_id,
              "document_sha256": digest(f"protocol:{protocol_id}"),
              "ratified": True, "ratified_at": "2026-05-01T00:00:00Z", "annex": "B"}
    fields.update(overrides)
    return t3.ProtocolBinding(**fields)


V8_CPU_BINARY = digest("v8-cpu-binary")
V8_GPU_BINARY = digest("v8-gpu-binary")
INCUMBENT_BINARIES = {"llama_cpu": V8_CPU_BINARY, "llama_gpu": V8_GPU_BINARY}
EVALUATOR_BUNDLE = digest("evaluator-bundle")


# -- T3 request pieces --------------------------------------------------------

def linkage_receipt(backend: str, **overrides) -> t3.LinkageReceipt:
    fields = {
        "backend": backend,
        "binary_path": f"{BUILD_ROOT}/bin/llama-server",
        "expected_tree_root": BUILD_ROOT,
        "verifier_path":
            f"/mnt/raid0/llm/epyc-inference-research/{t3.LINKAGE_VERIFIER_RELPATH}",
        "verifier_sha256": digest("verify_ggml_linkage.sh"),
        "exit_code": 0,
        "stdout": (f"binary : {BUILD_ROOT}/bin/llama-server\n"
                   f"  OK   libggml-base.so.0            -> {BUILD_ROOT}/bin/libggml-base.so.0\n"
                   f"PASS: all linked ggml libraries resolve inside {BUILD_ROOT}\n"),
        "ld_library_path": (f"{BUILD_ROOT}/bin", "/opt/rocm/lib"),
        "observed_at": NOW,
    }
    fields.update(overrides)
    return t3.LinkageReceipt(**fields)


def matrix_cells() -> list:
    cells: list = []
    for backend in LLAMA_BACKENDS:
        for workload_phase, protocol_id in (("prefill", "P-BENCH-PREFILL-1"),
                                            ("decode", "P-BENCH-1")):
            cells.append(t3.Cell(
                cell_id=f"{backend}.{workload_phase}",
                backend=backend, release_phase=t3.PHASE_PERFORMANCE_MATRIX,
                protocol_id=protocol_id,
                recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL,
                metric="tokens_per_s", metric_direction="higher_better",
                workload_phase=workload_phase,
                claim=f"{backend} {workload_phase} non-regression vs v8",
                roles_protected=("worker_general",),
                co_resident=(backend == "llama_cpu"),
                reps=10))
        for phase_id in (t3.PHASE_BACKEND_CORRECTNESS, t3.PHASE_QUALITY,
                         t3.PHASE_STABILITY, t3.PHASE_CAPACITY_UTILITY):
            cells.append(t3.Cell(
                cell_id=f"{backend}.{phase_id}", backend=backend,
                release_phase=phase_id, protocol_id="P-KERNEL-FREEZE-1",
                recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL, metric="pass_fail",
                metric_direction="higher_better",
                claim=f"{backend} {phase_id} parity", reps=1))
    return cells


def cell_results(cells) -> list:
    return [t3.CellResult(cell=cell, check=schemas.Check(schemas.PASS),
                          raw_samples_ref=f"data/ak/{cell.cell_id}.jsonl",
                          reducer_id="median_mad/v1")
            for cell in cells]


def sealed_candidate(**overrides) -> t3.SealedCandidate:
    fields = {
        "candidate_id": "akc-v9",
        "source_tree": "llama.cpp",
        "candidate_branch": "llama.cpp-experimental/v9",
        "production_base_commit": BASE_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "seal_sha256": digest("seal"),
        "evaluator_bundle_sha256": EVALUATOR_BUNDLE,
        "scope_manifest_sha256": digest("scope"),
        "evidence_tree_sha256": digest("evidence"),
        "binary_sha256": {b: digest(f"bin:{b}") for b in LLAMA_BACKENDS},
        "linkage_sha256": {b: digest(f"link:{b}") for b in LLAMA_BACKENDS},
        "build_dirs": {b: BUILD_ROOT for b in LLAMA_BACKENDS},
        "overlay_present": True,
        "tree_clean": True,
        "ancestry_clean": True,
    }
    fields.update(overrides)
    return t3.SealedCandidate(**fields)


def incumbent_archive(**overrides) -> t3.IncumbentArchive:
    entry_fields = {
        "generation": t3.ARCHIVE_GENERATION_N1,
        "branch": "production-consolidated-v8",
        "commit": V8_HEAD,
        "archive_root": ARCHIVE_ROOT,
        "binaries": ((f"{ARCHIVE_ROOT}/cpu/llama-server", V8_CPU_BINARY),
                     (f"{ARCHIVE_ROOT}/gpu/llama-server", V8_GPU_BINARY)),
        # One ggml runtime, both llama backends of the tree — the attribution is a
        # SET, and it is recorded here at the archive, never minted in the packager.
        "libraries": ((LLAMA_BACKENDS, f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0",
                       digest("v8-libggml-base")),),
        "rebuilt": False,
    }
    entry_fields.update(overrides)
    return t3.IncumbentArchive(entries=(t3.ArchivedBuild(**entry_fields),))


def transaction_plan(**overrides) -> t3.TransactionPlan:
    fields = {
        "next_branch": "production-consolidated-v9",
        "next_version_number": 9,
        "next_tag": "production-consolidated-v9",
        "install_path": INSTALL_PATH,
        "symlink_diff": (
            (CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin",
             "/mnt/raid0/llm/llama.cpp-v9/build/bin"),
            (GPU_LINK, "/mnt/raid0/llm/llama.cpp/build-hip/bin",
             "/mnt/raid0/llm/llama.cpp-v9/build-hip/bin"),
        ),
        "service_impact": ("llama-server restart at the inference owner's boundary",),
        "era_actions": ({"draft": True, "action": "write_era_registry_row",
                         "registry_path": ERA_REGISTRY},),
        "receipt_paths": ("artifacts/operator/v9-freeze/",),
        "rollback_branch": "production-consolidated-v8",
        "rollback_head": V8_HEAD,
    }
    fields.update(overrides)
    return t3.TransactionPlan(**fields)


def t3_request(**overrides) -> t3.T3Request:
    """A complete request that PASSes. Every test perturbs exactly one thing."""
    cells = overrides.pop("_cells", None) or matrix_cells()
    results = overrides.pop("_results", None)
    if results is None:
        results = cell_results(cells)
    plan = t3.ReleasePlanView(
        plan_id="akplan-v9", plan_sha256=digest("plan-v9"), source_tree="llama.cpp",
        backends=LLAMA_BACKENDS, cells=tuple(cells),
        incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
        incumbent_version_number=8)
    storage_state = storage.StorageState(
        state=storage.STORAGE_OK, free_bytes=200 * 1024 ** 3,
        total_bytes=3700 * 1024 ** 3, floor_bytes=50 * 1024 ** 3)
    fields = {
        "run_id": "akt3-v9-001",
        "campaign_id": "ak-v9",
        "mode": t3.MODE_DRY_RUN,
        "now": NOW,
        "protocol": t3.ProtocolBinding(
            protocol_id=t3.RELEASE_PROTOCOL_ID,
            document_sha256=digest("P-KERNEL-FREEZE-1-draft"), ratified=False),
        "sealed": sealed_candidate(),
        "plan": plan,
        "backend_unchanged": {
            b: t3.UnchangedView(backend=b, may_drop_cells=False,
                                unchanged_outcome=schemas.FAIL,
                                agreement_outcome=schemas.PASS, stage2_ran=True,
                                reasons=("the closure changed",))
            for b in LLAMA_BACKENDS},
        "host": guards.HostHealth(uptime_seconds=3600, observed_at=NOW,
                                  receipt="host-receipt-1"),
        "host_owner": "operator",
        "host_escalation_deadline": "2026-08-04T12:00:00Z",
        "resource_claims": tuple(
            guards.ResourceClaimObservation(
                resource=b, claim_kind="cpu_region" if b.endswith("cpu") else "gpu_device",
                acquired=True, observed_at=NOW, receipt=f"claim-{b}",
                held_by="akt3-v9-001")
            for b in LLAMA_BACKENDS),
        "storage_observation": guards.StorageObservation(
            path="/mnt/raid0", state=storage_state, expirable_backlog_bytes=0,
            receipt="storage-receipt-1"),
        "transaction": transaction_plan(),
        "archive": incumbent_archive(),
        "supplied_components": {name: digest(f"component:{name}")
                                for name in t3.SUPPLIED_COMPONENTS},
        "cooldown_seconds": 86400,
        # Where THIS RUN holds that operator attestations live. No suite can write to
        # `/workspace/artifacts/operator/` — the reader's default and the operator's
        # own tree — so every read waiver in this package comes from an
        # `artifacts/operator/` directory inside this checkout, which `verify_waiver`
        # refuses unless the RUN declares it. Production declares nothing and gets
        # the real root; see `t3.T3Request.attestation_roots`.
        "attestation_roots": (str(storage.REPO_ROOT / "artifacts" / "operator"),),
        "release_reps_by_protocol": {"P-BENCH-1": 10, "P-BENCH-PREFILL-1": 5,
                                     "P-KERNEL-FREEZE-1": 1},
        # Per-phase `ProtocolBinding`s, not bare ids: `P-BENCH-1` and
        # `P-BENCH-PREFILL-1` are ratified Annex B protocols and the gate now
        # requires that to be a DECLARED, hashed fact rather than a guess from a name.
        "phase_protocols": {b: {"prefill": ratified_protocol("P-BENCH-PREFILL-1"),
                                "decode": ratified_protocol("P-BENCH-1")}
                            for b in LLAMA_BACKENDS},
        "linkage_receipts": tuple(linkage_receipt(b) for b in LLAMA_BACKENDS),
        "backend_inventories": tuple(
            t3.BackendInventory(
                backend=b, entries=("CPU",) + (("HIP",) if b.endswith("gpu") else ()),
                device_entries=(("AMD Instinct MI210",) if b.endswith("gpu") else ()),
                source_ref=f"startup-log:{b}")
            for b in LLAMA_BACKENDS),
        "determinism": tuple(
            t3.DeterminismDeclaration(
                backend=b, anchor_class="bitwise_stable",
                candidate_class="bitwise_stable", evidence_ref=f"det:{b}")
            for b in LLAMA_BACKENDS),
        "cell_results": tuple(results),
        "standings": tuple(
            t3.PhaseStanding(backend=b, workload_phase=phase, protocol_id=protocol,
                             standing=standing, cell_ids=(f"{b}.{phase}",),
                             evidence_ref=f"standing:{b}.{phase}")
            for b in LLAMA_BACKENDS
            for phase, protocol, standing in (
                ("prefill", "P-BENCH-PREFILL-1", t3.STANDING_IMPROVED),
                ("decode", "P-BENCH-1", t3.STANDING_NON_INFERIOR))),
        "quality_evidence": tuple(
            t3.QualityEvidence(
                backend=b, mode=t3.QUALITY_MEASURED_PARITY,
                baseline_binary_path=f"{ARCHIVE_ROOT}/cpu/llama-server",
                baseline_binary_sha256=V8_CPU_BINARY,
                baseline_kernel="production-consolidated-v8",
                baseline_is_rebuild=False,
                evidence_refs=(f"data/ak/quality/{b}.json",),
                suites=("mmlu_pro", "gpqa"), shared_question_identity=True)
            for b in LLAMA_BACKENDS),
        "stability_evidence": tuple(
            t3.StabilityEvidence(
                backend=b, load_unload_cycles=5, memory_growth_bytes=0,
                memory_growth_allowance_bytes=1024, profiler_or_runtime_errors=0,
                cleanup_verified=True, mixed_prefill_decode_exercised=True,
                evidence_ref=f"stability:{b}")
            for b in LLAMA_BACKENDS),
        "stability_min_cycles": 5,
        "complexity": {
            b: integrity.ComplexityAssessment(
                requires_human_code_review=False, reasons=(), first_page_notice=None,
                measured={"total_changed_lines": 40, "files_touched": 2})
            for b in LLAMA_BACKENDS},
        "campaign_start_at": CAMPAIGN_START,
    }
    fields.update(overrides)
    return t3.T3Request(**fields)


# -- packager pieces ----------------------------------------------------------

def compute_window(**overrides) -> packager.ComputeWindow:
    fields = {"window_id": "cw-v9-freeze", "owner": "operator",
              "opens_at": "2026-08-04T08:00:00Z", "closes_at": "2026-08-04T20:00:00Z",
              "purpose": "T3 release matrix for the llama.cpp v9 candidate"}
    fields.update(overrides)
    return packager.ComputeWindow(**fields)


def freeze_request(**overrides) -> packager.OperatorFreezeRequest:
    fields = {"request_id": "akfr-v9-001", "campaign_id": "ak-v9",
              "source_tree": "llama.cpp", "requested_by": "daniele",
              "requested_at": "2026-08-03T09:00:00Z",
              "authority": packager.OPERATOR_AUTHORITY,
              "compute_window": compute_window(),
              "reason": "champion has accumulated four banked candidates since v8",
              "readiness_signal_ref": "journal://ak-v9/readiness/2026-08-03"}
    fields.update(overrides)
    return packager.OperatorFreezeRequest(**fields)


def sealed_release(**overrides) -> packager.SealedRelease:
    candidate = overrides.pop("candidate", None) or sealed_candidate()
    fields = {"champion_id": "akch-llama-v9", "candidate": candidate,
              "build_receipt_sha256": digest("build-receipt"),
              "seal_inputs_ref": "data/ak-v9/seal-inputs.json", "sealed_at": NOW}
    fields.update(overrides)
    return packager.SealedRelease(**fields)


def next_version(**overrides) -> packager.NextVersion:
    fields = {"incumbent_branch": "production-consolidated-v8",
              "existing_branches": ("production-consolidated-v7",
                                    "production-consolidated-v8"),
              "existing_tags": ()}
    fields.update(overrides)
    return packager.compute_next_version(**fields)


def rollback_plan(**overrides) -> packager.RollbackPlan:
    fields = {
        "archive": incumbent_archive(),
        "backends": LLAMA_BACKENDS,
        "incumbent_branch": "production-consolidated-v8",
        "incumbent_commit": V8_HEAD,
        "expected_binary_sha256": INCUMBENT_BINARIES,
        "stable_path_restore": ((CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin"),
                                (GPU_LINK, "/mnt/raid0/llm/llama.cpp/build-hip/bin")),
        "verified_at": NOW,
        # STATED, because the field is tristate and `None` means "nobody said". It
        # used to default to True, which answered §11.5's live-anchor requirement for
        # every caller who never considered it.
        "anchor_live": True,
    }
    fields.update(overrides)
    return packager.build_rollback_plan(**fields)


def era_rows() -> tuple:
    return (
        packager.EraRowDraft(
            era_id="E9-cpu-kernel", kind=packager.ERA_ROW_KIND_KERNEL,
            subject="llama.cpp kernel, CPU and GPU binaries",
            backends=LLAMA_BACKENDS, supersedes="E8-cpu-kernel",
            note="production-consolidated-v9 supersedes v8 for both llama binaries"),
        packager.EraRowDraft(
            era_id="E9-autopilot-speed", kind=packager.ERA_ROW_KIND_AUTOPILOT_SPEED,
            subject="AutoPilot speed era", backends=LLAMA_BACKENDS,
            supersedes="E8-autopilot-speed",
            note="throughput priors measured under E8 are not comparable across this "
                 "boundary and are re-derived"),
        packager.EraRowDraft(
            era_id="E9", kind=packager.ERA_ROW_KIND_UMBRELLA,
            subject="umbrella era", backends=LLAMA_BACKENDS, supersedes="E8",
            note="every number recorded after the cutover carries this era label"),
    )


def era_draft(**overrides) -> dict:
    fields = {"rows": era_rows(), "version": next_version(),
              "registry_path": ERA_REGISTRY, "incumbent_era": "E8", "drafted_at": NOW}
    fields.update(overrides)
    return packager.draft_era_registry_row(**fields)


def rebaseline_note(**overrides) -> str:
    fields = {"era_id": "E9", "baseline_path": AUTOPILOT_BASELINE,
              "affected_roles": AFFECTED_ROLES,
              "hold_reason": "a new llama kernel moves the speed era even when model "
                             "quality is identical",
              "drafted_at": NOW}
    fields.update(overrides)
    return packager.draft_autopilot_rebaseline_note(**fields)


def operator_commands() -> tuple:
    """A sequence that covers every element the transaction declares."""
    steps = [
        (f"git -C /mnt/raid0/llm/llama.cpp branch production-consolidated-v9 "
         f"{CANDIDATE_COMMIT}",
         "create the new production branch from the sealed candidate",
         "a new frozen branch exists at the candidate commit",
         ("production-consolidated-v9",),
         "git branch -D production-consolidated-v9"),
        ("git -C /mnt/raid0/llm/llama.cpp tag production-consolidated-v9",
         "tag the release", "the release tag exists",
         ("production-consolidated-v9",),
         "git tag -d production-consolidated-v9"),
        (f"install -d {INSTALL_PATH} && rsync -a {BUILD_ROOT}/ "
         "/mnt/raid0/llm/llama.cpp-v9/build/",
         "install the built binaries", "the v9 build tree is in place",
         (INSTALL_PATH,), "rm -rf /mnt/raid0/llm/llama.cpp-v9"),
        (f"ln -sfn /mnt/raid0/llm/llama.cpp-v9/build/bin {CPU_LINK}",
         "repoint the stable CPU kernel path",
         "every CPU launcher resolves the v9 binary", (CPU_LINK,),
         f"ln -sfn /mnt/raid0/llm/llama.cpp/build/bin {CPU_LINK}"),
        (f"ln -sfn /mnt/raid0/llm/llama.cpp-v9/build-hip/bin {GPU_LINK}",
         "repoint the stable GPU kernel path",
         "every GPU launcher resolves the v9 binary", (GPU_LINK,),
         f"ln -sfn /mnt/raid0/llm/llama.cpp/build-hip/bin {GPU_LINK}"),
        (f"$EDITOR {ERA_REGISTRY}  # write the three drafted E9 rows",
         "write the drafted era-registry rows",
         "E9-cpu-kernel, E9-autopilot-speed and E9 exist", (ERA_REGISTRY,),
         f"git checkout -- {ERA_REGISTRY}"),
        (f"$EDITOR {AUTOPILOT_BASELINE}  # open the fail-closed E9 rebaseline hold",
         "open the fail-closed AutoPilot rebaseline hold",
         "the E9 hold is closed until an operator-ratified reseed",
         (AUTOPILOT_BASELINE,), f"git checkout -- {AUTOPILOT_BASELINE}"),
        (f"sha256sum -c {ARCHIVE_ROOT}/SHA256SUMS",
         "verify the archived incumbent is intact before anything moves",
         "the rollback anchor is verified live", (ARCHIVE_ROOT,), None),
        ("cp -a artifacts/operator/v9-freeze/ artifacts/operator/v9-freeze.bak/",
         "record the freeze receipts", "the receipts are durable in-repo",
         ("artifacts/operator/v9-freeze/",), None),
    ]
    return tuple(
        packager.OperatorCommand(
            step=index + 1, command=command, purpose=purpose,
            expected_effect=effect, target_paths=targets,
            validation_receipt=digest(f"prevalidation:{index}"),
            validation_method="static pre-validation: shape, target scope, transaction "
                              "coverage, declared rollback",
            validated=True, rollback_command=rollback)
        for index, (command, purpose, effect, targets, rollback) in enumerate(steps))


def watch_bands(**overrides) -> tuple:
    edges = {
        packager.SIGNAL_THROUGHPUT: {"unit": "tokens_per_s", "lower": 46.0, "mde": 0.9,
                                     "roles": AFFECTED_ROLES},
        packager.SIGNAL_LATENCY: {"unit": "ms", "lower": 0.0, "upper": 1800.0,
                                  "roles": AFFECTED_ROLES},
        packager.SIGNAL_ERROR_RATES: {"unit": "fraction", "upper": 0.004},
        packager.SIGNAL_MEMORY: {"unit": "gib_headroom", "lower": 50.0},
        packager.SIGNAL_QUALITY: {"unit": "score", "lower": 0.71},
        packager.SIGNAL_SUPERVISOR: {"unit": "events", "upper": 0.0},
    }
    edges.update(overrides.pop("edges", {}))
    fields = {
        "basis_ref_by_signal": {
            signal: f"data/ak-v9/incumbent-era-E8/{signal}.jsonl"
            for signal in packager.REQUIRED_WATCH_SIGNALS},
        "noise_reference_ref": "data/ak-v9/standing-noise-reference-E8.json",
        "edges": edges,
    }
    fields.update(overrides)
    return packager.default_watch_bands(**fields)


def watch_window(**overrides) -> packager.WatchWindow:
    fields = {
        "window_id": "akww-v9-001", "package_id": "akr-v9-001", "owner": "operator",
        "incumbent_era": "E8", "candidate_era": "E9",
        "affected_roles": AFFECTED_ROLES,
        "min_duration_days": packager.DEFAULT_WATCH_WINDOW_DAYS,
        "min_volume_by_role": {"worker_general": 20000, "frontdoor": 5000},
        "bands": watch_bands(), "bands_fixed_at": NOW, "opens_at": NOW,
        "close_step": packager.WatchWindowCloseStep(owner="operator"),
        "rollback_anchor_ref": f"{ARCHIVE_ROOT} (production-consolidated-v8)",
    }
    fields.update(overrides)
    return packager.WatchWindow(**fields)


def watch_progress(*, days: float = 8.0, volumes=None, values=None, omit=(),
                   bands_sha256=None, era="E9") -> packager.WatchWindowProgress:
    """Progress that, by default, closes the window with every signal inside its band."""
    observed_at = "2026-08-11T12:00:00Z" if days >= 8 else "2026-08-05T12:00:00Z"
    defaults = {
        packager.SIGNAL_THROUGHPUT: 48.5,
        packager.SIGNAL_LATENCY: 900.0,
        packager.SIGNAL_ERROR_RATES: 0.001,
        packager.SIGNAL_MEMORY: 61.0,
        packager.SIGNAL_QUALITY: 0.74,
        packager.SIGNAL_SUPERVISOR: 0.0,
    }
    if values is not None:
        defaults.update(values)
    observations = tuple(
        packager.WatchObservation(
            signal_id=signal, value=value, observed_at=observed_at, era_label=era,
            samples_ref=f"data/ak-v9/watch/{signal}.jsonl")
        for signal, value in defaults.items() if signal not in omit)
    return packager.WatchWindowProgress(
        now=observed_at,
        volume_by_role=volumes if volumes is not None
        else {"worker_general": 40000, "frontdoor": 9000},
        bands_sha256=bands_sha256 or watch_window().bands_sha256(),
        observations=observations)


def cutover_request(**overrides) -> packager.CutoverRequest:
    fields = {"message_id": "msg-20260803T120000Z-001-autokernel",
              "from_agent": "autokernel", "to_agent": "coordinator-agent",
              "needs_routing_to": ("inference",), "task_id": "ak-v9-cutover",
              "created_at": NOW, "package_id": "akr-v9-001",
              "transaction": transaction_plan(), "rollback": rollback_plan()}
    fields.update(overrides)
    return packager.build_cutover_request(**fields)


def evaluation(**overrides) -> packager.TrustedEvaluation:
    request = overrides.pop("request", None) or t3_request()
    evaluator = overrides.pop("evaluator", None) or t3.T3Runner()
    return packager.run_release_evaluation(request, evaluator=evaluator)


DIFF_COMPLEXITY = {"diff_size": 40, "files_touched": 2, "touches_shared_core": False}


def release_package(**overrides) -> packager.ReleasePackage:
    """A complete READY package. Every test perturbs exactly one thing."""
    fields = {
        "package_id": "akr-v9-001",
        "created_at": NOW,
        "freeze_request": freeze_request(),
        "sealed": sealed_release(),
        "evaluation": overrides.pop("evaluation", None) or evaluation(),
        "version": next_version(),
        "transaction": transaction_plan(),
        "rollback": rollback_plan(),
        "era_row_draft": era_draft(),
        "rebaseline_note": rebaseline_note(),
        "commands": operator_commands(),
        "watch_window": watch_window(),
        "cutover_request": cutover_request(),
        "autopilot_baseline_path": AUTOPILOT_BASELINE,
        "change_classes": ("arithmetic",),
        "diff_complexity": dict(DIFF_COMPLEXITY),
    }
    fields.update(overrides)
    return packager.assemble_release_package(**fields)


def unverified_waiver_binding(**overrides) -> t3.WaiverBinding:
    """A waiver pinned into the package that the T3 run never saw, so never verified."""
    document = {
        "schema": schemas.SCHEMA_OPERATOR_WAIVER,
        "waiver_id": "WAIVE-Q8-v9",
        "campaign_id": "ak-v9",
        "decision": "release without the Q8 non-regression claim",
        "protocol": "P-BENCH-PREFILL-1",
        "protocol_changed": False,
        "candidate_head": CANDIDATE_COMMIT,
        "production_head": V8_HEAD,
        "scope": {"excluded_models": ["qwen36_q8"], "excluded_pairs": [],
                  "covers_cell_ids": ["llama_cpu.prefill"]},
        "reason": "the Q8 pair was never measured under this protocol",
        "consequences": ["no Q8 prefill non-regression claim"],
    }
    fields = {"waiver_id": "WAIVE-Q8-v9", "pinned_sha256": digest("waive-q8-v9"),
              "document": document, "document_path": "artifacts/operator/waive-q8-v9.json",
              "covers_cell_ids": ("llama_cpu.prefill",)}
    fields.update(overrides)
    return t3.WaiverBinding(**fields)


def codes_of(package: packager.ReleasePackage) -> set:
    return {f.code for f in package.findings}


# =============================================================================
# The boundary — the cardinal rule, proved and then proved to bite
# =============================================================================

class TestTheBoundary(unittest.TestCase):
    """AutoKernel never freezes and never cuts over, checked four ways."""

    def test_the_four_self_audits_pass_on_this_module(self):
        for audit in (packager.audit_no_write_or_process_paths,
                      packager.audit_no_clock_or_self_trigger,
                      packager.audit_verdict_is_delegated,
                      packager.audit_refusal_doors_raise_unconditionally):
            with self.subTest(audit=audit.__name__):
                check = audit()
                self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    def test_the_write_audit_bites_on_an_import(self):
        source = f'MODULE_ID = "{packager.MODULE_ID}"\nimport os\n'
        check = packager.audit_no_write_or_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("imports 'os'", " ".join(check.reasons))

    def test_the_write_audit_bites_on_the_pathlib_write_and_symlink_verbs(self):
        for snippet, needle in (('Path(p).open("w")', ".open()"),
                                ('Path(new).replace(link)', ".replace()"),
                                ('Path(a).symlink_to(b)', ".symlink_to()"),
                                ('handle.write_text(x)', ".write_text()"),
                                ('getattr(p, "unlink")()', "'unlink'")):
            with self.subTest(snippet=snippet):
                source = f'MODULE_ID = "{packager.MODULE_ID}"\n{snippet}\n'
                check = packager.audit_no_write_or_process_paths(source)
                self.assertEqual(check.outcome, schemas.FAIL)
                self.assertIn(needle, " ".join(check.reasons))

    def test_an_audit_over_foreign_or_empty_source_is_not_a_pass(self):
        """The guarantee must not be obtainable by deleting what it inspects."""
        for source in ("", "x = 1\n", 'MODULE_ID = "somebody.else/v1"\n'):
            with self.subTest(source=source):
                for audit in (packager.audit_no_write_or_process_paths,
                              packager.audit_no_clock_or_self_trigger,
                              packager.audit_verdict_is_delegated,
                              packager.audit_refusal_doors_raise_unconditionally):
                    check = audit(source)
                    self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK,
                                     f"{audit.__name__} passed on {source!r}")

    def test_the_clock_audit_bites(self):
        source = (f'MODULE_ID = "{packager.MODULE_ID}"\n'
                  'moment = datetime.now()\n')
        check = packager.audit_no_clock_or_self_trigger(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("reads a clock", " ".join(check.reasons))

    def test_the_self_trigger_audit_bites_on_minting_a_request(self):
        source = (f'MODULE_ID = "{packager.MODULE_ID}"\n'
                  'req = OperatorFreezeRequest(request_id="akfr-x")\n')
        check = packager.audit_no_clock_or_self_trigger(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("constructs OperatorFreezeRequest", " ".join(check.reasons))

    def test_the_delegation_audit_bites_on_running_the_gate_directly(self):
        for snippet in ("t3.run_t3(request)", "compute_verdict(a, b)",
                        "t3.phase_seal(request, results)"):
            with self.subTest(snippet=snippet):
                source = f'MODULE_ID = "{packager.MODULE_ID}"\n{snippet}\n'
                check = packager.audit_verdict_is_delegated(source)
                self.assertEqual(check.outcome, schemas.FAIL)
                self.assertIn("trusted evaluator", " ".join(check.reasons))

    def test_the_refusal_door_audit_bites_on_a_softened_door(self):
        """A door with an `if` in it is a capability with a precondition."""
        body = "\n".join(
            f"def {name}(*a, **k):\n"
            f"    if a:\n        raise RuntimeError('no')\n    return None\n"
            for name in packager.REFUSED_CAPABILITIES.values())
        source = f'MODULE_ID = "{packager.MODULE_ID}"\n{body}'
        check = packager.audit_refusal_doors_raise_unconditionally(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("single unconditional raise", " ".join(check.reasons))

    def test_the_refusal_door_audit_bites_on_a_missing_door(self):
        source = f'MODULE_ID = "{packager.MODULE_ID}"\n'
        check = packager.audit_refusal_doors_raise_unconditionally(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("no function named", " ".join(check.reasons))

    def test_every_refused_capability_raises(self):
        for capability, name in sorted(packager.REFUSED_CAPABILITIES.items()):
            with self.subTest(capability=capability):
                door = getattr(packager, name)
                with self.assertRaises(packager.PackagerError):
                    door("anything", keyword="at all")

    def test_the_four_human_only_boundaries_each_have_a_door(self):
        """§1.3 enumerates four; each must be refusable by name, not by absence."""
        doors = set(packager.REFUSED_CAPABILITIES)
        for boundary in ("perform_the_freeze", "write_an_era_registry_row",
                         "apply_an_autopilot_baseline", "move_a_stable_kernel_symlink"):
            self.assertIn(boundary, doors)

    def test_the_package_record_carries_no_authority_flavoured_key(self):
        package = release_package()
        self.assertEqual(schemas.find_authority_flavoured_keys(package.to_dict()), [])

    def test_the_module_declares_its_own_refusals_in_the_record(self):
        package = release_package()
        self.assertEqual(sorted(packager.REFUSED_CAPABILITIES),
                         package.to_dict()["refusals"])


# =============================================================================
# AK7 — the freeze request the operator invokes, which cannot self-trigger
# =============================================================================

class TestModuleSurface(unittest.TestCase):

    def test_every_public_name_is_exported_and_every_export_exists(self):
        self.assertEqual([n for n in packager.__all__ if not hasattr(packager, n)], [])
        imported = {"annotations", "ast", "re", "dataclass", "field", "datetime",
                    "timezone", "Path", "Any", "Iterable", "Mapping", "Optional",
                    "Sequence", "schemas", "storage", "t3", "integrity"}
        unexported = sorted(n for n in dir(packager)
                            if not n.startswith("_") and n not in packager.__all__
                            and n not in imported)
        self.assertEqual(unexported, [])

    def test_the_package_schema_is_the_shared_one_not_a_private_shape(self):
        self.assertEqual(packager.PACKAGE_SCHEMA, schemas.SCHEMA_RELEASE_PACKAGE)

    def test_the_cutover_envelope_is_the_bus_envelope(self):
        self.assertEqual(packager.CUTOVER_MESSAGE_SCHEMA, "session_bus.msg.v1")


class TestFreezeRequest(unittest.TestCase):

    def test_a_well_formed_operator_request_is_accepted(self):
        request = freeze_request()
        self.assertEqual(request.authority, packager.OPERATOR_AUTHORITY)
        self.assertEqual(request.to_dict()["schema"], packager.FREEZE_REQUEST_SCHEMA)
        self.assertIn("out of scope", request.to_dict()["scope_note"].lower())

    def test_machine_authority_is_refused(self):
        for authority in ("autokernel", "self", "controller", "", "loop"):
            with self.subTest(authority=authority):
                with self.assertRaises(packager.PackagerError):
                    freeze_request(authority=authority)

    def test_a_machine_requester_is_refused(self):
        for who in ("autokernel", "ak-controller", "coordinator daemon",
                    "packager-subagent", "cron"):
            with self.subTest(who=who):
                with self.assertRaises(packager.SelfTriggerRefused) as caught:
                    freeze_request(requested_by=who)
                self.assertIn("machine actor", str(caught.exception))

    def test_a_human_name_that_merely_contains_a_token_is_not_refused(self):
        """The guard must not forbid its own compliant idiom."""
        for who in ("daniele", "d.pinna", "operator-daniele", "Daniele Pinna"):
            with self.subTest(who=who):
                self.assertTrue(freeze_request(requested_by=who).request_id)

    def test_a_request_without_a_compute_window_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            freeze_request(compute_window=None)
        self.assertIn("compute window", str(caught.exception).lower())

    def test_a_window_that_has_already_closed_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            freeze_request(compute_window=compute_window(
                opens_at="2026-08-01T08:00:00Z", closes_at="2026-08-01T20:00:00Z"))
        self.assertIn("no window", str(caught.exception))

    def test_a_zero_length_window_is_refused(self):
        with self.assertRaises(packager.PackagerInputError):
            compute_window(opens_at="2026-08-04T08:00:00Z",
                           closes_at="2026-08-04T08:00:00Z")

    def test_serving_runtime_has_no_source_tree_and_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            freeze_request(source_tree="serving_runtime")
        self.assertIn("11.6", str(caught.exception))

    def test_the_readiness_signal_is_context_and_not_a_trigger(self):
        """AK-D3 / P-AK-SEARCH-1 denial 5: a readiness signal is not a freeze trigger."""
        fields = {f.name for f in dataclasses.fields(packager.OperatorFreezeRequest)}
        self.assertNotIn("triggered_by", fields)
        self.assertIn("readiness_signal_ref", fields)
        self.assertIsNone(freeze_request(readiness_signal_ref=None).readiness_signal_ref)

    def test_the_scope_note_says_a_freeze_needs_an_operator_and_a_compute_window(self):
        note = packager.AK7_SCOPE_NOTE.lower()
        self.assertIn("operator", note)
        self.assertIn("compute window", note)
        self.assertIn("out of scope", note)
        self.assertIn(packager.STATE_READY.lower(), note)


# =============================================================================
# Sealing the champion — the six refusals AK6 names
# =============================================================================

class TestSealing(unittest.TestCase):

    def seal(self, **overrides):
        fields = {"champion_id": "akch-llama-v9", "candidate": sealed_candidate(),
                  "build_receipt_sha256": digest("build-receipt"),
                  "seal_inputs_ref": "data/ak-v9/seal-inputs.json", "sealed_at": NOW,
                  "pinned_evaluator_bundle_sha256": EVALUATOR_BUNDLE,
                  "incumbent_branch": "production-consolidated-v8",
                  "incumbent_commit": BASE_COMMIT}
        fields.update(overrides)
        return packager.seal_champion(**fields)

    def test_a_complete_champion_seals(self):
        sealed = self.seal()
        self.assertEqual(sealed.backends, LLAMA_BACKENDS)
        self.assertEqual(sealed.candidate.candidate_id, "akc-v9")

    def test_evaluator_drift_is_refused(self):
        with self.assertRaises(packager.SealRefused) as caught:
            self.seal(pinned_evaluator_bundle_sha256=digest("some-other-bundle"))
        self.assertIn("evaluator drift", str(caught.exception))

    def test_missing_hashes_are_refused_per_backend(self):
        for field_name in ("binary_sha256", "linkage_sha256", "build_dirs"):
            with self.subTest(field=field_name):
                partial = dict(getattr(sealed_candidate(), field_name))
                partial.pop("llama_gpu")
                with self.assertRaises(packager.SealRefused) as caught:
                    self.seal(candidate=sealed_candidate(**{field_name: partial}))
                self.assertIn("missing hashes", str(caught.exception))
                self.assertIn(f"llama_gpu.{field_name}", str(caught.exception))

    def test_dirty_ancestry_tree_or_missing_overlay_is_refused(self):
        for field_name in ("ancestry_clean", "tree_clean", "overlay_present"):
            with self.subTest(field=field_name):
                with self.assertRaises(packager.SealRefused) as caught:
                    self.seal(candidate=sealed_candidate(**{field_name: False}))
                self.assertIn(field_name, str(caught.exception))

    def test_a_build_inside_a_frozen_production_tree_is_refused(self):
        with self.assertRaises(packager.IncumbentModificationRefused) as caught:
            self.seal(candidate=sealed_candidate(
                build_dirs={b: "/mnt/raid0/llm/llama.cpp/build" for b in LLAMA_BACKENDS}))
        self.assertIn("Invariant 3", str(caught.exception))

    def test_a_seal_anchored_off_the_incumbent_tip_is_refused(self):
        with self.assertRaises(packager.SealRefused) as caught:
            self.seal(incumbent_commit="f" * 39 + "e")
        self.assertIn("Invariant 1", str(caught.exception))

    def test_a_production_named_candidate_branch_is_refused(self):
        with self.assertRaises(t3.T3InputError):
            sealed_candidate(candidate_branch="production-consolidated-v9")

    def test_an_unparseable_incumbent_branch_is_refused(self):
        with self.assertRaises(packager.SealRefused) as caught:
            self.seal(incumbent_branch="main")
        self.assertIn("successor", str(caught.exception))

    def test_the_rollup_digests_do_not_lose_the_per_backend_map(self):
        sealed = self.seal()
        record = sealed.to_dict()
        self.assertEqual(set(record["binary_sha256_by_backend"]), set(LLAMA_BACKENDS))
        self.assertEqual(record["binary_sha256"],
                         schemas.content_hash(dict(sealed.candidate.binary_sha256)))
        self.assertNotEqual(record["binary_sha256"], record["linkage_sha256"])


# =============================================================================
# Running T3 through the trusted evaluator (invariant 4)
# =============================================================================

class _WrongTier:
    tier = "T1"

    def evaluate_release(self, request):  # pragma: no cover - never reached
        raise AssertionError("must not be called")


class _NoSeam:
    tier = "T3"


class _ReturnsRubbish:
    tier = "T3"

    def evaluate_release(self, request):
        return {"verdict": "PASS"}


class _GradesSomethingElse:
    """Returns a real result whose fingerprint belongs to a different request."""

    tier = "T3"

    def evaluate_release(self, request):
        result = t3.run_t3(request)
        return dataclasses.replace(result, fingerprint=digest("another-request")[:32])


class TestTrustedEvaluation(unittest.TestCase):

    def test_the_happy_path_delegates_and_verifies_the_seam(self):
        result = evaluation()
        self.assertEqual(result.verdict, "PASS")
        self.assertEqual(result.check.outcome, schemas.PASS)
        self.assertEqual(result.evaluator_class, "T3Runner")
        self.assertEqual(result.evaluator_tier, t3.TIER)
        self.assertIsNotNone(result.bundle_sha256)

    def test_a_non_release_tier_evaluator_is_refused(self):
        with self.assertRaises(packager.EvaluatorNotTrusted) as caught:
            packager.run_release_evaluation(t3_request(), evaluator=_WrongTier())
        self.assertIn("admit_tier", str(caught.exception))

    def test_an_evaluator_without_the_seam_is_refused(self):
        with self.assertRaises(packager.EvaluatorNotTrusted) as caught:
            packager.run_release_evaluation(t3_request(), evaluator=_NoSeam())
        self.assertIn("evaluate_release", str(caught.exception))

    def test_an_evaluator_returning_a_non_result_is_refused(self):
        with self.assertRaises(packager.EvaluatorNotTrusted) as caught:
            packager.run_release_evaluation(t3_request(), evaluator=_ReturnsRubbish())
        self.assertIn("this module computed", str(caught.exception))

    def test_a_verdict_for_a_different_request_does_not_verify(self):
        result = packager.run_release_evaluation(t3_request(),
                                                 evaluator=_GradesSomethingElse())
        self.assertEqual(result.check.outcome, schemas.FAIL)
        self.assertIn("different sealed candidate", " ".join(result.check.reasons))

    def test_a_seam_failure_blocks_the_package(self):
        package = release_package(
            evaluation=packager.run_release_evaluation(
                t3_request(), evaluator=_GradesSomethingElse()))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("EVALUATION_SEAM_UNSOUND", codes_of(package))


# =============================================================================
# The next version — reused names and a stale incumbent
# =============================================================================

class TestNextVersion(unittest.TestCase):

    def test_v8_yields_v9_and_era_e9(self):
        version = next_version()
        self.assertEqual(version.next_branch, "production-consolidated-v9")
        self.assertEqual(version.next_version_number, 9)
        self.assertEqual(version.era_prefix, "E9")

    def test_a_reused_version_name_is_refused(self):
        with self.assertRaises(packager.VersionCollision) as caught:
            next_version(existing_branches=("production-consolidated-v8",
                                            "production-consolidated-v9"))
        self.assertIn("already exist", str(caught.exception))

    def test_a_reused_tag_is_refused(self):
        with self.assertRaises(packager.VersionCollision):
            next_version(existing_tags=("production-consolidated-v9",))

    def test_a_stale_incumbent_is_refused(self):
        with self.assertRaises(packager.VersionCollision) as caught:
            next_version(existing_branches=("production-consolidated-v8",
                                            "production-consolidated-v10"))
        self.assertIn("not the tip", str(caught.exception))

    def test_a_different_family_does_not_collide(self):
        """The guard must not forbid its own idiom: speech and consolidated are series."""
        version = next_version(
            incumbent_branch="production-speech-v1",
            existing_branches=("production-speech-v1", "production-consolidated-v9"))
        self.assertEqual(version.next_branch, "production-speech-v2")

    def test_a_non_production_incumbent_has_no_successor(self):
        with self.assertRaises(packager.PackagerInputError):
            next_version(incumbent_branch="main")


# =============================================================================
# Rollback and the incumbent archive (§10.5)
# =============================================================================

class TestRollback(unittest.TestCase):

    def test_a_complete_archive_verifies(self):
        plan = rollback_plan()
        self.assertEqual(plan.archive_check.outcome, schemas.PASS)
        self.assertEqual(plan.rollback_branch, "production-consolidated-v8")
        self.assertEqual(set(plan.backends), set(LLAMA_BACKENDS))

    def test_a_rebuilt_archive_is_not_the_incumbent(self):
        check = packager.verify_archive_target(
            incumbent_archive(rebuilt=True), backends=LLAMA_BACKENDS,
            incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
            expected_binary_sha256=INCUMBENT_BINARIES)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("rebuilt", " ".join(check.reasons))

    def test_a_hash_mismatch_is_a_fail_not_a_note(self):
        check = packager.verify_archive_target(
            incumbent_archive(), backends=LLAMA_BACKENDS,
            incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
            expected_binary_sha256={"llama_cpu": digest("some-other-binary"),
                                    "llama_gpu": V8_GPU_BINARY})
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("is not among the archived binaries", " ".join(check.reasons))

    def test_an_unsupplied_incumbent_digest_is_could_not_check(self):
        check = packager.verify_archive_target(
            incumbent_archive(), backends=LLAMA_BACKENDS,
            incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
            expected_binary_sha256={"llama_cpu": V8_CPU_BINARY})
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("cannot be checked", " ".join(check.reasons))

    def test_a_scratch_archive_root_is_refused(self):
        check = packager.verify_archive_target(
            incumbent_archive(archive_root="/mnt/raid0/llm/tmp/v8-backup"),
            backends=LLAMA_BACKENDS, incumbent_branch="production-consolidated-v8",
            incumbent_commit=V8_HEAD, expected_binary_sha256=INCUMBENT_BINARIES)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("scratch path", " ".join(check.reasons))

    def test_an_empty_archive_leaves_nothing_to_roll_back_to(self):
        with self.assertRaises(packager.RollbackIncomplete):
            rollback_plan(archive=t3.IncumbentArchive(
                entries=(), no_incumbent_reason="the archive directory is empty"))

    def test_a_missing_backend_binary_is_an_incomplete_rollback(self):
        with self.assertRaises(packager.RollbackIncomplete) as caught:
            rollback_plan(expected_binary_sha256={"llama_cpu": V8_CPU_BINARY})
        self.assertIn("llama_gpu", str(caught.exception))

    def test_a_rollback_that_restores_no_stable_path_is_incomplete(self):
        with self.assertRaises(packager.RollbackIncomplete) as caught:
            rollback_plan(stable_path_restore=())
        self.assertIn("stable kernel paths", str(caught.exception))

    def test_an_unverified_archive_blocks_the_package(self):
        package = release_package(
            rollback=rollback_plan(archive=incumbent_archive(rebuilt=True)))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("ROLLBACK_ARCHIVE_UNVERIFIED", codes_of(package))

    def test_an_unsupplied_digest_makes_the_package_incomplete_not_blocked(self):
        """The third outcome survives all the way to the package's state."""
        plan = packager.build_rollback_plan(
            archive=incumbent_archive(), backends=("llama_cpu",),
            incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
            expected_binary_sha256={"llama_cpu": V8_CPU_BINARY},
            stable_path_restore=((CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin"),
                                 (GPU_LINK, "/mnt/raid0/llm/llama.cpp/build-hip/bin")),
            verified_at=NOW)
        self.assertEqual(plan.archive_check.outcome, schemas.PASS)
        package = release_package(rollback=plan)
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("ROLLBACK_BACKEND_UNCOVERED", codes_of(package))


class TestRollbackLibraryAttribution(unittest.TestCase):
    """`incumbent_libraries` used to carry `(path, sha256)` and a comment saying the
    attribution could not be invented here. Correct — so it was added at the SOURCE
    (`t3.ArchivedBuild`) and this field transports it. On a three-ggml-generation
    host it is the field a rollback would most want attributed."""

    def test_the_attribution_flows_from_the_archive_into_the_plan(self):
        """The bite: before the source carried it, there was nothing to flow."""
        plan = rollback_plan()
        self.assertEqual(
            plan.incumbent_libraries,
            ((("llama_cpu", "llama_gpu"), f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0",
              digest("v8-libggml-base")),))
        self.assertEqual(plan.to_dict()["incumbent_libraries"][0]["backends"],
                         ["llama_cpu", "llama_gpu"])

    def test_the_plan_will_not_carry_an_unattributed_library(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            dataclasses.replace(
                rollback_plan(),
                incumbent_libraries=((f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0",
                                      digest("v8-libggml-base")),))
        self.assertIn("(backends, path, sha256)", str(caught.exception))

    def test_an_unknown_backend_is_refused_in_the_plan_too(self):
        with self.assertRaises(packager.PackagerInputError):
            dataclasses.replace(
                rollback_plan(),
                incumbent_libraries=((("not_a_backend",),
                                      f"{ARCHIVE_ROOT}/cpu/lib.so", digest("lib")),))

    def test_a_backend_with_no_attributed_library_fails_the_archive_check(self):
        """What the attribution is FOR: an archive with one library and two backends
        was previously indistinguishable from one library PER backend."""
        cpu_only = incumbent_archive(libraries=(
            (("llama_cpu",), f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0",
             digest("v8-libggml-base")),))
        check = packager.verify_archive_target(
            cpu_only, backends=LLAMA_BACKENDS,
            incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
            expected_binary_sha256=INCUMBENT_BINARIES)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("attributes no preserved library to this backend",
                      " ".join(check.reasons))
        self.assertIn("llama_gpu", " ".join(check.reasons))

    def test_a_shared_library_covers_both_backends(self):
        """Control: the check must not demand one library per backend when one
        `libggml-base.so.0` genuinely serves both."""
        self.assertEqual(rollback_plan().archive_check.outcome, schemas.PASS)


# =============================================================================
# The transaction plan
# =============================================================================

class TestTransactionPlan(unittest.TestCase):

    def build(self, **overrides):
        fields = {
            "version": next_version(), "install_path": INSTALL_PATH,
            "stable_path_moves": (
                (CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin",
                 "/mnt/raid0/llm/llama.cpp-v9/build/bin"),
                (GPU_LINK, "/mnt/raid0/llm/llama.cpp/build-hip/bin",
                 "/mnt/raid0/llm/llama.cpp-v9/build-hip/bin")),
            "service_impact": ("llama-server restart at the owner's boundary",),
            "era_actions": ({"draft": True, "action": "write_era_registry_row",
                             "registry_path": ERA_REGISTRY},),
            "receipt_paths": ("artifacts/operator/v9-freeze/",),
            "rollback": rollback_plan(),
        }
        fields.update(overrides)
        return packager.build_transaction_plan(**fields)

    def test_the_transaction_is_a_dry_run_bound_to_the_rollback_anchor(self):
        transaction = self.build()
        self.assertFalse(transaction.executed)
        self.assertEqual(transaction.next_branch, "production-consolidated-v9")
        self.assertEqual(transaction.rollback_head, V8_HEAD)

    def test_an_executed_transaction_is_refused(self):
        with self.assertRaises(t3.ProductionWriteRefused):
            transaction_plan(executed=True)

    def test_an_era_action_without_draft_true_is_refused(self):
        with self.assertRaises(t3.T3InputError) as caught:
            transaction_plan(era_actions=({"action": "write_era_registry_row"},))
        self.assertIn("draft=True", str(caught.exception))

    def test_a_transaction_that_repoints_nothing_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            self.build(stable_path_moves=())
        self.assertIn("repoints nothing", str(caught.exception))

    def test_a_moved_path_the_rollback_cannot_restore_is_refused(self):
        with self.assertRaises(packager.RollbackIncomplete) as caught:
            self.build(stable_path_moves=(
                ("/mnt/raid0/llm/kernels/production/stt", "/a", "/b"),))
        self.assertIn("does not restore", str(caught.exception))

    def test_a_rollback_anchor_equal_to_the_target_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            self.build(version=packager.compute_next_version(
                incumbent_branch="production-consolidated-v7",
                existing_branches=("production-consolidated-v7",)))
        self.assertIn("no fallback", str(caught.exception))

    def test_a_next_target_inside_the_production_tree_is_still_expressible(self):
        """The guard must not forbid the operator's own normal action (§11.2)."""
        transaction = self.build(stable_path_moves=(
            (CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin",
             "/mnt/raid0/llm/llama.cpp/build/bin"),
            (GPU_LINK, "/a", "/b")))
        self.assertEqual(transaction.symlink_diff[0][2],
                         "/mnt/raid0/llm/llama.cpp/build/bin")

    def test_a_version_mismatch_between_plan_and_transaction_blocks(self):
        package = release_package(transaction=transaction_plan(
            next_branch="production-consolidated-v10", next_version_number=10,
            next_tag="production-consolidated-v10"))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("TRANSACTION_VERSION_MISMATCH", codes_of(package))


# =============================================================================
# The drafts (§1.3 items 2 and 3, §11.4)
# =============================================================================

class TestDrafts(unittest.TestCase):

    def test_the_era_block_is_a_draft_written_by_the_operator(self):
        draft = era_draft()
        self.assertIs(draft["draft"], True)
        self.assertEqual(draft["written_by"], packager.EXECUTED_BY)
        self.assertIs(draft["human_only_path"], True)
        self.assertEqual(draft["registry_path"], ERA_REGISTRY)
        self.assertEqual(sorted(draft["kinds_present"]), sorted(packager.ERA_ROW_KINDS))

    def test_the_v8_precedent_of_three_rows_is_the_requirement(self):
        """`E8-cpu-kernel`, `E8-autopilot-speed`, `E8` — three, not one."""
        self.assertEqual(len(era_draft()["rows"]), 3)
        partial = era_draft(rows=era_rows()[:1])
        package = release_package(era_row_draft=partial)
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("ERA_ROW_KIND_MISSING", codes_of(package))

    def test_an_era_id_without_the_successor_prefix_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            era_draft(rows=(dataclasses.replace(era_rows()[0], era_id="E8-cpu-kernel"),)
                      + era_rows()[1:])
        self.assertIn("successor prefix", str(caught.exception))

    def test_duplicate_era_ids_are_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            era_draft(rows=era_rows() + (era_rows()[0],))
        self.assertIn("duplicate", str(caught.exception))

    def test_an_era_row_may_not_predict_when_the_operator_acts(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            dataclasses.replace(era_rows()[0], effective_from="2026-08-04T12:00:00Z")
        self.assertIn("predicting when a human will act", str(caught.exception))

    def test_the_rebaseline_note_follows_the_e8_fail_closed_precedent(self):
        note = rebaseline_note()
        self.assertIn("FAIL-CLOSED", note)
        self.assertIn("operator-ratified", note)
        self.assertIn(AUTOPILOT_BASELINE, note)
        self.assertIn("human-only", note)
        self.assertIn("AutoKernel does not apply this", note)
        for role in AFFECTED_ROLES:
            self.assertIn(role, note)

    def test_a_note_that_names_no_baseline_file_blocks_the_package(self):
        package = release_package(rebaseline_note=rebaseline_note(
            baseline_path="orchestration/some_other_file.yaml"))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("REBASELINE_NOTE_NAMES_NO_BASELINE", codes_of(package))


# =============================================================================
# The pre-validated operator command sequence (MEASUREMENT.md:138-145)
# =============================================================================

class TestOperatorCommands(unittest.TestCase):

    def review(self, commands=None, **overrides):
        fields = {"transaction": transaction_plan(), "rollback": rollback_plan(),
                  "era_row": era_draft(),
                  "autopilot_baseline_path": AUTOPILOT_BASELINE}
        fields.update(overrides)
        return packager.validate_command_sequence(
            commands if commands is not None else operator_commands(), **fields)

    def test_the_reference_sequence_pre_validates(self):
        review = self.review()
        self.assertEqual(review.check.outcome, schemas.PASS, list(review.findings))
        self.assertEqual(review.unvalidated_commands, ())
        self.assertEqual(review.uncovered_elements, ())

    def test_human_only_is_derived_from_the_command_not_declared(self):
        commands = {c.step: c for c in operator_commands()}
        self.assertTrue(commands[4].human_only)   # repoints kernels/production/cpu
        self.assertTrue(commands[6].human_only)   # writes instrument_eras.yaml
        self.assertTrue(commands[7].human_only)   # applies autopilot_baseline.yaml
        self.assertFalse(commands[8].human_only)  # verifies the archive
        fields = {f.name for f in dataclasses.fields(packager.OperatorCommand)}
        self.assertNotIn("human_only", fields)

    def test_a_command_this_module_would_execute_is_refused(self):
        with self.assertRaises(packager.ProductionWriteRefused) as caught:
            dataclasses.replace(operator_commands()[0], executed_by="autokernel")
        self.assertIn("11.2", str(caught.exception))

    def test_an_unvalidated_command_is_filed_separately_and_blocks(self):
        commands = list(operator_commands())
        commands[3] = dataclasses.replace(commands[3], validated=False)
        review = self.review(tuple(commands))
        self.assertEqual(review.check.outcome, schemas.FAIL)
        self.assertEqual([c.step for c in review.unvalidated_commands], [4])
        self.assertIn("COMMAND_NOT_PRE_VALIDATED", " ".join(review.findings))
        package = release_package(commands=tuple(commands))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertNotIn(
            4, [c["step"] for c in package.to_dict()["operator_command_sequence"]])

    def test_an_unvalidated_command_never_rides_in_the_validated_array(self):
        """`schemas` refuses `validated: false` there; nothing may smuggle one in."""
        commands = list(operator_commands())
        commands[0] = dataclasses.replace(commands[0], validated=False)
        package = release_package(commands=tuple(commands))
        for entry in package.to_dict()["operator_command_sequence"]:
            self.assertIs(entry["validated"], True)
        self.assertEqual(package.to_dict()["command_review"]["unvalidated_steps"], [1])

    def test_a_human_only_command_without_a_rollback_is_a_finding(self):
        commands = list(operator_commands())
        commands[3] = dataclasses.replace(commands[3], rollback_command=None)
        review = self.review(tuple(commands))
        self.assertIn("HUMAN_ONLY_COMMAND_WITHOUT_ROLLBACK", " ".join(review.findings))

    def test_a_gap_in_the_sequence_is_a_finding(self):
        commands = list(operator_commands())
        commands[2] = dataclasses.replace(commands[2], step=99)
        review = self.review(tuple(commands))
        self.assertIn("COMMAND_SEQUENCE_NOT_CONTIGUOUS", " ".join(review.findings))

    def test_an_uncommanded_transaction_step_is_named(self):
        commands = [c for c in operator_commands() if c.step != 6]
        commands = [dataclasses.replace(c, step=i + 1)
                    for i, c in enumerate(commands)]
        review = self.review(tuple(commands))
        self.assertEqual(review.check.outcome, schemas.FAIL)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.uncovered_elements)
        self.assertIn("TRANSACTION_STEP_UNCOMMANDED", " ".join(review.findings))

    def test_an_unaddressed_autopilot_baseline_is_named(self):
        commands = [c for c in operator_commands() if c.step != 7]
        commands = [dataclasses.replace(c, step=i + 1)
                    for i, c in enumerate(commands)]
        review = self.review(tuple(commands))
        self.assertIn(f"autopilot_baseline:{AUTOPILOT_BASELINE}",
                      review.uncovered_elements)

    def test_a_command_outside_the_derived_transaction_is_a_finding(self):
        commands = list(operator_commands())
        commands[0] = dataclasses.replace(
            commands[0], target_paths=("/mnt/raid0/llm/whisper.cpp",))
        review = self.review(tuple(commands))
        self.assertIn("COMMAND_OUTSIDE_TRANSACTION_SCOPE", " ".join(review.findings))

    def test_a_broad_target_cannot_swallow_the_whole_surface(self):
        """Containment runs one way: `/` is not "inside" every declared element."""
        commands = list(operator_commands())
        commands[0] = dataclasses.replace(commands[0], target_paths=("/",))
        review = self.review(tuple(commands))
        self.assertIn("COMMAND_OUTSIDE_TRANSACTION_SCOPE", " ".join(review.findings))

    def test_a_sibling_tree_is_not_inside_the_production_tree(self):
        """`/mnt/raid0/llm/llama.cpp` is a string prefix of `…-experimental`."""
        commands = list(operator_commands())
        commands[0] = dataclasses.replace(
            commands[0], target_paths=("/mnt/raid0/llm/kernels/production-scratch",))
        review = self.review(tuple(commands))
        self.assertIn("COMMAND_OUTSIDE_TRANSACTION_SCOPE", " ".join(review.findings))

    def test_a_declared_child_of_a_declared_element_is_still_inside(self):
        """The guard must not forbid its own compliant idiom."""
        commands = list(operator_commands())
        commands[0] = dataclasses.replace(
            commands[0], target_paths=(f"{ARCHIVE_ROOT}/cpu/llama-server",))
        review = self.review(tuple(commands))
        self.assertNotIn("COMMAND_OUTSIDE_TRANSACTION_SCOPE", " ".join(review.findings))

    def test_an_empty_sequence_is_refused_outright(self):
        with self.assertRaises(packager.PackagerInputError):
            self.review(())


class TestCoverageRequiresAVerb(unittest.TestCase):
    """Coverage used to be `value in "\\n".join(every validated command's text)`, so
    a transaction element was "commanded" by any command that MENTIONED it —
    including in a shell comment, and including a mention in one step with the verb
    in a different step. `ELEMENT_VERBS` is what turns naming into acting."""

    def review(self, commands, **overrides):
        fields = {"transaction": transaction_plan(), "rollback": rollback_plan(),
                  "era_row": era_draft(),
                  "autopilot_baseline_path": AUTOPILOT_BASELINE}
        fields.update(overrides)
        return packager.validate_command_sequence(commands, **fields)

    def _with_step(self, index, **changes):
        commands = list(operator_commands())
        commands[index] = dataclasses.replace(commands[index], **changes)
        return tuple(commands)

    def test_every_element_kind_the_denominator_emits_has_a_vocabulary(self):
        """A kind with no vocabulary is not auto-covered — it is uncoverable. Both
        halves are wrong, so the two lists must not be allowed to drift."""
        elements = packager._transaction_elements(
            transaction_plan(), rollback_plan(), era_draft())
        kinds = {kind for kind, _value in elements} | {"autopilot_baseline"}
        self.assertEqual(sorted(kinds - set(packager.ELEMENT_VERBS)), [])

    def test_a_comment_mentioning_the_era_registry_does_not_cover_it(self):
        """The bite, in the README's own words: *"a comment mentioning
        instrument_eras.yaml covers the era-registry element."*"""
        commands = self._with_step(
            5, command=f"git status  # remember to hand-write {ERA_REGISTRY} after this",
            target_paths=())
        review = self.review(commands)
        self.assertEqual(review.check.outcome, schemas.FAIL)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.uncovered_elements)
        self.assertIn("naming a thing is not acting on it", " ".join(review.findings))

    def test_a_declared_target_with_no_verb_does_not_cover_it(self):
        """`target_paths` says what a command touches; it does not say it did."""
        commands = self._with_step(
            5, command=f"echo 'the operator will edit {ERA_REGISTRY} later'")
        review = self.review(commands)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.uncovered_elements)

    def test_a_name_in_one_step_and_a_verb_in_another_is_not_coverage(self):
        """Coverage is per COMMAND. Pooling the sequence's text let step 2 name the
        element and step 9 supply the verb, which is not a step that does the job."""
        commands = list(operator_commands())
        commands[5] = dataclasses.replace(
            commands[5], command=f"true {ERA_REGISTRY}", target_paths=(ERA_REGISTRY,))
        review = self.review(tuple(commands))
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.uncovered_elements)

    def test_an_unaddressed_autopilot_baseline_is_held_to_the_same_bar(self):
        commands = self._with_step(
            6, command=f"git status  # {AUTOPILOT_BASELINE} still needs opening",
            target_paths=())
        review = self.review(commands)
        self.assertIn(f"autopilot_baseline:{AUTOPILOT_BASELINE}",
                      review.uncovered_elements)

    # -- compliant-path controls ---------------------------------------------

    def test_the_reference_sequence_still_covers_everything(self):
        review = self.review(operator_commands())
        self.assertEqual(review.check.outcome, schemas.PASS, list(review.findings))
        self.assertEqual(review.uncovered_elements, ())

    def test_the_editor_idiom_is_a_verb_for_a_human_only_registry_write(self):
        """Control: the sanctioned way to write `instrument_eras.yaml` is to open it
        in an editor. A vocabulary that refused `$EDITOR` would forbid the only
        compliant way to perform the write it is checking for."""
        commands = self._with_step(
            5, command=f"$EDITOR {ERA_REGISTRY}")
        review = self.review(commands)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.covered_elements)

    def test_a_trailing_comment_on_a_real_command_is_still_coverage(self):
        """Control: comments are stripped, not banned. The reference sequence's own
        era step carries one, and it must go on covering."""
        commands = self._with_step(
            5, command=f"$EDITOR {ERA_REGISTRY}  # write the three drafted E9 rows")
        review = self.review(commands)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.covered_elements)

    def test_a_verifying_command_covers_without_mutating(self):
        """Control: the finding says "performs OR VERIFIES it". `sha256sum -c` over
        the archive is the correct command for the rollback anchor and touches
        nothing; a mutation-only vocabulary would have demanded a write."""
        review = self.review(operator_commands())
        self.assertIn(f"archive:{ARCHIVE_ROOT}", review.covered_elements)

    def test_a_kind_with_no_vocabulary_is_not_silently_covered(self):
        command = operator_commands()[0]
        self.assertFalse(packager._acts_on(command, "kind_nobody_declared",
                                           "production-consolidated-v9"))


# =============================================================================
# §11.3 — the cutover REQUEST
# =============================================================================

class TestCutoverRequest(unittest.TestCase):

    def test_the_request_is_a_bus_message_that_names_no_time(self):
        message = cutover_request().to_bus_message()
        self.assertEqual(message["schema_version"], packager.CUTOVER_MESSAGE_SCHEMA)
        self.assertEqual(message["kind"], "request")
        self.assertEqual(message["needs_routing_to"], ["inference"])
        self.assertIs(message["action_required"], True)
        self.assertEqual(message["payload"]["ask"], packager.CUTOVER_ASK)
        self.assertNotIn("scheduled_at", message["payload"])
        self.assertIn("at a moment it chooses", message["payload"]["scheduling_rule"])

    def test_routing_intent_is_structural_not_prose(self):
        """The 2026-07-29 bus lesson: prose routing intent gets truncated away."""
        message = cutover_request().to_bus_message()
        self.assertIn("needs_routing_to", message)
        self.assertIn("action_required", message)

    def test_an_addressee_less_request_is_refused(self):
        with self.assertRaises(packager.PackagerInputError):
            cutover_request(needs_routing_to=())

    def test_a_more_imperative_ask_is_refused(self):
        with self.assertRaises(packager.CutoverExecutionRefused) as caught:
            dataclasses.replace(cutover_request(), ask="restart_the_stack_now")
        self.assertIn("preemption", str(caught.exception))

    def test_the_record_says_it_was_not_sent(self):
        self.assertIs(cutover_request().to_dict()["sent"], False)

    def test_there_is_no_transport(self):
        with self.assertRaises(packager.CutoverExecutionRefused) as caught:
            packager.send_cutover_request(cutover_request())
        self.assertIn("ITS OWN outbox", str(caught.exception))

    def test_a_request_for_another_package_blocks(self):
        package = release_package(
            cutover_request=cutover_request(package_id="akr-somebody-else"))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("CUTOVER_REQUEST_FOREIGN", codes_of(package))

    def test_a_request_naming_a_different_version_blocks(self):
        package = release_package(cutover_request=cutover_request(
            transaction=transaction_plan(next_branch="production-consolidated-v11",
                                         next_version_number=11,
                                         next_tag="production-consolidated-v11")))
        self.assertIn("CUTOVER_REQUEST_VERSION_MISMATCH", codes_of(package))


# =============================================================================
# §11.5 — the post-cutover watch window
# =============================================================================

class TestWatchWindowDeclaration(unittest.TestCase):
    """Everything that must be fixed BEFORE the window opens."""

    def test_the_window_declares_all_six_signals_with_their_own_directions(self):
        window = watch_window()
        self.assertEqual(sorted(b.signal_id for b in window.bands),
                         sorted(packager.REQUIRED_WATCH_SIGNALS))
        for band in window.bands:
            self.assertEqual(band.alarm_rule,
                             packager.WATCH_SIGNAL_ALARM_RULES[band.signal_id])
            self.assertEqual(band.source,
                             packager.WATCH_SIGNAL_SOURCES[band.signal_id])

    def test_the_alarm_direction_is_a_property_of_the_signal_not_a_field(self):
        fields = {f.name for f in dataclasses.fields(packager.WatchSignalBand)}
        self.assertNotIn("alarm_rule", fields)
        self.assertNotIn("source", fields)

    def test_a_missing_signal_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            watch_window(bands=tuple(b for b in watch_bands()
                                     if b.signal_id != packager.SIGNAL_QUALITY))
        self.assertIn("six rows", str(caught.exception))

    def test_a_band_with_no_edge_on_its_alarm_side_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            packager.WatchSignalBand(
                signal_id=packager.SIGNAL_THROUGHPUT, unit="tokens_per_s",
                basis_ref="data/x", noise_reference_ref="data/y", upper=99.0)
        self.assertIn("lower edge", str(caught.exception))
        with self.assertRaises(packager.PackagerInputError):
            packager.WatchSignalBand(
                signal_id=packager.SIGNAL_SUPERVISOR, unit="events",
                basis_ref="data/x", noise_reference_ref="data/y", lower=0.0)

    def test_a_band_with_no_incumbent_era_basis_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            watch_bands(basis_ref_by_signal={packager.SIGNAL_THROUGHPUT: "data/x"})
        self.assertIn("incumbent-era basis", str(caught.exception))

    def test_bands_fixed_after_the_window_opens_are_refused(self):
        with self.assertRaises(packager.BandsNotFixedBeforeData) as caught:
            watch_window(bands_fixed_at="2026-08-05T00:00:00Z", opens_at=NOW)
        self.assertIn("before the data is seen", str(caught.exception))

    def test_no_edge_is_invented_when_one_is_not_declared(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            watch_bands(edges={packager.SIGNAL_QUALITY: None})
        self.assertIn("no edge spec", str(caught.exception))

    def test_the_duration_rule_is_the_later_of_days_and_volume(self):
        window = watch_window()
        self.assertEqual(window.min_duration_days, packager.DEFAULT_WATCH_WINDOW_DAYS)
        self.assertEqual(packager.DEFAULT_WATCH_WINDOW_DAYS, 7)
        self.assertIn("later_of", window.to_dict()["duration_rule"])

    def test_a_role_with_no_declared_minimum_volume_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            watch_window(min_volume_by_role={"worker_general": 20000})
        self.assertIn("quiet weekend", str(caught.exception))

    def test_the_comparison_must_be_era_labelled(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            watch_window(incumbent_era="E8", candidate_era="E8")
        self.assertIn("233", str(caught.exception))

    def test_the_owner_is_human_and_the_close_step_demands_a_verdict(self):
        window = watch_window()
        self.assertEqual(window.close_step.action, "close_with_verdict")
        self.assertIs(window.close_step.verdict_required, True)
        self.assertEqual(window.close_step.unclosed_state, "OPEN_QUESTION")
        with self.assertRaises(packager.PackagerInputError):
            watch_window(owner="autokernel")
        with self.assertRaises(packager.PackagerInputError):
            packager.WatchWindowCloseStep(owner="operator", verdict_required=False)

    def test_the_window_declares_itself_a_recommendation_not_a_claim(self):
        record = watch_window().to_dict()
        self.assertIn("NOT A CLAIM", record["output_class"])
        self.assertEqual(schemas.find_authority_flavoured_keys(record), [])


class TestWatchWindowEvaluation(unittest.TestCase):

    def test_a_clean_closed_window_recommends_closing(self):
        recommendation = packager.evaluate_watch_window(watch_window(), watch_progress())
        self.assertEqual(recommendation.recommendation, packager.WATCH_CLOSE_NO_REGRESSION)
        self.assertEqual(recommendation.state, packager.WATCH_STATE_CLOSEABLE)

    def test_duration_alone_does_not_close_the_window(self):
        recommendation = packager.evaluate_watch_window(
            watch_window(), watch_progress(volumes={"worker_general": 100,
                                                    "frontdoor": 10}))
        self.assertTrue(recommendation.duration_met)
        self.assertFalse(recommendation.volume_met)
        self.assertEqual(recommendation.state, packager.WATCH_STATE_OPEN)
        self.assertEqual(recommendation.recommendation, packager.WATCH_CONTINUE)

    def test_volume_alone_does_not_close_the_window(self):
        recommendation = packager.evaluate_watch_window(
            watch_window(), watch_progress(days=2))
        self.assertFalse(recommendation.duration_met)
        self.assertTrue(recommendation.volume_met)
        self.assertEqual(recommendation.state, packager.WATCH_STATE_OPEN)

    def test_a_signal_outside_its_band_raises_a_decision_package(self):
        recommendation = packager.evaluate_watch_window(
            watch_window(), watch_progress(values={packager.SIGNAL_THROUGHPUT: 41.0}))
        self.assertEqual(recommendation.recommendation,
                         packager.WATCH_RAISE_DECISION_PACKAGE)
        self.assertEqual(recommendation.alarms, (packager.SIGNAL_THROUGHPUT,))

    def test_the_recommendation_vocabulary_can_never_revert_anything(self):
        joined = " ".join(packager.WATCH_RECOMMENDATIONS)
        for verb in ("revert", "rollback", "roll_back", "restart", "cutover", "freeze"):
            self.assertNotIn(verb, joined)

    def test_an_unobserved_signal_is_incomplete_evidence_not_a_pass(self):
        recommendation = packager.evaluate_watch_window(
            watch_window(), watch_progress(omit=(packager.SIGNAL_SUPERVISOR,)))
        self.assertEqual(recommendation.recommendation, packager.WATCH_INCOMPLETE_EVIDENCE)
        self.assertEqual(recommendation.unevaluable, (packager.SIGNAL_SUPERVISOR,))

    def test_an_observation_from_the_wrong_era_does_not_count(self):
        recommendation = packager.evaluate_watch_window(
            watch_window(), watch_progress(era="E8"))
        self.assertEqual(recommendation.recommendation, packager.WATCH_INCOMPLETE_EVIDENCE)
        self.assertEqual(len(recommendation.unevaluable),
                         len(packager.REQUIRED_WATCH_SIGNALS))

    def test_evaluating_against_moved_bands_is_refused(self):
        with self.assertRaises(packager.BandsNotFixedBeforeData) as caught:
            packager.evaluate_watch_window(
                watch_window(),
                watch_progress(bands_sha256=digest("some-other-bands")))
        self.assertIn("chosen after seeing the data", str(caught.exception))

    def test_the_digest_actually_changes_when_a_band_moves(self):
        """A digest that ignores the numbers would make the previous test vacuous."""
        loosened = watch_bands(edges={packager.SIGNAL_THROUGHPUT: {
            "unit": "tokens_per_s", "lower": 1.0}})
        self.assertNotEqual(watch_window().bands_sha256(),
                            watch_window(bands=loosened).bands_sha256())

    def test_closing_an_open_window_is_refused(self):
        with self.assertRaises(packager.WatchWindowOpen) as caught:
            packager.close_watch_window(
                watch_window(), watch_progress(days=2), verdict="no_regression_observed",
                closed_by="daniele", closed_at="2026-08-05T13:00:00Z")
        self.assertIn("has not met its close condition", str(caught.exception))

    def test_closing_records_a_verdict_and_the_human_who_recorded_it(self):
        closure = packager.close_watch_window(
            watch_window(), watch_progress(), verdict="no_regression_observed",
            closed_by="daniele", closed_at="2026-08-11T13:00:00Z")
        record = closure.to_dict()
        self.assertEqual(record["state"], packager.WATCH_STATE_CLOSED)
        self.assertEqual(record["verdict"], "no_regression_observed")
        self.assertEqual(record["closed_by"], "daniele")
        self.assertIn("NOT A CLAIM", record["record_class"])

    def test_a_machine_may_not_close_the_window(self):
        with self.assertRaises(packager.PackagerInputError):
            packager.close_watch_window(
                watch_window(), watch_progress(), verdict="no_regression_observed",
                closed_by="autokernel", closed_at="2026-08-11T13:00:00Z")

    def test_no_regression_cannot_be_recorded_over_unevaluated_signals(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            packager.close_watch_window(
                watch_window(), watch_progress(omit=(packager.SIGNAL_QUALITY,)),
                verdict="no_regression_observed", closed_by="daniele",
                closed_at="2026-08-11T13:00:00Z")
        self.assertIn("did not look", str(caught.exception))

    def test_inconclusive_is_a_first_class_close_verdict(self):
        closure = packager.close_watch_window(
            watch_window(), watch_progress(omit=(packager.SIGNAL_QUALITY,)),
            verdict="inconclusive", closed_by="daniele",
            closed_at="2026-08-11T13:00:00Z")
        self.assertEqual(closure.verdict, "inconclusive")


# =============================================================================
# The four-part decision package (OPERATING_CONSTRAINTS.md:69-78)
# =============================================================================

class TestDecisionPackage(unittest.TestCase):

    def test_a_ready_package_offers_execute_defer_decline(self):
        decision = release_package().decision_package
        self.assertEqual([o.option_id for o in decision.ordered_options],
                         ["execute-in-window", "defer-to-a-later-window", "decline"])
        self.assertEqual(decision.recommendation.option_id, "execute-in-window")

    def test_every_option_carries_all_five_tradeoff_axes(self):
        for option in release_package().decision_package.options:
            with self.subTest(option=option.option_id):
                self.assertEqual(sorted(option.tradeoffs),
                                 sorted(packager.REQUIRED_TRADEOFF_AXES))

    def test_a_missing_tradeoff_axis_is_refused(self):
        option = release_package().decision_package.options[0]
        thin = {k: v for k, v in option.tradeoffs.items() if k != "reversibility"}
        with self.assertRaises(packager.PackagerInputError) as caught:
            dataclasses.replace(option, tradeoffs=thin)
        self.assertIn("reversibility", str(caught.exception))

    def test_no_option_is_something_autokernel_performs(self):
        for option in release_package().decision_package.options:
            with self.subTest(option=option.option_id):
                self.assertNotIn("autokernel will", option.entails.lower())
                self.assertIn(option.option_id,
                              ("execute-in-window", "defer-to-a-later-window", "decline"))

    def test_a_blocked_package_recommends_not_freezing(self):
        package = release_package(
            rollback=rollback_plan(archive=incumbent_archive(rebuilt=True)))
        decision = package.decision_package
        self.assertEqual(decision.recommendation.option_id, "fix-the-blockers")
        self.assertIn("invariant 6", decision.recommendation.why.lower())

    def test_a_waiver_option_appears_only_when_the_blockers_are_cell_scoped(self):
        """A waiver covers cells, never the integrity spine (§10.4)."""
        failing = list(cell_results(matrix_cells()))
        failing[0] = dataclasses.replace(
            failing[0], check=schemas.Check(schemas.FAIL, ("decode regressed 9%",)))
        cell_scoped = release_package(evaluation=evaluation(
            request=t3_request(_results=failing)))
        self.assertIn("scoped-operator-waiver",
                      [o.option_id for o in cell_scoped.decision_package.options])

        spine = release_package(
            rollback=rollback_plan(archive=incumbent_archive(rebuilt=True)),
            evaluation=evaluation(request=t3_request(
                linkage_receipts=(linkage_receipt("llama_cpu"),))))
        self.assertNotIn("scoped-operator-waiver",
                         [o.option_id for o in spine.decision_package.options])

    def test_a_quoted_waiver_is_reported_as_unread_not_merely_unverified(self):
        """"T3 refused it" and "nobody read it" are different states, and they
        collapsed into one finding, so the package could not tell an operator which
        had happened. `t3.WaiverBinding` carries a document, a path and a digest that
        are three independent assertions by the party being gated.
        """
        binding = unverified_waiver_binding()
        self.assertFalse(binding.was_read)
        package = release_package(waivers=(binding,))
        codes = codes_of(package)
        self.assertIn("WAIVER_PINNED_UNREAD", codes)
        self.assertIn("WAIVER_PINNED_BUT_UNVERIFIED", codes)
        unread = next(f for f in package.findings
                      if f.code == "WAIVER_PINNED_UNREAD")
        self.assertEqual(unread.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("QUOTATION", unread.detail)
        self.assertNotEqual(package.state, packager.STATE_READY)

    def test_a_binding_that_merely_declares_itself_read_is_still_unread(self):
        """The package is the durable record — the one place a lie outlives the run.

        `was_read` is a PROPERTY, and a property is something a caller overrides:

            class Liar(t3.WaiverBinding):
                @property
                def was_read(self): return True

        Measured, that object was written into the package as `read: True` and
        skipped `WAIVER_PINNED_UNREAD` entirely, so the durable record asserted that
        somebody opened a file nobody opened. The packager now asks
        `t3.waiver_read_violations`, which inspects the reader's mint token, and a
        property override cannot reach it.
        """
        class Liar(t3.WaiverBinding):
            @property
            def was_read(self):
                return True

        honest = unverified_waiver_binding()
        liar = Liar(waiver_id=honest.waiver_id, pinned_sha256=honest.pinned_sha256,
                    document=honest.document, document_path=honest.document_path,
                    covers_cell_ids=honest.covers_cell_ids,
                    observed_sha256=honest.observed_sha256)
        self.assertTrue(liar.was_read)  # the property it defeats
        package = release_package(waivers=(liar,))
        self.assertIn("WAIVER_PINNED_UNREAD", codes_of(package))
        self.assertIs(package.to_dict()["waiver_bindings"][0]["read"], False)
        self.assertNotEqual(package.state, packager.STATE_READY)

    def test_the_binding_record_says_whether_the_document_was_read(self):
        binding = unverified_waiver_binding()
        self.assertIs(binding.to_dict()["read"], False)
        self.assertIsNone(binding.to_dict()["read_receipt"])
        package = release_package(waivers=(binding,))
        self.assertIs(package.to_dict()["waiver_bindings"][0]["read"], False)

    def test_an_incomplete_package_recommends_supplying_the_evidence(self):
        package = release_package(waivers=(unverified_waiver_binding(),))
        self.assertEqual(package.state, packager.STATE_INCOMPLETE)
        self.assertIn("WAIVER_PINNED_BUT_UNVERIFIED", codes_of(package))
        decision = package.decision_package
        self.assertEqual(decision.recommendation.option_id,
                         "supply-the-missing-evidence")
        self.assertIn("proceed-on-a-declared-forfeit",
                      [o.option_id for o in decision.options])

    def test_an_open_ended_question_is_refused(self):
        decision = release_package().decision_package
        with self.assertRaises(packager.PackagerInputError) as caught:
            dataclasses.replace(decision, context="How should I proceed?")
        self.assertIn("open-ended question", str(caught.exception))
        with self.assertRaises(packager.PackagerInputError):
            dataclasses.replace(decision, default_outcome="What do you want to do?")

    def test_the_option_count_is_bounded_at_two_to_four(self):
        decision = release_package().decision_package
        with self.assertRaises(packager.PackagerInputError):
            dataclasses.replace(decision, options=decision.options[:1])
        with self.assertRaises(packager.PackagerInputError):
            dataclasses.replace(
                decision,
                options=decision.options + tuple(
                    dataclasses.replace(decision.options[2], option_id=f"extra-{i}")
                    for i in range(2)))

    def test_the_recommendation_must_name_an_option(self):
        decision = release_package().decision_package
        with self.assertRaises(packager.PackagerInputError):
            dataclasses.replace(decision, recommendation=packager.DecisionRecommendation(
                option_id="not-an-option", why="because"))

    def test_the_rendered_page_leads_with_the_recommended_option(self):
        page = packager.render_first_page(release_package())
        self.assertIn("### Option A — ", page)
        self.assertIn("**(Recommended)**", page)
        recommended_index = page.index("**(Recommended)**")
        self.assertLess(recommended_index, page.index("### Option B"))

    def test_the_rendered_page_has_all_four_parts_and_the_notice(self):
        page = packager.render_first_page(release_package())
        for heading in ("## 1. Context", "## 2. Options", "## 3. Recommendation",
                        "## 4. Default"):
            self.assertIn(heading, page)
        self.assertIn(packager.PACKAGE_NOTICE, page)

    def test_the_human_review_marker_is_on_the_first_page(self):
        package = release_package(change_classes=("core_header",))
        self.assertTrue(package.requires_human_code_review)
        page = packager.render_first_page(package)
        head = "\n".join(page.splitlines()[:4])
        self.assertIn(integrity.REQUIRES_HUMAN_CODE_REVIEW, head)
        self.assertIn("core_header", head)

    def test_the_review_marker_is_derived_from_three_independent_sources(self):
        self.assertFalse(release_package().requires_human_code_review)
        by_core_class = release_package(change_classes=("core_header",))
        by_shared_core = release_package(diff_complexity={
            "diff_size": 2, "files_touched": 1, "touches_shared_core": True})
        by_t3 = release_package(evaluation=evaluation(request=t3_request(complexity={
            "llama_cpu": integrity.ComplexityAssessment(
                requires_human_code_review=True,
                reasons=("changed lines 250000 exceeds the declared ceiling",),
                first_page_notice="REQUIRES_HUMAN_CODE_REVIEW — too large",
                measured={"total_changed_lines": 250000}),
            "llama_gpu": integrity.ComplexityAssessment(
                requires_human_code_review=False, reasons=(), first_page_notice=None,
                measured={}),
        })))
        for package in (by_core_class, by_shared_core, by_t3):
            self.assertTrue(package.requires_human_code_review)
            self.assertIsNotNone(package.first_page_notice)


# =============================================================================
# The package: derived state, cross-checks, and the schema contract
# =============================================================================

class TestReleasePackage(unittest.TestCase):

    def test_a_complete_package_is_ready_with_no_findings(self):
        package = release_package()
        self.assertEqual(package.state, packager.STATE_READY)
        self.assertEqual(package.findings, ())
        self.assertEqual(package.blocking_findings, ())

    def test_a_ready_package_satisfies_the_release_package_schema(self):
        self.assertEqual(release_package().schema_violations(), [])

    def test_the_terminal_success_state_is_release_package_ready(self):
        """§3.3: the terminal success state is RELEASE_PACKAGE_READY, not FREEZE_ELIGIBLE."""
        record = release_package().to_dict()
        self.assertEqual(record["terminal_success_state"], "RELEASE_PACKAGE_READY")
        self.assertEqual(record["executed_by"], "operator")
        self.assertNotIn("FREEZE_ELIGIBLE", packager.PACKAGE_STATES)

    def test_the_state_cannot_be_stamped_over_its_own_findings(self):
        package = release_package()
        blocked = packager.PackageFinding(
            code="X", detail="something failed", outcome=schemas.FAIL)
        with self.assertRaises(packager.StateNotDerived) as caught:
            dataclasses.replace(package, findings=(blocked,))
        self.assertIn("its own findings yield", str(caught.exception))

    def test_the_state_cannot_be_stamped_in_the_optimistic_direction_either(self):
        package = release_package(
            rollback=rollback_plan(archive=incumbent_archive(rebuilt=True)))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        with self.assertRaises(packager.StateNotDerived):
            dataclasses.replace(package, state=packager.STATE_READY)

    def test_the_three_states_are_the_three_outcomes(self):
        self.assertEqual(len(packager.PACKAGE_STATES), 3)
        self.assertIn(packager.STATE_INCOMPLETE, packager.PACKAGE_STATES)

    def test_a_t3_fail_blocks_the_package(self):
        failing = list(cell_results(matrix_cells()))
        failing[0] = dataclasses.replace(
            failing[0], check=schemas.Check(schemas.FAIL, ("prefill regressed 12%",)))
        package = release_package(
            evaluation=evaluation(request=t3_request(_results=failing)))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("T3_VERDICT_FAIL", codes_of(package))

    def test_a_tree_mismatch_between_request_and_seal_blocks(self):
        package = release_package(freeze_request=freeze_request(source_tree="whisper.cpp"))
        self.assertIn("FREEZE_REQUEST_TREE_MISMATCH", codes_of(package))

    def test_a_package_over_another_campaigns_evidence_blocks(self):
        package = release_package(
            freeze_request=freeze_request(campaign_id="ak-some-other-campaign"))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("CAMPAIGN_MISMATCH", codes_of(package))

    def test_a_foreign_watch_window_blocks(self):
        package = release_package(watch_window=watch_window(package_id="akr-elsewhere"))
        self.assertIn("WATCH_WINDOW_FOREIGN", codes_of(package))

    def test_a_rollback_to_something_other_than_the_incumbent_blocks(self):
        package = release_package(
            version=next_version(incumbent_branch="production-consolidated-v7",
                                 existing_branches=("production-consolidated-v7",)))
        codes = codes_of(package)
        self.assertIn("ROLLBACK_NOT_THE_INCUMBENT", codes)

    def test_the_release_plan_block_is_the_plan_t3_actually_graded(self):
        package = release_package()
        graded = package.evaluation.result.bundle.payload["release_plan"]
        self.assertEqual(schemas.content_hash(dict(package.release_plan)),
                         schemas.content_hash(dict(graded)))

    def test_a_supplied_plan_that_is_not_the_graded_plan_blocks(self):
        package = release_package(release_plan={"plan_id": "some-other-plan"})
        self.assertIn("RELEASE_PLAN_NOT_THE_GRADED_PLAN", codes_of(package))

    def test_linkage_is_derived_from_t3s_own_phase(self):
        package = release_package()
        self.assertEqual(package.linkage.status, schemas.PASS)
        self.assertEqual(sorted(package.linkage.per_backend), sorted(LLAMA_BACKENDS))
        self.assertIn(t3.LINKAGE_VERIFIER_RELPATH, package.linkage.receipt)

    def test_a_failed_linkage_blocks_the_package(self):
        broken = linkage_receipt(
            "llama_cpu",
            ld_library_path=("/mnt/raid0/llm/qwentts.cpp/build/bin",),
            stdout="BAD  libggml.so.0 -> /mnt/raid0/llm/qwentts.cpp/build/bin/libggml.so.0\n",
            exit_code=1)
        package = release_package(evaluation=evaluation(request=t3_request(
            linkage_receipts=(broken, linkage_receipt("llama_gpu")))))
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("LINKAGE_NOT_PROVEN", codes_of(package))

    def test_a_package_id_without_the_akr_prefix_is_refused(self):
        with self.assertRaises(packager.PackagerInputError):
            release_package(package_id="release-v9")

    def test_the_package_record_hashes_stably(self):
        self.assertEqual(release_package().sha256(), release_package().sha256())

    def test_the_record_names_the_packager_that_built_it(self):
        record = release_package().to_dict()
        self.assertEqual(record["packaged_by"], packager.MODULE_ID)
        self.assertEqual(record["schema"], schemas.SCHEMA_RELEASE_PACKAGE)
        self.assertIn("handoff", record["record_class"].lower())


class TestWaiversInThePackage(unittest.TestCase):
    """§10.4: pinned here, verified there. The packager never grants one."""

    def failing_request(self, **overrides):
        failing = list(cell_results(matrix_cells()))
        failing[0] = dataclasses.replace(
            failing[0], check=schemas.Check(schemas.FAIL, ("prefill regressed",)))
        return t3_request(_results=failing, **overrides)

    def test_a_pinned_but_unverified_waiver_is_could_not_check_not_active(self):
        package = release_package(waivers=(unverified_waiver_binding(),))
        self.assertEqual(package.state, packager.STATE_INCOMPLETE)
        self.assertEqual(package.to_dict()["active_waivers"], [])
        self.assertEqual(len(package.to_dict()["waiver_bindings"]), 1)

    def test_a_pass_package_pins_no_active_waiver(self):
        """`schemas` refuses a PASS package that pins waivers; the record must agree."""
        package = release_package()
        self.assertEqual(package.evaluation.verdict, "PASS")
        self.assertEqual(package.to_dict()["active_waivers"], [])
        self.assertEqual(package.schema_violations(), [])

    def test_the_packager_cannot_grant_a_waiver(self):
        with self.assertRaises(packager.ProductionWriteRefused) as caught:
            packager.waive_failed_evidence(cell_id="llama_cpu.prefill")
        self.assertIn("human-authored", str(caught.exception))


# =============================================================================
# End to end: one campaign, one package, zero production writes
# =============================================================================

class TestEndToEnd(unittest.TestCase):
    """§14 AK6 exit: *"campaigns produce correct, idempotent, operator-executable
    release packages and never write production."*"""

    def build(self):
        request = t3_request()
        sealed = packager.seal_champion(
            champion_id="akch-llama-v9", candidate=request.sealed,
            build_receipt_sha256=digest("build-receipt"),
            seal_inputs_ref="data/ak-v9/seal-inputs.json", sealed_at=NOW,
            pinned_evaluator_bundle_sha256=EVALUATOR_BUNDLE,
            incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD)
        result = packager.run_release_evaluation(request, evaluator=t3.T3Runner())
        version = packager.compute_next_version(
            incumbent_branch="production-consolidated-v8",
            existing_branches=("production-consolidated-v7", "production-consolidated-v8"))
        rollback = rollback_plan()
        era = packager.draft_era_registry_row(
            rows=era_rows(), version=version, registry_path=ERA_REGISTRY,
            incumbent_era="E8", drafted_at=NOW)
        transaction = packager.build_transaction_plan(
            version=version, install_path=INSTALL_PATH,
            stable_path_moves=(
                (CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin",
                 "/mnt/raid0/llm/llama.cpp-v9/build/bin"),
                (GPU_LINK, "/mnt/raid0/llm/llama.cpp/build-hip/bin",
                 "/mnt/raid0/llm/llama.cpp-v9/build-hip/bin")),
            service_impact=("llama-server restart at the inference owner's boundary",),
            era_actions=({"draft": True, "action": "write_era_registry_row",
                          "registry_path": ERA_REGISTRY},),
            receipt_paths=("artifacts/operator/v9-freeze/",), rollback=rollback)
        return packager.assemble_release_package(
            package_id="akr-v9-001", created_at=NOW, freeze_request=freeze_request(),
            sealed=sealed, evaluation=result, version=version, transaction=transaction,
            rollback=rollback, era_row_draft=era, rebaseline_note=rebaseline_note(),
            commands=operator_commands(), watch_window=watch_window(),
            cutover_request=packager.build_cutover_request(
                message_id="msg-20260803T120000Z-001-autokernel", from_agent="autokernel",
                to_agent="coordinator-agent", needs_routing_to=("inference",),
                task_id="ak-v9-cutover", created_at=NOW, package_id="akr-v9-001",
                transaction=transaction, rollback=rollback),
            autopilot_baseline_path=AUTOPILOT_BASELINE, change_classes=("arithmetic",),
            diff_complexity=dict(DIFF_COMPLEXITY))

    def test_the_whole_path_ends_at_a_ready_operator_executable_package(self):
        package = self.build()
        self.assertEqual(package.state, packager.STATE_READY)
        self.assertEqual(package.schema_violations(), [])
        record = package.to_dict()
        self.assertEqual(len(record["operator_command_sequence"]), 9)
        for entry in record["operator_command_sequence"]:
            self.assertIs(entry["validated"], True)
            self.assertEqual(entry["executed_by"], "operator")

    def test_the_package_is_idempotent_over_the_same_material(self):
        self.assertEqual(self.build().sha256(), self.build().sha256())

    def test_the_package_contains_every_part_section_7_6_names(self):
        record = self.build().to_dict()
        for block in ("sealed_candidate", "t3_verdict", "active_waivers", "release_plan",
                      "transaction_plan", "rollback_plan", "draft_era_registry_row",
                      "draft_autopilot_rebaseline_note", "linkage_verification",
                      "operator_command_sequence", "watch_window", "cutover_request",
                      "decision_package"):
            self.assertIn(block, record)

    def test_the_package_carries_no_production_write_and_no_authority_claim(self):
        record = self.build().to_dict()
        self.assertEqual(schemas.find_authority_flavoured_keys(record), [])
        self.assertIs(record["transaction_plan"]["executed"], False)
        self.assertIs(record["cutover_request"]["sent"], False)
        self.assertIs(record["draft_era_registry_row"]["draft"], True)

    def test_the_first_page_is_renderable_and_names_the_next_version(self):
        page = packager.render_first_page(self.build())
        self.assertIn("production-consolidated-v8 → production-consolidated-v9", page)
        self.assertIn("## 4. Default", page)


# =============================================================================
# Independent red team, 2026-08-03 — an author's self-mutation harness tests the
# guarantees the author thought of. Every case below PASSED before its fix: the
# module's own audits certified source that could write, alias a clock, mint a
# freeze request or replace a refusal door, and the assembly certified packages
# whose evidence did not support them. Each test is paired with a COMPLIANT-PATH
# control, because the recurring defect in this package is a guard that closes a
# hole by forbidding its own legitimate idiom.
# =============================================================================

class TestAuditsCannotBeWalkedAround(unittest.TestCase):
    """The four self-audits are the guarantee. They are what gets attacked first."""

    def bound(self, body: str) -> str:
        """Doctored source that BINDS to this module, so a clean result is a PASS."""
        return f'MODULE_ID = "{packager.MODULE_ID}"\n{body}'

    # -- the write audit ------------------------------------------------------

    def test_the_write_audit_bites_on_a_write_verb_bound_to_a_name(self):
        """`sink = Path(p).write_text` then `sink(x)` puts nothing in call position."""
        for snippet in ("sink = Path(p).write_text\nsink('x')\n",
                        "w = open\nw('/etc/passwd', 'w')\n",
                        "mover = Path(new).replace\nmover(link)\n"):
            with self.subTest(snippet=snippet):
                check = packager.audit_no_write_or_process_paths(self.bound(snippet))
                self.assertEqual(check.outcome, schemas.FAIL, list(check.reasons))

    def test_the_write_audit_bites_on_dispatch_it_cannot_read(self):
        """`builtins.__dict__['open'](p, 'w')` is `open(p, 'w')` with punctuation."""
        check = packager.audit_no_write_or_process_paths(
            self.bound("import builtins\nbuiltins.__dict__['open']('/x', 'w')\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("Subscript expression", " ".join(check.reasons))

    def test_the_write_audit_bites_on_a_getattr_name_it_cannot_resolve(self):
        check = packager.audit_no_write_or_process_paths(
            self.bound("verb = 'sys' + 'tem'\ngetattr(module, verb)('rm -rf /')\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("cannot resolve to constants", " ".join(check.reasons))

    def test_the_write_audit_reads_a_denied_verb_out_of_a_literal_loop(self):
        check = packager.audit_no_write_or_process_paths(
            self.bound('for name in ("unlink", "tree_clean"):\n    getattr(c, name)()\n'))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("'unlink'", " ".join(check.reasons))

    def test_the_compliant_literal_field_loop_is_not_forbidden(self):
        """CONTROL. This module reads its own fields exactly this way, twice."""
        check = packager.audit_no_write_or_process_paths(self.bound(
            'for name in ("overlay_present", "tree_clean", "ancestry_clean"):\n'
            '    value = getattr(candidate, name)\n'))
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    # -- the clock / self-trigger audit ---------------------------------------

    def test_the_clock_audit_bites_on_a_clock_bound_to_a_name(self):
        check = packager.audit_no_clock_or_self_trigger(
            self.bound("clock = datetime.now\nmoment = clock()\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("clock-bearing receiver", " ".join(check.reasons))

    def test_a_field_called_now_is_not_a_clock(self):
        """CONTROL. `WatchWindowProgress.now` is data; `progress.now` must stay legal."""
        check = packager.audit_no_clock_or_self_trigger(self.bound(
            "moment = _timestamp(progress.now, 'progress.now')\nlabel = self.now\n"))
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    def test_the_self_trigger_audit_bites_on_an_aliased_mint(self):
        check = packager.audit_no_clock_or_self_trigger(
            self.bound("mint = OperatorFreezeRequest\nrequest = mint(request_id='akfr-x')\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("binds OperatorFreezeRequest", " ".join(check.reasons))

    def test_naming_the_request_class_in_a_type_check_is_not_a_mint(self):
        """CONTROL. `assemble_release_package` names the class in its isinstance table."""
        check = packager.audit_no_clock_or_self_trigger(self.bound(
            'for label, value, klass in (("freeze_request", request, '
            'OperatorFreezeRequest),):\n    isinstance(value, klass)\n'))
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    # -- the delegation audit -------------------------------------------------

    def test_the_delegation_audit_bites_on_an_aliased_gate(self):
        check = packager.audit_verdict_is_delegated(
            self.bound("grade = t3.run_t3\nresult = grade(request)\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("binds run_t3", " ".join(check.reasons))

    def test_going_through_the_seam_is_still_allowed(self):
        """CONTROL. The whole point is that `evaluate_release` remains reachable."""
        check = packager.audit_verdict_is_delegated(self.bound(
            'evaluate = getattr(evaluator, "evaluate_release", None)\n'
            "result = evaluate(request)\n"))
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    # -- the refusal doors ----------------------------------------------------

    def doors(self) -> str:
        return "\n".join(f"def {name}(*a, **k):\n    raise RuntimeError('no')\n"
                         for name in packager.REFUSED_CAPABILITIES.values())

    def test_the_door_audit_bites_on_a_door_rebound_after_it_was_defined(self):
        """The AST is full of compliant raises and the module exports a no-op."""
        check = packager.audit_refusal_doors_raise_unconditionally(
            self.bound(self.doors() + "\nexecute_freeze = lambda *a, **k: None\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("rebinds execute_freeze()", " ".join(check.reasons))

    def test_the_door_audit_bites_on_a_door_imported_over(self):
        check = packager.audit_refusal_doors_raise_unconditionally(
            self.bound(self.doors() + "\nfrom somewhere import helper as schedule_cutover\n"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("rebinds schedule_cutover()", " ".join(check.reasons))

    def test_a_door_that_binds_no_module_attribute_does_not_count(self):
        """A nested `def` satisfies a walk and leaves `packager.execute_freeze` absent."""
        nested = "def _holder():\n" + "\n".join(
            f"    def {name}(*a, **k):\n        raise RuntimeError('no')\n"
            for name in packager.REFUSED_CAPABILITIES.values())
        check = packager.audit_refusal_doors_raise_unconditionally(self.bound(nested))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("no function named", " ".join(check.reasons))

    def test_the_twelve_real_doors_still_pass(self):
        """CONTROL, and the reason the three tests above are not just strictness."""
        check = packager.audit_refusal_doors_raise_unconditionally(
            self.bound(self.doors()))
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))
        for name in packager.REFUSED_CAPABILITIES.values():
            self.assertTrue(callable(getattr(packager, name)), name)


class TestPackageEvidenceCannotBeDeclaredAway(unittest.TestCase):
    """Each of these reached a package state its own evidence did not support."""

    def test_a_verdict_for_a_different_seal_blocks_the_package(self):
        foreign = sealed_release(candidate=sealed_candidate(
            candidate_id="akc-something-else", seal_sha256=digest("another-seal")))
        package = release_package(sealed=foreign)
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("SEALED_CANDIDATE_NOT_THE_GRADED_ONE", codes_of(package))

    def test_the_seal_the_evaluator_graded_is_still_accepted(self):
        """CONTROL. The cross-check must not refuse the matching pair."""
        self.assertEqual(release_package().state, packager.STATE_READY)

    def test_era_kinds_are_traced_from_the_rows_not_read_off_a_summary_key(self):
        declared_only = {"draft": True, "written_by": packager.OPERATOR_AUTHORITY,
                         "registry_path": ERA_REGISTRY,
                         "kinds_present": list(packager.ERA_ROW_KINDS), "rows": []}
        package = release_package(era_row_draft=declared_only)
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("ERA_ROW_KINDS_DECLARED_NOT_DRAFTED", codes_of(package))
        self.assertIn("ERA_ROW_KIND_MISSING", codes_of(package))

    def test_a_real_three_row_draft_is_still_complete(self):
        """CONTROL. `draft_era_registry_row()` output must remain sufficient."""
        self.assertNotIn("ERA_ROW_KINDS_DECLARED_NOT_DRAFTED",
                         codes_of(release_package()))

    def test_a_duplicate_binding_cannot_hide_a_self_granted_waiver(self):
        machine = dict(unverified_waiver_binding().document)
        machine["authorized_by"] = "autokernel-controller"
        shadowed = unverified_waiver_binding(document=machine,
                                             pinned_sha256=digest("self-granted"))
        clean = unverified_waiver_binding()
        for order in ((shadowed, clean), (clean, shadowed)):
            with self.subTest(order=[b.pinned_sha256[:8] for b in order]):
                package = release_package(waivers=order)
                self.assertEqual(package.state, packager.STATE_BLOCKED)
                self.assertIn("WAIVER_SELF_GRANTED", codes_of(package))
                self.assertIn("WAIVER_BINDING_DUPLICATE", codes_of(package))

    def test_two_different_waivers_are_not_a_duplicate(self):
        """CONTROL. Pinning two distinct waivers is the normal §10.4 case."""
        second = unverified_waiver_binding(
            waiver_id="WAIVE-STT-v9", pinned_sha256=digest("waive-stt-v9"),
            covers_cell_ids=("llama_gpu.decode",))
        package = release_package(waivers=(unverified_waiver_binding(), second))
        self.assertNotIn("WAIVER_BINDING_DUPLICATE", codes_of(package))

    def test_an_unstated_rollback_anchor_liveness_is_not_a_yes(self):
        package = release_package(rollback=rollback_plan(anchor_live=None))
        self.assertEqual(package.state, packager.STATE_INCOMPLETE)
        self.assertIn("ROLLBACK_ANCHOR_LIVENESS_UNSTATED", codes_of(package))

    def test_a_stated_live_anchor_still_clears_and_a_dead_one_still_blocks(self):
        """CONTROL, both directions: the tristate keeps the two states it had."""
        self.assertEqual(release_package().state, packager.STATE_READY)
        dead = release_package(rollback=rollback_plan(anchor_live=False))
        self.assertEqual(dead.state, packager.STATE_BLOCKED)
        self.assertIn("ROLLBACK_ANCHOR_NOT_LIVE", codes_of(dead))

    def test_an_unstated_shared_core_answer_is_not_a_no(self):
        package = release_package(diff_complexity={"diff_size": 4000,
                                                   "files_touched": 61})
        self.assertEqual(package.state, packager.STATE_INCOMPLETE)
        self.assertIn("DIFF_COMPLEXITY_SHARED_CORE_UNSTATED", codes_of(package))
        self.assertTrue(package.requires_human_code_review)

    def test_a_stated_shared_core_answer_still_clears(self):
        """CONTROL. `touches_shared_core: False` is an answer and must read as one."""
        package = release_package()
        self.assertEqual(package.state, packager.STATE_READY)
        self.assertFalse(package.requires_human_code_review)


class TestWatchWindowFoldsEveryObservation(unittest.TestCase):

    def progress_with(self, extra, *, first: bool = False):
        base = watch_progress()
        observations = (extra,) + base.observations if first \
            else base.observations + (extra,)
        return packager.WatchWindowProgress(
            now=base.now, volume_by_role=dict(base.volume_by_role),
            bands_sha256=base.bands_sha256, observations=observations)

    def excursion(self) -> packager.WatchObservation:
        return packager.WatchObservation(
            signal_id=packager.SIGNAL_THROUGHPUT, value=12.0,
            observed_at="2026-08-11T13:00:00Z", era_label="E9",
            samples_ref="data/ak-v9/watch/throughput-second-sample.jsonl")

    def test_a_second_observation_outside_the_band_still_alarms(self):
        """First-wins dedup dropped it and the window closed with `no regression`."""
        for first in (False, True):
            with self.subTest(position="first" if first else "last"):
                recommendation = packager.evaluate_watch_window(
                    watch_window(), self.progress_with(self.excursion(), first=first))
                self.assertEqual(recommendation.recommendation,
                                 packager.WATCH_RAISE_DECISION_PACKAGE)
                self.assertIn(packager.SIGNAL_THROUGHPUT, recommendation.alarms)

    def test_no_regression_cannot_be_recorded_over_an_alarming_signal(self):
        """'We looked and it alarmed' was accepted while 'we did not look' was not."""
        with self.assertRaises(packager.PackagerInputError) as caught:
            packager.close_watch_window(
                watch_window(), self.progress_with(self.excursion()),
                verdict="no_regression_observed", closed_by="daniele",
                closed_at="2026-08-11T14:00:00Z")
        self.assertIn("outside their bands", str(caught.exception))

    def test_the_same_window_closes_on_the_verdict_the_evidence_supports(self):
        """CONTROL. `regression_observed` over an alarm is the compliant closure."""
        closure = packager.close_watch_window(
            watch_window(), self.progress_with(self.excursion()),
            verdict="regression_observed", closed_by="daniele",
            closed_at="2026-08-11T14:00:00Z")
        self.assertEqual(closure.verdict, "regression_observed")
        self.assertIn(packager.SIGNAL_THROUGHPUT, closure.recommendation.alarms)

    def test_a_clean_window_still_closes_with_no_regression(self):
        """CONTROL. The refusal must not swallow the ordinary close."""
        closure = packager.close_watch_window(
            watch_window(), watch_progress(), verdict="no_regression_observed",
            closed_by="daniele", closed_at="2026-08-11T14:00:00Z")
        self.assertEqual(closure.verdict, "no_regression_observed")

    def test_one_observation_per_signal_still_closes_the_window(self):
        """CONTROL. Folding must not turn the ordinary window into an open one."""
        recommendation = packager.evaluate_watch_window(watch_window(), watch_progress())
        self.assertEqual(recommendation.recommendation,
                         packager.WATCH_CLOSE_NO_REGRESSION)

    def test_two_observations_inside_the_band_are_both_a_pass(self):
        """CONTROL. Repeated sampling is the normal case, not a finding."""
        second = packager.WatchObservation(
            signal_id=packager.SIGNAL_THROUGHPUT, value=49.9,
            observed_at="2026-08-11T13:00:00Z", era_label="E9",
            samples_ref="data/ak-v9/watch/throughput-second-sample.jsonl")
        recommendation = packager.evaluate_watch_window(
            watch_window(), self.progress_with(second))
        self.assertEqual(recommendation.recommendation,
                         packager.WATCH_CLOSE_NO_REGRESSION)

    def test_a_window_shorter_than_the_section_11_5_floor_is_refused(self):
        with self.assertRaises(packager.PackagerInputError) as caught:
            watch_window(min_duration_days=1)
        self.assertIn("floor", str(caught.exception))

    def test_the_declared_default_window_is_still_accepted(self):
        """CONTROL. Seven days is the rule, not the exception."""
        self.assertEqual(watch_window().min_duration_days,
                         packager.DEFAULT_WATCH_WINDOW_DAYS)


class TestVersionStalenessCountsTags(unittest.TestCase):

    def test_a_tag_that_moved_the_series_is_a_stale_incumbent(self):
        with self.assertRaises(packager.VersionCollision) as caught:
            packager.compute_next_version(
                incumbent_branch="production-consolidated-v8",
                existing_branches=("production-consolidated-v8",),
                existing_tags=("production-consolidated-v10",))
        self.assertIn("production-consolidated-v10", str(caught.exception))

    def test_an_older_tag_and_another_family_do_not_collide(self):
        """CONTROL. Only a HIGHER version of the SAME family is a moved series."""
        version = packager.compute_next_version(
            incumbent_branch="production-consolidated-v8",
            existing_branches=("production-consolidated-v7",
                               "production-consolidated-v8"),
            existing_tags=("production-consolidated-v7", "production-speech-v3"))
        self.assertEqual(version.next_branch, "production-consolidated-v9")


if __name__ == "__main__":
    unittest.main()
