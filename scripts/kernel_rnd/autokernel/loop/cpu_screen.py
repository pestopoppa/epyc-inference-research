"""Reduced common CPU conditions in the existing loop, not transfer authority.

Both source arms use the same reduced allocation/thread count. The original full
recipe remains the transfer target, and only its separate comparison can keep.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from ..controller import experiments
from ..execution.cpu_region_claim import parse_cpu_list
from . import archive, resolved_recipe as rr


class ScreenRefused(ValueError):
    pass


def prepare_launch(full: rr.CanonicalResolvedRecipe, scope: str, owned_cpus) -> dict:
    """Pure prospective preparation using the installed region owner's geometry."""
    from ..execution.cpu_region_claim import ATOMIC_REGIONS, REGION_CORE_RANGE, cpu_list_to_regions

    if type(full) is not rr.CanonicalResolvedRecipe or full.backend != "cpu":
        raise ScreenRefused("reduced screen requires an original canonical CPU launch")
    if scope not in {"quarter", "half"}:
        raise ScreenRefused("CPU screen scope must be quarter or half")
    allowed = set(owned_cpus)
    original = set(parse_cpu_list(full.template.cpu_list))
    if not original or not original <= allowed:
        raise ScreenRefused("original CPU launch is outside the declared owned CPUs")
    if any(cpu > 95 for cpu in original):
        raise ScreenRefused("reduced common scope does not infer SMT sibling geometry")
    # Use the existing physical-region owner, not a guessed NUMA/CCD or SMT map.
    # SMT-only placement is not represented by that owner's current region map.
    occupied = cpu_list_to_regions(full.template.cpu_list)
    count = 1 if scope == "quarter" else 2
    available = [name for name in ATOMIC_REGIONS if name in occupied]
    if len(available) < count:
        raise ScreenRefused("requested screen has insufficient original physical regions")
    regions = available[:count]
    cpus = sorted(cpu for cpu in original if any(
        REGION_CORE_RANGE[name][0] <= cpu <= REGION_CORE_RANGE[name][1] for name in regions))
    if not cpus or set(cpus) == original:
        raise ScreenRefused("requested reduced screen does not reduce the original allocation")
    cpu_list = ",".join(map(str, cpus))
    threads = min(full.template.threads, len(cpus))
    # Existing floor files are name/request keyed and validate recipe_hash inside.
    # Give this distinct common condition its own stable name; never overwrite the
    # original full target's floor or change the legacy filename contract.
    template = replace(full.template, cpu_list=cpu_list, threads=threads,
                       name=f"{full.template.name}.cpu-{scope}-{full.template.recipe_hash[:16]}")
    command, prefix = list(full.command_argv), list(full.topology_prefix)
    if prefix.count("taskset") != 1:
        raise ScreenRefused("original CPU topology does not expose one taskset")
    position = prefix.index("taskset")
    if prefix[position + 1:position + 2] != ["-c"] or position + 2 >= len(prefix):
        raise ScreenRefused("original CPU topology does not expose taskset -c")
    prefix[position + 2] = cpu_list
    for flag in ("-t", "-tb"):
        if flag not in command and flag == "-tb":
            continue
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ScreenRefused(f"original command does not expose one {flag}")
        command[command.index(flag) + 1] = str(threads)
    reduced = rr.resolve_canonical_launch(
        template, build_dir=full.build_dir, command_argv=command, topology_prefix=prefix,
        launch_environment=dict(full.launch_env),
        artifact_identities={"model": full.model.to_dict(),
            "drafter": full.drafter.to_dict() if full.drafter else None,
            "executable": full.executable.to_dict(), "dsos": [x.to_dict() for x in full.dsos]},
        backend=full.backend, environment_policy=full.environment_policy, port=full.port,
        runtime_binary_dir=full.runtime_binary_dir, runtime_ld_paths=full.runtime_ld_paths,
        provenance={**dict(full.provenance), "cpu_screen_full_target": full.snapshot_digest})
    return {"scope": scope, "full": full, "launch": reduced, "cpu_list": cpu_list,
            "regions": tuple(regions), "region_fraction": len(regions) / len(ATOMIC_REGIONS),
            "limitation": "both arms share reduced threads/affinity; not a measured gain from "
                          "changing those conditions, not NUMA-local equivalence, and no full-target transfer"}


def mechanism_hint(store_root: Path, full: rr.CanonicalResolvedRecipe) -> dict:
    """Read original qualitative mechanisms; hints neither grade nor retire them."""
    try:
        with experiments.ExperimentStore(store_root, read_only=True) as store:
            rows = store.recall(epoch="", limit=16, include_source_scope=True,
                statuses=("measured_null", "kept", "keep_candidate", "runtime_observed"))
        for row in rows:
            scope = row.get("research_scope") or {}
            model = scope.get("model")
            model = model.get("path") if isinstance(model, dict) else model
            if model != full.model.path:
                continue
            text = " ".join(str(row.get(key) or "") for key in (
                "mechanism_id", "statement", "target_symbol", "falsifier")).lower()
            if re.search(r"barrier|schedul|numa|scaling|thread.synchron", text):
                family, selection = "scale_sensitive", "full"
            elif re.search(r"prefetch|repack|layout|blocking|cache|traffic|cop(?:y|ies)", text):
                family, selection = "memory_geometry", "half"
            elif re.search(r"simd|arithmetic|redundant|dispatch|unpack|dequant|vec_dot|vecdot", text):
                family, selection = "local_work", "quarter"
            else:
                continue
            return {"scope": selection, "mechanism_family": family,
                    "source_store": str(store_root), "attempt_id": row["attempt_id"],
                    "mechanism_id": row["mechanism_id"], "statement": row["statement"],
                    "basis": "qualitative keyword hint from original recorded mechanism; not transfer evidence"}
    except Exception as exc:
        return {"scope": "full", "mechanism_family": "unknown",
                "reason": f"prior mechanism hint unavailable: {type(exc).__name__}: {exc}"[:512]}
    return {"scope": "full", "mechanism_family": "unknown",
            "reason": "no relevant recorded mechanism hint; retain the original full target"}


def read_candidate(reference: dict) -> dict:
    """Bounded immutable archive reopen; launch still requires the original run owner."""
    from . import serial_run

    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"} \
            or not all(isinstance(reference[k], str) for k in reference):
        raise ScreenRefused("invalid original screen candidate reference")
    path = Path(reference["path"])
    if not path.is_absolute():
        raise ScreenRefused("screen candidate path is not absolute")
    raw = serial_run._read(path)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ScreenRefused("original screen candidate bytes changed")
    body = json.loads(raw)
    required = {"schema", "origin_batch", "target", "original_head", "full_target",
                "evaluated_anchor", "candidate_launch", "request_digest", "hypothesis", "paths",
                "source_archive", "screen_capture", "screen_assessment"}
    if not isinstance(body, dict) or set(body) != required \
            or body["schema"] != "epyc.autokernel.cpu_screen_candidate.v1":
        raise ScreenRefused("invalid original screen candidate archive")
    return body


def retain_candidate(*, store_root, origin_batch, worker, target, hypothesis, paths,
                     full_target, comparison) -> dict:
    """Keep an original reduced positive without moving source or build directories."""
    if hypothesis.runtime_pair is not None:
        raise ScreenRefused("source-screen continuation does not select runtime recipes")
    row = comparison.to_dict()
    if not comparison.decisive or comparison.effect <= 0:
        raise ScreenRefused("reduced source candidate did not clear its own original floor")
    capture = row.get("belief_capture") or {}
    arms = (capture.get("inputs") or {}).get("resolved_arms") or {}
    if not arms.get("anchor") or not arms.get("candidate"):
        raise ScreenRefused("screen comparison lacks original resolved source arms")
    patch = archive.retain_patch(Path(store_root), worker.worktree, lane=worker.name,
                                 mechanism_id=hypothesis.mechanism_id)
    if patch is None:
        raise ScreenRefused("screen candidate has no original source patch")
    from . import serial_run
    metadata, metadata_sha = serial_run._json(patch.with_suffix(".json"))
    body = {"schema": "epyc.autokernel.cpu_screen_candidate.v1",
        "origin_batch": str(Path(origin_batch).resolve()), "target": target,
        "original_head": metadata["original_head"], "full_target": full_target.to_dict(),
        "evaluated_anchor": arms["anchor"], "candidate_launch": arms["candidate"],
        "request_digest": row.get("request_digest"), "hypothesis": hypothesis.to_dict(),
        "paths": list(paths), "source_archive": {"path": str(patch.with_suffix(".json").resolve()),
                                                   "sha256": metadata_sha},
        "screen_capture": {"capture_id": capture.get("capture_id"),
                           "native_sha256": capture.get("native_sha256")},
        "screen_assessment": {"effect": comparison.effect, "decisive": comparison.decisive,
                              "noise_floor_pct": row.get("noise_floor_pct"),
                              "scope": "reduced_only_full_target_confirmation_required"}}
    raw = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = patch.parent / f"candidate.{digest}.json"
    archive._retain_bytes(path, raw)
    return {"path": str(path.resolve()), "sha256": digest}


def confirmation_from(path, *, full_target, selected_target, request_digest, original_head):
    """Join a completed reduced batch to its retained source and full transfer target."""
    from . import serial_run

    path = Path(path).resolve()
    receipt, _ = serial_run.load_completed(path)
    screen = receipt.get("cpu_screen") or {}
    reference = screen.get("candidate")
    if (receipt["terminal"] != "complete" or receipt["iterations_completed"] != 1
            or receipt["outcome_counts"] != {"keep_candidate": 1}
            or screen.get("scope") not in {"quarter", "half"} or reference is None):
        raise ScreenRefused("original batch is not a completed pending reduced candidate")
    candidate = read_candidate(reference)
    recorded_full = rr.CanonicalResolvedRecipe.from_dict(candidate["full_target"])
    if (candidate["origin_batch"] != str(path.parent)
            or candidate["target"] != selected_target
            or candidate["original_head"] != original_head
            or receipt["current_anchor"]["commit"] != original_head
            or candidate["request_digest"] != request_digest
            or recorded_full.execution_digest != full_target.execution_digest):
        raise ScreenRefused("retained candidate differs from original base/target/request/full recipe")
    candidate["reference"] = reference
    candidate["scope"] = screen["scope"]
    return candidate


def verify_candidate(candidate, worker, reduced):
    """Original binary inventory and patch bytes, not a rebuild or renewed verdict."""
    from . import run, serial_run

    expected = rr.CanonicalResolvedRecipe.from_dict(candidate["candidate_launch"])
    if Path(expected.build_dir).resolve() != worker.build_dir.resolve():
        raise ScreenRefused("candidate build is not this original owned lane")
    actual = run._cpu_arm(reduced, worker.build_dir)
    if actual.execution_digest != expected.execution_digest:
        raise ScreenRefused("retained candidate binary/DSO/launch identity changed")
    ref = candidate["source_archive"]
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ScreenRefused("candidate source archive reference is malformed")
    metadata_path = Path(ref["path"])
    metadata, metadata_sha = serial_run._json(metadata_path)
    if (metadata_sha != ref["sha256"]
            or metadata.get("schema") != "epyc.autokernel.source_patch_archive.v1"
            or metadata.get("original_head") != candidate["original_head"]
            or metadata.get("worktree") != str(worker.worktree.resolve())
            or metadata.get("lane") != worker.name
            or metadata.get("mechanism_id") != candidate["hypothesis"].get("mechanism_id")):
        raise ScreenRefused("original source archive differs from candidate lane/base")
    patch_name = metadata.get("patch_file")
    if not isinstance(patch_name, str) or Path(patch_name).name != patch_name:
        raise ScreenRefused("original patch file is not a sibling of its metadata")
    patch = serial_run._read(metadata_path.parent / patch_name, limit=64 * 1024 * 1024)
    if hashlib.sha256(patch).hexdigest() != metadata.get("patch_sha256"):
        raise ScreenRefused("original retained source patch changed")
    if archive._git(worker.worktree, "rev-parse", "HEAD") != candidate["original_head"]:
        raise ScreenRefused("retained candidate base was superseded; no automatic rebase")
    return patch


def verify_restored(candidate, worker, reduced, store_root, hypothesis):
    """No source changes since restoration; reuse only the original proved binary."""
    original = verify_candidate(candidate, worker, reduced)
    path = archive.retain_patch(Path(store_root), worker.worktree, lane=worker.name,
                                mechanism_id=hypothesis.mechanism_id)
    from . import serial_run
    if path is None or serial_run._read(path, limit=64 * 1024 * 1024) != original:
        raise ScreenRefused("restored source differs from the originally built screen candidate")


class RetainedPlanner:
    """Only restores the already authored candidate; the real critic/gates still run."""

    def __init__(self, candidate, worker, reduced):
        self.candidate, self.worker, self.reduced = candidate, worker, reduced

    def propose(self, _context):
        from . import loop
        return loop.Hypothesis(**self.candidate["hypothesis"])

    def author(self, hypothesis, _context):
        if hypothesis.to_dict() != self.candidate["hypothesis"]:
            raise ScreenRefused("confirmation changed the original hypothesis")
        patch = verify_candidate(self.candidate, self.worker, self.reduced)
        if archive._git(self.worker.worktree, "status", "--porcelain", "--untracked-files=all"):
            raise ScreenRefused("confirmation lane is not reset before source restoration")
        archive._git(self.worker.worktree, "apply", "--check", "--binary", "-",
                     input_text=patch.decode("utf-8"))
        archive._git(self.worker.worktree, "apply", "--binary", "-", input_text=patch.decode("utf-8"))
        return tuple(self.candidate["paths"])


def preview_batch(original, prior, *, batch_iterations=1):
    """Read-only serial planning; a pending candidate outranks new work on its lane."""
    from . import campaign_cli, legacy_targets, serial_run

    if serial_run.option(original, "--cpu-serving-launch") is None:
        return {"scope": "full", "candidate": None}
    if prior is not None:
        receipt, sha = serial_run.load_completed(Path(prior["path"]))
        if sha != prior["sha256"]:
            raise ScreenRefused("original prior batch continuation changed")
        screen = receipt.get("cpu_screen") or {}
        if screen.get("candidate") is not None:
            if batch_iterations != 1:
                raise ScreenRefused("pending full confirmation requires its original single-candidate batch")
            read_candidate(screen["candidate"])
            return {"scope": "full_confirmation", "candidate": screen["candidate"],
                    "confirm_from": prior["path"]}
    if batch_iterations != 1 or serial_run.option(original, "--resolved-campaign") is None:
        return {"scope": "full", "candidate": None}
    full = rr.CanonicalResolvedRecipe.from_dict(serial_run._json(Path(
        serial_run.option(original, "--cpu-serving-launch")))[0])
    hint = mechanism_hint(Path(serial_run.option(original, "--store")), full)
    if hint["scope"] == "full":
        return {**hint, "candidate": None}
    if full.template.cpu_list is None:
        return {**hint, "scope": "full", "candidate": None,
                "reason": "original launch inherits owned affinity; no explicit reduced taskset geometry"}
    resolved = campaign_cli.load_previous(Path(serial_run.option(original, "--resolved-campaign")))
    legacy_targets.validate_resources(resolved.resources, full, backend="cpu", environment={})
    try:
        prepared = prepare_launch(full, hint["scope"], resolved.resources.cpu_logical)
    except (ScreenRefused, ValueError) as exc:
        return {**hint, "scope": "full", "candidate": None,
                "reason": f"reduced scope unavailable; original full target retained: {exc}"}
    return {**hint, "candidate": None, "cpu_list": prepared["cpu_list"],
            "region_fraction": prepared["region_fraction"]}


def prepare_batch(original, prior, directory, *, batch_iterations=1, previewed=None):
    """Add only existing-owner child arguments; no source/build/model writes."""
    from . import serial_run

    eligible = (serial_run.option(original, "--cpu-serving-launch") is not None
                and serial_run.option(original, "--resolved-campaign") is not None)
    selection_path = Path(directory) / "cpu-screen-selection.json"
    source = {"input_argv_sha256": serial_run._digest(list(original)),
              "binding": serial_run.input_binding(original), "prior": prior,
              "batch_iterations": batch_iterations}
    if eligible and selection_path.exists():
        retained, _sha = serial_run._json(selection_path)
        if not isinstance(retained, dict) or set(retained) != {"source", "selected"} \
                or retained["source"] != source:
            raise ScreenRefused("original batch CPU scope selection changed")
        selected = retained["selected"]
    else:
        selected = (preview_batch(original, prior, batch_iterations=batch_iterations)
                    if previewed is None else json.loads(json.dumps(previewed, allow_nan=False)))
        if eligible:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            archive._retain_bytes(selection_path, (json.dumps(
                {"source": source, "selected": selected}, sort_keys=True,
                allow_nan=False) + "\n").encode())
    if previewed is not None and selected != previewed:
        raise ScreenRefused("prepared CPU scope differs from original scheduler preview")
    if not isinstance(selected, dict) or selected.get("scope") not in {
            "full", "quarter", "half", "full_confirmation"}:
        raise ScreenRefused("invalid original batch scope selection")
    argv = serial_run._without(original, {"--cpu-screen-scope", "--cpu-confirm-from"})
    if selected["scope"] == "full_confirmation":
        argv += ["--cpu-confirm-from", selected["confirm_from"]]
    elif selected["scope"] in {"quarter", "half"}:
        argv += ["--cpu-screen-scope", selected["scope"]]
    return argv, selected


def planned_hint(directory, argv, scope):
    """The original preselection advice, not a new grading or applicability claim."""
    from . import serial_run
    path = Path(directory) / "cpu-screen-selection.json"
    if not path.exists():
        return {"scope": scope, "mechanism_family": "operator_selected",
                "reason": "explicit common scope; no historical mechanism hint supplied"}
    retained, _sha = serial_run._json(path)
    if (not isinstance(retained, dict) or set(retained) != {"source", "selected"}
            or not isinstance(retained["source"], dict) or not isinstance(retained["selected"], dict)
            or retained["source"].get("binding") != serial_run.input_binding(argv)
            or retained["selected"].get("scope") != scope):
        raise ScreenRefused("planned CPU scope advice differs from actual batch inputs")
    return dict(retained["selected"])


def scoped_proposal(original, preview):
    """Keep original time bound/policy; change only actually held common CPU scope."""
    from . import scheduling
    from ..execution.cpu_region_claim import ATOMIC_REGIONS, cpu_list_to_regions
    if not isinstance(preview, dict) or preview.get("scope") not in {
            "full", "quarter", "half", "full_confirmation"}:
        raise ScreenRefused("invalid prospective CPU scope")
    if preview["scope"] in {"full", "full_confirmation"}:
        return original
    if original.backend != "cpu" or original.estimated_claims.gpu_devices:
        raise ScreenRefused("reduced CPU scope cannot change a GPU proposal")
    cpus = parse_cpu_list(preview.get("cpu_list"))
    fraction = len(cpu_list_to_regions(preview["cpu_list"])) / len(ATOMIC_REGIONS)
    if (not cpus or fraction != preview.get("region_fraction")
            or fraction != {"quarter": .25, "half": .5}[preview["scope"]]
            or fraction > original.estimated_claims.physical_region_fraction):
        raise ScreenRefused("reduced scope resources differ from original owned proposal")
    claims = scheduling.ResourceVector(fraction, (), original.estimated_claims.memory_reservation_bytes)
    return replace(original, estimated_claims=claims, full_region=False)


def confirmation_debt(config, state, proposal, preview):
    """Do not replace an unaffordable full confirmation with new reduced work."""
    from . import scheduling
    if preview["scope"] != "full_confirmation":
        return None
    vector, capacity = proposal.estimated_claims, state.capacity
    if (vector.physical_region_fraction > capacity.physical_region_fraction
            or not set(vector.gpu_devices) <= set(capacity.gpu_devices)
            or vector.memory_reservation_bytes > capacity.memory_reservation_bytes):
        return "pending full confirmation exceeds current declared capacity"
    if (state.campaign_attempts >= config.campaign_attempt_cap
            or state.campaign_charged_seconds + scheduling._dominant_estimate(capacity, proposal)
            > config.campaign_charged_seconds_cap):
        return "pending full confirmation cannot fit remaining campaign attempt/time budget"
    for seed in state.seed_accounts:
        if (seed.backend, seed.target_revision, seed.alias_identity) == (
                proposal.backend, proposal.target_revision, proposal.alias_identity):
            if (seed.attempts >= config.seed_attempt_cap
                    or seed.charged_seconds + proposal.estimated_duration_seconds
                    > config.seed_charged_seconds_cap):
                return "pending full confirmation cannot fit remaining original seed budget"
    return None


def pending_collisions(targets, last_results):
    """Protect original pending source/build using existing continuation bindings."""
    from . import champion, serial_run

    def roots(argv):
        return [Path(serial_run.option(argv, key)).resolve() for key in (
            "--worktree", "--worker-root", "--worker-build-root")]

    def common(worktree):
        observed = champion._git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if observed.returncode:
            raise ScreenRefused("pending source Git ownership identity is unavailable")
        return Path(observed.stdout.strip()).resolve()

    blocked = {}
    for key, reference in last_results.items():
        index = int(key)
        receipt, sha = serial_run.load_completed(Path(reference["path"]),
            expected_binding=serial_run.input_binding(targets[index]))
        if sha != reference["sha256"]:
            raise ScreenRefused("pending continuation bytes changed")
        if (receipt.get("cpu_screen") or {}).get("candidate") is None:
            continue
        original = roots(receipt["input_argv"])
        original_common = None
        for other, argv in enumerate(targets):
            if other == index:
                continue
            candidate_roots = roots(argv)
            overlapping = any(left == right or left in right.parents or right in left.parents
                              for left in original for right in candidate_roots)
            branch = serial_run.option(argv, "--experimental-branch") or serial_run.option(
                argv, "--champion-branch", champion.CANONICAL_BRANCH)
            same_branch = False
            if branch == receipt["branch"]:
                original_common = original_common or common(original[0])
                same_branch = common(candidate_roots[0]) == original_common
            if overlapping or same_branch:
                blocked[other] = (f"original pending full confirmation for target "
                    f"{serial_run.option(targets[index], '--target-id')} protects this shared source/build")
    return blocked


def routing(value, argv):
    """Closed optional routing metadata, never a grade or a promoted-source receipt."""
    from . import serial_run
    fields = {"scope", "full_execution_digest", "measured_execution_digest", "candidate"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ScreenRefused("invalid CPU screen routing fields")
    scope = value["scope"]
    if scope not in {"quarter", "half", "full_confirmation"}:
        raise ScreenRefused("invalid CPU screen scope")
    if (serial_run.option(argv, "--cpu-serving-launch") is None
            or serial_run.option(argv, "--resolved-campaign") is None
            or int(serial_run.option(argv, "--iterations", "10")) != 1
            or (scope != "full_confirmation" and serial_run.option(argv, "--cpu-screen-scope") != scope)
            or (scope != "full_confirmation" and serial_run.option(argv, "--cpu-confirm-from"))
            or (scope == "full_confirmation" and (not serial_run.option(argv, "--cpu-confirm-from")
                                                  or serial_run.option(argv, "--cpu-screen-scope")))):
        raise ScreenRefused("CPU screen scope differs from original child inputs")
    for key in ("full_execution_digest", "measured_execution_digest"):
        if not isinstance(value[key], str) or not re.fullmatch("[0-9a-f]{64}", value[key]):
            raise ScreenRefused("invalid CPU screen execution digest")
    if scope == "full_confirmation" and (value["candidate"] is not None or
            value["full_execution_digest"] != value["measured_execution_digest"]):
        raise ScreenRefused("full confirmation cannot route another provisional candidate")
    if value["candidate"] is not None:
        read_candidate(value["candidate"])
    return dict(value)
