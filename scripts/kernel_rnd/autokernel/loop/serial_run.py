"""Serial finite batches of the existing loop, not another measurement owner."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time

from . import campaign_cli, champion, legacy_targets, status, worker_lifecycle

CONTINUATION_SCHEMA = "epyc.autokernel.loop_continuation.v1"
CONTINUATION_SCHEMA_V2 = "epyc.autokernel.loop_continuation.v2"
HELD_REFERENCE_SCHEMA = "epyc.autokernel.direct_held_reference.v1"
SERIAL_SCHEMA = "epyc.autokernel.serial_run.v1"
_DOCUMENT_FLAGS = ("--resolved-campaign", "--cpu-serving-launch", "--gpu-serving-launch",
                   "--frozen-prompts", "--serving-recipe", "--runtime-recipe-reference",
                   "--runtime-recovery-reference")
_CHANGING_FLAGS = frozenset({"--out", "--iterations", "--resume-run", "--anchor-build",
                            "--cor-build", "--cpu-calibrate-serving", "--gpu-calibrate-serving",
                            "--scheduler-selection", "--dry-run", "--cpu-screen-scope",
                            "--cpu-confirm-from", "--source-anchor-continuation",
                            "--source-anchor-sha256", "--runtime-recipe-reference",
                            "--runtime-recovery-reference", "--validate-source-continuation"})
_BOOLEAN_FLAGS = frozenset({"--dry-run", "--validate-source-continuation"})
_CONTINUATION_FIELDS = {"schema", "terminal", "input_argv", "input_argv_sha256", "binding",
                        "worktree", "branch", "model", "selected_target", "current_anchor", "cor_anchor",
                        "iterations_requested", "iterations_completed", "outcome_counts", "result_file"}
_CONTINUATION_FIELDS_V2 = _CONTINUATION_FIELDS | {"held_claim_evidence"}


class SerialRefused(ValueError):
    pass


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path: Path, *, limit: int = 2 * 1024 * 1024) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise SerialRefused(f"not a bounded regular file: {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SerialRefused(f"file changed while read: {path}")
    return raw


def _json(path: Path, *, limit: int = 2 * 1024 * 1024):
    raw = _read(path, limit=limit)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def option(argv, name, default=None):
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise SerialRefused(f"{name} has no value")
            values.append(argv[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg[len(name) + 1:])
    # Match the existing argparse last-value behavior, including old launchers
    # which append a finite iteration override to common arguments.
    return values[-1] if values else default


def _without(argv, names):
    out = []
    iterator = iter(argv)
    for arg in iterator:
        flag, equals, _value = arg.partition("=")
        if flag in names:
            if not equals and flag not in _BOOLEAN_FLAGS:
                next(iterator, None)
        else:
            out.append(arg)
    return out


def input_binding(argv) -> dict:
    """Original argument/workload bytes; never hash a model or derive a grant."""
    documents = {}
    for flag in _DOCUMENT_FLAGS:
        value = option(argv, flag)
        if value is not None:
            documents[flag] = hashlib.sha256(_read(Path(value))).hexdigest()
    return {"argv": _without(argv, _CHANGING_FLAGS), "documents": documents}


def resume_binding(argv) -> dict:
    """Prior-child binding before adding a newly selected runtime recipe."""
    return input_binding(_without(argv, {"--runtime-recipe-reference", "--runtime-recovery-reference"}))


def load_resume(path: Path, current_argv):
    """Reopen a self-bound prior, then compare only stable target inputs."""
    row, sha = load_completed(path)
    if resume_binding(row["input_argv"]) != resume_binding(current_argv):
        raise SerialRefused("result is for different stable resume inputs")
    return row, sha


def _held_reference(value):
    if not isinstance(value, dict) or set(value) != {"schema", "selection_digest", "evidence"} \
            or value.get("schema") != HELD_REFERENCE_SCHEMA:
        raise SerialRefused("held-claim reference shape is invalid")
    selection = value["selection_digest"]
    evidence = value["evidence"]
    if not isinstance(selection, str) or len(selection) != 64 \
            or any(char not in "0123456789abcdef" for char in selection):
        raise SerialRefused("held-claim selection digest is invalid")
    if not isinstance(evidence, dict) or set(evidence) != {"locator", "sha256", "verified"} \
            or not isinstance(evidence["locator"], str) or not evidence["locator"] \
            or not isinstance(evidence["sha256"], str) or len(evidence["sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in evidence["sha256"]) \
            or evidence["verified"] is not True:
        raise SerialRefused("held-claim evidence locator is invalid")
    return {"schema": HELD_REFERENCE_SCHEMA, "selection_digest": selection,
            "evidence": dict(evidence)}


def _runtime_recipe_reference(value):
    if not isinstance(value, dict) or set(value) != {"locator", "sha256", "verified"} \
            or not isinstance(value["locator"], str) or not value["locator"] \
            or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in value["sha256"]) \
            or value["verified"] is not True:
        raise SerialRefused("runtime recipe reference is malformed")
    return {"locator": value["locator"], "sha256": value["sha256"], "verified": True}


def last_outcome_reference(outcomes, *, store):
    """Point to the last completed outcome's EXISTING artifacts, not the batch."""
    if not outcomes or outcomes[-1].comparison is None:
        return None
    outcome = outcomes[-1]
    comparison = outcome.comparison.to_dict()
    native = comparison.get("belief_capture")
    path = comparison.get("belief_export_receipt")
    if path is None and isinstance(native, dict):
        path = Path(store) / "serving-beliefs" / f"{native['capture_id']}.json"
    receipt = None
    if path is not None:
        try:
            path = Path(path).absolute()
            if path.parent.resolve() != (Path(store) / "serving-beliefs").resolve():
                raise SerialRefused("last outcome receipt is outside its original store")
            receipt = {"path": str(path), "sha256": hashlib.sha256(_read(path, limit=16384)).hexdigest()}
        except (OSError, ValueError) as exc:
            print(f"warning: last outcome serving reference unavailable: {exc}", file=sys.stderr)
    runtime = comparison.get("runtime_admission")
    try:
        runtime = None if runtime is None else _runtime_recipe_reference(runtime)
    except (TypeError, ValueError) as exc:
        print(f"warning: last outcome runtime reference unavailable: {exc}", file=sys.stderr)
        runtime = None
    if receipt is None and runtime is None:
        return None
    return {"iteration_index": len(outcomes) - 1, "status": outcome.status,
            "mechanism_id": (None if outcome.hypothesis is None else outcome.hypothesis.mechanism_id),
            "serving_receipt": receipt,
            "runtime_result": runtime}


def read_last_outcome_reference(row):
    """Original-reader diagnostic. Never admits work, regrades, or needs a claim.

    A missing/corrupt optional source does not invalidate the owning continuation.
    Runtime verification below is stored-byte/namespace identity, not re-admission.
    """
    result = {"status": "absent", "serving": None, "runtime": None, "errors": [],
              "scope": "last completed outcome only; not full batch evidence",
              "scientific_eligibility": False}
    value = row.get("last_outcome_reference")
    if value is None:
        return result
    result["status"] = "unavailable"
    try:
        if (not isinstance(value, dict) or set(value) != {"iteration_index", "status", "mechanism_id",
                "serving_receipt", "runtime_result"}
                or len(json.dumps(value).encode()) > 12 * 1024
                or type(value["iteration_index"]) is not int
                or value["iteration_index"] != row["iterations_completed"] - 1
                or value["iteration_index"] < 0
                or not isinstance(value["status"], str) or len(value["status"]) > 512
                or row["outcome_counts"].get(value["status"], 0) < 1
                or (value["mechanism_id"] is not None and (
                    not isinstance(value["mechanism_id"], str) or len(value["mechanism_id"]) > 512))
                or value["serving_receipt"] is None and value["runtime_result"] is None):
            raise SerialRefused("last outcome reference shape or completed index differs")
        argv = row["input_argv"]
        store = Path(option(argv, "--store")).resolve()
        prompts = None
        if option(argv, "--frozen-prompts") is not None:
            from .planned_serving import FrozenPromptManifest
            prompts = FrozenPromptManifest.from_dict(_json(Path(option(argv, "--frozen-prompts")))[0])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"].append(str(exc)[:512])
        return result
    if value["serving_receipt"] is not None:
        try:
            import importlib
            reference = value["serving_receipt"]
            if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                    or not isinstance(reference["path"], str) or len(reference["path"]) > 4096):
                raise SerialRefused("last outcome serving reference shape differs")
            path = Path(reference["path"])
            if (not path.is_absolute() or path.parent.resolve() != store / "serving-beliefs"
                    or path.suffix != ".json" or len(path.stem) != 64
                    or any(char not in "0123456789abcdef" for char in path.stem)):
                raise SerialRefused("last outcome serving reference is not its original local capture")
            raw = _read(path, limit=16384)
            if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                raise SerialRefused("last outcome serving receipt bytes differ")
            root = Path(option(argv, "--belief-root-repo") or os.environ.get("EPYC_ROOT_REPO", "/workspace"))
            expected = (root / "scripts/vidya/adapters/autokernel_legacy_serving.py").resolve()
            python_path = str(root / "scripts/vidya")
            if python_path not in sys.path:
                sys.path.insert(0, python_path)
            reader = importlib.import_module("adapters.autokernel_legacy_serving")
            if Path(reader.__file__).resolve() != expected:
                raise SerialRefused("last outcome reader differs from the selected ROOT")
            receipt = json.loads(raw)
            if receipt.get("capture_id") != path.stem:
                raise SerialRefused("last outcome serving path/capture ID differs")
            source = reader._source(receipt, f"autokernel:{path}")
            reader._rows(source, receipt)  # Original capture/reducer, no new parser or grade.
            if source.get("mechanism_id") != value["mechanism_id"]:
                raise SerialRefused("last outcome serving mechanism differs")
            comparison = source["comparison"]
            inputs = comparison["belief_capture"]["inputs"]
            if Path(inputs["recipe"]["model"]).resolve() != Path(row["model"]).resolve():
                raise SerialRefused("last outcome serving model differs")
            if prompts is not None:
                from . import serving
                template = serving.Recipe.from_dict(inputs["recipe"])
                requests = prompts.requests(tuple(prompt.prompt_id for prompt in prompts.prompts), template)
                if serving.request_digest(template, requests) != comparison.get("request_digest"):
                    raise SerialRefused("last outcome serving original requests differ")
            result["serving"] = {"capture_id": receipt["capture_id"],
                "source_sha256": receipt["native_reference"]["sha256"],
                "epoch": source["epoch"], "recipe_hash": comparison["recipe_hash"],
                "request_digest": comparison.get("request_digest"),
                "verification": "original_observation_capture; not scientific qualification"}
        except Exception as exc:
            result["errors"].append(f"serving: {type(exc).__name__}: {exc}"[:512])
    if value["runtime_result"] is not None:
        try:
            from . import runtime_admission
            from .measurement_capture import ArtifactStore
            reference = _runtime_recipe_reference(value["runtime_result"])
            root = store / "runtime-preparation"
            if not root.is_dir():
                raise SerialRefused("original runtime artifact store is missing")
            artifacts = ArtifactStore(root)
            try:
                native = runtime_admission._read(artifacts, "direct-runtime-admission", reference)
            finally:
                artifacts.close()
            if (native["schema"] != runtime_admission.SCHEMA
                    or native["pair"]["dimension"]["dimension_id"] != value["mechanism_id"]
                    or any(Path(arm["model"]["path"]).resolve() != Path(row["model"]).resolve()
                           for arm in (native["pair"]["anchor"], native["pair"]["candidate"]))):
                raise SerialRefused("last outcome original runtime subject differs")
            if (prompts is not None and native["prompt_manifest_digest"] != prompts.digest
                    or native["source_commit"] != row["current_anchor"]["commit"]):
                raise SerialRefused("last outcome runtime original requests/source differ")
            result["runtime"] = {"reference": reference, "source_commit": native["source_commit"],
                "prompt_manifest_digest": native["prompt_manifest_digest"],
                "recorded_admitted": native["admitted"],
                "verification": "original artifact bytes/namespace only; not scientific re-admission"}
        except Exception as exc:
            result["errors"].append(f"runtime: {type(exc).__name__}: {exc}"[:512])
    result["status"] = ("partial" if result["errors"] else "available") if (
        result["serving"] is not None or result["runtime"] is not None) else "unavailable"
    return result


def _source_lineage_matches(row, receipts, validation_body=None):
    exact_local = (receipts and
        receipts[-1].kept_commit == row["current_anchor"]["commit"] and
        all(receipt.branch == row["branch"]
            and Path(receipt.repo).resolve() == Path(row["worktree"]).resolve()
            for receipt in receipts))
    exact_validated_foreign = False
    if receipts and validation_body is not None:
        latest = receipts[-1]
        try:
            from . import surface_validation as validation
            validation.shared_git_commit(
                Path(row["worktree"]), Path(latest.repo), latest.kept_commit)
        except validation.SurfaceValidationRefused:
            pass
        else:
            exact_validated_foreign = (
                (validation_body["source_commit"] == latest.kept_commit
                 or (validation_body["source_commit"] in {
                         receipts[0].parent_commit, receipts[0].kept_commit}
                     and latest.kept_commit == row["current_anchor"]["commit"]))
                and all(receipt.branch == latest.branch
                        and Path(receipt.repo).resolve() == Path(latest.repo).resolve()
                        for receipt in receipts))
    return exact_local or exact_validated_foreign


def continuation(*, argv, binding, terminal, worktree, branch, model, selected_target,
                 anchor_build, anchor_commit, iterations_requested, outcomes,
                 cor_build=None, cor_commit=None, held_claim_evidence=None, cpu_screen=None,
                 runtime_recipe_reference=None, experimental_source_keeps=None,
                 source_validation=None, source_lineage_keeps=None,
                 last_outcome_reference=None) -> dict:
    counts = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    row = {"schema": CONTINUATION_SCHEMA, "terminal": terminal,
            "input_argv": list(argv), "input_argv_sha256": _digest(list(argv)), "binding": binding,
            "worktree": str(Path(worktree).resolve()), "branch": branch,
            "model": str(Path(model).resolve()), "selected_target": selected_target,
            "iterations_requested": iterations_requested, "iterations_completed": len(outcomes),
            "outcome_counts": counts, "result_file": "loop-run.json",
            "current_anchor": {"path": str(Path(anchor_build).resolve()), "commit": anchor_commit},
            "cor_anchor": ({"path": str(Path(cor_build).resolve()), "commit": cor_commit}
                           if cor_build is not None else None)}
    if held_claim_evidence is not None:
        row["schema"] = CONTINUATION_SCHEMA_V2
        row["held_claim_evidence"] = _held_reference(held_claim_evidence)
    if cpu_screen is not None:
        from .cpu_screen import routing
        row["cpu_screen"] = routing(cpu_screen, argv)
    if runtime_recipe_reference is not None:
        row["runtime_recipe_reference"] = _runtime_recipe_reference(
            runtime_recipe_reference)
    if last_outcome_reference is not None:
        encoded = json.dumps(last_outcome_reference)
        if len(encoded.encode()) <= 12 * 1024:
            row["last_outcome_reference"] = json.loads(encoded)
        else:
            print("warning: last outcome reference exceeds its 12KiB transport bound", file=sys.stderr)
    if experimental_source_keeps:
        refs = list(experimental_source_keeps)
        if len(refs) > iterations_requested:
            raise SerialRefused("source keep references exceed completed iteration budget")
        from . import surface_fold
        for reference in refs:
            surface_fold.reopen_reference(reference)
        row["experimental_source_keeps"] = refs
    validation_body = None
    if source_validation is not None:
        from . import surface_validation as validation
        validation_body = validation.reopen_reference(source_validation)
        if validation_body["target"] != selected_target:
            raise SerialRefused("whole-source validation differs from continuation")
        row["source_validation"] = dict(source_validation)
    if source_lineage_keeps:
        from . import surface_fold
        refs = list(source_lineage_keeps)
        if len(refs) > 64:
            raise SerialRefused("source lineage keep membership exceeds its bound")
        receipts = [surface_fold.reopen_reference(reference) for reference in refs]
        if not _source_lineage_matches(row, receipts, validation_body):
            raise SerialRefused("source lineage differs from continuation source")
        if (source_validation is not None
                and validation_body["candidate_anchor"]["commit"]
                not in {receipt.kept_commit for receipt in receipts}):
            raise SerialRefused("validation candidate is outside retained source lineage")
        row["source_lineage_keeps"] = refs
    return row


def load_completed(path: Path, *, expected_argv=None, expected_binding=None):
    # The full result can legitimately contain hundreds of MiB of original
    # lifecycle observations. This original-owner routing receipt does not parse
    # or grade them, and never pretends that its file stat is a content hash.
    row, sha = _json(path)
    expected_fields = (_CONTINUATION_FIELDS_V2 if isinstance(row, dict)
                       and row.get("schema") == CONTINUATION_SCHEMA_V2
                       else _CONTINUATION_FIELDS)
    if isinstance(row, dict) and "cpu_screen" in row:
        expected_fields = expected_fields | {"cpu_screen"}
    if isinstance(row, dict) and "runtime_recipe_reference" in row:
        expected_fields = expected_fields | {"runtime_recipe_reference"}
    if isinstance(row, dict) and "experimental_source_keeps" in row:
        expected_fields = expected_fields | {"experimental_source_keeps"}
    if isinstance(row, dict) and "source_validation" in row:
        expected_fields = expected_fields | {"source_validation"}
    if isinstance(row, dict) and "source_lineage_keeps" in row:
        expected_fields = expected_fields | {"source_lineage_keeps"}
    if isinstance(row, dict) and "last_outcome_reference" in row:
        expected_fields = expected_fields | {"last_outcome_reference"}
    if not isinstance(row, dict) or set(row) != expected_fields \
            or row["schema"] not in {CONTINUATION_SCHEMA, CONTINUATION_SCHEMA_V2} \
            or row["terminal"] not in {"complete", "stopped"}:
        raise SerialRefused("missing or malformed terminal continuation")
    if row["schema"] == CONTINUATION_SCHEMA_V2:
        row["held_claim_evidence"] = _held_reference(row["held_claim_evidence"])
    argv = row["input_argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv) \
            or row["input_argv_sha256"] != _digest(argv):
        raise SerialRefused("continuation input arguments differ")
    if expected_argv is not None and argv != list(expected_argv):
        raise SerialRefused("result is for different actual child arguments")
    if "cpu_screen" in row:
        from .cpu_screen import routing
        row["cpu_screen"] = routing(row["cpu_screen"], argv)
        if row["cpu_screen"]["candidate"] is not None and (
                row["outcome_counts"] != {"keep_candidate": 1}):
            raise SerialRefused("pending screen candidate lacks its original completed provisional outcome")
    elif option(argv, "--cpu-screen-scope") or option(argv, "--cpu-confirm-from"):
        raise SerialRefused("CPU screen result omitted its original scope")
    if "runtime_recipe_reference" in row:
        row["runtime_recipe_reference"] = _runtime_recipe_reference(
            row["runtime_recipe_reference"])
    if "experimental_source_keeps" in row:
        from . import surface_fold
        refs = row["experimental_source_keeps"]
        if not isinstance(refs, list) or len(refs) > row["iterations_completed"]:
            raise SerialRefused("source keep reference count differs from completed iterations")
        for reference in refs:
            receipt = surface_fold.reopen_reference(reference)
            if (receipt.selected_target != row["selected_target"]
                    or ("source_validation" not in row
                        and (receipt.branch != row["branch"]
                             or Path(receipt.repo).resolve()
                             != Path(row["worktree"]).resolve()))):
                raise SerialRefused("source keep reference differs from original child identity")
    checked = None
    if "source_validation" in row:
        from . import surface_validation as validation
        checked = validation.reopen_reference(row["source_validation"])
        if checked["target"] != row["selected_target"]:
            raise SerialRefused("whole-source validation differs from original child identity")
    if "source_lineage_keeps" in row:
        from . import surface_fold
        refs = row["source_lineage_keeps"]
        if not isinstance(refs, list) or not refs or len(refs) > 64:
            raise SerialRefused("source lineage keep membership is malformed")
        receipts = [surface_fold.reopen_reference(reference) for reference in refs]
        if not _source_lineage_matches(row, receipts, checked):
            raise SerialRefused("source lineage keep membership differs")
        if ("source_validation" in row
                and checked["candidate_anchor"]["commit"]
                not in {receipt.kept_commit for receipt in receipts}):
            raise SerialRefused("validation candidate is outside source lineage")
    if not isinstance(row["binding"], dict) or set(row["binding"]) != {"argv", "documents"}:
        raise SerialRefused("invalid original input binding")
    if row["binding"] != input_binding(argv):
        raise SerialRefused("continuation binding does not match its own original arguments")
    if expected_binding is not None and row["binding"] != expected_binding:
        raise SerialRefused("result workload/input binding differs")
    for key in ("worktree", "model"):
        if not isinstance(row[key], str) or not Path(row[key]).is_absolute():
            raise SerialRefused(f"invalid continuation {key}")
    if not isinstance(row["branch"], str) or not row["branch"]:
        raise SerialRefused("invalid continuation branch")
    if Path(row["worktree"]).resolve() != Path(option(argv, "--worktree", "")).resolve():
        raise SerialRefused("recorded worktree differs from actual input arguments")
    cpu = option(argv, "--cpu-serving-launch") is not None
    gpu = option(argv, "--gpu-serving-launch") is not None
    experimental = cpu or (gpu and option(argv, "--experimental-branch") is not None)
    branch = option(argv, "--experimental-branch") if experimental else option(
        argv, "--champion-branch", champion.CANONICAL_BRANCH)
    if (cpu and gpu) or branch != row["branch"] or experimental != (row["cor_anchor"] is None):
        raise SerialRefused("recorded branch/backend differs from actual input arguments")
    if option(argv, "--resolved-campaign") is not None:
        resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
        selected = legacy_targets.select_target(resolved, option(argv, "--target-id"), cpu_serving=cpu)
        expected_target = {"campaign_id": resolved.campaign_id, "request_id": resolved.request_id,
                           "manifest_digest": resolved.manifest_digest,
                           "selected_id": option(argv, "--target-id"),
                           "scope": ("cpu_serving_selected_workload" if cpu else
                                     "gpu_serving_selected_workload" if gpu else "legacy_gpu_screen"),
                           "original_target": selected.to_dict()}
        if row["selected_target"] != expected_target \
                or Path(row["model"]).resolve() != Path(selected.execution.model.path).resolve():
            raise SerialRefused("recorded selected target/model differs from original inputs")
    elif row["selected_target"] is not None \
            or Path(row["model"]).resolve() != Path(option(argv, "--model", "")).resolve():
        raise SerialRefused("recorded model differs from actual input arguments")
    for key in ("current_anchor", "cor_anchor"):
        anchor = row[key]
        if key == "cor_anchor" and anchor is None:
            continue
        if not isinstance(anchor, dict) or set(anchor) != {"path", "commit"} \
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute() \
                or not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40 \
                or any(c not in "0123456789abcdef" for c in anchor["commit"]):
            raise SerialRefused(f"invalid {key}")
    if row["result_file"] != "loop-run.json" or Path(option(argv, "--out", "")).resolve() != path.parent.resolve():
        raise SerialRefused("continuation is not at its original output path")
    result = path.parent / row["result_file"]
    info = result.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise SerialRefused("original full result is missing or not a regular nonempty file")
    requested, count = row["iterations_requested"], row["iterations_completed"]
    if type(requested) is not int or type(count) is not int or count < 0 \
            or requested != int(option(argv, "--iterations", "10")):
        raise SerialRefused("invalid original iteration counts")
    counts = row["outcome_counts"]
    if not isinstance(counts, dict) or any(not isinstance(k, str) or type(v) is not int or v < 1
                                           for k, v in counts.items()) or sum(counts.values()) != count:
        raise SerialRefused("outcome counts differ from completed iterations")
    if requested <= 0 or count > requested or (row["terminal"] == "complete" and count != requested):
        raise SerialRefused("result does not cover its finite batch")
    return row, sha


def full_commit(worktree: Path, commit: str) -> str:
    done = champion._git(worktree, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if done.returncode != 0:
        raise SerialRefused("original commit cannot be resolved unambiguously")
    return done.stdout.strip()


def verify_exact_anchor(path: Path, worktree: Path, commit: str, *, experimental=False):
    """Reuse the original verifier, then require the named exact arm, not an ancestor."""
    commit = full_commit(worktree, commit)
    champion.verify_anchor(path, worktree, commit, experimental_identity=experimental)
    prov = path / "provenance.json"
    if prov.exists():
        body, _sha = _json(prov)
        named = body.get("champion_commit") if isinstance(body, dict) else None
        if not isinstance(named, str) or not named:
            raise SerialRefused("anchor provenance has no original commit")
        if full_commit(worktree, named) != commit:
            raise SerialRefused("anchor provenance is not the exact recorded arm")
    # Without provenance the existing experimental-identity verifier already
    # required exact head/source and original binary inventory; no new receipt.


def _target_args(path: Path) -> list[str]:
    argv, _sha = _json(path, limit=64 * 1024)
    return _validate_target_args(argv)


def _validate_target_args(argv, *, owner_anchor_waiver=False) -> list[str]:
    if not isinstance(argv, list) or not argv or len(argv) > 512 \
            or not all(isinstance(item, str) and "\0" not in item for item in argv):
        raise SerialRefused("target args must be a bounded JSON string array")
    reserved = {"--out", "--iterations", "--resume-run", "--scheduler-selection", "--dry-run",
                "--calibrate-surface",
                "--source-anchor-continuation", "--source-anchor-sha256",
                "--runtime-recipe-reference", "--runtime-recovery-reference",
                "--validate-source-continuation",
                "--allow-unverified-anchor", "--help", "-h"}
    if owner_anchor_waiver:
        reserved.remove("--allow-unverified-anchor")
    if any(item.partition("=")[0] in reserved for item in argv):
        raise SerialRefused("target args contain a serial-owned or non-execution option")
    for flag in ("--worktree", "--anchor-build", "--store", "--worker-root", "--worker-build-root",
                 "--resolved-campaign", "--target-id"):
        value = option(argv, flag)
        if not value or (flag != "--target-id" and not Path(value).is_absolute()):
            raise SerialRefused(f"serial target requires explicit absolute {flag}" if flag != "--target-id"
                                else "serial target requires --target-id")
    resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
    legacy_targets.select_target(resolved, option(argv, "--target-id"),
                                 cpu_serving=option(argv, "--cpu-serving-launch") is not None,
                                 model=Path(option(argv, "--model")) if option(argv, "--model") else None)
    return argv


def _scheduler_bindings(targets):
    from . import unified_planner
    bindings = {}
    for argv in targets:
        resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
        selected_id = option(argv, "--target-id")
        selected = legacy_targets.select_target(
            resolved, selected_id,
            cpu_serving=option(argv, "--cpu-serving-launch") is not None,
            model=Path(option(argv, "--model")) if option(argv, "--model") else None)
        if selected_id in bindings:
            raise SerialRefused("serial roster repeats a scheduled target ID")
        bindings[selected_id] = {
            "target_revision": unified_planner._target_digest(selected),
            "alias_identity": selected.workload_signature,
            "backend": selected.execution.backend,
            "eligibility_ref": f"resolved-campaign:{resolved.manifest_digest}:ready",
        }
    return bindings


def _derived_scheduler_manifest(targets, resolved_path, rounds):
    """Derive ordinary roster scheduling from its enrolled resource declaration."""
    # This pure mirror is structurally tested against the installed provider and
    # avoids importing the orchestrator's mutable package graph during routing.
    from ..execution.cpu_region_claim import ATOMIC_REGIONS, cpu_list_to_regions
    from . import claim, scheduling, serial_scheduling, unified_planner
    from .resolved_recipe import CanonicalResolvedRecipe
    resolved = campaign_cli.load_previous(Path(resolved_path))
    # One held invocation can contain planner, critic, validation and serving
    # phases in addition to a build. This is a declared scheduling bound, not a
    # duration observation; an overrun remains charged and successor-fenced.
    max_stage = resolved.resources.build_timeout_s + 4 * resolved.resources.stage_timeout_s
    attempt_cap = rounds * len(targets) if rounds else 1000
    proposals = {}
    has_gpu = False
    for argv in targets:
        selected_id = option(argv, "--target-id")
        cpu = option(argv, "--cpu-serving-launch") is not None
        selected = legacy_targets.select_target(resolved, selected_id, cpu_serving=cpu)
        launch_path = option(argv, "--cpu-serving-launch") or option(argv, "--gpu-serving-launch")
        launch_body, _sha = _json(Path(launch_path))
        launch = CanonicalResolvedRecipe.from_dict(launch_body)
        owned = legacy_targets.validate_resources(
            resolved.resources, launch, backend=selected.execution.backend,
            environment=os.environ)
        fraction = len(cpu_list_to_regions(owned)) / len(ATOMIC_REGIONS)
        gpu_devices = () if cpu else (claim.DEVICE_ID,)
        has_gpu = has_gpu or bool(gpu_devices)
        proposals[selected_id] = scheduling.StageProposal(
            proposal_id=f"template:{selected_id}", submitted_at=0.0,
            backend=selected.execution.backend,
            target_revision=unified_planner._target_digest(selected),
            alias_identity=selected.workload_signature,
            frontier_id=(unified_planner._target_digest(selected)
                         if "production" in selected.enrolled_as else None),
            production_frontier="production" in selected.enrolled_as,
            seed_id=(unified_planner._target_digest(selected)
                     if selected.seed_boost_units else None),
            stage_class="search", estimated_duration_seconds=max_stage,
            estimated_claims=scheduling.ResourceVector(fraction, gpu_devices, 0),
            eligible=True,
            eligibility_ref=f"resolved-campaign:{resolved.manifest_digest}:ready",
            reservation_kind=None, full_region=fraction == 1.0,
            compatibility_authority_refs=(), safe_chunking_declared=False)
    config = scheduling.SchedulerConfig(
        config_id=f"serial-derived:{resolved.manifest_digest}",
        max_stage_seconds=max_stage, noncoverage_slots=max(1, len(targets)),
        reservation_slots={}, reservation_shares={},
        campaign_attempt_cap=attempt_cap,
        campaign_charged_seconds_cap=max_stage * attempt_cap,
        seed_attempt_cap=attempt_cap, seed_charged_seconds_cap=max_stage * attempt_cap,
        capacity=scheduling.ResourceVector(1.0, (claim.DEVICE_ID,) if has_gpu else (), 0),
        weights_source="resolved-campaign seed provenance; fixed normal=1 seed=2",
        apportionment_rule="backend deficit over exact enrolled ready targets",
        adaptive_rule_id=None)
    return serial_scheduling.SerialSchedulerManifest.from_dict({
        "schema": serial_scheduling.MANIFEST_SCHEMA,
        "scheduler_id": f"serial:{resolved.manifest_digest}",
        "config": config.to_dict(),
        "targets": {key: value.to_dict() for key, value in proposals.items()}})


def _child_command(argv):
    return [sys.executable, "-m", "scripts.kernel_rnd.autokernel.loop.run", *argv]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target-args", type=Path, action="append")
    source.add_argument("--resolved-campaign", type=Path,
                        help="derive the roster from ready enrolled production/candidate targets")
    parser.add_argument("--owned-targets", type=Path,
                        help="target alias → original owned source/anchor/branch/request paths")
    parser.add_argument("--target-root", type=Path,
                        help="derived per-target store/lane roots (default: state-dir/targets)")
    parser.add_argument("--common-args", type=Path, help="optional shared actor/measurement argv JSON")
    parser.add_argument("--scheduler-manifest", type=Path,
                        help="closed resource-time budget and exact target proposals")
    parser.add_argument("--dry-run", action="store_true",
                        help="print inputs and run each existing owner dry-run; no execution or writes")
    parser.add_argument("--batch-iterations", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1,
                        help="0 schedules until its configured budget or STOP")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--control-listen", help="optional authenticated IPv4 loopback HOST:PORT")
    parser.add_argument("--control-origin", default=os.environ.get("AUTOKERNEL_TRUSTED_HUB_ORIGIN"),
                        help="exact trusted hub origin; token comes from AUTOKERNEL_CONTROL_TOKEN")
    args = parser.parse_args(argv)
    if args.batch_iterations <= 0 or args.rounds < 0:
        parser.error("batch iterations must be positive and rounds nonnegative")
    try:
        if args.control_listen:
            from . import campaign_service
            try:
                campaign_service.parse_listen(args.control_listen)
                campaign_service.validate_origin(args.control_origin)
            except campaign_service.ControlRefused as exc:
                raise SerialRefused(str(exc)) from exc
            if not os.environ.get("AUTOKERNEL_CONTROL_TOKEN"):
                raise SerialRefused("AUTOKERNEL_CONTROL_TOKEN is required with --control-listen")
        skipped = []
        child_prefix = ()
        if args.resolved_campaign:
            if args.owned_targets is None:
                raise SerialRefused("--resolved-campaign requires --owned-targets")
            from .serial_roster import build_targets
            targets, skipped, cpus = build_targets(args.resolved_campaign, args.owned_targets,
                target_root=args.target_root or args.state_dir / "targets", common_path=args.common_args,
                state_root=args.state_dir)
            targets = [_validate_target_args(row, owner_anchor_waiver=True) for row in targets]
            taskset = shutil.which("taskset")
            if taskset is None:
                raise SerialRefused("taskset is required to confine generated owned children")
            child_prefix = (taskset, "-c", ",".join(map(str, cpus)))
        else:
            if args.owned_targets or args.target_root or args.common_args:
                raise SerialRefused("roster options require --resolved-campaign")
            targets = [_target_args(path) for path in args.target_args]
        # No two configured targets may overwrite an active target's source,
        # build lanes or history. The owner never creates/repoints those roots.
        for flag in ("--worktree", "--store", "--worker-root", "--worker-build-root"):
            paths = [Path(option(row, flag)).resolve() for row in targets]
            overlaps = [(i, j, a, b) for i, a in enumerate(paths)
                        for j, b in enumerate(paths[i + 1:], start=i + 1)
                        if a == b or a in b.parents or b in a.parents]
            if flag == "--worktree":
                overlaps = [(i, j, a, b) for i, j, a, b in overlaps
                            if not (a == b and option(targets[i], "--experimental-branch",
                                                     champion.CANONICAL_BRANCH) ==
                                    option(targets[j], "--experimental-branch",
                                           champion.CANONICAL_BRANCH))]
            if overlaps:
                raise SerialRefused(f"target {flag} roots overlap")
        scheduler_manifest = None
        if args.scheduler_manifest is not None:
            if args.batch_iterations != 1:
                raise SerialRefused("scheduled serial mode requires one iteration per child")
            from . import serial_scheduling
            manifest_body, _manifest_sha = _json(args.scheduler_manifest, limit=256 * 1024)
            scheduler_manifest = serial_scheduling.SerialSchedulerManifest.from_dict(manifest_body)
            serial_scheduling.validate_target_bindings(
                scheduler_manifest, _scheduler_bindings(targets))
        elif args.resolved_campaign is not None:
            from . import serial_scheduling
            scheduler_manifest = _derived_scheduler_manifest(
                targets, args.resolved_campaign, args.rounds)
            serial_scheduling.validate_target_bindings(
                scheduler_manifest, _scheduler_bindings(targets))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"targets": targets, "skipped": skipped, "child_prefix": child_prefix,
                          "scheduler": (scheduler_manifest.to_dict()
                                        if scheduler_manifest is not None else None)}, indent=2))
        from . import run
        if scheduler_manifest is not None:
            from . import scheduling, serial_scheduling
            serial_scheduling.select_target(
                scheduler_manifest,
                scheduling.initial_state(scheduler_manifest.config,
                                         scheduler_manifest.scheduler_id),
                tuple(option(target, "--target-id") for target in targets), now=time.time(),
                stage_number=0)
        for target in targets:
            result = run.main([*target, "--dry-run"])
            if result:
                return result
        return 0
    for row in skipped:
        print(f"roster    {','.join(row['target_ids'])}: {row['reason']}", file=sys.stderr)
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "serial.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SerialRefused("serial session already has an owner") from exc
        return _drive(root, targets, args.batch_iterations, args.rounds, child_prefix=child_prefix,
                      scheduler_manifest=scheduler_manifest,
                      control_listen=args.control_listen, control_origin=args.control_origin)


def _source_owner_key(argv):
    identity = _selected_identity(argv)
    worktree = Path(option(argv, "--worktree")).resolve()
    done = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--path-format=absolute",
         "--git-common-dir"], capture_output=True, text=True, timeout=60)
    if done.returncode == 0:
        try:
            common = Path(done.stdout.strip()).resolve(strict=True)
            stat = common.stat()
            source_owner = {"common_git": str(common), "device": stat.st_dev,
                            "inode": stat.st_ino}
        except OSError:
            source_owner = {"unavailable_worktree": str(worktree)}
    else:
        source_owner = {"unavailable_worktree": str(worktree)}
    return _digest({"campaign_id": identity["campaign_id"],
                    "manifest_digest": identity["manifest_digest"],
                    "source_owner": source_owner})


def _source_result(state, target):
    """Recover old-key pointers and choose only a proven forward shared lineage."""
    key = _source_owner_key(target)
    direct = state["source_results"].get(key)
    if direct is not None:
        if not isinstance(direct, dict) or set(direct) != {"path", "sha256"}:
            raise SerialRefused("retained shared-source pointer is malformed")
        _body, sha = load_completed(Path(direct["path"]))
        if sha != direct["sha256"]:
            raise SerialRefused("retained shared-source result changed")
        return direct
    candidates = []
    identity = _selected_identity(target)
    for reference in state["source_results"].values():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise SerialRefused("retained shared-source pointer is malformed")
        body, sha = load_completed(Path(reference["path"]))
        if sha != reference["sha256"]:
            raise SerialRefused("retained shared-source result changed")
        selected = body.get("selected_target")
        lineage = body.get("source_lineage_keeps") or body.get("experimental_source_keeps")
        if (not isinstance(selected, dict) or not lineage
                or selected.get("campaign_id") != identity["campaign_id"]
                or selected.get("manifest_digest") != identity["manifest_digest"]):
            continue
        from . import surface_fold, surface_validation
        receipt = surface_fold.reopen_reference(lineage[-1])
        if body["current_anchor"]["commit"] != receipt.kept_commit:
            continue
        try:
            surface_validation.shared_git_commit(
                Path(option(target, "--worktree")), Path(receipt.repo), receipt.kept_commit)
        except surface_validation.SurfaceValidationRefused:
            continue
        candidates.append((dict(reference), body))
    if not candidates:
        return None
    chosen = next(((reference, body) for reference, body in candidates
                   if direct == reference), candidates[0])
    for candidate in candidates:
        if candidate[0] == chosen[0]:
            continue
        old = chosen[1]["current_anchor"]["commit"]
        new = candidate[1]["current_anchor"]["commit"]
        repo = option(target, "--worktree")
        if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", old, new],
                          capture_output=True, timeout=60).returncode == 0:
            chosen = candidate
        elif subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", new, old],
                            capture_output=True, timeout=60).returncode != 0 and direct is None:
            return None
    state["source_results"][key] = chosen[0]
    return chosen[0]


def _remember_source_result(state, body, target, reference):
    if not body.get("experimental_source_keeps"):
        return
    key = _source_owner_key(target)
    existing = _source_result(state, target)
    if existing is not None:
        prior, prior_sha = load_completed(Path(existing["path"]))
        if prior_sha != existing["sha256"]:
            raise SerialRefused("retained shared-source result changed")
        done = subprocess.run(
            ["git", "-C", option(target, "--worktree"), "merge-base", "--is-ancestor",
             prior["current_anchor"]["commit"], body["current_anchor"]["commit"]],
            capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            return
    state["source_results"][key] = dict(reference)


def _validation_subject(state, target, index, source_commit):
    prior_reference = state["last_results"].get(str(index))
    runtime_recipe = None
    if prior_reference is not None:
        prior, sha = load_completed(Path(prior_reference["path"]))
        if sha != prior_reference["sha256"]:
            raise SerialRefused("retained target continuation changed")
        runtime_recipe = prior.get("runtime_recipe_reference")
    return _digest({"source_commit": source_commit,
                    "target": _selected_identity(target),
                    "original_binding": input_binding(target),
                    "runtime_recipe_reference": runtime_recipe})


def _remember_source_validation(state, body, target, index, target_count):
    reference = body.get("source_validation")
    if reference is None:
        return
    from . import surface_validation
    row = surface_validation.reopen_reference(reference)
    if row["target"] != _selected_identity(target):
        raise SerialRefused("source validation target differs from original enrollment")
    key = _validation_subject(state, target, index, row["source_commit"])
    entry = state["source_validations"].setdefault(
        key, {"latest_reference": None, "disposition": None,
              "retry_after_search_count": 0,
              "attempts": 0, "history_digest": _digest([])})
    if (not isinstance(entry, dict)
            or set(entry) != {"latest_reference", "disposition",
                              "retry_after_search_count",
                              "attempts", "history_digest"}
            or type(entry["attempts"]) is not int or entry["attempts"] < 0):
        raise SerialRefused("source validation attempt history is malformed")
    if entry["latest_reference"] == reference:
        return
    entry["attempts"] += 1
    entry["history_digest"] = _digest({"prior": entry["history_digest"],
        "attempt": entry["attempts"], "reference": reference,
        "disposition": row["disposition"]})
    entry["latest_reference"] = dict(reference)
    entry["disposition"] = row["disposition"]
    search_count = state["source_search_counts"].get(str(index), 0)
    entry["retry_after_search_count"] = (
        search_count + 1 if row["disposition"] == "pending" else 0)


def _record_completed_stage(state, body, target, index, target_count):
    if body.get("source_validation") is not None:
        _remember_source_validation(state, body, target, index, target_count)
    else:
        key = str(index)
        state["source_search_counts"][key] = state["source_search_counts"].get(key, 0) + 1


def _validation_retry_due(entry, search_count):
    return (entry is None or
            (entry.get("disposition") == "pending"
             and search_count >= entry.get("retry_after_search_count", 0)))


def _pending_source_validations(state, targets):
    """Return exact target indexes due for an initial or bounded retry verdict."""
    pending = {}
    for index, target in enumerate(targets):
        reference = _source_result(state, target)
        if reference is None:
            continue
        body, sha = load_completed(Path(reference["path"]))
        if sha != reference["sha256"]:
            raise SerialRefused("retained shared-source continuation changed")
        lineage = body.get("source_lineage_keeps") or body.get("experimental_source_keeps")
        if not lineage:
            continue
        commit = body["current_anchor"]["commit"]
        subject = _validation_subject(state, target, index, commit)
        entry = state["source_validations"].get(subject)
        if _validation_retry_due(
                entry, state["source_search_counts"].get(str(index), 0)):
            pending[index] = subject
    return pending


def _required_source_validation(state, targets):
    """Fold exact current-tip rows for production and retained keep authors."""
    if not targets:
        return None
    source_reference = None
    for target in targets:
        source_reference = _source_result(state, target)
        if source_reference is not None:
            break
    if source_reference is None:
        return None
    source, sha = load_completed(Path(source_reference["path"]))
    if sha != source_reference["sha256"]:
        raise SerialRefused("retained shared-source continuation changed")
    references = (source.get("source_lineage_keeps")
                  or source.get("experimental_source_keeps") or ())
    if not references:
        return None
    from . import surface_fold, surface_validation
    receipts = [surface_fold.reopen_reference(reference) for reference in references]
    commit = source["current_anchor"]["commit"]
    if receipts[-1].kept_commit != commit:
        raise SerialRefused("required validation lineage is not the current source tip")
    authored = {receipt.selected_target["selected_id"] for receipt in receipts}
    intended = receipts[-1].selected_target["selected_id"]
    required = []
    for index, target in enumerate(targets):
        identity = _selected_identity(target)
        enrolled = identity["original_target"].get("enrolled_as", ())
        if "production" in enrolled or identity["selected_id"] in authored:
            required.append((index, target, identity))
    missing_authors = authored - {identity["selected_id"] for _, _, identity in required}
    if missing_authors:
        raise SerialRefused("retained keep author is absent from the owned target roster")
    rows, missing = [], []
    for index, target, identity in required:
        subject = _validation_subject(state, target, index, commit)
        entry = state["source_validations"].get(subject)
        reference = entry.get("latest_reference") if isinstance(entry, dict) else None
        if reference is None:
            missing.append(identity["selected_id"])
            continue
        row = surface_validation.reopen_reference(reference)
        expected_intended = identity["selected_id"] == intended
        if (row["source_commit"] != commit or row["target"] != identity
                or row.get("intended_target") is not expected_intended
                or row["disposition"] != entry.get("disposition")):
            raise SerialRefused("required source validation identity changed")
        rows.append({"selected_id": identity["selected_id"], "subject": subject,
                     "reference": dict(reference), "disposition": row["disposition"],
                     "intended_target": expected_intended})
    dispositions = {row["disposition"] for row in rows}
    disposition = ("pending" if missing or "pending" in dispositions else
                   "failed" if "failed" in dispositions else "passed")
    body = {"schema": "epyc.autokernel.required_source_validation.v1",
            "source_reference": dict(source_reference), "source_commit": commit,
            "required_target_ids": [identity["selected_id"]
                                    for _, _, identity in required],
            "intended_target_id": intended, "rows": rows,
            "missing_target_ids": missing, "disposition": disposition}
    body["aggregate_digest"] = _digest(body)
    return body


def _refresh_required_source_validation(state, targets):
    state["required_source_validation"] = _required_source_validation(state, targets)


def _batch_argv(original, prior, batch_iterations, directory, *, scheduler_selection=None,
                scope_preview=None, source_prior=None, recovery_reference=None,
                validate_source=False):
    child_argv = list(original)
    prior_body = None
    if prior is not None:
        prior_body, sha = load_resume(Path(prior["path"]), original)
        if sha != prior["sha256"]:
            raise SerialRefused("retained child result changed")
        child_argv = _without(child_argv, {"--cpu-calibrate-serving", "--gpu-calibrate-serving"})
        child_argv += ["--resume-run", prior["path"]]
        runtime_reference = prior_body.get("runtime_recipe_reference")
        if runtime_reference is not None:
            reference_path = Path(directory) / "runtime-recipe-reference.json"
            raw = json.dumps(runtime_reference, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            if reference_path.exists():
                if _read(reference_path, limit=4096) != raw:
                    raise SerialRefused("retained runtime recipe routing reference changed")
            else:
                from . import archive
                archive._retain_bytes(reference_path, raw)
            child_argv += ["--runtime-recipe-reference", str(reference_path.resolve())]
    same_prior_cross_checkout = False
    if source_prior is not None and source_prior == prior and prior_body is not None:
        lineage = (prior_body.get("source_lineage_keeps")
                   or prior_body.get("experimental_source_keeps") or ())
        if lineage:
            tip = surface_fold.reopen_reference(lineage[-1])
            same_prior_cross_checkout = (Path(tip.repo).resolve()
                                         != Path(prior_body["worktree"]).resolve())
    if source_prior is not None and (prior is None or source_prior != prior
                                     or same_prior_cross_checkout):
        _source_body, source_sha = load_completed(Path(source_prior["path"]))
        if source_sha != source_prior["sha256"]:
            raise SerialRefused("retained shared-source continuation changed")
        source_target = _source_body.get("selected_target")
        owns_source = (isinstance(source_target, dict)
                       and source_target.get("selected_id") == option(original, "--target-id"))
        authoring_ready = (prior_body is not None
                           and prior_body.get("source_validation") is not None)
        # The original owner resumes directly. Another target receives the pointer
        # only for validation or after its exact passed validation has bound a build
        # of this tip; source ownership then transfers without resetting either branch.
        if owns_source or validate_source or authoring_ready:
            child_argv += ["--source-anchor-continuation", source_prior["path"],
                           "--source-anchor-sha256", source_sha]
        if validate_source:
            child_argv.append("--validate-source-continuation")
    # Prospective common-scope selection remains inside these original child
    # arguments. Scheduler selection/claims must agree before this child launches.
    from . import cpu_screen
    child_argv, _screen_selection = cpu_screen.prepare_batch(
        child_argv, prior, directory, batch_iterations=batch_iterations, previewed=scope_preview)
    if recovery_reference is not None and _screen_selection["scope"] == "full":
        reference_path = Path(directory) / "runtime-recovery-reference.json"
        raw = json.dumps(recovery_reference, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        from . import archive
        archive._retain_bytes(reference_path, raw)
        child_argv += ["--runtime-recovery-reference", str(reference_path.resolve())]
    child_argv += ["--iterations", str(batch_iterations), "--out", str(directory)]
    if scheduler_selection is not None:
        child_argv += ["--scheduler-selection", str(Path(scheduler_selection).resolve())]
    return child_argv


def _original_child_terminal(active):
    """Read kernel identity only; never adopt, signal or grant from stale status."""
    pid = active.get("pid")
    if type(pid) is not int or pid < 1:
        raise SerialRefused("previous child PID was not captured; terminal ownership unresolved")
    original = active.get("process_identity")
    if original is not None and (
            not isinstance(original, dict) or set(original) != {"pid", "start_ticks", "boot_id"}
            or original["pid"] != pid or type(original["start_ticks"]) is not int
            or original["start_ticks"] < 0 or not isinstance(original["boot_id"], str)
            or not original["boot_id"]):
        raise SerialRefused("original child process identity is malformed")
    try:
        current = worker_lifecycle.process_identity(pid).to_dict()
    except worker_lifecycle.LifecycleRefused as exc:
        try:
            os.stat(f"/proc/{pid}")
        except FileNotFoundError:
            # Prove procfs is readable: an absent/unmounted /proc is not a dead PID.
            worker_lifecycle.process_identity(os.getpid())
            return "original_pid_absent; exit_status_unavailable"
        raise SerialRefused("previous child state unreadable; no relaunch") from exc
    if original is None:
        raise SerialRefused("legacy child PID still present without original start identity; no relaunch")
    if current != original:
        return "original_process_identity_no_longer_present; exit_status_unavailable"
    # A zombie is an exited original child, even if its reaper has not collected it.
    fd = os.open(f"/proc/{pid}/stat", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    tail = raw[raw.rfind(b")") + 1:].split()
    if len(raw) <= 4096 and tail and tail[0] in {b"Z", b"X"} \
            and worker_lifecycle.process_identity(pid).to_dict() == original:
        return "original_child_kernel_terminal; exit_status_unavailable"
    raise SerialRefused("previous original child is still live; no relaunch")


def _reconcile_completed(root, state, targets, batch_iterations):
    active = state["active"]
    if not isinstance(active, dict):
        raise SerialRefused("previous active batch is malformed")
    index, number = active.get("target_index"), state.get("next_batch")
    if type(index) is not int or not 0 <= index < len(targets) \
            or type(number) is not int or number < 0:
        raise SerialRefused("previous active batch index is malformed")
    directory = root / "batches" / f"batch-{number:06d}"
    original = targets[index]
    scheduled = active.get("scheduler_selection")
    scheduled_sha = active.get("scheduler_selection_sha256")
    if (scheduled is None) != (scheduled_sha is None):
        raise SerialRefused("previous active scheduler selection is incomplete")
    selection_path = directory / "scheduler-selection.json" if scheduled is not None else None
    if scheduled is not None:
        if not isinstance(scheduled, dict) or _digest(scheduled) != scheduled_sha:
            raise SerialRefused("previous active scheduler selection digest differs")
        selection_body, _selection_file_sha = _json(selection_path, limit=256 * 1024)
        if selection_body != scheduled:
            raise SerialRefused("previous scheduler selection file changed")
    argv = _batch_argv(
        original, state["last_results"].get(str(index)), batch_iterations,
        directory, scheduler_selection=selection_path,
        recovery_reference=state.get("runtime_recovery", {}).get(str(index)),
        source_prior=_source_result(state, original))
    expected = {"target_index": index, "selected_id": option(original, "--target-id"),
                "store": option(original, "--store"), "batch_dir": str(directory),
                "input_argv_sha256": _digest(argv)}
    if set(active) - {*expected, "pid", "process_identity", "scheduler_selection",
                      "scheduler_selection_sha256"} or any(
            active.get(key) != value for key, value in expected.items()):
        raise SerialRefused("previous active batch differs from original target/arguments")
    path = directory / "loop-continuation.json"
    if not path.exists() and scheduled is not None and option(original, "--cpu-serving-launch"):
        from . import runtime_recovery
        recovery = runtime_recovery.retain(directory, active, argv)
        if runtime_recovery.pending(recovery) is None:
            raise SerialRefused("failed child has no original recoverable runtime attempt")
        return {"target_index": index, "batch_number": number, "terminal": "failed",
            "result": None, "runtime_recovery": recovery,
            "process_terminal_basis": _original_child_terminal(active)}
    body, sha = load_completed(path, expected_argv=argv, expected_binding=input_binding(argv))
    identity = body["selected_target"]
    if not isinstance(identity, dict) or identity.get("selected_id") != active["selected_id"]:
        raise SerialRefused("previous terminal result belongs to another selected target")
    terminal_basis = _original_child_terminal(active)
    return {"target_index": index, "batch_number": number, "terminal": body["terminal"],
            "result": {"path": str(path), "sha256": sha}, "process_terminal_basis": terminal_basis}


def _selected_identity(argv):
    resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
    selected_id = option(argv, "--target-id")
    cpu = option(argv, "--cpu-serving-launch") is not None
    gpu = option(argv, "--gpu-serving-launch") is not None
    selected = legacy_targets.select_target(
        resolved, selected_id, cpu_serving=cpu,
        model=Path(option(argv, "--model")) if option(argv, "--model") else None)
    return {"campaign_id": resolved.campaign_id, "request_id": resolved.request_id,
            "manifest_digest": resolved.manifest_digest, "selected_id": selected_id,
            "scope": ("cpu_serving_selected_workload" if cpu else
                      "gpu_serving_selected_workload" if gpu else "legacy_gpu_screen"),
            "original_target": selected.to_dict()}


def _cost_body_scope(body, proposal, scheduler_state, *, preview=None, source_body=None):
    from . import serial_scheduling
    prospective = preview is not None
    geometry = (preview["scope"] if prospective else
                (body.get("cpu_screen") or {}).get("scope", "full"))
    if geometry == "full_confirmation":
        return None  # One retained candidate, not another ordinary search batch.
    argv = body["input_argv"]
    return serial_scheduling.cost_scope(
        binding=resume_binding(argv),
        anchor=(source_body or body)["current_anchor"], cor_anchor=body["cor_anchor"],
        runtime_recipe=body.get("runtime_recipe_reference"), geometry=geometry,
        preparation={"cpu_calibration": None if prospective else option(argv, "--cpu-calibrate-serving"),
                     "gpu_calibration": None if prospective else option(argv, "--gpu-calibrate-serving"),
                     "runtime_calibration": "--calibrate-runtime" in argv},
        proposal=proposal, state=scheduler_state)


def _cost_forecasts(state, manifest, scheduler_state, available, previews):
    """Read only original small continuations; absent/incompatible history keeps D."""
    from . import cpu_screen, serial_scheduling
    forecasts = {}
    if not state.get("cost_forecast"):
        return forecasts
    for index, original in available:
        selected_id = option(original, "--target-id")
        prior = state["last_results"].get(str(index))
        if prior is None or state["runtime_recovery"].get(str(index)) is not None:
            continue  # No guessed initial build identity or interrupted-stage forecast.
        try:
            body, sha = load_resume(Path(prior["path"]), original)
            if sha != prior["sha256"]:
                raise SerialRefused("cost forecast original continuation changed")
            source_body = None
            source = _source_result(state, original)
            if source is not None and source != prior:
                source_body, sha = load_completed(Path(source["path"]))
                if sha != source["sha256"]:
                    raise SerialRefused("cost forecast original source continuation changed")
            proposal = cpu_screen.scoped_proposal(manifest.proposals[selected_id], previews[selected_id])
            scope = _cost_body_scope(body, proposal, scheduler_state,
                                     preview=previews[selected_id], source_body=source_body)
            forecast = (serial_scheduling.duration_forecast(
                state["cost_forecast"], selected_id, scope, proposal=proposal,
                max_stage_seconds=manifest.config.max_stage_seconds) if scope is not None else None)
            if forecast is not None:
                forecasts[selected_id] = forecast
            state.get("cost_forecast_errors", {}).pop(selected_id, None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            state.setdefault("cost_forecast_errors", {})[selected_id] = str(exc)[:400]
    return forecasts


def _scheduled_account(state, manifest, active, body, batch_dir):
    from . import scheduling, serial_scheduling
    selection = scheduling.Selection.from_dict(active["scheduler_selection"])
    if active["scheduler_selection_sha256"] != selection.digest:
        raise SerialRefused("active selection digest differs from original selection")
    if body.get("schema") != CONTINUATION_SCHEMA_V2:
        raise SerialRefused("scheduled child lacks original held-resource evidence")
    receipts = serial_scheduling.reopen_held_receipts(
        batch_dir, body["held_claim_evidence"], selection=selection,
        target=body["selected_target"])
    scheduler_state = scheduling.SchedulerState.from_dict(state["scheduler_state"])
    outcome = serial_scheduling.one_iteration_outcome(
        body["terminal"], body["outcome_counts"])
    settled = scheduling.account_stage_components(
        manifest.config, scheduler_state, selection, receipts, outcome=outcome)
    # Only a NEW, completed measured search with unchanged source/runtime anchor
    # trains the forecast. A keep changes future setup; failures/invalids remain
    # charged above but cannot train successful duration from a truncated prefix.
    if (settled.campaign_attempts > scheduler_state.campaign_attempts
            and body["terminal"] == "complete"
            and body["outcome_counts"] in ({"measured_null": 1}, {"keep_candidate": 1})
            and selection.proposal.stage_class == "search"
            and settled.successor_fences == scheduler_state.successor_fences):
        selected_id = active["selected_id"]
        try:
            scope = _cost_body_scope(body, selection.proposal, scheduler_state)
            if scope is not None:
                state["cost_forecast"] = serial_scheduling.retain_cost_sample(
                    state.get("cost_forecast"), selected_id, scope, selection, receipts)
            state.get("cost_forecast_errors", {}).pop(selected_id, None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Optional scheduling feedback never undoes original held settlement.
            state.setdefault("cost_forecast_errors", {})[selected_id] = str(exc)[:400]
    return settled


def _scheduled_failure_account(state, manifest, active, batch_dir, original):
    from . import scheduling, serial_scheduling
    reference, _sha = _json(batch_dir / "loop-held-claims.json", limit=64 * 1024)
    selection = scheduling.Selection.from_dict(active["scheduler_selection"])
    if active["scheduler_selection_sha256"] != selection.digest:
        raise SerialRefused("active selection digest differs from original selection")
    receipts = serial_scheduling.reopen_held_receipts(
        batch_dir, reference, selection=selection, target=_selected_identity(original))
    return scheduling.account_stage_components(
        manifest.config, scheduling.SchedulerState.from_dict(state["scheduler_state"]),
        selection, receipts, outcome="failed")


def _drive(root, targets, batch_iterations, rounds, *, child_prefix=(),
           scheduler_manifest=None, control_listen=None, control_origin=None):
    config = _digest({"targets": targets, "batch_iterations": batch_iterations, "rounds": rounds,
                      **({"child_prefix": list(child_prefix)} if child_prefix else {}),
                      **({"scheduler_manifest": scheduler_manifest.digest}
                         if scheduler_manifest is not None else {})})
    state_path = root / "serial-state.json"
    if state_path.exists():
        state, _sha = _json(state_path)
        if state.get("schema") != SERIAL_SCHEMA or state.get("config_digest") != config:
            raise SerialRefused("serial state belongs to different inputs")
    else:
        state = {"schema": SERIAL_SCHEMA, "config_digest": config, "next_batch": 0,
                 "active": None, "last_results": {}, "source_results": {},
                 "source_validations": {},
                 "required_source_validation": None,
                 "source_search_counts": {},
                 "failed_targets": {}}
        if scheduler_manifest is not None:
            from . import scheduling
            state["scheduler_state"] = scheduling.initial_state(
                scheduler_manifest.config, scheduler_manifest.scheduler_id).to_dict()
    state.setdefault("source_results", {})
    state.setdefault("runtime_recovery", {})
    state.setdefault("source_validations", {})
    state.setdefault("required_source_validation", None)
    state.setdefault("source_search_counts", {})
    instruments = {option(row, "--target-id"): option(row, "--serving-instrument", "legacy_v1")
                   for row in targets}
    if state.get("serving_instruments", instruments) != instruments:
        raise SerialRefused("serial serving instrument selection differs from original target arguments")
    state["serving_instruments"] = instruments
    if scheduler_manifest is not None:
        from . import scheduling
        scheduling.SchedulerState.from_dict(state.get("scheduler_state"))
    elif "scheduler_state" in state:
        raise SerialRefused("unscheduled session contains scheduler state")
    # Diagnostic configuration only. The original digest above still controls
    # restart; old state files acquire these values from that same checked input.
    state.update(target_count=len(targets), rounds=rounds, batch_iterations=batch_iterations)
    stop = threading.Event()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def request_stop(_sig, _frame):
        stop.set()
        (root / "STOP").touch(exist_ok=True)

    def stopped():
        return stop.is_set() or (root / "STOP").exists()

    def save():
        status.write_json(root, state_path.name, state)

    control = service = None

    def publish(phase, active=None, reason=None):
        if isinstance(active, dict):
            active = {key: value for key, value in active.items()
                      if key not in {"scheduler_selection", "scheduler_selection_sha256"}}
        status.write(root, state=phase, epoch=config, campaign_id="legacy-serial",
                     anchor_commit="", surface="serial_targets", pairs=0, noise_floor_pct=None,
                     target=active, step=reason or "serial routing only; detailed original status stays in target store",
                     routing={"target_count": len(targets), "rounds": rounds,
                              "batch_iterations": batch_iterations, "next_batch": state["next_batch"],
                              "stop_requested": stopped(),
                              "failed_targets": {key: value[:240]
                                                 for key, value in state["failed_targets"].items()}},
                     serial_control=(control.pump(
                         state.get("active"), terminal=phase in {"complete", "failed"},
                         stopped=stopped(), failed=phase == "failed" or bool(state["failed_targets"]))
                         if control is not None else None),
                     stale_after_s=180)

    def retain_recovery(active, argv, directory):
        # Optional for ordinary source work. Failure remains a visible diagnostic,
        # not a new launch prerequisite or a fabricated success/held interval.
        if scheduler_manifest is None or option(argv, "--cpu-serving-launch") is None:
            return
        from . import runtime_recovery
        key = str(active["target_index"])
        try:
            reference = runtime_recovery.retain(directory, active, argv)
            interrupted = runtime_recovery.pending(reference)
            previous_pending = (runtime_recovery.pending(state["runtime_recovery"][key])
                                if key in state["runtime_recovery"] else None)
        except (OSError, ValueError) as exc:
            state.setdefault("runtime_recovery_errors", {})[key] = str(exc)[:400]
        else:
            if interrupted is not None:
                state["runtime_recovery"][key] = reference
                state.get("runtime_recovery_errors", {}).pop(key, None)
                # Reeligibility requires original pending runtime work, not merely
                # a closed CPU claim after an unrelated deterministic failure.
                state["failed_targets"].pop(key, None)
            elif key in state["runtime_recovery"] and previous_pending is None:
                state["runtime_recovery"].pop(key)

    for sig in handlers:
        signal.signal(sig, request_stop)
    try:
        if control_listen or "control" in state:
            from .serial_control import SerialControl, SerialHTTPService
            control = SerialControl(state, save, request_stop, config_digest=config)
            if (state["control"]["desired_state"] == "paused"
                    and not control_listen and not stopped()):
                raise SerialRefused("retained serial pause requires --control-listen and token to resume; "
                                    "or use the original STOP file to drain without resuming")
            if control_listen:
                service = SerialHTTPService(control, control_listen,
                    os.environ.get("AUTOKERNEL_CONTROL_TOKEN"), allowed_origin=control_origin)
            control.pump(state.get("active"), stopped=stopped())
            if service is not None:
                service.start()
        if state.get("active") is not None:
            recovered_active = dict(state["active"])
            recovered = _reconcile_completed(root, state, targets, batch_iterations)
            recovered_dir = root / "batches" / f"batch-{recovered['batch_number']:06d}"
            expected_recovered_argv = _batch_argv(
                targets[recovered["target_index"]],
                state["last_results"].get(str(recovered["target_index"])),
                batch_iterations, recovered_dir,
                recovery_reference=state["runtime_recovery"].get(str(recovered["target_index"])),
                source_prior=_source_result(
                    state, targets[recovered["target_index"]]),
                validate_source=(scheduler_manifest is not None and
                                 recovered["target_index"] in
                                 _pending_source_validations(state, targets)),
                scheduler_selection=(recovered_dir / "scheduler-selection.json"
                                     if scheduler_manifest is not None else None))
            if recovered["result"] is None:
                state["scheduler_state"] = _scheduled_failure_account(
                    state, scheduler_manifest, recovered_active, recovered_dir,
                    targets[recovered["target_index"]]).to_dict()
                state["runtime_recovery"][str(recovered["target_index"])] = recovered["runtime_recovery"]
            else:
                body, _sha = load_completed(
                    Path(recovered["result"]["path"]),
                    expected_binding=input_binding(expected_recovered_argv))
                if scheduler_manifest is not None:
                    state["scheduler_state"] = _scheduled_account(
                        state, scheduler_manifest, recovered_active, body,
                        recovered_dir).to_dict()
                _record_completed_stage(state, body,
                    targets[recovered["target_index"]], recovered["target_index"], len(targets))
                state["last_results"][str(recovered["target_index"])] = recovered["result"]
                _remember_source_result(state, body,
                    targets[recovered["target_index"]], recovered["result"])
                _refresh_required_source_validation(state, targets)
                retain_recovery(recovered_active, expected_recovered_argv, recovered_dir)
            state["last_reconciliation"] = recovered
            state["active"] = None
            state["next_batch"] += 1
            if recovered["terminal"] == "stopped":
                request_stop(None, None)
            save()  # original result and progress committed before selecting another batch
        while not stopped() and (rounds == 0 or state["next_batch"] < rounds * len(targets)):
            if control is not None:
                control.pump(stopped=stopped())
                paused_heartbeat = 0.0
                while state["control"]["desired_state"] == "paused" and not stopped():
                    if time.monotonic() >= paused_heartbeat:
                        publish("running", reason="paused between batches; no child or claims held")
                        paused_heartbeat = time.monotonic() + 30
                    time.sleep(.5)
                    previous = control.publish_snapshot()
                    control.pump(stopped=stopped())
                    if control.publish_snapshot() != previous:
                        paused_heartbeat = 0.0
                if stopped():
                    break
            if len(state["failed_targets"]) == len(targets):
                publish("failed")
                return 1  # Continuous mode must not spin over a failed roster.
            number = state["next_batch"]
            selection = None
            previews = None
            validation_targets = (_pending_source_validations(state, targets)
                                  if scheduler_manifest is not None else {})
            if scheduler_manifest is not None:
                from . import cpu_screen, scheduling, serial_scheduling
                available = tuple((i, target) for i, target in enumerate(targets)
                                  if str(i) not in state["failed_targets"])
                source_state = scheduling.SchedulerState.from_dict(state["scheduler_state"])
                previews, scope_debt, ready = {}, {}, []
                collisions = cpu_screen.pending_collisions(targets, state["last_results"])
                for i, target in available:
                    selected_id = option(target, "--target-id")
                    if i in collisions:
                        scope_debt[selected_id] = collisions[i]
                        continue
                    preview = ({"scope": "full", "candidate": None,
                                "reason": "whole-source validation uses the full enrolled target"}
                               if i in validation_targets else cpu_screen.preview_batch(
                                   target, state["last_results"].get(str(i)),
                                   batch_iterations=batch_iterations))
                    recovery_ref = state["runtime_recovery"].get(str(i))
                    if recovery_ref is not None:
                        from . import runtime_recovery
                        try:
                            interrupted = runtime_recovery.pending(recovery_ref)
                        except (OSError, ValueError) as exc:
                            state.setdefault("runtime_recovery_errors", {})[str(i)] = str(exc)[:400]
                            interrupted = None
                        if interrupted is not None and preview["scope"] != "full_confirmation":
                            preview = {"scope": "full", "candidate": None,
                                       "reason": "original pending full-target runtime pair"}
                    proposal = cpu_screen.scoped_proposal(scheduler_manifest.proposals[selected_id], preview)
                    debt = cpu_screen.confirmation_debt(
                        scheduler_manifest.config, source_state, proposal, preview)
                    if debt:
                        scope_debt[selected_id] = debt
                    else:
                        ready.append((i, target))
                        previews[selected_id] = preview
                state["scope_debt"] = scope_debt
                available = tuple(ready)
                if not available and scope_debt:
                    save()
                    publish("failed", reason="; ".join(f"{key}: {value}" for key, value in scope_debt.items()))
                    return 1  # Retain pending refs; no busy spin or new same-target search.
                scheduler_state, selection, available_index = serial_scheduling.select_target(
                    scheduler_manifest,
                    source_state,
                    tuple(option(target, "--target-id") for _i, target in available),
                    now=time.time(), stage_number=number, scope_previews=previews,
                    validation_ids=frozenset(option(target, "--target-id")
                        for i, target in available if i in validation_targets),
                    duration_forecasts=_cost_forecasts(
                        state, scheduler_manifest, source_state, available, previews))
                if available_index < 0:
                    state["scheduler_state"] = scheduler_state.to_dict()
                    save()
                    publish("complete")
                    return 0
                index = available[available_index][0]
                state["scheduler_state"] = scheduler_state.to_dict()
            else:
                index = number % len(targets)
            key = str(index)
            if key in state["failed_targets"]:
                state["next_batch"] += 1
                save()
                continue
            original = targets[index]
            directory = root / "batches" / f"batch-{number:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            prior = state["last_results"].get(key)
            selection_path = None
            if selection is not None:
                selection_path = directory / "scheduler-selection.json"
                status.write_json(directory, selection_path.name, selection.to_dict(),
                                  prefix=".scheduler-selection-")
            child_argv = _batch_argv(original, prior, batch_iterations, directory,
                                     recovery_reference=state["runtime_recovery"].get(key),
                                     scheduler_selection=selection_path,
                                     scope_preview=(previews[option(original, "--target-id")]
                                                    if previews is not None else None),
                                     source_prior=_source_result(state, original),
                                     validate_source=(scheduler_manifest is not None
                                                      and index in validation_targets))
            expected_binding = input_binding(child_argv)
            active = {"target_index": index, "selected_id": option(original, "--target-id"),
                      "store": option(original, "--store"), "batch_dir": str(directory),
                      "input_argv_sha256": _digest(child_argv), "pid": None}
            if selection is not None:
                active.update(scheduler_selection=selection.to_dict(),
                              scheduler_selection_sha256=selection.digest)
            state["active"] = active
            save()
            publish("starting", active)
            process = None
            try:
                with (directory / "stdout.log").open("xb") as stdout, \
                        (directory / "stderr.log").open("xb") as stderr:
                    process = subprocess.Popen([*child_prefix, *_child_command(child_argv)], stdout=stdout, stderr=stderr,
                                               cwd=Path(__file__).resolve().parents[4])
                    active["pid"] = process.pid
                    try:
                        active["process_identity"] = worker_lifecycle.process_identity(process.pid).to_dict()
                    except worker_lifecycle.LifecycleRefused:
                        # The original Popen still owns cleanup. Missing identity
                        # cannot permit later adoption/signaling of a reused PID.
                        pass
                    save()
                    publish("running", active)
                    sent = False
                    heartbeat_at = time.monotonic() + 30
                    while process.poll() is None:
                        if control is not None:
                            previous = control.publish_snapshot()
                            if control.pump(active, stopped=stopped()) != previous:
                                publish("running", active)
                        if stopped() and not sent:
                            process.send_signal(signal.SIGTERM)  # This captured child only; it drains its tail.
                            sent = True
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            pass
                        if time.monotonic() >= heartbeat_at:
                            publish("running", active)
                            heartbeat_at = time.monotonic() + 30
                    if process.returncode != 0:
                        if scheduler_manifest is not None:
                            state["scheduler_state"] = _scheduled_failure_account(
                                state, scheduler_manifest, active, directory, original).to_dict()
                        raise SerialRefused(f"child exited {process.returncode}; see retained logs")
                result_path = directory / "loop-continuation.json"
                body, sha = load_completed(result_path, expected_argv=child_argv,
                                           expected_binding=expected_binding)
                identity = body["selected_target"]
                if not isinstance(identity, dict) or identity.get("selected_id") != active["selected_id"]:
                    raise SerialRefused("child terminal belongs to another selected target")
                _record_completed_stage(state, body, original, index, len(targets))
                state["last_results"][key] = {"path": str(result_path), "sha256": sha}
                _remember_source_result(state, body, original,
                    {"path": str(result_path), "sha256": sha})
                _refresh_required_source_validation(state, targets)
                if scheduler_manifest is not None:
                    state["scheduler_state"] = _scheduled_account(
                        state, scheduler_manifest, active, body, directory).to_dict()
                if body["terminal"] == "stopped":
                    request_stop(None, None)
            except (OSError, ValueError) as exc:
                state["failed_targets"][key] = f"{type(exc).__name__}: {exc}"
            finally:
                # Enrolled immediately after Popen: even publication/reader failure
                # cannot abandon a child still holding the legacy loop's claims.
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait()  # The existing owner drains; never kill by name/group.
            if process is not None:
                retain_recovery(active, child_argv, directory)
            state["active"] = None
            state["next_batch"] += 1
            save()
        publish("complete", {"stop_requested": stopped(), "failed_targets": state["failed_targets"]})
        return 1 if state["failed_targets"] else 0
    finally:
        try:
            if service is not None:
                service.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
