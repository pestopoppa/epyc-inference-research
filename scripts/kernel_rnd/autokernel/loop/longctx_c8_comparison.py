"""SL-4 opt-in GPU np8 ~32k-token long-context cell: matched C8 full-prompt comparison.

OPT-IN: nothing in the loop calls this module; a caller (a runner holding a GPU mi210
region) must invoke `calibrate` then `compare` explicitly. Its floors live under their
OWN key, ``<root>/longctx-c8-floors/<recipe_hash>/<anchor execution digest>/``, never the
recipe's ordinary serving floor and never the CPU slot-restore (`longctx`) surface.
For a speculative recipe `serving.write_floor` also files the companion steps/s floor
there and `serving.compare` gates on it (SL-2) through the sealed `floor_record`.

The caller owns GPU admission and output retention.  This adapter refuses a floor
from any other request set, recipe, or anchor execution; it never treats an
uncalibrated comparison as evidence.  Workload/tokenizer validation belongs to
``longctx_c8`` and happens before any server is launched. The measured unit is the
ordinary serving path carrying eight complete 31–33k-token prompt requests from the
frozen manifest. It does not use the CPU-only ``longctx.SurfaceLaunch`` restored-slot
protocol; that is a separate measurement surface and does not represent this workload.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from . import longctx_c8, serving


def _validated(anchor, manifest, proof, texts, encode, tokenizer_sha256, pairs):
    if anchor.backend != "gpu" or anchor.template.np != 8:
        raise ValueError("long-context C8 comparison requires a GPU np=8 launch")
    if type(pairs) is not int or not 1 <= pairs <= 64:
        raise ValueError("matched pair count must be in 1..64")
    identity = anchor.execution_digest
    if (not isinstance(identity, str) or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)):
        raise ValueError("anchor execution identity is missing")
    frozen = longctx_c8.validate_manifest(anchor.template, manifest, proof, texts,
                                           encode, tokenizer_sha256)
    requests = frozen.requests(tuple(p.prompt_id for p in frozen.prompts), anchor.template)
    return requests, identity


FLOOR_STORE = "longctx-c8-floors"


def _store(root, anchor, identity):
    return Path(root) / FLOOR_STORE / anchor.template.recipe_hash / identity


@contextmanager
def _exclusive_floor_lock(path):
    """Hold a process-wide, nonblocking lock for one floor calibration target.

    Keep the lock file in place permanently: unlinking a locked inode can let a
    later process create and lock a different inode at the same path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise serving.ServingFloorMismatch(
                "C8 floor calibration is already in progress for this target") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def calibrate(root, anchor, manifest, proof, texts, encode, tokenizer_sha256, *, pairs,
              samples=serving.MATCHED_CALIBRATION_PAIRS):
    """Record an A/A floor for the full frozen long-prompt requests and anchor."""
    requests, identity = _validated(anchor, manifest, proof, texts, encode,
                                    tokenizer_sha256, pairs)
    store = _store(root, anchor, identity)
    path = serving.floor_path(store, anchor.template, frozen_requests=requests,
                             instrument=serving.MATCHED_INSTRUMENT, pairs=pairs)
    lock_path = path.with_name(path.name + ".lock")
    with _exclusive_floor_lock(lock_path):
        # Check only while holding the same per-target lock used by every calibrator.
        # Holding it through write_floor closes the check/calibrate/write overwrite race.
        if path.exists():
            raise serving.ServingFloorMismatch("refusing to overwrite an existing C8 floor")
        row = serving.calibrate_floor(anchor.template, Path(anchor.build_dir), samples=samples,
                                      port=anchor.port, resolved_recipe=anchor,
                                      frozen_requests=requests,
                                      instrument=serving.MATCHED_INSTRUMENT, pairs=pairs)
        serving._validate_matched_floor(row, anchor.template, requests, pairs, resolved=anchor)
        baseline = row.get("baseline_resolved_recipe")
        if not isinstance(baseline, dict) or baseline.get("execution_digest") != identity:
            raise serving.ServingFloorMismatch("C8 calibration anchor identity differs")
        return serving.write_floor(store, anchor.template, row, frozen_requests=requests,
                                   unit=serving.CALIBRATION_UNIT,
                                   instrument=serving.MATCHED_INSTRUMENT, pairs=pairs)


def compare(root, anchor, candidate, manifest, proof, texts, encode, tokenizer_sha256, *, pairs):
    """Compare full frozen long-prompt requests in fresh server processes against their floor."""
    requests, identity = _validated(anchor, manifest, proof, texts, encode,
                                    tokenizer_sha256, pairs)
    if candidate.backend != "gpu" or candidate.template.np != 8:
        raise ValueError("candidate must use a GPU np=8 launch")
    store = _store(root, anchor, identity)
    reading = serving.load_floor(store, anchor.template, frozen_requests=requests,
                                 instrument=serving.MATCHED_INSTRUMENT, pairs=pairs)
    if not reading.verified or reading.floor_pct is None:
        raise serving.ServingFloorMismatch("C8 comparison requires its own calibrated floor")
    serving._validate_matched_floor(reading.row, anchor.template, requests, pairs, resolved=anchor)
    baseline = reading.row.get("baseline_resolved_recipe")
    if not isinstance(baseline, dict) or baseline.get("execution_digest") != identity:
        raise serving.ServingFloorMismatch("C8 floor anchor execution identity differs")
    return serving.compare(anchor.template, Path(anchor.build_dir), Path(candidate.build_dir),
                           pairs=pairs, floor_pct=reading.floor_pct, floor_unit=reading.unit,
                           port=anchor.port, anchor_resolved_recipe=anchor,
                           candidate_resolved_recipe=candidate, frozen_requests=requests,
                           floor_request_digest=reading.request_digest,
                           instrument=serving.MATCHED_INSTRUMENT, floor_record=reading.row)
