"""Original serial teardown references, not resource grants or admission evidence."""
from pathlib import Path
import hashlib
import json
import threading

from . import archive, serial_run as sr, serial_scheduling, scheduling
from .claim import HeldCpuClaim
from .measurement_capture import ArtifactStore, _plain

SCHEMA = "epyc.autokernel.direct_runtime_recovery.v1"


def source_identity():
    from . import lifecycle_observation as lo
    return {"module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "loaded": [lo.callable_identity(fn) for fn in (holder_identity, _verify, reopen,
                pending, replacement, replace_window, sr._original_child_terminal,
                serial_scheduling.reopen_held_receipts)]}


def holder_identity(holder):
    if type(holder) is not HeldCpuClaim:
        raise sr.SerialRefused("recovery requires the original acquired CPU context")
    return {"context_id": holder._context_id, "domain": dict(holder._domain)}


def _verify(body):
    if not isinstance(body, dict) or set(body) != {
            "schema", "batch_dir", "active", "input_argv", "binding", "target",
            "held_claim_evidence", "interruption"} or body["schema"] != SCHEMA:
        raise sr.SerialRefused("runtime recovery reference shape differs")
    active, argv, directory = body["active"], body["input_argv"], Path(body["batch_dir"])
    if not isinstance(active, dict) or set(active) != {"target_index", "selected_id", "store",
            "batch_dir", "input_argv_sha256", "pid", "process_identity", "scheduler_selection",
            "scheduler_selection_sha256"}:
        raise sr.SerialRefused("runtime recovery original active shape differs")
    if not isinstance(argv, list) or not all(type(value) is str for value in argv):
        raise sr.SerialRefused("runtime recovery original arguments are malformed")
    target = sr._selected_identity(argv)
    if (target != body["target"] or target["scope"] not in {
            "cpu_serving_selected_workload", "gpu_serving_selected_workload"}
            or body["binding"] != sr.input_binding(argv)
            or active.get("input_argv_sha256") != sr._digest(argv)
            or active.get("selected_id") != target["selected_id"]
            or active.get("store") != sr.option(argv, "--store")
            or Path(active.get("batch_dir", "")) != directory
            or Path(sr.option(argv, "--out", "")) != directory):
        raise sr.SerialRefused("runtime recovery differs from original target/arguments")
    sr._original_child_terminal(active)  # Neither age nor a parsed flag proves exit.
    selection = scheduling.Selection.from_dict(active.get("scheduler_selection"))
    if active.get("scheduler_selection_sha256") != selection.digest:
        raise sr.SerialRefused("runtime recovery original selection differs")
    serial_scheduling.reopen_held_receipts(directory, body["held_claim_evidence"],
        selection=selection, target=target)
    reference = body["held_claim_evidence"]["evidence"]
    store = ArtifactStore(directory / "held-claim-artifacts")
    try:
        intervals = _plain(store.read(reference["locator"], reference["sha256"]))
    finally:
        store.close()
    cpu = next(row for row in intervals["components"] if row["device_id"] == "cpu")
    gpu = None
    if target["scope"] == "gpu_serving_selected_workload":
        from .claim import DEVICE_ID
        gpu = next((row for row in intervals["components"] if row["device_id"] == DEVICE_ID), None)
        if gpu is None or gpu["domain"] != cpu["domain"]:
            raise sr.SerialRefused("GPU runtime recovery lacks both original component contexts")
    identity = {"pid": cpu["domain"]["pid"],
        "start_ticks": cpu["domain"]["process_start_ticks"], "boot_id": cpu["domain"]["boot_id"]}
    if active.get("process_identity") != identity:
        raise sr.SerialRefused("runtime recovery lacks the original child/held-owner identity join")
    if body["interruption"] is not None:
        original_store = ArtifactStore(Path(sr.option(argv, "--store")) / "runtime-preparation")
        try:
            reference = body["interruption"]
            interruption = _plain(original_store.read(reference["locator"], reference["sha256"]))
        finally:
            original_store.close()
        if (interruption.get("schema") != "epyc.autokernel.direct_runtime_interruption.v1"
                or interruption.get("holder") != {"context_id": cpu["context_id"], "domain": cpu["domain"]}
                or interruption.get("reason") not in {"between_launch_budget", "terminal_invalid_arm"}):
            raise sr.SerialRefused("original runtime interruption/holder differs")
        if gpu is not None and interruption.get("gpu_holder") != {
                "context_id": gpu["context_id"], "domain": gpu["domain"]}:
            raise sr.SerialRefused("original runtime interruption/GPU holder differs")
    return {**cpu, **({"gpu_component": gpu} if gpu is not None else {})}


def retain(directory, active, argv):
    """Called after original child terminal; no successful iteration is inferred."""
    directory = Path(directory)
    held, _sha = sr._json(directory / "loop-held-claims.json", limit=64 * 1024)
    marker = directory / "loop-runtime-interruption.json"
    interruption = sr._json(marker, limit=4096)[0] if marker.exists() else None
    body = {"schema": SCHEMA, "batch_dir": str(directory), "active": dict(active),
        "input_argv": list(argv), "binding": sr.input_binding(argv),
        "target": sr._selected_identity(argv), "held_claim_evidence": held,
        "interruption": interruption}
    _verify(body)
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    path = directory / "runtime-recovery.json"
    archive._retain_bytes(path, raw)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def reopen(reference, *, current_argv=None):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise sr.SerialRefused("runtime recovery locator is malformed")
    body, sha = sr._json(Path(reference["path"]), limit=256 * 1024)
    if sha != reference["sha256"]:
        raise sr.SerialRefused("original runtime recovery bytes changed")
    cpu = _verify(body)
    if current_argv is not None and (
            sr._selected_identity(current_argv) != body["target"]
            or Path(sr.option(current_argv, "--store")).resolve()
            != Path(sr.option(body["input_argv"], "--store")).resolve()):
        raise sr.SerialRefused("runtime recovery belongs to another target/store")
    return cpu


def pending(reference):
    """Read the same original allocated attempt, not a newly inferred hypothesis."""
    reopen(reference)
    body, _sha = sr._json(Path(reference["path"]), limit=256 * 1024)
    ref = body["interruption"]
    if ref is None:
        return None
    root = Path(sr.option(body["input_argv"], "--store")) / "runtime-preparation"
    store = ArtifactStore(root)
    try:
        row = _plain(store.read(ref["locator"], ref["sha256"]))
    finally:
        store.close()
    if (type(row.get("scope")) is not str or len(row["scope"]) != 64
            or any(char not in "0123456789abcdef" for char in row["scope"])
            or row["state_name"] != f"runtime-selection-{row['scope']}.json"):
        raise sr.SerialRefused("interrupted runtime state locator differs")
    state, _sha = sr._json(root / row["state_name"], limit=128 * 1024)
    index = row["index"]
    if state.get("scope") != row["scope"] or type(index) is not int or not 0 <= index < len(state["attempts"]):
        raise sr.SerialRefused("interrupted runtime allocation is missing")
    attempt = state["attempts"][index]
    if attempt["pair"] != row["pair"] or attempt["candidate_id"] != row["candidate_id"]:
        raise sr.SerialRefused("interrupted runtime pair/allocation changed")
    return row if attempt["result"] is None else None


def replacement(*, reference, original_holder, new_holder):
    """Join released old ownership to an ACTUAL newly acquired context."""
    old = reopen(reference)
    if original_holder != {"context_id": old["context_id"], "domain": old["domain"]}:
        raise sr.SerialRefused("interrupted window belongs to another original holder")
    new = holder_identity(new_holder)
    if new == original_holder or new_holder.observe()["status"] != "held":
        raise sr.SerialRefused("replacement needs a different currently held original context")
    if (new["domain"]["boot_id"] == old["domain"]["boot_id"]
            and new_holder._started_at < old["ended_at"]):
        raise sr.SerialRefused("replacement holder overlaps the original interval")
    return {"recovery": dict(reference), "holder": new}


def replace_window(window, reference):
    # A pending unknown launch might have an orphan server. Released locks and
    # a dead parent do not prove that server's teardown, so never clear it here.
    if window.pending is not None or any(
            row.get("terminal_invalid") is not True for row in window.failures):
        from .loop import RunAborted
        raise RunAborted("interrupted launch cleanup remains unresolved; no replacement or fallback measurement")
    declaration = _plain(window.store.read(window.declaration.locator, window.declaration.sha256))
    binding = replacement(reference=reference, original_holder=declaration.get("holder"),
                          new_holder=window.held_claim)
    if window.pair.anchor.backend == "gpu":
        old = reopen(reference).get("gpu_component")
        original = declaration.get("gpu_holder")
        new_gpu = window.gpu_claim
        if (old is None or original != {"context_id": old["context_id"], "domain": old["domain"]}
                or new_gpu is None or holder_identity(new_gpu) == original
                or new_gpu.observe()["status"] != "held"
                or new_gpu._domain != window.held_claim._domain
                or (new_gpu._domain["boot_id"] == old["domain"]["boot_id"]
                    and new_gpu._started_at < old["ended_at"])):
            raise sr.SerialRefused("GPU replacement needs its released old and actual new component contexts")
        binding["gpu_holder"] = holder_identity(new_gpu)
    state, _sha = sr._json(window.store.root / window.state_name,
                          limit=16_384 + 4096 * (3 * window.maximum))
    checkpoint = window.store.write("interrupted-direct-pair-window", state)
    return {**binding, "checkpoint": checkpoint.to_dict()}


class PendingPlanner:
    """One existing pool draw consumes the original pair, without another actor call."""
    def __init__(self, ordinary, shared):
        self.ordinary, self.shared = ordinary, shared

    @staticmethod
    def slot(pair):
        return [threading.Lock(), pair]

    def propose(self, context):
        with self.shared[0]:
            pair, self.shared[1] = self.shared[1], None
        if pair is None:
            return self.ordinary.propose(context)
        from .loop import Hypothesis
        return Hypothesis(pair.dimension.dimension_id,
            "Resume the original interrupted runtime treatment",
            "The original calibrated comparison does not admit an improvement",
            "runtime", pair.dimension.kind, runtime_pair=pair)

    def author(self, hypothesis, context):
        return self.ordinary.author(hypothesis, context)
