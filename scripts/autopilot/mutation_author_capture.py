"""Default-off, best-effort native mutation-author context metadata. Never logs prompt text."""
from __future__ import annotations
import fcntl
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
FLAG = "AUTOPILOT_AUTHOR_CONTEXT_CAPTURE"


class AuthorCaptureBusyError(BlockingIOError):
    """The best-effort journal append is busy; never wait before the author call."""


def capture_enabled() -> bool:
    return os.environ.get(FLAG) == "1"


def author_capture_kwargs(action: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Dispatch-bound identities only; absent/wrong-trial pins remain unknown."""
    if not capture_enabled():
        return {}
    try:
        directory = ctx.journal.journal_dir
        if not isinstance(directory, (str, Path)) or not str(directory):
            raise ValueError("native journal directory absent")
        trial = ctx.state.get("trial_counter")
        marker = ctx.state.get("in_flight_trial")
        manifest = None
        if (type(trial) is int and isinstance(marker, dict)
                and type(marker.get("trial_id")) is int and marker["trial_id"] == trial):
            native = marker.get("run_manifest")
            if isinstance(native, dict):
                manifest = native
        return {"author_capture_context": {
            "trial_id": trial if type(trial) is int else None,
            "action_type": action.get("type") if type(action.get("type")) is str else None,
            "source_pins": manifest.get("sources") if manifest else None,
            "run_manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
            "journal_dir": str(directory),
        }}
    except Exception as exc:
        log.warning("Author context identity capture unavailable (%s)", type(exc).__name__)
        return {}


def capture_author_context(prompt: str, *, operator: str, target: str,
                           inputs: dict[str, str], context: dict[str, Any] | None,
                           parameters: dict[str, Any]) -> dict[str, Any] | None:
    """Capture complete input before invocation; failure never disrupts author flow."""
    if not capture_enabled():
        return None
    try:
        from context_budget import chars_for_tokens
        native = context if isinstance(context, dict) else {}
        raw = prompt.encode("utf-8")
        record = {
            "schema": "epyc.autopilot.mutation_author_context.v1",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "operator": operator, "target": target,
            "trial_id": native.get("trial_id"), "action_type": native.get("action_type"),
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "input_bytes": len(raw), "input_chars": len(prompt),
            "assembly_input_chars": {key: len(value) for key, value in inputs.items()},
            "assembly_input_count_semantics": "known assembler inputs; not rendered part sizes",
            "approximate_tokens": len(prompt) / chars_for_tokens(1),
            "token_estimator": "inverse context_budget.chars_for_tokens (4 characters/token)",
            "source_pins": native.get("source_pins"),
            "run_manifest_sha256": native.get("run_manifest_sha256"),
            "parameters": parameters, "model_pin": None, "human_identity": None,
        }
        # No default path/source identity is fabricated; real dispatch supplies the journal root.
        directory = native.get("journal_dir")
        if type(directory) is not str or not directory:
            raise ValueError("native journal directory absent")
        path = Path(directory) / "mutation_author_context.v1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with path.open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AuthorCaptureBusyError("author context journal busy") from exc
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record
    except Exception as exc:
        # No raw prompt or private identity in errors; this is diagnostic, never a gate.
        log.warning("Author context capture failed (%s)", type(exc).__name__)
        return {"schema": "epyc.autopilot.mutation_author_context.v1", "capture_error": type(exc).__name__}
