"""Authenticated commands queued to the existing serial owner, never a second writer."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import queue
import threading
import uuid

from . import campaign_service as transport

SCHEMA = "epyc.autokernel.serial_control.v1"
COMMAND_SCHEMA = "epyc.autokernel.serial_command.v1"
MAX_COMMANDS = 64


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SerialControl:
    """Only pump() runs on the original serial execution thread and saves state."""

    def __init__(self, state, save, stop, *, config_digest):
        self.state, self.save, self.stop = state, save, stop
        self.config_digest = config_digest
        self.owner_id = uuid.uuid4().hex
        self.endpoint = None
        self.allowed_origin = None
        self._queue = queue.Queue(maxsize=16)
        self._lock = threading.Lock()
        self._snapshot = None
        control = state.setdefault("control", {"revision": 0, "desired_state": "running",
                                               "commands": []})
        if (set(control) != {"revision", "desired_state", "commands"}
                or type(control["revision"]) is not int or control["revision"] < 0
                or control["desired_state"] not in {"running", "paused", "drained"}
                or not isinstance(control["commands"], list)
                or len(control["commands"]) > MAX_COMMANDS):
            raise ValueError("invalid retained serial control state")

    def publish_snapshot(self):
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError("serial owner has not published")
            return copy.deepcopy(self._snapshot)

    def apply_command(self, request):
        fields = {"schema", "config_digest", "owner_id", "request_id", "operation",
                  "expected_revision", "expected_batch", "expected_target"}
        if (not isinstance(request, dict) or set(request) != fields
                or request["schema"] != COMMAND_SCHEMA
                or request["config_digest"] != self.config_digest
                or request["owner_id"] != self.owner_id
                or not isinstance(request["request_id"], str)
                or not 1 <= len(request["request_id"]) <= 80
                or request["operation"] not in {"pause", "resume", "drain"}
                or type(request["expected_revision"]) is not int
                or request["expected_revision"] < 0
                or type(request["expected_batch"]) is not int or request["expected_batch"] < 0
                or (request["expected_target"] is not None
                    and not isinstance(request["expected_target"], str))):
            raise ValueError("command must bind this serial owner, configuration and revision")
        item = {"request": copy.deepcopy(request), "done": threading.Event()}
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            raise ValueError("serial command queue full; retry the same request") from exc
        if not item["done"].wait(1.0):
            raise RuntimeError("serial acknowledgement uncertain; retry the same request_id")
        if "error" in item:
            raise ValueError(item["error"])
        return item["result"]

    def pump(self, active=None, *, terminal=False, stopped=False, failed=False):
        """Apply and commit at existing polling boundaries; HTTP only queues."""
        control = self.state["control"]
        waiting = []
        changed = False
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            request = item["request"]
            prior = next((row for row in control["commands"]
                          if row["request"]["request_id"] == request["request_id"]), None)
            if prior is not None:
                if prior["request"] != request:
                    item["error"] = "request_id already belongs to another command"
                else:
                    item["result"] = prior["result"]
            elif request["expected_revision"] != control["revision"]:
                item["error"] = "stale serial revision; refresh before a new command"
            elif (request["expected_batch"] != self.state["next_batch"]
                  or request["expected_target"] != (active.get("selected_id") if active else None)):
                item["error"] = "serial batch/target changed; refresh before a new command"
            elif terminal or stopped or control["desired_state"] == "drained":
                item["error"] = "serial owner is terminal or draining"
            else:
                for row in control["commands"]:
                    if not row["result"]["completed"]:
                        row["result"].update(completed=True, outcome="superseded",
                                             completed_at=_now())
                control["revision"] += 1
                control["desired_state"] = {"pause": "paused", "resume": "running",
                                             "drain": "drained"}[request["operation"]]
                result = {"request_id": request["request_id"], "operation": request["operation"],
                          "revision": control["revision"], "accepted_at": _now(),
                          "completed": False, "outcome": "pending", "completed_at": None}
                control["commands"].append({"request": request, "result": result})
                del control["commands"][:-MAX_COMMANDS]
                item["result"] = result
                changed = True
            waiting.append(item)
        desired = control["desired_state"]
        observed = ("failed" if terminal and failed else
                    "drained" if terminal and (stopped or desired == "drained") else
                    "complete" if terminal else "draining" if stopped or desired == "drained" else
                    "pausing" if desired == "paused" and active else
                    "paused" if desired == "paused" else "running")
        for row in control["commands"]:
            result = row["result"]
            if not result["completed"] and (
                    (result["operation"] == "pause" and observed == "paused") or
                    (result["operation"] == "resume" and observed == "running") or
                    (result["operation"] == "drain" and observed == "drained")):
                result.update(completed=True, outcome="completed", completed_at=_now())
                changed = True
            elif terminal and not result["completed"]:
                result.update(completed=not failed, outcome="failed" if failed else "superseded",
                              completed_at=None if failed else _now())
                changed = True
        # No successful ACK or stop side effect precedes the original durable save.
        if changed:
            self.save()
        if desired == "drained":
            self.stop(None, None)
        snapshot = {"schema": SCHEMA, "config_digest": self.config_digest,
                    "owner_id": self.owner_id, "revision": control["revision"],
                    "desired_state": desired, "observed_state": observed,
                    "endpoint": self.endpoint, "allowed_origin": self.allowed_origin,
                    "active_target": (active.get("selected_id") if active else None),
                    "batch_number": self.state["next_batch"],
                    "commands": copy.deepcopy(control["commands"])}
        with self._lock:
            self._snapshot = snapshot
        for item in waiting:
            if "result" in item:
                item["result"] = copy.deepcopy(item["result"])
            item["done"].set()
        return snapshot


class SerialHTTPService:
    """Reuse the installed bounded authenticated transport, with serial identity."""

    def __init__(self, control, listen, token, *, allowed_origin=None):
        host, port = transport.parse_listen(listen)
        if not isinstance(token, str) or not token:
            raise ValueError("AUTOKERNEL_CONTROL_TOKEN is required with --control-listen")
        control.allowed_origin = transport.validate_origin(allowed_origin)
        self.server = transport._OwnedHTTPServer(
            (host, port), transport.make_handler(
                control, token, allowed_origin=control.allowed_origin,
                transport_health=lambda: (200, {
                    "ok": True, "transport": "serial-control-http",
                    "config_digest": control.config_digest, "owner_id": control.owner_id})),
            total_request_deadline=transport.TOTAL_REQUEST_DEADLINE_S)
        control.endpoint = f"http://{host}:{self.server.server_address[1]}"
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=.05),
            name="serial-control-http", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.server.close_owned_connections()
        if self.thread.ident is not None:
            self.server.shutdown()
            self.thread.join(transport.STOP_TIMEOUT_S)
        self.server.join_watchdog(transport.STOP_TIMEOUT_S)
        self.server.server_close()
        if self.thread.is_alive() or self.server.watchdog_thread.is_alive():
            raise RuntimeError("serial HTTP owned threads did not stop")
