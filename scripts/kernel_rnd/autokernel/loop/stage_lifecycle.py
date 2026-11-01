"""Durable, opt-in AK stage receipts. Incomplete intent is never replay authority.

The JSON envelope binds campaign pins, node inputs and predecessor output receipts.
Payload bytes live in the same SQLite WAL (FULL synchronous) as their receipts.
Authorization receipts describe historical dispatches; current resource gates still run.
"""
from __future__ import annotations

import contextvars
from contextlib import closing
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Mapping

STAGES = ("assignment", "planner", "critic", "authorization", "build",
          "correctness", "attribution", "runner", "classification")
IDENTITY_FIELDS = ("campaign_id", "deployment_id", "anchor_commit", "scope_id",
                   "source_digest", "model_id", "policy_graph_id")
RECEIPT_SCHEMA = "epyc.autokernel.stage_receipt.v2"
CARRY_SCHEMA = "epyc.autokernel.stage_graph_carry.v2"
NODE_OUTPUTS = {
    "assignment": "pinned-anchor-string.v1",
    "planner": "hypothesis-or-abstention-or-authored-paths-with-patch-cas.v1",
    "critic": "closed-review.v1", "authorization": "dispatch-reservation.v1",
    "build": "full-verdict-with-binary-artifact-digests.v1",
    "correctness": "full-verdict-with-case-records.v1",
    "attribution": "exact-candidate-and-instrument-scope-or-retained-profile.v2",
    "runner": "bench-or-serving-comparison.v1",
    "classification": "commit-head-with-typed-outcome-or-native-journal-disposition.v2",
}


class LifecycleError(RuntimeError):
    """A controller refusal, never an actor or measurement verdict."""


class UncertainStage(LifecycleError):
    """Intent survived without a result; reconcile exact native evidence or stop."""


class IdentityMismatch(LifecycleError):
    """Pinned identity, inputs or content-addressed evidence changed."""


class StageAnchorMoved(IdentityMismatch):
    """A restored assignment no longer names its pinned champion."""


@dataclass(frozen=True)
class Prepared:
    stage_id: str
    state: str
    output: Any = None


def _db_synchronized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._db_lock:
            try:
                return method(self, *args, **kwargs)
            except sqlite3.Error as exc:
                raise LifecycleError("durable stage ledger IO failed; no replay authority") from exc
    return guarded


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def interface_digest(stage):
    return digest({"schema": RECEIPT_SCHEMA, "stage": stage,
                   "input": "pinned-json-input-and-predecessor-envelope.v2",
                   "output": NODE_OUTPUTS[stage]})


def contains_graph(path, graph_version):
    """Read-only bootstrap discriminator; constructor still validates all evidence."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            return bool(conn.execute(
                "SELECT 1 FROM stage_receipts WHERE graph_version=? UNION ALL "
                "SELECT 1 FROM graph_carry WHERE new_graph=? LIMIT 1",
                (graph_version, graph_version)).fetchone())
    except sqlite3.DatabaseError as exc:
        raise IdentityMismatch("existing stage ledger cannot identify its graph") from exc


class StageLifecycle:
    """One campaign ledger; subject chains are thread/task local, writes serialized.

    Same-graph startup permits incomplete intent so an exact native reconciler can
    be supplied to ``perform``. Without one, prepare refuses before the operation.
    Graph carry refuses every unresolved intent. Construction closes on any failure.
    """

    def __init__(self, path: Path, *, identity: Mapping[str, str], graph_version: str):
        if set(identity) != set(IDENTITY_FIELDS) or any(not identity[k] for k in IDENTITY_FIELDS):
            raise ValueError("stage lifecycle requires exactly the non-empty pinned identity fields")
        if not graph_version or not graph_version.strip():
            raise ValueError("graph_version is required")
        self.identity = {k: str(identity[k]) for k in IDENTITY_FIELDS}
        self.graph_version = graph_version.strip()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()
        self._subject = contextvars.ContextVar(f"ak_stage_subject_{id(self)}", default=None)
        self._predecessor = contextvars.ContextVar(f"ak_stage_parent_{id(self)}", default=None)
        self._effects = contextvars.ContextVar(f"ak_stage_children_{id(self)}", default=())
        self._db = sqlite3.connect(self.path, timeout=30, isolation_level=None,
                                   check_same_thread=False)
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS stage_receipts (
                  stage_id TEXT PRIMARY KEY, stage TEXT NOT NULL, stage_key TEXT NOT NULL,
                  identity_json TEXT NOT NULL, graph_version TEXT NOT NULL,
                  input_digest TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('prepared','done')),
                  output_json TEXT, output_digest TEXT, prepared_at REAL NOT NULL, completed_at REAL);
                CREATE TABLE IF NOT EXISTS stage_payloads (
                  payload_sha256 TEXT PRIMARY KEY, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS stop_events (
                  receipt TEXT PRIMARY KEY, identity_json TEXT NOT NULL,
                  body_json TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS graph_carry (
                  carry_id TEXT PRIMARY KEY, old_graph TEXT NOT NULL, new_graph TEXT NOT NULL,
                  identity_json TEXT NOT NULL, proof_json TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS pool_contract (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1), body_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pool_draws (
                  worker TEXT NOT NULL, ordinal INTEGER NOT NULL, finished INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY(worker,ordinal));
                CREATE TABLE IF NOT EXISTS deployment_pin (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload_sha256 TEXT NOT NULL);
            """)
            self._validate_startup()
        except (ValueError, TypeError, KeyError, sqlite3.DatabaseError) as exc:
            self._db.close()
            raise IdentityMismatch("stage ledger metadata is malformed") from exc
        except BaseException:
            self._db.close()
            raise

    def _identity(self, *, identity=None, graph=None):
        return {**(self.identity if identity is None else identity),
                "graph_version": self.graph_version if graph is None else graph}

    def _stage_id(self, stage, key, *, identity=None, graph=None):
        return digest({"identity": self._identity(identity=identity, graph=graph),
                       "stage": stage, "key": key})

    def _validate_startup(self):
        current = self._identity()
        stable = set(IDENTITY_FIELDS) - {"policy_graph_id", "source_digest"}
        for stage_id, stage, key, identity_json, graph, state in self._db.execute(
                "SELECT stage_id,stage,stage_key,identity_json,graph_version,state FROM stage_receipts"):
            try:
                old = json.loads(identity_json)
            except (ValueError, TypeError) as exc:
                raise IdentityMismatch("corrupt receipt identity") from exc
            if old.get("campaign_id") != current["campaign_id"]:
                raise IdentityMismatch("ledger belongs to another campaign")
            if stage_id != digest({"identity": old, "stage": stage, "key": key}):
                raise IdentityMismatch("corrupt receipt stage identity")
            if graph == self.graph_version:
                if old != current:
                    raise IdentityMismatch("pinned campaign identity changed")
                if state == "done":
                    bound = json.loads(self.payload(self._read_done(stage_id)[0]["input_digest"]))
                    if bound["interface_sha256"] != interface_digest(stage):
                        raise IdentityMismatch("same graph changed its node interface")
            else:
                if any(old.get(k) != current[k] for k in stable):
                    raise IdentityMismatch("graph upgrade changes deployment/source/model/scope/anchor")
                if state != "done":
                    raise UncertainStage("graph carry cannot cross unresolved intent")
                if not self._carried_to_current(stage_id, graph, set()):
                    raise IdentityMismatch("graph upgrade requires a verified carry for each receipt")
            if state == "done":
                self._read_done(stage_id)

    def _carried_to_current(self, source_id, graph, visited):
        if (source_id, graph) in visited:
            return False
        visited = {*visited, (source_id, graph)}
        for carry_id, proof_json in self._db.execute(
                "SELECT carry_id,proof_json FROM graph_carry WHERE old_graph=?", (graph,)).fetchall():
            proof = json.loads(proof_json)
            if digest(proof) != carry_id:
                raise IdentityMismatch("corrupt graph carry receipt")
            self._validate_carry_evidence(proof)
            node = proof["nodes"].get(source_id)
            if node is None:
                continue
            source, source_sha = self._read_done(source_id)
            target, target_sha = self._read_done(node["target_stage_id"])
            source_bound = json.loads(self.payload(source["input_digest"]))
            target_bound = json.loads(self.payload(target["input_digest"]))
            if (source_sha != node["source_output_sha256"]
                    or target_sha != node["target_output_sha256"]
                    or source["input_digest"] != node["source_input_sha256"]
                    or target["input_digest"] != node["target_input_sha256"]
                    or source_bound["interface_sha256"] != node["source_interface_sha256"]
                    or target_bound["interface_sha256"] != node["target_interface_sha256"]
                    or source_bound["inputs"] != target_bound["inputs"]
                    or source["value"] != target["value"]
                    or source_bound["identity"] != {**proof["source_identity"], "graph_version": proof["from"]}
                    or target_bound["identity"] != {**proof["target_identity"], "graph_version": proof["to"]}):
                raise IdentityMismatch("carried node differs from its exact CAS evidence")
            if proof["to"] == self.graph_version and proof["target_identity"] == self.identity:
                return True
            if self._carried_to_current(node["target_stage_id"], proof["to"], visited):
                return True
        return False

    def close(self):
        with self._db_lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @property
    def identity_digest(self):
        return digest(self.identity)

    def _read_done(self, stage_id):
        row = self._db.execute(
            "SELECT output_json,output_digest,input_digest,stage,stage_key,identity_json FROM stage_receipts "
            "WHERE stage_id=? AND state='done'", (stage_id,)).fetchone()
        if row is None:
            raise IdentityMismatch("completed predecessor receipt is absent")
        try:
            body = json.loads(row[0])
            if digest(body) != row[1] or body["schema"] != RECEIPT_SCHEMA \
                    or body["stage_id"] != stage_id or body["input_digest"] != row[2]:
                raise IdentityMismatch("completed receipt digest or envelope changed")
            bound = json.loads(self.payload(row[2]))
            if digest(bound) != row[2] or bound["predecessors"] != body["predecessors"]:
                raise IdentityMismatch("receipt input CAS or predecessor scope changed")
            if bound["stage"] != row[3] or bound["key"] != row[4] \
                    or bound["identity"] != json.loads(row[5]):
                raise IdentityMismatch("receipt input CAS differs from node identity")
            for sha in body["payloads"]:
                self.payload(sha)
            for parent in [*body["predecessors"], *body.get("children", [])]:
                previous = self._db.execute(
                    "SELECT output_digest FROM stage_receipts WHERE stage_id=? AND state='done'",
                    (parent["stage_id"],)).fetchone()
                if previous != (parent["output_digest"],):
                    raise IdentityMismatch("receipt predecessor changed or disappeared")
            return body, row[1]
        except (ValueError, TypeError, KeyError) as exc:
            raise IdentityMismatch("corrupt completed receipt JSON") from exc

    def _advance(self, stage_id, output_digest):
        self._predecessor.set({"stage_id": stage_id, "output_digest": output_digest})

    def _track(self, stage_id, output_digest):
        row = {"stage_id": stage_id, "output_digest": output_digest}
        for parent_id, children in self._effects.get():
            if parent_id != stage_id and row not in children:
                children.append(row)

    @_db_synchronized
    def completed(self, *, stage, stage_key):
        stage_id = self._stage_id(stage, stage_key)
        row = self._db.execute("SELECT state FROM stage_receipts WHERE stage_id=?", (stage_id,)).fetchone()
        if row is None:
            return None
        if row[0] != "done":
            raise UncertainStage("commit/disposition intent requires exact native reconciliation")
        body, sha = self._read_done(stage_id)
        self._advance(stage_id, sha)
        self._track(stage_id, sha)
        return Prepared(stage_id, "done", body["value"])

    @_db_synchronized
    def prepare(self, *, stage, stage_key, inputs):
        if stage not in STAGES or not stage_key:
            raise ValueError("known stage and non-empty stage_key required")
        stage_id = self._stage_id(stage, stage_key)
        parents = [self._predecessor.get()] if self._predecessor.get() else []
        if parents and parents[0]["stage_id"] == stage_id:
            previous, _ = self._read_done(stage_id)
            parents = previous["predecessors"]
        bound = {"schema": RECEIPT_SCHEMA, "identity": self._identity(),
                 "stage": stage, "key": stage_key, "inputs": inputs,
                 "predecessors": parents, "interface_sha256": interface_digest(stage)}
        input_sha = digest(bound)
        identity_json = _canonical(self._identity()).decode()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute(
                "SELECT identity_json,input_digest,state FROM stage_receipts WHERE stage_id=?",
                (stage_id,)).fetchone()
            if row:
                if row[:2] != (identity_json, input_sha):
                    raise IdentityMismatch(f"stage {stage}/{stage_key} inputs or predecessors changed")
                if row[2] != "done":
                    raise UncertainStage(f"stage {stage}/{stage_key} has intent without result")
                body, output_sha = self._read_done(stage_id)
                self._db.execute("COMMIT")
                self._advance(stage_id, output_sha)
                self._track(stage_id, output_sha)
                return Prepared(stage_id, "done", body["value"])
            # Persist the exact input envelope too; a naked hash cannot be carried.
            self.store_payload(_canonical(bound))
            self._db.execute(
                "INSERT INTO stage_receipts VALUES(?,?,?,?,?,?,'prepared',NULL,NULL,?,NULL)",
                (stage_id, stage, stage_key, identity_json, self.graph_version, input_sha, time.time()))
            self._db.execute("COMMIT")
            return Prepared(stage_id, "prepared")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    @_db_synchronized
    def complete(self, prepared, output):
        if prepared.state != "prepared":
            raise ValueError("only newly prepared intent can complete")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute(
                "SELECT input_digest,identity_json FROM stage_receipts WHERE stage_id=? AND state='prepared'",
                (prepared.stage_id,)).fetchone()
            if row is None or row[1] != _canonical(self._identity()).decode():
                raise IdentityMismatch("prepared receipt absent or foreign")
            bound = json.loads(self.payload(row[0]))
            # Explicit payload references only; no arbitrary recursion over user JSON.
            payloads = ([output["payload_sha256"]] if isinstance(output, dict)
                        and "payload_sha256" in output else [])
            for sha in payloads:
                self.payload(sha)
            children = [row for parent_id, rows in self._effects.get()
                        if parent_id == prepared.stage_id for row in rows]
            if self._predecessor.get() and self._predecessor.get() not in bound["predecessors"] \
                    and self._predecessor.get() not in children:
                children.append(self._predecessor.get())
            body = {"schema": RECEIPT_SCHEMA, "stage_id": prepared.stage_id,
                    "input_digest": row[0], "predecessors": bound["predecessors"],
                    "payloads": payloads, "value": output,
                    "children": children}
            output_sha = digest(body)
            self._db.execute(
                "UPDATE stage_receipts SET state='done',output_json=?,output_digest=?,completed_at=? "
                "WHERE stage_id=? AND state='prepared'",
                (_canonical(body).decode(), output_sha, time.time(), prepared.stage_id))
            finish = bound["inputs"].get("finish_draw") if isinstance(bound["inputs"], dict) else None
            if finish is not None:
                worker, ordinal = finish
                if bound["stage"] != "classification" or bound["key"] != f"{worker}:{ordinal}:record":
                    raise IdentityMismatch("invalid terminal draw receipt")
                self._db.execute("UPDATE pool_draws SET finished=1 WHERE worker=? AND ordinal=?", (worker, ordinal))
            self._db.execute("COMMIT")
            self._advance(prepared.stage_id, output_sha)
            self._track(prepared.stage_id, output_sha)
            return output_sha
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def perform(self, *, stage, stage_key, inputs, operation,
                encode=lambda value: value, decode=lambda value: value,
                cas_payload=None, rehydrate=None, isolated=False):
        if isolated:
            token = self._predecessor.set(None)
            try:
                return self.perform(stage=stage, stage_key=stage_key, inputs=inputs,
                    operation=operation, encode=encode, decode=decode,
                    cas_payload=cas_payload, rehydrate=rehydrate)
            finally:
                completed = self._predecessor.get()
                self._predecessor.reset(token)
                if completed is not None:
                    self._advance(completed["stage_id"], completed["output_digest"])
        receipt = self.prepare(stage=stage, stage_key=stage_key, inputs=inputs)
        if receipt.state == "done":
            value = decode(receipt.output)
            if rehydrate is not None and rehydrate() != value:
                raise StageAnchorMoved("restored assignment differs from pinned completed assignment")
            return value
        token = self._effects.set((*self._effects.get(), (receipt.stage_id, [])))
        try:
            value = operation()
            output = encode(value)
            if cas_payload is not None:
                # Capture AFTER the successful operation, before the done transaction.
                raw = cas_payload() if callable(cas_payload) else cas_payload
                if not isinstance(output, dict):
                    raise TypeError("CAS-bearing output must be a closed JSON object")
                output = {**output, "payload_sha256": self.store_payload(raw)}
            self.complete(receipt, output)
            return value
        finally:
            self._effects.reset(token)

    @_db_synchronized
    def store_payload(self, payload):
        raw = bytes(payload)
        sha = hashlib.sha256(raw).hexdigest()
        self._db.execute("INSERT OR IGNORE INTO stage_payloads VALUES(?,?)", (sha, sqlite3.Binary(raw)))
        if self.payload(sha) != raw:
            raise IdentityMismatch("payload collision or corrupt CAS")
        return sha

    @_db_synchronized
    def payload(self, sha256):
        row = self._db.execute("SELECT payload FROM stage_payloads WHERE payload_sha256=?", (sha256,)).fetchone()
        if row is None or hashlib.sha256(row[0]).hexdigest() != sha256:
            raise IdentityMismatch("content-addressed stage payload absent or corrupt")
        return bytes(row[0])

    def bind_subject(self, subject):
        lifecycle = self
        class Binding:
            def __enter__(self):
                self.token = lifecycle._subject.set(str(subject))
                self.parent = lifecycle._predecessor.set(None)
                return lifecycle
            def __exit__(self, *_exc):
                lifecycle._predecessor.reset(self.parent)
                lifecycle._subject.reset(self.token)
        return Binding()

    @property
    def subject(self):
        return self._subject.get()

    @property
    def predecessor_digest(self):
        return digest(self._predecessor.get())

    @_db_synchronized
    def configure_pool(self, workers, iterations):
        body = _canonical({"workers": sorted(workers), "iterations": iterations}).decode()
        self._db.execute("INSERT OR IGNORE INTO pool_contract VALUES(1,?)", (body,))
        if self._db.execute("SELECT body_json FROM pool_contract").fetchone() != (body,):
            raise IdentityMismatch("same graph restart changed worker roster or iteration budget")

    @_db_synchronized
    def next_draw(self, worker, iterations):
        """Replay this worker's unfinished ordinal before allocating campaign budget."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            pending = self._db.execute(
                "SELECT ordinal FROM pool_draws WHERE worker=? AND finished=0 ORDER BY ordinal LIMIT 1",
                (worker,)).fetchone()
            if pending:
                result = pending[0]
            elif iterations is not None and self._db.execute(
                    "SELECT COUNT(*) FROM pool_draws").fetchone()[0] >= iterations:
                result = None
            else:
                result = self._db.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM pool_draws WHERE worker=?", (worker,)).fetchone()[0]
                self._db.execute("INSERT INTO pool_draws VALUES(?,?,0)", (worker, result))
            self._db.execute("COMMIT")
            return result
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    @_db_synchronized
    def finish_draw(self, worker, ordinal):
        self._db.execute("UPDATE pool_draws SET finished=1 WHERE worker=? AND ordinal=?", (worker, ordinal))

    @_db_synchronized
    def deployment_descriptor(self):
        row = self._db.execute("SELECT payload_sha256 FROM deployment_pin").fetchone()
        if row is None:
            return None
        value = json.loads(self.payload(row[0]))
        if digest(value) != self.identity["deployment_id"]:
            raise IdentityMismatch("immutable deployment pin changed")
        return value

    @_db_synchronized
    def register_controller_source(self, sources, *, expected_digest=None):
        manifest = {str(name): hashlib.sha256(bytes(raw)).hexdigest() for name, raw in sources.items()}
        source_sha = digest(manifest)
        if source_sha != (expected_digest or self.identity["source_digest"]):
            raise IdentityMismatch("executing controller source differs from declared graph source")
        for raw in sources.values():
            self.store_payload(raw)
        self.store_payload(_canonical(manifest))
        return source_sha

    @_db_synchronized
    def bind_deployment(self, *, contract, frontier, native_continuation=None):
        """Keep controller/deployment pins immutable while native keeps advance source.

        A new experimental frontier needs BOTH a completed commit in this graph's
        predecessor chain and the driver's already-verified native continuation.
        Neither an arbitrary new HEAD nor an unrecorded promotion can rebind it.
        """
        row = self._db.execute("SELECT payload_sha256 FROM deployment_pin").fetchone()
        if row is None:
            descriptor = {"schema": "epyc.autokernel.stage_deployment.v1",
                          "contract": contract, "initial_frontier": frontier}
            if digest(descriptor) != self.identity["deployment_id"] \
                    or frontier["commit"] != self.identity["anchor_commit"]:
                raise IdentityMismatch("initial deployment descriptor differs from declared pins")
            sha = self.store_payload(_canonical(descriptor))
            self._db.execute("INSERT INTO deployment_pin VALUES(1,?)", (sha,))
            return
        descriptor = json.loads(self.payload(row[0]))
        if digest(descriptor) != self.identity["deployment_id"] or descriptor["contract"] != contract:
            raise IdentityMismatch("controller deployment/model/policy/measurement contract changed")
        if frontier == descriptor["initial_frontier"]:
            return
        commit = frontier["commit"]
        if not isinstance(native_continuation, dict) \
                or native_continuation.get("current_anchor", {}).get("commit") != commit:
            raise IdentityMismatch("advanced kernel source requires exact native continuation evidence")
        for stage_id, in self._db.execute(
                "SELECT stage_id FROM stage_receipts WHERE identity_json=? AND stage='classification' "
                "AND stage_key LIKE '%:commit' AND state='done'", (_canonical(self._identity()).decode(),)):
            body, _ = self._read_done(stage_id)
            value = body["value"]
            if (value.get("head") if isinstance(value, dict) else value) == commit:
                # Retain exact verified native evidence on the write side. No
                # changes to any old stage identity, inputs or deployment pins.
                self.store_payload(_canonical({"schema": "epyc.autokernel.stage_frontier.v1",
                    "deployment_id": self.identity["deployment_id"], "commit_receipt": stage_id,
                    "frontier": frontier, "native_continuation": native_continuation}))
                return
        raise IdentityMismatch("advanced kernel source has no completed promotion receipt")

    @_db_synchronized
    def request_stop(self, reason):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.stop_receipt
            if prior is not None:
                self._db.execute("COMMIT")
                return prior
            body = {"schema": "epyc.autokernel.stage_stop.v1", "identity": self._identity(),
                    "reason": str(reason)[:1000], "requested_at_ns": time.time_ns()}
            receipt = digest(body)
            self._db.execute("INSERT INTO stop_events VALUES(?,?,?,0)",
                             (receipt, _canonical(self._identity()).decode(), _canonical(body).decode()))
            self._db.execute("COMMIT")
            return receipt
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    @property
    @_db_synchronized
    def stop_receipt(self):
        rows = self._db.execute("SELECT receipt,identity_json,body_json FROM stop_events WHERE acknowledged=0").fetchall()
        if not rows:
            return None
        if len(rows) != 1 or rows[0][1] != _canonical(self._identity()).decode() \
                or digest(json.loads(rows[0][2])) != rows[0][0]:
            raise IdentityMismatch("stop receipt is corrupt or belongs to another graph")
        return rows[0][0]

    @property
    def stop_requested(self):
        return self.stop_receipt is not None

    @_db_synchronized
    def check_resume_ack(self, acknowledgement):
        expected = {"schema": "epyc.autokernel.stage_resume.v1",
                    "identity": self._identity(), "stop_receipt": self.stop_receipt}
        if expected["stop_receipt"] is None or dict(acknowledgement) != expected:
            raise IdentityMismatch("resume acknowledgement does not match current stop/identity")
        return expected

    @_db_synchronized
    def acknowledge_resume(self, acknowledgement):
        """Fence clearing to one exact current stop and graph. Never auto-clear."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            expected = self.check_resume_ack(acknowledgement)
            count = self._db.execute("UPDATE stop_events SET acknowledged=1 WHERE receipt=? AND acknowledged=0",
                                     (expected["stop_receipt"],)).rowcount
            if count != 1:
                raise IdentityMismatch("stop was already acknowledged")
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def _validate_carry_evidence(self, proof):
        if proof.get("schema") != CARRY_SCHEMA:
            raise IdentityMismatch("unsupported graph carry schema")
        try:
            evidence = json.loads(self.payload(proof["verifier"]["evidence_sha256"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise IdentityMismatch("graph compatibility evidence is malformed") from exc
        required = {k: proof[k] for k in ("schema", "source_identity", "target_identity", "from", "to", "nodes")}
        if evidence != required:
            raise IdentityMismatch("compatibility evidence does not bind exact graph/node interfaces and pins")
        if not proof["verifier"].get("verifier"):
            raise IdentityMismatch("graph carry requires a named verifier")
        if proof["source_identity"]["source_digest"] != proof["target_identity"]["source_digest"]:
            for identity in (proof["source_identity"], proof["target_identity"]):
                manifest = json.loads(self.payload(identity["source_digest"]))
                if not isinstance(manifest, dict) or not manifest:
                    raise IdentityMismatch("graph source upgrade lacks its original source manifest")
                for source_sha in manifest.values():
                    self.payload(source_sha)

    @_db_synchronized
    def carry_forward(self, *, new_graph_version, new_identity=None, output_proofs,
                      verifier_receipt):
        """Versioned byte-preserving carry with exact input/output interface evidence.

        No arbitrary converter runs here. Each node must declare the same interface
        digest and exact old input/output scope; graph/deployment/model/source pins
        and node evidence are checked against CAS before any target row is written.
        """
        target = dict(self.identity if new_identity is None else new_identity)
        if set(target) != set(IDENTITY_FIELDS) or not all(target.values()):
            raise IdentityMismatch("carry target identity is incomplete")
        if not new_graph_version or new_graph_version == self.graph_version:
            raise ValueError("carry requires a distinct graph version")
        stable = set(IDENTITY_FIELDS) - {"policy_graph_id", "source_digest"}
        if any(target[k] != self.identity[k] for k in stable):
            raise IdentityMismatch("carry changes deployment/model/scope/anchor")
        if self.stop_requested:
            raise IdentityMismatch("acknowledge the exact old stop before upgrading")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            rows = self._db.execute(
                "SELECT stage_id,stage,stage_key,input_digest,state FROM stage_receipts WHERE identity_json=?",
                (_canonical(self._identity()).decode(),)).fetchall()
            if any(row[4] != "done" for row in rows):
                raise UncertainStage("graph carry cannot cross unresolved intent")
            if set(output_proofs) != {row[0] for row in rows}:
                raise IdentityMismatch("carry proof must cover every completed node")
            nodes = {}
            for old_id, stage, key, input_sha, _state in rows:
                body, output_sha = self._read_done(old_id)
                bound = json.loads(self.payload(input_sha))
                node = dict(output_proofs[old_id])
                interface = node.get("source_interface_sha256")
                if interface != bound["interface_sha256"] or interface != interface_digest(stage) \
                        or node.get("target_interface_sha256") != interface:
                    raise IdentityMismatch("carry requires unchanged declared node interface")
                expected = {"stage": stage, "stage_key": key, "input_scope_sha256": digest(bound["inputs"]),
                            "source_input_sha256": input_sha, "source_output_sha256": output_sha,
                            "output_scope_sha256": digest(body["value"]),
                            "source_interface_sha256": interface, "target_interface_sha256": interface,
                            "target_stage_id": self._stage_id(stage, key, identity=target, graph=new_graph_version)}
                if any(node.get(k) != v for k, v in expected.items()):
                    raise IdentityMismatch("carry node evidence differs from actual CAS receipt")
                # Target envelopes change with the graph and mapped predecessor ids.
                nodes[old_id] = {**expected}
            mapping = {old: node["target_stage_id"] for old, node in nodes.items()}
            pending = {row[0]: row for row in rows}
            carried = {}
            while pending:
                progressed = False
                for old_id, row in list(pending.items()):
                    body, _ = self._read_done(old_id)
                    if any(p["stage_id"] not in carried for p in
                           [*body["predecessors"], *body.get("children", [])]):
                        continue
                    bound = json.loads(self.payload(row[3]))
                    parents = [{"stage_id": mapping[p["stage_id"]],
                                "output_digest": carried[p["stage_id"]][1]}
                               for p in body["predecessors"]]
                    bound = {**bound, "identity": self._identity(identity=target, graph=new_graph_version),
                             "predecessors": parents}
                    new_input = digest(bound)
                    new_body = {**body, "stage_id": mapping[old_id], "input_digest": new_input,
                                "predecessors": parents,
                                "children": [{"stage_id": mapping[p["stage_id"]],
                                              "output_digest": carried[p["stage_id"]][1]}
                                             for p in body.get("children", [])]}
                    new_output = digest(new_body)
                    nodes[old_id].update(target_input_sha256=new_input, target_output_sha256=new_output)
                    carried[old_id] = (bound, new_output, new_body)
                    del pending[old_id]
                    progressed = True
                if not progressed:
                    raise IdentityMismatch("carry has cyclic or foreign predecessors")
            proof = {"schema": CARRY_SCHEMA, "source_identity": self.identity, "target_identity": target,
                     "from": self.graph_version, "to": new_graph_version, "nodes": nodes,
                     "verifier": dict(verifier_receipt)}
            # Evidence must have been authored externally against these exact nodes.
            self._validate_carry_evidence(proof)
            for old_id, row in {row[0]: row for row in rows}.items():
                bound, output_sha, body = carried[old_id]
                self.store_payload(_canonical(bound))
                self._db.execute("INSERT INTO stage_receipts VALUES(?,?,?,?,?,?,'done',?,?,?,?)",
                    (mapping[old_id], row[1], row[2], _canonical(bound["identity"]).decode(),
                     new_graph_version, digest(bound), _canonical(body).decode(), output_sha, time.time(), time.time()))
            carry_id = digest(proof)
            self._db.execute("INSERT INTO graph_carry VALUES(?,?,?,?,?,?)",
                (carry_id, self.graph_version, new_graph_version, _canonical(target).decode(),
                 _canonical(proof).decode(), time.time()))
            self._db.execute("COMMIT")
            return carry_id
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
