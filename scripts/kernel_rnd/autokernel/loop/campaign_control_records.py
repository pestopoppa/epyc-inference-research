"""Operation settlement, A2 records and native retention methods of CampaignController, moved verbatim (AK-SPLIT-CAMPAIGN-CONTROL)."""
from __future__ import annotations
from typing import Any, Mapping
import copy

from .. import journal as journal_module, schemas
from . import native_retention_catalog as retention_catalog_module
from . import retention_owner_journal as owner_journal
from . import operation_accounting as operation_accounting_module
from . import operation_child_proof as child_proof_module
from . import scheduling
from . import worker_lifecycle as worker_lifecycle_module
from .campaign_control import (
    CampaignController, ControlRefused, DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
    DRIVER_SETTLEMENT_SCHEMA, DRIVER_SETTLEMENT_SCHEMA_V2)


def append_a2_unit_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Append one AKU-04f v2 unit record (chain-verified) or replay an exact duplicate."""
    from . import a2_unit_permit

    with self._mutex:
        self._require_active_locked()
        assert self._journal is not None
        try:
            row = a2_unit_permit.validate_record(record)
            key = a2_unit_permit.unit_key(row)
            entries = self._a2_unit_entries.get(key, [])
            if any(item.payload["record_digest"] == row["record_digest"]
                   for item in entries):
                return copy.deepcopy(row)
            a2_unit_permit.project_unit([*(i.payload for i in entries), row])
        except a2_unit_permit.A2UnitRefused as exc:
            raise ControlRefused(f"invalid A2 unit record: {exc}") from exc
        if row["owner_incarnation"] != self.supervisor_incarnation:
            raise ControlRefused("A2 unit record is not by the current owner incarnation")
        entry = self._append_durable(
            journal_module.KIND_A2_RUNTIME_EXECUTION, row,
            campaign_id=self.resolved.campaign_id, record_id=row["execution_id"])
        self._a2_unit_entries.setdefault(key, []).append(copy.deepcopy(entry))
        return copy.deepcopy(row)

def replay_a2_unit_records(self, key: str) -> tuple:
    """Replayed (never authorizing) unit records for one `a2_unit_permit.unit_key`."""
    with self._mutex:
        self._require_active_locked()
        return tuple(copy.deepcopy(i.payload) for i in self._a2_unit_entries.get(key, []))

def append_a2_runtime_transition(self, *, logical_id: str, plan,
                                 event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Append or exactly replay one current-owner A2 phase event."""
    from . import a2_execution_state

    with self._mutex:
        self._require_active_locked()
        assert self._journal is not None
        try:
            from . import experiment_plan as experiment_plan_module
            if not isinstance(plan, experiment_plan_module.ExperimentPlan):
                raise a2_execution_state.A2ExecutionStateRefused(
                    "A2 append requires an ExperimentPlan")
            plan = experiment_plan_module.ExperimentPlan.from_dict(plan.to_dict())
            if plan.campaign_id != self.resolved.campaign_id:
                raise a2_execution_state.A2ExecutionStateRefused(
                    "A2 plan and controller campaign differ")
            event = a2_execution_state.validate_event_membership(event, plan)
            transition = a2_execution_state.make_transition(
                campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation,
                config_digest=self.config_digest,
                supervisor_incarnation=self.supervisor_incarnation,
                logical_id=logical_id, event=event)
        except a2_execution_state.A2ExecutionStateRefused as exc:
            raise ControlRefused(f"invalid A2 runtime transition: {exc}") from exc
        execution_id = transition["execution_id"]
        prior_execution = self._a2_logical_executions.get(transition["logical_id"])
        if prior_execution not in {None, execution_id}:
            raise ControlRefused("A2 logical execution changes fixed identity")
        entries = self._a2_execution_entries.get(execution_id, [])
        event_digest = transition["event"]["event_digest"]
        prior = next((item for item in entries
                      if item.payload["event"].get("event_digest") == event_digest), None)
        if prior is not None:
            return copy.deepcopy(prior.payload["event"])
        try:
            a2_execution_state.project_transitions(
                [*(item.payload for item in entries), transition])
        except a2_execution_state.A2ExecutionStateRefused as exc:
            raise ControlRefused(f"A2 runtime transition is inconsistent: {exc}") from exc
        entry = self._append_durable(
            journal_module.KIND_A2_RUNTIME_EXECUTION, transition,
            campaign_id=self.resolved.campaign_id, record_id=execution_id)
        self._a2_logical_executions[transition["logical_id"]] = execution_id
        self._a2_execution_entries.setdefault(execution_id, []).append(
            copy.deepcopy(entry))
        if transition["event"]["phase"] == "anchor_bank" \
                and transition["event"]["state"] == "SEALED":
            bank_digest = transition["event"]["payload"]["bank_digest"]
            sources = self._a2_bank_sources.setdefault(bank_digest, [])
            if execution_id not in sources:
                sources.append(execution_id)
        return copy.deepcopy(transition["event"])

def _a2_resolve_bank_source_locked(self, bank, *, target_plan_digest: str,
                                   target_frame_digest: str):
    from . import a2_execution_state, discovery_screen

    if not self._mutex._is_owned():
        raise ControlRefused("A2 bank source resolution requires controller lock")
    try:
        bank = discovery_screen.BaselineBank.from_dict(bank.to_dict())
    except Exception as exc:
        raise ControlRefused(f"A2 reused bank is invalid: {exc}") from exc
    matches = []
    for source_id in self._a2_bank_sources.get(bank.bank_digest, []):
        entries = self._a2_execution_entries.get(source_id)
        if not entries:
            continue
        try:
            reference = a2_execution_state.make_bank_reference(
                source_values=[item.payload for item in entries],
                source_journal_entry_ids=[item.event_id for item in entries],
                target_plan_digest=target_plan_digest,
                target_frame_digest=target_frame_digest)
            projection = a2_execution_state.project_transitions(
                [item.payload for item in entries])
            seal = next(event for event in projection.events
                        if event["phase"] == "anchor_bank"
                        and event["state"] == "SEALED")
            stored = discovery_screen.BaselineBank.from_dict(seal["payload"])
        except (a2_execution_state.A2ExecutionStateRefused,
                discovery_screen.DiscoveryScreenRefused, StopIteration):
            continue
        if (reference["source_frame_digest"] == target_frame_digest
                and stored.to_dict() == bank.to_dict()):
            matches.append((reference, stored))
    if len(matches) != 1:
        raise ControlRefused(
            "A2 reused bank lacks one exact indexed original sealed source")
    return matches[0]

def _bank_for_a2_reference_locked(self, reference: Mapping[str, Any]):
    from . import a2_execution_state, discovery_screen

    if not self._mutex._is_owned():
        raise ControlRefused("A2 bank reference reopening requires controller lock")
    source = self._a2_execution_entries.get(reference["source_execution_id"])
    if not source:
        raise ControlRefused("A2 bank reference source is missing")
    try:
        expected = a2_execution_state.make_bank_reference(
            source_values=[item.payload for item in source],
            source_journal_entry_ids=[item.event_id for item in source],
            target_plan_digest=reference["target_plan_digest"],
            target_frame_digest=reference["target_frame_digest"])
        projection = a2_execution_state.project_transitions(
            [item.payload for item in source])
        seal = next(event for event in projection.events
                    if event["phase"] == "anchor_bank"
                    and event["state"] == "SEALED")
        bank = discovery_screen.BaselineBank.from_dict(seal["payload"])
    except (a2_execution_state.A2ExecutionStateRefused,
            discovery_screen.DiscoveryScreenRefused, StopIteration) as exc:
        raise ControlRefused(f"A2 bank reference source cannot be reopened: {exc}") from exc
    if expected != dict(reference):
        raise ControlRefused("A2 bank reference differs from original source history")
    return bank

def append_a2_bank_reference(self, *, logical_id: str, plan,
                             frame_digest: str, bank) -> Mapping[str, Any]:
    """Durably bind one imported bank to its indexed original anchor history."""
    from . import a2_execution_state
    from . import experiment_plan as experiment_plan_module

    with self._mutex:
        self._require_active_locked()
        assert self._journal is not None
        if not isinstance(plan, experiment_plan_module.ExperimentPlan):
            raise ControlRefused("A2 bank reuse requires an ExperimentPlan")
        plan = experiment_plan_module.ExperimentPlan.from_dict(plan.to_dict())
        if plan.campaign_id != self.resolved.campaign_id:
            raise ControlRefused("A2 bank reuse plan and controller campaign differ")
        execution_id = a2_execution_state.execution_identity(
            campaign_id=self.resolved.campaign_id,
            config_generation=self.config_generation,
            config_digest=self.config_digest, logical_id=logical_id,
            plan_digest=plan.digest, frame_digest=frame_digest)
        prior_execution = self._a2_logical_executions.get(logical_id)
        if prior_execution not in {None, execution_id}:
            raise ControlRefused("A2 logical execution changes fixed identity")
        entries = self._a2_execution_entries.get(execution_id, [])
        if entries:
            try:
                projection = a2_execution_state.project_transitions(
                    [item.payload for item in entries])
            except a2_execution_state.A2ExecutionStateRefused as exc:
                raise ControlRefused(f"A2 bank reuse history is invalid: {exc}") from exc
            reference = projection.bank_reference
            if reference is None:
                raise ControlRefused(
                    "A2 bank reference must precede every local phase event")
            expected, stored = self._a2_resolve_bank_source_locked(
                bank, target_plan_digest=plan.digest,
                target_frame_digest=frame_digest)
            if reference != expected:
                raise ControlRefused("A2 bank reuse retry changes its original source")
            return {"bank_reference": copy.deepcopy(reference),
                    "bank": stored.to_dict()}
        reference, stored = self._a2_resolve_bank_source_locked(
            bank, target_plan_digest=plan.digest,
            target_frame_digest=frame_digest)
        transition = a2_execution_state.make_transition(
            campaign_id=self.resolved.campaign_id,
            config_generation=self.config_generation,
            config_digest=self.config_digest,
            supervisor_incarnation=self.supervisor_incarnation,
            logical_id=logical_id, event=reference)
        try:
            a2_execution_state.project_transitions([transition])
            entry = self._journal.append(
                journal_module.KIND_A2_RUNTIME_EXECUTION, transition,
                campaign_id=self.resolved.campaign_id, record_id=execution_id)
            self._verify_journal_layout(self.store / "journal")
        except BaseException:
            self._poisoned = True
            raise
        self._journal_cursor = entry.seq
        self._a2_logical_executions[logical_id] = execution_id
        self._a2_execution_entries[execution_id] = [copy.deepcopy(entry)]
        return {"bank_reference": copy.deepcopy(reference),
                "bank": stored.to_dict()}

def _attest_a2_bank_reference(self, *, execution_id: str,
                              bank_reference: Mapping[str, Any], bank) -> bool:
    from . import a2_execution_state, discovery_screen

    with self._mutex:
        self._require_active_locked()
        entries = self._a2_execution_entries.get(execution_id)
        if not entries:
            return False
        try:
            projection = a2_execution_state.project_transitions(
                [item.payload for item in entries])
            if projection.bank_reference != bank_reference:
                return False
            stored = self._bank_for_a2_reference_locked(bank_reference)
        except (a2_execution_state.A2ExecutionStateRefused,
                discovery_screen.DiscoveryScreenRefused, ControlRefused):
            return False
        return stored.to_dict() == dict(bank)

def _attest_a2_runtime_history(self, *, execution_id: str,
                               plan_digest: str, frame_digest: str,
                               events) -> bool:
    """Private half of discovery_screen's actual-owner verifier mint."""
    from . import a2_execution_state

    with self._mutex:
        self._require_active_locked()
        entries = self._a2_execution_entries.get(execution_id)
        if not entries:
            return False
        try:
            projection = a2_execution_state.project_transitions(
                [item.payload for item in entries])
        except a2_execution_state.A2ExecutionStateRefused:
            return False
        return (projection.plan_digest == plan_digest
                and projection.frame_digest == frame_digest
                and tuple(projection.events) == tuple(events))

def replay_a2_runtime_execution(self, *, logical_id: str,
                                plan_digest: str,
                                frame_digest: str) -> Mapping[str, Any]:
    """Return one bounded indexed replay plus live current-owner attestation."""
    from . import a2_execution_state, discovery_screen

    with self._mutex:
        self._require_active_locked()
        execution_id = a2_execution_state.execution_identity(
            campaign_id=self.resolved.campaign_id,
            config_generation=self.config_generation,
            config_digest=self.config_digest, logical_id=logical_id,
            plan_digest=plan_digest, frame_digest=frame_digest)
        prior_execution = self._a2_logical_executions.get(logical_id)
        if prior_execution not in {None, execution_id}:
            raise ControlRefused("A2 logical execution has identity drift")
        entries = self._a2_execution_entries.get(execution_id, [])
        if not entries:
            return {"execution_id": execution_id, "logical_id": logical_id,
                    "plan_digest": plan_digest, "frame_digest": frame_digest,
                    "events": (), "pending_intents": (), "sealed_phases": (),
                    "bank_reference": None, "reused_bank": None,
                    "bank_verifier": None,
                    "journal_entry_ids": (), "journal_cursor": 0,
                    "history_digest": None, "phase_verifier": None}
        try:
            projection = a2_execution_state.project_transitions(
                [item.payload for item in entries])
            verifier = (discovery_screen.attest_controller_phase_history(
                self, execution_id=execution_id, plan_digest=plan_digest,
                frame_digest=frame_digest, events=projection.events)
                if projection.events else None)
            reused_bank = None
            bank_verifier = None
            if projection.bank_reference is not None:
                stored = self._bank_for_a2_reference_locked(
                    projection.bank_reference)
                reused_bank = stored.to_dict()
                bank_verifier = discovery_screen.attest_controller_bank_reference(
                    self, execution_id=execution_id,
                    bank_reference=projection.bank_reference, bank=reused_bank)
        except (a2_execution_state.A2ExecutionStateRefused,
                discovery_screen.DiscoveryScreenRefused) as exc:
            raise ControlRefused(f"A2 runtime replay refused: {exc}") from exc
        return {"execution_id": execution_id, "logical_id": logical_id,
                "plan_digest": plan_digest, "frame_digest": frame_digest,
                "events": tuple(copy.deepcopy(list(projection.events))),
                "pending_intents": tuple(copy.deepcopy(list(
                    projection.pending_intents))),
                "sealed_phases": projection.sealed_phases,
                "bank_reference": copy.deepcopy(projection.bank_reference),
                "reused_bank": copy.deepcopy(reused_bank),
                "bank_verifier": bank_verifier,
                "journal_entry_ids": tuple(item.event_id for item in entries),
                "journal_cursor": entries[-1].seq,
                "history_digest": projection.history_digest,
                "phase_verifier": verifier}

def _require_retention_coverage_locked(self) -> None:
    """Every owner class registered, current (re-derived now) and gap-free."""
    missing = []
    for owner_class in retention_catalog_module.OWNER_CLASSES:
        record = self._retention_owner_records.get(owner_class)
        if record is None:
            missing.append(f"missing:{owner_class}")
        elif record.record_digest != self._derive_retention_owner_locked(
                owner_class).record_digest:
            missing.append(f"stale:{owner_class}")
        else:
            missing.extend(f"gap:{owner_class}:{gap}" for gap in record.gaps)
    if missing:
        raise ControlRefused(
            "native retention owner coverage is incomplete: " + ",".join(missing))

def install_native_retention_catalog(
        self, seed, *, runtime_anchors, model_preparations, artifact_root,
        runtime_recipes=None, runtime_recipe_snapshots=None) -> dict[str, Any]:
    """Durably install the one token-built static catalog before driver work."""
    if not isinstance(seed, retention_catalog_module.NativeRetentionCatalogSeed):
        raise TypeError("seed must be NativeRetentionCatalogSeed")
    expected = retention_catalog_module.build_seed(
        self.resolved, runtime_anchors, config_digest=self.config_digest,
        model_preparations=model_preparations, runtime_recipes=runtime_recipes,
        runtime_recipe_snapshots=runtime_recipe_snapshots,
        artifact_root=artifact_root)
    if seed is not expected and seed != expected:
        raise ControlRefused("native retention catalog was not rederived from owning inputs")
    with self._mutex:
        self._require_active_locked()
        if self.snapshot_version != 3 or seed.campaign_id != self.resolved.campaign_id \
                or seed.config_digest != self.config_digest \
                or seed.manifest_digest != self.resolved.manifest_digest:
            raise ControlRefused("native retention catalog binding differs")
        if self._retention_catalog_event is not None:
            if self._retention_catalog_seed != seed:
                raise ControlRefused("native retention catalog is already installed differently")
            return copy.deepcopy(self._retention_catalog_event)
        row = retention_catalog_module.make_install_event(
            seed, config_generation=self.config_generation,
            supervisor_incarnation=self.supervisor_incarnation)
        assert self._journal is not None
        entry = self._append_durable(
            journal_module.KIND_RETENTION_CATALOG_INSTALLED, row,
            record_id=seed.seed_digest)
        self._retention_catalog_seed = seed
        self._retention_catalog_event = copy.deepcopy(row)
        self._register_retention_owners_locked(*retention_catalog_module.OWNER_CLASSES)
        return copy.deepcopy(row)

def _register_retention_owners_locked(self, *owner_classes: str) -> None:
    """Owner write hook: record each class's own roots for the current generation."""
    seed = self._retention_catalog_seed
    if seed is None:
        self._retention_owner_records = {}
        return
    for owner_class in owner_classes:
        record = self._derive_retention_owner_locked(owner_class)
        if getattr(self._retention_owner_records.get(owner_class), "record_digest", None) \
                != record.record_digest:  # AKU-10f3: journal each changed record
            self._append_durable(journal_module.KIND_RETENTION_OWNER_RECORDED,
                                 owner_journal.make_payload(record))
        self._retention_owner_records[owner_class] = record

def _derive_retention_owner_locked(self, owner_class: str):
    return retention_catalog_module.derive_owner_record(
        owner_class, seed=self._retention_catalog_seed,
        generation=self._retention_generation_locked(),
        candidate_records=self._candidate_completed_records,
        native_records=self._native_records,
        worker_results=tuple(self._retention_worker_results.values()),
        store_root=str(self.store))

def _retention_generation_locked(self) -> tuple[Any, ...]:
    return (self._retention_catalog_seed.seed_digest, self.config_generation)

def _retention_frontier_locked(self) -> tuple[Any, ...]:
    seed = self._retention_catalog_seed
    if seed is None:
        raise ControlRefused("native retention catalog is unavailable")
    return (seed.seed_digest,
            schemas.content_hash(seed.to_dict()),
            self._candidate_projection_position,
            self._worker_lifecycle_revision,
            tuple(sorted(self._native_payload_digests.items())),
            tuple(sorted((key, row["transition_id"])
                         for key, row in self._driver_issued.items())),
            tuple(sorted(self._driver_settled)),
            self.supervisor_incarnation, self.control_revision,
            tuple(sorted((key, row.record_digest)
                         for key, row in self._retention_owner_records.items())))

def native_retention_catalog_capture(self) -> Mapping[str, Any]:
    """Capture declarations and exact dynamic owner frontier; performs no file I/O."""
    with self._mutex:
        self._require_active_locked()
        frontier = self._retention_frontier_locked()
        return {
            "seed": self._retention_catalog_seed,
            "frontier": frontier,
            "candidate_records": copy.deepcopy(self._candidate_completed_records),
            "active_workers": copy.deepcopy(self._active_worker_events),
            "active_acquisitions": copy.deepcopy(self._active_acquisition_events),
            "native_records": copy.deepcopy(self._native_records),
            "driver_issued": copy.deepcopy(self._driver_issued),
            "driver_settled": copy.deepcopy(self._driver_settled),
            "generation": self._retention_generation_locked(),
            "owner_records": dict(self._retention_owner_records),
            "worker_results": copy.deepcopy(tuple(self._retention_worker_results.values())),
            "store_root": str(self.store),
        }

def _native_retention_catalog_capture_locked(self, candidate_state) -> Mapping[str, Any]:
    """Capture a settled candidate state while its replay context owns the mutex."""
    if not self._mutex._is_owned() or self._candidate_context_token is None:
        raise ControlRefused("native retention capture requires candidate owner context")
    capture = dict(self.native_retention_catalog_capture())
    capture["candidate_state"] = copy.deepcopy(candidate_state.to_dict())
    return capture

def validate_native_retention_frontier(self, frontier: tuple[Any, ...]) -> None:
    """Recheck a completed external catalog collection against current authority."""
    if not isinstance(frontier, tuple):
        raise TypeError("frontier must be a tuple")
    with self._mutex:
        self._require_active_locked()
        if frontier != self._retention_frontier_locked():
            raise ControlRefused("native retention catalog changed during collection")

def bind_prepared_retention_job(self, job, *, prepared, policy) -> None:
    """Bind an externally collected immutable plan to the unchanged owner frontier."""
    from .retention_consumer import RetentionJob
    if (not isinstance(job, RetentionJob)
            or not isinstance(prepared, retention_catalog_module.PreparedCatalogView)):
        raise TypeError("prepared retention binding requires native typed capability")
    from . import retention as retention_module
    from . import retention_consumer as retention_consumer_module
    snapshot = retention_consumer_module.collect_native_snapshot(prepared.view)
    expected_plan = retention_module.plan_retention(snapshot)
    if (job.plan != expected_plan
            or job.policy_digest != retention_consumer_module._policy_digest(policy)
            or tuple(job.selected_artifact_ids) != job.plan.expirable_ids[:len(
                job.selected_artifact_ids)]):
        raise ControlRefused("maintenance job differs from native preparation")
    with self._mutex:
        self._require_active_locked()
        current = self._retention_frontier_locked()
        if (prepared.frontier != current
                or job.plan.snapshot_id != self._retention_catalog_seed.seed_digest
                or job.plan.snapshot_generation != max(1, int(current[2]) + 1)
                or job.plan.snapshot_digest != prepared.view_digest):
            raise ControlRefused("native retention catalog changed during preparation")
        if prepared.view.uncertain_scopes:
            raise ControlRefused("native retention view has uncertain scopes")
        self._require_retention_coverage_locked()
        self._prepared_retention_jobs[job.plan.plan_digest] = (
            current, prepared.view_digest, job.policy_digest,
            tuple(job.selected_artifact_ids))

def native_retention_catalog_seed_digest(self) -> str | None:
    with self._mutex:
        self._require_active_locked()
        return (None if self._retention_catalog_seed is None
                else self._retention_catalog_seed.seed_digest)

def _operation_preview_locked(self, row: Mapping[str, Any]):
    """Derive the atomic scheduler preview of one v2 settlement from durable children."""
    assert self._scheduler_engine is not None
    children = list(self._driver_children.get(row["transition_id"], ()))
    if [child["digest"] for child in children] != list(row["children"]):
        raise ControlRefused("operation settlement children differ from the durable prefix")
    operation_accounting_module.check_outcome_against_children(
        row["outcome"], children, row["denial_ref"],
        enrollment_ref=row.get("enrollment_ref"),
        uncertain=len(self._driver_intent_resolutions.get(row["transition_id"], {})))
    single = operation_accounting_module.single_stage(children)
    if single != ("operation" in row) or (
            single and row["operation"] != children[0]["operation"]):
        raise ControlRefused("operation settlement names another operation kind")
    selection = scheduling.Selection.from_dict(row["selection"])
    if any(child["selection_digest"] != selection.digest for child in children):
        raise ControlRefused("operation child is bound to another selection")
    if single:  # the exact v1 scheduler transition over the one original receipt
        return self._scheduler_engine.preview_accounting(
            selection, children[0]["receipt"], outcome=row["outcome"])
    if children:
        return self._scheduler_engine.preview_operation_accounting(
            selection, operation_accounting_module.operation_children(children),
            outcome=row["outcome"])
    return self._scheduler_engine.preview_operation_cancellation(
        selection, denial_ref=row["denial_ref"])

def _apply_settlement_preview(self, preview) -> bool:
    assert self._scheduler_engine is not None
    if isinstance(preview, scheduling.AccountingPreview):
        return self._scheduler_engine.apply_accounting_preview(preview)
    return self._scheduler_engine.apply_operation_preview(preview)

def unified_driver_account_child(
        self, *, binding, terminal, receipt, role: str,
        operation: str = "source_build") -> dict[str, Any]:
    """Durably account one original provider child receipt of a selected operation.

    Accounting evidence only. Joins the issued transition, the exact terminal and
    the exact receipt the lifecycle retained; accepts no caller receipt mapping."""
    if not isinstance(binding, worker_lifecycle_module.AccountingBinding):
        raise ControlRefused("child accounting requires a typed accounting binding")
    if not isinstance(receipt, scheduling.HeldClaimReceipt):
        raise ControlRefused("child accounting requires the exact held receipt object")
    with self._mutex:
        self._require_active_locked()
        if self.snapshot_version != 3 or self._scheduler_engine is None \
                or self._worker_lifecycle is None:
            raise ControlRefused("child accounting requires the unified v3 controller")
        issued = self._driver_issued.get(binding.catalog_digest)
        try:
            binding.verify_issued(issued)
            binding.verify_receipt(receipt)
        except worker_lifecycle_module.LifecycleRefused as exc:
            raise ControlRefused(f"child accounting binding refused: {exc}") from exc
        transition = issued["transition_id"]
        try:  # AKU-06l3: an exact retry is a duplicate, also after a lost settle reply
            duplicate = child_proof_module.duplicate_child(
                self._driver_children.get(transition, ()), terminal, receipt, role,
                binding.digest)
        except child_proof_module.ChildProofRefused as exc:
            raise ControlRefused(f"child accounting {exc}") from exc
        if duplicate is not None:
            return duplicate
        if transition in self._driver_settled:
            raise ControlRefused("child accounting targets an already settled transition")
        exact, trusted = self._operation_child_proof_locked(terminal, binding.digest)
        if (exact != terminal or trusted is None or trusted != receipt
                or receipt.ownership_generation != terminal.worker_generation
                or receipt.allocation_generation != terminal.grant_generation):
            raise ControlRefused(
                "child accounting terminal/receipt is not the exact lifecycle proof")
        chain = self._driver_children.setdefault(transition, [])
        terminal_body = child_proof_module.terminal_body(terminal)
        if (role not in (*operation_accounting_module.ROLES,
                         operation_accounting_module.SINGLE_STAGE_ROLE)
                or terminal.request_id in self._driver_intent_resolutions.get(
                    transition, {})):
            raise ControlRefused("child accounting role/request is unsupported")
        # Every original invocation is its own child, in order, exactly once: a
        # fallback repeats its role (v2 attempt); exact retries returned above.
        last = chain[-1] if chain else None
        attempt = (operation_accounting_module.child_attempt(last) + 1
                   if last is not None and last["role"] == role else 0)
        if not operation_accounting_module.ordered_roles(
                operation, [*(c["role"] for c in chain), role]):
            raise ControlRefused("child accounting role is out of order")
        try:
            record = operation_accounting_module.make_child_record(
                campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation,
                config_digest=self.config_digest,
                supervisor_incarnation=self.supervisor_incarnation,
                catalog_id=issued["catalog_id"], transition_id=transition,
                selection_digest=scheduling.Selection.from_dict(
                    issued["selection"]).digest, operation=operation,
                role=role, binding_digest=binding.digest, terminal=terminal_body,
                receipt=receipt.to_dict(),
                disposition=("accepted" if terminal.accepted
                             and terminal.return_code == 0 else "diagnostic"),
                predecessor=chain[-1]["digest"] if chain else None, attempt=attempt)
        except operation_accounting_module.OperationAccountingRefused as exc:
            raise ControlRefused(f"child accounting refused: {exc}") from exc
        try:
            operation_accounting_module.validate_child_chain([*chain, record])
        except operation_accounting_module.OperationAccountingRefused as exc:
            raise ControlRefused(f"child accounting refused: {exc}") from exc
        self._join_child_intent_locked(record)
        self._verify_store()
        assert self._journal is not None
        entry = self._append_durable(
            journal_module.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED, record,
            record_id=record["digest"])
        chain.append(copy.deepcopy(record))
        return {"status": "accepted", "digest": record["digest"], "role": role}

def _operation_child_proof_locked(self, terminal, binding_digest):
    try:  # AKU-06l3: current lifecycle terminal, else the exact restored held index
        return child_proof_module.child_proof(
            self._worker_lifecycle, self._restored_held_accounting, terminal, binding_digest)
    except child_proof_module.ChildProofRefused as exc:
        raise ControlRefused(f"child proof refused: {exc}") from exc

def operation_child_proof(self, terminal, binding_digest: str):
    """Read-only (terminal, held receipt) proof of one accounted v2 child (AKU-06l3)."""
    with self._mutex:
        self._require_active_locked()
        if self._worker_lifecycle is None:
            raise ControlRefused("worker lifecycle is unavailable")
        return self._operation_child_proof_locked(terminal, binding_digest)

def _join_child_intent_locked(self, record: Mapping[str, Any]) -> None:
    """A source-build child joins its durable intent-before-launch (AKU-06l)."""
    if record["operation"] != "source_build" or record["role"] != "build":
        return
    intent = self._driver_child_intents.get(record["transition_id"], {}).get(
        record["request_id"])
    if intent is None or any(intent[name] != record[name] for name in (
            "binding_digest", "plan_digest", "lineage_id", "stage_id")):
        raise ControlRefused("build child has no matching durable intent before launch")

def unified_driver_child_intent(self, request: Any) -> dict[str, Any]:
    """Durably record a bound source-build child's intent before its launch.

    ``replayed`` marks an intent from an earlier controller lifetime: with no exact
    terminal it is accounting uncertainty, so the caller must not relaunch."""
    binding = getattr(request, "accounting_binding", None)
    if (not isinstance(request, worker_lifecycle_module.StageRequest)
            or request.stage != "build"
            or not isinstance(binding, worker_lifecycle_module.AccountingBinding)):
        raise ControlRefused("child intent requires a bound typed build request")
    with self._mutex:
        self._require_active_locked()
        if self.snapshot_version != 3 or self._scheduler_engine is None:
            raise ControlRefused("child intent requires the unified v3 controller")
        issued = self._driver_issued.get(binding.catalog_digest)
        try:
            binding.verify_issued(issued)
        except worker_lifecycle_module.LifecycleRefused as exc:
            raise ControlRefused(f"child intent binding refused: {exc}") from exc
        transition = issued["transition_id"]
        if transition in self._driver_settled:
            raise ControlRefused("child intent targets an already settled transition")
        try:
            row = operation_accounting_module.make_child_intent(
                campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation,
                config_digest=self.config_digest,
                supervisor_incarnation=self.supervisor_incarnation,
                catalog_id=issued["catalog_id"], transition_id=transition,
                selection_digest=scheduling.Selection.from_dict(
                    issued["selection"]).digest, operation="source_build",
                role="build", binding_digest=binding.digest,
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id,
                artifact_contract_digest=request.artifact_contract_digest)
        except operation_accounting_module.OperationAccountingRefused as exc:
            raise ControlRefused(f"child intent refused: {exc}") from exc
        if self._driver_intent_resolutions.get(transition):
            raise ControlRefused("a resolved uncertain build intent is never relaunched")
        intents = self._driver_child_intents.setdefault(transition, {})
        prior = intents.get(request.request_id)
        if prior is not None:
            if {**prior, "supervisor_incarnation": 0, "digest": ""} != {
                    **row, "supervisor_incarnation": 0, "digest": ""}:
                raise ControlRefused("child intent retry differs from durable record")
            return {"status": "duplicate", "digest": prior["digest"],
                    "replayed": prior["supervisor_incarnation"]
                    < self.supervisor_incarnation}
        self._verify_store()
        self._append_durable(journal_module.KIND_UNIFIED_DRIVER_CHILD_INTENT, row,
                             record_id=row["digest"])
        intents[request.request_id] = copy.deepcopy(row)
        return {"status": "accepted", "digest": row["digest"], "replayed": False}

def unified_driver_uncertain_children(self, transition_id: str) -> tuple[str, ...]:
    """Request ids with a durable intent but no accounted child: accounting uncertainty."""
    with self._mutex:
        self._require_active_locked()
        return self._uncertain_children_locked(transition_id)

def _uncertain_children_locked(self, transition_id: str) -> tuple[str, ...]:
    accounted = {c["request_id"] for c in self._driver_children.get(transition_id, ())}
    resolved = set(self._driver_intent_resolutions.get(transition_id, {}))
    return tuple(sorted(set(self._driver_child_intents.get(transition_id, {}))
                        - accounted - resolved))

def _check_intent_resolution_locked(self, row: Mapping[str, Any]) -> None:
    """A resolution names one exact unaccounted intent of an unsettled transition."""
    intent = self._driver_child_intents.get(row["transition_id"], {}).get(
        row["request_id"])
    if (intent is None or intent["digest"] != row["intent_digest"]
            or intent["supervisor_incarnation"] != row["intent_incarnation"]
            or intent["catalog_id"] != row["catalog_id"]
            or (row["campaign_id"], row["config_generation"], row["config_digest"])
            != (self.resolved.campaign_id, self.config_generation, self.config_digest)
            or row["transition_id"] in self._driver_settled
            or row["request_id"] in self._driver_intent_resolutions.get(
                row["transition_id"], {})
            or any(c["request_id"] == row["request_id"]
                   for c in self._driver_children.get(row["transition_id"], ()))):
        raise ControlRefused("resolution does not name one unaccounted durable intent")

def unified_driver_resolve_uncertain_intent(
        self, transition_id: str, request_id: str, *, reason: str) -> dict[str, Any]:
    """Explicitly journal an earlier lifetime's unaccounted build intent as an
    uncertain diagnostic (AKU-06l): the transition may then settle only ``failed``
    over its accounted prefix; the build is never relaunched or counted as success."""
    with self._mutex:
        self._require_active_locked()
        if self.snapshot_version != 3 or self._scheduler_engine is None:
            raise ControlRefused("intent resolution requires the unified v3 controller")
        prior = self._driver_intent_resolutions.get(transition_id, {}).get(request_id)
        if prior is not None:
            return {"status": "duplicate", "digest": prior["digest"]}
        intent = self._driver_child_intents.get(transition_id, {}).get(request_id)
        if intent is None:
            raise ControlRefused("resolution names no durable build intent")
        if self.worker_terminal_for_request(
                request_id=intent["request_id"], plan_digest=intent["plan_digest"],
                lineage_id=intent["lineage_id"], stage_id=intent["stage_id"]) is not None:
            raise ControlRefused("intent has an exact terminal; account it instead")
        try:
            row = operation_accounting_module.make_intent_resolution(
                intent, supervisor_incarnation=self.supervisor_incarnation,
                reason=reason)
        except operation_accounting_module.OperationAccountingRefused as exc:
            raise ControlRefused(f"intent resolution refused: {exc}") from exc
        self._check_intent_resolution_locked(row)
        self._verify_store()
        self._append_durable(journal_module.KIND_UNIFIED_DRIVER_CHILD_INTENT_RESOLVED,
                             row, record_id=row["digest"])
        self._driver_intent_resolutions.setdefault(transition_id, {})[request_id] = \
            copy.deepcopy(row)
        return {"status": "accepted", "digest": row["digest"]}

def unified_driver_intent_resolutions(self, transition_id: str) -> tuple[dict, ...]:
    with self._mutex:
        self._require_active_locked()
        return tuple(copy.deepcopy(list(
            self._driver_intent_resolutions.get(transition_id, {}).values())))

def unified_driver_children(self, transition_id: str) -> tuple[dict[str, Any], ...]:
    """Read the durable ordered child prefix of one transition (copies)."""
    with self._mutex:
        self._require_active_locked()
        return tuple(copy.deepcopy(self._driver_children.get(transition_id, ())))

def unified_driver_settle_v2(self, value: Mapping[str, Any]) -> dict[str, Any]:
    """Settle one selected operation once from its durable ordered child prefix.

    Carries child digests, never a caller-chosen receipt. An empty prefix with a
    ``denial_ref`` is the explicit zero-cost no-receipt cancellation. A registered
    settlement validator is the sole execution validator for v2 as for v1: it is
    consulted outside the mutex and must return the exact request (a v1-only
    validator therefore refuses v2)."""
    fields = {"schema", "catalog_id", "transition_id", "selection", "outcome",
              "children", "denial_ref"}
    if not isinstance(value, Mapping) or not fields <= set(value) \
            or set(value) - fields - {"evidence_refs", "enrollment_ref"} \
            or value.get("schema") != DRIVER_SETTLEMENT_SCHEMA_V2:
        raise ControlRefused("unified v2 settlement request fields/schema differ")
    request = copy.deepcopy(dict(value))
    with self._mutex:
        self._require_active_locked()
        checked = self._operation_settle_locked(request, append=False)
        if checked["status"] == "duplicate":
            return checked
        validator = self._driver_settlement_validator
        token, incarnation = self._lifetime_token, self.supervisor_incarnation
    if validator is not None:
        try:
            verified = validator(copy.deepcopy(request))
        except Exception as exc:
            raise ControlRefused(
                f"trusted operation settlement verification failed: {exc}") from exc
        if not isinstance(verified, Mapping) or dict(verified) != request:
            raise ControlRefused("trusted operation settlement verifier returned other data")
    with self._mutex:
        self._require_active_locked()
        if (token is not self._lifetime_token or incarnation != self.supervisor_incarnation
                or validator is not self._driver_settlement_validator):
            raise ControlRefused(
                "settlement verification belongs to a stale controller lifetime")
        return self._operation_settle_locked(request, append=True)

def _operation_settle_locked(self, request: dict[str, Any], *, append: bool
                             ) -> dict[str, Any]:
    """Derive (and, when ``append``, durably settle) one v2 operation under the mutex."""
    if self.snapshot_version != 3 or self._scheduler_engine is None:
        raise ControlRefused("operation settlement requires the unified v3 controller")
    issued = self._driver_issued.get(request["catalog_id"])
    if (issued is None or issued["transition_id"] != request["transition_id"]
            or issued["selection"] != request["selection"]):
        raise ControlRefused("settlement does not bind an issued transition")
    children = list(self._driver_children.get(request["transition_id"], ()))
    single = operation_accounting_module.single_stage(children)
    evidence = request.get("evidence_refs")
    if single != (evidence is not None) or (single and (
            not isinstance(evidence, list) or not evidence
            or any(not isinstance(i, str) or not i for i in evidence)
            or len(evidence) != len(set(evidence)))):
        raise ControlRefused("evidence refs are required exactly for one single-stage child")
    resolved = self._driver_intent_resolutions.get(request["transition_id"], {})
    refs = (list(evidence) if single else
            [f"lifecycle:{child['terminal_digest']}" for child in children]
            + ([f"denial:{request['denial_ref']}"]
               if request["denial_ref"] is not None else [])
            + [f"uncertain-intent:{row['digest']}" for row in resolved.values()])
    optional = ({"operation": children[0]["operation"]} if single else {}) | (
        {"enrollment_ref": request["enrollment_ref"]} if "enrollment_ref" in request
        else {})
    prior = self._driver_settled.get(request["transition_id"])
    if prior is not None:
        if (prior["schema"] == journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA_V2
                and prior["terminal_refs"] == refs
                and {k: prior.get(k) for k in ("operation", "enrollment_ref")}
                == {k: optional.get(k) for k in ("operation", "enrollment_ref")}
                and all(prior[name] == request[name] for name in (
                    "catalog_id", "transition_id", "selection", "outcome",
                    "children", "denial_ref"))):
            return {"schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                    "transition_id": request["transition_id"],
                    "status": "duplicate",
                    "accounting_projection_digest":
                        prior["after_projection_digest"]}
        raise ControlRefused("settlement retry conflicts with durable content")
    if self._uncertain_children_locked(request["transition_id"]):
        raise ControlRefused(
            "operation has a child intent without accounted cost; accounting is uncertain")
    event = {
        "schema": journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA_V2,
        "catalog_id": request["catalog_id"], "transition_id": request["transition_id"],
        "campaign_id": self.resolved.campaign_id,
        "config_generation": self.config_generation,
        "config_digest": self.config_digest,
        "supervisor_incarnation": self.supervisor_incarnation,
        "selection": request["selection"], "outcome": request["outcome"],
        "children": list(request["children"]), "denial_ref": request["denial_ref"],
        "terminal_refs": refs, "prior_projection_digest": "0" * 64,
        "after_projection_digest": "0" * 64, **optional}
    try:
        preview = self._operation_preview_locked(event)
    except (scheduling.SchedulingRefused,
            operation_accounting_module.OperationAccountingRefused) as exc:
        raise ControlRefused(f"operation settlement refused: {exc}") from exc
    event["prior_projection_digest"] = preview.prior.projection_digest
    event["after_projection_digest"] = preview.after.projection_digest
    violations = journal_module._validate_native_payload(
        journal_module.KIND_UNIFIED_DRIVER_SETTLED, event)
    if violations:
        raise ControlRefused("operation settlement is invalid: " + "; ".join(violations))
    if not append:
        return {"status": "checked"}
    self._verify_store()
    assert self._journal is not None
    self._append_durable(
        journal_module.KIND_UNIFIED_DRIVER_SETTLED, event,
        record_id=request["transition_id"],
        after=lambda: self._apply_settlement_preview(preview))
    self._driver_settled[request["transition_id"]] = copy.deepcopy(event)
    return {"schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
            "transition_id": request["transition_id"], "status": "accepted",
            "accounting_projection_digest": preview.after.projection_digest}

def unified_driver_settle(self, value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify outside the mutex, then journal and apply exact held accounting."""
    if isinstance(value, Mapping) and value.get("schema") == DRIVER_SETTLEMENT_SCHEMA_V2:
        return self.unified_driver_settle_v2(value)
    with self._mutex:
        self._require_active_locked()
        if not isinstance(value, Mapping):
            raise ControlRefused("unified settlement request must be a mapping")
        supplied = copy.deepcopy(dict(value))
        transition_id = supplied.get("transition_id")
        if isinstance(transition_id, str):
            prior_settlement = self._driver_settled.get(transition_id)
            if prior_settlement is not None:
                if prior_settlement["schema"] != journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA:
                    raise ControlRefused(
                        "settlement retry conflicts with durable content")
                semantic_names = (
                    "catalog_id", "transition_id", "selection", "receipt", "outcome",
                    "terminal_refs")
                expected_fields = {"schema", *semantic_names}
                if (set(supplied) != expected_fields
                        or supplied.get("schema") != DRIVER_SETTLEMENT_SCHEMA):
                    raise ControlRefused(
                        "settlement retry conflicts with durable content")
                semantic = {name: supplied[name] for name in semantic_names}
                durable = {name: prior_settlement[name] for name in semantic_names}
                if semantic != durable:
                    raise ControlRefused(
                        "settlement retry conflicts with durable content")
                return {
                    "schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                    "transition_id": transition_id, "status": "duplicate",
                    "accounting_projection_digest":
                        prior_settlement["after_projection_digest"],
                }
        validator = self._driver_settlement_validator
        if validator is None:
            raise ControlRefused("trusted unified settlement validator is unavailable")
        lifetime_token = self._lifetime_token
        incarnation = self.supervisor_incarnation
    try:
        verified = validator(supplied)
    except Exception as exc:
        raise ControlRefused(f"trusted unified settlement verification failed: {exc}") from exc
    if not isinstance(verified, Mapping):
        raise ControlRefused("trusted unified settlement verifier returned malformed data")
    row = copy.deepcopy(dict(verified))
    expected = {"schema", "catalog_id", "transition_id", "selection", "receipt",
                "outcome", "terminal_refs"}
    if set(row) != expected or row.get("schema") != DRIVER_SETTLEMENT_SCHEMA:
        raise ControlRefused("unified settlement request fields/schema differ")
    if (not isinstance(row["terminal_refs"], list) or not row["terminal_refs"]
            or any(not isinstance(item, str) or not item for item in row["terminal_refs"])
            or len(row["terminal_refs"]) != len(set(row["terminal_refs"]))):
        raise ControlRefused("unified settlement terminal refs are invalid")
    try:
        selection = scheduling.Selection.from_dict(row["selection"])
        receipt = scheduling.HeldClaimReceipt.from_dict(row["receipt"])
    except Exception as exc:
        raise ControlRefused(
            f"trusted unified settlement verifier returned invalid bindings: {exc}") from exc
    if row["outcome"] not in scheduling.OUTCOMES:
        raise ControlRefused("unified settlement outcome is unsupported")
    with self._mutex:
        self._require_active_locked()
        if (lifetime_token is None or lifetime_token is not self._lifetime_token
                or incarnation != self.supervisor_incarnation
                or validator is not self._driver_settlement_validator):
            raise ControlRefused(
                "settlement verification belongs to a stale controller lifetime")
        prior_settlement = self._driver_settled.get(row["transition_id"])
        if prior_settlement is not None:
            if prior_settlement["schema"] != journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA:
                raise ControlRefused("settlement retry conflicts with durable content")
            semantic = {name: row[name] for name in (
                "catalog_id", "transition_id", "selection", "receipt", "outcome",
                "terminal_refs")}
            durable = {name: prior_settlement[name] for name in semantic}
            if semantic != durable:
                raise ControlRefused("settlement retry conflicts with durable content")
            return {"schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                    "transition_id": row["transition_id"], "status": "duplicate",
                    "accounting_projection_digest":
                        prior_settlement["after_projection_digest"]}
        issued = self._driver_issued.get(row["catalog_id"])
        if (issued is None or issued["transition_id"] != row["transition_id"]
                or issued["selection"] != selection.to_dict()):
            raise ControlRefused("settlement does not bind an issued transition")
        assert self._scheduler_engine is not None
        preview = self._scheduler_engine.preview_accounting(
            selection, receipt, outcome=row["outcome"])
        event = {
            "schema": journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA,
            "catalog_id": row["catalog_id"], "transition_id": row["transition_id"],
            "campaign_id": self.resolved.campaign_id,
            "config_generation": self.config_generation,
            "config_digest": self.config_digest,
            "supervisor_incarnation": self.supervisor_incarnation,
            "selection": selection.to_dict(), "receipt": receipt.to_dict(),
            "outcome": row["outcome"], "terminal_refs": list(row["terminal_refs"]),
            "prior_projection_digest": preview.prior.projection_digest,
            "after_projection_digest": preview.after.projection_digest,
        }
        self._verify_store()
        assert self._journal is not None
        entry = self._append_durable(
            journal_module.KIND_UNIFIED_DRIVER_SETTLED, event,
            record_id=row["transition_id"],
            after=lambda: self._scheduler_engine.apply_accounting_preview(preview))
        self._driver_settled[row["transition_id"]] = copy.deepcopy(event)
        return {"schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                "transition_id": row["transition_id"], "status": "accepted",
                "accounting_projection_digest": preview.after.projection_digest}


for _method in (append_a2_unit_record, replay_a2_unit_records, append_a2_runtime_transition, _a2_resolve_bank_source_locked, _bank_for_a2_reference_locked, append_a2_bank_reference, _attest_a2_bank_reference, _attest_a2_runtime_history, replay_a2_runtime_execution, _require_retention_coverage_locked, install_native_retention_catalog, _register_retention_owners_locked, _derive_retention_owner_locked, _retention_generation_locked, _retention_frontier_locked, native_retention_catalog_capture, _native_retention_catalog_capture_locked, validate_native_retention_frontier, bind_prepared_retention_job, native_retention_catalog_seed_digest, _operation_preview_locked, _apply_settlement_preview, unified_driver_account_child, _operation_child_proof_locked, operation_child_proof, _join_child_intent_locked, unified_driver_child_intent, unified_driver_uncertain_children, _uncertain_children_locked, _check_intent_resolution_locked, unified_driver_resolve_uncertain_intent, unified_driver_intent_resolutions, unified_driver_children, unified_driver_settle_v2, _operation_settle_locked, unified_driver_settle):
    setattr(CampaignController, _method.__name__, _method)
