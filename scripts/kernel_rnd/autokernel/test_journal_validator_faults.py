"""Environment/validator faults must not label durable native records corrupt."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

from . import journal as J

# Public native-record kinds and each owning validator's documented refusal.
VALIDATORS = [
    (
        J.KIND_A2_RUNTIME_EXECUTION,
        "a2_execution_state",
        "validate_transition",
        "A2ExecutionStateRefused",
    ),
    (J.KIND_ACTOR_PREPARATION, "actor_preparation_state", "validate_event", "ActorStateRefused"),
    (J.KIND_WORKER_LIFECYCLE, "worker_lifecycle", "validate_event", "LifecycleRefused"),
    (
        J.KIND_WORKER_ACQUISITION,
        "worker_lifecycle",
        "validate_acquisition_transition",
        "LifecycleRefused",
    ),
    (
        J.KIND_CAMPAIGN_COMMAND_V2,
        "worker_lifecycle",
        "validate_command_transition_v2",
        "LifecycleRefused",
    ),
    (J.KIND_CAMPAIGN_COMMAND_V3, "campaign_command_v2", "validate_transition", "CommandV2Refused"),
    (
        J.KIND_MAINTENANCE_EXECUTION,
        "maintenance_execution",
        "validate_event",
        "MaintenanceExecutionRefused",
    ),
    (
        J.KIND_RETENTION_CATALOG_INSTALLED,
        "native_retention_catalog",
        "validate_install_event",
        "NativeRetentionCatalogRefused",
    ),
]
FAULTS = [ImportError, NameError, AttributeError, TypeError, ValueError, RuntimeError]


def owner(name):
    return importlib.import_module(f"autokernel.loop.{name}")


def journal_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind,module,function,refused", VALIDATORS)
@pytest.mark.parametrize("package", ["autokernel", ""])
def test_documented_native_payload_refusals_stay_invalid_before_append(
    tmp_path, monkeypatch, kind, module, function, refused, package
):
    monkeypatch.setattr(J, "__package__", package)
    validator = owner(module)
    with pytest.raises(getattr(validator, refused)):
        getattr(validator, function)({})
    assert J._validate_native_payload(kind, {})
    journal = J.Journal(str(tmp_path))
    before = journal_bytes(tmp_path)
    with pytest.raises(ValueError, match="payload is invalid"):
        journal.append(kind, {})
    assert journal_bytes(tmp_path) == before


@pytest.mark.parametrize("kind,module,function,refused", VALIDATORS)
@pytest.mark.parametrize("fault", FAULTS)
def test_validator_programming_or_environment_fault_propagates_without_append(
    tmp_path, monkeypatch, kind, module, function, refused, fault
):
    def broken(payload):
        raise fault("fixture validator fault")

    monkeypatch.setattr(owner(module), function, broken)
    with pytest.raises(fault, match="fixture validator fault"):
        J._validate_native_payload(kind, {})
    journal = J.Journal(str(tmp_path))
    before = journal_bytes(tmp_path)
    with pytest.raises(fault, match="fixture validator fault"):
        journal.append(kind, {})
    assert journal_bytes(tmp_path) == before


def actor_intent():
    """A complete synthetic control event, accepted by the actual native validator."""
    state = owner("actor_preparation_state")
    zero = {key: 0.0 for key in state.BUDGET_KEYS}
    return {
        "schema": state.INTENT_SCHEMA,
        "event": "INTENT",
        "reservation_id": "reservation",
        "campaign_id": "campaign",
        "config_generation": 1,
        "config_digest": "1" * 64,
        "supervisor_id": "supervisor",
        "supervisor_incarnation": 2,
        "control_revision": 3,
        "catalog_id": "2" * 64,
        "transition_id": "3" * 64,
        "request_digest": "4" * 64,
        "stage_plan_digest": "5" * 64,
        "target_revision_digest": "6" * 64,
        "target_profile_digest": "7" * 64,
        "target_profile_receipt_digest": "8" * 64,
        "actor_profile_digest": "9" * 64,
        "backend_key": "fixture:model@low",
        "clock_domain": "monotonic:fixture",
        "deadline": 100.0,
        "occurred_at": "2026-10-30T00:00:00Z",
        "budgets": {key: 10 for key in state.BUDGET_KEYS},
        "debits": {**zero, "actor_calls_per_target": 1.0, "actor_calls_per_campaign": 1.0},
    }


@pytest.mark.parametrize("fault", FAULTS)
def test_scan_of_valid_durable_record_propagates_fault_and_recovers(tmp_path, monkeypatch, fault):
    journal = J.Journal(str(tmp_path), campaign_id="campaign")
    journal.initialize()
    event = journal.append(J.KIND_ACTOR_PREPARATION, actor_intent())
    before = journal_bytes(tmp_path)
    validator = owner("actor_preparation_state")
    with monkeypatch.context() as patch:

        def broken(payload):
            raise fault("fixture scan fault")

        patch.setattr(validator, "validate_event", broken)
        with pytest.raises(fault, match="fixture scan fault"):
            journal.scan(validate_payloads=True)
    assert journal_bytes(tmp_path) == before
    report = journal.scan(validate_payloads=True)
    assert not report.defects and [row.event_id for row in report.entries] == [event.event_id]


def test_scan_still_marks_documented_native_refusal_invalid(tmp_path, monkeypatch):
    journal = J.Journal(str(tmp_path), campaign_id="campaign")
    journal.initialize()
    journal.append(J.KIND_ACTOR_PREPARATION, actor_intent())
    validator = owner("actor_preparation_state")
    before = journal_bytes(tmp_path)
    with monkeypatch.context() as patch:

        def refused(payload):
            raise validator.ActorStateRefused("fixture invalid event")

        patch.setattr(validator, "validate_event", refused)
        report = journal.scan(validate_payloads=True)
        assert not report.entries and len(report.defects) == 1
        assert report.defects[0].reason == "payload fails validation"
    assert journal_bytes(tmp_path) == before
    assert not journal.scan(validate_payloads=True).defects


@pytest.mark.parametrize("standalone", [False, True])
def test_fresh_process_missing_yaml_is_an_import_fault_and_recovers(standalone):
    code = """
import importlib.util, sys
from pathlib import Path
from autokernel import journal as J
if STANDALONE:
    sys.path.insert(0, str(Path(J.__file__).parent))
    spec = importlib.util.spec_from_file_location("standalone_journal", J.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    J = module
previous = sys.modules.pop("yaml", None)
sys.modules["yaml"] = None
try:
    try:
        J._validate_native_payload(J.KIND_A2_RUNTIME_EXECUTION, {"schema":"x"})
    except ImportError:
        pass
    else:
        raise AssertionError("missing dependency became a payload violation")
    assert J._validate_native_payload(J.KIND_STOP_STATE, {})
finally:
    sys.modules.pop("yaml", None)
    if previous is not None:
        sys.modules["yaml"] = previous
# Recovery yields a normal documented payload refusal, not an environment fault.
assert J._validate_native_payload(J.KIND_A2_RUNTIME_EXECUTION, {"schema":"x"})
""".replace("STANDALONE", repr(standalone))
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
