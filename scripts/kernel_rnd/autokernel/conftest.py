"""Host-isolation for every AutoKernel unit test.

Since 2026-10-04 the CPU/GPU measurement quiet window defaults ON (`run.py`,
`--cpu-measurement-gpu-quiet lock`): a CPU run holds the orchestrator's host-wide
gpu-quiet flock SHARED with its region claim, and a GPU measurement holds it EXCLUSIVE.
A unit test must never take either on the REAL host -- a live GPU bench or CPU
measurement would block it, or it would block them. So:

* `ORCHESTRATOR_TMP_DIR` points into the test's tmp dir, so any real orchestrator
  region or gpu-quiet lock a test reaches is a temporary file;
* `claim.DEVICE_LOCK` points into the test's tmp dir (tests that patch it themselves
  still win: their monkeypatch runs after this one);
* `claim.hold_gpu_quiet_measurement` is a recording no-op (`gpu_quiet_holds`) and
  `claim.gpu_quiet_preflight` a no-op; tests of the window itself inject their own
  `hold`, and `real_gpu_quiet` hands tests the unpatched originals;
* the planner's MANDATORY belief reader (`belief_context.RUNNER`, ROOT's
  `loop_beliefs.py` as a child) answers a canned, valid `ready` with empty evidence, so no
  unit test reads or ingests the canonical ledger or raises a real operator alarm. It is a
  fake ANSWER, still validated by `belief_context.validate_ready` -- never a bypass. Tests of
  the reader itself inject their own runner (`belief_ready_answer` builds the canned body).
"""
from contextlib import contextmanager
import json
import subprocess

import pytest

from .loop import belief_context, claim

_REAL_GPU_QUIET = {"hold_gpu_quiet_measurement": claim.hold_gpu_quiet_measurement,
                   "gpu_quiet_preflight": claim.gpu_quiet_preflight}


@pytest.fixture(autouse=True)
def _isolate_host_claims(tmp_path_factory):
    # Its OWN MonkeyPatch, not the shared `monkeypatch` fixture: requesting that here
    # would set it up before a test's `tmp_path` and so undo it after tmp_path's
    # teardown, running pytest's cleanup under a test's patched shutil.rmtree.
    root = tmp_path_factory.mktemp("host-claims")
    holds = []

    @contextmanager
    def hold_gpu_quiet_measurement(timeout_s: float = 1.0):
        holds.append("enter")
        try:
            yield {"device_id": "host", "gpu_quiet": "exclusive", "test_double": True}
        finally:
            holds.append("exit")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ORCHESTRATOR_TMP_DIR", str(root / "orchestrator-locks"))
        patch.setattr(claim, "DEVICE_LOCK", root / "gpu_device.mi210_0.lock")
        patch.setattr(claim, "hold_gpu_quiet_measurement", hold_gpu_quiet_measurement)
        patch.setattr(claim, "gpu_quiet_preflight", lambda: root / "gpu_quiet.lock")
        yield holds


@pytest.fixture
def gpu_quiet_holds(_isolate_host_claims):
    return _isolate_host_claims


@pytest.fixture
def real_gpu_quiet():
    return dict(_REAL_GPU_QUIET)


def belief_ready_answer(target=None, *, evidence=None, sync_action="already-fresh", **extra):
    """A canned, VALID `loop_beliefs.py prepare` ready answer (schema v1)."""
    state = {"schema": belief_context.LEDGER_STATE_SCHEMA,
             "ledger_path": "/test/ledger.jsonl", "frame_count": 3, "head_seq": 2,
             "head_hash": "f" * 64, "verified_at": "2026-10-21T00:00:00Z", "fresh": True,
             "freshness": {"sync_action": sync_action, "was_fresh": sync_action == "already-fresh",
                           "frames_emitted": 0}}
    body = {"status": "omitted", "reasons": ["inapplicable_target"], "frontier": 3,
            "as_of": "2026-10-21T00:00:00Z", "claim_ids": [], "contested_claim_ids": [],
            "withdrawn_claim_ids": [], "claims": [], "flagged": {"contested": [], "withdrawn": []},
            "runs": [], "omitted": [], "omitted_count": 0, "section": "", "section_sha256": "",
            "section_bytes": 0}
    return {"schema": belief_context.ANSWER_SCHEMA, "consumer": belief_context.CONSUMER,
            "status": "ready", "reason": None, "detail": "", "targets": [target or {}],
            "policy": {"use": "planner-context"}, "ledger_state": state,
            "ledger_state_digest": belief_context.state_digest(state),
            "evidence": {**body, **(evidence or {})}, "flag_counts": {"contested": 0,
                                                                     "withdrawn": 0},
            "known_good_error": None, "alarm": {"cleared": {}}, **extra}


def _belief_runner(argv, **_kw):
    command = argv[2] if len(argv) > 2 else ""
    if command == "prepare":
        body = belief_ready_answer()
    elif command == "check-seeds":
        items = json.loads(argv[argv.index("--items-json") + 1])
        body = {"schema": belief_context.SEED_CHECK_SCHEMA, "checked_against": None,
                "checked_against_digest": None, "admissible_ids": [], "flagged_ids": [],
                "items": [{**item, "admissible": False, "reasons": ["no_known_good_state"],
                           "flagged_premises": []} for item in items]}
    else:
        body = {"key": "test-double", "raised": None}
    return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(body), stderr="")


@pytest.fixture(autouse=True)
def _isolate_belief_reader():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(belief_context, "RUNNER", _belief_runner)
        yield


@pytest.fixture
def belief_answer():
    """`belief_ready_answer`, for tests that build their own canned reader answers."""
    return belief_ready_answer
