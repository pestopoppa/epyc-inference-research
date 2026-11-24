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
* the serial current-run pointer is redirected into the test's tmp dir, including
  in subprocesses, even when the caller inherited a production override;
* `claim.hold_gpu_quiet_measurement` is a recording no-op (`gpu_quiet_holds`) and
  `claim.gpu_quiet_preflight` a no-op; tests of the window itself inject their own
  `hold`, and `real_gpu_quiet` hands tests the unpatched originals;
* the planner's MANDATORY belief reader (`belief_context.RUNNER`, ROOT's
  `loop_beliefs.py` as a child) answers a canned, valid `ready` with empty evidence, so no
  unit test reads or ingests the canonical ledger or raises a real operator alarm. It is a
  fake ANSWER, still validated by `belief_context.validate_ready` -- never a bypass. Tests of
  the reader itself inject their own runner (`belief_ready_answer` builds the canned body).
* the bandwidth-census journal that `loop.program_text()` appends is an absent temporary
  path, so the planner program a test sees is program.md alone;
* `kernel_coverage`'s process-global launch-stderr capture starts off in every test.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

import pytest

from .loop import belief_context, claim, gpu_window, kernel_coverage, loop as loop_module, tailsplit

# Current hub view first (the /workspace shared clone can lag origin/main), then /workspace.
_REAL_ROOT_CANDIDATES = (Path("/mnt/raid0/llm/views/epyc-root-main"), Path("/workspace"))
_ORCHESTRATOR_CANDIDATES = (Path("/workspace/repos/epyc-orchestrator"),
                            Path("/mnt/raid0/llm/epyc-orchestrator"))

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
        patch.setenv("AUTOKERNEL_CURRENT_SERIAL_RUN_POINTER", str(root / "current-serial-run.json"))
        patch.setenv("AUTOKERNEL_CURRENT_SERIAL_RUN_TRUSTED_ROOT", str(tmp_path_factory.getbasetemp()))
        patch.setattr(claim, "DEVICE_LOCK", root / "gpu_device.mi210_0.lock")
        patch.setattr(claim, "hold_gpu_quiet_measurement", hold_gpu_quiet_measurement)
        patch.setattr(claim, "gpu_quiet_preflight", lambda: root / "gpu_quiet.lock")
        # G1: the MI210 window file, schedule and session bus are live shared state;
        # code that resolves them at call time sees temporary paths instead, and no
        # test inherits a real serial_run parent's device-hold fd.
        patch.setattr(gpu_window, "WINDOW_FILE", root / "gpu-window" / "mi210.json")
        patch.setattr(gpu_window, "SCHEDULE_FILE", root / "gpu-window" / "schedule.json")
        patch.setattr(gpu_window, "BUS_ROOT", root / "session-bus")
        patch.delenv(claim.INHERITED_DEVICE_LOCK_ENV, raising=False)
        # The live bandwidth-census journal (ak_runend hook, outside git) is appended to the
        # planner's program by program_text(); a test must see program.md alone, in-process and
        # in subprocesses. Tests of the journal pass their own path.
        absent_journal = root / "ak-bw-census" / "journal.md"
        patch.setenv("AK_BW_CENSUS_JOURNAL", str(absent_journal))
        patch.setattr(loop_module, "BW_CENSUS_JOURNAL", absent_journal)
        # The AKPL-P3 candidate spool is LIVE shared state (flow_status.py counts its cand-*.json as
        # real candidates); a pooled run in a test hands its records to a per-test spool instead.
        test_spool = root / "akchain-spool"
        patch.setenv("AKCHAIN_SPOOL", str(test_spool))
        patch.setattr(tailsplit, "SPOOL", test_spool)
        # run.main turns the process-global launch-stderr capture on for its store; a test
        # that ran main must not leave it on for the next one (it would swallow stderr).
        patch.setattr(kernel_coverage, "_CAPTURE", dict(kernel_coverage._CAPTURE, root=None))
        yield holds


@pytest.fixture
def real_store_guard():
    """Request this (usefixtures) to keep the real, refusing experiments.require_existing_store
    for a test that asserts a wrong --store is never created."""


@pytest.fixture(autouse=True)
def _loop_tests_may_create_stores(request):
    """C2a/C2b: production refuses a `--store` with no experiments.db and a missing
    inbox. The pre-existing loop tests drive `run._main` on fresh tmp dirs; they get a
    permissive guard that materialises the inbox. Tests outside `loop/` (the guard's
    own tests live in controller/) see the real, refusing function."""
    from pathlib import Path as _P
    if ("/loop/" not in str(request.node.fspath)
            or "real_store_guard" in request.fixturenames):
        yield
        return
    from .controller import experiments

    def permissive(root, *, new_store=False):
        (_P(root) / "inbox").mkdir(parents=True, exist_ok=True)
        return _P(root)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(experiments, "require_existing_store", permissive)
        yield


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
def _isolate_belief_reader(tmp_path_factory):
    """The canned reader above replaces the child process, but the ROOT checkout is resolved
    before the child is called (EPYC_ROOT_REPO, then EPYC_ROOT; production never guesses one).
    When the caller names no root, give the test a throwaway one with the adapter dir, so
    canned-reader tests do not fail on the caller's environment. A caller-named root is left
    alone: integration tests read real root files through it. Tests that exercise resolution
    itself still `monkeypatch.delenv`/`setenv` and override this."""
    import os
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(belief_context, "RUNNER", _belief_runner)
        if not (os.environ.get("EPYC_ROOT_REPO") or os.environ.get("EPYC_ROOT")):
            # Several tests read real ROOT files (the VB-AK-SEAT contract, the real-mask
            # verifier, the PII hook, the legacy-serving adapter) and skip or fail on a
            # throwaway root. Prefer a real checkout when one is present on this host; the
            # empty root is the last resort for a runner that has none.
            real = next((c for c in _REAL_ROOT_CANDIDATES
                         if (c / "scripts/vidya/adapters").is_dir()), None)
            root = real or tmp_path_factory.mktemp("epyc-root")
            (root / "scripts/vidya/adapters").mkdir(parents=True, exist_ok=True)
            patch.setenv("EPYC_ROOT_REPO", str(root))
            patch.setenv("EPYC_ROOT", str(root))
        if not os.environ.get("EPYC_ORCHESTRATOR_REPO"):
            owner = next((c for c in _ORCHESTRATOR_CANDIDATES
                          if (c / "src/runtime/cpu_region_lock.py").is_file()), None)
            if owner is not None:
                patch.setenv("EPYC_ORCHESTRATOR_REPO", str(owner))
        yield


@pytest.fixture
def belief_answer():
    """`belief_ready_answer`, for tests that build their own canned reader answers."""
    return belief_ready_answer


@pytest.fixture(autouse=True)
def _numerics_chain_off_unless_asked(monkeypatch):
    """AKX-ALL-31.DEFAULT (2026-11-18) turned the numerics chain ON by default. Unit tests that
    fake a passing test-backend-ops run print no AK_NUMERICS_V1 receipt, so they opt out here;
    the numerics tests set the flag (or clear the env to test the default) themselves."""
    monkeypatch.setenv("AK_NUMERICS_GATES", "0")
    # AKX-ALL-29.5a: same for the default-on GPU guard-band receipt requirement.
    monkeypatch.setenv("AK_GUARD_BANDS_GATE", "0")
