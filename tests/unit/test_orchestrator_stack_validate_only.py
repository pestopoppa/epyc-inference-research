#!/usr/bin/env python3
"""`start --validate-only` must SHORT-CIRCUIT, not merely parse.

Found by `mainA` 2026-08-12: the flag was declared at
`orchestrator_stack.py:2795` with help text *"Validate stack template and exit"*
and *"Use --validate-only to check without launching"* — and **nothing read it**.
`grep validate_only` returned exactly one hit, the declaration. `main()` dispatched
`start -> cmd_start(args)` unconditionally, so anyone trusting the help text
launched the production stack instead.

A dry-run flag that is not wired is worse than no dry-run flag: it manufactures
the confidence to run the command. The blast radius is documented in the file
itself — `guard_against_running_bench` exists because a lifecycle action destroyed
1h09m of decision-gating measurement on 2026-07-27.

**These tests assert the BEHAVIOUR, not the declaration.** mainA's sharpest point
was that "a test asserting the flag EXISTS would pass today", so the load-bearing
assertion here is that `cmd_start` is NEVER CALLED. A test that only checked the
exit code, or only that argparse accepted the flag, would have passed against the
broken build.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def stack_mod():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_orchestrator_stack_vo", _ROOT / "scripts" / "server" / "orchestrator_stack.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_validate_only_helper_exists_and_is_reachable(stack_mod):
    """Weakest assertion, kept only as a locator for the ones below."""
    assert hasattr(stack_mod, "_cmd_validate_only")


def test_validate_only_never_calls_cmd_start(stack_mod, monkeypatch, capsys):
    """THE load-bearing assertion — and it must exercise main(), not mimic it.

    My first draft of this test simulated the dispatch branch itself
    (`rc = _cmd_validate_only(args) if args.validate_only else boom(args)`).
    That asserts the TEST's logic, not the CODE's: it passes against the broken
    build, which is precisely the vacuous-verification shape being documented
    across the fleet tonight. Rewritten to drive `main()` through argv with
    `cmd_start` booby-trapped, so a regression that removes the branch fails
    here instead of launching a production stack under a test run.
    """
    import scripts.server.stack_commands as sc

    def _boom(_args):
        raise AssertionError(
            "cmd_start was reached under --validate-only — the flag is inert "
            "again and the production stack would LAUNCH"
        )

    monkeypatch.setattr(sc, "cmd_start", _boom, raising=True)
    monkeypatch.setattr(
        stack_mod, "guard_against_running_bench", lambda *_a, **_k: True, raising=True
    )
    monkeypatch.setattr(
        stack_mod, "_cmd_validate_only", lambda _a: 0, raising=True
    )
    monkeypatch.setattr(
        sys, "argv", ["orchestrator_stack.py", "start", "--stack-profile", "default", "--validate-only"]
    )

    rc = stack_mod.main()
    assert rc == 0


def test_without_the_flag_dispatch_still_reaches_cmd_start(stack_mod, monkeypatch):
    """Negative control: prove the booby trap above CAN fire.

    Without this, `test_validate_only_never_calls_cmd_start` would also pass if
    main() simply never reached dispatch at all for an unrelated reason.
    """
    import scripts.server.stack_commands as sc

    reached = {"v": False}

    def _mark(_args):
        reached["v"] = True
        return 0

    monkeypatch.setattr(sc, "cmd_start", _mark, raising=True)
    monkeypatch.setattr(
        stack_mod, "guard_against_running_bench", lambda *_a, **_k: True, raising=True
    )
    monkeypatch.setattr(
        sys, "argv", ["orchestrator_stack.py", "start", "--stack-profile", "default"]
    )

    stack_mod.main()
    assert reached["v"] is True, "dispatch never reached cmd_start — the trap cannot fire"


def test_validate_only_reports_invalid_template_as_nonzero(stack_mod, monkeypatch):
    """Invalid template must EXIT NONZERO — a dry run that always says OK is no check."""
    import src.config.stack_templates as st

    monkeypatch.setattr(st, "load_template", lambda _p: object(), raising=True)
    monkeypatch.setattr(
        st,
        "validate_template",
        lambda _t: SimpleNamespace(
            valid=False, errors=["port conflict on 8070"], warnings=[], summary="FAIL"
        ),
        raising=True,
    )

    rc = stack_mod._cmd_validate_only(SimpleNamespace(stack_profile="default"))
    assert rc == 1


def test_validate_only_reports_valid_template_as_zero(stack_mod, monkeypatch):
    """The compliant path — a guard that only ever fails is also useless."""
    import src.config.stack_templates as st

    monkeypatch.setattr(st, "load_template", lambda _p: object(), raising=True)
    monkeypatch.setattr(
        st,
        "validate_template",
        lambda _t: SimpleNamespace(valid=True, errors=[], warnings=["ram close to cap"], summary="PASS"),
        raising=True,
    )

    rc = stack_mod._cmd_validate_only(SimpleNamespace(stack_profile="default"))
    assert rc == 0


def test_unloadable_template_fails_closed_rather_than_launching(stack_mod, monkeypatch):
    """A template that will not load must return 1, not fall through to a launch."""
    import src.config.stack_templates as st

    def _raise(_p):
        raise FileNotFoundError("no such profile")

    monkeypatch.setattr(st, "load_template", _raise, raising=True)
    rc = stack_mod._cmd_validate_only(SimpleNamespace(stack_profile="ghost"))
    assert rc == 1


def test_validate_only_works_while_a_bench_is_running(stack_mod, monkeypatch):
    """Validation must survive `guard_against_running_bench` REFUSING.

    Residual found by `mainA` on my first placement: the branch sat BELOW the
    bench guard, which covers start/stop/reload because those mutate the host.
    Validation mutates nothing, so refusing it there made pure config validation
    unavailable exactly when the host is busy — which is when you most want to
    check config without touching it. Fail-closed, never dangerous; just useless
    at the moment of need.

    The guard is forced to REFUSE here. If the branch ever slips back below it,
    this returns 2 and the test fails.
    """
    import scripts.server.stack_commands as sc

    monkeypatch.setattr(
        stack_mod, "guard_against_running_bench", lambda *_a, **_k: False, raising=True
    )
    monkeypatch.setattr(
        sc, "cmd_start",
        lambda _a: (_ for _ in ()).throw(AssertionError("cmd_start reached")),
        raising=True,
    )
    monkeypatch.setattr(stack_mod, "_cmd_validate_only", lambda _a: 0, raising=True)
    monkeypatch.setattr(
        sys, "argv",
        ["orchestrator_stack.py", "start", "--stack-profile", "default", "--validate-only"],
    )

    assert stack_mod.main() == 0, "bench guard blocked a validation that launches nothing"


def test_a_real_start_is_still_refused_while_a_bench_is_running(stack_mod, monkeypatch):
    """Negative control for the hoist: the guard must still bite a REAL start.

    Without this, the hoist could have been implemented by weakening the guard
    for every start, which is the opposite of what is wanted.
    """
    import scripts.server.stack_commands as sc

    monkeypatch.setattr(
        stack_mod, "guard_against_running_bench", lambda *_a, **_k: False, raising=True
    )
    monkeypatch.setattr(
        sc, "cmd_start",
        lambda _a: (_ for _ in ()).throw(AssertionError("cmd_start reached during a bench")),
        raising=True,
    )
    monkeypatch.setattr(
        sys, "argv", ["orchestrator_stack.py", "start", "--stack-profile", "default"]
    )

    assert stack_mod.main() == 2, "a real start must still be refused while a bench runs"


def test_dry_run_without_migrate_to_never_calls_cmd_start(stack_mod, monkeypatch):
    """`start --dry-run` alone LAUNCHED the production stack (2026-10-03, post-reboot).

    --dry-run is only read by the --migrate-to path; without it, dispatch fell through
    to cmd_start, the same inert-flag shape as --validate-only above. It must refuse
    (rc 2) and never reach cmd_start.
    """
    import scripts.server.stack_commands as sc

    monkeypatch.setattr(
        sc, "cmd_start",
        lambda _a: (_ for _ in ()).throw(AssertionError("cmd_start reached under --dry-run")),
        raising=True,
    )
    monkeypatch.setattr(
        stack_mod, "guard_against_running_bench", lambda *_a, **_k: True, raising=True
    )
    monkeypatch.setattr(sys, "argv", ["orchestrator_stack.py", "start", "--dry-run"])

    assert stack_mod.main() == 2


def test_dry_run_with_migrate_to_still_dispatches(stack_mod, monkeypatch):
    """Negative control: the migration plan path (--migrate-to NAME --dry-run) is untouched."""
    import scripts.server.stack_commands as sc

    reached = {"v": False}

    def _mark(_args):
        reached["v"] = True
        return 0

    monkeypatch.setattr(sc, "cmd_start", _mark, raising=True)
    monkeypatch.setattr(
        stack_mod, "guard_against_running_bench", lambda *_a, **_k: True, raising=True
    )
    monkeypatch.setattr(
        sys, "argv", ["orchestrator_stack.py", "start", "--migrate-to", "default", "--dry-run"]
    )

    stack_mod.main()
    assert reached["v"] is True, "the --migrate-to --dry-run plan path no longer reaches cmd_start"
