#!/usr/bin/env python3
"""test_surface_seam.py — the AK6 seam, and the restart chaos test.

WHY A THIRD SUITE, WHEN BOTH HALVES ARE GREEN
---------------------------------------------
`surface/test_dashboard_contract.py` proves the PRODUCER cannot make a dead loop
look alive. `epyc-root/tests/test_dashboard_panels*.py` prove the CONSUMER cannot
render absence as silence. Neither can fail when the two halves *disagree*, and
they were written by different agents against a schema only one of them owns —
which is exactly where two green modules produce a broken page. This suite is the
only place where the real producer writes a real file that the real hub reads.

It found two disagreements, both of which every existing test passed over:

  * the hub OVERWROTE `contract_version` — a key the producer owns and writes as
    the integer `2` — with the string `"v2"`, so the document served at
    `/api/kernel` no longer validated under its own producer's validator; and
  * `static/kernel.html` renders a v1 run log and nothing else, so a FULLY
    REPORTED contract v2 drew an empty page over the sentence *"no runs recorded
    yet — the kernel-R&D loop has not exported any results"*. The producer was
    alive, all seven owners had reported, and the page said the loop had exported
    nothing: the absence-tolerance scar rebuilt in the render layer, pointing the
    wrong way.

THE SCAR THIS WHOLE SURFACE EXISTS FOR
--------------------------------------
> Today's `/kernel` page is **absence-tolerant over a missing directory** — it
> renders clean when its producer is dead, which is the exact shape of AutoPilot
> dying at trial 1302 and staying dead ~23 HOURS with every dashboard green.

`RestartChaosTest` reproduces that incident end to end: a producer that is alive
and reporting, then dies, then time passes, and the board goes from green to
NAMING it. Absence tolerance is preserved throughout — no step may raise, and the
last step deletes the export entirely and still renders — but no step may read
clean (`assertNotClean`).

NO PROCESS MANAGEMENT. Nothing here starts, stops, signals, kills or restarts
anything: the hub on :8100 is a live service other sessions are using, and a test
that needs a real process is the wrong test for this. Death is simulated the way
death actually presents to a consumer — the producer stops writing, and the clock
moves on — with the clock INJECTED (`panels.envelope(now=...)`) and the file, the
producer, the reader, the watchdog and the fold all real.
`test_this_suite_manages_no_processes` audits this module's own source for it.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO PRODUCTION WRITE. Every byte written
goes into a `tempfile.TemporaryDirectory` this suite creates and removes.

Run:
    python3 -m pytest scripts/kernel_rnd/autokernel/surface/test_surface_seam.py
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

# The producer half. Package-relative (README convention) so this suite shares
# ONE copy of `schemas` with the modules under test; a flat import would load a
# second copy and every `isinstance` seam would silently stop biting.
from .. import schemas as S
from ..controller import state_machine as SM
from ..release import test_packager as FPK
from . import dashboard_contract as DC
from . import test_dashboard_contract as F


# ---------------------------------------------------------------------------
# The consumer half lives in the OTHER repository
# ---------------------------------------------------------------------------
#: Where `epyc-root` is. Both entries are the same clone (CLAUDE.md,
#: "Working-tree identity": `/workspace` is a symlink to `/mnt/raid0/llm/epyc-root`
#: and parallel sessions on either path share one checkout), listed so this suite
#: runs from either side.
_EPYC_ROOT_CANDIDATES = ("/workspace", "/mnt/raid0/llm/epyc-root")


def _epyc_root() -> pathlib.Path:
    """The epyc-root checkout, or a hard failure naming the seam it cannot check.

    DELIBERATELY NOT A SKIP. A seam test that skips when one half is missing
    reports "green" for a check that never ran, which is the same defect as a
    panel that renders clean when its producer is dead — one level up, in the
    test suite. If the consumer half cannot be found, the gate must say so.
    """
    for candidate in _EPYC_ROOT_CANDIDATES:
        if (pathlib.Path(candidate) / "dashboard" / "server.py").is_file():
            return pathlib.Path(candidate)
    raise RuntimeError(
        "the AK6 consumer half (epyc-root/dashboard/server.py) was not found at "
        f"any of {_EPYC_ROOT_CANDIDATES}. This suite is the ONLY thing that "
        "checks the producer and the hub against each other; it fails rather "
        "than skipping, because a skipped seam check reads exactly like a "
        "passing one.")


_ROOT = _epyc_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard import panels  # noqa: E402  (path set above)
from dashboard import server  # noqa: E402

KERNEL_HTML = _ROOT / "dashboard" / "static" / "kernel.html"

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)
HOUR = 3600.0
DAY = 86400.0


def _iso(when: dt.datetime) -> str:
    return when.isoformat()


def _epoch(when: dt.datetime) -> float:
    return when.timestamp()


class _Wired:
    """Point the hub's two env-overridable artifacts at a temp directory.

    The ENVIRONMENT is what is set, not just the module attribute, because
    `RestartChaosTest` reloads `dashboard.server` to simulate a hub restart and a
    reload re-reads the environment. Nothing outside the temp directory is
    touched, and no service is signalled — the running hub keeps its own paths.
    """

    def __init__(self, test: unittest.TestCase) -> None:
        # `dir=` keeps the temp tree on the array: `/tmp` is scratch, and
        # `assert_exportable_destination` refuses to export there (correctly).
        tmp = tempfile.TemporaryDirectory(dir="/mnt/raid0/llm")
        test.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.kernel = self.dir / "kernel_dashboard.json"
        self.outcome = self.dir / "outcome_contract.json"
        for name, value in (("KERNEL_DASHBOARD_JSON", self.kernel),
                            ("AUTOPILOT_OUTCOME_JSON", self.outcome)):
            previous = os.environ.get(name)
            os.environ[name] = str(value)
            test.addCleanup(self._restore, name, previous)
        test.addCleanup(self.reload_hub)
        self.reload_hub()

    @staticmethod
    def _restore(name: str, previous) -> None:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous

    def reload_hub(self):
        """Re-import the hub — a HUB RESTART, with no process involved.

        `importlib.reload` re-executes `server.py` in place: the module re-reads
        the environment and, crucially, drops `_watchdog_state`. That in-process
        memory is the watchdog's SECOND arm; the first arm (the producer's own
        semantic timestamp going stale) is stateless, and the point of the
        restart steps below is that the detector survives on the stateless arm.
        """
        return importlib.reload(server)


# ---------------------------------------------------------------------------
# Contract fixtures dated by a VIRTUAL clock
# ---------------------------------------------------------------------------
def contract(when: dt.datetime, *, seq: int = 7,
             champion_when: dt.datetime | None = None,
             exported_when: dt.datetime | None = None,
             stopped_state: str | None = None) -> dict:
    """A real v2 contract whose journaled RECORDS are dated `when`.

    Every liveness section (`campaign`, `champion`, `backend_standing`,
    `release_package`) carries a record timestamp of `when`, so `produced_at` —
    which the producer derives, and its validator recomputes — is `when`. The
    live host readings (`headroom`, `resource_claims`) are dated `exported_when`
    on purpose: they are what an exporter that is still alive over a dead loop
    would report, and they must not lift `produced_at`.
    """
    exported = exported_when or when
    ts, champion_ts, exported_ts = _iso(when), _iso(champion_when or when), _iso(exported)
    controller = F.controller(
        seq=seq,
        state=stopped_state or SM.POST_RUN_CRITIC,
        stopped=stopped_state is not None,
        last_transition=F.transition(
            seq=seq, at=ts, to_state=stopped_state or SM.POST_RUN_CRITIC))
    return F.document(
        controller=controller,
        champion=F.champion_record(created_at=champion_ts),
        readiness=F.readiness_report(computed_at=ts),
        release_package=FPK.release_package(created_at=ts),
        headroom=F.headroom(measured_at=exported_ts),
        claims=DC.ClaimsObservation(receipts=(F.claim_receipt(),),
                                    observed_at=exported_ts),
        exported_at=exported_ts)


def outcome_contract(when: dt.datetime, *, trial: int = 1302,
                     paused: bool = False) -> dict:
    """The orchestrator autopilot's outcome contract, as the hub reads it.

    Hand-built, because there is no exporter for it yet (`phase_health_report.py`
    only prints to stdout) — which is precisely why the `outcome` panel is the
    one the trial-1302 outage is named after, and why it is watched anyway.
    """
    progress = {"status": "paused" if paused else "ok",
                "latest_trial_id": trial,
                "rates": {"keepable": 0.41, "wasted_eval": 0.12},
                "blockers": []}
    if paused:
        progress["paused"] = True
    return {"generated_at": _iso(when), "outcome_progress": progress}


class _SeamCase(unittest.TestCase):
    """Shared wiring: a temp export path, and readers that go through the hub."""

    def setUp(self) -> None:
        self.wire = _Wired(self)

    # -- writing --------------------------------------------------------------
    def export(self, document: dict) -> dict:
        """Write with the REAL producer, to the path the REAL hub reads."""
        receipt = DC.export_contract(document, path=str(self.wire.kernel))
        self.assertEqual(receipt.path, str(self.wire.kernel))
        return document

    def export_outcome(self, document: dict) -> dict:
        self.wire.outcome.write_text(json.dumps(document), encoding="utf-8")
        return document

    # -- reading, through the hub's own code path ------------------------------
    def kernel_env(self, now: float) -> dict:
        present, data = server._read_kernel_contract()
        obs = server._kernel_observation(data, artifact_present=present)
        return server._panel_envelope("kernel", obs, now=now)

    def outcome_env(self, now: float) -> dict:
        present, data = server._read_outcome_contract()
        obs = server._outcome_observation(data, artifact_present=present)
        return server._panel_envelope("outcome", obs, now=now)

    def fold(self, now: float, **override) -> dict:
        """The real `/api/health` fold over a healthy universe plus overrides.

        The universe is the REGISTRY (`panels.fold`'s own rule), so a panel that
        drops out is named rather than subtracted. The non-overridden panels are
        given healthy envelopes rather than being read off this host, so the
        verdict under test is caused by the panel under test.
        """
        envs = {}
        for name, src in panels.PANELS.items():
            if src.kind in panels.LIVE_KINDS:
                obs = panels.Observation(artifact_present=True, timestamp=None,
                                         source="live-scan", populated=True)
            else:
                obs = panels.Observation(artifact_present=True, timestamp=now - 60,
                                         source=src.timestamp_field, populated=True,
                                         watermark=f"{name}:1")
            envs[name] = panels.envelope(src, obs, now=now)
        envs.update(override)
        return panels.fold(envs)

    # -- the assertion this suite is built around ------------------------------
    def assertNotClean(self, env: dict, why: str) -> None:
        """Fail if NOTHING on the wire distinguishes this panel from a healthy one.

        "Clean" is defined positively, so the assertion cannot be satisfied by a
        cosmetic difference: fresh, observed, watchdog quiet, no unreported
        sections, and no absence sentence travelling with the card. A panel in
        that state is one an operator reads as "nothing is wrong".
        """
        clean = (env["staleness_class"] == panels.CLASS_FRESH
                 and env["reporting"] == panels.REPORTING_OBSERVED
                 and env["watchdog"]["state"] in (panels.WATCHDOG_OK,
                                                  panels.WATCHDOG_UNWATCHED)
                 and not env["unreported"]
                 and "absence_means" not in env)
        self.assertFalse(clean, f"{why}: the panel reads CLEAN — {env}")

    def assertClean(self, env: dict, why: str) -> None:
        """The compliant-path control for `assertNotClean`.

        Without it, `assertNotClean` would pass on a surface that is permanently
        alarmed — which is a dashboard nobody reads, i.e. the same failure with
        the opposite sign.
        """
        self.assertEqual(env["staleness_class"], panels.CLASS_FRESH, why)
        self.assertEqual(env["reporting"], panels.REPORTING_OBSERVED, why)
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_OK, why)
        self.assertEqual(env["unreported"], [], why)
        self.assertNotIn("absence_means", env, why)


# =============================================================================
# 1 — the two halves name the same things
# =============================================================================
class TheTwoHalvesNameTheSameThingsTest(_SeamCase):
    """Constants the hub COPIES rather than imports.

    The hub is stdlib-only and must never import `autokernel` to render a page (a
    consumer that needs its producer's code installed goes dark when the
    producer's repo moves), so the schema strings, the section status and the
    export path are literals over there. Literals drift silently; this class is
    the thing that makes them not.
    """

    def test_the_schema_strings_are_identical(self):
        self.assertEqual(server.KERNEL_SCHEMA_V2, S.SCHEMA_KERNEL_DASHBOARD_V2)
        self.assertEqual(server.KERNEL_SCHEMA_V1, S.SCHEMA_KERNEL_DASHBOARD_V1)
        self.assertEqual(server.KERNEL_SECTION_OBSERVED, S.SECTION_OBSERVED)

    def test_the_registry_points_at_the_path_the_producer_writes(self):
        """The panel→producer registry's `evidence` is where the export lands.

        If the producer moves its output, this fails — instead of the hub
        silently reading a path nobody writes and rendering the absence of a file
        that exists somewhere else.
        """
        self.assertEqual(panels.PANELS["kernel"].evidence, DC.DEFAULT_EXPORT_PATH)

    def test_the_hub_default_is_the_producer_default(self):
        """Pinned against the SOURCE, because this process overrides the env var.

        Reading `server.KERNEL_DASHBOARD_JSON` here would read the temp path this
        suite set, so the check would pass while saying nothing.
        """
        source = (_ROOT / "dashboard" / "server.py").read_text(encoding="utf-8")
        self.assertIn(f'"{DC.DEFAULT_EXPORT_PATH}"', source)

    def test_the_registry_names_the_producing_module(self):
        stem = DC.MODULE_ID.split("/")[0]
        self.assertIn(stem, panels.PANELS["kernel"].producer)
        self.assertEqual(panels.PANELS["kernel"].producer_repo,
                         "epyc-inference-research")

    def test_the_absence_of_this_panel_has_a_declared_meaning(self):
        """The registry's own rule, checked at the seam: the sentence an operator
        reads when the card is blank must exist, and must say the producer is
        dead rather than that nothing happened."""
        means = panels.PANELS["kernel"].absence_means
        self.assertIn("NOBODY IS REPORTING", means)
        self.assertTrue(panels.PANELS["kernel"].absence_is_anomalous)


# =============================================================================
# 2 — a real contract round-trips into a real panel, field by field
# =============================================================================
class ARealContractRendersFieldByFieldTest(_SeamCase):

    def setUp(self) -> None:
        super().setUp()
        self.now = _epoch(T0) + HOUR
        self.doc = self.export(contract(T0))

    def payload(self) -> dict:
        return server.kernel_payload()

    def test_every_producer_field_survives_the_hub_unchanged(self):
        """The hub ADDS; it does not rewrite.

        THE DEFECT THIS CLOSES: `kernel_payload` assigned
        `data["contract_version"] = "v2"` over the producer's integer `2`. One
        key, changed type, silently, by the consumer — see the next test for what
        it cost. Hub-derived facts now live under `_contract_version`,
        `_freshness` and `_render`, which the producer will never collide with.
        """
        payload = self.payload()
        for key, value in self.doc.items():
            self.assertEqual(payload[key], value,
                             f"the hub altered the producer-owned key {key!r}")
        added = set(payload) - set(self.doc)
        self.assertEqual(added, {"_contract_version", "_freshness", "_render"})
        for key in added:
            self.assertTrue(key.startswith("_"), key)

    def test_what_the_hub_serves_still_validates_under_the_producers_validator(self):
        """The strongest statement of the seam: /api/kernel's body IS a contract.

        BITE: with the old in-place overwrite this fails with
        `contract_version: expected an integer, got str`.
        """
        self.assertEqual(S.validate_kernel_dashboard_v2(self.payload()), [])

    def test_the_panel_is_dated_by_produced_at_and_nothing_else(self):
        env = self.kernel_env(self.now)
        self.assertEqual(env["source"], "produced_at")
        self.assertEqual(env["timestamp"],
                         round(_epoch(T0), 3))
        self.assertEqual(env["age_s"], round(HOUR, 1))
        self.assertClean(env, "a loop that reported an hour ago")

    def test_the_exporters_own_clock_cannot_date_the_panel(self):
        """The producer's central guarantee, arriving intact at the consumer.

        The document is re-exported a month later with untouched RECORDS and a
        fresh `exported_at` and fresh live host readings — the exact shape of an
        exporter that is alive over a loop that is dead.
        """
        later = T0 + dt.timedelta(days=30)
        self.export(contract(T0, exported_when=later))
        env = self.kernel_env(_epoch(later))
        self.assertEqual(env["timestamp"], round(_epoch(T0), 3))
        self.assertEqual(env["staleness_class"], panels.CLASS_STALE)
        self.assertNotClean(env, "a month-dead loop with a live exporter")

    def test_the_evidence_named_is_the_file_actually_read(self):
        """The registry declares the DEFAULT; the envelope reports what was read.

        An envelope that names the default while the hub is reading an override
        sends an investigation to a file nobody wrote.
        """
        env = self.kernel_env(self.now)
        self.assertEqual(env["evidence"], str(self.wire.kernel))
        self.assertEqual(env["declared_evidence"], DC.DEFAULT_EXPORT_PATH)

    def test_all_seven_sections_arrive_with_their_status_and_record_time(self):
        payload = self.payload()
        self.assertEqual(set(payload["sections"]), set(S.DASHBOARD_SECTIONS))
        for name in S.DASHBOARD_SECTIONS:
            with self.subTest(section=name):
                served = payload["sections"][name]
                self.assertEqual(served, self.doc["sections"][name])
                self.assertIn(served["status"], S.DASHBOARD_SECTION_STATUSES)
                self.assertEqual(served["status"], S.SECTION_OBSERVED)
                if name in S.DASHBOARD_LIVENESS_SECTIONS:
                    # An observed LIVENESS section must carry the record time it
                    # observed, or it reads healthy while contributing nothing to
                    # `produced_at` — the hole this contract was built around.
                    self.assertIsNotNone(served["as_of"])

    def test_the_derived_section_dates_nothing_when_it_has_nothing_to_date(self):
        """`blocking_conditions` is observed with `as_of: null` on a healthy loop.

        It is not an oversight and it is not a liveness source: every condition
        in it is RESTATED from another section, so it dates itself from the
        conditions' own record times and has none when there are no conditions.
        Asserting a record time here would be asserting the exact hole that let a
        month-dead loop read `produced_at: now` — the exporter's clock filling in
        for a section no journaled record backs.
        """
        section = self.payload()["sections"][S.DASHBOARD_SECTION_BLOCKING]
        self.assertEqual(section["status"], S.SECTION_OBSERVED)
        self.assertIsNone(section["as_of"])
        self.assertEqual(section["open"], [])
        self.assertNotIn(S.DASHBOARD_SECTION_BLOCKING, S.DASHBOARD_LIVENESS_SECTIONS)
        # ...and the panel is still dated, by the sections that own records.
        self.assertEqual(self.payload()["produced_at"], _iso(T0))

    def test_the_run_identity_survives_so_two_exports_can_be_compared(self):
        run = self.payload()["producer"]["run"]
        self.assertEqual(run["campaign_id"], self.doc["campaign_id"])
        self.assertEqual(run["controller_seq"], 7)
        self.assertEqual(run["controller_state"], SM.POST_RUN_CRITIC)

    def test_both_halves_compute_the_SAME_absence_set(self):
        """Two independent implementations, one answer.

        The producer summarises with `schemas.dashboard_unreported_sections`; the
        hub DERIVES its own set from the sections and unions the producer's
        summary in (so a producer that under-reports its own absence cannot hand
        the hub a clean panel). Same document, same answer — checked over a
        healthy contract AND a degraded one, because agreeing only when
        everything is fine is the case that never mattered.
        """
        for absent in ((), ("champion",), ("champion", "release_package")):
            with self.subTest(absent=absent):
                overrides = {("controller" if a == "campaign" else a):
                             DC.Unreported(reason=f"{a} owner did not report")
                             for a in absent}
                doc = self.export(contract(T0, **{}) if not overrides
                                  else F.document(
                                      controller=F.controller(
                                          last_transition=F.transition(at=_iso(T0))),
                                      champion=overrides.get(
                                          "champion", F.champion_record(created_at=_iso(T0))),
                                      release_package=overrides.get(
                                          "release_package",
                                          FPK.release_package(created_at=_iso(T0))),
                                      readiness=F.readiness_report(computed_at=_iso(T0)),
                                      exported_at=_iso(T0)))
                env = self.kernel_env(self.now)
                self.assertEqual(
                    env["unreported"],
                    S.dashboard_unreported_sections(doc["sections"]))
                self.assertEqual(doc["unreported_sections"], env["unreported"])

    def test_a_silent_owner_carries_its_reason_all_the_way_to_the_card(self):
        reason = "champion store unreadable: the composition ledger is torn"
        self.export(F.document(champion=DC.Unreported(reason=reason)))
        payload = self.payload()
        self.assertEqual(payload["sections"]["champion"]["reason"], reason)
        env = payload["_freshness"]
        self.assertEqual(env["unreported"], ["champion"])
        # The registry's declared meaning travels WITH the incomplete report, so
        # a renderer cannot draw the gap without the sentence explaining it.
        self.assertIn("absence_means", env)
        verdict, why = panels.panel_verdict(env)
        self.assertEqual(verdict, panels.STATUS_ABSENT)
        self.assertIn("champion", why)
        self.assertNotEqual(self.fold(self.now, kernel=env)["status"],
                            panels.STATUS_OK)

    def test_a_contract_in_which_nobody_reported_is_not_an_empty_one(self):
        silent = {name: DC.Unreported(reason=f"{name} owner did not report")
                  for name in ("controller", "champion", "readiness", "headroom",
                               "blocking", "claims", "release_package")}
        doc = self.export(F.document(**silent))
        self.assertIsNone(doc["produced_at"])
        self.assertTrue(doc["degraded"])
        env = self.kernel_env(self.now)
        self.assertEqual(env["reporting"], panels.REPORTING_ABSENT)
        self.assertEqual(env["content"], panels.CONTENT_UNKNOWN)
        self.assertEqual(env["staleness_class"], panels.CLASS_MISSING)
        self.assertTrue(env["artifact_present"])  # the exporter DID run
        self.assertNotClean(env, "an export in which every owner was silent")
        self.assertEqual(self.fold(self.now, kernel=env)["status"],
                         panels.STATUS_ABSENT)

    def test_a_declared_stop_reads_idle_rather_than_dead(self):
        """COMPLIANT-PATH CONTROL. A controller that says it has stopped is
        allowed to be silent — only the producer may say so, and it does
        (`sections.campaign.stopped`). Without this the surface would alarm on
        every finished campaign and teach an operator to ignore it."""
        self.export(contract(T0, stopped_state=SM.RELEASE_PACKAGE_READY))
        env = self.kernel_env(_epoch(T0) + 10 * DAY)
        self.assertTrue(server.kernel_payload()["sections"]["campaign"]["stopped"])
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_IDLE)
        self.assertEqual(self.fold(_epoch(T0) + 10 * DAY, kernel=env)["status"],
                         panels.STATUS_OK)


# =============================================================================
# 3 — the render layer: what the page is given to draw
# =============================================================================
class TheRenderedPanelSaysWhatWasActuallyReadTest(_SeamCase):
    """The second seam defect: v2 has no run log, and the page only drew one.

    The page is JavaScript and cannot be executed here, so the guarantee is put
    where it can be checked: the empty-state SENTENCE is derived on the wire
    (`server._kernel_render`) from the document that was read, and the page
    prints it. A sentence hardcoded in the page cannot be tested and cannot know
    which contract it is looking at — which is how "the kernel-R&D loop has not
    exported any results" came to be printed over a complete contract.
    """

    def test_a_fully_reported_v2_contract_is_not_described_as_nothing_exported(self):
        self.export(contract(T0))
        render = server.kernel_payload()["_render"]
        self.assertEqual(render["mode"], server.RENDER_MODE_V2)
        self.assertNotIn("has not exported any results", render["note"])
        self.assertIn("7 of 7 sections reported", render["note"])

    def test_the_page_no_longer_carries_the_sentence_as_a_literal(self):
        """BITE: this is the exact string the deployed page printed over a
        complete v2 contract. It may not be a literal in the page again — the
        v1-only version of it now comes from the wire, where the reader knows
        whether it is true."""
        html = KERNEL_HTML.read_text(encoding="utf-8")
        self.assertNotIn("the kernel-R&D loop has not exported any results", html)
        self.assertIn("_render", html)
        # ...and the page must still show the absence sentence and the section
        # table, or the wire fields would be renderable-but-unrendered.
        self.assertIn("absence_means", html)
        self.assertIn("d.sections", html)

    def test_an_absent_producer_gets_an_absence_sentence_not_an_empty_run_log(self):
        env_before = server.kernel_payload()  # nothing exported yet
        self.assertEqual(env_before["_render"]["mode"], server.RENDER_MODE_ABSENT)
        self.assertIn("NO PRODUCER REPORTED", env_before["_render"]["note"])
        self.assertIn("NOBODY IS REPORTING", env_before["_render"]["note"])
        self.assertIsNone(env_before["_contract_version"])
        # Absence tolerance, unchanged: the page's v1 accessors are null, and the
        # deployed page reads every one of them through `x || []`.
        for key in ("runs", "pareto", "best_per_model", "totals"):
            self.assertIsNone(env_before[key], key)

    def test_a_corrupt_export_is_not_described_as_a_loop_that_never_ran(self):
        self.wire.kernel.write_text("{truncated", encoding="utf-8")
        payload = server.kernel_payload()
        self.assertEqual(payload["_render"]["mode"], server.RENDER_MODE_UNREADABLE)
        self.assertIn("PRESENT AND UNREADABLE", payload["_render"]["note"])
        self.assertIsNone(payload["_contract_version"])

    def test_a_legacy_v1_run_log_keeps_its_own_sentence(self):
        """COMPLIANT-PATH CONTROL: the v1 claim is preserved where it is true."""
        self.wire.kernel.write_text(json.dumps(
            {"runs": [], "totals": {}, "generated_at": _iso(T0)}), encoding="utf-8")
        payload = server.kernel_payload()
        self.assertEqual(payload["_render"]["mode"], server.RENDER_MODE_V1)
        self.assertIn("empty run log", payload["_render"]["note"])
        self.assertEqual(payload["_contract_version"], "v1")


# =============================================================================
# 4 — THE RESTART CHAOS TEST
# =============================================================================
class RestartChaosTest(_SeamCase):
    """Producer alive → producer DIES → time passes → the board NAMES it.

    Every step reads through the real hub, and every step asserts BOTH halves of
    the contract this surface exists for:

      * absence tolerance — no step raises, and the page's accessors stay
        readable even after the export is deleted; and
      * absence VISIBILITY — no step after the death reads clean.

    Nothing is killed. `assertNotClean` is the assertion that would have caught
    the original outage, and `assertClean` at the live steps is the control that
    stops this suite from passing on a surface that simply alarms forever.
    """

    def test_the_incident_end_to_end(self):
        # -- 1. alive and reporting -------------------------------------------
        self.export(contract(T0, seq=7))
        now = _epoch(T0) + 60
        env = self.kernel_env(now)
        self.assertClean(env, "the loop exported a minute ago")
        fold = self.fold(now, kernel=env)
        self.assertEqual(fold["status"], panels.STATUS_OK)
        self.assertIsNone(fold["status_set_by"])

        # -- 2. the producer dies. Nothing is written from here on -------------
        # (simulated by NOT exporting again — which is exactly what a dead
        # producer does, and requires killing nothing.)

        # -- 3. inside the silence budget, silence is not death ----------------
        now = _epoch(T0) + 6 * HOUR
        env = self.kernel_env(now)
        self.assertClean(env, "6h of silence inside a 3-day budget")
        self.assertEqual(self.fold(now, kernel=env)["status"], panels.STATUS_OK)

        # -- 4. past the budget: the board NAMES the dead producer -------------
        budget = panels.PANELS["kernel"].silent_after_s
        now = _epoch(T0) + budget + HOUR
        env = self.kernel_env(now)
        self.assertNotClean(env, "the producer has been dead past its budget")
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_STOPPED)
        self.assertEqual(env["reporting"], panels.REPORTING_SILENT)
        # The last report is still SHOWN — the card is not blanked — but it is no
        # longer presented as current. Rendering the absence, not hiding the data.
        self.assertEqual(env["content"], panels.CONTENT_POPULATED)
        self.assertIn("absence_means", env)
        fold = self.fold(now, kernel=env)
        self.assertEqual(fold["status"], panels.STATUS_DEGRADED)
        self.assertEqual(fold["status_set_by"]["panel"], "kernel")
        self.assertIn("produced_at", fold["status_set_by"]["why"])
        self.assertIn("dashboard_contract", fold["status_set_by"]["why"])
        self.assertEqual(fold["worst"]["panel"], "kernel")

        # -- 5. THE HUB RESTARTS. The verdict must not go green ----------------
        # The watchdog's in-process memory is gone; the age arm is stateless and
        # is the one that matters here.
        self.wire.reload_hub()
        self.assertEqual(server._watchdog_state, {})
        now = _epoch(T0) + budget + 2 * HOUR
        env = self.kernel_env(now)
        self.assertNotClean(env, "after a hub restart over a dead producer")
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_STOPPED)
        self.assertEqual(self.fold(now, kernel=env)["status"], panels.STATUS_DEGRADED)

        # -- 6. the EXPORTER is alive over the dead loop -----------------------
        # A re-export moves `exported_at` and the live host readings and nothing
        # else. A heartbeat-shaped rewrite may not resurrect the panel.
        rewritten = self.export(contract(T0, seq=7, exported_when=T0 + dt.timedelta(
            seconds=budget + 3 * HOUR)))
        self.assertEqual(rewritten["produced_at"], _iso(T0))
        now = _epoch(T0) + budget + 3 * HOUR
        env = self.kernel_env(now)
        self.assertNotClean(env, "a no-op re-export over a dead loop")
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_STOPPED)

        # -- 7. the export is swept away entirely ------------------------------
        # Absence tolerance under chaos: the hub must render, not 500.
        self.wire.kernel.unlink()
        payload = server.kernel_payload()          # must not raise
        env = payload["_freshness"]
        self.assertFalse(env["artifact_present"])
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_NEVER)
        self.assertNotClean(env, "the export was deleted while the producer was dead")
        fold = self.fold(now, kernel=env)
        self.assertEqual(fold["status"], panels.STATUS_ABSENT)
        self.assertEqual([a["panel"] for a in fold["absent"]], ["kernel"])
        self.assertTrue(fold["absent"][0]["anomalous"])

        # -- 8. the producer comes back ---------------------------------------
        # COMPLIANT-PATH CONTROL: the alarm CLEARS. A detector that never returns
        # to green is a detector nobody keeps looking at.
        back = T0 + dt.timedelta(seconds=budget + 4 * HOUR)
        self.export(contract(back, seq=9))
        now = _epoch(back) + 60
        env = self.kernel_env(now)
        self.assertClean(env, "the producer restarted and exported again")
        self.assertEqual(self.fold(now, kernel=env)["status"], panels.STATUS_OK)

    def test_a_producer_that_reports_without_advancing_is_caught_too(self):
        """The other death: alive enough to re-export, dead enough to make no
        progress. `produced_at` keeps moving (the champion record is rewritten),
        so the age arm never fires; the controller sequence — the loop's own
        progress identity — has not moved. Two polls a budget apart.
        """
        budget = panels.PANELS["kernel"].silent_after_s
        first = _epoch(T0)
        self.export(contract(T0, seq=7))
        env = self.kernel_env(first)
        self.assertClean(env, "first poll of a stalled-but-writing producer")

        later = T0 + dt.timedelta(seconds=budget + HOUR)
        # Records advance; the controller does not.
        self.export(contract(later, seq=7))
        env = self.kernel_env(_epoch(later))
        self.assertEqual(env["staleness_class"], panels.CLASS_FRESH,
                         "the timestamps really are fresh — that is the point")
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_NOT_ADVANCING)
        self.assertNotClean(env, "fresh timestamps, no progress")
        self.assertEqual(self.fold(_epoch(later), kernel=env)["status"],
                         panels.STATUS_DEGRADED)

    def test_a_producer_that_is_advancing_is_left_alone(self):
        """COMPLIANT-PATH CONTROL for the watermark arm: the same two polls a
        budget apart, with the controller sequence moving, stay green."""
        budget = panels.PANELS["kernel"].silent_after_s
        self.export(contract(T0, seq=7))
        self.kernel_env(_epoch(T0))
        later = T0 + dt.timedelta(seconds=budget + HOUR)
        self.export(contract(later, seq=8))
        env = self.kernel_env(_epoch(later))
        self.assertClean(env, "a producer that kept advancing")

    def test_the_literal_trial_1302_shape_on_the_panel_it_belongs_to(self):
        """~23 HOURS of silence with every dashboard green — the actual incident.

        The autopilot outcome panel is the one the outage happened on, and its
        silence budget is 6 h, so hour 23 is a NAMED verdict rather than a green
        card. The kernel panel's budget is 3 days because its producer exports
        per campaign round, not per trial — different cadence, different budget,
        same rule.
        """
        self.export_outcome(outcome_contract(T0, trial=1302))
        now = _epoch(T0) + HOUR
        env = self.outcome_env(now)
        self.assertClean(env, "the autopilot exported an hour ago")
        self.assertEqual(self.fold(now, outcome=env)["status"], panels.STATUS_OK)

        now = _epoch(T0) + 23 * HOUR      # the outage, to the hour
        env = self.outcome_env(now)
        self.assertNotClean(env, "AutoPilot dead for 23 hours")
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_STOPPED)
        fold = self.fold(now, outcome=env)
        self.assertEqual(fold["status"], panels.STATUS_DEGRADED)
        self.assertEqual(fold["status_set_by"]["panel"], "outcome")
        # It gates DESPITE being a non-health-gating panel: a watchdog alarm is
        # the fact the fold exists to publish.
        self.assertFalse(env["gates_health"])

    def test_a_declared_pause_is_not_a_dead_autopilot(self):
        """COMPLIANT-PATH CONTROL: a Phase-0 stop-loss pause is a legitimate long
        silence — but only the loop can tell a pause from a crash, so it must be
        DECLARED. Same 23 hours, `paused: true`, reads idle."""
        self.export_outcome(outcome_contract(T0, trial=1302, paused=True))
        now = _epoch(T0) + 23 * HOUR
        env = self.outcome_env(now)
        self.assertEqual(env["watchdog"]["state"], panels.WATCHDOG_IDLE)
        self.assertEqual(self.fold(now, outcome=env)["status"], panels.STATUS_OK)

    def test_nothing_in_the_chaos_run_ever_raises(self):
        """Absence tolerance is still REQUIRED. Every payload builder the hub
        serves must answer over each chaos state — a hub that 500s because a
        producer died has replaced a lying page with no page."""
        states = (
            ("nothing exported", lambda: None),
            ("a complete contract", lambda: self.export(contract(T0))),
            ("a truncated write", lambda: self.wire.kernel.write_text(
                "{trunc", encoding="utf-8")),
            ("a JSON array", lambda: self.wire.kernel.write_text(
                "[]", encoding="utf-8")),
            ("an unknown schema", lambda: self.wire.kernel.write_text(json.dumps(
                {"schema": "epyc.autokernel.kernel_dashboard.v9"}), encoding="utf-8")),
            ("a v2 document with no sections", lambda: self.wire.kernel.write_text(
                json.dumps({"schema": S.SCHEMA_KERNEL_DASHBOARD_V2,
                            "produced_at": _iso(T0)}), encoding="utf-8")),
            ("the file deleted again", lambda: self.wire.kernel.unlink()),
        )
        for label, act in states:
            with self.subTest(state=label):
                act()
                payload = server.kernel_payload()
                self.assertIsInstance(payload, dict)
                self.assertIn("_freshness", payload)
                self.assertIn("_render", payload)
                json.dumps(payload)   # the wire must be serialisable
                env = payload["_freshness"]
                if label != "a complete contract":
                    self.assertNotClean(env, label)


# =============================================================================
# 5 — this suite's own constraints
# =============================================================================
class TheSuiteObeysItsOwnConstraintsTest(unittest.TestCase):

    def test_this_suite_manages_no_processes(self):
        """STRUCTURAL, not documented. The hub on :8100 is a live service other
        sessions are using, and this host has already been bitten by a broad
        process-pattern kill (INC-20260731). A chaos test that reaches for a
        process is the wrong test, so the ban is audited rather than promised.
        """
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        # Strip the module docstring, whose prose necessarily NAMES what is
        # banned. The needles below are SPLIT so that this list is not itself a
        # hit — the guard must not be satisfiable, or defeatable, by the text of
        # the guard (`feedback_guard_must_not_forbid_its_own_idiom`).
        body = source.split('"""', 2)[-1]
        forbidden = ("sub" + "process", "os." + "kill", "os." + "system",
                     "sig" + "nal.", "Pop" + "en", "pk" + "ill", "pg" + "rep",
                     "systemc" + "tl", "os." + "spawn", "os." + "fork")
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, body)

    def test_the_compliant_path_control_can_actually_fail(self):
        """`assertNotClean` is only worth something if a clean panel exists to be
        distinguished from. The healthy universe this suite folds over must be
        green, or every "not clean" assertion below would pass vacuously."""
        now = _epoch(T0)
        envs = {}
        for name, src in panels.PANELS.items():
            obs = (panels.Observation(artifact_present=True, timestamp=None,
                                      source="live-scan", populated=True)
                   if src.kind in panels.LIVE_KINDS else
                   panels.Observation(artifact_present=True, timestamp=now - 60,
                                      source=src.timestamp_field, populated=True))
            envs[name] = panels.envelope(src, obs, now=now)
        self.assertEqual(panels.fold(envs)["status"], panels.STATUS_OK)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
