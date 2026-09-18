"""Controller-owned durable persistence for A2 runtime discovery screens.

Native unit execution is deliberately not implemented here yet.  This module joins
the existing A2 consumer to the actual CampaignController/Journal durability owner;
the pending bridge supplies the already-declared ``TrustedInvoker`` implementation.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import a2_execution_state
from . import campaign_control
from . import discovery_screen
from . import experiment_plan as ep
from . import unified_planner


class DiscoveryExecutionRefused(RuntimeError):
    pass


class DiscoveryExecution:
    """One stable logical A2 execution backed by the live campaign controller."""

    def __init__(self, *, controller: campaign_control.CampaignController,
                 logical_id: str, plan: ep.ExperimentPlan,
                 pair: unified_planner.RuntimeArmPair,
                 context: discovery_screen.RuntimeFrameContext | Mapping[str, Any],
                 invoker: discovery_screen.TrustedInvoker) -> None:
        if type(controller) is not campaign_control.CampaignController:
            raise DiscoveryExecutionRefused(
                "A2 execution requires the actual CampaignController owner")
        if plan.campaign_id != controller.resolved.campaign_id:
            raise DiscoveryExecutionRefused("A2 plan and controller campaign differ")
        if not isinstance(logical_id, str) or not logical_id.strip() or len(logical_id) > 512:
            raise DiscoveryExecutionRefused("logical_id must be bounded nonempty text")
        self.controller = controller
        self.logical_id = logical_id
        self.plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        self.pair = unified_planner.RuntimeArmPair.from_dict(pair.to_dict())
        self.context = (discovery_screen.RuntimeFrameContext.from_dict(context.to_dict())
                        if isinstance(context, discovery_screen.RuntimeFrameContext)
                        else discovery_screen.RuntimeFrameContext.from_dict(context))
        self.invoker = invoker

        # Construction performs all A2 frame validation but invokes neither sink nor
        # model.  The resulting digest selects the controller's bounded replay index.
        probe = discovery_screen.A2RuntimeScreen(
            self.plan, self.pair, self.context, invoker=invoker,
            phase_sink=self._construction_sink)
        replay = controller.replay_a2_runtime_execution(
            logical_id=logical_id, plan_digest=self.plan.digest,
            frame_digest=probe.frame_digest)
        self._validate_replay(replay, probe.frame_digest)
        self.execution_id = replay["execution_id"]
        self.screen = discovery_screen.A2RuntimeScreen(
            self.plan, self.pair, self.context, invoker=invoker,
            phase_sink=self._append_phase, events=replay["events"],
            replay_verifier=replay["phase_verifier"])
        self.pending_intents = tuple(replay["pending_intents"])
        self.bank_reference = replay["bank_reference"]
        self.reused_bank = (None if replay["reused_bank"] is None else
                            discovery_screen.BaselineBank.from_dict(replay["reused_bank"]))
        self._bank_verifier = replay["bank_verifier"]

    @staticmethod
    def _construction_sink(_event: Mapping[str, Any]) -> Mapping[str, Any]:
        raise DiscoveryExecutionRefused("A2 construction unexpectedly emitted a phase event")

    def _validate_replay(self, replay: Mapping[str, Any], frame_digest: str) -> None:
        fields = {"execution_id", "logical_id", "plan_digest", "frame_digest", "events",
                  "pending_intents", "sealed_phases", "journal_entry_ids",
                  "journal_cursor", "history_digest", "phase_verifier",
                  "bank_reference", "reused_bank", "bank_verifier"}
        if not isinstance(replay, Mapping) or set(replay) != fields:
            raise DiscoveryExecutionRefused("controller A2 replay fields differ")
        if (replay["logical_id"] != self.logical_id
                or replay["plan_digest"] != self.plan.digest
                or replay["frame_digest"] != frame_digest):
            raise DiscoveryExecutionRefused("controller A2 replay identity differs")
        events = replay["events"]
        if not isinstance(events, tuple) or len(events) > 14:
            raise DiscoveryExecutionRefused("controller A2 replay event bound differs")
        if events and not isinstance(
                replay["phase_verifier"], discovery_screen.RegisteredPhaseVerifier):
            raise DiscoveryExecutionRefused("persisted A2 replay lacks live attestation")
        if not events and replay["phase_verifier"] is not None:
            raise DiscoveryExecutionRefused("empty A2 replay carries invented authority")
        reference = replay["bank_reference"]
        if reference is None:
            if replay["reused_bank"] is not None or replay["bank_verifier"] is not None:
                raise DiscoveryExecutionRefused("A2 replay invents bank reuse authority")
        elif (replay["reused_bank"] is None
              or not isinstance(replay["bank_verifier"],
                                discovery_screen.RegisteredBankVerifier)):
            raise DiscoveryExecutionRefused("A2 bank reference lacks reopened owner authority")
        try:
            for event in events:
                a2_execution_state.validate_event_membership(event, self.plan)
        except a2_execution_state.A2ExecutionStateRefused as exc:
            raise DiscoveryExecutionRefused(
                f"persisted A2 event differs from frozen plan: {exc}") from exc

    def _append_phase(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.controller.append_a2_runtime_transition(
            logical_id=self.logical_id, plan=self.plan, event=event)

    def create_bank(self) -> discovery_screen.BaselineBank:
        return self.screen.create_bank()

    def run_screen(self, bank: discovery_screen.BaselineBank | Mapping[str, Any], *,
                   bank_verifier: discovery_screen.RegisteredBankVerifier | None = None) \
            -> discovery_screen.ScreenReceipt:
        bank = (discovery_screen.BaselineBank.from_dict(bank.to_dict())
                if isinstance(bank, discovery_screen.BaselineBank)
                else discovery_screen.BaselineBank.from_dict(bank))
        if self.screen.bank_verifier().verify(bank):
            return self.screen.screen(bank, bank_verifier=bank_verifier)
        effective_verifier = bank_verifier
        if self.bank_reference is not None:
            if self.reused_bank is None or self.reused_bank.to_dict() != bank.to_dict():
                raise DiscoveryExecutionRefused(
                    "screen bank differs from its durable original-source reference")
            effective_verifier = self._bank_verifier
        elif bank_verifier is not None:
            if (not isinstance(bank_verifier, discovery_screen.RegisteredBankVerifier)
                    or not bank_verifier.verify(bank)):
                raise DiscoveryExecutionRefused(
                    "imported bank lacks its existing trusted phase authority")
            committed = self.controller.append_a2_bank_reference(
                logical_id=self.logical_id, plan=self.plan,
                frame_digest=self.screen.frame_digest, bank=bank)
            replay = self.controller.replay_a2_runtime_execution(
                logical_id=self.logical_id, plan_digest=self.plan.digest,
                frame_digest=self.screen.frame_digest)
            self._validate_replay(replay, self.screen.frame_digest)
            if (committed["bank_reference"] != replay["bank_reference"]
                    or committed["bank"] != replay["reused_bank"]):
                raise DiscoveryExecutionRefused(
                    "durable bank reference replay differs immediately after append")
            self.bank_reference = replay["bank_reference"]
            self.reused_bank = discovery_screen.BaselineBank.from_dict(
                replay["reused_bank"])
            self._bank_verifier = replay["bank_verifier"]
            effective_verifier = self._bank_verifier
        return self.screen.screen(bank, bank_verifier=effective_verifier)


__all__ = ["DiscoveryExecution", "DiscoveryExecutionRefused"]
