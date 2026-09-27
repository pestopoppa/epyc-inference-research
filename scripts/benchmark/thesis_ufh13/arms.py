"""UFH-13 arms, as specified by TE-1 (epyc-orchestrator 9959e8db + the opt-in follow-up).

All three arms go through the SAME orchestrator ``/v1/chat/completions`` endpoint in client
tool mode. They differ only in these body keys:

| arm | what serves the answer | keys |
|---|---|---|
| A0 | the consultant alone (architect_general; Flash-Next after the role swap) | ``x_force_role=architect_general``, ``x_escalation=off`` |
| A1 | frontdoor alone | ``x_escalation=off`` |
| A2 | frontdoor, escalating to the consultant | ``x_escalation=architect_general`` |

A0 sends ``x_escalation=off`` only to get the escalation receipt, whose
``request_device_seconds`` is A0's consultant time (every call of an A0 request is served by the
pinned consultant). A role-overridden request never escalates either way.

A2 NAMES the consultant instead of ``auto``: frontdoor's native chain goes through
``coder_escalation``, which stays on the 27B after the swap, so ``auto`` would measure the wrong
model.

Every arm requires the orchestrator flag ``v1_escalation`` ON: with it off the receipt says
``disabled_reason=flag_off`` and the runner refuses the record set (see ``records.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONSULTANT_ROLE = "architect_general"
FRONTDOOR_ROLE = "frontdoor"


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    body_keys: dict[str, Any] = field(default_factory=dict)
    # Whose server time is "consultant device-seconds" for this arm.
    consultant_is_whole_request: bool = False
    expect_escalation_enabled: bool = False


ARMS: dict[str, Arm] = {
    "A0": Arm(
        name="A0",
        description="consultant alone (architect_general pinned), no escalation",
        body_keys={"x_force_role": CONSULTANT_ROLE, "x_escalation": "off"},
        consultant_is_whole_request=True,
    ),
    "A1": Arm(
        name="A1",
        description="frontdoor alone, escalation off",
        body_keys={"x_escalation": "off"},
    ),
    "A2": Arm(
        name="A2",
        description="frontdoor with escalation pinned to the consultant (architect_general)",
        body_keys={"x_escalation": CONSULTANT_ROLE},
        expect_escalation_enabled=True,
    ),
}

# Pre-registered generation settings (handoff "Generation"). enable_thinking=false is NOT a /v1
# body key: the orchestrator applies each role's registry chat_template_kwargs server-side, so the
# runner records it as registry-controlled rather than pretending to send it.
GENERATION = {"temperature": 0, "seed": 42, "max_tokens": 16384}
ENABLE_THINKING = "registry-controlled (chat_template_kwargs per role); pre-registered false"
