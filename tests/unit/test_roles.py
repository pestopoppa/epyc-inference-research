#!/usr/bin/env python3
"""Unit tests for src/roles.py."""

from src.roles import (
    Role,
    Tier,
    _ESCALATION_MAP,
    chain_name_to_role,
    get_escalation_chain,
    get_tier,
    role_to_chain_name,
)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


# ── Derivation helpers ───────────────────────────────────────────────────────
#
# The tables in src/roles.py (_TIER_MAP, _ESCALATION_MAP) are hand restatements
# of registry data / topology. Tests below DERIVE their expectations from those
# same sources instead of re-pasting literals, so a future role move fails on
# the real invariant rather than on a stale name (the 2026-08-01 W1 cutover
# moved the 122B from architect_general to architect_critic and broke every
# literal here at once).


def _registry_role(role_name: str):
    """RoleConfig for ``role_name`` from the compiled registry, or None."""
    from src.registry_loader import RegistryLoader

    try:
        return RegistryLoader().get_role(role_name)
    except Exception:
        return None


def _registry_tier(role_name: str) -> str | None:
    cfg = _registry_role(role_name)
    return None if cfg is None else cfg.tier


def _registry_model_path(role: Role) -> str | None:
    cfg = _registry_role(role.value)
    return None if cfg is None else cfg.model.path


def _terminal_role(start: Role) -> Role:
    """Walk _ESCALATION_MAP from ``start`` to the rung that does not escalate."""
    current = start
    seen = {current}
    while True:
        nxt = _ESCALATION_MAP.get(current)
        if nxt is None or nxt in seen:
            return current
        seen.add(nxt)
        current = nxt


def _assert_walks_escalation_map(chain: list[Role]) -> None:
    """Every consecutive pair is a declared edge and the last rung is terminal.

    Derived directly from ``_ESCALATION_MAP`` — the table the chain builder
    consumes — so no test below has to name the terminal role.
    """
    assert chain, "escalation chain must not be empty"
    for src_role, dst_role in zip(chain, chain[1:]):
        assert _ESCALATION_MAP.get(src_role) is dst_role, (
            f"{src_role.value} -> {dst_role.value} is not a declared escalation edge"
        )
    assert _ESCALATION_MAP.get(chain[-1]) is None, (
        f"chain does not terminate: {chain[-1].value} still escalates"
    )


class TestTierEnum:
    """Test Tier enum."""

    def test_tier_values(self):
        """Test Tier enum values."""
        assert Tier.A == "A"
        assert Tier.B == "B"
        assert Tier.C == "C"
        assert Tier.D == "D"


class TestRoleEnum:
    """Test Role enum."""

    def test_role_values(self):
        """Test Role enum has expected values."""
        assert Role.FRONTDOOR == "frontdoor"
        assert Role.CODER_ESCALATION == "coder_escalation"
        assert Role.WORKER_GENERAL == "worker_general"
        assert Role.DRAFT_CODER == "draft_coder"

    def test_role_str_conversion(self):
        """Test Role.__str__ returns value."""
        assert str(Role.CODER_ESCALATION) == "coder_escalation"
        assert str(Role.WORKER_MATH) == "worker_math"

    def test_role_is_valid(self):
        """Test Role.is_valid() validates strings."""
        assert Role.is_valid("frontdoor") is True
        assert Role.is_valid("coder_escalation") is True
        assert Role.is_valid("coder") is True
        assert Role.is_valid("worker_fast") is True
        assert Role.is_valid(_RETIRED_ARCHITECT_ROLE) is True
        assert Role.is_valid("invalid_role") is False
        assert Role.is_valid("") is False

    def test_role_from_string_valid(self):
        """Test Role.from_string() with valid roles."""
        role = Role.from_string("coder_escalation")
        assert role == Role.CODER_ESCALATION

        role2 = Role.from_string("worker_math")
        assert role2 == Role.WORKER_MATH

        assert Role.from_string("coder") == Role.CODER_ESCALATION
        assert Role.from_string("coder_agent") == Role.CODER_ESCALATION
        assert Role.from_string("researcher") == Role.WORKER_GENERAL
        assert Role.from_string("reviewer") == Role.ARCHITECT_CRITIC  # ARCHSWAP-20260927: reviewer default = the 27B
        assert Role.from_string("worker_explore") == Role.WORKER_GENERAL
        assert Role.from_string("worker_fast") == Role.WORKER_GENERAL

    def test_retired_architect_role_string_normalizes_to_live_architect(self):
        """Old serialized coding-architect role strings resolve to the live architect."""
        assert Role.from_string(_RETIRED_ARCHITECT_ROLE) == Role.ARCHITECT_GENERAL
        assert Role(_RETIRED_ARCHITECT_ROLE) == Role.ARCHITECT_GENERAL
        assert Role.ARCHITECT_CODING == Role.ARCHITECT_GENERAL
        assert str(Role.ARCHITECT_CODING) == "architect_general"
        assert _RETIRED_ARCHITECT_ROLE not in {role.value for role in Role}

    def test_role_from_string_invalid(self):
        """Test Role.from_string() with invalid role returns default."""
        role = Role.from_string("invalid_role")
        assert role is None

        role2 = Role.from_string("invalid", default=Role.WORKER_GENERAL)
        assert role2 == Role.WORKER_GENERAL

    def test_role_tier_property(self):
        """Test Role.tier property."""
        assert Role.FRONTDOOR.tier == Tier.A
        assert Role.CODER_ESCALATION.tier == Tier.B
        assert Role.WORKER_GENERAL.tier == Tier.C
        assert Role.DRAFT_CODER.tier == Tier.D

    def test_role_is_specialist(self):
        """is_specialist must agree with the registry's DECLARED tier for every role.

        ``_TIER_MAP`` in roles.py is a hand restatement of ``roles.<name>.tier`` in
        the compiled registry, and it has drifted before (architect_critic was
        missing entirely, so the whole-machine 122B read as Tier.C and the approval
        gate never fired). Deriving from the registry means the next drift fails
        here instead of passing silently, and it removes the pre-W1 literal that
        claimed architect_general is tier B (the registry declares A since the 27B
        took over that role and the 122B moved to architect_critic).
        """
        declared = {
            role: _registry_tier(role.value)
            for role in Role
            if _registry_tier(role.value) is not None
        }

        # Coverage floor: these roles must stay registry-declared, so the loop
        # below cannot quietly degrade into asserting nothing.
        for role in (
            Role.CODER_ESCALATION,
            Role.ARCHITECT_GENERAL,
            Role.ARCHITECT_CRITIC,
            Role.WORKER_GENERAL,
            Role.FRONTDOOR,
        ):
            assert role in declared, f"{role.value} is not declared in the registry"

        for role, tier in declared.items():
            assert role.is_specialist is (tier == Tier.B.value), (
                f"{role.value}: is_specialist={role.is_specialist} but the registry "
                f"declares tier {tier}"
            )

        # Both polarities are genuinely exercised.
        assert any(role.is_specialist for role in declared)
        assert any(not role.is_specialist for role in declared)

    def test_role_is_worker(self):
        """Test Role.is_worker property."""
        assert Role.WORKER_GENERAL.is_worker is True
        assert Role.WORKER_MATH.is_worker is True
        assert Role.CODER_ESCALATION.is_worker is False

    def test_role_is_draft(self):
        """Test Role.is_draft property."""
        assert Role.DRAFT_CODER.is_draft is True
        assert Role.DRAFT_GENERAL.is_draft is True
        assert Role.WORKER_GENERAL.is_draft is False


class TestEscalationChain:
    """Test escalation chain logic."""

    def test_worker_escalates_to_coder(self):
        """Test worker roles escalate to coder."""
        assert Role.WORKER_GENERAL.escalates_to() == Role.CODER_ESCALATION
        assert Role.WORKER_MATH.escalates_to() == Role.CODER_ESCALATION
        assert Role.WORKER_SUMMARIZE.escalates_to() == Role.CODER_ESCALATION

    def test_frontdoor_escalates_to_coder(self):
        """Test frontdoor escalates to coder."""
        assert Role.FRONTDOOR.escalates_to() == Role.CODER_ESCALATION

    def test_coder_escalates_to_architect(self):
        """Coder escalation reaches an architect rung, and it is a REAL hop.

        The literal ARCHITECT_GENERAL was stale after the 2026-08-01 W1 cutover:
        the registry declares ``roles.coder_escalation.alias_of: architect_general``
        and ``server_mode.architect_general.shared_with: [coder_escalation]``, i.e.
        both names are the SAME :8083 process serving the SAME GGUF. That edge
        could not change model or hardware — it only burned a rung of the ladder.
        The invariant the literal stood for is asserted directly here: the hop
        lands on an architect and serves a different model, derived from the
        registry rather than restated.
        """
        target = Role.CODER_ESCALATION.escalates_to()
        assert target is not None, "coder escalation must have a target"
        assert target.value.startswith("architect_"), target.value
        assert target is _terminal_role(Role.CODER_ESCALATION)

        coder_model = _registry_model_path(Role.CODER_ESCALATION)
        target_model = _registry_model_path(target)
        assert coder_model and target_model
        assert target_model != coder_model, (
            f"coder_escalation escalates to {target.value}, which serves the same "
            f"GGUF ({coder_model}) — a null hop, not an escalation"
        )

    def test_ingest_escalates_to_architect(self):
        """Test ingest escalates to architect."""
        assert Role.INGEST_LONG_CONTEXT.escalates_to() == Role.ARCHITECT_GENERAL

    def test_architect_no_escalation(self):
        """The ladder terminates on exactly one architect rung, from every entry point.

        ARCHITECT_GENERAL stopped being that rung on 2026-08-01 when the 122B moved
        to ARCHITECT_CRITIC, so the terminal role is derived rather than named. The
        second original line was really protecting the alias equivalence
        (ARCHITECT_CODING is the live architect), which is asserted explicitly.
        """
        terminal = _terminal_role(Role.CODER_ESCALATION)
        assert terminal.escalates_to() is None
        assert terminal.value.startswith("architect_"), terminal.value

        # Every entry point converges on the SAME terminal rung — no second top,
        # no cycle (the walk would otherwise not terminate).
        for start in (
            Role.WORKER_GENERAL,
            Role.WORKER_MATH,
            Role.FRONTDOOR,
            Role.CODER_ESCALATION,
            Role.ARCHITECT_GENERAL,
            Role.INGEST_LONG_CONTEXT,
        ):
            assert _terminal_role(start) is terminal, start.value

        # Retired alias behaves identically to the live architect it normalizes to.
        assert Role.ARCHITECT_CODING.escalates_to() is Role.ARCHITECT_GENERAL.escalates_to()

    def test_draft_no_escalation(self):
        """Test draft models don't escalate."""
        assert Role.DRAFT_CODER.escalates_to() is None
        assert Role.DRAFT_GENERAL.escalates_to() is None


class TestGetTier:
    """Test get_tier() function."""

    def test_get_tier_from_role_enum(self):
        """Test get_tier with Role enum."""
        assert get_tier(Role.FRONTDOOR) == Tier.A
        assert get_tier(Role.CODER_ESCALATION) == Tier.B
        assert get_tier(Role.WORKER_GENERAL) == Tier.C
        assert get_tier(Role.DRAFT_CODER) == Tier.D

    def test_get_tier_from_string(self):
        """Test get_tier with string role."""
        assert get_tier("frontdoor") == Tier.A
        assert get_tier("coder_escalation") == Tier.B
        assert get_tier("worker_math") == Tier.C
        assert get_tier("draft_general") == Tier.D

    def test_get_tier_unknown_defaults_to_c(self):
        """Test get_tier returns Tier.C for unknown roles."""
        assert get_tier("unknown_role") == Tier.C


class TestGetEscalationChain:
    """Test get_escalation_chain() function."""

    def test_worker_escalation_chain(self):
        """Test escalation chain from worker (terminal rung derived, not named)."""
        chain = get_escalation_chain(Role.WORKER_GENERAL)
        _assert_walks_escalation_map(chain)
        assert len(chain) == 3
        assert chain[0] == Role.WORKER_GENERAL
        assert chain[1] == Role.CODER_ESCALATION
        assert chain[2] == _terminal_role(Role.WORKER_GENERAL)

    def test_frontdoor_escalation_chain(self):
        """Test escalation chain from frontdoor."""
        chain = get_escalation_chain(Role.FRONTDOOR)
        _assert_walks_escalation_map(chain)
        assert len(chain) == 3
        assert chain[0] == Role.FRONTDOOR
        assert chain[1] == Role.CODER_ESCALATION
        assert chain[2] == _terminal_role(Role.FRONTDOOR)

    def test_coder_escalation_chain(self):
        """Test escalation chain from coder."""
        chain = get_escalation_chain(Role.CODER_ESCALATION)
        _assert_walks_escalation_map(chain)
        assert len(chain) == 2
        assert chain[0] == Role.CODER_ESCALATION
        assert chain[1] == _terminal_role(Role.CODER_ESCALATION)

    def test_architect_escalation_chain(self):
        """The terminal architect rung's chain is length 1; architect_general ends there.

        Pre-W1 architect_general WAS the terminal rung, so this asserted length 1 on
        it directly. Since the 122B moved to architect_critic the terminal rung is
        derived; architect_general's chain must still converge on it.
        """
        terminal = _terminal_role(Role.ARCHITECT_GENERAL)
        terminal_chain = get_escalation_chain(terminal)
        _assert_walks_escalation_map(terminal_chain)
        assert terminal_chain == [terminal]

        general_chain = get_escalation_chain(Role.ARCHITECT_GENERAL)
        _assert_walks_escalation_map(general_chain)
        assert general_chain[0] == Role.ARCHITECT_GENERAL
        assert general_chain[-1] == terminal

    def test_escalation_chain_from_string(self):
        """Test get_escalation_chain with string role."""
        chain = get_escalation_chain("worker_math")
        assert len(chain) == 3
        assert chain[0] == Role.WORKER_MATH

    def test_escalation_chain_unknown_role(self):
        """Test get_escalation_chain with unknown role returns empty."""
        chain = get_escalation_chain("unknown_role")
        assert chain == []


class TestChainNameMapping:
    """Test chain name to role mapping."""

    def test_chain_name_to_role(self):
        """Test chain_name_to_role conversion."""
        assert chain_name_to_role("worker") == Role.WORKER_GENERAL
        assert chain_name_to_role("coder") == Role.CODER_ESCALATION
        assert chain_name_to_role("architect") == Role.ARCHITECT_GENERAL
        assert chain_name_to_role("ingest") == Role.INGEST_LONG_CONTEXT
        assert chain_name_to_role("frontdoor") == Role.FRONTDOOR

    def test_chain_name_to_role_unknown(self):
        """Test chain_name_to_role returns None for unknown."""
        assert chain_name_to_role("unknown") is None

    def test_role_to_chain_name(self):
        """Test role_to_chain_name conversion."""
        assert role_to_chain_name(Role.WORKER_GENERAL) == "worker"
        assert role_to_chain_name(Role.CODER_ESCALATION) == "coder"
        assert role_to_chain_name(Role.ARCHITECT_GENERAL) == "architect"

    def test_role_to_chain_name_variants(self):
        """Test role_to_chain_name for variant roles."""
        assert role_to_chain_name(Role.WORKER_MATH) == "worker"
        assert role_to_chain_name(Role.CODER_ESCALATION) == "coder"
        assert role_to_chain_name(Role.ARCHITECT_CODING) == "architect"

    def test_role_to_chain_name_fallback(self):
        """Test role_to_chain_name returns value for unmapped roles."""
        # Thinking reasoning should map to coder
        assert role_to_chain_name(Role.THINKING_REASONING) == "coder"


class TestArchswapPlannerAndReviewerBindings:
    """ARCHSWAP-20260927: review/plan work stays on the 27B; only escalation moves."""

    def test_planner_default_is_architect_critic(self, monkeypatch):
        from src.roles import DEFAULT_PLANNER_ROLE, resolve_planner_role

        monkeypatch.delenv("ORCHESTRATOR_PLANNER_ROLE", raising=False)
        assert DEFAULT_PLANNER_ROLE is Role.ARCHITECT_CRITIC
        assert resolve_planner_role() is Role.ARCHITECT_CRITIC

    def test_planner_env_override_and_unknown_fallback(self, monkeypatch):
        from src.roles import resolve_planner_role

        monkeypatch.setenv("ORCHESTRATOR_PLANNER_ROLE", "architect_general")
        assert resolve_planner_role() is Role.ARCHITECT_GENERAL
        monkeypatch.setenv("ORCHESTRATOR_PLANNER_ROLE", "no_such_role")
        assert resolve_planner_role() is Role.ARCHITECT_CRITIC
        assert resolve_planner_role(override="worker_math") is Role.WORKER_MATH

    def test_reviewer_default_is_architect_critic(self, monkeypatch):
        from src.roles import resolve_reviewer_role

        monkeypatch.delenv("ORCHESTRATOR_REVIEWER_ROLE", raising=False)
        assert resolve_reviewer_role(config=object()) is Role.ARCHITECT_CRITIC

    def test_escalation_terminal_is_architect_general(self):
        assert Role.CODER_ESCALATION.escalates_to() is Role.ARCHITECT_GENERAL
        assert Role.ARCHITECT_CRITIC.escalates_to() is Role.ARCHITECT_GENERAL
        assert Role.ARCHITECT_GENERAL.escalates_to() is None
