"""Tests for GraphRouter integration with HybridRouter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


@dataclass
class MockMemoryEntry:
    """Minimal mock MemoryEntry for retriever tests."""

    action: str = "frontdoor:direct"
    q_value: float = 0.8
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)
    update_count: int = 2
    similarity_score: float = 0.9
    task_description: str = "test"

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.default_rng(42).normal(size=1024).astype(np.float64)


class MockGraphRouterPredictor:
    """Mock GraphRouterPredictor for testing HybridRouter blending."""

    def __init__(self, scores=None, ready=True):
        self._scores = scores or {
            "frontdoor": 0.4,
            "coder_escalation": 0.3,
            "architect_general": 0.2,
            "worker_general": 0.1,
        }
        self._ready = ready

    @property
    def is_ready(self):
        return self._ready

    def predict(self, query_embedding, task_type):
        return self._scores.copy()


class TestHybridRouterGraphBlending:
    """Test GraphRouter score blending in HybridRouter."""

    @pytest.fixture
    def retriever(self):
        """Mock TwoPhaseRetriever."""
        from orchestration.repl_memory.retriever import RetrievalResult

        mock = MagicMock()
        mock.config = MagicMock()
        mock.config.cost_lambda = 0.1
        mock.config.risk_control_enabled = False
        mock.config.risk_gate_kill_switch = False
        mock.config.risk_budget_id = "test"
        mock.config.prior_strength = 0.5
        mock.config.confidence_threshold = 0.3
        mock.config.calibrated_confidence_threshold = None
        mock.config.conformal_margin = 0.0
        mock.config.risk_gate_min_samples = 3
        mock.config.risk_abstain_target_role = "architect_general"
        mock.config.risk_gate_rollout_ratio = 0.0

        # Create mock results
        results = [
            RetrievalResult(
                memory=MockMemoryEntry(action="frontdoor:direct", q_value=0.9,
                                       context={"role": "frontdoor"}),
                similarity=0.95, q_value=0.9, combined_score=0.92,
                selection_score=0.85, posterior_score=0.85,
                q_confidence=0.8,
            ),
            RetrievalResult(
                memory=MockMemoryEntry(action="coder_escalation:direct", q_value=0.7,
                                       context={"role": "coder_escalation"}),
                similarity=0.8, q_value=0.7, combined_score=0.75,
                selection_score=0.65, posterior_score=0.65,
                q_confidence=0.8,
            ),
        ]
        mock.retrieve_for_routing.return_value = results
        mock.should_use_learned.return_value = True
        mock.get_best_action.return_value = ("frontdoor:direct", 0.8)
        mock.evaluate_risk_gate.return_value = {"enforced": False, "passed": True, "action": "not_enforced"}
        mock.get_effective_confidence_threshold.return_value = 0.3
        mock.update_last_role.return_value = None
        mock.embedder = MagicMock()
        mock.embedder.embed_task_ir.return_value = np.random.default_rng(42).normal(size=1024)
        mock._extract_role_from_memory = lambda m: (m.context or {}).get("role", m.action.split(":")[0])
        mock.store = MagicMock()
        mock.store.count.return_value = 1000

        return mock

    @pytest.fixture
    def rule_router(self):
        mock = MagicMock()
        mock.route.return_value = ["frontdoor"]
        return mock

    def test_no_graph_router(self, retriever, rule_router):
        """Without graph_router, HybridRouter works normally."""
        from orchestration.repl_memory.retriever import HybridRouter

        router = HybridRouter(retriever=retriever, rule_based_router=rule_router)
        routing, strategy = router.route({"task_type": "code"})
        assert strategy == "learned"
        assert "frontdoor" in routing

    def test_control_plane_memory_is_filtered_before_learned_selection(
        self, retriever, rule_router
    ):
        """A high-Q plan-review episode can never become a serving route."""
        from orchestration.repl_memory.retriever import HybridRouter, RetrievalResult

        poisoned = RetrievalResult(
            memory=MockMemoryEntry(action="plan_review:drop", q_value=1.0),
            similarity=0.99,
            q_value=1.0,
            combined_score=1.0,
            selection_score=1.0,
            posterior_score=1.0,
            q_confidence=1.0,
        )
        valid = RetrievalResult(
            memory=MockMemoryEntry(action="frontdoor:direct", q_value=0.8),
            similarity=0.8,
            q_value=0.8,
            combined_score=0.8,
            selection_score=0.8,
            posterior_score=0.8,
            q_confidence=0.8,
        )
        retriever.retrieve_for_routing.return_value = [poisoned, valid]
        retriever.should_use_learned.side_effect = lambda results: bool(results)
        retriever.get_best_action.side_effect = (
            lambda results: (results[0].memory.action, results[0].q_confidence)
        )

        router = HybridRouter(retriever=retriever, rule_based_router=rule_router)
        routing, strategy = router.route({"task_type": "chat"})

        assert (routing, strategy) == (["frontdoor"], "learned")
        assert router.last_decision_meta["action_topk"] == ["frontdoor:direct"]

    def test_only_control_plane_memories_fall_back_to_rules(
        self, retriever, rule_router
    ):
        """An all-invalid learned neighborhood fails closed to rule routing."""
        from orchestration.repl_memory.retriever import HybridRouter, RetrievalResult

        retriever.retrieve_for_routing.return_value = [
            RetrievalResult(
                memory=MockMemoryEntry(action="plan_review:add", q_value=1.0),
                similarity=0.99,
                q_value=1.0,
                combined_score=1.0,
                selection_score=1.0,
                posterior_score=1.0,
                q_confidence=1.0,
            )
        ]
        retriever.should_use_learned.side_effect = lambda results: bool(results)

        router = HybridRouter(retriever=retriever, rule_based_router=rule_router)
        routing, strategy = router.route({"task_type": "chat"})

        assert (routing, strategy) == (["frontdoor"], "rules")
        retriever.get_best_action.assert_not_called()

    def test_route_parser_rejects_non_serving_and_canonicalizes_alias(
        self, retriever, rule_router
    ):
        from orchestration.repl_memory.retriever import HybridRouter

        router = HybridRouter(retriever=retriever, rule_based_router=rule_router)
        assert router._parse_routing_action("plan_review:drop") == []
        assert router._parse_routing_action("reviewer:direct") == ["architect_critic"]  # ARCHSWAP-20260927 reviewer default

    def test_with_graph_router(self, retriever, rule_router):
        """With graph_router, blending happens before routing decision."""
        from orchestration.repl_memory.retriever import HybridRouter

        gr = MockGraphRouterPredictor()
        router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=gr,
        )
        routing, strategy = router.route({"task_type": "code"})
        assert strategy == "learned"

    def test_graph_router_not_ready(self, retriever, rule_router):
        """GraphRouter not ready should be skipped gracefully."""
        from orchestration.repl_memory.retriever import HybridRouter

        gr = MockGraphRouterPredictor(ready=False)
        router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=gr,
        )
        routing, strategy = router.route({"task_type": "code"})
        assert strategy == "learned"

    def test_decision_meta_includes_graph_router(self, retriever, rule_router):
        """Decision metadata should include graph_router fields."""
        from orchestration.repl_memory.retriever import HybridRouter

        gr = MockGraphRouterPredictor()
        router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=gr,
        )
        router.route({"task_type": "code"})
        meta = router.last_decision_meta
        assert "graph_router_ready" in meta
        assert meta["graph_router_ready"] is True
        assert "graph_router_weight" in meta

    def test_adaptive_weight_low_memory(self, retriever, rule_router):
        """Low memory count should give low graph weight."""
        from orchestration.repl_memory.retriever import HybridRouter

        retriever.store.count.return_value = 100
        gr = MockGraphRouterPredictor()
        router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=gr,
        )
        w = router._get_adaptive_graph_weight()
        assert w == pytest.approx(0.1, abs=0.01)

    def test_adaptive_weight_high_memory(self, retriever, rule_router):
        """High memory count should give max graph weight."""
        from orchestration.repl_memory.retriever import HybridRouter

        retriever.store.count.return_value = 5000
        gr = MockGraphRouterPredictor()
        router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=gr,
            graph_router_weight=0.3,
        )
        w = router._get_adaptive_graph_weight()
        assert w == pytest.approx(0.3, abs=0.01)

    def test_adaptive_weight_mid_memory(self, retriever, rule_router):
        """Mid memory count should give interpolated weight."""
        from orchestration.repl_memory.retriever import HybridRouter

        retriever.store.count.return_value = 1250  # Midpoint of 500-2000
        gr = MockGraphRouterPredictor()
        router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=gr,
            graph_router_weight=0.3,
        )
        w = router._get_adaptive_graph_weight()
        assert 0.1 < w < 0.3
