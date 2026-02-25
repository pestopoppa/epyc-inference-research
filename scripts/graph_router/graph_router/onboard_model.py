#!/usr/bin/env python3
"""New model onboarding script for GraphRouter inductive cold-start.

Adds a new LLM role to the bipartite routing graph and uses GAT to
predict routing distribution from the capability embedding alone (inductive).
Optionally runs few-shot eval queries to collect actual performance data.

This is the key value proposition of GraphRouter: hours of organic data
accumulation -> minutes of scripted onboarding.

Usage:
    python3 scripts/graph_router/onboard_model.py \\
        --role new_coder_v2 \\
        --description "Qwen4-Coder-32B, optimized for refactoring, 55 t/s" \\
        --port 8086 --tps 55.0 --memory-tier HOT --memory-gb 20 \\
        --few-shot-queries 80
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("onboard_model")

DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "orchestration/repl_memory/graph_router_weights.npz"


def onboard(
    role_id: str,
    description: str,
    port: int,
    tps: float,
    memory_tier: str,
    memory_gb: float,
    few_shot_queries: int = 0,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
):
    """Onboard a new model into the GraphRouter.

    Steps:
    1. Embed capability description via TaskEmbedder
    2. Create LLMRole node in BipartiteRoutingGraph
    3. GAT forward pass — predict routing distribution inductively
    4. Report predicted routing distribution
    """
    from orchestration.repl_memory.embedder import TaskEmbedder
    from orchestration.repl_memory.lightweight_gat import LightweightGAT
    from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor
    from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

    logger.info("=== Model Onboarding: %s ===", role_id)

    # 1. Initialize components
    embedder = TaskEmbedder()
    graph = BipartiteRoutingGraph()
    gat = LightweightGAT()

    if weights_path.exists():
        gat.load(weights_path)
        logger.info("Loaded GAT weights from %s", weights_path)
    else:
        logger.warning("No GAT weights found at %s — predictions will be random", weights_path)

    # 2. Embed capability description
    role_embedding = embedder.embed_text(description)
    logger.info("Embedded description: %d-dim vector", len(role_embedding))

    # 3. Add LLMRole node
    graph.add_llm_role(
        role_id=role_id,
        description=description,
        embedding=role_embedding,
        port=port,
        tps=tps,
        tier=memory_tier,
        gb=memory_gb,
    )
    logger.info("Created LLMRole node: %s (port=%d, tps=%.1f, tier=%s)", role_id, port, tps, memory_tier)

    # 4. Create predictor and check readiness
    predictor = GraphRouterPredictor(graph, gat, embedder)

    if not predictor.is_ready:
        logger.warning(
            "GraphRouter not ready (graph stats: %s). "
            "Need to run train_graph_router.py first.",
            graph.get_stats(),
        )
        return

    # 5. Predict routing distribution for sample task types
    task_types = ["code", "chat", "architecture", "ingest", "general"]
    sample_queries = {
        "code": "Write a Python function to sort a list of dictionaries by key",
        "chat": "Explain the difference between TCP and UDP protocols",
        "architecture": "Design a microservices architecture for an e-commerce platform",
        "ingest": "Summarize this 50-page research paper about transformer architectures",
        "general": "Help me debug this error in my application",
    }

    logger.info("\n--- Predicted Routing Distribution ---")
    for task_type, query in sample_queries.items():
        query_emb = embedder.embed_text(query)
        scores = predictor.predict(query_emb, task_type)
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            score_str = ", ".join(f"{r}={s:.3f}" for r, s in sorted_scores)
            # Highlight if new model gets significant routing share
            new_model_score = scores.get(role_id, 0.0)
            marker = " <<<" if new_model_score > 0.15 else ""
            logger.info("  %s: %s%s", task_type, score_str, marker)
        else:
            logger.info("  %s: no predictions available", task_type)

    # 6. Few-shot evaluation (optional)
    if few_shot_queries > 0:
        logger.info("\n--- Few-Shot Evaluation (%d queries) ---", few_shot_queries)
        logger.info(
            "NOTE: Few-shot evaluation requires the model server at port %d to be running.",
            port,
        )
        logger.info("Skipping actual eval (not implemented in offline script).")
        logger.info(
            "To run few-shot eval, use the orchestrator's /chat endpoint with force_role=%s",
            role_id,
        )

    logger.info("\n=== Onboarding Complete ===")
    logger.info("Graph stats: %s", graph.get_stats())
    logger.info(
        "Next steps:\n"
        "  1. Run few-shot queries via orchestrator to collect PERFORMANCE_ON edges\n"
        "  2. Re-train GAT: python3 scripts/graph_router/train_graph_router.py\n"
        "  3. Enable graph_router: ORCHESTRATOR_GRAPH_ROUTER=1",
    )


def main():
    parser = argparse.ArgumentParser(description="Onboard a new model into GraphRouter")
    parser.add_argument("--role", required=True, help="Role ID (e.g., new_coder_v2)")
    parser.add_argument("--description", required=True, help="Capability description")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--tps", type=float, required=True, help="Tokens per second")
    parser.add_argument("--memory-tier", choices=["HOT", "WARM"], default="HOT", help="Memory tier")
    parser.add_argument("--memory-gb", type=float, required=True, help="VRAM usage in GB")
    parser.add_argument("--few-shot-queries", type=int, default=0, help="Number of few-shot eval queries")
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH),
        help="GAT weights path",
    )
    args = parser.parse_args()

    onboard(
        role_id=args.role,
        description=args.description,
        port=args.port,
        tps=args.tps,
        memory_tier=args.memory_tier,
        memory_gb=args.memory_gb,
        few_shot_queries=args.few_shot_queries,
        weights_path=Path(args.weights),
    )


if __name__ == "__main__":
    main()
