#!/usr/bin/env python3
"""Offline training pipeline for GraphRouter GAT weights.

Steps:
1. Load EpisodicStore snapshot
2. BipartiteRoutingGraph.sync_from_episodic_store() — cluster + edge compute
3. Export node features + adjacency + edge labels
4. Train GAT: edge masking (20% held out) -> predict -> BCE loss -> SGD
5. Save graph_router_weights.npz
6. Validate: held-out edge prediction accuracy

Usage:
    python3 scripts/graph_router/train_graph_router.py [--epochs 100] [--lr 0.001]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_graph_router")

DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "orchestration/repl_memory/graph_router_weights.npz"

# Model fleet from model_registry.yaml
MODEL_FLEET = [
    {"role_id": "frontdoor", "description": "Qwen3-Coder-30B-A3B front door orchestrator, interactive chat, MoE6+spec+lookup 47 t/s", "port": 8080, "tps": 47.0, "tier": "HOT", "gb": 20.0},
    {"role_id": "coder_escalation", "description": "Qwen2.5-Coder-32B code generation escalation specialist, spec+lookup 39 t/s", "port": 8081, "tps": 39.0, "tier": "HOT", "gb": 20.0},
    {"role_id": "worker_general", "description": "Qwen2.5-7B general purpose worker, explore summarize, spec+lookup 50 t/s", "port": 8082, "tps": 50.0, "tier": "HOT", "gb": 8.0},
    {"role_id": "architect_general", "description": "Qwen3-235B-A22B general architecture specialist, system design, full+spec 6.1 t/s", "port": 8083, "tps": 6.1, "tier": "WARM", "gb": 133.0},
    {"role_id": "architect_coding", "description": "Qwen3-Coder-480B-A35B coding architecture specialist, complex code design, full+spec 9.0 t/s", "port": 8084, "tps": 9.0, "tier": "WARM", "gb": 271.0},
    {"role_id": "ingest_long_context", "description": "Qwen3-Next-80B-A3B long context ingestion specialist, SSM no spec, 6.3 t/s", "port": 8085, "tps": 6.3, "tier": "WARM", "gb": 46.0},
]


def populate_llm_roles(graph, embedder):
    """Populate LLMRole nodes from model fleet definition."""
    for model in MODEL_FLEET:
        emb = embedder.embed_text(model["description"])
        graph.add_llm_role(
            role_id=model["role_id"],
            description=model["description"],
            embedding=emb,
            port=model["port"],
            tps=model["tps"],
            tier=model["tier"],
            gb=model["gb"],
        )
    logger.info("Populated %d LLM role nodes", len(MODEL_FLEET))


def build_training_data(graph):
    """Export graph structure for GAT training.

    Returns:
        node_features, edge_index, targets, qc_indices, llm_indices
    """
    node_feats = graph.get_node_features()
    edge_idx = graph.get_edge_index()
    perf_edges = graph.get_performance_edges()

    qc_ids = node_feats["query_cluster_ids"]
    llm_ids = node_feats["llm_role_ids"]

    if not qc_ids or not llm_ids:
        logger.error("Graph has no query clusters or LLM roles — cannot train")
        return None

    qc_idx_map = {id: i for i, id in enumerate(qc_ids)}
    llm_idx_map = {id: i for i, id in enumerate(llm_ids)}

    # Build target matrix (N_qc, N_llm) with success_rate as soft labels
    targets = np.zeros((len(qc_ids), len(llm_ids)), dtype=np.float32)
    for e in perf_edges:
        qi = qc_idx_map.get(e["to_cluster"])
        li = llm_idx_map.get(e["from_role"])
        if qi is not None and li is not None:
            targets[qi, li] = e["success_rate"]

    return node_feats, edge_idx, targets, np.arange(len(qc_ids)), np.arange(len(llm_ids))


def train(
    epochs: int = 100,
    lr: float = 0.001,
    val_split: float = 0.2,
    patience: int = 20,
    output_path: Path = DEFAULT_WEIGHTS_PATH,
    min_memories: int = 500,
):
    """Run offline GAT training pipeline."""
    from orchestration.repl_memory.embedder import TaskEmbedder
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.lightweight_gat import LightweightGAT
    from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

    logger.info("=== GraphRouter Training Pipeline ===")
    t0 = time.time()

    # 1. Load episodic store
    store = EpisodicStore()
    mem_count = store.count()
    logger.info("Episodic store: %d memories", mem_count)

    if mem_count < min_memories:
        logger.warning(
            "Insufficient memories (%d < %d). Skipping training.",
            mem_count, min_memories,
        )
        return False

    # 2. Initialize components
    embedder = TaskEmbedder()
    graph = BipartiteRoutingGraph()

    # 3. Populate LLM roles
    populate_llm_roles(graph, embedder)

    # 4. Sync graph from episodic store
    sync_stats = graph.sync_from_episodic_store(store, embedder)
    logger.info("Graph sync: %s", sync_stats)

    # 5. Build training data
    result = build_training_data(graph)
    if result is None:
        return False
    node_feats, edge_idx, targets, qc_indices, llm_indices = result
    logger.info(
        "Training data: %d query clusters, %d LLM roles, target shape %s",
        len(qc_indices), len(llm_indices), targets.shape,
    )

    # 6. Create validation mask (stratified by role)
    rng = np.random.default_rng(42)
    val_mask = np.zeros_like(targets, dtype=bool)
    for li in range(targets.shape[1]):
        nonzero = np.where(targets[:, li] > 0)[0]
        if len(nonzero) >= 2:
            n_val = max(1, int(len(nonzero) * val_split))
            val_idx = rng.choice(nonzero, n_val, replace=False)
            val_mask[val_idx, li] = True

    train_mask = ~val_mask
    logger.info(
        "Train edges: %d, Val edges: %d",
        int(train_mask.sum()), int(val_mask.sum()),
    )

    # 7. Train GAT
    gat = LightweightGAT()
    logger.info("GAT parameters: %d", gat.param_count)

    gat_edge_index = {
        "belongs_to": edge_idx["belongs_to"],
        "performance_on": edge_idx["performance_on"],
    }

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(epochs):
        # Forward pass
        out = gat.forward(node_feats, gat_edge_index, training=True)
        preds = gat.predict_edges(out["query_cluster"], out["llm_role"])

        # Losses
        train_loss = gat.compute_loss(preds, targets, mask=train_mask.astype(np.float32))
        val_loss = gat.compute_loss(preds, targets, mask=val_mask.astype(np.float32))

        # Cosine LR decay
        current_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        # Compute gradients and update
        gradients = gat.get_gradients(
            node_feats, gat_edge_index, targets,
            qc_indices, llm_indices,
        )
        gat.update_weights(gradients, current_lr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.copy() for k, v in gat._weights.items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.6f  patience=%d/%d",
                epoch, epochs, train_loss, val_loss, current_lr, no_improve, patience,
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # 8. Restore best weights and save
    if best_weights:
        gat._weights = best_weights

    gat.save(output_path)

    # 9. Final validation
    out = gat.forward(node_feats, gat_edge_index)
    preds = gat.predict_edges(out["query_cluster"], out["llm_role"])
    final_loss = gat.compute_loss(preds, targets)

    # Edge prediction accuracy (threshold 0.5)
    binary_preds = (preds > 0.5).astype(np.float32)
    binary_targets = (targets > 0.5).astype(np.float32)
    accuracy = float((binary_preds == binary_targets).mean())

    elapsed = time.time() - t0
    logger.info(
        "=== Training complete in %.1fs ===\n"
        "  Final loss: %.4f\n"
        "  Edge accuracy: %.2f%%\n"
        "  Weights: %s\n"
        "  Graph: %s",
        elapsed, final_loss, accuracy * 100, output_path, graph.get_stats(),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Train GraphRouter GAT weights")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stop patience")
    parser.add_argument("--min-memories", type=int, default=500, help="Min episodic memories required")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_WEIGHTS_PATH),
        help="Output path for weights",
    )
    args = parser.parse_args()

    success = train(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_path=Path(args.output),
        min_memories=args.min_memories,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
