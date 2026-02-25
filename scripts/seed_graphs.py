#!/usr/bin/env python3
from __future__ import annotations

"""
Seed the FailureGraph and HypothesisGraph with documented patterns.

This script:
1. Loads failure patterns and hypotheses from graph_seeds.yaml
2. Creates FailureMode, Symptom, and Mitigation nodes in FailureGraph
3. Creates Hypothesis nodes with calibrated confidence in HypothesisGraph
4. Links Q-value outlier memories from EpisodicStore to the graphs

Usage:
    uv run python scripts/seed_graphs.py [--dry-run] [--verbose]
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path

import yaml

# Add workspace to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.episodic_store import EpisodicStore, extract_symptoms
from orchestration.repl_memory.failure_graph import FailureGraph
from orchestration.repl_memory.hypothesis_graph import HypothesisGraph

# Default paths (on RAID array per CLAUDE.md)
DEFAULT_SEEDS_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/graph_seeds.yaml")
DEFAULT_KUZU_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/kuzu_db")
DEFAULT_EPISODIC_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/sessions")

# Fallback for development (workspace)
WORKSPACE_SEEDS_PATH = Path(__file__).parent.parent / "orchestration/repl_memory/graph_seeds.yaml"
WORKSPACE_KUZU_PATH = Path(__file__).parent.parent / "orchestration/repl_memory/kuzu_db"
WORKSPACE_EPISODIC_PATH = Path(__file__).parent.parent / "orchestration/repl_memory/sessions"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_seeds(seeds_path: Path) -> dict:
    """Load seed data from YAML file."""
    with open(seeds_path) as f:
        return yaml.safe_load(f)


def seed_failure_graph(
    failure_graph: FailureGraph,
    failure_modes: list,
    dry_run: bool = False,
) -> dict:
    """
    Seed FailureGraph with documented failure patterns.

    Args:
        failure_graph: The FailureGraph instance
        failure_modes: List of failure mode definitions from YAML
        dry_run: If True, only log what would be done

    Returns:
        Statistics dict
    """
    stats = {"failure_modes": 0, "symptoms": 0, "mitigations": 0}

    for fm in failure_modes:
        logger.info(f"Seeding failure mode: {fm['id']}")

        if dry_run:
            stats["failure_modes"] += 1
            stats["symptoms"] += len(fm.get("symptoms", []))
            stats["mitigations"] += len(fm.get("mitigations", []))
            continue

        # Generate a seed memory ID
        memory_id = f"seed_{fm['id']}_{uuid.uuid4().hex[:8]}"

        # Record the failure with its symptoms
        failure_id = failure_graph.record_failure(
            memory_id=memory_id,
            symptoms=fm.get("symptoms", ["unknown"]),
            description=fm["description"],
            severity=fm.get("severity", 3),
        )
        stats["failure_modes"] += 1
        stats["symptoms"] += len(fm.get("symptoms", []))

        # Record mitigations
        for mitigation in fm.get("mitigations", []):
            failure_graph.record_mitigation(
                failure_id=failure_id,
                action=mitigation["action"],
                worked=mitigation.get("success_rate", 1.0) > 0.5,
            )
            stats["mitigations"] += 1
            logger.debug(f"  Mitigation: {mitigation['action']}")

    return stats


def seed_hypothesis_graph(
    hypothesis_graph: HypothesisGraph,
    hypotheses: list,
    dry_run: bool = False,
) -> dict:
    """
    Seed HypothesisGraph with documented action-task patterns.

    Args:
        hypothesis_graph: The HypothesisGraph instance
        hypotheses: List of hypothesis definitions from YAML
        dry_run: If True, only log what would be done

    Returns:
        Statistics dict
    """
    stats = {"hypotheses": 0}

    for hyp in hypotheses:
        logger.info(f"Seeding hypothesis: {hyp['claim']} (confidence={hyp['initial_confidence']})")

        if dry_run:
            stats["hypotheses"] += 1
            continue

        # Generate a seed memory ID
        memory_id = f"seed_hypothesis_{uuid.uuid4().hex[:8]}"

        # Create hypothesis with calibrated confidence
        hypothesis_graph.create_hypothesis(
            claim=hyp["claim"],
            memory_id=memory_id,
            initial_confidence=hyp["initial_confidence"],
        )
        stats["hypotheses"] += 1

    return stats


def link_memory_outliers(
    episodic_store: EpisodicStore,
    failure_graph: FailureGraph,
    hypothesis_graph: HypothesisGraph,
    low_threshold: float = 0.3,
    high_threshold: float = 0.8,
    dry_run: bool = False,
) -> dict:
    """
    Link Q-value outlier memories to the graphs.

    Args:
        episodic_store: The EpisodicStore instance
        failure_graph: The FailureGraph instance
        hypothesis_graph: The HypothesisGraph instance
        low_threshold: Q-values below this are linked to failure graph
        high_threshold: Q-values above this provide evidence for hypotheses
        dry_run: If True, only log what would be done

    Returns:
        Statistics dict
    """
    stats = {"outliers_found": 0, "failures_linked": 0, "hypotheses_updated": 0}

    try:
        outliers = episodic_store.get_q_outliers(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            limit=1000,
        )
    except Exception as e:
        logger.warning(f"Could not get Q outliers: {e}")
        return stats

    stats["outliers_found"] = len(outliers)
    logger.info(f"Found {len(outliers)} Q-value outliers to link")

    for memory in outliers:
        if dry_run:
            if memory.q_value < low_threshold:
                stats["failures_linked"] += 1
            else:
                stats["hypotheses_updated"] += 1
            continue

        # Low Q-value = failure -> link to failure graph
        if memory.q_value < low_threshold and memory.outcome:
            try:
                symptoms = extract_symptoms(memory.context, memory.outcome)
                matching = failure_graph.find_matching_failures(symptoms)
                if matching:
                    # Memory is already linked via record_failure mechanism
                    stats["failures_linked"] += 1
                    logger.debug(f"  Linked failure memory {memory.id[:8]}... (Q={memory.q_value:.2f})")
            except Exception as e:
                logger.debug(f"  Could not link failure memory: {e}")

        # High Q-value = success -> update hypothesis confidence
        if memory.q_value > high_threshold:
            try:
                task_type = memory.context.get("task_type", "general")
                hypothesis_id = hypothesis_graph.get_or_create_hypothesis(
                    action=memory.action,
                    task_type=task_type,
                    memory_id=memory.id,
                )
                # Add supporting evidence
                hypothesis_graph.add_evidence(
                    hypothesis_id=hypothesis_id,
                    outcome="success",
                    source=memory.id,
                )
                stats["hypotheses_updated"] += 1
                logger.debug(f"  Updated hypothesis for {memory.action}|{task_type}")
            except Exception as e:
                logger.debug(f"  Could not update hypothesis: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Seed MemRL graphs with documented patterns")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--seeds", type=Path, help="Path to graph_seeds.yaml")
    parser.add_argument("--kuzu-path", type=Path, help="Path to Kuzu database directory")
    parser.add_argument("--episodic-path", type=Path, help="Path to EpisodicStore directory")
    parser.add_argument("--skip-linking", action="store_true", help="Skip linking Q-outlier memories")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths (try RAID first, fallback to workspace)
    seeds_path = args.seeds or (DEFAULT_SEEDS_PATH if DEFAULT_SEEDS_PATH.exists() else WORKSPACE_SEEDS_PATH)
    kuzu_path = args.kuzu_path or (DEFAULT_KUZU_PATH if DEFAULT_KUZU_PATH.parent.exists() else WORKSPACE_KUZU_PATH)
    episodic_path = args.episodic_path or (DEFAULT_EPISODIC_PATH if DEFAULT_EPISODIC_PATH.exists() else WORKSPACE_EPISODIC_PATH)

    logger.info(f"Seeds path: {seeds_path}")
    logger.info(f"Kuzu path: {kuzu_path}")
    logger.info(f"Episodic path: {episodic_path}")

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    # Load seeds
    if not seeds_path.exists():
        logger.error(f"Seeds file not found: {seeds_path}")
        sys.exit(1)

    seeds = load_seeds(seeds_path)
    logger.info(f"Loaded {len(seeds.get('failure_modes', []))} failure modes")
    logger.info(f"Loaded {len(seeds.get('hypotheses', []))} hypotheses")

    # Initialize graphs
    if not args.dry_run:
        # Kuzu needs parent directory to exist, but not the path itself
        kuzu_path.parent.mkdir(parents=True, exist_ok=True)

        failure_graph = FailureGraph(path=kuzu_path / "failure_graph")
        hypothesis_graph = HypothesisGraph(path=kuzu_path / "hypothesis_graph")
    else:
        failure_graph = None
        hypothesis_graph = None

    # Seed failure graph
    logger.info("=== Seeding FailureGraph ===")
    failure_stats = seed_failure_graph(
        failure_graph,
        seeds.get("failure_modes", []),
        dry_run=args.dry_run,
    )
    logger.info(f"FailureGraph: {failure_stats}")

    # Seed hypothesis graph
    logger.info("=== Seeding HypothesisGraph ===")
    hypothesis_stats = seed_hypothesis_graph(
        hypothesis_graph,
        seeds.get("hypotheses", []),
        dry_run=args.dry_run,
    )
    logger.info(f"HypothesisGraph: {hypothesis_stats}")

    # Link Q-outlier memories
    if not args.skip_linking and episodic_path.exists():
        logger.info("=== Linking Q-Outlier Memories ===")
        try:
            episodic_store = EpisodicStore(db_path=episodic_path, use_faiss=True)
            link_stats = link_memory_outliers(
                episodic_store,
                failure_graph,
                hypothesis_graph,
                dry_run=args.dry_run,
            )
            logger.info(f"Memory linking: {link_stats}")
        except Exception as e:
            logger.warning(f"Could not link memories: {e}")
    else:
        logger.info("Skipping memory linking (--skip-linking or no episodic store)")

    # Print final stats
    if not args.dry_run and failure_graph and hypothesis_graph:
        logger.info("=== Final Graph Statistics ===")
        logger.info(f"FailureGraph: {failure_graph.get_stats()}")
        logger.info(f"HypothesisGraph: {hypothesis_graph.get_stats()}")

    logger.info("Seeding complete!")


if __name__ == "__main__":
    main()
