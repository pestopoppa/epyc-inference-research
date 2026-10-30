"""Shared fixed T0 plans; campaign and live measurement callers use identical values."""
from __future__ import annotations

from . import correctness, recipes
from ..execution import t0_provider


def t0_tools() -> t0_provider.ToolPaths:
    return t0_provider.ToolPaths(
        bash="/bin/bash",
        verify_ggml_linkage_sh=str(
            recipes.REPO_ROOT / "scripts" / "utils" / "verify_ggml_linkage.sh"),
        cmake="/usr/bin/cmake")

def t0_evaluator_policy(backend: str) -> correctness.T0Policy:
    """Return the fixed evaluator policy for an executed campaign.

    The policy lives on the evaluator side of the boundary, but construction
    happens here because the campaign supplies the backend identity.  It is
    intentionally explicit: :class:`T0Policy` has no defaults, and a bare
    construction would turn a protocol omission into a live-run failure.
    """
    return correctness.T0Policy(
        required_backend_ops=correctness.MANDATORY_BACKEND_OPS,
        symbol_shrinkage_reject_ratio=0.6,
        diff_ceiling=correctness.DiffComplexityCeiling(
            backend=backend,
            max_changed_lines=400,
            max_files_touched=10,
            shared_core_forces_review=True),
        determinism_min_runs=2,
        coherence_tolerance_floor=0.98,
        policy_ref="ak-policy/v1")

def t0_generation_plan(model: str) -> t0_provider.GenerationPlan:
    """The T0 graph/output probe, bound to this campaign's exact model.

    The direct completion executable needs the model explicitly to
    construct a graph and generate output. Keep it in the typed plan,
    rather than relying on an ambient default or a build-local model path.
    ``llama-cli`` is intentionally not used: its current implementation
    starts a loopback HTTP server, while T0's candidate sandbox is
    network-denied.
    """
    return t0_provider.GenerationPlan(
        prompt="The capital of France is", prompt_ref="ak-prompt-001",
        n_predict=32, seed=42, extra_argv=("-m", model))
