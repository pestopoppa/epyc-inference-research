"""boost_guard.py — CS-8.LEX0: boosting harness guard + effective-config record.

Source pins (NeMo-Speech.cpp, read in /mnt/raid0/llm/tmp/dive-2344): TdtGreedyDecoder takes no
DecoderConfig and drops `speech_contexts` silently (docs: "Parakeet TDT ignore[s] it"), so a TDT
trial would record a false null. RnntGreedyDecoder::set_request_options: alpha_eff =
boosting_tree_alpha x clamp(max request boost, 0.1, boosting_max_boost [default 5.0]); a request
with no positive boost uses factor 1.0 (unclamped); unk/off-trie token score 0; per-token
context_score 1.0; depth scaling = boosting_depth_scaling (default 2.0). The runner prints NO
positive 'boosting active' line (only a 'boosting disabled' warning), so the positive-evidence
regex is caller-supplied (LEX2 supplies it from the runner build it pins); absent evidence refuses.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

NO_RUNNER_PATH = "NO_RUNNER_PATH"
COULD_NOT_CHECK = "COULD_NOT_CHECK"
DISABLED_BY_RUNNER = "DISABLED_BY_RUNNER"
ACTIVE = "ACTIVE"
NOT_REQUESTED = "NOT_REQUESTED"

BOOSTABLE_CLASSES = ("rnnt", "nemotron", "ctc-flashlight")
# Verbatim warning in rnnt_greedy_decoder.cpp when boosting silently turns off.
DEFAULT_DISABLED_PATTERNS = (r"word boosting requested but no phrases could be tokenized",
                             r"boosting disabled")


class BoostingRefused(RuntimeError):
    def __init__(self, record: dict):
        super().__init__(f"{record['status']}: {record['reason']}")
        self.record = record


def _rec(status, reason, model_class, evidence=None):
    return {"status": status, "reason": reason, "model_class": model_class, "evidence": evidence}


def assert_boosting_active(model_class: str, request_has_speech_contexts: bool, runner_log_text: str,
                           *, active_pattern: Optional[str] = None,
                           disabled_patterns: Sequence[str] = DEFAULT_DISABLED_PATTERNS) -> dict:
    """Return an ACTIVE (or NOT_REQUESTED) record, or raise BoostingRefused carrying the record."""
    mc = model_class.strip().lower()
    if not request_has_speech_contexts:
        return _rec(NOT_REQUESTED, "no speech_contexts in request; unboosted arm", mc)
    if mc == "tdt":
        raise BoostingRefused(_rec(NO_RUNNER_PATH, "no runner path",
                                   mc, "TdtGreedyDecoder drops speech_contexts"))
    if mc not in BOOSTABLE_CLASSES:
        raise BoostingRefused(_rec(COULD_NOT_CHECK, f"unknown model class {model_class!r}", mc))
    for pat in disabled_patterns:
        m = re.search(pat, runner_log_text or "", re.I)
        if m:
            raise BoostingRefused(_rec(DISABLED_BY_RUNNER, "runner log reports boosting disabled",
                                       mc, m.group(0)))
    if not active_pattern:
        raise BoostingRefused(_rec(COULD_NOT_CHECK, "no active_pattern supplied; absence of "
                                   "evidence is not evidence of boosting", mc))
    m = re.search(active_pattern, runner_log_text or "", re.M)
    if not m:
        raise BoostingRefused(_rec(COULD_NOT_CHECK, "positive evidence not found in runner log", mc))
    return _rec(ACTIVE, "positive evidence in runner log", mc, m.group(0))


def effective_config_record(*, runner_commit: str, boosting_tree_alpha: float,
                            request_boost: Optional[float], clamp_lo: float, clamp_hi: float,
                            depth_scaling: float, unk_score: float, case_policy: str,
                            greedy: bool, tree_node_count: int, lexicon_sha256: str) -> dict:
    """Effective boosting config. alpha = boosting_tree_alpha x clamp(request_boost, lo, hi);
    request_boost None or <= 0 => factor 1.0 (the runner's no-boost default)."""
    if clamp_lo > clamp_hi:
        raise ValueError("clamp_lo > clamp_hi")
    if not re.fullmatch(r"[0-9a-f]{64}", lexicon_sha256 or ""):
        raise ValueError("lexicon_sha256 must be 64 lowercase hex chars")
    if not isinstance(tree_node_count, int) or tree_node_count < 0:
        raise ValueError("tree_node_count must be a non-negative int")
    if request_boost is not None and request_boost > 0:
        factor = max(clamp_lo, min(request_boost, clamp_hi))
    else:
        factor = 1.0
    return {"runner_commit": runner_commit, "boosting_tree_alpha": boosting_tree_alpha,
            "request_boost": request_boost, "clamp_lo": clamp_lo, "clamp_hi": clamp_hi,
            "clamped_request_boost": factor, "alpha": boosting_tree_alpha * factor,
            "depth_scaling": depth_scaling, "unk_score": unk_score, "case_policy": case_policy,
            "greedy": greedy, "tree_node_count": tree_node_count, "lexicon_sha256": lexicon_sha256}
