"""YARN-AGG-1 window plans: WHICH cells each filler runs, frozen to disk before the first item.

A plan is pre-registered: ``arm_runner plan`` materialises it once into ``<out>/plan/`` (items + blobs +
``plan.json`` with ``plan_sha256``) and every later window reloads that frozen copy, so both arms and every
resumed slice see byte-identical prompts. Edit the specs below ONLY before the first freeze; a changed spec
after a freeze is refused (the plan_sha256 of the frozen copy is what the decision record cites).

Item order is a round-robin over the segments' context units (one unit = the items sharing one cached
context), so a slice of any length covers every cell instead of finishing one cell first.

Segment keys
  name      unique segment id
  builder   oolong | ruler | helmet | longbio
  args      builder kwargs
  arms      the arms that run it (["A0", "A1"] = paired; ["A1"] = beyond-native, A0 cannot serve it)
  gate      native_mcnemar (counts toward the ratified "Net, paired" rule, only for pairs whose BOTH real
            prompt_n land in the native zone) | report_only
  required  True: plan freeze refuses while the builder reports the data unavailable
            False: the segment is dropped at freeze time and the reason is recorded in plan.json
"""

from __future__ import annotations

from typing import Any

PAIRED = ["A0", "A1"]

#: yarn-native-gate (YARN-AGG-1(i)): the in-native-window gate, all paired, thinking off.
#: Sized for ~2 h per arm on the MI210 (E1 rates: ~400 tok/s prefill at 128K, ~12 tok/s decode at depth).
NATIVE_GATE: list[dict[str, Any]] = [
    {"name": "ruler_mk2_32k", "builder": "ruler", "args": {"task": "niah_multikey_2", "target_real": 32768, "n": 6},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk3_32k", "builder": "ruler", "args": {"task": "niah_multikey_3", "target_real": 32768, "n": 6},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mv_32k", "builder": "ruler", "args": {"task": "niah_multivalue", "target_real": 32768, "n": 6},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "oolong_131072", "builder": "oolong", "args": {"nominal": 131072, "max_contexts": 8},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk2_64k", "builder": "ruler", "args": {"task": "niah_multikey_2", "target_real": 65536, "n": 3},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk3_64k", "builder": "ruler", "args": {"task": "niah_multikey_3", "target_real": 65536, "n": 3},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mv_64k", "builder": "ruler", "args": {"task": "niah_multivalue", "target_real": 65536, "n": 3},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk2_128k", "builder": "ruler", "args": {"task": "niah_multikey_2", "target_real": 126976, "n": 2},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk3_128k", "builder": "ruler", "args": {"task": "niah_multikey_3", "target_real": 126976, "n": 2},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mv_128k", "builder": "ruler", "args": {"task": "niah_multivalue", "target_real": 126976, "n": 2},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
]


def _within(target: int, frac: float = 0.10):
    return lambda n: n is not None and abs(n - target) <= frac * target


#: yarn-extension-arms (YARN-AGG-1(ii)): 240K in-native cells (paired, same McNemar rule) and the
#: beyond-native cells (A1 only, report-only), plus HELMET json_kv / RAG and LongBioBench. Judge-free:
#: judged categories go through YARN-JUDGE-1 (run_local_judge_window.sh) and are not in this plan.
EXTENSION_ARMS: list[dict[str, Any]] = [
    # --- 240K, inside the native window: paired, counts toward the native rule -----------------------
    {"name": "oolong_240k", "builder": "oolong",
     "args": {"nominal": 262144, "max_contexts": 4, "context_filter_target": 245760, "label": "240k"},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk2_240k", "builder": "ruler", "args": {"task": "niah_multikey_2", "target_real": 245760, "n": 1},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mk3_240k", "builder": "ruler", "args": {"task": "niah_multikey_3", "target_real": 245760, "n": 1},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "ruler_mv_240k", "builder": "ruler", "args": {"task": "niah_multivalue", "target_real": 245760, "n": 1},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "helmet_json_kv_128k", "builder": "helmet", "args": {"task": "json_kv", "target_real": 126976, "n": 2},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "helmet_json_kv_240k", "builder": "helmet", "args": {"task": "json_kv", "target_real": 245760, "n": 1},
     "arms": PAIRED, "gate": "native_mcnemar", "required": True},
    {"name": "longbio_standard_128k", "builder": "longbio",
     "args": {"variant": "standard", "target_real": 126976, "n": 2}, "arms": PAIRED, "gate": "native_mcnemar",
     "required": True},
    {"name": "longbio_multi5_128k", "builder": "longbio",
     "args": {"variant": "multi_standard_5", "target_real": 126976, "n": 1}, "arms": PAIRED,
     "gate": "native_mcnemar", "required": True},
    {"name": "longbio_multi10_128k", "builder": "longbio",
     "args": {"variant": "multi_standard_10", "target_real": 126976, "n": 1}, "arms": PAIRED,
     "gate": "native_mcnemar", "required": True},
    {"name": "longbio_paraphrase_128k", "builder": "longbio",
     "args": {"variant": "paraphrase", "target_real": 126976, "n": 2}, "arms": PAIRED, "gate": "native_mcnemar",
     "required": True},
    {"name": "longbio_standard_240k", "builder": "longbio",
     "args": {"variant": "standard", "target_real": 245760, "n": 1}, "arms": PAIRED, "gate": "native_mcnemar",
     "required": True},
    {"name": "helmet_nq_128k", "builder": "helmet", "args": {"task": "nq", "target_real": 126976, "n": 2},
     "arms": PAIRED, "gate": "native_mcnemar", "required": False},
    {"name": "helmet_hotpotqa_128k", "builder": "helmet", "args": {"task": "hotpotqa", "target_real": 126976, "n": 2},
     "arms": PAIRED, "gate": "native_mcnemar", "required": False},
    # --- beyond the native window: A1 only, report-only ----------------------------------------------
    {"name": "oolong_400k", "builder": "oolong",
     "args": {"nominal": 524288, "max_contexts": 3, "context_filter_target": 409600, "label": "400k"},
     "arms": ["A1"], "gate": "report_only", "required": True},
    {"name": "ruler_mk2_400k", "builder": "ruler", "args": {"task": "niah_multikey_2", "target_real": 409600, "n": 1},
     "arms": ["A1"], "gate": "report_only", "required": True},
    {"name": "ruler_mv_400k", "builder": "ruler", "args": {"task": "niah_multivalue", "target_real": 409600, "n": 1},
     "arms": ["A1"], "gate": "report_only", "required": True},
    {"name": "helmet_json_kv_400k", "builder": "helmet", "args": {"task": "json_kv", "target_real": 409600, "n": 1},
     "arms": ["A1"], "gate": "report_only", "required": True},
    {"name": "longbio_standard_400k", "builder": "longbio",
     "args": {"variant": "standard", "target_real": 409600, "n": 1}, "arms": ["A1"], "gate": "report_only",
     "required": True},
    {"name": "ruler_mk2_500k", "builder": "ruler", "args": {"task": "niah_multikey_2", "target_real": 491520, "n": 1},
     "arms": ["A1"], "gate": "report_only", "required": True},
    # --- OOL-1 thinking_on arm (fixed cap 16384, pre-registered; R2 precedent), native bucket, paired ---
    {"name": "oolong_131072_think", "builder": "oolong",
     "args": {"nominal": 131072, "max_contexts": 1, "thinking": True, "max_questions_per_context": 6},
     "arms": PAIRED, "gate": "report_only", "required": True},
]

PLANS = {"native-gate": NATIVE_GATE, "extension-arms": EXTENSION_ARMS}

#: Server arm configs (E1 launch_arm.sh A0 / A1, unchanged). ARM_CTX is the -c the server is launched with.
ARM_CTX = {"A0": 262144, "A1": 524288}
