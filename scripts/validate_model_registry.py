#!/usr/bin/env python3
"""Validate model_registry.yaml internal consistency.

Added 2026-05-27 after a disk cleanup left dangling references (deleted GGUFs still
named by active roles; a deprecated role still in escalation chains/routing). Catches
the four drift classes the operator called out:

  1. active role -> model file exists      (ERROR if the role is *deployable*; WARNING
                                             if it is a pure catalogue entry — the research
                                             registry is the comprehensive record, so an
                                             off-disk catalogue model is allowed, see
                                             feedback_registry_scope)
  2. deprecated role NOT in process_layout                                  (ERROR)
  3. deprecated role NOT an escalation-chain / routing-hint target          (ERROR)
  4. section drift: server_mode.<role>.model == roles.<role>.model.path     (WARNING)

"Deployable" = referenced by process_layout (what actually launches), escalation_chains,
or routing_hints (what routing actually targets). server_mode alone is a launch *spec*
catalogue, so a spec for a not-currently-deployed role is not itself an error.

SEC-ART-5 (2026-10-21, handoffs/active/model-artifact-integrity.md in epyc-root) adds a fifth class:

  5. every deployable GGUF-serving server_mode role carries chat_template.{upstream_template_sha256,
     served_template_sha256, template_diff}: both digests 64 lowercase hex, `template_diff` an existing
     non-empty file (repo-relative; the reviewed upstream -> served diff), and served_template_sha256_12
     (when present) the prefix of the 64-hex pin                                                  (ERROR)
     Produce them with epyc-orchestrator `scripts/validate/template_provenance.py` (new-model Step 1.5).

Exit code: 1 if any ERROR, else 0 (WARNINGs do not fail — they are hygiene the operator
reviews). Designed to run standalone or in CI:

    python3 scripts/validate_model_registry.py [path/to/model_registry.yaml]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from registry import ModelRegistry, DEFAULT_REGISTRY_PATH  # noqa: E402

# server_mode 'model:' is a bare filename — it resolves against the GGUF models dir,
# NOT the lmstudio model_base_path that roles.<role>.model.path (relative) uses.
MODELS_DIR = "/mnt/raid0/llm/models"
MODEL_BASE_PATH = "/mnt/raid0/llm/lmstudio/models"


def _is_local_gguf(path: str | None) -> bool:
    """True only for local GGUF file paths — skips remote/HF-id models (whisper, ColBERT,
    image models) whose 'path' is a repo id, not a file we expect on disk."""
    return bool(path) and path.endswith(".gguf")


def _resolve_server_mode_model(model) -> str | None:
    """server_mode model is a bare filename (-> MODELS_DIR), an absolute path, or a dict."""
    if isinstance(model, dict):
        p = model.get("path")
        if not p:
            return None
        return p if p.startswith("/") else os.path.join(MODEL_BASE_PATH, p)
    if not isinstance(model, str) or not model:
        return None
    if model.startswith("/"):
        return model
    return os.path.join(MODELS_DIR, model)


def _deployable_roles(data: dict) -> set[str]:
    """Roles that are actually launched or routed-to (must have a present model file)."""
    refs: set[str] = set()
    for members in data.get("process_layout", {}).values():
        if isinstance(members, list):
            refs.update(members)
    for chain in data.get("escalation_chains", {}).values():
        if isinstance(chain, dict):
            refs.update(chain.get("chain", []) or [])
    for hint in data.get("routing_hints", []) or []:
        if isinstance(hint, dict):
            refs.update(hint.get("use", []) or [])
            esc = hint.get("escalate_to")
            if esc:
                refs.add(esc)
    return refs


_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_FIELDS = ("upstream_template_sha256", "served_template_sha256", "template_diff")


def check_template_provenance(data: dict, deployable: set[str], repo_root: Path = _REPO_ROOT) -> list[tuple[str, str]]:
    """SEC-ART-5 over the deployable, non-alias GGUF-serving `server_mode` roles. Returns ERROR tuples."""
    errors: list[tuple[str, str]] = []
    for role, cfg in sorted((data.get("server_mode") or {}).items()):
        if role not in deployable or not isinstance(cfg, dict) or cfg.get("alias_of"):
            continue
        model = cfg.get("model")
        mpath = model.get("path") if isinstance(model, dict) else model
        if not (isinstance(mpath, str) and mpath.endswith(".gguf")):
            continue
        block = cfg.get("chat_template") if isinstance(cfg.get("chat_template"), dict) else {}
        missing = [f for f in TEMPLATE_FIELDS if not block.get(f)]
        if missing:
            errors.append(("MISSING_TEMPLATE_PROVENANCE",
                           f"deployable role '{role}': chat_template lacks {', '.join(missing)} "
                           f"(run epyc-orchestrator scripts/validate/template_provenance.py)"))
            continue
        for f in ("upstream_template_sha256", "served_template_sha256"):
            if not (isinstance(block[f], str) and _SHA64.match(block[f])):
                errors.append(("BAD_TEMPLATE_PIN", f"role '{role}': chat_template.{f}={block[f]!r} is not 64 lowercase hex"))
        short = block.get("served_template_sha256_12")
        if short is not None and isinstance(block["served_template_sha256"], str) \
                and str(short) != block["served_template_sha256"][:12]:
            errors.append(("TEMPLATE_PIN_PREFIX_MISMATCH",
                           f"role '{role}': served_template_sha256_12 {short!r} != prefix of served_template_sha256"))
        diff = Path(str(block["template_diff"]))
        diff = diff if diff.is_absolute() else repo_root / diff
        if not diff.is_file() or diff.stat().st_size == 0:
            errors.append(("TEMPLATE_DIFF_MISSING", f"role '{role}': template_diff {block['template_diff']} is not a non-empty file"))
    return errors


def validate(registry_path: str) -> tuple[list, list]:
    reg = ModelRegistry(registry_path)
    data = reg.data
    roles = data.get("roles", {})
    deprecated = {r for r, c in roles.items() if isinstance(c, dict) and c.get("deprecated")}
    active = set(reg.get_all_roles())  # non-deprecated
    deployable = _deployable_roles(data)

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []

    # --- Check 1: active role model files exist ---
    for role in sorted(active):
        path = reg.get_model_path(role)
        if not _is_local_gguf(path) or os.path.exists(path):
            continue
        if role in deployable:
            errors.append(("MISSING_DEPLOYABLE_MODEL",
                           f"active+deployable role '{role}' -> missing model file: {path}"))
        else:
            warnings.append(("MISSING_CATALOGUE_MODEL",
                             f"active catalogue role '{role}' -> off-disk model: {path} "
                             f"(deprecate, download, or accept as research record)"))

    # --- Check 2: deprecated roles must not be in process_layout ---
    pl_roles: set[str] = set()
    for members in data.get("process_layout", {}).values():
        if isinstance(members, list):
            pl_roles.update(members)
    for r in sorted(deprecated & pl_roles):
        errors.append(("DEPRECATED_IN_PROCESS_LAYOUT", f"deprecated role '{r}' is in process_layout"))

    # --- Check 3: deprecated roles must not be escalation/routing targets ---
    for cname, chain in data.get("escalation_chains", {}).items():
        if isinstance(chain, dict):
            for r in chain.get("chain", []) or []:
                if r in deprecated:
                    errors.append(("DEPRECATED_IN_CHAIN",
                                   f"deprecated role '{r}' in escalation_chain '{cname}'"))
    for i, hint in enumerate(data.get("routing_hints", []) or []):
        if not isinstance(hint, dict):
            continue
        cond = hint.get("if", "?")
        for r in hint.get("use", []) or []:
            if r in deprecated:
                errors.append(("DEPRECATED_IN_ROUTING_USE",
                               f"deprecated role '{r}' in routing_hint #{i} (if: {cond})"))
        if hint.get("escalate_to") in deprecated:
            errors.append(("DEPRECATED_IN_ESCALATE_TO",
                           f"deprecated role '{hint['escalate_to']}' as escalate_to in "
                           f"routing_hint #{i} (if: {cond})"))

    # --- Check 4: section drift between server_mode and roles for active runtime roles ---
    # Compares more than the model basename: a matching GGUF can still hide stale runtime
    # semantics (e.g. a Qwen3.6 path left with a Qwen3.5 MoE-reduction recipe or qwen35
    # model_role), which is the class of drift the basename-only check missed.
    for role, cfg in data.get("server_mode", {}).items():
        if role in deprecated or role not in roles or not isinstance(cfg, dict):
            continue
        rc = roles[role] if isinstance(roles.get(role), dict) else {}
        rcm = rc.get("model") or {}
        # 4a model basename
        sm_path = _resolve_server_mode_model(cfg.get("model"))
        roles_path = reg.get_model_path(role)
        if _is_local_gguf(sm_path) and _is_local_gguf(roles_path) \
                and os.path.basename(sm_path) != os.path.basename(roles_path):
            warnings.append(("SECTION_DRIFT_MODEL",
                             f"role '{role}': server_mode model '{os.path.basename(sm_path)}' "
                             f"!= roles model '{os.path.basename(roles_path)}'"))
        # 4b acceleration.type
        sm_accel = (cfg.get("acceleration") or {}).get("type")
        roles_accel = (rc.get("acceleration") or {}).get("type")
        if sm_accel and roles_accel and sm_accel != roles_accel:
            warnings.append(("SECTION_DRIFT_ACCEL",
                             f"role '{role}': server_mode acceleration.type '{sm_accel}' "
                             f"!= roles acceleration.type '{roles_accel}'"))
        # 4c thinking: server_mode chat_template_kwargs.enable_thinking should be the
        #     negation of roles.model.disable_thinking
        sm_think = (cfg.get("chat_template_kwargs") or {}).get("enable_thinking")
        roles_think_off = rcm.get("disable_thinking")
        if sm_think is not None and roles_think_off is not None \
                and bool(sm_think) == bool(roles_think_off):
            warnings.append(("SECTION_DRIFT_THINKING",
                             f"role '{role}': server_mode enable_thinking={sm_think} inconsistent "
                             f"with roles disable_thinking={roles_think_off}"))
        # 4d model_role version token must appear in the model filename (catches qwen35 model_role
        #    left on a Qwen3.6 GGUF — exactly finding #1)
        mr_norm = str(cfg.get("model_role") or "").replace(".", "").replace("_", "").lower()
        smf_norm = os.path.basename(sm_path or "").replace(".", "").replace("_", "").lower()
        mtok = re.search(r"qwen3[0-9]", mr_norm)
        if mtok and smf_norm and mtok.group(0) not in smf_norm:
            warnings.append(("SECTION_DRIFT_MODEL_ROLE",
                             f"role '{role}': model_role '{cfg.get('model_role')}' version token "
                             f"'{mtok.group(0)}' absent from model '{os.path.basename(sm_path or '')}'"))

    # --- Check 5: serving_floors (measured noise floors; the only floor store) ---
    # Rules live in floor_registry.validate_section so the reader, the applier and this
    # validator cannot disagree. The module is loaded by file path (it is stdlib + yaml and has no
    # package-relative imports), not through the heavy `autokernel` package.
    if "serving_floors" in data:
        import importlib.util
        fr_path = (Path(__file__).resolve().parent / "kernel_rnd" / "autokernel" / "loop"
                   / "floor_registry.py")
        spec = importlib.util.spec_from_file_location("floor_registry_validate", fr_path)
        fr = importlib.util.module_from_spec(spec)
        sys.modules["floor_registry_validate"] = fr
        spec.loader.exec_module(fr)
        floor_errors, floor_warnings = fr.validate_section(data, Path(registry_path).resolve().parent.parent)
        errors.extend(("SERVING_FLOOR", m) for m in floor_errors)
        warnings.extend(("SERVING_FLOOR", m) for m in floor_warnings)

    # --- Check 6 (SEC-ART-5, docstring class 5): template provenance for deployable GGUF-serving roles ---
    errors += check_template_provenance(data, deployable)

    return errors, warnings


def main() -> int:
    registry_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REGISTRY_PATH
    errors, warnings = validate(registry_path)

    print(f"model_registry.yaml consistency check: {registry_path}")
    for code, msg in errors:
        print(f"  ERROR   [{code}] {msg}")
    for code, msg in warnings:
        print(f"  WARNING [{code}] {msg}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
