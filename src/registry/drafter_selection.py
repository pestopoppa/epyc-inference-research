"""Drafter SELECTION: master lists every acceptable drafter, topology picks one.

DRAFT-SEL-1 (proposed 2026-10-01). Origin: production :8083 served Qwen3.8-27B with
`--spec-type draft-mtp` although the operator had ruled DFlash2 the 27B's spec-decode
path (2026-08-27, autokernel-champion-aggregate ruling 3; restated 2026-10-01 "ALWAYS
use dflash2"). The drafter was a hand-copied per-role field (`acceleration.spec_type`,
`draft_model`, `draft_max`) restated in five places for one process, under a global
`speculative_decoding_policy.production_recipe: draft-mtp`. Every model swap
(7483d7fb, ARCHSWAP-20260927) changed the model path and re-inherited the copied
MTP fields; nothing structural could carry a different drafter to the launch.

The contract this module enforces:

* MASTER  ``roles.<model_role>.drafters`` — a mapping ``{drafter_id: recipe}`` of every
  ACCEPTABLE drafter for THAT MODEL, each with its launch parameters.
* TOPOLOGY ``stack_topology.yaml -> drafter_selection.<server>`` — which drafter_id
  the server (the launching PRIMARY, never an alias) runs. ``none`` is reserved and
  means "launch without speculation" on purpose.
* COMPILER projects the selected recipe into the lean view (``server_mode.<server>``,
  ``roles.<server>`` and every alias in ``shared_with``) and stamps the provenance.
* A server whose model declares ``drafters`` may NOT hand-carry drafter fields in the
  master; the projection is the only writer.

Models without ``drafters`` are LEGACY rows: their hand-carried fields still pass
through unchanged (with a warning) so the fix can migrate one model at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPOLOGY_PATH = _REPO_ROOT / "orchestration" / "stack_topology.yaml"

SELECTION_SECTION = "drafter_selection"
NO_SPECULATION = "none"
# serving_shape key naming the drafter `vram_non_kv_gib` was derived for (optional).
VRAM_DRAFTER_KEY = "vram_non_kv_drafter"

# Acceleration keys that DESCRIBE THE DRAFTER. When a model declares `drafters`, none
# of these may be written by hand on a server/role/alias that serves it.
DRAFTER_FIELDS: tuple[str, ...] = (
    "spec_type",
    "draft_model",
    "draft_model_path",
    "draft_max",
    "k",
    "draft_min",
    "draft_p_min",
    "draft_p_split",
    "n_gpu_layers_draft",
    "threads_draft",
    "ngram_mod_n_min",
    "ngram_mod_n_max",
    "ngram_mod_n_match",
)

# Recipe keys a `drafters.<id>` entry may carry, and the acceleration key each
# projects to. Anything else in a drafter entry is metadata (evidence, rulings,
# recipe pointer); only `recipe` is copied into the lean's `drafter_selection` stamp.
_RECIPE_TO_ACCEL: dict[str, str] = {
    "spec_type": "spec_type",
    "draft_model": "draft_model",
    "draft_max": "draft_max",
    "draft_min": "draft_min",
    "draft_p_min": "draft_p_min",
    "draft_p_split": "draft_p_split",
    "ngld": "n_gpu_layers_draft",
    "threads_draft": "threads_draft",
}
_REQUIRED_RECIPE_KEYS = ("spec_type", "draft_model")

# --spec-type tokens the production kernel (v10, ffc1bac82) accepts for a drafter
# recipe: common/speculative.cpp:42 maps "draft-dflash"; draft-mtp is the NEXTN path.
KNOWN_SPEC_TYPES = frozenset({"draft-mtp", "draft-dflash", "draft-simple", "draft-eagle3"})


class DrafterSelectionError(ValueError):
    """The master/topology pair does not determine exactly one drafter per server."""


@dataclass
class Resolution:
    selected: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_drafter_selection(topology_path: Path | None = None) -> dict[str, str]:
    """Return ``{server_role: drafter_id}`` from stack_topology.yaml (empty if absent)."""
    path = topology_path or DEFAULT_TOPOLOGY_PATH
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get(SELECTION_SECTION) or {}
    if not isinstance(raw, dict):
        raise DrafterSelectionError(
            f"{path}: `{SELECTION_SECTION}` must be a mapping server_role -> drafter_id"
        )
    out: dict[str, str] = {}
    for server, drafter in raw.items():
        if not isinstance(drafter, str) or not drafter:
            raise DrafterSelectionError(
                f"{path}: {SELECTION_SECTION}.{server} must be a drafter id string, "
                f"got {drafter!r}"
            )
        out[str(server)] = drafter
    return out


def selection_cache_bytes(selection: dict[str, str]) -> bytes:
    """Canonical bytes of the selection, for the lean compile cache key."""
    return json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _model_role_of(server: str, entry: dict[str, Any]) -> str:
    model_role = entry.get("model_role")
    return model_role if isinstance(model_role, str) and model_role else server


def _alias_set(server_mode: dict[str, Any]) -> dict[str, str]:
    """alias role -> host server, from every server_mode.<host>.shared_with."""
    aliases: dict[str, str] = {}
    for host, entry in server_mode.items():
        if isinstance(entry, dict) and isinstance(entry.get("shared_with"), list):
            for alias in entry["shared_with"]:
                aliases[str(alias)] = host
    return aliases


def _hand_carried(block: Any) -> list[str]:
    if not isinstance(block, dict):
        return []
    return [key for key in DRAFTER_FIELDS if key in block]


def _validate_recipe(model_role: str, drafter_id: str, recipe: Any) -> list[str]:
    where = f"roles.{model_role}.drafters.{drafter_id}"
    if not isinstance(recipe, dict):
        return [f"{where} must be a mapping"]
    errors = [f"{where} is missing required key {key!r}" for key in _REQUIRED_RECIPE_KEYS
              if not recipe.get(key)]
    spec_type = recipe.get("spec_type")
    if isinstance(spec_type, str) and spec_type not in KNOWN_SPEC_TYPES:
        errors.append(
            f"{where}.spec_type {spec_type!r} is not a drafter spec type the production "
            f"kernel accepts ({sorted(KNOWN_SPEC_TYPES)})"
        )
    draft_max = recipe.get("draft_max")
    if draft_max is not None and (
        not isinstance(draft_max, int) or isinstance(draft_max, bool) or draft_max < 1
    ):
        errors.append(f"{where}.draft_max must be a positive int, got {draft_max!r}")
    return errors


def resolve(
    master: dict[str, Any],
    active_roles: set[str],
    selection: dict[str, str],
) -> Resolution:
    """Decide the drafter for every ACTIVE launching server. Pure; never raises."""
    res = Resolution()
    server_mode = master.get("server_mode") or {}
    roles = master.get("roles") or {}
    aliases = _alias_set(server_mode)

    for server, drafter_id in sorted(selection.items()):
        if server in aliases:
            res.errors.append(
                f"stack_topology.{SELECTION_SECTION}.{server}: {server!r} is an ALIAS on "
                f"{aliases[server]!r}'s process; a drafter is a property of the launching "
                f"process. Select it on {aliases[server]!r}."
            )
        elif server not in active_roles:
            res.warnings.append(
                f"stack_topology.{SELECTION_SECTION}.{server}={drafter_id!r} names a role "
                f"that is not in the active launch set; the selection is inert."
            )

    for server in sorted(active_roles):
        entry = server_mode.get(server)
        if not isinstance(entry, dict) or server in aliases:
            continue
        model_role = _model_role_of(server, entry)
        model_row = roles.get(model_role) if isinstance(roles.get(model_role), dict) else {}
        drafters = model_row.get("drafters")
        chosen = selection.get(server)

        if not drafters:
            if chosen is not None and chosen != NO_SPECULATION:
                res.errors.append(
                    f"stack_topology.{SELECTION_SECTION}.{server}={chosen!r}, but the model "
                    f"it serves (roles.{model_role}) lists no `drafters`. Add the drafter "
                    f"to the master registry first; topology may only select from that list."
                )
            elif _hand_carried(entry.get("acceleration")):
                res.warnings.append(
                    f"LEGACY drafter on {server!r}: master hand-carries "
                    f"{_hand_carried(entry.get('acceleration'))} for roles.{model_role}, which "
                    f"declares no `drafters`. Migrate it (DRAFT-SEL-1)."
                )
            continue

        if not isinstance(drafters, dict):
            res.errors.append(f"roles.{model_role}.drafters must be a mapping id -> recipe")
            continue
        for drafter_id, recipe in drafters.items():
            res.errors.extend(_validate_recipe(model_role, str(drafter_id), recipe))

        # The projection is the ONLY writer once a model declares drafters.
        holders = [
            (f"server_mode.{server}.acceleration", entry.get("acceleration")),
            (f"roles.{server}.acceleration", (roles.get(server) or {}).get("acceleration")),
        ]
        for alias in entry.get("shared_with") or []:
            holders.append((f"server_mode.{alias}.acceleration",
                            (server_mode.get(alias) or {}).get("acceleration")))
            holders.append((f"roles.{alias}.acceleration",
                            (roles.get(alias) or {}).get("acceleration")))
        for where, block in holders:
            carried = _hand_carried(block)
            if carried:
                res.errors.append(
                    f"{where} hand-carries drafter fields {carried} although "
                    f"roles.{model_role} declares `drafters`. Delete them: the compiler "
                    f"projects the drafter chosen by stack_topology.{SELECTION_SECTION}."
                )
        for key in ("draft_model", "draft_model_path"):
            if key in entry:
                res.errors.append(
                    f"server_mode.{server}.{key} is hand-carried although "
                    f"roles.{model_role} declares `drafters`. Delete it; the projection "
                    f"writes server_mode.{server}.draft_model from the selected drafter."
                )

        if chosen is None:
            if len(drafters) == 1:
                (only_id,) = drafters
                res.selected[server] = {"model_role": model_role, "drafter": str(only_id),
                                        "source": "sole_drafter"}
            else:
                res.errors.append(
                    f"{server!r} serves roles.{model_role}, which lists {len(drafters)} "
                    f"acceptable drafters {sorted(drafters)}, and stack_topology."
                    f"{SELECTION_SECTION} selects none. Add `{server}: <one of "
                    f"{sorted(drafters)}>` there. There is deliberately no default: a "
                    f"silent default is how :8083 re-inherited draft-mtp."
                )
            continue
        if chosen == NO_SPECULATION:
            res.selected[server] = {"model_role": model_role, "drafter": NO_SPECULATION,
                                    "source": f"stack_topology.{SELECTION_SECTION}"}
            continue
        if chosen not in drafters:
            res.errors.append(
                f"stack_topology.{SELECTION_SECTION}.{server}={chosen!r} is not an acceptable "
                f"drafter for roles.{model_role} (master lists {sorted(drafters)})."
            )
            continue
        res.selected[server] = {"model_role": model_role, "drafter": chosen,
                                "source": f"stack_topology.{SELECTION_SECTION}"}

    # The VRAM capacity figure is a function of the drafter (weights, draft context, GDN
    # rollback ring sized by draft_max). A server that names the drafter its
    # `serving_shape.vram_non_kv_gib` was derived for must run THAT drafter; flipping the
    # selection without re-deriving the figure would pass the capacity gate on a stale
    # number and fail at HIP allocation time (STACKCHG-DFLASH2-20261003).
    for server, sel in sorted(res.selected.items()):
        shape = (server_mode.get(server) or {}).get("serving_shape")
        derived_for = shape.get(VRAM_DRAFTER_KEY) if isinstance(shape, dict) else None
        if derived_for is not None and derived_for != sel["drafter"]:
            res.errors.append(
                f"server_mode.{server}.serving_shape.vram_non_kv_gib was derived for drafter "
                f"{derived_for!r} ({VRAM_DRAFTER_KEY}), but {server!r} runs {sel['drafter']!r}. "
                f"Re-derive vram_non_kv_gib for the selected drafter and update "
                f"{VRAM_DRAFTER_KEY} in the same change."
            )
    return res


def _accel_fields(recipe: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for recipe_key, accel_key in _RECIPE_TO_ACCEL.items():
        if recipe.get(recipe_key) is not None:
            out[accel_key] = recipe[recipe_key]
    if "draft_max" in out:
        out["k"] = out["draft_max"]  # RegistryLoader reads `k` (registry_loader.py:407)
    return out


def _stamp(server: str, sel: dict[str, Any], recipe: dict[str, Any] | None) -> dict[str, Any]:
    stamp = {"server": server, "model_role": sel["model_role"],
             "drafter": sel["drafter"], "source": sel["source"]}
    if recipe and isinstance(recipe.get("recipe"), str):
        stamp["recipe"] = recipe["recipe"]  # evidence/rulings stay on the master row
    return stamp


def project(lean: dict[str, Any], master: dict[str, Any], resolution: Resolution) -> None:
    """Write the selected drafter into the lean view, in place."""
    server_mode = lean.get("server_mode") or {}
    roles = lean.get("roles") or {}
    master_roles = master.get("roles") or {}

    for server, sel in resolution.selected.items():
        host = server_mode.get(server)
        if not isinstance(host, dict):
            continue
        if sel["drafter"] == NO_SPECULATION:
            recipe = None
            fields: dict[str, Any] = {}
            accel_type = "none"
        else:
            recipe = master_roles[sel["model_role"]]["drafters"][sel["drafter"]]
            fields = _accel_fields(recipe)
            accel_type = "speculative_decoding"
        stamp = _stamp(server, sel, recipe)

        for block_owner in (host, roles.get(server)):
            if not isinstance(block_owner, dict):
                continue
            accel = dict(block_owner.get("acceleration") or {})
            for key in DRAFTER_FIELDS:
                accel.pop(key, None)
            accel["type"] = accel_type
            accel.update(fields)
            accel["drafter_selection"] = stamp
            block_owner["acceleration"] = accel
        host.pop("draft_model_path", None)
        if "draft_model" in fields:
            # stack_priors._server_mode_launch_requirement_overrides turns this into
            # requirements.draft_model_path, which outranks acceleration.draft_model.
            host["draft_model"] = fields["draft_model"]
        else:
            host.pop("draft_model", None)

        # Aliases share the process, so they read the host's drafter, never their own.
        for alias in host.get("shared_with") or []:
            for block_owner in (server_mode.get(alias), roles.get(alias)):
                if not isinstance(block_owner, dict):
                    continue
                accel = dict(block_owner.get("acceleration") or {})
                for key in DRAFTER_FIELDS:
                    accel.pop(key, None)
                accel["type"] = "none"  # "launches no draft of its own"
                accel["inherits_spec_from"] = server
                accel.update(fields)    # what the process actually runs, for readers
                accel["drafter_selection"] = dict(stamp, inherited_by=alias)
                block_owner["acceleration"] = accel


def check_lean(lean: dict[str, Any], selection: dict[str, str]) -> list[str]:
    """Validator rule over the COMPILED lean: projection present, intact, and current."""
    errors: list[str] = []
    server_mode = lean.get("server_mode") or {}
    roles = lean.get("roles") or {}
    aliases = _alias_set(server_mode)
    for server, entry in sorted(server_mode.items()):
        if not isinstance(entry, dict) or server in aliases:
            continue
        model_role = _model_role_of(server, entry)
        drafters = (roles.get(model_role) or {}).get("drafters")
        chosen = selection.get(server)
        if chosen is not None and chosen != NO_SPECULATION and (
            not isinstance(drafters, dict) or chosen not in drafters
        ):
            errors.append(
                f"drafter_selection: topology selects {chosen!r} for {server!r}; "
                f"roles.{model_role}.drafters does not list it"
            )
            continue
        if not drafters:
            continue
        if chosen is None and len(drafters) > 1:
            errors.append(
                f"drafter_selection: {server!r} serves {model_role!r} "
                f"({len(drafters)} drafters) with no topology selection"
            )
            continue
        expected_id = chosen or next(iter(drafters))
        stamp = (entry.get("acceleration") or {}).get("drafter_selection") or {}
        if stamp.get("drafter") != expected_id:
            errors.append(
                f"drafter_selection: lean server_mode.{server} carries drafter "
                f"{stamp.get('drafter')!r}, topology selects {expected_id!r} - the lean is "
                f"stale or hand-edited; recompile from master"
            )
            continue
        if expected_id == NO_SPECULATION:
            continue
        want = _accel_fields(drafters[expected_id])
        have = {k: (entry.get("acceleration") or {}).get(k) for k in want}
        if have != want:
            errors.append(
                f"drafter_selection: lean server_mode.{server}.acceleration {have} differs "
                f"from roles.{model_role}.drafters.{expected_id} {want}"
            )
    return errors
