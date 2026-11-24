"""AK-SURFACE-SERVING-1: serving-recipe knobs on the AutoKernel manipulation surface.

Policy: epyc-root docs/guides/agent-workflows/agent-loop-design.md, "Serving knobs are on the
AutoKernel manipulation surface".

WHY A SIDECAR AND NOT A RECIPE FIELD.  `serving.Recipe.recipe_hash` covers every field, so a
new field would invalidate every calibrated serving floor (the same reason `production_surfaces`
is a sidecar). A knob delta is therefore its own immutable object, bound to the base recipe by
`base_recipe_hash` and carried beside it; `Recipe` is never touched.

Three pure parts, none of which launches anything:

  * schema + actuator: a CLOSED knob table, validated against the registry `server_mode.<server>`
    shape, rendered into an argv delta by `render`. Nobody hand-types an argv: the base is
    `Recipe.server_argv`, then the registry shape is overlaid, then the delta.
  * screening: ONE evaluator, `serving_knob_measure.screen_cases` (GiB, two-case schema), reached by
    `serving_knob_measure.screen_stage`. SCREENING ONLY: arms differ in np / batch / drafter, so it is
    not an apples-to-apples gate and never promotes (which is why the kind is NOT in
    candidate_manifest.KEEP_KINDS: a screening candidate is not a keep).
  * seed + planner emission: `load_seed`, `portfolio_rows`, `emit_candidate`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import sampling_axis

SCHEMA = "epyc.autokernel.serving_knob.v1"
SEED_SCHEMA = "epyc.autokernel.serving_knob_seed.v1"
CANDIDATE_SCHEMA = "epyc.autokernel.serving_knob_candidate.v1"
SCREEN_SCHEMA = "epyc.autokernel.serving_knob_screen.v2"
SEED_KIND = "serving-knob"
SEED_SUFFIX = ".serving-knob.json"
CONTROL = "control"
ITL_REGRESSION_MAX = 0.20


class KnobError(ValueError):
    """A knob delta, seed or base shape cannot be honestly rendered."""


class ScreeningInvalid(ValueError):
    """The measurements cannot support a ranking (bad numbers, or the control itself failed)."""


# --- schema ------------------------------------------------------------------------------
#: knob -> (flag aliases, kind). The actuator writes the FIRST alias.
_FLAGS: dict[str, tuple[tuple[str, ...], str]] = {
    "batch": (("-b", "--batch-size"), "int"), "ubatch": (("-ub", "--ubatch-size"), "int"),
    "np": (("-np", "--parallel"), "int"), "kv_unified": (("--kv-unified", "--no-kv-unified"), "switch"),
    "ctx_checkpoints": (("--ctx-checkpoints", "-ctxcp"), "int"), "cache_ram": (("--cache-ram", "-cram"), "int"),
    "spec_draft_n_max": (("--spec-draft-n-max",), "int"), "drafter": ((), "drafter"),
    "temperature": (("--temp", "--temperature"), "float"), "top_p": (("--top-p",), "float"),
    "top_k": (("--top-k",), "int"),
    "n_ctx": (("-c", "--ctx-size"), "int"), "ctk": (("-ctk", "--cache-type-k"), "enum"),
    "ctv": (("-ctv", "--cache-type-v"), "enum"),
}
KNOBS = tuple(_FLAGS)
SAMPLING = ("temperature", "top_p", "top_k")
#: KV cache types a K/V knob may take. A quantized V cache needs flash-attn (checked at render, against the argv).
KV_TYPES = ("f16", "bf16", "q8_0", "q4_0")
_QUANTIZED_KV = frozenset({"q8_0", "q4_0"})
_BOUNDS = {"batch": (1, 65536), "ubatch": (1, 65536), "np": (1, 16), "ctx_checkpoints": (0, 256),
           "cache_ram": (0, 1 << 21), "spec_draft_n_max": (1, 32), "top_k": (0, 1000),
           "n_ctx": (1, 1 << 24)}


@dataclass(frozen=True)
class ServingBase:
    """The registry `server_mode.<server>` serving shape the delta is measured over."""
    server: str
    values: Mapping[str, Any]                 # knob -> registry value (absent = registry silent)
    drafters: Mapping[str, Mapping[str, Any]]  # eligible drafter id -> {spec_type, draft_model, draft_max, ngld}
    sampling_allowed: Mapping[str, tuple]     # sampling knob -> values some served role declares
    model: str | None = None

    def digest(self) -> str:
        return _digest({"server": self.server, "values": dict(self.values),
                        "drafters": {k: dict(v) for k, v in self.drafters.items()},
                        "sampling": {k: list(v) for k, v in self.sampling_allowed.items()}})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def base_from_registry(server: str, registry: Any = None) -> ServingBase:
    """Read one server's shape from the orchestrator registry. An unreadable or missing entry
    raises: a delta is never validated against an invented base."""
    reg = sampling_axis._registry(registry)
    entry = (reg.get("server_mode") or {}).get(server)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("serving_shape"), Mapping):
        raise KnobError(f"registry server_mode.{server} has no serving_shape")
    shape, accel = entry["serving_shape"], entry.get("acceleration") or {}
    roles = reg.get("roles") or {}
    drafters = {k: dict(v) for k, v in ((roles.get(entry.get("model_role")) or {}).get("drafters") or {}).items()
                if isinstance(v, Mapping)}
    chosen = (accel.get("drafter_selection") or {}).get("drafter")
    kv = shape.get("kv_quant") or {}
    values: dict[str, Any] = {"np": entry.get("slots"), "batch": shape.get("batch"), "ubatch": shape.get("ubatch"),
                              "kv_unified": shape.get("kv_unified"), "cache_ram": entry.get("cache_ram"),
                              "ctx_checkpoints": entry.get("ctx_checkpoints"),
                              "spec_draft_n_max": accel.get("draft_max"), "drafter": chosen,
                              "n_ctx": shape.get("n_ctx"), "ctk": kv.get("k"), "ctv": kv.get("v")}
    # A knob renders as a SERVER-WIDE flag (--temp ...), so it is allowed only when EVERY role served on this
    # server declares it (a single-role server is the one-role case); the values are those the roles declare.
    served = list(dict.fromkeys([entry.get("model_role"), *(entry.get("shared_with") or ())]))
    allowed: dict[str, set] = {k: set() for k in SAMPLING}
    for key in SAMPLING:
        declared = [(roles.get(r) or {}).get("generation_defaults", {}).get(key) for r in served]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in declared):
            allowed[key] = set(declared)
    return ServingBase(
        server=server, values={k: v for k, v in values.items() if v is not None},
        drafters=drafters, sampling_allowed={k: tuple(sorted(v)) for k, v in allowed.items()},
        model=entry.get("model"))


def vram_capacity_gib(server: str, registry: Any = None) -> float:
    """The fit bound for the screen: the card ceiling the loop already enforces
    (`surface_validation.GPU_CAPACITY_CEILING_BYTES`, 62 GiB), NOT the role's `vram_gib` (that is a footprint: the
    served control already measures above it). A GPU server is one with `device: ROCm*` or a role that declares
    `vram_gib`. The loop has no host-RAM budget for CPU servers, so one raises KnobError MISSING_INPUT."""
    from .surface_validation import GPU_CAPACITY_CEILING_BYTES
    reg = sampling_axis._registry(registry)
    entry = (reg.get("server_mode") or {}).get(server)
    if not isinstance(entry, Mapping):
        raise KnobError(f"MISSING_INPUT: registry server_mode.{server} not found")
    role = (reg.get("roles") or {}).get(entry.get("model_role")) or {}
    if str(entry.get("device") or "").lower().startswith(("rocm", "gpu", "hip")) or role.get("vram_gib") is not None:
        return GPU_CAPACITY_CEILING_BYTES / (1 << 30)
    raise KnobError(f"MISSING_INPUT: server {server!r} is not a GPU server and the loop has no host-RAM budget")


@dataclass(frozen=True)
class KnobDelta:
    """An immutable set of knob values for ONE server. Structure is checked at construction;
    `validate(base)` checks it against the registry shape."""
    server: str
    knobs: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.server, str) or not self.server:
            raise KnobError("delta needs a server")
        names = [k for k, _ in self.knobs]
        if not names or names != sorted(set(names)):
            raise KnobError("delta needs at least one knob, sorted and unique")
        for name, value in self.knobs:
            if name not in _FLAGS:
                raise KnobError(f"unknown serving knob {name!r}; closed set: {list(KNOBS)}")
            kind = _FLAGS[name][1]
            if kind == "switch":
                ok = isinstance(value, bool)
            elif kind == "drafter":
                ok = isinstance(value, str) and bool(value)
            elif kind == "enum":
                ok = value in KV_TYPES
            elif kind == "float":
                ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
                    and value >= 0
            else:
                ok = isinstance(value, int) and not isinstance(value, bool)
            if not ok:
                raise KnobError(f"knob {name}={value!r} is not a valid {kind}"
                                + (f" (one of {list(KV_TYPES)})" if kind == "enum" else ""))
            lo, hi = _BOUNDS.get(name, (None, None))
            if lo is not None and not lo <= value <= hi:
                raise KnobError(f"knob {name}={value} outside [{lo}, {hi}]")

    @classmethod
    def of(cls, server: str, knobs: Mapping[str, Any]) -> "KnobDelta":
        return cls(server, tuple(sorted(dict(knobs).items())))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "KnobDelta":
        if not isinstance(row, Mapping) or row.get("schema") != SCHEMA or set(row) != {"schema", "server", "knobs"} \
                or not isinstance(row["knobs"], Mapping):
            raise KnobError("delta must be {schema, server, knobs} at " + SCHEMA)
        return cls.of(row["server"], row["knobs"])

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "server": self.server, "knobs": dict(self.knobs)}

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def validate(self, base: ServingBase) -> "KnobDelta":
        if base.server != self.server:
            raise KnobError(f"delta is for {self.server}, base is {base.server}")
        want = dict(self.knobs)
        eff = {**base.values, **want}
        for name, value in want.items():
            if name in SAMPLING and value not in base.sampling_allowed.get(name, ()):
                raise KnobError(f"{name}={value} is not a value any role on {self.server} declares "
                                f"({list(base.sampling_allowed.get(name, ()))}): the role does not allow it")
            if name == "drafter" and value not in base.drafters:
                raise KnobError(f"drafter {value!r} is not eligible for {self.server}: {sorted(base.drafters)}")
        if all(base.values.get(k) == v for k, v in want.items()):
            raise KnobError(f"{want} equals the registry shape: not a delta")  # a redundant knob inside a real delta is fine
        if "n_ctx" in want or "np" in want:
            if eff.get("n_ctx") and eff.get("np") and eff["n_ctx"] % eff["np"]:
                raise KnobError(f"n_ctx {eff['n_ctx']} is not a multiple of np {eff['np']}")
        if "ubatch" in eff and "batch" in eff and eff["ubatch"] > eff["batch"]:
            raise KnobError(f"ubatch {eff['ubatch']} exceeds batch {eff['batch']}")
        spec = (base.drafters.get(eff.get("drafter")) or {}).get("spec_type")
        if spec == "draft-dspark" and eff.get("np", 1) > 1:
            raise KnobError("draft-dspark is refused by the server at np > 1")
        if "spec_draft_n_max" in want and not eff.get("drafter"):
            raise KnobError("spec_draft_n_max needs a spec-decode drafter in the base shape")
        return self


# --- actuator ----------------------------------------------------------------------------
def _find(argv: Sequence[str], aliases: Sequence[str]) -> int:
    """Index of the first alias AFTER the server executable: `taskset -c` / `numactl` flags are not ours."""
    start = next((i for i, t in enumerate(argv) if Path(t).name == "llama-server"), 0)
    return next((i for i in range(start, len(argv)) if argv[i] in aliases), -1)


def _set(argv: list[str], aliases: Sequence[str], value: Any, *, switch: bool = False) -> None:
    i = _find(argv, aliases)
    if switch:
        flag = aliases[0] if value else aliases[1]
        if i >= 0:
            argv[i] = flag
        else:
            argv.append(flag)
    elif i >= 0 and i + 1 < len(argv):
        argv[i + 1] = str(value)
    else:
        argv += [aliases[0], str(value)]


def _drop(argv: list[str], aliases: Sequence[str]) -> None:
    i = _find(argv, aliases)
    if i >= 0:
        del argv[i:i + 2]


def _apply_drafter(argv: list[str], base: ServingBase, drafter: str, *, n_max: bool) -> None:
    d = base.drafters[drafter]
    target = argv[_find(argv, ("-m",)) + 1] if _find(argv, ("-m",)) >= 0 else base.model
    _set(argv, ("--spec-type",), d["spec_type"])
    if d.get("draft_model") and d["draft_model"] != target:       # a self-drafting model passes no -md
        _set(argv, ("-md",), d["draft_model"])
        if _find(argv, ("-ngld",)) < 0:
            _set(argv, ("-ngld",), d.get("ngld", 99))
    else:
        _drop(argv, ("-md",)); _drop(argv, ("-ngld",))
    if n_max and "draft_max" in d:
        _set(argv, _FLAGS["spec_draft_n_max"][0], d["draft_max"])


def _apply_knobs(argv: list[str], base: ServingBase, knobs: Mapping[str, Any]) -> None:
    if "drafter" in knobs:
        _apply_drafter(argv, base, knobs["drafter"], n_max="spec_draft_n_max" not in knobs)
    for name, value in knobs.items():
        aliases, kind = _FLAGS[name]
        if kind != "drafter":
            _set(argv, aliases, value, switch=kind == "switch")


def read_knobs(argv: Sequence[str]) -> dict[str, Any]:
    """The knob values an argv actually carries (the readback that proves a render)."""
    out: dict[str, Any] = {}
    for name, (aliases, kind) in _FLAGS.items():
        i = _find(argv, aliases)
        if i < 0 or kind == "drafter":
            continue
        if kind == "switch":
            out[name] = argv[i] == aliases[0]
        elif i + 1 < len(argv):
            out[name] = argv[i + 1] if kind == "enum" else (float if kind == "float" else int)(argv[i + 1])
    return out


@dataclass(frozen=True)
class Rendered:
    argv: tuple[str, ...]
    base_argv: tuple[str, ...]          # recipe argv with the registry shape overlaid
    argv_delta: tuple[tuple[str, str | None, str | None], ...]   # (flag, base token, new token)
    delta_digest: str


def _flag_values(argv: Sequence[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for i, tok in enumerate(argv):
        if tok.startswith("-") and len(tok) > 1 and not tok[1:].replace(".", "").isdigit():
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            out[tok] = nxt if nxt is not None and not nxt.startswith("-") else None
    return out


def render(recipe_argv: Sequence[str], base: ServingBase, delta: KnobDelta) -> Rendered:
    """Argv for one arm: `recipe_argv` (from `Recipe.server_argv`), the registry shape overlaid,
    then the delta. Raises KnobError if the delta fails validation or the readback of the result
    disagrees with the delta."""
    delta.validate(base)
    argv = list(recipe_argv)
    reg = {k: v for k, v in base.values.items() if k != "drafter"}
    _apply_knobs(argv, base, reg)
    if base.values.get("drafter") in base.drafters:
        _apply_drafter(argv, base, base.values["drafter"], n_max=False)
    base_argv = tuple(argv)
    _apply_knobs(argv, base, dict(delta.knobs))
    fa = _find(argv, ("-fa", "--flash-attn"))
    final = {**base.values, **dict(delta.knobs)}
    if final.get("ctv") in _QUANTIZED_KV and fa >= 0 and fa + 1 < len(argv) and argv[fa + 1] == "off":
        raise KnobError(f"ctv {final['ctv']} is a quantized V cache and needs flash-attn, but the argv has -fa off")
    wanted = {k: v for k, v in delta.knobs if k != "drafter"}
    got = read_knobs(argv)
    bad = {k: (v, got.get(k)) for k, v in wanted.items() if got.get(k) != v}
    if bad:
        raise KnobError(f"rendered argv does not carry the delta: {bad}")
    before, after = _flag_values(base_argv), _flag_values(argv)
    changed = tuple((f, before.get(f), after.get(f)) for f in sorted({*before, *after})
                    if (f in before, before.get(f)) != (f in after, after.get(f)))
    return Rendered(tuple(argv), base_argv, changed, delta.digest)


# --- seed + planner emission -------------------------------------------------------------
def load_seed(path: Path | str, registry: Any = None) -> dict:
    """Load a `serving-knob` seed. Structure is always checked; with a `registry` every arm is
    validated against the server's shape too (seed.install runs structure-only so installing a
    seed never depends on the registry file)."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    need = {"kind", "schema", "seed_id", "server", "summary", "falsifier", "arms"}
    if not isinstance(body, Mapping) or set(body) != need or body["kind"] != SEED_KIND \
            or body["schema"] != SEED_SCHEMA:
        raise KnobError(f"{path}: a serving-knob seed is exactly {sorted(need)} at {SEED_SCHEMA}")
    for key in ("seed_id", "server", "summary", "falsifier"):
        if not isinstance(body[key], str) or not body[key].strip():
            raise KnobError(f"{path}: {key} must be non-empty text")
    arms = body["arms"]
    if not isinstance(arms, list) or not arms:
        raise KnobError(f"{path}: arms must be a non-empty list")
    ids = [a.get("arm_id") if isinstance(a, Mapping) else None for a in arms]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i and i != CONTROL for i in ids):
        raise KnobError(f"{path}: arm ids must be unique non-empty strings other than {CONTROL!r}")
    base = base_from_registry(body["server"], registry) if registry is not None else None
    for a in arms:
        if set(a) != {"arm_id", "knobs"}:
            raise KnobError(f"{path}: arm {a.get('arm_id')!r} is exactly {{arm_id, knobs}}")
        delta = KnobDelta.of(body["server"], a["knobs"])
        if base is not None:
            delta.validate(base)
    return body


def render_seed_markdown(seed: Mapping[str, Any]) -> str:
    """The planner-visible inbox text for a seed (the inbox reader only reads *.md)."""
    lines = [f"# Serving-knob seed `{seed['seed_id']}` (server `{seed['server']}`)", "", seed["summary"], "",
             f"Falsifier: {seed['falsifier']}", "",
             "Arms (serving-knob deltas over the registry serving_shape; argv is rendered by the actuator, never typed):"]
    lines += [f"- `{a['arm_id']}`: {json.dumps(a['knobs'], sort_keys=True)}" for a in seed["arms"]]
    lines += ["", "Screening only: rank by aggregate tok/s with fit-memory, ITL-during-prefill <= +20% and "
                  "correct-output guards; a winner still needs a paired A/B."]
    return "\n".join(lines) + "\n"


def portfolio_rows(seed: Mapping[str, Any]) -> list[dict]:
    """One typed_planner portfolio row per arm, so the planner picks from a closed list."""
    return [{"mechanism_id": f"{SEED_KIND}:{seed['seed_id']}:{a['arm_id']}", "kind": SEED_KIND,
             "summary": f"{seed['server']} {json.dumps(a['knobs'], sort_keys=True)}: {seed['summary']}"[:300],
             "serving_knob": KnobDelta.of(seed["server"], a["knobs"]).to_dict()} for a in seed["arms"]]


def emit_candidate(choice: Mapping[str, Any], base: ServingBase, recipe_argv: Sequence[str], *,
                   base_recipe_hash: str, champion_sha: str | None = None) -> dict:
    """Turn a typed_planner serving-knob choice into a candidate: the delta, its rendered argv and
    the base recipe identity it rides beside. Pure; nothing is launched. A knob screen has no patch, so `base_sha` and
    `patch_sha` both equal the champion sha the CALLER names (`champion_sha`, 40 hex); omitted, neither is emitted."""
    if choice.get("kind") != SEED_KIND or "serving_knob" not in choice:
        raise KnobError("not a serving-knob choice")
    if not isinstance(base_recipe_hash, str) or len(base_recipe_hash) != 64:
        raise KnobError("base_recipe_hash must be the base Recipe.recipe_hash")
    if champion_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", str(champion_sha)):
        raise KnobError("champion_sha must be the champion's 40-hex commit sha")
    delta = KnobDelta.from_dict(choice["serving_knob"])
    out = render(recipe_argv, base, delta)
    shas = {"base_sha": champion_sha, "patch_sha": champion_sha} if champion_sha else {}
    return {**shas, "schema": CANDIDATE_SCHEMA, "kind": SEED_KIND, "mechanism_id": choice["choice"],
            "server": delta.server, "delta": delta.to_dict(), "delta_digest": out.delta_digest,
            "base_recipe_hash": base_recipe_hash, "registry_base_digest": base.digest(),
            "argv": list(out.argv), "argv_delta": [list(t) for t in out.argv_delta],
            "text": dict(choice.get("text") or {}), "screening_only": True}


__all__ = ["CONTROL", "KNOBS", "KnobDelta", "KnobError", "Rendered", "SEED_KIND", "SEED_SUFFIX", "ScreeningInvalid",
           "ServingBase", "base_from_registry", "emit_candidate", "load_seed", "portfolio_rows", "read_knobs",
           "render", "render_seed_markdown", "vram_capacity_gib"]
