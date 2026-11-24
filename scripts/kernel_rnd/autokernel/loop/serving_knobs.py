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
  * `screen`: ranks arms by aggregate tok/s and rejects an arm that fails a guard rail
    (fit-in-memory, ITL-during-prefill regression <= 20%, correct output). SCREENING ONLY: arms
    differ in np / batch / drafter, so this is not an apples-to-apples gate and never promotes.
  * seed + planner emission: `load_seed`, `portfolio_rows`, `emit_candidate`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from . import sampling_axis

SCHEMA = "epyc.autokernel.serving_knob.v1"
SEED_SCHEMA = "epyc.autokernel.serving_knob_seed.v1"
CANDIDATE_SCHEMA = "epyc.autokernel.serving_knob_candidate.v1"
SCREEN_SCHEMA = "epyc.autokernel.serving_knob_screen.v1"
SEED_KIND = "serving-knob"
SEED_SUFFIX = ".serving-knob.json"
CONTROL = "control"
ITL_REGRESSION_MAX = 0.20
MIN_LAUNCHES = 3


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
    allowed: dict[str, set] = {k: set() for k in SAMPLING}
    for role in [entry.get("model_role"), *(entry.get("shared_with") or ())]:
        defaults = (roles.get(role) or {}).get("generation_defaults") or {}
        for key in SAMPLING:
            if isinstance(defaults.get(key), (int, float)) and not isinstance(defaults[key], bool):
                allowed[key].add(defaults[key])
    return ServingBase(
        server=server, values={k: v for k, v in values.items() if v is not None},
        drafters=drafters, sampling_allowed={k: tuple(sorted(v)) for k, v in allowed.items()},
        model=entry.get("model"))


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


# --- screening evaluator -----------------------------------------------------------------
def _num(rec: Mapping, key: str, label: str, *, positive: bool = False) -> float | None:
    v = rec.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 \
            or (positive and v == 0):
        raise ScreeningInvalid(f"{label}: {key}={v!r} is not a finite non-negative number")
    return float(v)


def _arm_verdict(arm_id: str, launches: Sequence[Mapping], min_launches: int, itl_ref: float | None,
                 itl_max: float) -> dict:
    """Aggregate one arm and apply every guard that does not need the control's ITL (or needs it, when given)."""
    reasons: list[str] = []
    live = [r for r in launches if not r.get("warmup")]
    for n, r in enumerate(launches):
        label = f"{arm_id}[{n}]"
        if r.get("status", "ok") != "ok":
            reasons.append("does-not-fit-memory" if r.get("status") == "oom" else f"launch-failed:{r.get('status')}")
        elif r.get("output_ok") is not True:   # absent is NOT correct
            reasons.append("incorrect-output")
        used, cap = _num(r, "mem_used_gib", label), _num(r, "mem_capacity_gib", label, positive=True)
        if r.get("status", "ok") == "ok" and (used is None or cap is None):
            reasons.append("memory-unmeasured")
        elif used is not None and cap is not None and used > cap:
            reasons.append("does-not-fit-memory")
    ok = [r for r in live if r.get("status", "ok") == "ok"]
    if len(live) < min_launches:
        reasons.append(f"insufficient-launches:{len(live)}<{min_launches}")
    toks = [t for t in (_num(r, "aggregate_tok_s", arm_id, positive=True) for r in ok) if t is not None]
    itls = [t for t in (_num(r, "itl_prefill_ms", arm_id, positive=True) for r in ok) if t is not None]
    if ok and len(toks) != len(ok):
        reasons.append("throughput-unmeasured")
    out = {"arm_id": arm_id, "launches": len(live), "median_tok_s": statistics.median(toks) if toks else None,
           "itl_prefill_ms": statistics.median(itls) if itls else None, "itl_regression_pct": None}
    if itl_ref is not None:
        if not itls:
            reasons.append("itl-unmeasured")      # fail closed: an unmeasurable guard is not a pass
        else:
            out["itl_regression_pct"] = round(100.0 * (out["itl_prefill_ms"] - itl_ref) / itl_ref, 3)
            if out["itl_prefill_ms"] > itl_ref * (1 + itl_max):
                reasons.append(f"itl-regression>{itl_max * 100:g}%")
    out["reasons"] = sorted(set(reasons))
    return out


def screen(arms: Mapping[str, Sequence[Mapping[str, Any]]], *, control: str = CONTROL,
           min_launches: int = MIN_LAUNCHES, itl_regression_max: float = ITL_REGRESSION_MAX) -> dict:
    """Rank arms by median post-warm-up aggregate tok/s; reject any that fails a guard rail.

    `arms` maps arm id -> launch records {status, warmup, aggregate_tok_s, itl_prefill_ms (decode
    inter-token latency while a prefill is in flight), output_ok, mem_used_gib, mem_capacity_gib}.
    The registry-shape arm `control` is required and must itself be sound, because the ITL guard is
    relative to it. Output is SCREENING ONLY (`promotable` is always False)."""
    if control not in arms:
        raise ScreeningInvalid(f"no {control!r} arm: the ITL guard has nothing to be relative to")
    if not 0 <= itl_regression_max < 1 or min_launches < 1:
        raise ScreeningInvalid("guard parameters out of range")
    base = _arm_verdict(control, arms[control], min_launches, None, itl_regression_max)
    if base["reasons"] or base["itl_prefill_ms"] is None or base["median_tok_s"] is None:
        raise ScreeningInvalid(f"control arm is unsound: {base['reasons'] or ['itl/throughput-unmeasured']}")
    kept, rejected = [], []
    for arm_id in sorted(a for a in arms if a != control):
        v = _arm_verdict(arm_id, arms[arm_id], min_launches, base["itl_prefill_ms"], itl_regression_max)
        (rejected if v["reasons"] else kept).append(v)
    for v in kept:
        v["vs_control_pct"] = round(100.0 * (v["median_tok_s"] - base["median_tok_s"]) / base["median_tok_s"], 3)
    ranking = sorted([dict(base, vs_control_pct=0.0, is_control=True), *kept],
                     key=lambda v: (-v["median_tok_s"], v["arm_id"] != control, v["arm_id"]))
    for i, v in enumerate(ranking, 1):
        v["rank"] = i
    return {"schema": SCREEN_SCHEMA, "screening_only": True, "promotable": False, "control": control,
            "metric": "aggregate_tok_s", "guards": {"itl_regression_max": itl_regression_max,
                                                     "min_launches": min_launches,
                                                     "fit_memory": True, "correct_output": True},
            "ranking": ranking, "rejected": rejected,
            "best_arm": None if ranking[0].get("is_control") else ranking[0]["arm_id"],
            "note": "screen, not a gate: arms differ in np/batch/drafter, so confirm a winner with a paired A/B"}


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
                   base_recipe_hash: str) -> dict:
    """Turn a typed_planner serving-knob choice into a candidate: the delta, its rendered argv and
    the base recipe identity it rides beside. Pure; nothing is launched."""
    if choice.get("kind") != SEED_KIND or "serving_knob" not in choice:
        raise KnobError("not a serving-knob choice")
    if not isinstance(base_recipe_hash, str) or len(base_recipe_hash) != 64:
        raise KnobError("base_recipe_hash must be the base Recipe.recipe_hash")
    delta = KnobDelta.from_dict(choice["serving_knob"])
    out = render(recipe_argv, base, delta)
    return {"schema": CANDIDATE_SCHEMA, "kind": SEED_KIND, "mechanism_id": choice["choice"],
            "server": delta.server, "delta": delta.to_dict(), "delta_digest": out.delta_digest,
            "base_recipe_hash": base_recipe_hash, "registry_base_digest": base.digest(),
            "argv": list(out.argv), "argv_delta": [list(t) for t in out.argv_delta],
            "text": dict(choice.get("text") or {}), "screening_only": True}


__all__ = ["CONTROL", "KNOBS", "KnobDelta", "KnobError", "Rendered", "SEED_KIND", "SEED_SUFFIX", "ScreeningInvalid",
           "ServingBase", "base_from_registry", "emit_candidate", "load_seed", "portfolio_rows", "read_knobs",
           "render", "render_seed_markdown", "screen"]
