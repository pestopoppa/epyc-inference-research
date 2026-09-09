#!/usr/bin/env python3
"""Serving-throughput measurement: a keep is real only if it improves SERVING, not a
llama-bench proxy (R23-43, operator directive 2026-09-04: "the only performance that
matters is serving performance").

WHY THIS EXISTS. llama-bench measures a fixed-workload forward pass. It is deterministic
and cheap -- the right tool for the planner to SCREEN hypotheses -- but it is a PROXY, and
2026-09-04 proved the proxy diverges: two keeps worth +23.3% / +10.1% on the dec-b4 bench
surface moved DFlash2 serving decode by ~0% (71.22 t/s, flat). So the KEEP GATE and the
HEADLINE move to `llama-server` under the champion's CANONICAL RECIPE, which is also the
recipe production needs at promotion -- built once, used for both.

THE RECIPE IS GENERAL. `spec_decode.type` is one of {none, draft-dflash, draft-mtp, ...}; a
model that does not use speculative decode carries `none` and its own optimal `np`. Nothing
about DFlash2 is baked into the framework; today's champion just happens to serve the 27B on
gfx90a with DFlash2 at np4 (the aggregate-throughput knee measured by DF2-5).

THE METRIC. `aggregate_tok_s` is the sum of each concurrent slot's reported
`predicted_per_second`. It is not common-window wall throughput; changing to that estimator would
require a new metric identity and calibration.

This module is backend-blind about the kernel: it takes two BUILD DIRECTORIES (champion vs
candidate) and the recipe, and returns a paired A/B `Comparison`-shaped result the loop's
keep gate already knows how to read.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import concurrent.futures as cf
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import TYPE_CHECKING
import urllib.request
import urllib.error

from . import lifecycle_observation, residency, status
from . import native_server_response as server_response

if TYPE_CHECKING:
    from .resolved_recipe import ResolvedRecipe

#: Schema of the on-disk recipe file, and part of the recipe's identity: a schema bump
#: changes what the fields MEAN, so it must change the hash too.
RECIPE_SCHEMA = "epyc.autokernel.canonical_recipe.v1"

#: Variables the LOADER owns. A recipe may neither set nor unset these, and the refusal is
#: a hard error rather than a silent override: `LD_LIBRARY_PATH` is what pins the build's
#: OWN ggml, and three ggml generations live on this host -- a binary that inherits another
#: tree's ggml runs silently wrong and no exit code reports it. `HSA_OVERRIDE_GFX_VERSION` is
#: deliberately UNSET by the loader env for the same reason. An arm that needs either of
#: these is not an env arm; it is a different build or a different device.
LOADER_OWNED_ENV = ("LD_LIBRARY_PATH", "HSA_OVERRIDE_GFX_VERSION")

#: `env_readback` key standing for "the recipe does not set this variable at all".
UNSET = "unset"

#: Schema of the per-launch residency block. Additive: every consumer reads it via
#: `.get()`, and a record written before R23-60 simply has no block.
RESIDENCY_SCHEMA = "epyc.autokernel.serving_residency.v1"
#: Sampled DURING the launch, over a window that encloses the request phase, and the
#: device held at least `residency.RESIDENT_FLOOR_BYTES`.
RESIDENCY_PROVEN = "proven"
#: The instrument could not be read, or the window did not cover the request phase.
#: NOT a claim that the launch ran on the CPU -- a claim that nobody knows. It is a
#: recorded fact rather than a refusal because an unreadable sysfs node is an
#: INSTRUMENT failure, and refusing every launch on one would stop the campaign on a
#: fault that says nothing about the measurement.
RESIDENCY_UNPROVEN = "unproven"
#: CPU launches neither require nor earn a GPU residency warrant.
RESIDENCY_NOT_APPLICABLE = "not_applicable"

#: Characters a recipe name may carry VERBATIM into `serving-floor.<name>.json`: safe in a
#: filename and as an unquoted shell word -- no `/`, no whitespace, no glob metacharacter,
#: no leading dot. `+` and `=` are in the set deliberately, so a `with_env` arm name
#: (`base+GGML_NOHUGEPAGE_PROCESS=1`) stays readable on disk instead of being hashed away.
_FLOOR_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+=@-")

#: Cap on the floor-file key, leaving room for the `serving-floor.` prefix and the `.json`
#: suffix inside the 255-byte filename limit. A name derived by `with_env` twice grows
#: without bound otherwise, and the failure would land as ENAMETOOLONG mid-campaign.
FLOOR_KEY_MAX = 120

#: Prompts fired at the server. Distinct so the slots do not share a KV prefix (a shared
#: prefix would understate the real per-request work); enough of them to cover np up to 8.
_PROMPTS = (
    "Prove by induction that the sum of the first n odd numbers is n squared, then compute n=20.",
    "Explain how a red-black tree keeps its height logarithmic, then insert 7,3,18,10,22,8,11.",
    "Derive the closed form of the Fibonacci sequence via generating functions, step by step.",
    "A train leaves A at 60 km/h and another leaves B at 90 km/h 200 km apart; when do they meet? Show work.",
    "Prove that there are infinitely many primes, then list the first ten primes above 1000.",
    "Explain the CAP theorem and give a concrete example system for each of CP, AP, and CA.",
    "Compute the eigenvalues of [[2,1],[1,2]] and explain what they mean geometrically.",
    "Describe Dijkstra's algorithm and trace it on a 6-node weighted graph you define.",
)


@dataclass(frozen=True)
class Recipe:
    """A champion's canonical serving configuration. General over spec-decode type."""
    name: str
    model: str
    device: str = "ROCm0"
    ngl: int = 99
    #: {"type": "none"} | {"type": "draft-dflash"|"draft-mtp", "drafter": path, "ngld": int,
    #: "draft_n_max": int}
    spec_decode: dict = field(default_factory=lambda: {"type": "none"})
    np: int = 4
    ctx: int = 16384
    threads: int = 8
    batch: int = 2048
    ubatch: int = 2048
    ctk: str = "f16"
    ctv: str = "f16"
    fa: str = "on"
    kv_unified: bool = False
    extra_flags: tuple = ()
    #: CPU affinity for the server's host threads, e.g. "184-191". None = UNPINNED, which is
    #: what every floor calibrated to date measured, so it stays the default: setting it
    #: changes the measured condition and INVALIDATES the recipe's serving floor until that
    #: floor is re-calibrated (R23-49). Unpinned host threads are free to land on 0-95, the
    #: CPU campaign's bench region -- kernel-verified 2026-09-07, every logical CPU on this
    #: host shares a physical core with 0-95, so this pollutes their arms and our own.
    cpu_list: str | None = None
    #: workload
    n_predict: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    metric: str = "aggregate_tok_s"
    #: Extra environment variables set on the server process at LAUNCH, on top of the
    #: loader env. This is what makes a launch-time knob expressible as an ARM at all:
    #: `GGML_NOHUGEPAGE_PROCESS=1` (a `prctl(PR_SET_THP_DISABLE)` shim, distinct from the
    #: already-on madvise `GGML_NOHUGEPAGE`) cannot be reached through argv, and on the CPU
    #: surface it cut between-launch variance 25.3x (sd 2.510% -> 0.481%). Default None
    #: keeps every shipped recipe launching EXACTLY as it does today: this makes the arm
    #: expressible, it does not adopt it. Setting it CHANGES THE MEASURED CONDITION and so
    #: invalidates the recipe's serving floor until that floor is re-calibrated, exactly as
    #: `cpu_list` does (R23-49).
    env: dict | None = None
    #: Declarative post-launch readback: prove the env arm TOOK EFFECT. Each entry is
    #: {"field": <'/proc/<pid>/status' field>, "expect": <value>} for an unconditional
    #: check, or {"field": ..., "env": <variable>, "expect": {<env value>: <field value>,
    #: "unset": <field value>}} for one that asserts in BOTH directions from a single
    #: declaration -- e.g.
    #:     {"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
    #:      "expect": {"1": "0", "unset": "1"}}
    #: reads THP_enabled=0 on the treatment arm and THP_enabled=1 on the control. An env
    #: state the declaration does not cover is a REFUSAL at construction, not a skipped
    #: check: a control arm whose readback was never declared is a control that was never
    #: checked, and a control that is secretly the treatment cannot be detected afterwards.
    env_readback: tuple = ()
    #: Environment variables explicitly REMOVED from the launched process after the
    #: inherited and loader environments are assembled.  This is deliberately separate
    #: from `env`: absence from `env` means "inherit", while presence here means "unset".
    #: A tuple keeps the declaration immutable; construction canonicalises its order so
    #: equivalent recipes have one serialized identity. Kept last to preserve the legacy
    #: positional constructor shape for every pre-existing field.
    explicit_unsets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key, value in (self.env or {}).items():
            self._validate_env_key(key, "env")
            if not isinstance(value, str):
                raise RecipeError(
                    f"env {key!r}={value!r}: environment variables are strings. Quote the "
                    f"value in the recipe JSON -- coercing it here would let the record "
                    f"claim a value the process never saw.")
            if key in LOADER_OWNED_ENV:
                raise RecipeError(
                    f"recipe env may not set {key!r}: it is owned by the loader env, and "
                    f"overriding it would break the three-ggml-generations linkage "
                    f"guarantee the residency check depends on. An arm that needs a "
                    f"different {key} is a different BUILD, not an env arm.")
        if isinstance(self.explicit_unsets, str):
            raise RecipeError("explicit_unsets must be a sequence of environment keys, not a string")
        try:
            unsets = tuple(self.explicit_unsets)
        except TypeError as exc:
            raise RecipeError("explicit_unsets must be a sequence of environment keys") from exc
        for key in unsets:
            self._validate_env_key(key, "explicit_unsets")
            if key in LOADER_OWNED_ENV:
                raise RecipeError(
                    f"recipe env may not unset {key!r}: it is owned by the loader env, and "
                    f"removing it would break the loader's linkage/device guarantee. An arm "
                    f"that needs a different {key} is not an env arm.")
        if len(set(unsets)) != len(unsets):
            duplicates = sorted(key for key in set(unsets) if unsets.count(key) > 1)
            raise RecipeError(f"explicit_unsets contains duplicate keys: {duplicates}")
        unsets = tuple(sorted(unsets))
        conflict = sorted(set(self.env or {}).intersection(unsets))
        if conflict:
            raise RecipeError(
                f"recipe env keys cannot be both set and explicitly unset: {conflict}")
        object.__setattr__(self, "explicit_unsets", unsets)
        # Resolve the readback declaration NOW, so an uncovered env state fails at
        # construction (including through `with_env`) rather than mid-measurement.
        self.readback_expectations()

    @staticmethod
    def _validate_env_key(key: object, source: str) -> None:
        """Reject keys that cannot be passed faithfully through an OS environment."""
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise RecipeError(
                f"{source} key {key!r}: environment variable names must be non-empty "
                f"strings containing neither '=' nor NUL")

    @classmethod
    def load(cls, path: Path | str) -> "Recipe":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: Mapping) -> "Recipe":
        d = dict(d)
        d.pop("schema", None)
        d["extra_flags"] = tuple(d.get("extra_flags", ()))
        if d.get("env") is not None:
            d["env"] = dict(d["env"])
        d["explicit_unsets"] = d.get("explicit_unsets") or ()
        d["env_readback"] = tuple(dict(c) for c in (d.get("env_readback") or ()))
        return cls(**d)

    def to_dict(self) -> dict:
        """Every field, in the shape `from_dict` reads back. There are no cosmetic fields:
        each one either changes what is launched or changes what the number means."""
        out = {"schema": RECIPE_SCHEMA, "name": self.name, "model": self.model,
               "device": self.device, "ngl": self.ngl,
               "spec_decode": dict(self.spec_decode), "np": self.np, "ctx": self.ctx,
               "threads": self.threads, "batch": self.batch, "ubatch": self.ubatch,
               "ctk": self.ctk, "ctv": self.ctv, "fa": self.fa,
               "kv_unified": self.kv_unified, "extra_flags": list(self.extra_flags),
               "cpu_list": self.cpu_list, "n_predict": self.n_predict,
               "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
               "metric": self.metric, "env": dict(self.env or {}),
               "env_readback": [dict(c) for c in self.env_readback]}
        # Additive only when the new semantic is used: legacy recipe serialization and
        # hashes remain byte-for-byte identical, so old floors are not reinterpreted.
        if self.explicit_unsets:
            out["explicit_unsets"] = list(self.explicit_unsets)
        return out

    @property
    def recipe_hash(self) -> str:
        """Content-addressed identity of the SERVING recipe, emitted with every serving row.

        A champion identified only by a commit hash is under-specified once a launch recipe
        is part of the artifact (R23-59): the same commit launched at a different `np`, a
        different pin, or -- now that `env` exists -- with a different launch-time knob is a
        different measurement wearing the same name. This digest is what lets a gate REFUSE
        a measurement whose recipe does not match the one the floor was calibrated under,
        instead of silently comparing two conditions.

        It covers `to_dict()`, i.e. every field including `env`, `explicit_unsets`, and
        `env_readback`. `None` and `{}` env normalise to the same digest, because "no extra
        env" is one condition. The explicit-unset field is omitted when empty to preserve
        every legacy digest.
        """
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()

    def readback_expectations(self) -> tuple[tuple[str, str], ...]:
        """`(status field, expected value)` pairs this recipe's env must produce at launch."""
        env = self.env or {}
        out: list[tuple[str, str]] = []
        for check in self.env_readback:
            try:
                field_name, expect = check["field"], check["expect"]
            except (TypeError, KeyError) as exc:
                raise RecipeError(
                    f"env_readback entry {check!r} needs 'field' and 'expect'") from exc
            if isinstance(expect, str):
                out.append((field_name, expect))
                continue
            var = check.get("env")
            if not var:
                raise RecipeError(
                    f"env_readback for {field_name!r} maps expectations by env value but "
                    f"names no 'env' variable to read them from")
            state = env.get(var, UNSET)
            if state not in expect:
                raise RecipeError(
                    f"env_readback for {field_name!r} does not cover {var}={state!r} "
                    f"(declared: {sorted(expect)}). Declare BOTH directions or drop the "
                    f"check -- an undeclared state is an UNCHECKED arm, and a control that "
                    f"is secretly the treatment is unrecoverable after the fact.")
            out.append((field_name, str(expect[state])))
        return tuple(out)

    def server_env(self, build_dir: Path, *,
                   base: Mapping[str, str] | None = None) -> dict[str, str]:
        """The exact environment the server is launched with: inherited env, then the
        loader pin, then the recipe's own `env`, then its explicit unsets LAST.

        Precedence is explicit unset > recipe set > loader > inherited, and it is total only
        because the loader's own variables are refused to both recipe states
        (`LOADER_OWNED_ENV`) -- so the recipe can override any host variable it likes
        without ever being able to silently drop the linkage pin the residency check
        depends on.
        """
        env = dict(os.environ if base is None else base)
        env.pop("HSA_OVERRIDE_GFX_VERSION", None)
        env["LD_LIBRARY_PATH"] = str(Path(build_dir) / "bin")
        env.update(self.env or {})
        for key in self.explicit_unsets:
            env.pop(key, None)
        return env

    def with_env(self, *, name: str | None = None, **overrides: str | None) -> "Recipe":
        """A sibling recipe differing only in `env` -- the two arms of a paired env A/B from
        ONE recipe file, with no JSON editing. A `None` value explicitly UNSETS a variable
        after inherited env is assembled, so the control arm of an ON recipe is
        `with_env(KNOB=None)`. Setting a value replaces an unset marker, and unsetting a
        value removes any recipe override; the source recipe is never mutated.

        The name gets the override appended by default, because the serving floor FILE is
        keyed by recipe name (`serving-floor.<name>.json`) and a different env is a
        different measured condition: reusing the name would point the new arm at the old
        arm's floor file. Pass `name=` to override deliberately.

        The name is no longer the only guard, and was never a sufficient one: `load_floor`
        refuses any floor whose stamped `recipe_hash` is not this recipe's, so even a
        deliberate `name=` collision is caught at the point of use instead of being judged
        against the wrong bar. The filename is `floor_key(name)`, not the raw name -- an
        env VALUE reaches the name here and may carry `/` or whitespace.
        """
        merged = dict(self.env or {})
        unsets = set(self.explicit_unsets)
        for key, value in overrides.items():
            if value is None:
                merged.pop(key, None)
                unsets.add(key)
            else:
                merged[key] = value
                unsets.discard(key)
        if name is None:
            name = self.name + "".join(
                f"+{k}={UNSET if v is None else v}" for k, v in sorted(overrides.items()))
        return replace(self, name=name, env=merged, explicit_unsets=tuple(sorted(unsets)))

    def server_argv(self, build_dir: Path, port: int) -> list[str]:
        argv = ["taskset", "-c", self.cpu_list] if self.cpu_list else []
        argv += [str(Path(build_dir) / "bin" / "llama-server"),
                "-m", self.model, "-np", str(self.np), "-c", str(self.ctx),
                "-t", str(self.threads), "-tb", str(self.threads),
                "-b", str(self.batch), "-ub", str(self.ubatch),
                "-ctk", self.ctk, "-ctv", self.ctv, "--device", self.device,
                "-ngl", str(self.ngl), "-fa", self.fa,
                "--host", "127.0.0.1", "--port", str(port), "--metrics", "--slots"]
        sd = self.spec_decode
        if sd.get("type", "none") != "none":
            # A SELF-DRAFTING model carries its draft head inside the weights (MTP, and any
            # future variant of it), so there is no second GGUF and `-md` must NOT be passed.
            # Requiring `drafter` unconditionally made every self-drafting model inexpressible
            # as a recipe -- found 2026-09-08 trying to sweep Qwen3.6-35B-A3B-MTP. `drafter`
            # is therefore OPTIONAL, and its absence is the declaration that the model drafts
            # for itself; `ngld` is likewise only meaningful with a separate drafter.
            if sd.get("drafter"):
                argv += ["-md", sd["drafter"], "-ngld", str(sd.get("ngld", self.ngl))]
            argv += ["--spec-type", sd["type"]]
            if "draft_n_max" in sd:
                argv += ["--spec-draft-n-max", str(sd["draft_n_max"])]
        argv += ["--kv-unified" if self.kv_unified else "--no-kv-unified"]
        argv += list(self.extra_flags)
        return argv

    def describe(self) -> str:
        """One line naming every condition the number is conditional on.

        `env` is rendered even when empty (`env=none`), like `cpu=unpinned`: a record must
        never be able to claim an arm it did not run, and the reader of a serving row can
        only tell the treatment from the control by what the recipe says it launched.
        """
        sd = self.spec_decode.get("type", "none")
        pin = f" cpu={self.cpu_list}" if self.cpu_list else " cpu=unpinned"
        env = self.env or {}
        env_states = ([f"{k}={v}" for k, v in sorted(env.items())]
                      + [f"{k}=<unset>" for k in self.explicit_unsets])
        envs = " env=" + ",".join(env_states) if env_states else " env=none"
        rb = self.readback_expectations()
        rbs = " readback=" + ",".join(f"{f}={v}" for f, v in rb) if rb else ""
        return (f"{self.name} [np{self.np} {sd} {self.metric}{pin}{envs}{rbs} "
                f"#{self.recipe_hash[:12]}]")


class RecipeError(ValueError):
    """A recipe declares something that cannot be measured honestly."""


class EnvReadbackFailed(RuntimeError):
    """The launched server does not show the state its env claims to have set.

    "I set the knob" is not evidence the knob took effect. This is raised in BOTH
    directions -- a CONTROL arm that is secretly the treatment is unrecoverable after
    the fact, so verifying only that ON is on is not verification.
    """


class ServerDied(RuntimeError):
    """The server exited during load or measurement -- a build/config fault, not noise."""


class ServingNotResident(RuntimeError):
    """The launch was SAMPLED, over a window covering the request phase, and the device
    never held the model. The measurement did not happen on the GPU.

    Deliberately NOT the same failure as an unreadable instrument. This is a positive
    observation of absence over a window that DID overlap the phenomenon, so it is a
    refusal for the same reason `EnvReadbackFailed` is one: a CPU-resident arm silently
    filed as a GPU number is unrecoverable after the fact -- no later reader can tell,
    and no re-analysis can rescue it. Distinct class and distinct message, because
    "the knob did not take effect" and "this did not run on the device" are different
    defects with different fixes.
    """


class ServingFloorMismatch(RuntimeError):
    """The floor on disk was calibrated under a DIFFERENT recipe than the one running.

    A floor is a property of the measured CONDITION, not of the recipe's NAME. The store
    keys it by name (`serving-floor.<name>.json`) and nothing used to check that the file
    was produced by the recipe now being gated -- so any recipe edit silently reused the
    old floor and the gate judged one condition against another's bar. R23-49 is the
    standing proof: pinning `cpu_list` to 184-191 voided the unpinned floor, and only a
    human noticing forced the recalibration. An `env` arm makes this sharper still: its
    entire purpose is to change DISPERSION, which is to change the floor itself.
    """


def _status_fields(text: str) -> dict[str, str]:
    """`/proc/<pid>/status` as a mapping. `THP_enabled:\t1` -> {"THP_enabled": "1"}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def verify_env_readback(recipe: Recipe, pid: int, *,
                        status_text: str | None = None,
                        expectations: Sequence[tuple[str, str]] | None = None
                        ) -> dict[str, str | None]:
    """Prove the recipe's env arm took effect on the LAUNCHED process, or refuse.

    Fail-closed in both directions and in every failure mode: a mismatch raises, a field the
    kernel does not expose raises, and an unreadable `/proc` raises. It aborts rather than
    warns because the failure this guards is a mislabelled CONTROL arm -- if the control
    silently ran as the treatment, the A/B measured nothing and nothing downstream can tell.

    `pid` is `Popen.pid`, which is the SERVER even under `taskset`: taskset `exec`s the
    binary in place rather than forking, and `PR_SET_THP_DISABLE` survives `exec`.
    """
    expectations = (recipe.readback_expectations() if expectations is None
                    else tuple(expectations))
    if not expectations:
        return {}
    if status_text is None:
        try:
            status_text = Path(f"/proc/{pid}/status").read_text()
        except OSError as exc:
            raise EnvReadbackFailed(
                f"{recipe.describe()}: cannot read /proc/{pid}/status ({exc}), so the env "
                f"arm is unverified -- refusing the measurement.") from exc
    fields = _status_fields(status_text)
    observed: dict[str, str | None] = {}
    for field_name, expect in expectations:
        got = fields.get(field_name)
        observed[field_name] = got
        if got != expect:
            raise EnvReadbackFailed(
                f"{recipe.describe()}: /proc/{pid}/status {field_name}={got!r}, but this "
                f"recipe's env requires {expect!r}. Setting the variable is not evidence it "
                f"took effect -- refusing the measurement.")
    return observed



# ---------------------------------------------------------------------------
# GPU residency for the SERVING path (R23-60).
#
# The bench path has proven residency per invocation since the rebuild; this path did
# not, so every serving number taken before 2026-09-08 -- the 4.581% floor and the gate
# reading included -- is un-PROVEN as GPU-resident. That is a missing proof, not a
# suspected defect, and it cannot be retro-fitted: a residency tuple invented on read
# claims warrant the original run never captured.
#
# One mechanism, shared with bench: `residency.Sampler`, `residency.RESIDENT_FLOOR_BYTES`.
# No second threshold is defined here.
# ---------------------------------------------------------------------------

def covers_request_phase(record: Mapping) -> bool:
    """Did the sampling window actually ENCLOSE the phase it claims to be evidence about?

    Containment, which is strictly stronger than overlap -- the launch controls both
    clocks, so anything less is a bug rather than a limitation. This is the predicate a
    READER applies to the recorded timestamps: a measurement whose window does not
    overlap the phenomenon is not evidence of its absence, which is exactly why a
    post-hoc 0% VRAM reading is the NORMAL result on a finished llama-bench and proves
    nothing. Missing timestamps are False: an unknown window is not a covering one.
    """
    window_start, window_end = record.get("window_start"), record.get("window_end")
    request_start, request_end = record.get("request_start"), record.get("request_end")
    if None in (window_start, window_end, request_start, request_end):
        return False
    return window_start <= request_start and request_end <= window_end


def _residency_record(sampler, *, window_start: float, window_end: float,
                      request_start: float | None,
                      request_end: float | None, backend: str = "gpu") -> dict:
    """One launch's residency evidence: what bench records, plus the window it covers.

    `status` is three-valued in effect: `proven`, `unproven`, or -- when the window DID
    cover the request phase and the device was empty throughout -- a refusal raised by
    `_refuse_if_not_resident`, which never reaches a record a reader could mistake for a
    result.
    """
    proof = dict(sampler.proof)
    sampled = bool(proof.get("samples")) and bool(proof.get("vram_reads"))
    record = {"schema": RESIDENCY_SCHEMA, "backend": backend, **proof,
              # Distinguishes an unreadable instrument from a device read as empty.
              "sampled": sampled,
              "resident_floor_bytes": residency.RESIDENT_FLOOR_BYTES,
              "window_start": window_start, "window_end": window_end,
              "window_s": round(window_end - window_start, 3),
              "request_start": request_start, "request_end": request_end}
    record["covers_request_phase"] = covers_request_phase(record)
    record["status"] = (RESIDENCY_NOT_APPLICABLE if backend == "cpu" else
                        RESIDENCY_PROVEN
                        if sampled and record["covers_request_phase"] and proof.get("resident")
                        else RESIDENCY_UNPROVEN)
    record["gpu_residency"] = record["status"]
    if backend == "cpu":
        # These need a CPU lifecycle sampler; absence is never a clean observation.
        record["cpu_placement"] = "unproven"
        record["contention"] = "unproven"
    return record


def _refuse_if_not_resident(recipe: Recipe, record: Mapping, *, backend: str = "gpu") -> None:
    """Abort a launch MEASURED non-resident. Record, but do not abort, an unsampled one.

    The asymmetry is the same one `verify_env_readback` enforces, and it is not a
    preference: a launch whose window covered the request phase and read an empty device
    is positive evidence that the number is not a GPU number, and a CPU-resident arm
    filed as GPU is unrecoverable. A launch nobody could sample is an instrument fault --
    it carries `unproven` and travels with the number, so a reader can refuse it later
    with the whole record in hand.
    """
    if record.get("status") == RESIDENCY_PROVEN:
        return
    if (record.get("status") == RESIDENCY_NOT_APPLICABLE and backend == "cpu"
            and recipe.ngl == 0 and recipe.device == "none"):
        return
    if not (record.get("sampled") and record.get("covers_request_phase")):
        return
    raise ServingNotResident(
        f"{recipe.describe()}: GPU RESIDENCY REFUTED -- peak VRAM "
        f"{record.get('peak_vram_bytes')} B (median {record.get('median_vram_bytes')} B) "
        f"over {record.get('samples')} samples spanning the request phase "
        f"[{record.get('request_start')}, {record.get('request_end')}], below the "
        f"{residency.RESIDENT_FLOOR_BYTES} B resident floor. This launch did not execute "
        f"on the device -- refusing to return a serving number that would be filed as a "
        f"GPU result.")


def _residency_fold(records: Sequence[Mapping]) -> dict:
    """Aggregate per-launch evidence for a row, in the shape `bench.compare` folds it.

    `status` is `proven` only when EVERY launch in the row is proven: a row is a claim
    about all of its launches, and one unproven launch makes the row's provenance
    unproven no matter how the rest sampled.
    """
    rows = [dict(record) for record in records]
    cpu_only = bool(rows) and all(row.get("backend") == "cpu" for row in rows)
    fold = {"schema": RESIDENCY_SCHEMA,
            "backend": "cpu" if cpu_only else "gpu",
            "status": RESIDENCY_NOT_APPLICABLE if cpu_only else (
                RESIDENCY_UNPROVEN if not rows else (
                RESIDENCY_PROVEN
                if all(r.get("status") == RESIDENCY_PROVEN for r in rows)
                else RESIDENCY_UNPROVEN)),
            "invocations": len(rows),
            "resident": sum(1 for r in rows if r.get("resident")),
            "proven": sum(1 for r in rows if r.get("status") == RESIDENCY_PROVEN),
            "resident_floor_bytes": residency.RESIDENT_FLOOR_BYTES}
    fold["gpu_residency"] = fold["status"]
    if cpu_only:
        fold["cpu_placement"] = "unproven"
        fold["contention"] = "unproven"
    if not rows:
        return fold
    fold.update({
        "peak_vram_bytes": max((r.get("peak_vram_bytes") or 0) for r in rows),
        "median_vram_bytes": int(statistics.median(
            [(r.get("median_vram_bytes") or 0) for r in rows])),
        "peak_kfd_processes": max((r.get("peak_kfd_processes") or 0) for r in rows),
        "sclk_min_mhz": min((r.get("sclk_min_mhz") or 0) for r in rows),
        "sclk_max_mhz": max((r.get("sclk_max_mhz") or 0) for r in rows),
        "clock_stable": all(r.get("clock_stable") for r in rows),
        "samples": sum((r.get("samples") or 0) for r in rows),
        "covers_request_phase": all(r.get("covers_request_phase") for r in rows),
        "window_start": min((r.get("window_start") or 0.0) for r in rows),
        "window_end": max((r.get("window_end") or 0.0) for r in rows)})
    return fold


def _measure_once(recipe: Recipe, build_dir: Path, port: int,
                  boot_timeout_s: int = 360, *,
                  evidence: list | None = None,
                  resolved_recipe: "ResolvedRecipe | None" = None,
                  frozen_requests: Sequence[tuple[str, bytes]] | None = None,
                  observation: list | None = None,
                  observation_session: lifecycle_observation.ObservationSession | None = None,
                  response_capture: server_response.ServerResponseCapture | None = None
                  ) -> float:
    """Launch the server under `recipe`, fire `np` concurrent requests, return aggregate
    tok/s. The server is always stopped, even on error.

    GPU residency is sampled ACROSS THE WHOLE LAUNCH -- the sampler starts before
    `Popen` and stops after teardown, so the window encloses both the model load and the
    request phase, and the record carries both pairs of timestamps so a reader can
    confirm the overlap instead of taking it on trust. "I invoked the HIP build" is not
    evidence of a HIP run and `ldd` cannot supply one: llama.cpp dlopens
    `libggml-hip.so`, so the executable shows no HIP linkage either way.

    `evidence`, when given, receives this launch's residency record -- appended in a
    `finally`, so a failed launch still leaves its window on the record rather than
    vanishing. Passing a sink is OPTIONAL and the residency REFUSAL is not: a launch
    measured non-resident raises `ServingNotResident` whether or not anyone asked for
    the record.
    """
    request_rows: list[dict] = []
    process_pid: int | None = None
    teardown = "not_started"
    failure: str | None = None
    observer_finish_ok = observation_session is None
    response_reference = None
    if response_capture is not None and type(response_capture) is not server_response.ServerResponseCapture:
        raise RecipeError("response capture must be the concrete native server recorder")
    if response_capture is not None and (resolved_recipe is None or frozen_requests is None):
        raise RecipeError("server response capture requires the frozen resolved launch")
    if response_capture is not None:
        response_capture.validate_launch(resolved_recipe, frozen_requests)

    def observe(method: str, *args) -> bool:
        if observation_session is None:
            return True
        try:
            getattr(observation_session, method)(*args)
            return True
        except Exception as exc:
            # The session retains its own diagnostic whenever it can. Observation is
            # auxiliary: it must never skip owned teardown or replace a serving error.
            try:
                observation_session.note_hook_failure(method, exc)
            except Exception:
                pass
            return False
    if frozen_requests is not None:
        if len(frozen_requests) != recipe.np:
            raise RecipeError("frozen request count must equal recipe.np")
        for prompt_id, body in frozen_requests:
            if not isinstance(prompt_id, str) or not prompt_id.strip() \
                    or not isinstance(body, bytes) or not body:
                raise RecipeError("frozen requests need non-empty prompt IDs and body bytes")
        if len({item[0] for item in frozen_requests}) != len(frozen_requests):
            raise RecipeError("frozen request prompt IDs must be unique within a launch")
    backend = "gpu"
    if resolved_recipe is not None:
        # Refuse unsupported capabilities or moved inputs before sampler/Popen.
        resolved_recipe.validate_launch(recipe, build_dir, port)
        argv = list(resolved_recipe.argv)
        launch_env = dict(resolved_recipe.launch_env)
        backend = resolved_recipe.backend
    else:
        argv = recipe.server_argv(build_dir, port)
        launch_env = recipe.server_env(build_dir)
    observe("start", "setup")
    sampler = None
    window_start = time.time()
    request_start: float | None = None
    request_end: float | None = None
    try:
        sampler = residency.Sampler()
        with sampler:
            observe("phase", "load")
            srv = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   env=launch_env)
            process_pid = srv.pid
            observe("attach_target", srv.pid)
            # Placement is an overlapping launcher marker inside the load window;
            # load itself remains open until the health marker below.
            observe("phase", "placement")
            try:
                for _ in range(boot_timeout_s // 2):
                    if srv.poll() is not None:
                        raise ServerDied(f"server exited {srv.returncode} during load ({recipe.describe()})")
                    try:
                        health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                        if callable(getattr(health, "close", None)):
                            health.close()
                        break
                    except urllib.error.HTTPError as exc:
                        exc.close()
                        time.sleep(2)
                    except Exception:
                        time.sleep(2)
                else:
                    raise ServerDied("server not healthy within boot timeout")
                observe("phase", "health")
                # The env arm is verified on the LIVE process, before a single token is measured.
                if resolved_recipe is None:
                    verify_env_readback(recipe, srv.pid)
                else:
                    verify_env_readback(
                        recipe, srv.pid,
                        expectations=resolved_recipe.readback_expectations)

                def one(i: int, phase: str) -> tuple:
                    if frozen_requests is None:
                        prompt_id = f"legacy-slot-{i}"
                        body = json.dumps({"prompt": _PROMPTS[i % len(_PROMPTS)],
                                           "n_predict": recipe.n_predict,
                                           "temperature": recipe.temperature,
                                           "top_p": recipe.top_p, "top_k": recipe.top_k,
                                           "cache_prompt": False}).encode()
                    else:
                        prompt_id, body = frozen_requests[i]
                    record = {"phase": phase, "slot_index": i, "prompt_id": prompt_id,
                              "request_sha256": hashlib.sha256(body).hexdigest(),
                              "predicted_n": None, "predicted_per_second": None,
                              "terminal": False, "error": None}
                    response_bytes = None
                    started_monotonic = time.monotonic()
                    ended_monotonic = started_monotonic
                    result = (0, 0.0, False)
                    try:
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{port}/completion", data=body,
                            headers={"Content-Type": "application/json"})
                        http_error = None
                        try:
                            opened = urllib.request.urlopen(req, timeout=600)
                        except urllib.error.HTTPError as exc:
                            if response_capture is None:
                                exc.close()
                                raise
                            opened, http_error = exc, exc
                        try:
                            if response_capture is None:
                                response_bytes = opened.read()
                            else:
                                response_bytes = opened.read(server_response.MAX_RESPONSE_BYTES + 1)
                                if len(response_bytes) > server_response.MAX_RESPONSE_BYTES:
                                    response_bytes = None
                                    raise ValueError("server response exceeds raw capture byte budget")
                        finally:
                            if callable(getattr(opened, "close", None)):
                                opened.close()
                        ended_monotonic = time.monotonic()
                        if http_error is not None:
                            raise http_error
                        response = json.loads(response_bytes)
                        timings = response.get("timings", {})
                        # per-request decode rate, NOT wall-clock: each slot reports its own
                        # predicted_n / predicted_ms; the aggregate remains the sum of rates.
                        if frozen_requests is not None:
                            if not isinstance(timings, Mapping):
                                raise ValueError("response timings must be an object")
                            tokens = timings.get("predicted_n")
                            rate = timings.get("predicted_per_second")
                            if isinstance(tokens, bool) or not isinstance(tokens, int) \
                                    or tokens < 0:
                                raise ValueError("predicted_n must be a non-negative integer")
                            if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                                    or not math.isfinite(float(rate)) or rate < 0:
                                raise ValueError(
                                    "predicted_per_second must be a finite non-negative number")
                            rate = float(rate)
                        else:
                            tokens = int(timings.get("predicted_n", 0))
                            rate = float(timings.get("predicted_per_second", 0.0))
                        terminal = response.get("stop") is True
                        record.update(predicted_n=tokens, predicted_per_second=rate,
                                      terminal=terminal)
                        result = (tokens, rate, terminal)
                    except Exception as exc:
                        ended_monotonic = time.monotonic()
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    captured = None if response_capture is None else server_response.RawServerResponse(
                        phase, i, prompt_id, body, response_bytes, started_monotonic,
                        ended_monotonic, record["error"])
                    return (*result, record, captured)

                # The request phase proper starts HERE, warmup included: the
                # residency window must overlap the work, not merely the boot.
                request_start = time.time()
                request_started_monotonic = time.monotonic()
                # Warmup: one full np-wide round discarded, so cold-cache/clock-ramp does not
                # land in the measured sample (the first calibration run read high, then settled).
                observe("phase", "warmup")
                with cf.ThreadPoolExecutor(recipe.np) as ex:
                    warmup_rows = list(ex.map(lambda i: one(i, "warmup"), range(recipe.np)))
                observe("phase", "measurement")
                with cf.ThreadPoolExecutor(recipe.np) as ex:
                    rows = list(ex.map(lambda i: one(i, "measurement"), range(recipe.np)))
                observe("checkpoint", "measurement_end")
                request_end = time.time()
                request_ended_monotonic = time.monotonic()
                request_rows = [row[3] for row in warmup_rows + rows]
                if response_capture is not None:
                    # Persistence is outside both timed rounds and still inside the
                    # existing owned lifecycle, before its finally-driven teardown.
                    response_reference = response_capture.seal(
                        tuple(row[4] for row in warmup_rows + rows), process_pid=srv.pid,
                        request_started_monotonic_s=request_started_monotonic,
                        request_ended_monotonic_s=request_ended_monotonic)
                toks = [row[0] for row in rows]
                if any(row[3]["error"] for row in warmup_rows + rows):
                    raise ServerDied("one or more serving slots failed; see collected observation")
                if frozen_requests is None and min(toks) < recipe.n_predict // 2:
                    raise ServerDied(f"degenerate measurement: tokens={toks}")
                if frozen_requests is not None and not all(row[2] for row in warmup_rows + rows):
                    raise ServerDied("frozen request lacks explicit terminal completion")
                value = sum(row[1] for row in rows)
            finally:
                observe("phase", "teardown")
                srv.terminate()
                try:
                    srv.wait(30)
                    teardown = "terminated"
                except Exception:
                    srv.kill()
                    srv.wait(10)
                    teardown = "killed"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        window_end = time.time()
        try:
            if sampler is not None:
                try:
                    record = _residency_record(sampler, window_start=window_start,
                                               window_end=window_end,
                                               request_start=request_start,
                                               request_end=request_end, backend=backend)
                    if evidence is not None:
                        evidence.append(record)
                    if observation is not None:
                        exported = {
                            "schema": "epyc.autokernel.serving_observation.v1",
                            "process_pid": process_pid, "requests": request_rows,
                            "residency": record, "teardown": teardown, "failure": failure}
                        if response_reference is not None:
                            exported["server_responses"] = response_reference
                        observation.append(exported)
                except Exception:
                    if failure is None:
                        raise
                    # Preserve the serving exception already in flight. The auxiliary
                    # export failure cannot replace the launch/readback/request cause.
        finally:
            observer_finish_ok = observe("finish")
    # Success path only. An exception already in flight carries its own reason, and
    # replacing it with a residency refusal would hide the real fault -- while the
    # record above is appended either way, so a failed launch still leaves its window.
    _refuse_if_not_resident(recipe, record, backend=backend)
    if observation_session is not None and (
            not observer_finish_ok or not observation_session.shutdown_resolved):
        raise lifecycle_observation.ObserverShutdownUnresolved(
            "lifecycle observer ownership remains unresolved; refusing a successor unit")
    return value


def _spread(runs: Sequence[float]) -> dict:
    """Per-arm dispersion, in the SAME grammar the serving floor is defined in.

    WHY A MEANS-ONLY COMPARATOR IS NOT ENOUGH. The `GGML_NOHUGEPAGE_PROCESS` effect on the
    CPU surface was a COMPRESSED DOWNSIDE TAIL, not a shifted mean: between-launch sd fell
    25.3x (2.510% -> 0.481%) while the median moved +5.23%. A comparator that reports only
    medians can return "no effect" on a real one -- the arm that removes the bad launches
    looks identical to the arm that keeps them once the tail is averaged away. So every
    serving row now carries, per arm, the same p95-deviation-from-median the floor itself is
    (`calibrate_floor`), plus sd/min/max and the per-launch values.

    REPORTING ONLY. No decision rule reads these fields; nothing here changes a verdict.
    """
    runs = list(runs)
    med = statistics.median(runs)
    devs = sorted(abs(r / med - 1.0) * 100.0 for r in runs) if med else [0.0] * len(runs)
    p95 = devs[min(len(devs) - 1, int(round(0.95 * (len(devs) - 1))))]
    sd = statistics.pstdev(runs)
    return {"n": len(runs), "median": med, "mean": statistics.fmean(runs), "sd": sd,
            "cv_pct": round(sd / med * 100.0, 3) if med else None,
            "min": min(runs), "max": max(runs),
            "range_pct": round((max(runs) - min(runs)) / med * 100.0, 3) if med else None,
            # p95 |deviation from median|, IDENTICAL in definition to `floor_pct`, so an
            # arm's own spread is directly comparable to the floor it must clear.
            "p95_dev_pct": round(p95, 3), "max_dev_pct": round(devs[-1], 3),
            "runs": runs}


def compare(recipe: Recipe, anchor_build: Path, candidate_build: Path, *, pairs: int,
            floor_pct: float | None, port: int = 18311) -> dict:
    """Paired, alternating serving A/B: anchor vs candidate, `pairs` times, each pair a
    fresh server per side (drift control). Effect = median(candidate)/median(anchor) - 1.
    `decisive` is None when uncalibrated (no floor), so the keep gate fails closed."""
    a_runs, c_runs = [], []
    # Per-launch residency evidence, one record per server launch, per arm. Written by
    # `_measure_once`; a launch it could not sample lands here as `unproven` and a launch
    # it sampled as non-resident never gets here at all -- it raises.
    a_residency: list[dict] = []
    c_residency: list[dict] = []
    for _ in range(pairs):
        a_runs.append(_measure_once(recipe, anchor_build, port, evidence=a_residency))
        c_runs.append(_measure_once(recipe, candidate_build, port, evidence=c_residency))
    a_med, c_med = statistics.median(a_runs), statistics.median(c_runs)
    effect = c_med / a_med - 1.0
    decisive = None if floor_pct is None else (abs(effect) * 100.0 >= floor_pct)
    return {"schema": "epyc.autokernel.serving_ab.v1", "recipe": recipe.name,
            # The recipe is part of the artifact, so its identity travels with the number:
            # a reader (or a gate) can tell whether this row and the floor it was judged
            # against were even produced under the same launch conditions (R23-59).
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(),
            "metric": recipe.metric, "np": recipe.np, "pairs": pairs,
            "anchor_tok_s": a_med, "candidate_tok_s": c_med,
            "effect": effect, "effect_pct": effect * 100.0,
            "noise_floor_pct": floor_pct, "decisive": decisive,
            "anchor_samples": a_runs, "candidate_samples": c_runs,
            # Reporting only -- no decision rule reads these (see `_spread`).
            "anchor_spread": _spread(a_runs), "candidate_spread": _spread(c_runs),
            # PROVENANCE, not a decision input: no gate reads these. The row states
            # whether its own launches were shown to run on the device, so a later
            # reader never has to assume it -- and cannot be handed a tuple invented
            # after the fact, which is the one thing no re-analysis can supply.
            "residency": _residency_fold(a_residency + c_residency),
            "anchor_residency": a_residency, "candidate_residency": c_residency}


def calibrate_floor(recipe: Recipe, build_dir: Path, *, samples: int, port: int = 18311) -> dict:
    """A/A the serving metric `samples` times on ONE build: the run-to-run spread IS the
    noise floor a keep must clear. floor = p95 of |pairwise effect| against the median,
    reported at a few sample counts so a keep at N pairs is judged against the N-pair bar."""
    launch_residency: list[dict] = []
    runs = [_measure_once(recipe, build_dir, port, evidence=launch_residency)
            for _ in range(samples)]
    # `floor_pct` IS this arm's p95 deviation from its own median -- taken from `_spread`
    # so the floor and the per-arm spread reported by `compare` can never drift apart.
    sp = _spread(runs)
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(),
            "metric": recipe.metric, "np": recipe.np, "samples": samples,
            "median_tok_s": sp["median"], "floor_pct": sp["p95_dev_pct"],
            "runs": runs, "cv_pct": sp["cv_pct"], "spread": sp,
            # A floor is a bar every future keep is judged against, so the row records
            # whether the launches that DEFINED it were proven resident. `write_floor`
            # carries this to disk.
            "residency": _residency_fold(launch_residency),
            "launch_residency": launch_residency}


# ---------------------------------------------------------------------------
# The floor FILE: keyed by recipe IDENTITY, not by recipe NAME.
#
# `serving-floor.<name>.json` is a name-keyed cache of a number that is only meaningful
# under one measured condition. Loading it by name alone is how a stale floor gets to
# judge a new condition without anyone being told. Everything below exists so that the
# recipe's identity is WRITTEN into the file and CHECKED when it is read back.
# ---------------------------------------------------------------------------

def floor_key(name: str) -> str:
    """The filesystem-safe, deterministic key `serving-floor.<key>.json` uses.

    A name that is already safe and bounded is used VERBATIM, so the shipped
    `serving-floor.qwen3.8-27b-q8-gpu-dflash2-np4.json` keeps exactly the path it has
    today. Anything else is sanitised AND suffixed with a digest OF THE ORIGINAL NAME:
    sanitising alone would map two different recipes onto one floor file, which is the
    very defect this module now refuses -- an env value like `/tmp/a b` reaches the name
    through `with_env`, and `_`-substitution on its own is not injective.
    """
    if not isinstance(name, str) or not name:
        raise RecipeError("recipe name is empty: a floor file cannot be keyed by it")
    if (len(name) <= FLOOR_KEY_MAX and not name.startswith(".")
            and all(ch in _FLOOR_SAFE_CHARS for ch in name)):
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    cleaned = "".join(ch if ch in _FLOOR_SAFE_CHARS else "_" for ch in name).lstrip(".")
    cleaned = cleaned[:FLOOR_KEY_MAX - len(digest) - 1] or "recipe"
    return f"{cleaned}-{digest}"


def floor_path(store: Path | str, recipe: Recipe) -> Path:
    """Where THIS recipe's serving floor lives. One place, so a reader and a writer can
    never disagree about the filename."""
    return Path(store) / f"serving-floor.{floor_key(recipe.name)}.json"


@dataclass(frozen=True)
class FloorReading:
    """A floor loaded FOR A GATE DECISION, with the provenance of its identity check.

    `provenance` is three-valued and every value is explicit, because "no floor" and
    "a floor whose provenance nobody checked" are different facts and only one of them
    is safe to grandfather:

      * ``verified``   -- the file carries this recipe's `recipe_hash`;
      * ``unverified`` -- the file predates identity stamping and carries no hash at all;
        it is USED (hard-failing would block the loop on every existing floor) but every
        record it touches is stamped so a reader can tell it apart. Recalibrate.
      * ``absent``     -- no floor file. Uncalibrated: both gate triggers stay blocked
        (R23-54) and `compare` returns `decisive: None`.

    A file whose hash DISAGREES is not a provenance value -- it raises.
    """
    floor_pct: float | None
    provenance: str
    path: Path
    row: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.provenance == "verified"

    @property
    def residency_status(self) -> str:
        """`proven` only when the file says so. Absent evidence reads as `unproven`,
        never as proven -- the same fail-closed direction `provenance` takes."""
        block = self.row.get("residency") or {}
        return str(block.get("status") or RESIDENCY_UNPROVEN)


def _stamped_residency(block: object) -> dict:
    """The residency block a floor file carries, never absent and never invented."""
    if isinstance(block, Mapping) and block:
        return dict(block)
    return dict(_residency_fold(()),
                note="this floor carries no per-launch residency evidence: it was "
                     "calibrated before the serving path sampled (R23-60), or by a "
                     "harness that did not record it. Not a claim that it ran off the "
                     "device -- a statement that nothing proves it ran on one. "
                     "Recalibrate to obtain the proof; it cannot be added afterwards.")


def write_floor(store: Path | str, recipe: Recipe, row: Mapping, *,
                conditions: Mapping | None = None) -> Path:
    """Persist a calibrated floor WITH the identity of the recipe it was calibrated under.

    THE ONE WRITER. `calibrate_floor` already returns `recipe_hash` / `recipe_describe` /
    `recipe`, but a returned dict proves nothing about what reached the disk -- and the
    file is the only thing a later run sees. This stamps them at top level, refuses a row
    produced by a DIFFERENT recipe (the copy-paste that would otherwise file one arm's
    A/A under another arm's name), and writes atomically so a crashed calibration cannot
    leave a half-written floor for a gate to read.

    `conditions` is free-form provenance for humans (host state, harness, timestamp); it
    is merged, never allowed to overwrite the identity keys.
    """
    body = dict(row)
    stamped = body.get("recipe_hash")
    if stamped is not None and stamped != recipe.recipe_hash:
        raise ServingFloorMismatch(
            f"refusing to file this floor under {recipe.name!r}: the row was produced by "
            f"recipe_hash {stamped}, but the recipe writing it is {recipe.recipe_hash} "
            f"({recipe.describe()}). A floor filed under the wrong recipe is worse than "
            f"no floor -- it is a bar nobody will question.")
    if conditions:
        body["conditions"] = {**dict(body.get("conditions") or {}), **dict(conditions)}
    body["recipe"] = recipe.name
    body["recipe_hash"] = recipe.recipe_hash
    body["recipe_describe"] = recipe.describe()
    body["recipe_env"] = dict(recipe.env or {})
    # A floor whose provenance is silent reads as a floor whose provenance is fine. A row
    # with no residency block is stamped `unproven` EXPLICITLY -- it is not a claim that
    # the calibration ran on the CPU, it is a refusal to let the absence pass unremarked.
    body["residency"] = _stamped_residency(body.get("residency"))
    target = floor_path(store, recipe)
    return status.write_json(target.parent, target.name, body, prefix=".sv-floor-")


def load_floor(store: Path | str, recipe: Recipe) -> FloorReading:
    """Load the serving floor for `recipe`, or REFUSE one calibrated under another.

    Fail-closed, and deliberately NOT by degrading to "no floor": an absent floor already
    blocks both gate triggers (R23-54), so a silent downgrade would surface as a cadence
    bug -- the gate quietly never firing -- rather than as the stale floor it is. The
    caller gets an exception naming both hashes, or a reading it can trust the provenance
    of.
    """
    path = floor_path(store, recipe)
    if not path.is_file():
        return FloorReading(None, "absent", path, {})
    row = json.loads(path.read_text(encoding="utf-8"))
    stamped = row.get("recipe_hash")
    if stamped is None:
        # Grandfathered: written before floors carried an identity. Proceed -- hard-failing
        # here would block the loop at relaunch on every floor that exists today -- but the
        # provenance travels with every record the number touches.
        return FloorReading(row.get("floor_pct"), "unverified", path, row)
    if stamped != recipe.recipe_hash:
        raise ServingFloorMismatch(
            f"serving floor {path.name} was calibrated under a DIFFERENT recipe: the file "
            f"carries recipe_hash {stamped}, the live recipe is {recipe.recipe_hash} "
            f"({recipe.describe()}). A floor is a property of the measured CONDITION, not "
            f"of the recipe's NAME -- recalibrate the floor for THIS recipe "
            f"(serving.calibrate_floor + serving.write_floor) before any gate is judged "
            f"against it. Refusing rather than falling back to 'no floor': an absent floor "
            f"merely blocks both gate triggers (R23-54) and would read as a cadence bug "
            f"instead of a stale floor.")
    return FloorReading(row.get("floor_pct"), "verified", path, row)


__all__ = ["FLOOR_KEY_MAX", "LOADER_OWNED_ENV", "RECIPE_SCHEMA", "RESIDENCY_PROVEN",
           "RESIDENCY_NOT_APPLICABLE", "RESIDENCY_SCHEMA", "RESIDENCY_UNPROVEN", "UNSET",
           "EnvReadbackFailed", "FloorReading", "Recipe", "RecipeError", "ServerDied",
           "ServingFloorMismatch", "ServingNotResident", "calibrate_floor", "compare",
           "covers_request_phase", "floor_key", "floor_path", "load_floor",
           "verify_env_readback", "write_floor"]
