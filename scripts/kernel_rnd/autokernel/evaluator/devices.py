#!/usr/bin/env python3
"""devices.py — the device-NAME vocabulary, once, for the whole bundle.

WHY THIS MODULE EXISTS
----------------------
Both speech adapters carried the same hole: `check_device_evidence(expected_lane=
"gpu")` required a `Device N: <name>` line and never asked whether `<name>` denotes
a GPU, so **`Device 0: CPU` satisfied a GPU cell**. That is the 2026-07-31 silent
CPU fallback (INC-20260731-ggml-linkage-silent-cpu-fallback) surviving the very
check written to catch it: the incident's binary printed `use gpu = 1`, the check
was tightened to demand a device line instead — and a device line naming the CPU
backend is exactly what a silently-fallen-back ggml prints.

Closing it needs a vocabulary: a table saying which device names denote a GPU and
which denote the host. That table lives **here**, in the evaluator bundle, and not
in either adapter, because two copies diverge — this package has already paid for
that lesson three times (two `proposal_fingerprint`s writing one journal field, two
§6.5 oracle registries sharing one id out of nineteen, two harvest-class
vocabularies). A backend adapter owns its own log GRAMMAR (`whisper_init:` versus
`ggml_cuda_init:` line shapes); it does not own what a device name means.

WHAT IT IS NOT
--------------
It is **not** a device inventory, a claim, or a probe. It reads no file, runs no
process, opens no device, and imports nothing that could. It classifies a string
that a runner captured from an engine's own startup log.

It is also **not** a complete catalogue of the world's accelerators. The vocabulary
is scoped to the devices this host actually has — an AMD Instinct MI210 (`gfx90a`,
CDNA2) under ROCm 6.2, and an AMD EPYC 9655 host — plus the generic names ggml's
own backends register (`CPU`, `BLAS`). A name outside the table classifies as
`unknown`, and an unknown name is `COULD_NOT_CHECK`, **never** `PASS`: a device this
table cannot name is a device this table cannot vouch for. That is the same third
outcome the rest of the package is built on, and it is why extending the table is
safe — the failure mode of an incomplete table is an unrunnable check, not a false
one.

THE AMBIGUITY RULE
------------------
`AMD Instinct MI210` and `AMD EPYC 9655` share a vendor word, so vendor words are
NOT tokens. A name that matches BOTH classes is `ambiguous` and is
`COULD_NOT_CHECK` — never resolved by precedence. A precedence rule here would
decide, silently and forever, which of two contradictory readings of one string
wins; a device line that reads as both is evidence of nothing.

Design context: §13.3/§13.4 (the speech adapters), §10.2 phase 2 (build+linkage),
`handoffs/active/autokernel-research-loop.md`.
"""
from __future__ import annotations

MODULE_ID = "autokernel.evaluator.devices/v1"

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .. import schemas

# =============================================================================
# Errors
# =============================================================================


class DeviceVocabularyError(Exception):
    """A refusal. Never a degraded answer."""


class DeviceStateParseError(DeviceVocabularyError):
    """A required numeric ROCm state field could not be parsed."""


# RVP-C3-5's duration floor is local measurement evidence, not a convention
# copied from another accelerator.  The 60-second RVP-T0-1 saturation receipt
# began at 800 MHz and first observed the MI210's nominal 1700 MHz SCLK at this
# offset.  A shorter ranked timing window can therefore fit wholly inside the
# startup transition this host actually exhibited.  Round upward to an integer
# nanosecond so float truncation cannot admit a window just below the observation.
GFX90A_DURATION_FLOOR_DERIVATION_ID = (
    "rvp-t0-1-first-observed-nominal-sclk/v1")
GFX90A_DURATION_FLOOR_EVIDENCE_REF = (
    "rvp-t0-1-20260811T0906Z/receipt.json@sha256:"
    "07788e1d488ecec062e8133dd9e11d379e5075afbcc20f80b6da37e345533431"
    "#device_sampling.samples[1]")
GFX90A_FIRST_NOMINAL_SCLK_OFFSET_S = 0.25009090290404856
GFX90A_MIN_RANKED_WINDOW_NS = math.ceil(
    GFX90A_FIRST_NOMINAL_SCLK_OFFSET_S * 1_000_000_000)


@dataclass(frozen=True)
class RankedDurationWindowAdmission:
    """Device-local absolute floor for a timing window used in a speed rank.

    This is separate from the relative paired-block noise test.  An arbitrarily
    repeatable microsecond observation can still measure only gfx90a's clock-ramp
    transient.  RVP-C3-5 requires both: the complete repetition window must first
    clear this absolute, locally measured floor; only then may paired statistics
    compare its constituent repetitions.
    """

    architecture: str
    device_id: str
    min_window_ns: int
    first_nominal_sclk_offset_s: float
    evidence_ref: str
    derivation_id: str = GFX90A_DURATION_FLOOR_DERIVATION_ID

    def __post_init__(self) -> None:
        for name in ("architecture", "device_id", "evidence_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ranked duration admission {name} must be non-empty")
        if isinstance(self.min_window_ns, bool) or not isinstance(
                self.min_window_ns, int) or self.min_window_ns <= 0:
            raise ValueError("ranked duration admission min_window_ns must be positive")
        if isinstance(self.first_nominal_sclk_offset_s, bool) or not isinstance(
                self.first_nominal_sclk_offset_s, (int, float)) or not math.isfinite(
                    float(self.first_nominal_sclk_offset_s)) \
                or self.first_nominal_sclk_offset_s <= 0:
            raise ValueError(
                "ranked duration admission first_nominal_sclk_offset_s must be positive")
        expected = math.ceil(self.first_nominal_sclk_offset_s * 1_000_000_000)
        if self.min_window_ns != expected:
            raise ValueError(
                "ranked duration admission min_window_ns must be the upward-rounded "
                "first nominal-SCLK observation")
        if self.derivation_id != GFX90A_DURATION_FLOOR_DERIVATION_ID:
            raise ValueError(
                "ranked duration admission derivation_id is not implemented")

    def check(self, observed_ns: Sequence[Any], *, device_id: str) -> schemas.Check:
        """Refuse a missing, foreign-device, malformed, or sub-floor window."""
        if device_id != self.device_id:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the ranked duration floor was measured for {self.device_id} "
                f"({self.architecture}), but the receipt names {device_id!r}; a floor "
                "from one device may not be imported onto another",))
        if isinstance(observed_ns, (str, bytes)) or not isinstance(
                observed_ns, Sequence) or not observed_ns:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "the ranked GPU cell carries no raw repetition durations; an aggregate "
                "throughput cannot establish that its complete timing window cleared the "
                "absolute gfx90a floor",))
        values = []
        for index, value in enumerate(observed_ns):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) <= 0:
                return schemas.Check(schemas.COULD_NOT_CHECK, (
                    f"ranked GPU duration window {index} is not a finite positive "
                    f"nanosecond count: {value!r}",))
            values.append(float(value))
        total_ns = sum(values)
        if total_ns < self.min_window_ns:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"ranked GPU duration window totals {total_ns:.0f} ns across "
                f"{len(values)} repetition(s), below the "
                f"local {self.architecture} floor {self.min_window_ns} ns; the cell is "
                "inconclusive and receives no speed rank",
                f"floor derivation {self.derivation_id}: first nominal SCLK observed "
                f"at {self.first_nominal_sclk_offset_s:.12g}s in {self.evidence_ref}",))
        return schemas.Check(schemas.PASS, (
            f"the ranked GPU duration window totals {total_ns:.0f} ns across "
            f"{len(values)} repetition(s), at or above the local {self.architecture} "
            f"floor {self.min_window_ns} ns ({self.evidence_ref})",))

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "device_id": self.device_id,
            "min_window_ns": self.min_window_ns,
            "first_nominal_sclk_offset_s": self.first_nominal_sclk_offset_s,
            "evidence_ref": self.evidence_ref,
            "derivation_id": self.derivation_id,
        }


GFX90A_RANKED_DURATION_ADMISSION = RankedDurationWindowAdmission(
    architecture="gfx90a",
    device_id="ROCm0",
    min_window_ns=GFX90A_MIN_RANKED_WINDOW_NS,
    first_nominal_sclk_offset_s=GFX90A_FIRST_NOMINAL_SCLK_OFFSET_S,
    evidence_ref=GFX90A_DURATION_FLOOR_EVIDENCE_REF)


@dataclass(frozen=True)
class DeviceStateSample:
    """One parsed ROCm device-state snapshot; no raw text masquerades as data."""

    sclk_mhz: float
    mclk_mhz: float
    power_w: float
    temperature_c: float
    under_measurement_load: bool

    def __post_init__(self) -> None:
        for name in ("sclk_mhz", "mclk_mhz", "power_w", "temperature_c"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"device sample {name} must be a finite non-negative number")
        if not isinstance(self.under_measurement_load, bool):
            raise TypeError("under_measurement_load must be bool")

    def to_dict(self) -> dict:
        return {
            "sclk_mhz": float(self.sclk_mhz),
            "mclk_mhz": float(self.mclk_mhz),
            "power_w": float(self.power_w),
            "temperature_c": float(self.temperature_c),
            "under_measurement_load": self.under_measurement_load,
        }


@dataclass(frozen=True)
class DeviceState:
    """Parsed state across a window with a mechanically derived throttle verdict."""

    device_id: str
    source: str
    nominal_sclk_mhz: float
    min_sclk_ratio: float
    samples: tuple
    receipt_ref: str

    def __post_init__(self) -> None:
        for name in ("device_id", "source", "receipt_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"device state {name} must be a non-empty string")
        if isinstance(self.nominal_sclk_mhz, bool) or not isinstance(
                self.nominal_sclk_mhz, (int, float)) or self.nominal_sclk_mhz <= 0:
            raise ValueError("nominal_sclk_mhz must be positive")
        if isinstance(self.min_sclk_ratio, bool) or not isinstance(
                self.min_sclk_ratio, (int, float)) or not 0 < self.min_sclk_ratio <= 1:
            raise ValueError("min_sclk_ratio must be in (0, 1]")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("device state samples must be a non-empty tuple")
        if not all(isinstance(row, DeviceStateSample) for row in self.samples):
            raise TypeError("device state samples must contain DeviceStateSample")

    @property
    def throttle_observed(self) -> bool:
        floor = self.nominal_sclk_mhz * self.min_sclk_ratio
        return any(row.under_measurement_load and row.sclk_mhz < floor
                   for row in self.samples)

    def check(self) -> schemas.Check:
        loaded = tuple(row for row in self.samples if row.under_measurement_load)
        if not loaded:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("no device-state sample was captured under measurement load; an idle "
                 "clock cannot establish absence of throttling",))
        if self.throttle_observed:
            floor = self.nominal_sclk_mhz * self.min_sclk_ratio
            observed = min(row.sclk_mhz for row in loaded)
            return schemas.Check(
                schemas.FAIL,
                (f"GPU throttle observed: loaded sclk fell to {observed:g} MHz below "
                 f"the declared floor {floor:g} MHz",))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "source": self.source,
            "nominal_sclk_mhz": float(self.nominal_sclk_mhz),
            "min_sclk_ratio": float(self.min_sclk_ratio),
            "samples": [row.to_dict() for row in self.samples],
            "throttle_observed": self.throttle_observed,
            "receipt_ref": self.receipt_ref,
        }


_SCLK_RE = re.compile(r"\bsclk clock level:\s*\d+:\s*\(([0-9.]+)\s*Mhz\)", re.I)
_MCLK_RE = re.compile(r"\bmclk clock level:\s*\d+:\s*\(([0-9.]+)\s*Mhz\)", re.I)
_POWER_RE = re.compile(r"Average Graphics Package Power\s*\(W\):\s*([0-9.]+)", re.I)
_JUNCTION_TEMP_RE = re.compile(
    r"Temperature \(Sensor junction\) \(C\):\s*([0-9.]+)", re.I)


def parse_rocm_smi_snapshot(*, clocks_text: str, power_text: str,
                            temperature_text: str,
                            under_measurement_load: bool) -> DeviceStateSample:
    """Parse the four numeric fields RVP-C3-3 requires from captured text."""
    values = {}
    for name, pattern, text in (
            ("sclk_mhz", _SCLK_RE, clocks_text),
            ("mclk_mhz", _MCLK_RE, clocks_text),
            ("power_w", _POWER_RE, power_text),
            ("temperature_c", _JUNCTION_TEMP_RE, temperature_text)):
        if not isinstance(text, str):
            raise TypeError(f"{name} capture must be text")
        match = pattern.search(text)
        if match is None:
            raise DeviceStateParseError(
                f"{name} was not present as a numeric field in the rocm-smi capture")
        values[name] = float(match.group(1))
    return DeviceStateSample(
        **values, under_measurement_load=under_measurement_load)


# =============================================================================
# The classes a device name can fall into
# =============================================================================

#: The name denotes an accelerator: the measurement's GPU lane is credible.
GPU = "gpu"
#: The name denotes the host: a CPU backend, or a host-side BLAS device.
CPU = "cpu"
#: No vocabulary entry matched. NOT a class, an ABSENCE of one.
UNKNOWN = "unknown"
#: Entries of both classes matched. A contradiction, never resolved by precedence.
AMBIGUOUS = "ambiguous"

DEVICE_CLASSES = frozenset({GPU, CPU, UNKNOWN, AMBIGUOUS})

#: The two classes a vocabulary entry may declare. `unknown` and `ambiguous` are
#: RESULTS of matching, so an entry cannot declare them.
DECLARABLE_CLASSES = frozenset({GPU, CPU})

#: The lanes a caller may grade against — `schemas.RESOURCE_LANES` minus `stack`,
#: which is the serving-runtime lane and names no device at all (§13.5).
GRADEABLE_LANES = frozenset({CPU, GPU})

#: Tokens the vocabulary MUST declare, in the class it MUST declare them in, and why
#: each one is load-bearing.
#:
#: "Both classes are represented" is NOT enough on its own. That rule is satisfied by
#: any table with one row of each class, so deleting the `cpu` row leaves the import
#: green, `epyc` and `blas` still represent the host class — and `Device 0: CPU`, the
#: exact string the 2026-07-31 silent fallback prints, degrades from FAIL to
#: COULD_NOT_CHECK. The defect this module exists to close would be reopened by
#: deleting one row of the table that closes it, which is the package's standing
#: definition of a guarantee that is not one.
#:
#: The class is pinned as well as the token: a `cpu` row re-declared as GPU would
#: satisfy a presence-only rule and invert the verdict.
MANDATORY_TOKENS = {
    "cpu": (CPU, "the string ggml's CPU backend registers as its device name; "
                 "`Device 0: CPU` on a GPU cell is INC-20260731's signature and "
                 "must be a FAIL, not an unrecognised name"),
    "mi210": (GPU, "the host's only accelerator, named in "
                   "artifacts/operator/ratify_speech_kernel_freeze_20260731.json"),
    "gfx90a": (GPU, "the ISA the frozen speech kernels' load-bearing patches target "
                    "(CLAUDE.md kernel-set freeze note); ROCm prints it in the "
                    "device's own ISA field"),
}


@dataclass(frozen=True)
class DeviceVocabularyEntry:
    """One device-name token and what it denotes.

    `token` is matched case-insensitively on WORD boundaries, never as a bare
    substring: `cpu` must not match inside `libggml-cpu.so`, and a substring match
    is how a library name becomes a device classification.
    """

    token: str
    device_class: str
    denotes: str
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token.strip():
            raise DeviceVocabularyError("vocabulary token must be a non-empty string")
        if self.token != self.token.lower():
            raise DeviceVocabularyError(
                f"vocabulary token {self.token!r} must be lowercase; matching is "
                f"case-insensitive and a mixed-case token hides that")
        if self.device_class not in DECLARABLE_CLASSES:
            raise DeviceVocabularyError(
                f"entry {self.token!r} declares class {self.device_class!r}; an entry "
                f"may only declare {sorted(DECLARABLE_CLASSES)} — {UNKNOWN!r} and "
                f"{AMBIGUOUS!r} are results of matching, not declarations")
        for field in ("denotes", "provenance"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise DeviceVocabularyError(
                    f"entry {self.token!r} must carry a non-empty {field}: a token whose "
                    f"meaning and origin are unrecorded is not auditable")

    def to_dict(self) -> dict:
        return {"token": self.token, "device_class": self.device_class,
                "denotes": self.denotes, "provenance": self.provenance}


#: The vocabulary. Scoped to THIS host, and every row says where the name came from.
#:
#: Vendor words (`amd`) are deliberately absent: they are shared by the MI210 and
#: the EPYC, so a vendor token would make every device name ambiguous.
#: Runtime words (`rocm`, `hip`) are deliberately absent too: they name a RUNTIME,
#: not a device — `found 1 ROCm devices:` is a header line, and `libggml-hip.so` is
#: a library. Neither establishes that a device loaded.
HOST_DEVICE_VOCABULARY = (
    DeviceVocabularyEntry(
        token="mi210", device_class=GPU,
        denotes="AMD Instinct MI210 (gfx90a, CDNA2, 64 GB HBM2e)",
        provenance="the host's only accelerator; named in "
                   "artifacts/operator/ratify_speech_kernel_freeze_20260731.json "
                   "as 'HIP / gfx90a (MI210)'"),
    DeviceVocabularyEntry(
        token="instinct", device_class=GPU,
        denotes="the AMD Instinct accelerator family the MI210 belongs to",
        provenance="ggml's ROCm device line prints the marketing name, "
                   "'AMD Instinct MI210'"),
    DeviceVocabularyEntry(
        token="gfx90a", device_class=GPU,
        denotes="the LLVM target of the MI210; ROCm prints it in the device's "
                "ISA field (gfx90a:sramecc+:xnack-)",
        provenance="CLAUDE.md kernel-set freeze note; the load-bearing speech "
                   "patches are gfx90a-specific"),
    DeviceVocabularyEntry(
        token="radeon", device_class=GPU,
        denotes="AMD's Radeon/Radeon Pro GPU families, which ROCm also enumerates "
                "under this marketing name",
        provenance="declared so a Radeon-branded accelerator is not silently "
                   "UNKNOWN; not present on this host"),
    DeviceVocabularyEntry(
        token="cpu", device_class=CPU,
        denotes="ggml's CPU backend, which registers its device as 'CPU'",
        provenance="THE defect this table closes: `Device 0: CPU` satisfied a GPU "
                   "cell in both speech adapters"),
    DeviceVocabularyEntry(
        token="epyc", device_class=CPU,
        denotes="the host processor, an AMD EPYC 9655 (96 cores / 192 threads)",
        provenance="the host CPU; some backends print the /proc/cpuinfo model name"),
    DeviceVocabularyEntry(
        token="blas", device_class=CPU,
        denotes="ggml's BLAS backend — a HOST device, not an accelerator",
        provenance="ggml enumerates it alongside the CPU backend as 'Device N: BLAS'"),
)


def _validate_vocabulary(entries: Sequence[DeviceVocabularyEntry]) -> None:
    """Both classes represented, every MANDATORY token present in its own class.

    Run at IMPORT time. A vocabulary that lost its GPU rows would classify every
    accelerator as `unknown`, and every GPU cell would degrade to COULD_NOT_CHECK
    forever — fail-safe, but silently unrunnable. The package's rule for this is
    `statistics.py`'s: a registry that drifts is an ImportError, not a surprise at
    call time.

    The mandatory-token rule is what makes the table's guarantee undeletable. A
    "both classes represented" rule alone is satisfied after deleting the `cpu` row
    (`epyc` and `blas` still carry the host class), and with that row gone the
    module's own headline defect — `Device 0: CPU` satisfying a GPU cell — comes
    back as COULD_NOT_CHECK instead of FAIL.
    """
    if not entries:
        raise DeviceVocabularyError("the device vocabulary is empty")
    seen: set = set()
    for entry in entries:
        if entry.token in seen:
            raise DeviceVocabularyError(f"duplicate vocabulary token {entry.token!r}")
        seen.add(entry.token)
    classes = {e.device_class for e in entries}
    if classes != DECLARABLE_CLASSES:
        raise DeviceVocabularyError(
            f"the vocabulary declares {sorted(classes)}; both {sorted(DECLARABLE_CLASSES)} "
            f"must be represented or one lane can never be graded")
    by_token = {e.token: e for e in entries}
    for token, (device_class, why) in sorted(MANDATORY_TOKENS.items()):
        entry = by_token.get(token)
        if entry is None:
            raise DeviceVocabularyError(
                f"the vocabulary does not declare the mandatory token {token!r} "
                f"({why}). Deleting it does not make the check stricter, it makes the "
                f"name unrecognised — and an unrecognised name is COULD_NOT_CHECK, "
                f"which is how this table's own guarantee would be deleted")
        if entry.device_class != device_class:
            raise DeviceVocabularyError(
                f"the mandatory token {token!r} is declared as {entry.device_class!r} "
                f"but must denote {device_class!r} ({why}); re-classing it inverts every "
                f"verdict that depends on it")


_validate_vocabulary(HOST_DEVICE_VOCABULARY)

_TOKEN_PATTERNS = tuple(
    (entry, re.compile(r"(?<![0-9a-z])" + re.escape(entry.token) + r"(?![0-9a-z])",
                       re.IGNORECASE))
    for entry in HOST_DEVICE_VOCABULARY
)


def device_vocabulary() -> tuple:
    """The vocabulary, as data. One table, read by every consumer."""
    return HOST_DEVICE_VOCABULARY


# =============================================================================
# Classification
# =============================================================================


@dataclass(frozen=True)
class DeviceNameVerdict:
    """What a single device name was classified as, and by which entries."""

    name: str
    device_class: str
    matched_tokens: tuple

    def __post_init__(self) -> None:
        if self.device_class not in DEVICE_CLASSES:
            raise DeviceVocabularyError(f"invalid device class {self.device_class!r}")

    @property
    def is_gpu(self) -> bool:
        return self.device_class == GPU

    @property
    def is_cpu(self) -> bool:
        return self.device_class == CPU

    def to_dict(self) -> dict:
        return {"name": self.name, "device_class": self.device_class,
                "matched_tokens": list(self.matched_tokens)}


def classify_device_name(name: Any) -> DeviceNameVerdict:
    """Classify one device name against the vocabulary.

    Four outcomes, and the last two are the reason this returns a verdict object
    rather than a bool: `unknown` (nothing matched) and `ambiguous` (both classes
    matched) are distinct states, and neither is a GPU and neither is a CPU.
    """
    if not isinstance(name, str):
        raise DeviceVocabularyError(f"device name must be a string, got "
                                    f"{type(name).__name__}")
    text = name.strip()
    if not text:
        return DeviceNameVerdict(name="", device_class=UNKNOWN, matched_tokens=())
    matched = tuple(entry for entry, pattern in _TOKEN_PATTERNS if pattern.search(text))
    classes = {entry.device_class for entry in matched}
    tokens = tuple(entry.token for entry in matched)
    if not classes:
        return DeviceNameVerdict(name=text, device_class=UNKNOWN, matched_tokens=())
    if len(classes) > 1:
        return DeviceNameVerdict(name=text, device_class=AMBIGUOUS, matched_tokens=tokens)
    return DeviceNameVerdict(name=text, device_class=classes.pop(), matched_tokens=tokens)


def names_look_like_a_device(name: Any) -> bool:
    """True when the vocabulary recognises `name` as denoting SOME device.

    Used by the adapter self-audit below to tell a device NAME from a library name
    or a path — not by any gate.
    """
    return classify_device_name(name).device_class in (GPU, CPU)


# =============================================================================
# The lane gate
# =============================================================================


def _require_lane(expected_lane: Any) -> str:
    if expected_lane not in GRADEABLE_LANES:
        raise DeviceVocabularyError(
            f"expected_lane must be one of {sorted(GRADEABLE_LANES)}, got "
            f"{expected_lane!r}")
    return expected_lane


def check_device_names(names: Iterable[Any], *, expected_lane: str) -> schemas.Check:
    """Grade every device name an engine's startup log named, against one lane.

    The reduction, and why each branch is what it is:

      * **GPU lane, any name denotes a GPU  ->  PASS.** A GPU was loaded. A CPU or
        BLAS device listed alongside it is normal — ggml enumerates the host
        backends too — and does not unload the accelerator.
      * **GPU lane, every name denotes the host  ->  FAIL.** `Device 0: CPU` is the
        silent-fallback signature, and accepting it is the carried-forward defect
        this module closes.
      * **GPU lane, nothing classifiable  ->  COULD_NOT_CHECK.** An unrecognised
        device name is a device this table cannot vouch for, in either direction.
      * **CPU lane, any name denotes a GPU  ->  FAIL**, whatever else is listed: a
        CPU cell that loaded an accelerator did not measure the declared footprint.
      * **CPU lane, a host device and no GPU  ->  PASS.**

    An EMPTY name list is COULD_NOT_CHECK: no names is no evidence. Whether the
    absence of a device line is itself a finding is the caller's log grammar to
    decide, and the two speech adapters decide it differently on the CPU lane.
    """
    lane = _require_lane(expected_lane)
    verdicts = tuple(classify_device_name(n) for n in names)
    if not verdicts:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no device names were supplied, so nothing establishes which device "
            "loaded; an empty enumeration is not evidence of a lane",))

    gpus = tuple(v for v in verdicts if v.is_gpu)
    cpus = tuple(v for v in verdicts if v.is_cpu)
    unresolved = tuple(v for v in verdicts if v.device_class in (UNKNOWN, AMBIGUOUS))
    listed = ", ".join(repr(v.name) for v in verdicts)

    if lane == GPU:
        if gpus:
            return schemas.Check(schemas.PASS)
        if cpus:
            return schemas.Check(schemas.FAIL, (
                f"the GPU cell's device enumeration names only host devices ({listed}); "
                f"a `Device N: <name>` line proves a device was ENUMERATED, not that an "
                f"accelerator was loaded, and a CPU device line is the exact signature "
                f"of the 2026-07-31 silent CPU fallback "
                f"(INC-20260731-ggml-linkage-silent-cpu-fallback)",))
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"no supplied device name is in the declared device vocabulary ({listed}); "
            f"an unrecognised or ambiguous name establishes neither lane. Declared "
            f"tokens: {sorted(e.token for e in HOST_DEVICE_VOCABULARY)}",))

    if gpus:
        return schemas.Check(schemas.FAIL, (
            f"a CPU cell's log enumerates an accelerator "
            f"({', '.join(repr(v.name) for v in gpus)}); the measured footprint is not "
            f"the declared one",))
    if cpus:
        return schemas.Check(schemas.PASS)
    return schemas.Check(schemas.COULD_NOT_CHECK, (
        f"no supplied device name is in the declared device vocabulary ({listed}); an "
        f"unrecognised or ambiguous name establishes neither lane. Unresolved: "
        f"{[v.device_class for v in unresolved]}",))


def check_device_name(name: Any, *, expected_lane: str) -> schemas.Check:
    """`check_device_names` for a single name. Same rules, no shortcut."""
    return check_device_names((name,), expected_lane=expected_lane)


# =============================================================================
# The audit — one vocabulary, proved from a consumer's own AST
# =============================================================================

#: Collection literals in a consumer's source whose STRINGS are paths, sonames or
#: shell globs are not device vocabularies. `libggml-cpu.so` contains the token
#: `cpu` on a word boundary, and `EXPECTED_SHARED_LIBRARIES` legitimately names it.
_NOT_A_DEVICE_NAME = ("/", ".so", ".a", ".sh", "*", ".py", ".json", ".md")

#: A bare LANE identifier is not a device name. `schemas.RESOURCE_LANES` is the
#: lane vocabulary — `qwentts_tts.SIBLING_STABLE_TARGETS` is keyed `cpu`/`gpu`/`stt`
#: and names four production paths, not four devices — and an audit that forbade a
#: consumer's own lane keys would be a guard banning the idiom it exists to protect
#: (`feedback_guard_must_not_forbid_its_own_idiom`).
#:
#: The exclusion is deliberately CASE-SENSITIVE: lane identifiers are lowercase by
#: definition, while ggml prints its host device as `CPU`. `"CPU"` in a collection
#: literal is therefore still caught, and that is the string a divergent local
#: vocabulary would actually contain.
_LANE_IDENTIFIERS = frozenset(schemas.RESOURCE_LANES)


def _is_device_name_literal(text: str) -> bool:
    if text in _LANE_IDENTIFIERS:
        return False
    if any(marker in text for marker in _NOT_A_DEVICE_NAME):
        return False
    return names_look_like_a_device(text)


def _collection_elements(value: Any) -> list:
    """The string-constant elements of a collection literal, keys included."""
    if isinstance(value, ast.Dict):
        return list(value.keys) + list(value.values)
    if isinstance(value, (ast.Set, ast.Tuple, ast.List)):
        return list(value.elts)
    return []


def _vocabulary_sites(tree: ast.AST) -> list:
    """Every collection literal a consumer could USE as a device vocabulary.

    Two site kinds, and neither is "any collection literal anywhere":

      * the value of a **binding** (`X = (...)`, annotated or augmented), at module
        level OR inside a function. Module level alone is not enough — moving the
        table inside `check_device_evidence` evades a module-level rule completely,
        and a table one indent deeper is the same table.
      * a **membership comparator** (`if name in ("MI210", "CPU")`), which is the
        table written without ever being bound to a name.

    Everything else is deliberately out of scope, and that exclusion is what keeps
    the guard off its own consumers' idioms: both speech adapters build a FAIL
    reason tuple containing the sentence *"a CPU cell's log carries `use gpu = 1`"*.
    That string names the device the checker SAW, in prose, as an argument to
    `schemas.Check` — it is a finding, not a vocabulary, and an audit that forbade a
    checker from naming the device it rejected would be routed around rather than
    obeyed (`feedback_guard_must_not_forbid_its_own_idiom`).
    """
    sites = []
    for node in ast.walk(tree):
        values = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if node.value is not None:
                values = [node.value]
        elif isinstance(node, ast.Compare):
            values = list(node.comparators)
        for value in values:
            for element in _collection_elements(value):
                sites.append((node, element))
    return sites


def _module_identity(tree: ast.AST, *, expected_backend: str,
                     required_functions: Sequence[str]) -> Optional[str]:
    """None when the AST is recognisably the expected consumer; else the reason."""
    backend = None
    defined: set = set()
    for node in getattr(tree, "body", ()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "BACKEND"
                        and isinstance(node.value, ast.Constant)):
                    backend = node.value.value
    if backend != expected_backend:
        return (f"the supplied source declares BACKEND = {backend!r}, not "
                f"{expected_backend!r}, so the AST audited is not this consumer's")
    missing = [name for name in required_functions if name not in defined]
    if missing:
        return (f"the supplied source does not define {missing}, so it is not the "
                f"consumer whose delegation is being audited")
    return None


def _delegating_calls(tree: ast.AST, function_name: str) -> set:
    """Names of `<something>.<attr>` calls made inside `function_name`."""
    calls: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    calls.add(inner.func.attr)
    return calls


def audit_delegates_device_vocabulary(
        source: Optional[str], *, expected_backend: str,
        checker_name: str = "check_device_evidence",
        identity_functions: Sequence[str] = ()) -> schemas.Check:
    """Prove, from a consumer's own AST, that it has no device vocabulary of its own.

    Two rules, and they are complementary — either alone is passable by deleting the
    other's subject:

      1. the consumer's device checker CALLS one of this module's graders
         (`check_device_names` / `check_device_name` / `classify_device_name`); and
      2. no collection literal BOUND to a name — at module level or inside any
         function — and no membership comparator in the consumer's source contains a
         string this vocabulary would classify as a device name. Module scope alone
         was not enough: a table moved inside `check_device_evidence` is the same
         table, and rule 1 is satisfied by a checker that consults its local table
         first and delegates only on the fall-through.

    Rule 2 excludes paths, sonames and globs on purpose: `libggml-cpu.so` carries
    the token `cpu` on a word boundary and the adapters' `EXPECTED_SHARED_LIBRARIES`
    legitimately names it. A guard that forbade its own consumer's necessary idiom
    would be worked around rather than obeyed, which is the failure the package
    keeps re-learning (`feedback_guard_must_not_forbid_its_own_idiom`).

    Returns **COULD_NOT_CHECK on empty, unparsable or foreign source** — the audit
    is a statement about a named consumer, and a clean audit of text nobody bound to
    that consumer (the empty string satisfies every rule) is not evidence about it.
    A FAIL is returned unbound: a local vocabulary is a finding about the text
    whoever the text belongs to.
    """
    if source is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no source supplied; this module reads no file, so the caller passes the "
            "consumer's module text",))
    if not isinstance(source, str):
        raise DeviceVocabularyError("source must be a string")
    if not isinstance(expected_backend, str) or not expected_backend.strip():
        raise DeviceVocabularyError("expected_backend must be a non-empty string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse source: {exc}",))

    findings = []
    for node, element in _vocabulary_sites(tree):
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            if _is_device_name_literal(element.value):
                findings.append(
                    f"line {node.lineno}: a collection literal names the device "
                    f"{element.value!r}. Device names live in `evaluator/devices.py`; "
                    f"a second copy diverges from the first and one of the two then "
                    f"grades a lane by a stale table. Scope makes no difference — a "
                    f"table inside the checker is the same table")
    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))

    identity_problem = _module_identity(tree, expected_backend=expected_backend,
                                        required_functions=tuple(identity_functions))
    if identity_problem is not None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (identity_problem,))

    delegated = _delegating_calls(tree, checker_name)
    graders = {"check_device_names", "check_device_name", "classify_device_name"}
    if checker_name not in {n.name for n in ast.walk(tree)
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the supplied source defines no {checker_name!r}, so there is no device "
            f"checker to audit",))
    if not (delegated & graders):
        return schemas.Check(schemas.FAIL, (
            f"{checker_name}() calls none of {sorted(graders)}; it therefore decides "
            f"what a device name means on its own, which is the divergence this module "
            f"exists to prevent",))
    return schemas.Check(schemas.PASS)
