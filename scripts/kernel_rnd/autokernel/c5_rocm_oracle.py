"""Hardware-free C5 seam for the SOL-ExecBench ROCm correctness oracle.

The external port can compile for the GPU named by ``LOCAL`` and run its live
reference comparison on fresh inputs.  Its numeric artifacts are gfx950/ROCm
7.2 measurements, however, so this seam deliberately exposes no timing or SOL
score surface for gfx90a.  It only seals an audited source/config plan and
validates compile/correctness-only result envelopes.

No function in this module imports torch, touches a GPU, builds code, profiles,
or computes a score.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import c5_seed_corpus


SCHEMA = "epyc.autokernel.c5_rocm_correctness_provider.v1"
PLAN_SCHEMA = "epyc.autokernel.c5_rocm_correctness_plan.v1"
AUDIT_SCHEMA = "epyc.autokernel.c5_rocm_primary_artifact_audit.v1"
RESULT_SCHEMA = "epyc.autokernel.c5_rocm_correctness_result.v1"
PROVIDER_ID = "sol-execbench-rocm-c5-gfx90a-correctness-v1"
TARGET_ARCH = "gfx90a"
TARGET_HARDWARE = "LOCAL"
AUTHORITY = "correctness_oracle_only"
EXPECTED_WORKLOADS = 193
EXPECTED_PROBLEMS = {
    "k138": "L2__044_mamba_discretization_and_segsum",
    "k145": "L2__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block",
    "k154": "L2__060_chunk_gated_delta_rule_linear_attention",
    "k175": "L2__081_moe_sparse_expert_dispatch",
    "k215": "FlashInfer-Bench__006_gemm_n2048_k4096",
    "k225": "FlashInfer-Bench__016_gqa_ragged_prefill_causal_h32_kv4_d128",
    "k227": "FlashInfer-Bench__018_mla_paged_decode_h16_ckv512_kpe64_ps1",
    "k228": "FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
}
EXPECTED_ORACLE_DTYPES = {
    "k138": ("bf16",),
    # The port's tracked workload/tolerance records say fp32.  HyRA's k145
    # solution uses fp16 internally, so these two metadata surfaces must not be
    # collapsed into the inaccurate claim that every oracle workload is 16-bit.
    "k145": ("fp32",),
    "k154": ("bf16",),
    "k175": ("bf16",),
    "k215": ("fp16",),
    "k225": ("bf16",),
    "k227": ("bf16",),
    "k228": ("bf16",),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+._a-zA-Z0-9]*)")
_TOLERANCE_DTYPE_RE = re.compile(r"torch\.(bfloat16|float16|float32) epsilon")
_TOLERANCE_DTYPE = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
_FORBIDDEN_RESULT_KEYS = frozenset({
    "score", "sol_score", "t_sol", "t_sol_ms", "t_sol_cycles", "t_b",
    "t_b_ms", "t_k", "t_k_ms", "latency", "latency_ms", "speedup",
    "throughput", "bandwidth", "flops",
})
_CORRECTNESS_ONLY_ENV = "EPYC_AUTOKERNEL_C5_CORRECTNESS_ONLY"
_EVAL_ANCHOR = """    if _correctness_failed:
        continue

    # -- Monkey-patch defense before timing --
"""
_EVAL_REPLACEMENT = f"""    if _correctness_failed:
        continue

    # EPYC AutoKernel C5: stop at the trusted live-reference boundary.  The
    # port's subsequent timing path depends on part-local constants that do not
    # exist for gfx90a.  Emit correctness with no Performance object and never
    # enter that path when the sealed provider plan requests oracle mode.
    if os.environ.get(\"{_CORRECTNESS_ONLY_ENV}\") == \"1\":
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_workload,
                evaluation=_make_eval(
                    EvaluationStatus.PASSED,
                    _device,
                    None,
                    correctness=_correctness,
                    performance=None,
                    extra_msg=\"EPYC AutoKernel correctness oracle; timing and SOL scoring disabled\",
                ),
            )
        )
        continue

    # -- Monkey-patch defense before timing --
"""


class OracleRefusal(ValueError):
    """The requested surface would exceed correctness-oracle authority."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OracleRefusal(f"{label} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OracleRefusal(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise OracleRefusal(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}")


def _relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise OracleRefusal(f"{label} must be repository-relative")
    return text


@dataclass(frozen=True)
class OracleSeed:
    seed_id: str
    problem_id: str
    workload_count: int
    oracle_workload_dtypes: tuple[str, ...]
    workload_path: str
    workload_sha256: str


@dataclass(frozen=True)
class OracleConfig:
    document: Mapping[str, Any]
    seeds: tuple[OracleSeed, ...]

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.document["source"], "source")

    def select(self, seed_ids: Sequence[str] | None = None) -> tuple[OracleSeed, ...]:
        if seed_ids is None:
            return self.seeds
        selected = tuple(seed_ids)
        if not selected or len(selected) != len(set(selected)):
            raise OracleRefusal("seed selection must be non-empty and unique")
        by_id = {seed.seed_id: seed for seed in self.seeds}
        unknown = sorted(set(selected) - set(by_id))
        if unknown:
            raise OracleRefusal(f"unknown C5 seed ids: {unknown}")
        return tuple(by_id[seed_id] for seed_id in selected)


def _config_path() -> Path:
    return Path(__file__).with_name("c5_rocm_oracle.json")


def _parse_config(document: Mapping[str, Any]) -> OracleConfig:
    _exact_keys(document, {
        "schema", "provider_id", "source", "source_runtime_provenance",
        "target", "correctness", "scoring", "primary_artifacts", "seeds",
    }, "provider config")
    if document["schema"] != SCHEMA or document["provider_id"] != PROVIDER_ID:
        raise OracleRefusal("unexpected provider schema or id")

    source = _mapping(document["source"], "source")
    _exact_keys(source, {"url", "commit", "license"}, "source")
    if source != {
        "url": "https://github.com/williamqwu/sol-execbench-rocm",
        "commit": "7e751eccb8e45a0d0efbcb8c0db8f6eac57a837e",
        "license": "Apache-2.0",
    } or not _COMMIT_RE.fullmatch(str(source["commit"])):
        raise OracleRefusal("provider source identity drifted from the audited pin")

    runtime = _mapping(document["source_runtime_provenance"], "source runtime")
    _exact_keys(runtime, {
        "container_image", "rocm_version", "torch_version", "torch_hip_version",
    }, "source runtime")
    for field in ("container_image", "rocm_version", "torch_version", "torch_hip_version"):
        _text(runtime[field], f"source_runtime_provenance.{field}")
    if runtime["rocm_version"] != "7.2.0" or "rocm7.2" not in runtime["container_image"]:
        raise OracleRefusal("source ROCm 7.2 provenance must remain explicit")

    target = _mapping(document["target"], "target")
    if target != {
        "hardware": TARGET_HARDWARE,
        "architecture": TARGET_ARCH,
        "compile_arch_source": "torch.cuda.get_device_properties(0).gcnArchName",
        "compile_flag": "--offload-arch=gfx90a",
    }:
        raise OracleRefusal("oracle target must stay LOCAL/gfx90a")

    correctness = _mapping(document["correctness"], "correctness")
    if correctness != {
        "enabled": True, "rounds": 10, "fresh_inputs_each_round": True,
        "live_reference_each_round": True, "authority": AUTHORITY,
    }:
        raise OracleRefusal("correctness oracle must keep ten fresh live-reference rounds")

    scoring = _mapping(document["scoring"], "scoring")
    if scoring != {
        "enabled": False, "authority": "none",
        "reason": "no_measured_gfx90a_t_sol_t_b_or_tolerances",
        "source_constant_architectures": ["gfx950"],
        "import_source_constants": False,
    }:
        raise OracleRefusal("SOL scoring must stay disabled with gfx950 constants isolated")
    if TARGET_ARCH in scoring["source_constant_architectures"]:
        raise OracleRefusal("no measured gfx90a constants exist")

    primary = _mapping(document["primary_artifacts"], "primary_artifacts")
    _exact_keys(primary, {
        "manifest_path", "manifest_sha256", "problem_packager_path",
        "problem_packager_sha256", "build_template_path", "build_template_sha256",
        "eval_template_path", "eval_template_sha256",
    }, "primary_artifacts")
    for field in ("manifest", "problem_packager", "build_template", "eval_template"):
        _relative_path(primary[f"{field}_path"], f"primary_artifacts.{field}_path")
        if not _SHA256_RE.fullmatch(str(primary[f"{field}_sha256"])):
            raise OracleRefusal(f"primary {field} digest must be a lowercase SHA-256")

    rows = document["seeds"]
    if not isinstance(rows, list):
        raise OracleRefusal("seeds must be a list")
    parsed: list[OracleSeed] = []
    for row_value in rows:
        row = _mapping(row_value, "seed")
        _exact_keys(row, {
            "seed_id", "problem_id", "workload_count", "oracle_workload_dtypes",
            "workload_path", "workload_sha256",
        }, "seed")
        seed_id = _text(row["seed_id"], "seed.seed_id")
        count = row["workload_count"]
        dtypes = row["oracle_workload_dtypes"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise OracleRefusal(f"{seed_id}: workload_count must be positive")
        if not isinstance(dtypes, list) or not dtypes or len(dtypes) != len(set(dtypes)):
            raise OracleRefusal(f"{seed_id}: oracle dtypes must be a unique non-empty list")
        dtypes_tuple = tuple(_text(item, f"{seed_id}.dtype") for item in dtypes)
        if seed_id not in EXPECTED_PROBLEMS:
            raise OracleRefusal(f"unknown seed id {seed_id!r}")
        if row["problem_id"] != EXPECTED_PROBLEMS[seed_id]:
            raise OracleRefusal(f"{seed_id}: SOL-ExecBench problem join drifted")
        if dtypes_tuple != EXPECTED_ORACLE_DTYPES[seed_id]:
            raise OracleRefusal(f"{seed_id}: oracle workload dtype evidence drifted")
        digest = _text(row["workload_sha256"], f"{seed_id}.workload_sha256")
        if not _SHA256_RE.fullmatch(digest):
            raise OracleRefusal(f"{seed_id}: workload digest must be a lowercase SHA-256")
        parsed.append(OracleSeed(
            seed_id=seed_id, problem_id=str(row["problem_id"]), workload_count=count,
            oracle_workload_dtypes=dtypes_tuple,
            workload_path=_relative_path(row["workload_path"], f"{seed_id}.workload_path"),
            workload_sha256=digest,
        ))
    ids = tuple(seed.seed_id for seed in parsed)
    if ids != c5_seed_corpus.EXPECTED_SEED_IDS:
        raise OracleRefusal("oracle seed ids/order differ from the C5 corpus")
    if sum(seed.workload_count for seed in parsed) != EXPECTED_WORKLOADS:
        raise OracleRefusal("oracle workload population must total 193")

    # Keep HyRA candidate metadata and the port's actual oracle-workload dtype
    # evidence distinct.  In particular k145 and k227 are not identical.
    corpus = c5_seed_corpus.load()
    if tuple(seed.seed_id for seed in corpus.seeds) != ids:
        raise OracleRefusal("HyRA C5 corpus and oracle provider seed join differ")
    return OracleConfig(document=json.loads(json.dumps(document)), seeds=tuple(parsed))


def load(path: str | Path | None = None) -> OracleConfig:
    config_path = _config_path() if path is None else Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OracleRefusal("ROCm correctness-provider config is not valid JSON") from exc
    return _parse_config(_mapping(document, "provider config"))


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args), text=True, capture_output=True, check=False)
    if completed.returncode:
        raise OracleRefusal(
            f"provider source git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _workload_dtypes(path: Path) -> tuple[tuple[str, ...], int]:
    seen: set[str] = set()
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OracleRefusal(f"invalid workload JSONL in {path}") from exc
        tolerance = _mapping(_mapping(row, "workload").get("tolerance"), "tolerance")
        provenance = _text(tolerance.get("_provenance"), "tolerance._provenance")
        match = _TOLERANCE_DTYPE_RE.search(provenance)
        if match is None:
            raise OracleRefusal(f"workload dtype provenance missing in {path}")
        seen.add(_TOLERANCE_DTYPE[match.group(1)])
    return tuple(sorted(seen)), count


def audit_primary_artifacts(source_root: str | Path,
                            config: OracleConfig | None = None) -> dict[str, Any]:
    """Verify the pinned port checkout and the eight tracked primary records."""
    oracle = load() if config is None else config
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir() or not (root / ".git").exists():
        raise OracleRefusal("provider source must be an existing Git checkout")
    commit = str(oracle.source["commit"])
    if _git(root, "rev-parse", "HEAD") != commit:
        raise OracleRefusal("provider source HEAD differs from the audited commit")
    if _git(root, "status", "--porcelain"):
        raise OracleRefusal("provider source checkout is not clean")

    primary = _mapping(oracle.document["primary_artifacts"], "primary_artifacts")
    manifest_path = (root / str(primary["manifest_path"])).resolve(strict=True)
    if not manifest_path.is_relative_to(root) or _file_sha256(manifest_path) != primary["manifest_sha256"]:
        raise OracleRefusal("provider primary manifest identity drifted")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    problems = _mapping(manifest.get("problems"), "manifest.problems")

    provider_code = {}
    for field in ("problem_packager", "build_template", "eval_template"):
        path = (root / str(primary[f"{field}_path"])).resolve(strict=True)
        expected = str(primary[f"{field}_sha256"])
        if not path.is_relative_to(root) or _file_sha256(path) != expected:
            raise OracleRefusal(f"provider {field} identity drifted")
        provider_code[field] = {"path": str(primary[f"{field}_path"]), "sha256": expected}

    seed_rows = []
    for seed in oracle.seeds:
        workload_path = (root / seed.workload_path).resolve(strict=True)
        if not workload_path.is_relative_to(root) or _file_sha256(workload_path) != seed.workload_sha256:
            raise OracleRefusal(f"{seed.seed_id}: workload artifact identity drifted")
        dtypes, count = _workload_dtypes(workload_path)
        problem = _mapping(problems.get(seed.problem_id), f"manifest.{seed.problem_id}")
        deferred = problem.get("deferred")
        if (
            count != seed.workload_count
            or problem.get("n_workloads") != seed.workload_count
            or problem.get("n_scoreable") != seed.workload_count
            or deferred not in (None, [])
        ):
            raise OracleRefusal(f"{seed.seed_id}: workload/scoreable population drifted")
        if dtypes != tuple(sorted(seed.oracle_workload_dtypes)):
            raise OracleRefusal(f"{seed.seed_id}: workload dtype evidence drifted")
        seed_rows.append({
            "seed_id": seed.seed_id, "problem_id": seed.problem_id,
            "workload_count": count, "oracle_workload_dtypes": list(dtypes),
            "workload_sha256": seed.workload_sha256,
        })
    receipt = {
        "schema": AUDIT_SCHEMA, "provider_id": PROVIDER_ID,
        "source_root": str(root), "source_commit": commit,
        "manifest_sha256": primary["manifest_sha256"],
        "provider_code": provider_code,
        "seed_count": len(seed_rows), "workload_count": sum(
            row["workload_count"] for row in seed_rows),
        "seeds": seed_rows, "hardware_accessed": False,
        "build_executed": False, "scoring_constants_imported": False,
        "authority": AUTHORITY,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _render_correctness_driver(root: Path, oracle: OracleConfig) -> str:
    primary = _mapping(oracle.document["primary_artifacts"], "primary_artifacts")
    path = (root / str(primary["eval_template_path"])).resolve(strict=True)
    source = path.read_text(encoding="utf-8")
    if source.count(_EVAL_ANCHOR) != 1:
        raise OracleRefusal("provider evaluation template correctness/timing boundary drifted")
    rendered = source.replace(_EVAL_ANCHOR, _EVAL_REPLACEMENT)
    if rendered.count(_CORRECTNESS_ONLY_ENV) != 1:
        raise OracleRefusal("correctness-only driver overlay was not applied exactly once")
    return rendered


def stage_correctness_driver(
    source_root: str | Path, staging_root: str | Path,
    config: OracleConfig | None = None,
) -> dict[str, Any]:
    """Write the audited no-timing eval driver into an existing staging root.

    This is the only provider overlay.  It leaves the source checkout untouched
    and overwrites only the byte-exact eval template emitted by
    ``ProblemPackager.execute()``.  The caller applies it after the packager has
    prepared the other inputs but before any evaluation process starts.
    """
    oracle = load() if config is None else config
    source = Path(source_root).resolve(strict=True)
    audit = audit_primary_artifacts(source, oracle)
    staging = Path(staging_root).resolve(strict=True)
    if not staging.is_dir() or staging == source or staging.is_relative_to(source):
        raise OracleRefusal("correctness-driver staging root must exist outside provider source")
    destination = staging / "eval_driver.py"
    primary = _mapping(oracle.document["primary_artifacts"], "primary_artifacts")
    replaced_packager_template = destination.exists()
    if replaced_packager_template and (
        not destination.is_file()
        or _file_sha256(destination) != primary["eval_template_sha256"]
    ):
        raise OracleRefusal("staged eval_driver.py differs from the audited packager template")
    rendered = _render_correctness_driver(source, oracle)
    with destination.open("w" if replaced_packager_template else "x", encoding="utf-8") as handle:
        handle.write(rendered)
    receipt = {
        "schema": "epyc.autokernel.c5_rocm_correctness_driver_stage.v1",
        "provider_id": PROVIDER_ID, "source_audit_sha256": audit["receipt_sha256"],
        "destination": str(destination), "driver_sha256": _file_sha256(destination),
        "required_environment": {_CORRECTNESS_ONLY_ENV: "1"},
        "replaced_packager_template": replaced_packager_template,
        "timing_path_reachable": False, "sol_scoring_path_reachable": False,
        "hardware_accessed": False, "build_executed": False, "authority": AUTHORITY,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _runtime_provenance(value: Mapping[str, Any]) -> dict[str, str]:
    runtime = _mapping(value, "runtime_provenance")
    _exact_keys(runtime, {
        "rocm_version", "torch_version", "torch_hip_version", "driver_version",
    }, "runtime_provenance")
    observed = {key: _text(runtime[key], f"runtime_provenance.{key}") for key in runtime}
    for field in ("rocm_version", "torch_hip_version", "driver_version"):
        if not _VERSION_RE.fullmatch(observed[field]):
            raise OracleRefusal(f"runtime_provenance.{field} is not an exact version")
    return observed


def compile_plan(
    source_root: str | Path, *, runtime_provenance: Mapping[str, Any],
    seed_ids: Sequence[str] | None = None, config: OracleConfig | None = None,
) -> dict[str, Any]:
    """Seal a compile+correctness plan; never authorize timing or scoring."""
    oracle = load() if config is None else config
    audit = audit_primary_artifacts(source_root, oracle)
    source = Path(source_root).resolve(strict=True)
    correctness_driver = _render_correctness_driver(source, oracle)
    runtime = _runtime_provenance(runtime_provenance)
    selected = oracle.select(seed_ids)
    hyra_by_id = {seed.seed_id: seed for seed in c5_seed_corpus.load().seeds}
    source_runtime = dict(_mapping(
        oracle.document["source_runtime_provenance"], "source runtime"))
    plan = {
        "schema": PLAN_SCHEMA, "provider_id": PROVIDER_ID,
        "source_audit": audit,
        "runtime_provenance": runtime,
        "source_runtime_provenance": source_runtime,
        "runtime_compatibility": {
            "rocm_exact_match": runtime["rocm_version"] == source_runtime["rocm_version"],
            "compatibility_claimed": False,
            "authority": "record_only_no_cross_version_inference",
        },
        "target": dict(_mapping(oracle.document["target"], "target")),
        "operations": {
            "compile": True, "correctness": True,
            "correctness_rounds": 10, "fresh_inputs_each_round": True,
            "live_reference_each_round": True,
            "timing": False, "profiling": False, "sol_scoring": False,
        },
        "execution_seam": {
            "staged_driver": "eval_driver.py",
            "driver_sha256": hashlib.sha256(correctness_driver.encode("utf-8")).hexdigest(),
            "required_environment": {_CORRECTNESS_ONLY_ENV: "1"},
            "timing_path_reachable": False, "sol_scoring_path_reachable": False,
        },
        "corpus": {
            "seed_count": len(selected),
            "workload_count": sum(seed.workload_count for seed in selected),
            "seeds": [{
                "seed_id": seed.seed_id, "problem_id": seed.problem_id,
                "workload_count": seed.workload_count,
                "oracle_workload_dtypes": list(seed.oracle_workload_dtypes),
                "hyra_reference_dtypes": list(hyra_by_id[seed.seed_id].dtypes),
            } for seed in selected],
        },
        "scoring": dict(_mapping(oracle.document["scoring"], "scoring")),
        "authority": AUTHORITY,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _reject_numeric_claim_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_RESULT_KEYS:
                raise OracleRefusal(f"correctness-only result cannot carry {key!r}")
            _reject_numeric_claim_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_numeric_claim_keys(child)


def validate_result(result: Mapping[str, Any], *, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an unscored provider result and refuse every numeric claim path."""
    row = _mapping(result, "result")
    _reject_numeric_claim_keys(row)
    _exact_keys(row, {
        "schema", "provider_id", "plan_sha256", "target", "runtime_provenance",
        "authority", "seed_results",
    }, "result")
    if (
        row["schema"] != RESULT_SCHEMA
        or row["provider_id"] != PROVIDER_ID
        or row["plan_sha256"] != plan.get("plan_sha256")
        or row["target"] != {"hardware": TARGET_HARDWARE, "architecture": TARGET_ARCH}
        or row["runtime_provenance"] != plan.get("runtime_provenance")
        or row["authority"] != AUTHORITY
    ):
        raise OracleRefusal("result identity/authority differs from its plan")
    planned = {seed["seed_id"]: seed for seed in plan["corpus"]["seeds"]}
    seed_results = row["seed_results"]
    if not isinstance(seed_results, list):
        raise OracleRefusal("seed_results must be a list")
    seen: set[str] = set()
    for value in seed_results:
        seed = _mapping(value, "seed_result")
        _exact_keys(seed, {
            "seed_id", "compile_status", "correctness_status",
            "correctness_rounds_run", "workloads_checked", "error",
        }, "seed_result")
        seed_id = _text(seed["seed_id"], "seed_result.seed_id")
        if seed_id in seen or seed_id not in planned:
            raise OracleRefusal(f"unexpected or duplicate seed result {seed_id!r}")
        seen.add(seed_id)
        if seed["compile_status"] not in {"passed", "failed"}:
            raise OracleRefusal(f"{seed_id}: invalid compile status")
        if seed["correctness_status"] not in {"passed", "failed", "not_run"}:
            raise OracleRefusal(f"{seed_id}: invalid correctness status")
        rounds = seed["correctness_rounds_run"]
        checked = seed["workloads_checked"]
        if (
            isinstance(rounds, bool) or not isinstance(rounds, int) or not 0 <= rounds <= 10
            or isinstance(checked, bool) or not isinstance(checked, int)
            or not 0 <= checked <= planned[seed_id]["workload_count"]
        ):
            raise OracleRefusal(f"{seed_id}: invalid correctness coverage")
        if seed["correctness_status"] == "passed" and (
            seed["compile_status"] != "passed" or rounds != 10
            or checked != planned[seed_id]["workload_count"]
        ):
            raise OracleRefusal(f"{seed_id}: correctness pass lacks full ten-round coverage")
        if seed["correctness_status"] == "not_run" and (rounds or checked):
            raise OracleRefusal(f"{seed_id}: not_run cannot claim coverage")
        if seed["error"] is not None and not isinstance(seed["error"], str):
            raise OracleRefusal(f"{seed_id}: error must be text or null")
    if seen != set(planned):
        raise OracleRefusal("result does not cover every planned seed")
    return json.loads(json.dumps(row))


__all__ = [
    "AUDIT_SCHEMA", "AUTHORITY", "EXPECTED_WORKLOADS", "OracleConfig",
    "OracleRefusal", "OracleSeed", "PLAN_SCHEMA", "PROVIDER_ID", "RESULT_SCHEMA",
    "SCHEMA", "TARGET_ARCH", "audit_primary_artifacts", "compile_plan", "load",
    "stage_correctness_driver", "validate_result",
]
