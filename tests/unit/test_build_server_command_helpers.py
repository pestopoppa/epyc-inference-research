"""Tests for the per-mode helpers that back build_server_command()."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from scripts.server import orchestrator_stack as oss
from scripts.server import stack_commands


def _stack_prior_role(role: str) -> dict[str, Any]:
    priors_path = oss._PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"
    payload = yaml.safe_load(priors_path.read_text(encoding="utf-8"))
    role_record = payload["roles"][role]
    assert isinstance(role_record, dict)
    return role_record


def _write_launch_prior(
    tmp_path: Path,
    role: str,
    *,
    requirements: dict[str, Any],
    runtime: dict[str, Any],
) -> Path:
    path = tmp_path / "stack_priors.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "roles": {
                    role: {
                        "deployment_status": "live_stack",
                        "serving": {
                            "launch": {
                                "requirements": requirements,
                                "runtime": runtime,
                            }
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _flag_value(cmd: list[str], flag: str) -> str | None:
    if flag not in cmd:
        return None
    idx = cmd.index(flag)
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def _all_flag_values(cmd: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for idx, value in enumerate(cmd):
        if value == flag and idx + 1 < len(cmd):
            values.append(cmd[idx + 1])
    return values


def _optional_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: str | None) -> float | None:
    return float(value) if value is not None else None


def _command_runtime_signature(cmd: list[str]) -> dict[str, Any]:
    spec_enabled = "-md" in cmd or "--spec-type" in cmd
    draft_model_path = _flag_value(cmd, "-md")
    if draft_model_path is None and spec_enabled:
        draft_model_path = _flag_value(cmd, "-m")
    return {
        "binary_path": cmd[0],
        "cache": {
            "context_tokens": _optional_int(_flag_value(cmd, "-c")),
            "slots": _optional_int(_flag_value(cmd, "-np")),
            "ubatch": _optional_int(_flag_value(cmd, "-ub")),
            "kv_type_k": _flag_value(cmd, "-ctk"),
            "kv_type_v": _flag_value(cmd, "-ctv"),
            "no_mmap": "--no-mmap" in cmd,
            "mlock": "--mlock" in cmd,
            "slot_save_path": _flag_value(cmd, "--slot-save-path"),
        },
        "flags": {
            "device": _flag_value(cmd, "--device"),
            "flash_attn": _flag_value(cmd, "--flash-attn") == "on",
            "jinja": "--jinja" in cmd,
            "reasoning": _flag_value(cmd, "--reasoning"),
            "override_kv": sorted(_all_flag_values(cmd, "--override-kv")),
            "spec": {
                "enabled": spec_enabled,
                # 2026-06-26 v6 cutover: --spec-type token is now 'draft-mtp' (the
                # bare 'mtp' token was removed); flag name itself is unchanged.
                "type": _flag_value(cmd, "--spec-type") if spec_enabled else None,
                "draft_model_path": draft_model_path if spec_enabled else None,
                # 2026-06-26 v6 cutover: n-max value now carried by --spec-draft-n-max
                # (v6 removed --draft-max); same schema field 'draft_max'.
                "draft_max": (
                    _optional_int(_flag_value(cmd, "--spec-draft-n-max")) if spec_enabled else None
                ),
                "draft_p_min": (
                    _optional_float(_flag_value(cmd, "--draft-p-min"))
                    if spec_enabled
                    else None
                ),
                "threads_draft": (
                    _optional_int(_flag_value(cmd, "--threads-draft"))
                    if spec_enabled
                    else None
                ),
            },
        },
    }


def _stack_prior_runtime_signature(runtime: dict[str, Any]) -> dict[str, Any]:
    cache = runtime["cache"]
    flags = runtime["flags"]
    spec = flags["spec"]
    return {
        "binary_path": runtime["binary_path"],
        "cache": {
            "context_tokens": cache["context_tokens"],
            "slots": cache["slots"],
            "ubatch": cache["ubatch"],
            "kv_type_k": cache["kv_type_k"],
            "kv_type_v": cache["kv_type_v"],
            "no_mmap": cache["no_mmap"],
            "mlock": cache["mlock"],
            "slot_save_path": cache["slot_save_path"],
        },
        "flags": {
            "device": flags.get("device"),
            "flash_attn": flags["flash_attn"],
            "jinja": flags["jinja"],
            "reasoning": flags["reasoning"],
            "override_kv": sorted(flags["override_kv"]),
            "spec": {
                "enabled": spec["enabled"],
                "type": spec["type"],
                "draft_model_path": spec["draft_model_path"],
                "draft_max": spec["draft_max"],
                "draft_p_min": spec["draft_p_min"],
                "threads_draft": spec["threads_draft"],
            },
        },
    }


def _assert_detached_popen(popen) -> None:
    kwargs = popen.call_args.kwargs
    assert kwargs["stdin"] is oss.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True


def test_descriptor_active_roles_are_canonical_launch_roles() -> None:
    """Canonical launch roles are the HOSTS; aliases compile through their host.

    2026-09-22 lineup cutover (860b0b2d): worker_general stopped being its own
    CPU server and became an alias on frontdoor's :8070 process
    (``server_mode.frontdoor.shared_with``). It therefore must NOT appear as a
    canonical launch role any more — ``write_model_descriptors()`` expands the
    alias from the registry on its own. The alias set is recomputed from the
    master declaration rather than restated, so the next alias move lands here
    without an edit.
    """
    from scripts.server import stack_manifest

    active_roles = stack_commands._descriptor_active_roles()

    # worker_general resolves to its host through master's shared_with binding,
    # and that host IS a canonical launch role.
    host, _row, binding = stack_manifest.master_server_row("worker_general")
    assert binding == "shared_with"
    assert host == "frontdoor"
    assert host in active_roles
    assert "worker_general" not in active_roles

    # No alias of a live host may be a canonical launch role (a second launch
    # record for one process is the defect the alias binding removes).
    aliases = {
        alias
        for name, cfg in stack_manifest.MASTER_SERVER_MODE.items()
        if name in active_roles and isinstance(cfg.get("shared_with"), list)
        for alias in cfg["shared_with"]
    }
    assert "worker_general" in aliases  # non-vacuous
    assert not (aliases & active_roles)

    assert "architect_general" in active_roles
    assert "worker_explore" not in active_roles
    assert "architect_coding" not in active_roles  # stack-change-guard: allow legacy retired-role coverage


# -----------------------------------------------------------------------------
# Vision / embedding / dev / worker-mode shape regressions
# -----------------------------------------------------------------------------


def test_build_vision_command_escalation_uses_worker_vision_model() -> None:
    """2026-08-01 W1 cutover — renamed from
    ``test_build_vision_command_escalation_uses_worker_vision_safety_alias``.

    vision_escalation is no longer a "temporary safety alias" pending a
    higher-quality replacement: it is a routing label on worker_vision's ONE
    :8086 MI210 process (Qwen3-VL-30B-A3B-Instruct Q4_K_M). Nothing launches this
    branch any more (ROLE_LAUNCH_META has no vision_escalation entry), but the
    branch is still live code, so its shape stays pinned here.
    """
    cmd = oss._build_vision_command(port=8087, vision_type="escalation")
    assert oss.VISION_ESCALATION_MODEL == oss.VISION_WORKER_MODEL
    assert oss.VISION_ESCALATION_MMPROJ == oss.VISION_WORKER_MMPROJ
    # Was Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf / mmproj-model-f16.gguf.
    assert "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf" in oss.VISION_ESCALATION_MODEL
    assert "mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf" in oss.VISION_ESCALATION_MMPROJ
    assert oss.VISION_ESCALATION_MODEL in cmd
    assert oss.VISION_ESCALATION_MMPROJ in cmd
    assert "--mmproj" in cmd
    # Was "none" (CPU-only 7B lane), then ROCm0 (MI210, 2026-08-01). STACKCHG-DFLASH2-20261003: the VL
    # process is a COLD CPU role while the MI210 carries DFlash2 -> "none" again.
    assert _flag_value(cmd, "--device") == "none"
    assert _flag_value(cmd, "--reasoning") == "off"
    # Still no expert-count override: the registry corrected int:4 to "use the
    # GGUF default of 8" on 2026-07-31.
    assert "--override-kv" not in cmd
    # 8192 -> 16384 (2026-08-01 W1) -> 65536 (2026-08-02, operator-ratified).
    # DERIVED now, not declared launcher-side: server_mode.worker_vision
    # .serving_shape.n_ctx, which vision_escalation inherits through shared_with.
    # 65536 / -np 4 = 16384 per slot, i.e. the same per-slot depth the MMMU-250
    # cutover measured — the total grew, the per-request context did not.
    assert cmd[cmd.index("-c") + 1] == "65536"
    # Was "24", from NUMA_CONFIG["vision_escalation"]. That entry was DELETED by
    # the cutover (the role launches nothing), so _resolve_thread_count now falls
    # through to its unknown-role default of 96. Pinned deliberately: if anyone
    # re-rosters vision_escalation as a process, this flips and demands a review.
    assert cmd[cmd.index("-t") + 1] == "96"
    assert "vision_escalation" not in oss.NUMA_CONFIG


def test_build_vision_command_worker_uses_stack_prior_shape() -> None:
    """2026-08-01 W1 cutover — renamed from
    ``test_build_vision_command_worker_uses_small_model``: the VL worker is no
    longer the small model (Qwen2.5-VL-7B, 4.4 GB CPU) but Qwen3-VL-30B-A3B
    Q4_K_M on MI210."""
    cmd = oss._build_vision_command(port=8086, vision_type="worker")
    assert oss.VISION_WORKER_MODEL in cmd
    assert oss.VISION_WORKER_MMPROJ in cmd
    # 8192 -> 16384 (measured shape) -> 65536 (2026-08-02, operator-ratified,
    # DERIVED from server_mode.worker_vision.serving_shape.n_ctx). Per-slot depth
    # is unchanged at 65536/-np 4 = 16384.
    assert cmd[cmd.index("-c") + 1] == "65536"
    # -t was "24", the NPS4 node-1 CPU quarter this role held while it was the
    # 4.4 GB Qwen2.5-VL-7B. The 2026-08-01 W1 cutover moved worker_vision onto
    # the MI210, and a GPU-resident role takes the SHARED GPU host lane, not a
    # CPU quarter — its host threads only tokenise, sample and marshal.
    #
    # There is no longer a disagreement about which host shape the VL role owns;
    # all four declarations agree, so assert against them rather than a literal:
    #   * stack_numa.GPU_HOST_LANE — canonical BECAUSE MEASURED: the MMMU-250
    #     cutover that chose this model ran `taskset -c 184-191 ... -t 8`.
    #   * NUMA_CONFIG["worker_vision"] — gpu_host_lane, cpuset 184-191.
    #   * the compiled stack prior — launch entry cpu_shape_class gpu_host_lane.
    # If anyone re-rosters the VL role onto a CPU quarter, all three must move
    # together or this fails.
    # STACKCHG-DFLASH2-20261003: re-rostered, as this test demands, all together: the VL role is a COLD
    # CPU role on NUMA_HALF_A (0-47,96-143, -t 48) while the MI210 carries DFlash2.
    from scripts.server.stack_numa import NUMA_HALF_A

    half_cpus, half_threads = NUMA_HALF_A
    assert "gpu_host_lane" not in oss.NUMA_CONFIG["worker_vision"]
    assert oss.NUMA_CONFIG["worker_vision"]["instances"][0][0] == half_cpus
    assert {
        entry["cpu_shape_class"]
        for entry in _stack_prior_role("worker_vision")["serving"]["launch"]["entries"]
    } == {"half"}
    assert cmd[cmd.index("-t") + 1] == str(half_threads)
    assert _flag_value(cmd, "--device") == "none"


@pytest.mark.parametrize(
    ("role", "port", "vision_type"),
    [
        ("worker_vision", 8086, "worker"),
        # 2026-08-01 W1 cutover: the ("vision_escalation", 8087, "escalation")
        # case is VOID, not skipped. vision_escalation has no ROLE_LAUNCH_META
        # entry and no NUMA_CONFIG entry any more, so no launch witness exists for
        # it to be compared against — its stack-prior record is worker_vision's
        # entry with alias=True and vision_type="worker", already covered by the
        # row above. Port 8087 is retired.
    ],
)
def test_build_vision_command_matches_stack_prior_launch_witness(
    role: str,
    port: int,
    vision_type: str,
) -> None:
    role_record = _stack_prior_role(role)
    launch = role_record["serving"]["launch"]
    requirements = launch["requirements"]
    runtime = launch["runtime"]

    cmd = oss._build_vision_command(port=port, vision_type=vision_type)

    assert _flag_value(cmd, "-m") == requirements["model_path"]
    assert _flag_value(cmd, "--mmproj") == requirements["mmproj_path"]
    assert _command_runtime_signature(cmd) == _stack_prior_runtime_signature(runtime)


def test_build_vision_command_prefers_stack_prior_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    priors = _write_launch_prior(
        tmp_path,
        "vision_escalation",
        requirements={
            "model_path": "/prior/vision.gguf",
            "mmproj_path": "/prior/mmproj.gguf",
        },
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {"context_tokens": 12000, "slots": 1, "no_mmap": True},
            "flags": {
                "flash_attn": False,
                "device": "ROCm0",
                "reasoning": "off",
                "override_kv": ["qwen3vlmoe.expert_used_count=int:2"],
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)

    cmd = oss._build_vision_command(port=9087, vision_type="escalation")

    assert cmd[0] == "/prior/llama-server"
    assert _flag_value(cmd, "-m") == "/prior/vision.gguf"
    assert _flag_value(cmd, "--mmproj") == "/prior/mmproj.gguf"
    assert _all_flag_values(cmd, "--override-kv") == [
        "qwen3vlmoe.expert_used_count=int:2"
    ]
    assert _flag_value(cmd, "-np") == "1"
    assert _flag_value(cmd, "-c") == "12000"
    assert _flag_value(cmd, "--device") == "ROCm0"
    assert _flag_value(cmd, "--reasoning") == "off"
    assert "--no-mmap" in cmd
    assert "--flash-attn" not in cmd


def test_dispatcher_resolves_vision_escalation_to_worker_vision_gpu_process() -> None:
    """2026-08-01 W1 cutover — REPLACES
    ``test_dispatcher_uses_cpu_device_for_temporary_vision_escalation_alias``.

    That test asserted ``--device none`` on a dedicated :8087 CPU escalation
    server. The concept is void: the cutover deleted that server, so there is no
    CPU device left to assert. The surviving invariant — where does
    vision_escalation actually resolve to — is asserted instead, with the device
    check kept at identical strength and its polarity flipped (ROCm0 in, "none"
    out).
    """
    # vision_escalation resolves onto worker_vision's process, not one of its own.
    assert oss.PORT_MAP["vision_escalation"] == oss.PORT_MAP["worker_vision"] == 8086
    assert "vision_escalation" not in oss.ROLE_LAUNCH_META  # launches nothing
    assert "vision_escalation" not in oss.NUMA_CONFIG  # no CPU wiring of its own
    assert oss.ROLE_LAUNCH_META["worker_vision"]["shared_with_first_n"] == [
        "vision_escalation"
    ]

    cmd = oss.build_server_command(None, 8086, vision_mode=True, vision_type="worker")

    # STACKCHG-DFLASH2-20261003: the shared :8086 process is a COLD CPU role -> exactly one `--device none`,
    # never a GPU device (the dispatcher must not re-add ROCm0 behind the builder).
    assert _all_flag_values(cmd, "--device") == ["none"]


def test_build_embedding_command_enables_embeddings_and_cls_pool() -> None:
    cmd = oss._build_embedding_command(port=8090)
    assert "--embeddings" in cmd
    assert "--pooling" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "cls"
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[cmd.index("-t") + 1] == "4"


def test_build_embedding_command_uses_warm_candidate_recipes() -> None:
    granite = oss._build_embedding_command(port=8096)
    assert granite[granite.index("-m") + 1].endswith(
        "granite-embedding-97m-multilingual-r2-Q8_0.gguf"
    )
    assert granite[granite.index("-c") + 1] == "32768"
    assert granite[granite.index("-t") + 1] == "16"
    assert granite[granite.index("--pooling") + 1] == "cls"

    e5 = oss._build_embedding_command(port=8097)
    assert e5[e5.index("-m") + 1].endswith("multilingual-e5-base-Q8_0.gguf")
    assert e5[e5.index("-c") + 1] == "512"
    assert e5[e5.index("--pooling") + 1] == "mean"

    bge_m3 = oss._build_embedding_command(port=8098)
    assert bge_m3[bge_m3.index("-m") + 1].endswith("bge-m3-Q8_0.gguf")
    assert bge_m3[bge_m3.index("-c") + 1] == "8192"
    assert bge_m3[bge_m3.index("--pooling") + 1] == "cls"


def test_build_dev_command_short_context_small_threads() -> None:
    cmd = oss._build_dev_command(port=9999)
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("-t") + 1] == "16"
    assert "--flash-attn" in cmd


def test_build_worker_fast_command_uses_4_slots() -> None:
    cmd = oss._build_worker_fast_command(port=8102, model_path="/m/fast.gguf")
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[cmd.index("-m") + 1] == "/m/fast.gguf"
    assert cmd[cmd.index("-c") + 1] == "16384"


def test_build_worker_general_command_engages_mtp_path() -> None:
    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    # MTP-specific flags must all be present. The broad worker default remains
    # native MTP; ngram-mod,draft-mtp is a task-specific candidate lane.
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "2"
    assert cmd[cmd.index("--draft-p-min") + 1] == "0.0"
    assert cmd[cmd.index("--threads-draft") + 1] == "16"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"
    assert "--no-mmap" in cmd
    assert "--jinja" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"


def test_worker_general_launch_witness_is_its_host_primary_role_command() -> None:
    """Renamed from ``test_build_worker_general_command_matches_stack_prior_launch_witness``.

    Since the 2026-09-22 lineup cutover (860b0b2d) worker_general launches NOTHING:
    it is an alias on frontdoor's :8070 process, has no role_launch_meta entry, and
    its compiled prior is an ALIAS record (every entry ``alias: true``, the host in
    ``primary_roles``, spec disabled because aliases inherit the host's NEXTN draft
    rather than launching their own — see ``src/registry/stack_priors.py``). The
    command that actually serves worker_general is the host's
    ``_build_role_command``, so THAT is what must match the compiled witness.

    The old ``-md == draft_model_path`` assertion was also stale on its own terms:
    Qwen3.6-35B-A3B-MTP is a NEXTN self-draft (draft path == model path), and the
    launcher deliberately suppresses same-realpath ``-md`` while keeping
    ``--spec-type``/``--spec-draft-n-max``.
    """
    alias_launch = _stack_prior_role("worker_general")["serving"]["launch"]
    entries = alias_launch["entries"]
    assert entries and all(entry["alias"] is True for entry in entries)
    (host,) = alias_launch["primary_roles"]
    assert {entry["primary_role"] for entry in entries} == {host}
    assert alias_launch["runtime"]["flags"]["spec"]["enabled"] is False

    host_launch = _stack_prior_role(host)["serving"]["launch"]
    requirements = host_launch["requirements"]
    runtime = host_launch["runtime"]
    # The alias serves the host's model — one process, one GGUF.
    assert alias_launch["requirements"]["model_path"] == requirements["model_path"]
    primary_entry = next(
        entry for entry in host_launch["entries"] if entry["numa_instance"] == 0
    )
    assert primary_entry["alias"] is False
    assert primary_entry["port"] in {entry["port"] for entry in entries}

    role_config = oss.RegistryLoader().get_role(host)
    cmd = oss._build_role_command(
        role_config, primary_entry["port"], 0, prepare_runtime_dirs=False
    )

    assert _flag_value(cmd, "-m") == requirements["model_path"]
    if runtime["flags"]["spec"]["draft_model_path"] == requirements["model_path"]:
        assert "-md" not in cmd  # NEXTN self-draft: same-file -md suppressed
    else:
        assert _flag_value(cmd, "-md") == runtime["flags"]["spec"]["draft_model_path"]
    assert _command_runtime_signature(cmd) == _stack_prior_runtime_signature(runtime)


def test_build_worker_general_command_prefers_stack_prior_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    priors = _write_launch_prior(
        tmp_path,
        "worker_general",
        requirements={
            "model_path": "/prior/gemma.gguf",
            "draft_model_path": "/prior/draft.gguf",
        },
        runtime={
            # 2026-06-26 v6 cutover: worker now runs on canonical llama.cpp (v6),
            # not a separate ik build.
            "binary_path": "/prior/llama.cpp/llama-server",
            "cache": {
                "context_tokens": 12288,
                "slots": 1,
                "ubatch": 256,
                "kv_type_k": "q5_0",
                "kv_type_v": "q6_0",
                "no_mmap": False,
            },
            "flags": {
                "flash_attn": False,
                "jinja": False,
                "reasoning": "off",
                "override_kv": [],
                "spec": {
                    "enabled": True,
                    # 2026-06-26 v6 cutover: spec type token is now 'draft-mtp'.
                    "type": "draft-mtp",
                    "draft_model_path": "/prior/draft.gguf",
                    "draft_max": 4,
                    "draft_min": 1,
                    "draft_p_min": 0.25,
                    "draft_p_split": 0.75,
                    "threads_draft": 8,
                    "ngram_mod_n_min": 0,
                    "ngram_mod_n_max": 96,
                    "ngram_mod_n_match": 12,
                },
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)

    cmd = oss._build_worker_general_command(
        port=8072,
        model_path="/fallback/gemma.gguf",
        binary_override=None,
    )

    assert cmd[0] == "/prior/llama.cpp/llama-server"
    assert _flag_value(cmd, "-m") == "/prior/gemma.gguf"
    assert _flag_value(cmd, "-md") == "/prior/draft.gguf"
    # 2026-06-26 v6 cutover: n-max emitted via --spec-draft-n-max (was --draft-max).
    assert _flag_value(cmd, "--spec-draft-n-max") == "4"
    assert _flag_value(cmd, "--spec-draft-n-min") == "1"
    assert _flag_value(cmd, "--draft-p-min") == "0.25"
    assert _flag_value(cmd, "--draft-p-split") == "0.75"
    assert _flag_value(cmd, "--threads-draft") == "8"
    assert _flag_value(cmd, "--spec-ngram-mod-n-min") == "0"
    assert _flag_value(cmd, "--spec-ngram-mod-n-max") == "96"
    assert _flag_value(cmd, "--spec-ngram-mod-n-match") == "12"
    assert _flag_value(cmd, "-ub") == "256"
    assert _flag_value(cmd, "-c") == "12288"
    assert _flag_value(cmd, "-ctk") == "q5_0"
    assert _flag_value(cmd, "-ctv") == "q6_0"
    assert "--no-mmap" not in cmd
    assert "--jinja" not in cmd
    assert "--flash-attn" not in cmd


def test_build_worker_general_command_rejects_boolean_runtime_numbers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    priors = _write_launch_prior(
        tmp_path,
        "worker_general",
        requirements={
            "model_path": "/prior/gemma.gguf",
            "draft_model_path": "/prior/draft.gguf",
        },
        runtime={
            # 2026-06-26 v6 cutover: worker on canonical llama.cpp (v6), not ik.
            "binary_path": "/prior/llama.cpp/llama-server",
            "cache": {
                "context_tokens": True,
                "slots": True,
                "ubatch": True,
                "kv_type_k": "q8_0",
                "kv_type_v": "q8_0",
            },
            "flags": {
                "flash_attn": True,
                "jinja": True,
                "reasoning": "off",
                "spec": {
                    "enabled": True,
                    # 2026-06-26 v6 cutover: spec type token is now 'draft-mtp'.
                    "type": "draft-mtp",
                    "draft_model_path": "/prior/draft.gguf",
                    "draft_max": True,
                    "draft_p_min": True,
                    "threads_draft": True,
                },
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)

    cmd = oss._build_worker_general_command(
        port=8072,
        model_path="/fallback/gemma.gguf",
        binary_override=None,
    )

    # The prior fixture above declares no slots, so `-np` falls through to the
    # ambient declaration for :8072 — worker's FULL instance, 16 as of the
    # 2026-08-02 per-shape ratification (was 1). What this test is actually
    # about is unchanged: a boolean in a numeric runtime field must be rejected
    # rather than coerced, which the spec assertions below still pin.
    from scripts.server.stack_manifest import DECLARED_SLOTS

    assert _flag_value(cmd, "-np") == str(DECLARED_SLOTS["worker_general"])
    fallback = oss._WORKER_GENERAL_DEGRADED_FALLBACK
    assert _flag_value(cmd, "-c") == str(fallback["context_tokens"])
    assert _flag_value(cmd, "-ub") == str(fallback["ubatch"])
    # 2026-06-26 v6 cutover: n-max emitted via --spec-draft-n-max (was --draft-max);
    # value now comes from the launcher's explicit degraded fallback.
    assert _flag_value(cmd, "--spec-draft-n-max") == str(fallback["draft_max"])
    assert _flag_value(cmd, "--draft-p-min") == str(fallback["draft_p_min"])
    assert _flag_value(cmd, "--threads-draft") == str(fallback["threads_draft"])


def test_build_worker_general_command_uses_binary_override_when_set() -> None:
    # 2026-06-26 v6 cutover: worker runs on canonical llama.cpp (v6); ik_llama.cpp
    # is deprecated. An explicit binary_override is still honored verbatim — assert
    # against a v6 llama.cpp path rather than the retired ik build.
    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf",
        binary_override="/opt/llama.cpp/build/bin/llama-server",
    )
    assert cmd[0] == "/opt/llama.cpp/build/bin/llama-server"


def test_build_worker_general_command_prefers_live_stack_prior_binary() -> None:
    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    runtime = _stack_prior_role("worker_general")["serving"]["launch"]["runtime"]
    assert cmd[0] == runtime["binary_path"]


def test_build_worker_general_command_falls_back_to_llama_server_without_priors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", tmp_path / "missing.yaml")

    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )

    assert cmd[0] == str(oss.LLAMA_SERVER)


def test_worker_lane_host_command_uses_numa_thread_count_per_instance() -> None:
    """Sub-full instances must get the per-instance thread count from NUMA_CONFIG.

    Renamed from ``test_build_worker_general_command_uses_numa_thread_count_for_port``.
    Since the 2026-09-22 lineup cutover (860b0b2d) worker_general is an ALIAS on
    frontdoor's process: its own numa_config was deleted and the worker lane's full
    and halves are frontdoor's instances, launched by ``_build_role_command``. The
    witness moves to the role that actually launches them.

    Post-da1aed6 the thread count is resolved by ``numa_instance`` *index* (not by
    port): the start loop / dispatcher pass the index of the instance being
    launched, and _resolve_thread_count picks instances[numa_instance].threads.
    That is what gives the halves their own -t while the full instance keeps the
    full machine's — the over-subscription the launcher bug once applied to every
    sub-full instance.
    """
    from scripts.server import stack_manifest

    host, _row, _binding = stack_manifest.master_server_row("worker_general")
    assert host == "frontdoor"
    cfg = oss.NUMA_CONFIG[host]
    instances = cfg["instances"]
    role_config = SimpleNamespace(
        name=host,
        model=SimpleNamespace(full_path=f"/models/{host}.gguf", name=host),
        acceleration=SimpleNamespace(type="none", draft_role=None, experts=None, k=None),
    )
    for idx, (_cpus, port, expected_threads) in enumerate(instances):
        cmd = oss._build_role_command(role_config, port, idx, prepare_runtime_dirs=False)
        assert cmd[cmd.index("-t") + 1] == str(expected_threads), (
            f"{host} instance {idx} (port {port}) expected -t {expected_threads}"
        )
    # Non-vacuous: the sub-full instances really do differ from the full, so a
    # builder that ignored numa_instance would fail above.
    full_threads = instances[cfg["full_instance_idx"]][2]
    assert any(inst[2] != full_threads for inst in instances)


def test_build_worker_general_command_unknown_port_uses_fallback_96() -> None:
    cmd = oss._build_worker_general_command(
        port=9999, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[cmd.index("-t") + 1] == "96"


def test_build_worker_explore_command_keeps_compatibility_wrapper() -> None:
    cmd = oss._build_worker_explore_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"


# -----------------------------------------------------------------------------
# Default-role builder sub-helpers
# -----------------------------------------------------------------------------


def test_append_runtime_spec_args_omits_md_for_embedded_nextn_self_draft() -> None:
    cmd = ["llama-server", "-m", "/models/qwen-mtp.gguf"]
    runtime = {
        "flags": {
            "spec": {
                "enabled": True,
                "type": "draft-mtp",
                "draft_model_path": "/models/qwen-mtp.gguf",
                "draft_max": 4,
            }
        }
    }

    oss._append_runtime_spec_args(cmd, runtime, "/models/qwen-mtp.gguf")

    assert "-md" not in cmd
    assert _flag_value(cmd, "--spec-type") == "draft-mtp"
    assert _flag_value(cmd, "--spec-draft-n-max") == "4"


def test_append_runtime_spec_args_keeps_md_for_separate_draft_model() -> None:
    cmd = ["llama-server", "-m", "/models/gemma.gguf"]
    runtime = {
        "flags": {
            "spec": {
                "enabled": True,
                "type": "draft-mtp",
                "draft_model_path": "/models/gemma-assistant.gguf",
                "draft_max": 2,
            }
        }
    }

    oss._append_runtime_spec_args(cmd, runtime, "/models/gemma.gguf")

    assert _flag_value(cmd, "-md") == "/models/gemma-assistant.gguf"
    assert _flag_value(cmd, "--spec-type") == "draft-mtp"
    assert _flag_value(cmd, "--spec-draft-n-max") == "2"


def test_append_runtime_serving_flags_emits_chat_template_file(tmp_path: Path) -> None:
    template = tmp_path / "role-template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    cmd: list[str] = []

    oss._append_runtime_serving_flags(cmd, {"chat_template_file": str(template)})

    assert _flag_value(cmd, "--chat-template-file") == str(template)


def test_append_runtime_serving_flags_omits_chat_template_file_when_undeclared() -> None:
    # Absent key and compiled-null key are both "undeclared": no flag, no default.
    for flags in ({}, {"chat_template_file": None}):
        cmd: list[str] = []
        oss._append_runtime_serving_flags(cmd, flags)
        assert "--chat-template-file" not in cmd


def test_append_runtime_serving_flags_refuses_missing_chat_template_file(
    tmp_path: Path,
) -> None:
    # A declared template that is not on disk must refuse at launch-build time
    # (same fail-fast posture as validate_model_paths for a missing model), not
    # launch a server that silently falls back to the GGUF-embedded template.
    missing = tmp_path / "deleted-template.jinja"
    cmd = ["llama-server"]

    with pytest.raises(ValueError, match="chat_template_file"):
        oss._append_runtime_serving_flags(cmd, {"chat_template_file": str(missing)})

    assert "--chat-template-file" not in cmd


def test_build_role_command_emits_declared_chat_template_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # End-to-end over the DATA path: compiled prior flags -> _build_role_command
    # -> --chat-template-file on the llama-server cmdline.
    template = tmp_path / "frontdoor-template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    priors = _write_launch_prior(
        tmp_path,
        "frontdoor",
        requirements={},
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {
                "context_tokens": 24576,
                "slots": 1,
                "ubatch": 2048,
            },
            "flags": {
                "flash_attn": False,
                "jinja": False,
                "reasoning": None,
                "override_kv": [],
                "chat_template_file": str(template),
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)
    role = SimpleNamespace(
        name="frontdoor",
        model=SimpleNamespace(full_path="/fallback/frontdoor.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )

    cmd = oss._build_role_command(role, port=9070)

    assert _flag_value(cmd, "--chat-template-file") == str(template)


def test_build_role_command_sub_full_instance_emits_declared_serving_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A half/quarter must emit the SAME declared serving flags as the full.

    The compiled prior is role-level: one runtime.flags record covers every
    instance of the role. This pins that numa_instance>0 builds read it — a
    per-instance path that fell back to empty/legacy flags (the 2026-08-21
    --jinja fallback family) would strip chat_template_file, -ngl,
    --cache-ram and --jinja from every sub-full launch.
    """
    template = tmp_path / "frontdoor-template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    priors = _write_launch_prior(
        tmp_path,
        "frontdoor",
        requirements={},
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {
                "context_tokens": 24576,
                "slots": 1,
                "ubatch": 2048,
            },
            "flags": {
                "flash_attn": False,
                "jinja": True,
                "reasoning": None,
                "override_kv": [],
                "n_gpu_layers": 999,
                "cache_ram": 0,
                "chat_template_file": str(template),
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)
    role = SimpleNamespace(
        name="frontdoor",
        model=SimpleNamespace(full_path="/fallback/frontdoor.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )

    def declared_serving_flags(cmd: list[str]) -> dict[str, object]:
        return {
            "chat_template_file": _flag_value(cmd, "--chat-template-file"),
            "n_gpu_layers": _flag_value(cmd, "-ngl"),
            "cache_ram": _flag_value(cmd, "--cache-ram"),
            "jinja": "--jinja" in cmd,
        }

    full = oss._build_role_command(role, port=8070, numa_instance=0)
    half = oss._build_role_command(role, port=8080, numa_instance=1)

    expected = {
        "chat_template_file": str(template),
        "n_gpu_layers": "999",
        "cache_ram": "0",
        "jinja": True,
    }
    assert declared_serving_flags(full) == expected
    assert declared_serving_flags(half) == expected, (
        "sub-full instance dropped declared serving flags the full emits"
    )


def test_resolve_thread_count_from_numa_config() -> None:
    # frontdoor[0] is NUMA_NODE0 = 96 threads
    assert oss._resolve_thread_count("frontdoor") == "96"


def test_resolve_thread_count_fallback_for_unknown_role() -> None:
    assert oss._resolve_thread_count("nonexistent_role") == "96"


def test_resolve_binary_for_role_defaults_to_llama_server() -> None:
    # _V2_ROLES is currently empty, so every role gets LLAMA_SERVER
    assert oss._resolve_binary_for_role("frontdoor") == oss.LLAMA_SERVER
    assert oss._resolve_binary_for_role("architect_general") == oss.LLAMA_SERVER


@pytest.mark.parametrize(
    ("declared", "expected_tail"),
    [
        (True, ["--kv-unified"]),
        (False, ["--no-kv-unified"]),
        (None, []),
    ],
)
def test_append_runtime_kv_args_emits_declared_kv_unified_both_ways(
    declared: bool | None, expected_tail: list[str]
) -> None:
    # A declared fact is emitted explicitly in BOTH directions; an undeclared one
    # emits nothing (every CPU role keeps its current cmdline byte-for-byte).
    cmd: list[str] = []
    oss._append_runtime_kv_args(
        cmd, {"kv_type_k": "q8_0", "kv_type_v": "q8_0", "kv_unified": declared}
    )
    assert cmd == ["-ctk", "q8_0", "-ctv", "q8_0", *expected_tail]


def test_append_runtime_kv_args_emits_declared_draft_kv_types() -> None:
    cmd: list[str] = []
    oss._append_runtime_kv_args(
        cmd,
        {
            "kv_type_k": "q8_0",
            "kv_type_v": "q8_0",
            "draft_kv_type_k": "q8_0",
            "draft_kv_type_v": "q8_0",
            "kv_unified": True,
        },
    )
    assert cmd == ["-ctk", "q8_0", "-ctv", "q8_0", "-ctkd", "q8_0", "-ctvd", "q8_0", "--kv-unified"]


def test_append_runtime_kv_args_omits_half_declared_draft_kv_types() -> None:
    cmd: list[str] = []
    oss._append_runtime_kv_args(cmd, {"draft_kv_type_k": "q8_0", "draft_kv_type_v": None})
    assert cmd == []


def test_build_role_command_emits_declared_kv_unified_with_explicit_np(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # End-to-end over the DATA path, at the :8083 shape: -np 2 stays, and -kvu
    # is added so ONE request may use the whole -c pool.
    priors = _write_launch_prior(
        tmp_path,
        "architect_general",
        requirements={},
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {
                "context_tokens": 196608,
                "slots": 2,
                "ubatch": 2048,
                "kv_type_k": "q8_0",
                "kv_type_v": "q8_0",
                "kv_unified": True,
            },
            "flags": {
                "flash_attn": True,
                "jinja": True,
                "reasoning": None,
                "override_kv": [],
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)
    role = SimpleNamespace(
        name="architect_general",
        model=SimpleNamespace(full_path="/fallback/27b.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )

    cmd = oss._build_role_command(role, port=9083, prepare_runtime_dirs=False)

    assert _flag_value(cmd, "-np") == "2"
    assert _flag_value(cmd, "-c") == "196608"
    assert cmd.count("--kv-unified") == 1
    assert "--no-kv-unified" not in cmd
    assert stack_commands._live_kv_unified(cmd) is True


def test_append_kv_quant_args_emits_q8_for_frontdoor() -> None:
    cmd: list[str] = []
    oss._append_kv_quant_args(cmd, "frontdoor")
    assert cmd == ["-ctk", "q8_0", "-ctv", "q8_0"]


def test_append_kv_quant_args_emits_declared_pair_for_flash_next_architect() -> None:
    """Renamed from ``test_append_kv_quant_args_emits_q4_f16_for_architect_critic``.

    History: the (q4_0, f16) pair belonged to the Qwen3.5-122B UD-Q4_K_M and moved
    WITH it to architect_critic on 2026-08-01. The 2026-09-22 lineup cutover
    (860b0b2d) RETIRED the 122B; architect_critic is now Qwen3.8-Flash-Next
    UD-IQ4_XS and master declares ``server_mode.architect_critic.serving_shape
    .kv_quant = f16/f16``. The KV pair follows the model, so the expectation is
    read from the role's OWN declaration (not an alias's) instead of restated.
    2026-09-27 ARCHITECT SWAP: the Flash-Next process is architect_general now.
    """
    from scripts.server import stack_manifest

    # 2026-09-27 ARCHITECT SWAP: the Flash-Next :8074 process now serves
    # architect_general, so its declared pair is read under that host row.
    declared, source = stack_manifest.master_declared("architect_general", "kv_quant")
    assert source == "architect_general/direct.serving_shape"
    assert isinstance(declared, dict) and declared.get("k") and declared.get("v")

    cmd: list[str] = []
    oss._append_kv_quant_args(cmd, "architect_general")
    assert cmd == ["-ctk", str(declared["k"]), "-ctv", str(declared["v"])]


def test_append_kv_quant_args_emits_q8_for_the_27b_architect() -> None:
    """2026-08-01 W1 cutover: the MI210 27B dense Q8 takes q8_0/q8_0 (was q4_0/f16
    under the 122B). It MUST match coder_escalation, which is an alias on the very
    same :8083 process — two aliases declaring different KV shapes for one server is
    incoherent. The 27B serves architect_critic since the 2026-09-27 ARCHITECT SWAP
    (architect_general before)."""
    cmd: list[str] = []
    oss._append_kv_quant_args(cmd, "architect_critic")
    assert cmd == ["-ctk", "q8_0", "-ctv", "q8_0"]


def test_append_kv_quant_args_noop_for_role_without_config() -> None:
    """A launch role with no derived KV-quant entry must get NO -ctk/-ctv.

    2026-08-02: `worker_vision` was this test's example of a KV-config-free role.
    It is not one any more — the MMMU-250 paired A/B (f16 153/250 vs q8_0
    155/250, +0.80 pp, CI [-1.90,+3.03], non-inferior at a pre-registered 3 pp
    margin) moved it to q8_0/q8_0. `_KV_QUANT_CONFIGS` is also no longer
    hand-listed: it is DERIVED from `server_mode.<role>.serving_shape.kv_quant`.
    So take the examples FROM the derived table instead of naming a role, and
    this cannot go stale the same way again.
    """
    uncovered = sorted(set(oss.ROLE_LAUNCH_META) - set(oss._KV_QUANT_CONFIGS))
    # The role that made this test stale is now covered — that is the point.
    assert "worker_vision" not in uncovered
    # Guard against the loop below going vacuous if every role gains a config.
    assert uncovered, "expected at least one launch role without a derived KV-quant config"
    for role_name in uncovered:
        cmd = ["pre-existing"]
        oss._append_kv_quant_args(cmd, role_name)
        assert cmd == ["pre-existing"], role_name


def test_apply_numa_spec_overrides_rewrites_draft_max() -> None:
    # 2026-06-26 v6 cutover: rewrite targets the emitted --spec-draft-n-max flag
    # (v6 removed --draft-max); the spec_overrides schema key stays 'draft_max'.
    cmd = ["--spec-draft-n-max", "16", "--other", "thing"]
    numa_cfg = {"spec_overrides": {"draft_max": 2}}
    oss._apply_numa_spec_overrides(cmd, numa_cfg)
    assert cmd == ["--spec-draft-n-max", "2", "--other", "thing"]


def test_apply_numa_spec_overrides_noop_without_overrides() -> None:
    # 2026-06-26 v6 cutover: --spec-draft-n-max replaces --draft-max.
    cmd = ["--spec-draft-n-max", "16"]
    oss._apply_numa_spec_overrides(cmd, {"instances": []})
    assert cmd == ["--spec-draft-n-max", "16"]


def test_apply_numa_spec_overrides_noop_when_numa_cfg_none() -> None:
    # 2026-06-26 v6 cutover: --spec-draft-n-max replaces --draft-max.
    cmd = ["--spec-draft-n-max", "16"]
    oss._apply_numa_spec_overrides(cmd, None)
    assert cmd == ["--spec-draft-n-max", "16"]


def test_append_acceleration_args_moe_expert_reduction() -> None:
    accel = SimpleNamespace(
        type="moe_expert_reduction",
        experts=4,
        override_key="qwen3vlmoe.expert_used_count",
        draft_role=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "vision_escalation", accel, "/m/x.gguf")
    assert cmd == ["--override-kv", "qwen3vlmoe.expert_used_count=int:4"]


def test_append_acceleration_args_architect_general_uses_nextn_self_draft() -> None:
    """2026-06-26 v6 cutover: architect_general now runs native NEXTN MTP (draft-mtp).
    It was removed from _NO_SPEC_DECODE (the M-RoPE assertion is resolved on v6), and
    its draft is the NEXTN self-draft (draft_model == base) resolved by the compiled
    stack_priors / _append_runtime_spec_args path, which later omits -md for
    same-realpath drafts — NOT via this helper's draft_role branch. So with
    draft_role=None this helper emits no spec args (and must not crash)."""
    accel = SimpleNamespace(
        type="speculative_decoding",
        draft_role=None,  # v6 NEXTN self-draft via draft_model; spec emitted from stack_priors
        k=4,
        experts=None,
        n_layer_exit_draft=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "architect_general", accel, "/m/x.gguf")
    assert cmd == []


def test_append_acceleration_args_self_speculation_omits_same_file_md() -> None:
    accel = SimpleNamespace(
        type="self_speculation",
        n_layer_exit_draft=4,
        k=8,
        draft_role=None,
        experts=None,
        n_layer_exit_intermediate=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "some_role", accel, "/m/target.gguf")
    assert "-md" not in cmd
    assert cmd[cmd.index("--n-layer-exit-draft") + 1] == "4"
    # 2026-06-26 v6 cutover: n-max emitted via --spec-draft-n-max (was --draft-max).
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "8"


def test_append_acceleration_args_hierarchical_speculation_with_intermediate() -> None:
    accel = SimpleNamespace(
        type="hierarchical_speculation",
        n_layer_exit_draft=3,
        n_layer_exit_intermediate=7,
        k=12,
        draft_role=None,
        experts=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "some_role", accel, "/m/x.gguf")
    assert "-md" not in cmd
    assert "--hierarchical-spec" in cmd
    assert cmd[cmd.index("--n-layer-exit-intermediate") + 1] == "7"


def test_build_role_command_prefers_stack_prior_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    slot_dir = tmp_path / "slot-cache" / "frontdoor"
    priors = _write_launch_prior(
        tmp_path,
        "frontdoor",
        requirements={},
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {
                "context_tokens": 24576,
                "slots": 1,
                "ubatch": 2048,
                "kv_type_k": "q5_0",
                "kv_type_v": "q6_0",
                "mlock": False,
                "slot_save_path": str(slot_dir),
            },
            "flags": {
                "flash_attn": False,
                "jinja": False,
                "reasoning": None,
                "override_kv": ["qwen36.expert_used_count=int:3"],
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)
    role = SimpleNamespace(
        name="frontdoor",
        model=SimpleNamespace(full_path="/fallback/frontdoor.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )

    cmd = oss._build_role_command(role, port=9070)

    assert cmd[0] == "/prior/llama-server"
    assert _flag_value(cmd, "-m") == "/fallback/frontdoor.gguf"
    assert _flag_value(cmd, "-np") == "1"
    assert _flag_value(cmd, "-c") == "24576"
    assert _flag_value(cmd, "-ub") == "2048"
    assert _flag_value(cmd, "-ctk") == "q5_0"
    assert _flag_value(cmd, "-ctv") == "q6_0"
    assert _all_flag_values(cmd, "--override-kv") == [
        "qwen36.expert_used_count=int:3"
    ]
    assert _flag_value(cmd, "--slot-save-path") == str(slot_dir)
    assert "--mlock" not in cmd
    assert "--jinja" not in cmd
    assert "--flash-attn" not in cmd


def test_build_role_command_omits_md_for_embedded_nextn_stack_prior(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = "/prior/qwen-mtp.gguf"
    runtime = {
        "binary_path": "/prior/llama-server",
        "cache": {
            "context_tokens": 24576,
            "slots": 1,
            "ubatch": 2048,
            "kv_type_k": "q8_0",
            "kv_type_v": "q8_0",
            "no_mmap": False,
            "mlock": True,
            "slot_save_path": str(oss.SLOT_SAVE_DIR / "frontdoor"),
        },
        "flags": {
            "flash_attn": True,
            "jinja": True,
            "reasoning": "off",
            "override_kv": [],
            "spec": {
                "enabled": True,
                "type": "draft-mtp",
                "draft_model_path": model_path,
                "draft_max": 4,
                "draft_p_min": None,
                "threads_draft": None,
            },
        },
    }
    priors = _write_launch_prior(
        tmp_path,
        "frontdoor",
        requirements={
            "model_path": model_path,
            "draft_model_path": model_path,
        },
        runtime=runtime,
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)
    role = SimpleNamespace(
        name="frontdoor",
        model=SimpleNamespace(full_path=model_path),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )

    cmd = oss._build_role_command(role, port=9070)

    assert _flag_value(cmd, "-m") == model_path
    assert "-md" not in cmd
    assert _flag_value(cmd, "--spec-type") == "draft-mtp"
    assert _flag_value(cmd, "--spec-draft-n-max") == "4"
    assert _command_runtime_signature(cmd) == _stack_prior_runtime_signature(runtime)


def test_eval_batch_frontdoor_command_uses_declared_serving_shape() -> None:
    cmd = oss._build_eval_batch_frontdoor_command(18070)

    assert _flag_value(cmd, "--port") == "18070"
    assert _flag_value(cmd, "-np") == "2"
    assert _flag_value(cmd, "-c") == "32768"
    # 2026-07-31 HALF FLEET: 18070 sits on NUMA_HALF_A, which is 48 physical cores.
    # The prior value 96 pinned the 2x SMT oversubscription that 982adb0c removed.
    assert _flag_value(cmd, "-t") == "48"
    # K4 (3cb53971): -ub is emitted without -b, so llama.cpp clamps it to the
    # n_batch default; the declared 8192 was never real and the manifest default
    # became 2048. The lane reuses frontdoor's runtime priors, falling back to
    # DEFAULT_UBATCH_TOKENS — both are read here rather than restated.
    from scripts.server.stack_manifest import DEFAULT_UBATCH_TOKENS

    frontdoor_ubatch = _stack_prior_role("frontdoor")["serving"]["launch"]["runtime"][
        "cache"
    ]["ubatch"]
    assert frontdoor_ubatch == DEFAULT_UBATCH_TOKENS
    assert _flag_value(cmd, "-ub") == str(frontdoor_ubatch)
    assert _flag_value(cmd, "-ctk") == "q8_0"
    assert _flag_value(cmd, "-ctv") == "q8_0"
    # CORRECT AS WRITTEN — do not "fix" this to ngram-mod,draft-mtp. Master reversed
    # the composed recipe at a126e43d; derived/stack_priors.yaml has not been
    # regenerated, so this assertion fails against stale data, not a stale expectation.
    assert _flag_value(cmd, "--spec-type") == "draft-mtp"
    assert _flag_value(cmd, "--spec-draft-n-max") == "4"
    assert "--jinja" in cmd
    assert "--mlock" in cmd
    assert "--log-colors" in cmd
    assert "-md" not in cmd


# -----------------------------------------------------------------------------
# Dispatcher routing
# -----------------------------------------------------------------------------


def test_dispatcher_routes_vision_mode() -> None:
    with patch.object(oss, "_build_vision_command", return_value=["VISION"]) as m:
        out = oss.build_server_command(None, 8087, vision_mode=True, vision_type="escalation")
    # 2026-08-01: was ["VISION", "--device", "none"]. The tail is the dispatcher's
    # device-pinning step, which used to ASSUME CPU regardless of the role; it now
    # reads the role's compiled declaration, and vision_escalation declares ROCm0.
    # Asserting "none" here was pinning the defect that launched GPU-declared roles
    # with every device disabled. Polarity flipped, strength unchanged.
    # STACKCHG-DFLASH2-20261003: vision_escalation rides the COLD CPU :8086 process -> "none" (declared, not
    # assumed: the tail still reads the role's compiled declaration).
    assert out == ["VISION", "--device", "none"]
    # numa_instance defaults to 0 (full) and is forwarded post-da1aed6 so quarters
    # get NUMA_CONFIG -t (was always -t 96).
    m.assert_called_once_with(8087, "escalation", 0)


def test_dispatcher_routes_embedding_mode() -> None:
    with patch.object(oss, "_build_embedding_command", return_value=["EMB"]) as m:
        out = oss.build_server_command(None, 8090, embedding_mode=True)
    assert out == ["EMB", "--device", "none"]
    m.assert_called_once_with(8090)


def test_dispatcher_routes_eval_batch_frontdoor_mode() -> None:
    with patch.object(oss, "_build_eval_batch_frontdoor_command", return_value=["EVAL"]) as m:
        out = oss.build_server_command(None, 18070, eval_batch_frontdoor_mode=True)
    assert out == ["EVAL", "--device", "none"]
    m.assert_called_once_with(18070, 0)


def test_dispatcher_routes_worker_fast() -> None:
    with patch.object(oss, "_build_worker_fast_command", return_value=["FAST"]) as m:
        out = oss.build_server_command(
            None, 8102, worker_pool_mode=True, worker_type="fast",
        )
    assert out == ["FAST", "--device", "none"]
    m.assert_called_once()
    assert m.call_args.args[0] == 8102


def test_dispatcher_routes_worker_general_with_binary_override() -> None:
    # 2026-06-26 v6 cutover: worker runs on canonical llama.cpp (v6); ik is
    # deprecated. This is a pure binary_override pass-through assertion.
    with patch.object(oss, "_build_worker_general_command", return_value=["GEMMA"]) as m:
        out = oss.build_server_command(
            None, 8072, worker_pool_mode=True, worker_type="explore",
            binary_override="/opt/llama.cpp/llama-server",
        )
    assert out == ["GEMMA", "--device", "none"]
    m.assert_called_once()
    # signature: (port, model_path, binary_override)
    assert m.call_args.args[0] == 8072
    assert m.call_args.args[2] == "/opt/llama.cpp/llama-server"


def test_dispatcher_raises_on_unknown_worker_type() -> None:
    with pytest.raises(ValueError, match="Unknown worker type"):
        oss.build_server_command(
            None, 8000, worker_pool_mode=True, worker_type="ghost",
        )


def test_dispatcher_routes_dev_mode() -> None:
    with patch.object(oss, "_build_dev_command", return_value=["DEV"]) as m:
        out = oss.build_server_command(None, 9999, dev_mode=True)
    assert out == ["DEV", "--device", "none"]
    m.assert_called_once_with(9999)


def test_dispatcher_routes_default_to_role_builder() -> None:
    fake_role = SimpleNamespace(name="frontdoor")
    with patch.object(oss, "_build_role_command", return_value=["ROLE"]) as m:
        out = oss.build_server_command(fake_role, 8070)
    assert out == ["ROLE", "--device", "none"]
    # numa_instance (default 0) forwarded post-da1aed6.
    m.assert_called_once_with(fake_role, 8070, 0)


def test_start_server_vision_forwards_numa_instance_to_prefix(tmp_path, monkeypatch) -> None:
    """Vision quarter launches must inherit their NUMA_CONFIG CPU mask, not idx0."""
    calls: list[tuple[str, int]] = []

    def fake_prefix(role: str, instance_idx: int = 0) -> list[str]:
        calls.append((role, instance_idx))
        return ["taskset", "-c", f"{role}:{instance_idx}"]

    fake_proc = SimpleNamespace(pid=4242)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    with (
        patch.object(oss, "_numa_prefix", side_effect=fake_prefix),
        patch.object(oss, "build_server_command", return_value=["llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8187,
            roles=["vision_escalation"],
            registry=SimpleNamespace(),
            vision_mode=True,
            vision_type="escalation",
            numa_instance=1,
        )

    assert info is not None
    assert calls == [("vision_escalation", 1)]
    assert popen.call_args.args[0][:3] == ["taskset", "-c", "vision_escalation:1"]
    _assert_detached_popen(popen)


def test_start_server_vision_applies_stack_prior_runtime_ld_path(tmp_path, monkeypatch) -> None:
    fake_proc = SimpleNamespace(pid=4243)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        oss,
        "build_launch_env",
        lambda *a, **kw: {"GGML_IQK": "1", "GGML_TEST_FLAG": "remove"},
    )
    monkeypatch.setattr(
        oss,
        "_stack_prior_runtime_overrides",
        lambda role: ("/tmp/v7-hip/bin/llama-server", ["/tmp/v7-hip/bin"]),
    )
    with (
        patch.object(oss, "_numa_prefix", return_value=[]),
        patch.object(oss, "build_server_command", return_value=["/tmp/v7-hip/bin/llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8087,
            roles=["vision_escalation"],
            registry=SimpleNamespace(),
            vision_mode=True,
            vision_type="escalation",
        )

    assert info is not None
    env = popen.call_args.kwargs["env"]
    assert env["GGML_IQK"] == "1"
    assert "GGML_TEST_FLAG" not in env
    assert env["LD_LIBRARY_PATH"].startswith("/tmp/v7-hip/bin")
    assert env["KMP_BLOCKTIME"] == "10"
    _assert_detached_popen(popen)


def test_start_server_worker_pool_forwards_numa_instance_to_prefix(tmp_path, monkeypatch) -> None:
    """Worker quarter launches must use the quarter CPU mask as well as -t 48."""
    calls: list[tuple[str, int]] = []

    def fake_prefix(role: str, instance_idx: int = 0) -> list[str]:
        calls.append((role, instance_idx))
        return ["taskset", "-c", f"{role}:{instance_idx}"]

    fake_proc = SimpleNamespace(pid=4343)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    monkeypatch.setattr(oss, "_runtime_requirements_for_role", lambda *a, **kw: (None, None))
    with (
        patch.object(oss, "_numa_prefix", side_effect=fake_prefix),
        patch.object(oss, "build_server_command", return_value=["llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8282,
            roles=["worker_general"],
            registry=SimpleNamespace(),
            worker_pool_mode=True,
            worker_type="explore",
            numa_instance=3,
        )

    assert info is not None
    assert calls == [("worker_general", 3)]
    assert popen.call_args.args[0][:3] == ["taskset", "-c", "worker_general:3"]
    _assert_detached_popen(popen)


def test_every_instance_interleaves_over_only_the_nodes_it_spans() -> None:
    """REPLACES test_worker_general_numa_policy_is_full_instance_only (2026-07-31).

    That test asserted the sub-full instances carry NO numactl policy. Post-982adb0c
    that is the defect, not the contract: a cpuset spanning two NPS4 nodes with no
    policy places weights on nodes its threads cannot reach locally, which is what
    cost frontdoor and ingest_long_context ~2x. It would have failed the fix rather
    than the bug, so it is replaced rather than adjusted.
    """
    # 2026-09-22 lineup cutover (860b0b2d): worker_general's own numa_config was
    # deleted (it is now an alias on frontdoor's :8070 process), so the witness is
    # retargeted to EVERY declared instance. The expected node set is recomputed
    # from each instance's cpuset through the host-fact NPS4 map, not restated.
    from scripts.server import stack_numa

    all_nodes = sorted(stack_numa._NPS4_NODES)
    checked_sub_full: set[tuple[str, int]] = set()
    for role, cfg in oss.NUMA_CONFIG.items():
        for idx, (cpus, port, _threads) in enumerate(cfg["instances"]):
            prefix = oss._numa_prefix(role, idx)
            policies = [p for p in prefix if p.startswith("--interleave=")]
            if not policies:
                continue
            spanned = stack_numa._nodes_touched(cpus)
            value = policies[0].split("=", 1)[1]
            interleaved = all_nodes if value == "all" else sorted(
                int(n) for n in value.split(",")
            )
            assert interleaved == spanned, (
                f"{role}[{idx}] :{port} interleaves over {interleaved} but its cpuset "
                f"{cpus!r} spans NPS4 nodes {spanned}"
            )
            assert prefix[:3] == ["numactl", policies[0], "--"]
            if spanned != all_nodes:
                # A sub-full instance must never interleave over all four nodes —
                # that is the original defect.
                assert value != "all"
                checked_sub_full.add((role, idx))

    # The shape that motivated this test: the worker lane's host, frontdoor, runs
    # a full + halves. Non-vacuous: every one of its sub-full instances was checked.
    frontdoor = oss.NUMA_CONFIG["frontdoor"]
    frontdoor_sub_full = {
        ("frontdoor", idx)
        for idx in range(len(frontdoor["instances"]))
        if idx != frontdoor["full_instance_idx"]
    }
    assert frontdoor_sub_full
    assert frontdoor_sub_full <= checked_sub_full


def test_role_level_numa_policy_still_applies_to_all_instances() -> None:
    # 2026-08-01 W1 cutover: the full-machine `interleave=all` instance under test
    # here used to be architect_general's; it moved WITH the 122B to
    # architect_critic (:8074, cpus 0-95). Same policy, same shape, new role name.
    # 2026-09-27 ARCHITECT SWAP: the labels swapped back over unchanged processes —
    # the :8074 full-machine instance is architect_general, the GPU host-lane
    # membind=3 instance (:8083) is architect_critic.
    assert oss._numa_prefix("architect_general", 0)[:3] == [
        "numactl",
        "--interleave=all",
        "--",
    ]
    # The GPU host-lane 27B keeps a role-level (not per-instance) policy — NPS4
    # node 3. Asserted so the "role-level policy applies" contract still has a
    # witness on this process.
    assert oss._numa_prefix("architect_critic", 0)[:3] == [
        "numactl",
        "--membind=3",
        "--",
    ]


def test_start_server_embedding_candidate_reports_recipe_model_path(
    tmp_path, monkeypatch
) -> None:
    fake_proc = SimpleNamespace(pid=4344)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    # UFH-12 Phase 0: :8096 now has a declared placement, so the spawn is
    # `numactl --membind=2 -- taskset -c 152-167 llama-server ...`. The bench
    # guard is stubbed so the test never reads a live bench claim from /proc.
    with (
        patch.object(oss, "_numa_prefix", return_value=[]),
        patch.object(oss, "enforce_placement", return_value=None),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8096,
            roles=["embedder_granite_97m_r2"],
            registry=SimpleNamespace(),
            embedding_mode=True,
        )

    assert info is not None
    assert info.model_path.endswith("granite-embedding-97m-multilingual-r2-Q8_0.gguf")
    argv = popen.call_args.args[0]
    assert argv[:6] == ["numactl", "--membind=2", "--", "taskset", "-c", "152-167"]
    assert argv[6].endswith("llama-server")
    _assert_detached_popen(popen)


def test_start_server_default_detaches_child_stdio(tmp_path, monkeypatch) -> None:
    """Stack-managed llama-server children must outlive non-interactive launchers."""
    fake_proc = SimpleNamespace(pid=4444)
    fake_role = SimpleNamespace(
        model=SimpleNamespace(name="frontdoor-model", full_path="/m/frontdoor.gguf"),
    )
    registry = SimpleNamespace(get_role=lambda _role: fake_role)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    with (
        patch.object(oss, "_numa_prefix", return_value=["taskset", "-c", "frontdoor:0"]),
        patch.object(oss, "build_server_command", return_value=["llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8070,
            roles=["frontdoor"],
            registry=registry,
        )

    assert info is not None
    assert popen.call_args.args[0][:3] == ["taskset", "-c", "frontdoor:0"]
    _assert_detached_popen(popen)


def test_start_document_formalizer_detaches_child_stdio(tmp_path, monkeypatch) -> None:
    fake_proc = SimpleNamespace(pid=4545)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    with patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen:
        info = oss.start_document_formalizer()

    assert info is not None
    _assert_detached_popen(popen)


# =============================================================================
# Phase 2 — `-np` on the FALLBACK path comes from the declaration
# =============================================================================
# Compiled priors normally supply `slots`, so the launcher's own fallback is
# only reached when a role has no usable prior record. That is exactly why the
# divergence survived: the fallback was `1 if role in SERIAL_ROLES else 2`,
# which disagreed with `server_mode.<role>.slots` for 6 of the 11 roles master
# declares it for, and nothing exercised it. These tests exercise it.


def _np_without_priors(build, *args, **kwargs) -> str:
    """Build a command with the compiled priors suppressed."""
    with patch.object(oss, "_stack_prior_launch", lambda role: ({}, {})):
        cmd = build(*args, **kwargs)
    return cmd[cmd.index("-np") + 1]


@pytest.mark.parametrize(
    ("role", "port"),
    [
        ("frontdoor", 8070),
        # 2026-09-27 ARCHITECT SWAP: labels swapped over unchanged processes.
        ("architect_general", 8074),
        ("architect_critic", 8083),
        ("ingest_long_context", 8085),
    ],
)
def test_role_command_np_fallback_equals_the_declared_slots(role: str, port: int) -> None:
    from scripts.server.stack_manifest import DECLARED_SLOTS

    role_config = SimpleNamespace(
        name=role,
        model=SimpleNamespace(full_path=f"/models/{role}.gguf", name=role),
        acceleration=SimpleNamespace(type="none", draft_role=None, experts=None, k=None),
    )
    assert _np_without_priors(oss._build_role_command, role_config, port) == str(
        DECLARED_SLOTS[role]
    )


def test_vision_worker_np_fallback_is_the_declared_one_not_the_old_literal() -> None:
    """The exhibit: this branch hardcoded 2 while worker_vision declared 1.

    The declared value has since moved 1 -> 4 (2026-08-02, operator-ratified),
    which is exactly the point — the assertion tracks the DECLARATION rather than
    restating a number, so a ratified change lands here without an edit to the
    launcher. Only the constant below had to move.
    """
    from scripts.server.stack_manifest import DECLARED_SLOTS

    assert _np_without_priors(oss._build_vision_command, 8086, "worker") == str(
        DECLARED_SLOTS["worker_vision"]
    )


def test_worker_general_np_fallback_is_the_declared_one() -> None:
    from scripts.server.stack_manifest import DECLARED_SLOTS

    # 1 -> 16: :8072 is worker's FULL instance and `slots_by_shape.full` is 16.
    # DECLARED_SLOTS is the per-ROLE compat scalar, which the parity guard pins to
    # the primary instance's value, so the two agree by construction.
    assert _np_without_priors(
        oss._build_worker_general_command, 8072, "/models/w.gguf", None
    ) == str(DECLARED_SLOTS["worker_general"])


def test_worker_lane_host_np_differs_between_the_full_and_a_half() -> None:
    """The per-INSTANCE claim: one role, one record, two different `-np`.

    Renamed from ``test_worker_general_np_differs_between_the_full_and_a_half``.
    Since the 2026-09-22 lineup cutover (860b0b2d) worker_general is an ALIAS on
    frontdoor's process and has no instances of its own (its :8072/:8082/:8182
    numa_config was deleted), so the worker lane's full and halves are
    frontdoor's. The witness moves to the role that actually launches them.

    This is what a per-role slot count could not express: one role whose full
    instance and half instances take different slot counts, both resolved from the
    same `serving_shape.slots_by_shape` joined against the `cpu_shape`
    stack_topology.yaml declares for each instance.
    """
    from scripts.server import stack_manifest
    from scripts.server.stack_manifest import DECLARED_SLOTS

    host, _row, _binding = stack_manifest.master_server_row("worker_general")
    assert host == "frontdoor"
    cfg = oss.NUMA_CONFIG[host]
    full_idx = cfg["full_instance_idx"]
    half_idx = next(
        idx
        for idx in range(len(cfg["instances"]))
        if stack_manifest.instance_shape_class(host, idx) == "half"
    )
    role_config = SimpleNamespace(
        name=host,
        model=SimpleNamespace(full_path=f"/models/{host}.gguf", name=host),
        acceleration=SimpleNamespace(type="none", draft_role=None, experts=None, k=None),
    )
    full = _np_without_priors(
        oss._build_role_command,
        role_config,
        cfg["instances"][full_idx][1],
        full_idx,
        prepare_runtime_dirs=False,
    )
    half = _np_without_priors(
        oss._build_role_command,
        role_config,
        cfg["instances"][half_idx][1],
        half_idx,
        prepare_runtime_dirs=False,
    )
    by_shape, _source = stack_manifest.slots_by_shape_for(host)

    # Derived, not restated. The claim is per-INSTANCE differentiation: the full
    # carries the role's compat scalar (the parity guard pins them equal) and a half
    # is strictly narrower. Pinning literals here made a ratified slot change fail
    # this test instead of the registry — which is the defect the docstring warns of.
    assert full == str(DECLARED_SLOTS[host]) == str(by_shape["full"])
    assert half == str(by_shape["half"])
    assert int(half) < int(full)
    # The alias's compat scalar is its host's — one process, one declaration.
    assert DECLARED_SLOTS["worker_general"] == DECLARED_SLOTS[host]


def test_serial_roles_no_longer_shrinks_the_launched_slot_count() -> None:
    """SERIAL_ROLES is an admission policy; adding a role to it must not move -np."""
    from scripts.server import stack_manifest

    # The 27B GPU process on :8083 (architect_critic since the 2026-09-27 ARCHITECT
    # SWAP) is NOT serial, so adding it to SERIAL_ROLES is a real perturbation.
    # (architect_general is now the serial Flash-Next process, already in the set.)
    role_config = SimpleNamespace(
        name="architect_critic",
        model=SimpleNamespace(full_path="/models/a.gguf", name="architect_critic"),
        acceleration=SimpleNamespace(type="none", draft_role=None, experts=None, k=None),
    )
    assert "architect_critic" not in stack_manifest.SERIAL_ROLES
    before = _np_without_priors(oss._build_role_command, role_config, 8083)
    with patch.object(
        stack_manifest, "SERIAL_ROLES", stack_manifest.SERIAL_ROLES | {"architect_critic"}
    ):
        after = _np_without_priors(oss._build_role_command, role_config, 8083)
    # 2 -> 8 (operator-ratified 2026-08-02). The invariant under test is the
    # EQUALITY, not the number: adding a role to SERIAL_ROLES must leave `-np`
    # exactly where it was.
    from scripts.server.stack_manifest import DECLARED_SLOTS

    assert before == after
    assert before == str(DECLARED_SLOTS["architect_critic"])
