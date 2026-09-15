"""Zero-hardware fixtures for AKX-P0b's HIP route layer."""
from pathlib import Path
from unittest import mock

from autokernel.loop import census, hip_routes


IDENTITY = {"workload_id": "w1", "workload_signature": "sig",
            "model": {"sha256": "model"}, "recipe": {"sha256": "recipe"},
            "kernel": {"commit": "abc", "build_variant": "production",
                       "dso_digests": {"libggml-hip.so": "dso"}},
            "host": {"gpu": "gfx90a"}}


# Recorded W1 decode lines from /mnt/raid0/llm/tmp/mmvq-probe3.log.  Prefixes are
# retained because the parser must tolerate the logger's timestamp/severity wrapper.
W1_DECODE = """\
0.11.920.903 I GGML_CUDA_MUL_MAT_ROUTE route=MMQ src0=blk.0.attn_qkv.weight src1=attn_norm-0 dst=node_13 type=q8_0 cc=16779530 src0_ne=[5120,10240,1,1] src1_ne=[5120,2,1,1] dst_ne=[10240,2,1,1] ne11=2
0.11.988.049 I GGML_CUDA_MUL_MAT_ROUTE route=MMQ src0=blk.0.ffn_gate.weight src1=attn_post_norm-0 dst=ffn_gate-0 type=q8_0 cc=16779530 src0_ne=[5120,17408,1,1] src1_ne=[5120,2,1,1] dst_ne=[17408,2,1,1] ne11=2
0.12.031.701 I GGML_CUDA_MUL_MAT_ROUTE route=MMVQ src0=output.weight src1=result_norm dst=result_output type=q8_0 cc=16779530 src0_ne=[5120,248320,1,1] src1_ne=[5120,1,1,1] dst_ne=[248320,1,1,1] ne11=1
"""


def test_recorded_w1_multiset_has_stable_canonical_schema():
    parsed = hip_routes.parse_hip_matmul_routes(W1_DECODE)
    assert parsed["state"] == census.OBSERVED
    assert parsed["hip_matmul_routes"] == [
        {"route": "MMQ", "type": "q8_0", "ne11": 2, "count": 2},
        {"route": "MMVQ", "type": "q8_0", "ne11": 1, "count": 1},
    ]
    assert parsed["multiset_sha256"] == hip_routes.parse_hip_matmul_routes(W1_DECODE)["multiset_sha256"]


def test_order_does_not_change_multiset_or_digest():
    reverse = "\n".join(reversed(W1_DECODE.strip().splitlines()))
    assert (hip_routes.parse_hip_matmul_routes(reverse)["hip_matmul_routes"] ==
            hip_routes.parse_hip_matmul_routes(W1_DECODE)["hip_matmul_routes"])


def test_empty_and_malformed_route_logs_fail_closed():
    empty = hip_routes.parse_hip_matmul_routes("ordinary llama log\n")
    malformed = hip_routes.parse_hip_matmul_routes(
        "GGML_CUDA_MUL_MAT_ROUTE route=MMQ type=q8_0 ne11=2\n")
    assert empty["state"] == census.UNKNOWN
    assert not empty["vacuous_guards"]["route_events_gt_zero"]
    assert malformed["state"] == census.UNKNOWN
    assert malformed["parse_errors"]


def test_unknown_route_fails_closed_even_when_another_line_is_valid():
    bad = W1_DECODE + W1_DECODE.splitlines()[0].replace("route=MMQ", "route=NEW_ROUTE") + "\n"
    assert hip_routes.parse_hip_matmul_routes(bad)["state"] == census.UNKNOWN


def test_two_run_entrypoint_sets_level_2_and_requires_structural_identity():
    seen = []
    def runner(argv, *, env, timeout):
        seen.append((list(argv), dict(env), timeout))
        return mock.Mock(returncode=0, stdout="", stderr=W1_DECODE)
    with mock.patch.object(hip_routes.residency, "loader_env", return_value={"BASE": "1"}):
        result = hip_routes.run_aa(Path("/b/bin/llama-bench"), Path("/m/w1.gguf"),
                                   IDENTITY, runner=runner)
    assert result["state"] == census.OBSERVED
    assert result["structurally_identical"]
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0][1]["GGML_CUDA_LOG_MMVQ_ROUTE"] == "2"


def test_two_different_multisets_are_unknown():
    logs = iter([W1_DECODE, W1_DECODE.splitlines()[0] + "\n"])
    def runner(*_args, **_kwargs):
        return mock.Mock(returncode=0, stdout=next(logs), stderr="")
    with mock.patch.object(hip_routes.residency, "loader_env", return_value={}):
        result = hip_routes.run_aa(Path("/b/bin/llama-bench"), Path("/m/w1.gguf"),
                                   IDENTITY, runner=runner)
    assert result["state"] == census.UNKNOWN
    assert not result["structurally_identical"]


def test_cli_is_dry_by_default_and_does_not_read_inputs(capsys):
    rc = hip_routes.main(["--binary", "/missing", "--model", "/missing",
                          "--identity-json", "/missing", "--out", "/missing"])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_identity_missing_build_binding_is_refused():
    parsed = hip_routes.parse_hip_matmul_routes(W1_DECODE)
    try:
        hip_routes.aa_identity(parsed, parsed, {"kernel": {"commit": "abc"}})
    except ValueError as exc:
        assert "identity missing required fields" in str(exc)
    else:
        raise AssertionError("partial identity must fail closed")
