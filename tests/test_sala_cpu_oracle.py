import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
SPEC = importlib.util.spec_from_file_location(
    "sala_cpu_oracle", Path(__file__).parents[1] / "scripts/benchmark/sala_cpu_oracle.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def inputs(length=7):
    torch.manual_seed(17)
    return (
        torch.randn(2, length, 3, 4),
        torch.randn(2, length, 3, 4),
        torch.randn(2, length, 3, 5),
        -m.slopes(3),
        torch.randn(2, 3, 4, 5),
    )


@pytest.mark.parametrize("length", [0, 1, 7, 63, 64, 65, 129])
def test_reference_matches_independent_causal_sum_and_continuation(length):
    q, k, v, g, initial = inputs(length)
    actual, state = m.simple_gla(q, k, v, g, initial_state=initial)
    expected = []
    for i in range(length):
        row = (
            torch.einsum("bhk,bhkv->bhv", q[:, i] * 0.5, initial)
            * torch.exp(g * (i + 1))[None, :, None]
        )
        for j in range(i + 1):
            row += (
                (q[:, i] * k[:, j]).sum(-1)[..., None]
                * v[:, j]
                * 0.5
                * torch.exp(g * (i - j))[None, :, None]
            )
        expected.append(row)
    want = torch.stack(expected, dim=1) if expected else v.new_empty((2, 0, 3, 5))
    torch.testing.assert_close(actual, want, rtol=3e-5, atol=3e-5)
    split = length // 2
    a, mid = m.simple_gla(q[:, :split], k[:, :split], v[:, :split], g, initial_state=initial)
    b, end = m.simple_gla(q[:, split:], k[:, split:], v[:, split:], g, initial_state=mid)
    torch.testing.assert_close(torch.cat((a, b), dim=1), actual)
    torch.testing.assert_close(end, state)
    assert torch.equal(initial, inputs(length)[4])


def test_packed_sequences_reset_state_and_optional_final_state():
    q, k, v, g, state = inputs(7)
    boundaries = torch.tensor([0, 3, 7], dtype=torch.int32)
    actual, final = m.chunk_simple_gla(
        q[:1],
        k[:1],
        v[:1],
        g_gamma=g,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=boundaries,
    )
    a, sa = m.simple_gla(q[:1, :3], k[:1, :3], v[:1, :3], g, initial_state=state[:1])
    b, sb = m.simple_gla(q[:1, 3:], k[:1, 3:], v[:1, 3:], g, initial_state=state[1:])
    torch.testing.assert_close(actual, torch.cat((a, b), dim=1))
    torch.testing.assert_close(final, torch.cat((sa, sb), dim=0))
    _, omitted = m.fused_recurrent_simple_gla(q, k, v, g_gamma=g)
    assert omitted is None
    with pytest.raises(ValueError):
        m.chunk_simple_gla(q[:1], k[:1], v[:1], g_gamma=g, cu_seqlens=torch.tensor([0, 8]))


def test_refuses_invalid_shapes_dtypes_and_scale():
    q, k, v, g, state = inputs()
    for args in ((q, k, v, g[:2]), (q.double(), k, v, g)):
        with pytest.raises(ValueError):
            m.simple_gla(*args)
    with pytest.raises(ValueError):
        m.simple_gla(q, k, v, g, scale=float("nan"))
    with pytest.raises(ValueError):
        m.simple_gla(q, k, v, g, initial_state=state[:1])


def test_shim_restores_modules_and_refuses_overwriting_existing_fla():
    assert "fla" not in sys.modules
    with m.cpu_fla_modules():
        assert sys.modules["fla.ops.simple_gla"].chunk_simple_gla is m.chunk_simple_gla
        with pytest.raises(RuntimeError):
            with m.cpu_fla_modules():
                pass
    assert "fla" not in sys.modules


def test_native_sala_attention_and_slopes_when_source_is_available():
    value = os.environ.get("SALA_HF_SOURCE_DIR")
    if not value:
        pytest.skip("set SALA_HF_SOURCE_DIR for the optional pinned vendor-source control")
    root = Path(value)
    expected = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("configuration_minicpm_sala.py", "modeling_minicpm_sala.py")
    }
    assert (
        expected["modeling_minicpm_sala.py"]
        == "b6060c08cfd45868f0e90726e48292689b15f01bf9828cf07f6257530477719c"
    )
    assert (
        expected["configuration_minicpm_sala.py"]
        == "0bd0668a2c771911c6cc5bf64481a28b23a56aa63dbb60cbc9c75031b6b2cae2"
    )
    native = m.load_native_sala_cpu(root, expected)
    assert hasattr(native.CPUForCausalLM, "generate")
    for heads in (1, 3, 7, 32):
        torch.testing.assert_close(m.slopes(heads), native._build_slope_tensor(heads))
    for length in (5, 65):
        q, k, v, g, state = inputs(length)
        mask = torch.ones(2, length, dtype=torch.int64)
        mask[1, :2] = 0
        actual, final = native.LightningAttention.attn_fn(
            None, q, k, v, g, initial_state=state, attention_mask=mask
        )
        for row in range(2):
            indices = mask[row].bool()
            want, last = m.simple_gla(
                q[row : row + 1, indices],
                k[row : row + 1, indices],
                v[row : row + 1, indices],
                g,
                initial_state=state[row : row + 1],
            )
            torch.testing.assert_close(actual[row, indices], want[0])
            torch.testing.assert_close(final[row], last[0])
            assert torch.count_nonzero(actual[row, ~indices]) == 0
    with pytest.raises(ValueError):
        m.load_native_sala_cpu(root, dict(expected, **{"modeling_minicpm_sala.py": "0" * 64}))
