"""Float32 CPU reference for MiniCPM-SALA constant-decay Simple GLA.

State update: S[t] = exp(g_gamma) * S[t-1] + k[t] outer v[t].
Output: o[t] = (scale * q[t]) @ S[t]. SALA supplies negative ALiBi slopes
as g_gamma. This is an inference reference, not a Triton performance backend.
"""

import math
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import types
import uuid

import torch


def slopes(n):
    if type(n) is not int or n < 1:
        raise ValueError("head count must be positive")

    def pow2(m):
        a = 2 ** (-(2 ** -(math.log2(m) - 3)))
        return [a ** (i + 1) for i in range(m)]

    p = 1 << (n.bit_length() - 1)
    return torch.tensor(pow2(p) + pow2(2 * p)[0::2][: n - p], dtype=torch.float32)


def simple_gla(q, k, v, g_gamma, scale=None, initial_state=None):
    # SALA input is [batch, time, heads, features]; state is [batch, heads, key, value].
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("incompatible q/k/v shapes")
    b, t, h, dk = q.shape
    if (
        min(b, h, dk, v.shape[-1]) < 1
        or g_gamma.shape != (h,)
        or any(x.device.type != "cpu" for x in (q, k, v, g_gamma))
    ):
        raise ValueError("CPU and nonempty key dimensions required")
    if any(x.dtype != torch.float32 for x in (q, k, v, g_gamma)):
        raise ValueError("SALA passes float32 inputs")
    shape = (b, h, dk, v.shape[-1])
    if initial_state is None:
        state = torch.zeros(shape, dtype=torch.float32)
    elif (
        initial_state.shape != shape
        or initial_state.dtype != torch.float32
        or initial_state.device.type != "cpu"
    ):
        raise ValueError("invalid initial state")
    else:
        state = initial_state.clone()
    multiplier = dk**-0.5 if scale is None else float(scale)
    if not math.isfinite(multiplier):
        raise ValueError("invalid scale")
    decay = g_gamma.exp()[None, :, None, None]
    out = []
    for i in range(t):
        state = decay * state + torch.einsum("bhk,bhv->bhkv", k[:, i], v[:, i])
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * multiplier, state))
    return (torch.stack(out, dim=1) if out else v.new_empty((b, 0, h, v.shape[-1]))), state


def _dispatch(q, k, v, g_gamma, scale, initial_state, output_final_state, cu_seqlens):
    if cu_seqlens is None:
        output, state = simple_gla(q, k, v, g_gamma, scale, initial_state)
    else:
        if (
            q.ndim != 4
            or q.shape[0] != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.device.type != "cpu"
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid packed sequence metadata")
        boundaries = cu_seqlens.tolist()
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != q.shape[1]
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("invalid packed boundaries")
        n = len(boundaries) - 1
        if initial_state is not None and initial_state.shape[0] != n:
            raise ValueError("packed state count")
        rows = []
        states = []
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            state = None if initial_state is None else initial_state[i : i + 1]
            output, final = simple_gla(
                q[:, start:end], k[:, start:end], v[:, start:end], g_gamma, scale, state
            )
            rows.append(output)
            states.append(final)
        output = torch.cat(rows, dim=1)
        state = torch.cat(states, dim=0)
    return output, state if output_final_state else None


def chunk_simple_gla(
    q,
    k,
    v,
    g_gamma=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    head_first=False,
    cu_seqlens=None,
    **kwargs,
):
    if head_first is not False or kwargs or g_gamma is None:
        raise ValueError("unsupported FLA arguments")
    return _dispatch(q, k, v, g_gamma, scale, initial_state, output_final_state, cu_seqlens)


def fused_recurrent_simple_gla(
    q,
    k,
    v,
    g_gamma=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    **kwargs,
):
    if kwargs or g_gamma is None:
        raise ValueError("unsupported FLA arguments")
    return _dispatch(q, k, v, g_gamma, scale, initial_state, output_final_state, cu_seqlens)


@contextmanager
def cpu_fla_modules():
    """Temporarily expose just SALA's FLA imports in an isolated CPU process.

    Existing FLA imports are refused, never replaced. The native SALA module
    binds these function objects during import and can retain them afterward.
    No CUDA branch, Triton kernel, weights or model request runs here.
    """
    names = (
        "fla",
        "fla.ops",
        "fla.ops.simple_gla",
        "fla.ops.simple_gla.fused_recurrent",
        "fla.ops.utils",
        "fla.ops.utils.index",
        "fla.utils",
    )
    if torch.cuda.is_available():
        raise RuntimeError("use a CPU-only PyTorch process for the native SALA CPU oracle")
    if any(name in sys.modules for name in names):
        raise RuntimeError("FLA is already imported; use an isolated benchmark process")
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        sys.modules[name] = module
    modules["fla.ops.simple_gla"].chunk_simple_gla = chunk_simple_gla
    modules[
        "fla.ops.simple_gla.fused_recurrent"
    ].fused_recurrent_simple_gla = fused_recurrent_simple_gla
    modules["fla.utils"].tensor_cache = lambda fn: fn
    index = modules["fla.ops.utils.index"]
    index.prepare_lens_from_mask = lambda mask: mask.sum(-1, dtype=torch.int32)
    index.prepare_cu_seqlens_from_mask = lambda mask: torch.nn.functional.pad(
        torch.cumsum(index.prepare_lens_from_mask(mask), dim=0, dtype=torch.int32), (1, 0)
    )
    try:
        yield
    finally:
        for name, module in modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


def load_native_sala_cpu(directory, expected_sources):
    """Load hash-pinned local HF source using the CPU reference, without weights.

    expected_sources must name exactly configuration_minicpm_sala.py and
    modeling_minicpm_sala.py. Call only in an isolated benchmark process with
    CPU PyTorch and the matching Transformers version. The returned module's
    CPUForCausalLM class adds GenerationMixin compatibility for Transformers
    4.57; loading weights or generating tokens remains the caller's action.
    """
    names = ("configuration_minicpm_sala.py", "modeling_minicpm_sala.py")
    if set(expected_sources) != set(names):
        raise ValueError("provide both exact native source digests")
    directory = Path(directory)
    sources = {}
    for name in names:
        path = directory / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sources[name]:
            raise ValueError("native SALA source digest mismatch: " + name)
        sources[name] = raw
    package_name = "_sala_cpu_" + uuid.uuid4().hex
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    loaded = []
    try:
        with cpu_fla_modules():
            for name in names:
                fullname = package_name + "." + name[:-3]
                module = types.ModuleType(fullname)
                module.__file__ = str(directory / name)
                module.__package__ = package_name
                sys.modules[fullname] = module
                loaded.append(fullname)
                exec(compile(sources[name], module.__file__, "exec"), module.__dict__)
        from transformers.generation import GenerationMixin

        class CPUForCausalLM(module.MiniCPMSALAForCausalLM, GenerationMixin):
            pass

        module.CPUForCausalLM = CPUForCausalLM
        return module
    except BaseException:
        for name in loaded:
            sys.modules.pop(name, None)
        sys.modules.pop(package_name, None)
        raise
