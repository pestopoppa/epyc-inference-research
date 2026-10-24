"""perf-script line parsing and DSO-aware sample classification (AKX-ALL-LB3.2 items (a) and (b)).

The 2026-10-22 premeasure parsed `perf script -F tid,time,ip,sym,dso` lines with a non-greedy regex, which cut every
templated C++ symbol at its first space (`void (anonymous namespace)::tinyBLAS_Q0_AVX<...>::gemm4xN<3>(...)` became
`void`), and classified by symbol only, so a restore memcpy (`[unknown]` in stripped libc) fell through to `other`.

Here the DSO is split off from the RIGHT (`rest.rpartition(" (")`): a symbol may contain " (" itself (anonymous
namespaces, argument lists), a DSO path never does. Classification looks at the DSO first, then the symbol.

Expected field order: `perf script -F pid,tid,time,ip,sym,dso` (`pid/tid` token) or `-F tid,time,ip,sym,dso`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

PERF_FIELDS = "pid,tid,time,ip,sym,dso"


@dataclass(frozen=True)
class Sample:
    pid: Optional[int]
    tid: int
    t: float          # seconds, the perf clock (record with -k CLOCK_MONOTONIC)
    ip: str
    sym: str
    dso: str

    @property
    def main_thread(self) -> bool:
        """The llama-bench main thread (OpenMP thread 0 inside a graph, the host otherwise). Needs the pid field."""
        return self.pid is not None and self.pid == self.tid


def parse_line(line: str) -> Optional[Sample]:
    """One perf-script line -> Sample, or None for anything that is not a sample line."""
    s = line.strip()
    if not s:
        return None
    parts = s.split(None, 3)
    if len(parts) < 4:
        return None
    ids, tok_t, ip, rest = parts
    if not tok_t.endswith(":"):
        return None
    try:
        t = float(tok_t[:-1])
        if "/" in ids:
            p, _, q = ids.partition("/")
            pid, tid = int(p), int(q)
        else:
            pid, tid = None, int(ids)
        int(ip, 16)
    except ValueError:
        return None
    sym, sep, dso = rest.rpartition(" (")
    if sep and dso.endswith(")"):
        dso = dso[:-1]
    else:                       # no DSO field on this line: everything is the symbol
        sym, dso = rest, ""
    return Sample(pid=pid, tid=tid, t=t, ip=ip, sym=sym.strip(), dso=dso.strip())


# ---------------------------------------------------------------------------------------------- classification
# DSO families (basename match)
_DSO_GOMP = re.compile(r"libgomp\.so|libomp\.so|libiomp")
_DSO_LIBC = re.compile(r"(^|/)libc\.so|(^|/)libc-[0-9.]+\.so|ld-linux")
_DSO_LIBLLAMA = re.compile(r"(^|/)libllama\.so|(^|/)libllama-[^/]*\.so|(^|/)llama-bench$")
_DSO_GGML = re.compile(r"(^|/)libggml[^/]*\.so")
_DSO_CXX = re.compile(r"libstdc\+\+\.so|libgcc_s\.so")
_DSO_KERNEL = re.compile(r"kernel\.kallsyms|\[kernel|vmlinux|\[vdso\]")

# symbols inside libllama
_SYM_RESTORE = re.compile(r"state_read|state_write|llama_state_|llama_io_read|llama_io_write|state_seq_")
HOST_SYM_CLASSES = [           # first match wins; all are host (main-thread, outside the graph) work
    ("host_prev_tokens", re.compile(r"get_prev_tokens")),
    ("host_apply_ubatch", re.compile(r"apply_ubatch|find_slot|slot_info")),
    ("host_mask_input", re.compile(r"set_input_kq_mask|kq_mask|set_input_k_idxs|set_input_v_idxs|set_input_pos_bucket")),
    ("host_set_input", re.compile(r"set_input")),
    ("host_graph_build", re.compile(r"build_graph|graph_build|llm_build|build_attn|build_ffn|llm_graph_context|"
                                    r"ggml_backend_sched|ggml_gallocr|ggml_graph_|can_reuse")),
]
# symbols inside ggml (the graph)
GGML_SYM_CLASSES = [
    ("fa", re.compile(r"flash_attn|ggml_vec_dot_f16|ggml_vec_mad1?_f16|ggml_vec_scale_f16|iqk_flash_attn|"
                      r"FlashAttn|fa_kernel")),
    ("indexer", re.compile(r"lightning_indexer|top_k|cmp_top_k|argsort")),
    ("mask", re.compile(r"kq_mask|diag_mask|forward_fill|forward_set_rows")),
    ("ssm_gdn", re.compile(r"gated_delta|ssm_conv|ssm_scan")),
    ("matmul", re.compile(r"mul_mat|vec_dot_q|vec_dot_iq|iqk|tinyBLAS|gemm|gemv|repack|quantize_row|"
                          r"ggml_gemv_|ggml_gemm_|from_float|mul_mat_id")),
    ("get_rows", re.compile(r"get_rows")),
    ("norm", re.compile(r"rms_norm|norm_f32|forward_norm")),
]

CLASSES = ["fa", "indexer", "mask", "ssm_gdn", "matmul", "get_rows", "norm", "ggml_other",
           "omp_spin", "libc", "kernel", "state_restore",
           "host_prev_tokens", "host_apply_ubatch", "host_mask_input", "host_set_input", "host_graph_build",
           "host_llama", "host_cxx", "other"]
HOST_CLASSES = {"host_prev_tokens", "host_apply_ubatch", "host_mask_input", "host_set_input", "host_graph_build",
                "host_llama", "host_cxx"}
# classes that can never be part of a timed denominator, wherever their samples fall
NEVER_TIMED = {"state_restore"}


def classify(sym: str, dso: str) -> str:
    """DSO first, then symbol. A libc sample is `libc` whatever its symbol (stripped libc reports `[unknown]`),
    a libgomp sample is `omp_spin`, a libllama state_read/llama_io_read sample is `state_restore`."""
    if _DSO_GOMP.search(dso):
        return "omp_spin"
    if _DSO_LIBC.search(dso):
        return "libc"
    if _DSO_KERNEL.search(dso):
        return "kernel"
    if _DSO_LIBLLAMA.search(dso):
        if _SYM_RESTORE.search(sym):
            return "state_restore"
        for name, rx in HOST_SYM_CLASSES:
            if rx.search(sym):
                return name
        return "host_llama"
    if _DSO_CXX.search(dso):
        return "host_cxx"
    if _DSO_GGML.search(dso) or "ggml" in sym:
        for name, rx in GGML_SYM_CLASSES:
            if rx.search(sym):
                return name
        return "ggml_other"
    # unknown DSO (or none): fall back to symbol classes so a missing DSO never silently zeroes a class
    if _SYM_RESTORE.search(sym):
        return "state_restore"
    for name, rx in GGML_SYM_CLASSES:
        if rx.search(sym):
            return name
    if re.search(r"gomp_|GOMP_", sym):
        return "omp_spin"
    return "other"


# restore-family on the MAIN thread: what llama_state_seq_set_data runs (libc memcpy, libllama state/apply_ubatch
# cell restore, libstdc++ containers). Used only to locate where a restore starts, never to classify timed work.
def is_restore_family(sample: Sample) -> bool:
    c = classify(sample.sym, sample.dso)
    return c in ("libc", "state_restore", "host_apply_ubatch", "host_llama", "host_cxx")
