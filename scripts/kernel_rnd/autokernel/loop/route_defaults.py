"""Default (generated) CPU kernel routes: file family -> oracle class.

AKX-AUTO-9.  `gates.CPU_SOURCE_ROUTES` is hand-listed per function, so a new hot
file has no route until a human writes one (2026-10-06: Q38FN's IQ3_S/IQ4_NL kernels
had none).  This pure module classifies every compiled CPU kernel file by FAMILY and
names the oracle class that can judge an edit to it:

  scalar_reference_nmse  quant GEMM / dequant / repack / quantize kernels: the
                         independent scalar reference (`cpu_quant_reference`) plus
                         op-level NMSE (`ppl_contract`).
  fa_anchor_bits        CPU flash-attention: `cpu_fa_reference` anchor-vs-candidate bits.
  model_identity        graph / scheduling / elementwise-op / loader code: whole-model
                         greedy output identity (`model_identity`).

and REFUSES files for which no oracle can exist: build files, headers and shared
config/feature code compiled into BOTH arms, stubs, tests, and trees not compiled on
this host.  It never overrides a hand-listed route: a file that `gates` already names
keeps its hand-listed verdict (including the standing `PPL_CONTRACT_NO_OP_ORACLE`
refusals); the default only covers files no hand-listed route names.  A default route
says the FILE is in scope with a named oracle -- it does not admit any particular
patch; hunk confinement still belongs to the hand-listed route table.
"""
from __future__ import annotations

from dataclasses import dataclass
import posixpath

SCALAR_NMSE = "scalar_reference_nmse"
FA_ANCHOR_BITS = "fa_anchor_bits"
MODEL_IDENTITY = "model_identity"
ORACLE_CLASSES = (SCALAR_NMSE, FA_ANCHOR_BITS, MODEL_IDENTITY)

_CPU = "ggml/src/ggml-cpu/"

#: Exact-path families (checked first).
_EXACT = {
    _CPU + "ggml-cpu.c": ("graph_scheduling", MODEL_IDENTITY),
    _CPU + "ggml-cpu.cpp": ("graph_scheduling", MODEL_IDENTITY),
    _CPU + "traits.cpp": ("graph_scheduling", MODEL_IDENTITY),
    _CPU + "hbm.cpp": ("graph_scheduling", MODEL_IDENTITY),
    _CPU + "ops.cpp": ("graph_ops", MODEL_IDENTITY),
    _CPU + "vec.cpp": ("graph_ops", MODEL_IDENTITY),
    _CPU + "unary-ops.cpp": ("graph_ops", MODEL_IDENTITY),
    _CPU + "binary-ops.cpp": ("graph_ops", MODEL_IDENTITY),
    _CPU + "quants.c": ("quant_kernel", SCALAR_NMSE),
    _CPU + "repack.cpp": ("quant_kernel", SCALAR_NMSE),
    _CPU + "arch/x86/quants.c": ("quant_kernel", SCALAR_NMSE),
    _CPU + "arch/x86/repack.cpp": ("quant_kernel", SCALAR_NMSE),
    _CPU + "llamafile/sgemm.cpp": ("quant_kernel", SCALAR_NMSE),
    _CPU + "iqk/iqk_flash_attn.cpp": ("flash_attention", FA_ANCHOR_BITS),
    _CPU + "iqk/iqk_mul_mat.cpp": ("quant_kernel", SCALAR_NMSE),
    _CPU + "iqk/iqk_dispatch.cpp": ("quant_kernel", SCALAR_NMSE),
    _CPU + "iqk/iqk_cpu_ops.cpp": ("graph_ops", MODEL_IDENTITY),
}

#: Exact-path refusals: (reason).
_REFUSED_EXACT = {
    _CPU + "CMakeLists.txt": "build file: no kernel code, no oracle",
    _CPU + "iqk/iqk_stubs.cpp": "disabled-build stub: not compiled in the GGML_IQK=1 arms",
    _CPU + "arch/x86/cpu-feats.cpp": "feature detection shared by both arms",
    _CPU + "iqk/iqk_config.h": "configuration shared by both arms",
}

_MODEL_PREFIXES = ("src/models/",)
_MODEL_FILES = ("src/llama-model-loader.cpp", "src/llama-graph.cpp", "src/llama-model.cpp")
_NOT_COMPILED_DIRS = ("kleidiai/", "spacemit/", "amx/")


@dataclass(frozen=True)
class DefaultRoute:
    path: str
    family: str
    oracle_class: str
    numerics: str            # "ppl_contract" | "bit_exact" | "model_identity"
    route: str               # generated route name, never collides with a hand-listed one

    def as_dict(self) -> dict:
        return {"path": self.path, "family": self.family, "oracle_class": self.oracle_class,
                "numerics": self.numerics, "route": self.route, "basis": "default_route"}


@dataclass(frozen=True)
class RouteRefusal:
    path: str
    reason: str


_NUMERICS = {SCALAR_NMSE: "ppl_contract", FA_ANCHOR_BITS: "bit_exact",
             MODEL_IDENTITY: "model_identity"}


def _norm(path: str) -> str:
    return posixpath.normpath(str(path).replace("\\", "/")).lstrip("./") if path else ""


def classify_path(path: str) -> "DefaultRoute | RouteRefusal":
    """The default route for `path`, or the refusal naming why no oracle can exist."""
    p = _norm(path)
    if not p:
        return RouteRefusal(str(path), "empty path")
    base = posixpath.basename(p)
    if base == "CMakeLists.txt" or base.endswith((".cmake", ".txt")):
        return RouteRefusal(p, "build file: no kernel code, no oracle")
    if p in _REFUSED_EXACT:
        return RouteRefusal(p, _REFUSED_EXACT[p])
    if "/tests/" in p or p.startswith("tests/") or base.startswith("test-"):
        return RouteRefusal(p, "test code: not part of the compiled kernel")
    if p.startswith(_CPU):
        rel = p[len(_CPU):]
        if rel.startswith(_NOT_COMPILED_DIRS):
            return RouteRefusal(p, "tree not compiled on this x86 host: nothing to measure or judge")
        if rel.startswith("cmake/"):
            return RouteRefusal(p, "build file: no kernel code, no oracle")
        if base.endswith((".h", ".hpp")):
            return RouteRefusal(p, "header shared by both arms or declarations only: edit the "
                                   "compiled .c/.cpp that owns the kernel")
        if p in _EXACT:
            family, oracle = _EXACT[p]
        elif rel.startswith("iqk/iqk_gemm_") or rel.startswith("iqk/iqk_quantize"):
            family, oracle = "quant_kernel", SCALAR_NMSE
        elif rel.startswith("iqk/") and base.endswith((".c", ".cpp")):
            family, oracle = "graph_ops", MODEL_IDENTITY
        elif base.endswith((".c", ".cpp")):
            # A compiled kernel file nobody classified yet is still in scope: whole-model
            # identity can judge any edit to code that runs in the model.
            family, oracle = "unclassified_kernel", MODEL_IDENTITY
        else:
            return RouteRefusal(p, "not a compiled kernel source")
    elif p.startswith(_MODEL_PREFIXES) or p in _MODEL_FILES:
        if not base.endswith((".cpp", ".c")):
            return RouteRefusal(p, "header: edit the compiled source")
        family, oracle = "graph_scheduling", MODEL_IDENTITY
    else:
        return RouteRefusal(p, "not a compiled CPU kernel file")
    return DefaultRoute(p, family, oracle, _NUMERICS[oracle], f"default:{family}")


def default_route(path: str) -> "DefaultRoute | None":
    c = classify_path(path)
    return c if isinstance(c, DefaultRoute) else None


def refusal_for(path: str) -> "str | None":
    c = classify_path(path)
    return c.reason if isinstance(c, RouteRefusal) else None


def default_route_uncovered_by_hand_list(path: str, hand_listed_paths) -> "DefaultRoute | None":
    """The default route only for a file no hand-listed route names."""
    if _norm(path) in {_norm(x) for x in hand_listed_paths}:
        return None
    return default_route(path)


__all__ = ["DefaultRoute", "RouteRefusal", "ORACLE_CLASSES", "SCALAR_NMSE", "FA_ANCHOR_BITS",
           "MODEL_IDENTITY", "classify_path", "default_route", "refusal_for",
           "default_route_uncovered_by_hand_list"]
