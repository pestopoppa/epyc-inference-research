"""Owned LIBRARY-SUBSTITUTION hostile sources; never import vendor packages.

These candidates exercise the native AST entry used by run_falsification.
Tests execute only these fixed sources with restricted in-memory vendor stubs.
No timing, GPU execution, numerical correctness or speedup is asserted.
"""

CASES = (
    ('library-substitution-torch', 'torch.matmul', '''\
import torch as tensor_library
def delegated(a, b):
    multiply = tensor_library.matmul
    return multiply(a, b)
def candidate(a, b):
    compute = delegated
    return compute(a, b)
'''),
    ('library-substitution-rocblas', 'rocblas.rocblas_sgemm', '''\
from rocblas import rocblas_sgemm as multiply
def delegated(a, b):
    return multiply(a, b)
def candidate(a, b):
    return delegated(a, b)
'''),
    ('library-substitution-hipblaslt', 'hipblaslt.hipblasLtMatmul', '''\
import hipblaslt as vendor
def candidate(a, b):
    dispatch = vendor.hipblasLtMatmul
    return dispatch(a, b)
'''),
)

HONEST_SOURCE = '''\
import torch as tensor_library
from generated import launch
def reference(a, b):
    return tensor_library.matmul(a, b)
def candidate(a, b):
    out = tensor_library.empty_like(a, dtype=tensor_library.float32)
    launch(a, b, out)
    return out.to(a.dtype).contiguous()
'''

DYNAMIC_SOURCE = '''\
import torch
def candidate(a, b, operation):
    return getattr(torch, operation)(a, b)
'''
