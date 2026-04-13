#!/usr/bin/env python3
"""Attention Matching KV Cache Compaction — HighestAttnKeys-fast variant.

Port of adamzweiger/compaction (MIT license) for EPYC CPU evaluation.
Paper: arxiv:2602.16284 (Zweiger, Fu, Guo, Yoon Kim)

Algorithm overview:
  1. Score each key position by attention weight (RMS/max/mean across query set)
  2. Select top-t positions as compact keys C1
  3. Fit per-position bias beta via NNLS to preserve attention mass
  4. Fit compact values C2 via OLS to preserve attention output

Usage:
    from attention_matching import AttentionMatchingCompactor

    compactor = AttentionMatchingCompactor(score_method='max')
    C1, beta, C2, indices = compactor.compact(K, V, queries, target_size=t)

    # Decode with compact cache:
    # scores = Q @ C1.T / sqrt(d) + beta
    # attn = softmax(scores)
    # output = attn @ C2
"""

import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import numpy as np
import torch


@dataclass
class CompactionResult:
    """Result of KV cache compaction."""
    C1: torch.Tensor          # (t, d) compact keys
    beta: torch.Tensor        # (t,) per-position bias
    C2: torch.Tensor          # (t, d) compact values
    indices: List[int]        # indices of selected keys in original K
    compression_ratio: float  # T / t
    timing: dict = field(default_factory=dict)  # per-step wall-clock seconds


class AttentionMatchingCompactor:
    """HighestAttnKeys-fast KV cache compaction.

    Selects keys with highest attention scores, fits bias via NNLS,
    fits values via OLS. All operations are closed-form (no gradient descent).

    Parameters
    ----------
    score_method : str
        How to score key importance: 'max', 'rms', or 'mean'.
    beta_method : str
        How to compute bias: 'nnls' (fit to match attention mass) or 'zero'.
    c2_solver : str
        OLS solver: 'lstsq' (default, most robust), 'cholesky', or 'pinv'.
    c2_ridge_lambda : float
        Ridge regularization for C2 solve. 0 = no regularization.
    """

    def __init__(
        self,
        score_method: str = 'max',
        beta_method: str = 'nnls',
        c2_solver: str = 'lstsq',
        c2_ridge_lambda: float = 0.0,
    ):
        assert score_method in ('max', 'rms', 'mean')
        assert beta_method in ('nnls', 'zero')
        assert c2_solver in ('lstsq', 'cholesky', 'pinv')
        self.score_method = score_method
        self.beta_method = beta_method
        self.c2_solver = c2_solver
        self.c2_ridge_lambda = c2_ridge_lambda

    def compact(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        target_size: int,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> CompactionResult:
        """Compact KV cache from T entries to target_size entries.

        Parameters
        ----------
        K : (T, d) original keys
        V : (T, d) original values
        queries : (n, d) query vectors for fitting (from calibration set)
        target_size : int
            Number of compact KV entries (t)
        attention_bias : (T,) or (n, T), optional
            Additive bias on original attention scores

        Returns
        -------
        CompactionResult with C1, beta, C2, indices, compression_ratio, timing
        """
        T, d = K.shape
        n = queries.shape[0]
        t = target_size
        assert t <= T, f"target_size {t} > original size {T}"
        assert queries.shape[1] == d

        timing = {}

        # Step 1: Compute attention weights (fp32 for numerical stability)
        t0 = time.perf_counter()
        inv_sqrt_d = (1.0 / d) ** 0.5
        scores_raw = queries.float() @ K.float().T  # (n, T)
        scores32 = scores_raw * inv_sqrt_d

        if attention_bias is not None:
            scores32 = scores32 + attention_bias.float().broadcast_to(scores32.shape)

        max_scores = scores32.max(dim=1, keepdim=True).values  # numerical stability
        exp_scores = torch.exp(scores32 - max_scores)  # (n, T)
        sum_exp = exp_scores.sum(dim=1, keepdim=True)
        attn_weights = exp_scores / sum_exp  # (n, T) normalized
        timing['attention'] = time.perf_counter() - t0

        # Step 2: Score and select top-t keys
        t0 = time.perf_counter()
        if self.score_method == 'rms':
            key_scores = torch.sqrt((attn_weights ** 2).mean(dim=0))
        elif self.score_method == 'max':
            key_scores = attn_weights.max(dim=0).values
        else:  # mean
            key_scores = attn_weights.mean(dim=0)

        _, top_indices = torch.topk(key_scores, t, largest=True)
        top_indices_sorted = top_indices.sort().values  # preserve order
        C1 = K[top_indices_sorted]  # (t, d)
        timing['selection'] = time.perf_counter() - t0

        # Step 3: Fit beta via NNLS
        t0 = time.perf_counter()
        if self.beta_method == 'zero':
            beta = torch.zeros(t, dtype=K.dtype)
        else:
            beta = self._fit_beta_nnls(exp_scores, top_indices_sorted, t)
        timing['nnls'] = time.perf_counter() - t0

        # Step 4: Fit C2 via OLS
        t0 = time.perf_counter()
        C2 = self._fit_c2(C1, beta, K, V, queries, attention_bias)
        timing['ols'] = time.perf_counter() - t0

        # Convert back to original dtype
        beta_out = beta.to(K.dtype)
        C2_out = C2.to(K.dtype)

        return CompactionResult(
            C1=C1,
            beta=beta_out,
            C2=C2_out,
            indices=top_indices_sorted.cpu().tolist(),
            compression_ratio=T / t,
            timing=timing,
        )

    def _fit_beta_nnls(
        self,
        exp_scores: torch.Tensor,
        selected_indices: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Fit bias beta via NNLS to match attention partition function.

        Solves: min_B ||M @ B - target||^2  s.t. B >= 0
        Then: beta = log(B)

        M = exp_scores[:, selected] (n, t)
        target = exp_scores.sum(dim=1) (n,)
        """
        M = exp_scores[:, selected_indices].float()  # (n, t)
        target = exp_scores.sum(dim=1).float()  # (n,)

        # Solve via lstsq + clamp (fast path, matches reference impl with nnls_iters=0)
        try:
            B = torch.linalg.lstsq(M, target.unsqueeze(1), driver='gels').solution.squeeze(1)
            if torch.isnan(B).any():
                raise RuntimeError("NaN in lstsq")
        except Exception:
            # Fallback: regularized solve
            n, t_ = M.shape
            if n >= t_:
                MtM = M.T @ M
                MtM = 0.5 * (MtM + MtM.T)
                MtM.diagonal().add_(1e-6)
                L = torch.linalg.cholesky(MtM)
                Mty = M.T @ target
                B = torch.cholesky_solve(Mty.unsqueeze(1), L).squeeze(1)
            else:
                MMt = M @ M.T
                MMt = 0.5 * (MMt + MMt.T)
                MMt.diagonal().add_(1e-6)
                L = torch.linalg.cholesky(MMt)
                alpha = torch.cholesky_solve(target.unsqueeze(1), L).squeeze(1)
                B = M.T @ alpha

        # Clamp to positive (NNLS constraint) and compute log
        B = B.clamp(min=1e-12)
        beta = torch.log(B)  # (t,) fp32

        return beta

    def _fit_c2(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fit compact values C2 via OLS.

        Solves: X @ C2 = Y
        Where:
            Y = softmax(Q @ K.T / sqrt(d)) @ V   (original attention output)
            X = softmax(Q @ C1.T / sqrt(d) + beta)  (compact attention weights)
        """
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Y = softmax(QK/sqrt(d)) @ V
        sK = (queries.float() @ K.float().T) * inv_sqrt_d
        if attention_bias is not None:
            sK = sK + attention_bias.float().broadcast_to(sK.shape)
        m_K = sK.max(dim=1, keepdim=True).values
        exp_K = torch.exp(sK - m_K)
        attn_K = exp_K / exp_K.sum(dim=1, keepdim=True)
        Y = attn_K @ V.float()  # (n, d)

        # X = softmax(Q @ C1.T / sqrt(d) + beta)
        sC = (queries.float() @ C1.float().T) * inv_sqrt_d + beta.float().unsqueeze(0)
        m_C = sC.max(dim=1, keepdim=True).values
        exp_C = torch.exp(sC - m_C)
        X = exp_C / exp_C.sum(dim=1, keepdim=True)  # (n, t)

        # Solve X @ C2 = Y
        lam = self.c2_ridge_lambda
        n, t = X.shape

        if self.c2_solver == 'lstsq' and lam == 0:
            try:
                C2 = torch.linalg.lstsq(X, Y, driver='gels').solution
                if not torch.isnan(C2).any():
                    return C2
            except Exception:
                pass
            # Fallback to cholesky with small regularization
            lam = 1e-6

        # Ridge regression
        if n >= t:
            XtX = X.T @ X
            XtX = 0.5 * (XtX + XtX.T)
            XtX.diagonal().add_(lam)
            L = torch.linalg.cholesky(XtX)
            XtY = X.T @ Y
            C2 = torch.cholesky_solve(XtY, L)
        else:
            XXt = X @ X.T
            XXt = 0.5 * (XXt + XXt.T)
            XXt.diagonal().add_(lam)
            L = torch.linalg.cholesky(XXt)
            Z = torch.cholesky_solve(Y, L)
            C2 = X.T @ Z

        return C2


def evaluate_compaction_quality(
    K: torch.Tensor,
    V: torch.Tensor,
    queries: torch.Tensor,
    result: CompactionResult,
    eval_queries: Optional[torch.Tensor] = None,
) -> dict:
    """Evaluate quality of compacted cache vs original.

    Compares attention outputs on eval_queries (or queries if not provided):
        original_output = softmax(Q @ K.T / sqrt(d)) @ V
        compact_output  = softmax(Q @ C1.T / sqrt(d) + beta) @ C2

    Returns dict with:
        mse: mean squared error between outputs
        cosine_sim: mean cosine similarity
        max_abs_error: maximum absolute error
        relative_error: ||compact - original|| / ||original||
    """
    Q = (eval_queries if eval_queries is not None else queries).float()
    K_f, V_f = K.float(), V.float()
    d = K.shape[1]
    inv_sqrt_d = (1.0 / d) ** 0.5

    # Original output
    sK = (Q @ K_f.T) * inv_sqrt_d
    m_K = sK.max(dim=1, keepdim=True).values
    exp_K = torch.exp(sK - m_K)
    attn_K = exp_K / exp_K.sum(dim=1, keepdim=True)
    orig_out = attn_K @ V_f  # (n, d)

    # Compact output
    C1_f = result.C1.float()
    beta_f = result.beta.float()
    C2_f = result.C2.float()
    sC = (Q @ C1_f.T) * inv_sqrt_d + beta_f.unsqueeze(0)
    m_C = sC.max(dim=1, keepdim=True).values
    exp_C = torch.exp(sC - m_C)
    attn_C = exp_C / exp_C.sum(dim=1, keepdim=True)
    comp_out = attn_C @ C2_f  # (n, d)

    # Metrics
    diff = comp_out - orig_out
    mse = (diff ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(orig_out, comp_out, dim=1).mean().item()
    max_abs = diff.abs().max().item()
    orig_norm = orig_out.norm().item()
    rel_error = diff.norm().item() / max(orig_norm, 1e-12)

    return {
        'mse': mse,
        'cosine_similarity': cosine,
        'max_abs_error': max_abs,
        'relative_error': rel_error,
    }


def run_synthetic_validation(
    T: int = 512,
    d: int = 128,
    n_queries: int = 64,
    compression_ratios: list = None,
    seed: int = 42,
) -> dict:
    """Run validation on synthetic random KV cache.

    Useful for verifying the implementation is correct before running
    on real model outputs. Random data is a worst case (no structure).

    Returns dict mapping compression_ratio -> quality metrics.
    """
    if compression_ratios is None:
        compression_ratios = [2, 5, 10]

    torch.manual_seed(seed)
    K = torch.randn(T, d)
    V = torch.randn(T, d)
    queries = torch.randn(n_queries, d)

    # Split queries: half for fitting, half for evaluation
    fit_queries = queries[:n_queries // 2]
    eval_queries = queries[n_queries // 2:]

    compactor = AttentionMatchingCompactor(score_method='max', beta_method='nnls')
    results = {}

    for ratio in compression_ratios:
        t = max(1, T // ratio)
        result = compactor.compact(K, V, fit_queries, target_size=t)
        quality = evaluate_compaction_quality(K, V, fit_queries, result, eval_queries)
        results[ratio] = {
            **quality,
            'target_size': t,
            'timing': result.timing,
        }

    return results


if __name__ == '__main__':
    import json
    import sys

    print("Attention Matching KV Compaction — Synthetic Validation")
    print("=" * 60)

    results = run_synthetic_validation(
        T=512, d=128, n_queries=64,
        compression_ratios=[2, 5, 10, 20],
    )

    for ratio, metrics in sorted(results.items()):
        print(f"\n{ratio}x compression (T=512 -> t={metrics['target_size']}):")
        print(f"  MSE:              {metrics['mse']:.6f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"  Relative error:    {metrics['relative_error']:.6f}")
        print(f"  Timing: selection={metrics['timing']['selection']:.3f}s, "
              f"nnls={metrics['timing']['nnls']:.3f}s, "
              f"ols={metrics['timing']['ols']:.3f}s")

    # Also test with larger synthetic (closer to real model dimensions)
    print("\n" + "=" * 60)
    print("Large synthetic (T=4096, d=128, n=256):")
    large_results = run_synthetic_validation(
        T=4096, d=128, n_queries=256,
        compression_ratios=[5, 10, 20],
    )
    for ratio, metrics in sorted(large_results.items()):
        print(f"\n{ratio}x compression (T=4096 -> t={metrics['target_size']}):")
        print(f"  MSE:              {metrics['mse']:.6f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"  Relative error:    {metrics['relative_error']:.6f}")
        print(f"  Timing: selection={metrics['timing']['selection']:.3f}s, "
              f"nnls={metrics['timing']['nnls']:.3f}s, "
              f"ols={metrics['timing']['ols']:.3f}s")

    # Write results to JSON for persistence
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    if output_path:
        combined = {'synthetic_512': results, 'synthetic_4096': large_results}
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nResults written to {output_path}")
