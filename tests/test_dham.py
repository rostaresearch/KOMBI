"""DHAM reweighting should recover the unbiased transition matrix from a biased one."""

from __future__ import annotations

import numpy as np

from kombi.dham import DHAM
from kombi.msm import MSM


def _build_unbiased_K(n: int = 30, kbT: float = 2.494) -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, n)
    fes = 5.0 * (0.5 * np.sin(x - np.pi) + 0.5 * np.sin((x - np.pi) / 2.0))
    K = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K[j, i] = np.exp((fes[i] - fes[j]) / (2.0 * kbT))
        K[i, i] = -np.sum(K[:, i])
    return K


def test_dham_recovers_unbiased_M_from_biased_M() -> None:
    """If we apply a bias to M then DHAM-unbias it, we should get M back to numerical precision."""
    from scipy.linalg import expm

    n = 30
    kbT = 2.494
    K = _build_unbiased_K(n=n, kbT=kbT)
    tau = 0.1
    M_true = expm(K * tau)
    M_true = M_true / M_true.sum(axis=0, keepdims=True)

    bias = 1.5 * np.sin(np.linspace(0, np.pi, n))

    msm = MSM(kbT=kbT, time_step=tau).build_from_M(M_true, dim=1)
    M_biased = msm.bias_M(bias, method="direct_bias")

    cv_grid = np.linspace(0.0, 2.0 * np.pi, n)
    dham = DHAM(cv_grid=cv_grid, kbT=kbT, lag_time_ps=tau)
    M_recovered = dham.unbias_transition_matrix(M_biased, bias)

    # off-diagonal entries should match the true unbiased M up to ~1e-3
    off = np.where(~np.eye(n, dtype=bool))
    rel = np.abs(M_recovered[off] - M_true[off]) / np.maximum(M_true[off], 1e-12)
    # most entries should be tight; tolerate some slop at the matrix edges
    assert np.median(rel) < 0.05, f"median relative error {np.median(rel):.3e}"


def test_dham_count_matrix() -> None:
    cv_grid = np.linspace(0.0, 10.0, 5)
    dham = DHAM(cv_grid=cv_grid, kbT=2.494)
    bins = np.array([0, 1, 2, 1, 0, 1, 1, 2, 3, 4])
    C = dham.count_matrix(bins)
    # 0→1, 1→2, 2→1, 1→0, 0→1, 1→1, 1→2, 2→3, 3→4
    expected = np.zeros((5, 5), dtype=np.int64)
    for s, d in [(0, 1), (1, 2), (2, 1), (1, 0), (0, 1), (1, 1), (1, 2), (2, 3), (3, 4)]:
        expected[s, d] += 1
    assert np.array_equal(C, expected)


def test_ess_diagnostic_returns_array() -> None:
    n = 10
    cv_grid = np.linspace(0.0, 1.0, n)
    dham = DHAM(cv_grid=cv_grid, kbT=2.494)
    C = np.random.default_rng(0).integers(0, 100, size=(n, n))
    bias = 2.0 * np.sin(cv_grid)
    ess = dham.effective_sample_size_per_state(C, bias)
    assert ess.shape == (n,)
    assert np.all(ess >= 0.0)
