"""recover_unbiased_fes must use the same (antisymmetric) DHAM reweight as kombi.dham."""

from __future__ import annotations

import inspect

import numpy as np

from kombi.analysis import recover_unbiased_fes
from kombi.dham import DHAM
from kombi.msm import MSM


def _db_matrix(V: np.ndarray, kbT: float) -> np.ndarray:
    """Column-stochastic, detailed-balance banded matrix, M[i, j] = P(j -> i)."""
    n = V.size
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) <= 2:
                M[i, j] = 0.05 * np.exp((V[j] - V[i]) / (2.0 * kbT))
    for j in range(n):
        M[j, j] = 1.0 - M[:, j].sum()
    return M


def test_recover_unbiased_fes_source_has_no_symmetric_factor() -> None:
    """REGRESSION GUARD: recover_unbiased_fes once carried its own copy of the
    reweight with the symmetric exp((u_i + u_j)/2kT) factor (fixed in
    kombi.dham on 18 Jul 2026 but not here). It must delegate to
    DHAM.unbias_transition_matrix."""
    src = inspect.getsource(recover_unbiased_fes)
    assert "u[:, None] + u[None, :]" not in src, src
    assert "unbias_transition_matrix" in src, src


def test_recover_unbiased_fes_matches_dham_reweight() -> None:
    n, kbT = 12, 2.5
    rng = np.random.default_rng(1)
    V = rng.uniform(-1, 1, n)
    bias = rng.uniform(-0.5, 0.5, n)
    M_biased = _db_matrix(V + bias, kbT)

    msm_b = MSM(kbT=kbT, time_step=1.0)
    msm_b.build_from_M(M_biased, dim=1, time_step=1.0)
    fes = recover_unbiased_fes(msm_b, bias)

    M_ref = DHAM(cv_grid=np.arange(n, dtype=float), kbT=kbT).unbias_transition_matrix(M_biased, bias)
    ref = MSM(kbT=kbT, time_step=1.0)
    ref.build_from_M(M_ref, dim=1, time_step=1.0)
    ref.compute_peq_fes(from_="M")
    assert np.allclose(fes - np.nanmin(fes), ref.free_energy - np.nanmin(ref.free_energy), atol=1e-10)
