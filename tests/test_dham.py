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


def test_dham_reweight_uses_antisymmetric_exponent() -> None:
    """REGRESSION GUARD: the DHAM reweight must use the *antisymmetric*
    ``exp((u_i - u_j) / (2 k_B T))`` factor (column convention). The
    symmetric ``exp((u_i + u_j) / (2 k_B T))`` form does NOT invert the
    detailed-balance-preserving bias tilt and gives ~2e-3 errors that
    propagate into biased free energies — this bug has been reintroduced
    multiple times.

    Tests the raw reweight factor (before column renormalisation) recovers
    M_true off-diagonal to machine precision from a manually tilted M.
    """
    n = 12
    kbT = 2.5
    rng = np.random.default_rng(0)
    V = rng.uniform(-1, 1, n)

    # Manually construct M_true (column-stochastic, detailed balance)
    M_true = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) <= 2:
                # column convention M[i, j] = P(j -> i)
                M_true[i, j] = 0.05 * np.exp((V[j] - V[i]) / (2.0 * kbT))
    for j in range(n):
        M_true[j, j] = 1.0 - M_true[:, j].sum()

    # Apply an additive bias u — build M_biased by the same DB rule on V + u
    bias = rng.uniform(-0.5, 0.5, n)
    Vb = V + bias
    M_biased = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) <= 2:
                M_biased[i, j] = 0.05 * np.exp((Vb[j] - Vb[i]) / (2.0 * kbT))
    for j in range(n):
        M_biased[j, j] = 1.0 - M_biased[:, j].sum()

    # DHAM undo: the RAW reweight (before the internal col-renorm) should
    # give back M_true off-diagonal to machine precision if and only if
    # the sign convention is antisymmetric.
    cv_grid = np.linspace(0.0, 1.0, n)
    dham = DHAM(cv_grid=cv_grid, kbT=kbT)
    F_antisym = np.exp((bias[:, None] - bias[None, :]) / (2.0 * kbT))
    F_sym = np.exp((bias[:, None] + bias[None, :]) / (2.0 * kbT))
    off = ~np.eye(n, dtype=bool)
    err_antisym = np.max(np.abs((M_biased * F_antisym - M_true)[off]))
    err_sym = np.max(np.abs((M_biased * F_sym - M_true)[off]))

    assert err_antisym < 1e-12, (
        f"Antisymmetric reweight should be exact: got max err {err_antisym:.3e}"
    )
    assert err_sym > 1e-3, (
        "Symmetric reweight sanity-check failed: expected it to be off by "
        f">1e-3; got {err_sym:.3e}. If this triggers the test is broken, "
        "not the code."
    )

    # Round-trip through the actual dham.unbias_transition_matrix code:
    # with the antisymmetric fix in place, off-diagonal ratios within each
    # column must recover M_true's; with the symmetric buggy form the
    # recovered M is off by O(1) on entries whose bias difference is large.
    M_rec = dham.unbias_transition_matrix(M_biased, bias)
    off = ~np.eye(n, dtype=bool)
    # Absolute recovery: dham normalises col sums to 1 and so does M_true;
    # off-diagonal ratios within a column depend only on F's shape, so
    # ratio recovery is the strongest column-invariant check.
    # For the antisymmetric form the recovered off-diagonals should match
    # M_true's after both are col-normalised to 1.
    def _col_norm(A):
        Ac = A.copy()
        np.fill_diagonal(Ac, 0.0)
        s = Ac.sum(axis=0, keepdims=True)
        return np.divide(Ac, s, where=s > 0, out=np.zeros_like(Ac))

    diff = _col_norm(M_rec) - _col_norm(M_true)
    max_ratio_err = np.max(np.abs(diff[off]))
    assert max_ratio_err < 1e-10, (
        f"Round-trip DHAM recovery failed: max off-diagonal col-normalised "
        f"error {max_ratio_err:.3e}. Wrong sign in dham.unbias_transition_matrix?"
    )


def test_dham_source_uses_antisymmetric_form() -> None:
    """Direct source inspection: the reweight literal must be antisymmetric.

    A round-trip numerical test can be defeated by clever renormalisation.
    This source-level assertion is the last line of defence against
    reintroducing the (u_i + u_j) symmetric form in
    ``kombi.dham.DHAM.unbias_transition_matrix``.
    """
    import inspect

    from kombi.dham import DHAM as _DHAM

    src = inspect.getsource(_DHAM.unbias_transition_matrix)
    # The correct antisymmetric exponent (column convention).
    assert "u[:, None] - u[None, :]" in src, (
        "DHAM reweight must contain the ANTISYMMETRIC exponent "
        "'u[:, None] - u[None, :]' (column convention); found source:\n" + src
    )
    # And must NOT contain the symmetric form.
    assert "u[:, None] + u[None, :]" not in src, (
        "DHAM reweight contains the SYMMETRIC exp((u_i+u_j)/(2kT)) form — "
        "this is the recurring bug that gives biased free energies. Use "
        "the antisymmetric (u_i - u_j) form; see docstring for derivation."
    )


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
