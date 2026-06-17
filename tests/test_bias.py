"""GaussianBias serialization + L-BFGS-B optimisation sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from kombi.bias import GaussianBias, optimize_bias_parameters
from kombi.msm import MSM


def test_gaussian_bias_round_trip() -> None:
    b = GaussianBias(
        amplitudes=np.array([1.0, -0.5, 2.0]),
        centres=np.array([0.0, 1.0, 2.5]),
        widths=np.array([0.3, 0.5, 0.4]),
    )
    v = b.to_vector()
    assert v.shape == (9,)
    b2 = GaussianBias.from_vector(v)
    np.testing.assert_allclose(b2.amplitudes, b.amplitudes)
    np.testing.assert_allclose(b2.centres, b.centres)
    np.testing.assert_allclose(b2.widths, b.widths)


def test_gaussian_bias_evaluate() -> None:
    # Single Gaussian centred at 0, width 1, amplitude 2
    b = GaussianBias(amplitudes=[2.0], centres=[0.0], widths=[1.0])
    x = np.array([0.0, 1.0, -1.0, 3.0])
    expected = 2.0 * np.exp(-0.5 * (x**2))
    np.testing.assert_allclose(b.evaluate(x), expected)


def test_optimize_bias_finds_lower_mfpt_than_zero_bias() -> None:
    """Even one restart should beat the zero-bias MFPT on a double-well."""
    from scipy.linalg import expm

    n = 30
    kbT = 2.494
    x = np.linspace(0.0, 2.0 * np.pi, n)
    fes = 5.0 * (0.5 * np.sin(x - np.pi) + 0.5 * np.sin((x - np.pi) / 2.0))

    K = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K[j, i] = np.exp((fes[i] - fes[j]) / (2.0 * kbT))
        K[i, i] = -np.sum(K[:, i])

    tau = 0.1
    M = expm(K * tau)
    M /= M.sum(axis=0, keepdims=True)

    msm = MSM(kbT=kbT, time_step=tau).build_from_M(M, dim=1).compute_peq_fes(from_="M")
    msm.build_mfpt_matrix(method="diag", check=False)
    baseline_mfpt = msm.mfpts[0, n // 2 + 5]
    assert np.isfinite(baseline_mfpt)

    best_bias, best_mfpt = optimize_bias_parameters(
        msm=msm,
        cv_grid=x,
        start_idx=0,
        target_idx=n // 2 + 5,
        n_gaussians=3,
        amp_bounds=(-3.0, 3.0),
        width_bounds=(0.3, 1.0),
        n_restarts=5,
        rng=np.random.default_rng(0),
    )
    assert best_mfpt < baseline_mfpt, (
        f"Optimised MFPT {best_mfpt:.3f} should beat zero-bias baseline {baseline_mfpt:.3f}"
    )
