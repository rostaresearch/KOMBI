"""Algorithm-level sanity checks for the MSM class."""

from __future__ import annotations

import numpy as np
import pytest

from kombi.msm import MSM


def _build_test_K(n: int = 50, kbT: float = 2.494, amplitude: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Tilted double-well on [0, 2π], discretised into nearest-neighbour rates."""
    x = np.linspace(0.0, 2.0 * np.pi, n)
    y1 = np.sin(x - np.pi)
    y2 = np.sin((x - np.pi) / 2.0)
    fes = amplitude * (0.5 * y1 + 0.5 * y2)
    # Detailed-balance Boltzmann-split rates: K[j,i] = rate from i to j.
    # Going downhill (fes[j] < fes[i]) should have rate > 1, so the exponent is
    # +(fes[i] - fes[j]) / (2 kT) — matches the prototype's _init_sample_K.
    K = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K[j, i] = np.exp((fes[i] - fes[j]) / (2.0 * kbT))
        K[i, i] = -np.sum(K[:, i])
    return K, fes


def test_peq_from_K_is_real_nonnegative_and_normalized() -> None:
    K, _ = _build_test_K()
    msm = MSM(kbT=2.494).build_from_K(K, dim=1).compute_peq_fes(from_="K")
    peq = msm.peq
    assert np.iscomplexobj(peq) is False or np.allclose(np.imag(peq), 0.0)
    assert np.all(peq >= 0.0)
    assert pytest.approx(peq.sum()) == 1.0


def test_free_energy_matches_input_fes_up_to_constant() -> None:
    K, fes_in = _build_test_K()
    msm = MSM(kbT=2.494).build_from_K(K, dim=1).compute_peq_fes(from_="K")
    F = msm.free_energy
    # Detailed balance encoded in K means F = fes_in + const
    # The "const" is the integration constant for normalization
    F_aligned = F - np.nanmin(F)
    in_aligned = fes_in - fes_in.min()
    # Boltzmann splitting (u/2 in K) means the recovered FES has amplitude
    # equal to fes_in (the splitting cancels at the equilibrium distribution).
    rms = np.sqrt(np.nanmean((F_aligned - in_aligned) ** 2))
    assert rms < 0.5, f"FES mismatch RMS = {rms:.3f} kJ/mol"


def test_kemeny_constant() -> None:
    from scipy.linalg import expm

    K, _ = _build_test_K(n=20)
    tau = 0.1
    M = expm(K * tau)
    # column-normalise
    M = M / M.sum(axis=0, keepdims=True)
    msm = MSM(kbT=2.494, time_step=tau).build_from_M(M, dim=1).compute_peq_fes(from_="M")
    msm.build_mfpt_matrix(method="diag", check=False)
    assert msm.kemeny_constant_check(tol=1e-3)


def test_bias_M_preserves_column_stochasticity() -> None:
    from scipy.linalg import expm

    K, _ = _build_test_K(n=20)
    M = expm(K * 0.1)
    M = M / M.sum(axis=0, keepdims=True)
    msm = MSM(kbT=2.494, time_step=0.1).build_from_M(M, dim=1)
    bias = 2.0 * np.sin(np.linspace(0, np.pi, 20))
    M_b = msm.bias_M(bias, method="direct_bias")
    assert np.allclose(M_b.sum(axis=0), 1.0, atol=1e-9)
