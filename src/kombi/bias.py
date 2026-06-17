"""Sum-of-Gaussians bias kernel + L-BFGS-B optimisation of the parameters.

The bias is

    u(x) = Σ_k  a_k · exp(- (x − μ_k)² / (2 σ_k²))

for the 1D case, and the analogous separable form in 2D.  Bias parameters are
flattened into a 1D vector of length 3·N_g (1D) or 5·N_g (2D) for the
optimiser, with the convention ``[a₁, μ₁, σ₁, a₂, μ₂, σ₂, …]``.

The optimisation step:

    Given the current unbiased rate matrix K and a target state s,
    find the bias parameter vector θ that minimises MFPT_biased(start → s)

uses ``scipy.optimize.minimize`` with method ``L-BFGS-B`` and bounded
parameters (amplitude ∈ ``RunConfig.bias_amplitude_bounds`` × kT, width ∈
``RunConfig.bias_width_bounds`` in CV-units).  The previous prototype did a
single L-BFGS-B run; we instead do
``RunConfig.lbfgs_restarts_per_gaussian × N_g`` restarts from random initial
guesses (matching the procedure stated in the manuscript).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from kombi.msm import MSM


@dataclass
class GaussianBias:
    """Sum-of-Gaussians bias defined on a regular CV grid.

    Parameters
    ----------
    amplitudes
        Length-N_g array of Gaussian heights (kJ/mol, may be negative).
    centres
        Length-N_g array of Gaussian centres in CV-units.
    widths
        Length-N_g array of Gaussian standard deviations in CV-units.
    """

    amplitudes: np.ndarray
    centres: np.ndarray
    widths: np.ndarray

    def __post_init__(self) -> None:
        self.amplitudes = np.asarray(self.amplitudes, dtype=float).ravel()
        self.centres = np.asarray(self.centres, dtype=float).ravel()
        self.widths = np.asarray(self.widths, dtype=float).ravel()
        n = self.amplitudes.size
        assert self.centres.size == n and self.widths.size == n, "amp/centre/width sizes differ"

    @property
    def n_gaussians(self) -> int:
        return int(self.amplitudes.size)

    # ------------------------------------------------------------------
    # Vector ↔ struct conversions for the optimiser
    # ------------------------------------------------------------------
    def to_vector(self) -> np.ndarray:
        """Flatten to ``[a₁, μ₁, σ₁, a₂, μ₂, σ₂, …]`` for L-BFGS-B."""
        return np.column_stack([self.amplitudes, self.centres, self.widths]).ravel()

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "GaussianBias":
        v = np.asarray(vec).reshape(-1, 3)
        return cls(amplitudes=v[:, 0], centres=v[:, 1], widths=v[:, 2])

    # ------------------------------------------------------------------
    # Evaluate the bias on a 1D CV grid
    # ------------------------------------------------------------------
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Return ``u(x)`` on a 1D CV grid ``x`` of arbitrary shape."""
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        for a, mu, sig in zip(self.amplitudes, self.centres, self.widths, strict=True):
            out = out + a * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        return out


def _bias_vec_to_bias_array(theta: np.ndarray, cv_grid: np.ndarray) -> np.ndarray:
    """Flatten optimisation vector to a per-state bias array via :class:`GaussianBias`."""
    return GaussianBias.from_vector(theta).evaluate(cv_grid)


def _mfpt_objective_factory(
    msm: MSM,
    cv_grid: np.ndarray,
    start_idx: int,
    target_idx: int,
) -> Callable[[np.ndarray], float]:
    """Return a closure ``f(theta) → MFPT(start, target)`` for L-BFGS-B."""

    def f(theta: np.ndarray) -> float:
        bias = _bias_vec_to_bias_array(theta, cv_grid)
        M_b = msm.bias_M(bias, method="direct_bias")
        msm_local = MSM(kbT=msm.kbT, time_step=msm.time_step, verbose=False)
        msm_local.build_from_M(M_b, dim=msm.num_dimensions, time_step=msm.time_step)
        msm_local.compute_peq_fes(from_="M")
        try:
            mfpts = msm_local.build_mfpt_matrix(method="diag", check=False)
            return float(mfpts[start_idx, target_idx])
        except Exception:
            # singular matrices etc.  Return a large penalty so L-BFGS-B
            # backs out of this region without crashing.
            return 1e12

    return f


def optimize_bias_parameters(
    msm: MSM,
    cv_grid: np.ndarray,
    start_idx: int,
    target_idx: int,
    n_gaussians: int,
    amp_bounds: tuple[float, float],
    width_bounds: tuple[float, float],
    tol: float = 1e-4,
    n_restarts: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[GaussianBias, float]:
    """Find Gaussian-bias parameters minimising MFPT(start → target).

    Parameters
    ----------
    msm
        Current MSM estimate (must already have ``M`` and ``peq`` populated).
    cv_grid
        1D CV-grid centres of length ``N_states``.
    start_idx, target_idx
        Indices into ``cv_grid`` defining the source and target states.
    n_gaussians
        Number of Gaussian kernels in the bias.
    amp_bounds, width_bounds
        Bounds passed to L-BFGS-B for each amplitude and width.  Centres are
        bounded by the CV range.
    tol
        Convergence tolerance on the MFPT objective.
    n_restarts
        Number of random initial guesses (best result is returned).
    rng
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    best_bias
        :class:`GaussianBias` corresponding to the lowest MFPT found.
    best_mfpt
        The MFPT at that bias.
    """
    if rng is None:
        rng = np.random.default_rng()

    cv_lo, cv_hi = float(cv_grid.min()), float(cv_grid.max())
    centre_bounds = (cv_lo, cv_hi)
    bounds = []
    for _ in range(n_gaussians):
        bounds.extend([amp_bounds, centre_bounds, width_bounds])

    objective = _mfpt_objective_factory(msm, cv_grid, start_idx, target_idx)

    best_mfpt = np.inf
    best_theta: np.ndarray | None = None
    for _ in range(n_restarts):
        theta0 = np.empty(3 * n_gaussians)
        for k in range(n_gaussians):
            theta0[3 * k] = rng.uniform(*amp_bounds)
            theta0[3 * k + 1] = rng.uniform(cv_lo, cv_hi)
            theta0[3 * k + 2] = rng.uniform(*width_bounds)
        try:
            res = minimize(
                objective,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                tol=tol,
                options={"maxiter": 200},
            )
            if res.fun < best_mfpt and np.isfinite(res.fun):
                best_mfpt = float(res.fun)
                best_theta = res.x
        except Exception:
            # let other restarts try; the bias landscape can be brutal
            continue

    if best_theta is None:
        raise RuntimeError(
            "L-BFGS-B failed on every restart — check that the MSM is irreducible and "
            "that the bounds are not pathologically tight."
        )

    return GaussianBias.from_vector(best_theta), best_mfpt
