"""Dynamic histogram analysis method (DHAM) — reweighting biased to unbiased.

Recovers the unbiased Markov transition matrix from biased trajectories via
the standard Rosta–Hummer reweighting

    K_unbiased[i,j] = K_biased[i,j] · exp( β · (u_i - u_j) / 2 )

(column convention, K[i,j] = rate j -> i; the factor is antisymmetric in i, j)
applied to off-diagonal transitions, with diagonal restored so that columns
sum to 1.  The previous prototype had this same math but with the lag time
and convergence threshold hardcoded inside the per-system folders; this
version pulls both into parameters of :class:`DHAM`.

References
----------
Rosta, E. and Hummer, G. *Free Energies from Dynamic Weighted Histogram
Analysis Using Unbiased Markov State Model.* J. Chem. Theory Comput. **11**,
276 (2015).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DHAM:
    """DHAM reweighting on a single CV grid.

    Parameters
    ----------
    cv_grid
        1D array of CV-bin centres.
    kbT
        Thermal energy (kJ/mol) — must match the simulation temperature.
    lag_time_ps
        Markov lag time τ used when accumulating transition counts.
    max_iter, tol
        Self-consistency parameters when iterating to convergence; the
        default direct (non-iterative) DHAM rarely needs more than one pass
        for a well-converged count matrix.
    """

    cv_grid: np.ndarray
    kbT: float
    lag_time_ps: float = 1.0
    max_iter: int = 100
    tol: float = 1e-6

    def __post_init__(self) -> None:
        self.cv_grid = np.asarray(self.cv_grid).ravel()
        self.n_states = int(self.cv_grid.size)

    # ------------------------------------------------------------------
    # Counts → transition matrix → biased rate matrix
    # ------------------------------------------------------------------
    def count_matrix(self, trajectory_bins: np.ndarray) -> np.ndarray:
        """Build a transition count matrix at the configured lag time.

        Parameters
        ----------
        trajectory_bins
            1D integer array of bin indices, shape ``(n_frames,)``.

        Returns
        -------
        C
            ``(n_states, n_states)`` integer count matrix; ``C[i,j]`` counts
            transitions from state ``i`` at time ``t`` to state ``j`` at
            time ``t + lag_steps``.
        """
        # one trajectory frame == lag_time_ps if we record at lag_time_ps cadence
        bins = np.asarray(trajectory_bins).astype(int)
        C = np.zeros((self.n_states, self.n_states), dtype=np.int64)
        # standard lag-1 count (the trajectory should already be sub-sampled to τ)
        for src, dst in zip(bins[:-1], bins[1:], strict=False):
            if 0 <= src < self.n_states and 0 <= dst < self.n_states:
                C[src, dst] += 1
        return C

    def transition_matrix(self, counts: np.ndarray) -> np.ndarray:
        """Column-stochastic transition matrix from a count matrix.

        Empty states (zero column-sum) are kept as zero columns; ``compute_peq``
        handles them by masking.
        """
        C = np.asarray(counts, dtype=float)
        col_sums = C.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.where(col_sums > 0, C / col_sums, 0.0)
        return M

    # ------------------------------------------------------------------
    # DHAM reweighting
    # ------------------------------------------------------------------
    def unbias_transition_matrix(self, M_biased: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Apply the DHAM correction to a biased transition matrix.

        Column-stochastic convention: ``M[i, j] = P(state_j -> state_i)``, so
        the reweight factor that inverts the detailed-balance-preserving bias
        tilt is **antisymmetric** in (i, j):

        ``M_unbiased[i, j] = M_biased[i, j] * exp((u_i - u_j) / (2 k_B T))``

        applied to every entry, with the diagonal then restored so each column
        sums to unity. The antisymmetric form is Rosta & Hummer, JCTC 11, 276
        (2015); using the symmetric ``exp((u_i + u_j)/(2 k_B T))`` here is a
        recurring bug — it fails to invert the tilt and gives biased free
        energies. A regression test in ``tests/test_dham.py`` guards against
        reintroducing it.

        Parameters
        ----------
        M_biased
            Column-stochastic transition matrix from the biased trajectory.
        bias
            Per-state bias values ``u_i`` (length n_states), same energy units
            as ``self.kbT``.

        Returns
        -------
        M_unbiased
            Column-stochastic unbiased transition matrix.
        """
        M = np.asarray(M_biased, dtype=float)
        u = np.asarray(bias, dtype=float).reshape(-1)
        assert M.shape[0] == M.shape[1] == u.size

        # Antisymmetric reweight: F[i, j] = exp((u_i - u_j) / (2 k_B T))
        F = np.exp((u[:, None] - u[None, :]) / (2.0 * self.kbT))
        M_unb = M * F
        # zero negatives (numerical), zero diagonal, then renormalise columns.
        M_unb = np.clip(M_unb, 0.0, None)
        np.fill_diagonal(M_unb, 0.0)
        col_sums = M_unb.sum(axis=0)
        # set diagonal so each column sums to 1 (population conservation)
        for j in range(M_unb.shape[1]):
            s = col_sums[j]
            if s > 0:
                M_unb[:, j] = M_unb[:, j] / s
        return M_unb

    # ------------------------------------------------------------------
    # Effective sample size diagnostic
    # ------------------------------------------------------------------
    def effective_sample_size_per_state(
        self, counts: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        """Per-state effective sample size after DHAM reweighting.

        ESS_i = (Σ w_j)² / Σ w_j² where the sum is over transitions out of
        state i and ``w_j = exp(β · u_j)``.  Low ESS means a state's
        unbiased estimate is variance-dominated.
        """
        C = np.asarray(counts, dtype=float)
        u = np.asarray(bias, dtype=float).reshape(-1)
        w = np.exp(u / self.kbT)
        # weight outgoing transitions from i by destination bias
        Wsum = (C * w[None, :]).sum(axis=1)
        W2sum = (C * (w[None, :] ** 2)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = np.where(W2sum > 0, Wsum**2 / W2sum, 0.0)
        return ess
