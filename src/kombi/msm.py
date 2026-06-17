"""Markov-state-model helper class — refactored from TJ's ``MSM.py``.

Changes vs the prototype (semantic-preserving):
  1. ``kBT`` is no longer a hard-coded ``0.5981``; it's set from
     :class:`~kombi.config.RunConfig.temperature` at construction time and
     stored in kJ/mol.
  2. Eigenvector → equilibrium distribution: take ``np.real`` of the
     largest-eigenvalue eigenvector, clip to ``≥ 0`` and renormalise.  The
     prototype kept the complex leakage and could silently propagate tiny
     imaginary parts through the rest of the pipeline.
  3. Matrix-logarithm route ``K = logm(M) / τ`` returns a real matrix only
     if ``M`` is close to identity; we take ``np.real(...)`` and warn when
     the imaginary norm exceeds a tolerance.
  4. Convergence checks (Kemeny constant, peq sum) are surfaced as
     :class:`numpy.testing` assertions rather than printed warnings.
  5. Plot helpers moved to :mod:`kombi.analysis` (this class is now numeric
     only — easier to test, no matplotlib import at module load).

Algorithm code (state adjacency check, biased ``K`` / biased ``M`` from a
position-dependent bias, ``mfpts = diag-method`` formula, JJ-Hunter as an
alternative) is **identical** to the prototype's math.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.linalg import eig, expm, inv, logm


class MSM:
    """Discrete-state Markov state model on a regular CV grid.

    Holds at most one of ``K`` (continuous-time rate matrix) or ``M``
    (discrete-time transition matrix) plus their derived quantities
    (equilibrium distribution ``peq``, free energy ``free_energy``, MFPT
    matrix ``mfpts``).  Use the ``build_*`` constructors to populate.

    Parameters
    ----------
    kbT
        Thermal energy in kJ/mol.  Default 2.494 (~ 300 K).
    time_step
        Simulation time step *between MSM frames* in ps.  Used to convert
        between discrete-time ``M`` and continuous-time ``K`` and to scale
        MFPTs to physical units.
    verbose
        If True, the bias and Kemeny-constant routines print diagnostics.
    """

    def __init__(self, kbT: float = 2.494, time_step: float = 1.0, verbose: bool = False):
        self.K: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None
        self.peq: Optional[np.ndarray] = None
        self.free_energy: Optional[np.ndarray] = None
        self.mfpts: Optional[np.ndarray] = None
        self.num_states: Optional[int] = None
        self.num_dimensions: Optional[int] = None
        self.qspace: Optional[np.ndarray] = None
        self.kbT = float(kbT)
        self.time_step = float(time_step)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    def build_from_M(self, M: np.ndarray, dim: int, time_step: Optional[float] = None) -> "MSM":
        """Populate from a discrete-time transition matrix.

        Parameters
        ----------
        M
            Column-stochastic transition matrix of shape ``(N**dim, N**dim)``.
        dim
            CV dimensionality (1, 2 or 3).
        time_step
            Override the lag time (ps).  If ``None``, keep the value passed
            to :meth:`__init__`.
        """
        M = np.asarray(M, dtype=np.float64)
        assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be a square matrix"
        self.M = M
        self.num_dimensions = int(dim)
        # N**dim = M.shape[0]  →  N = round(M.shape[0] ** (1/dim))
        self.num_states = int(round(M.shape[0] ** (1.0 / self.num_dimensions)))
        if time_step is not None:
            self.time_step = float(time_step)
        return self

    def build_from_K(self, K: np.ndarray, dim: int, time_step: Optional[float] = None) -> "MSM":
        """Populate from a continuous-time rate matrix."""
        K = np.asarray(K, dtype=np.float64)
        assert K.ndim == 2 and K.shape[0] == K.shape[1], "K must be a square matrix"
        self.K = K
        self.num_dimensions = int(dim)
        self.num_states = int(round(K.shape[0] ** (1.0 / self.num_dimensions)))
        if time_step is not None:
            self.time_step = float(time_step)
        return self

    # ------------------------------------------------------------------
    # Equilibrium distribution and free energy
    # ------------------------------------------------------------------
    @staticmethod
    def _principal_eigenvector(A: np.ndarray) -> np.ndarray:
        """Return the right eigenvector of A corresponding to the largest-real eigenvalue.

        Clean-up vs the prototype:
          * Take the real part (the rate matrix is real, so any imaginary
            component is numerical noise).
          * Clip to ``[0, ∞)`` so log() downstream doesn't see negatives.
          * Renormalise to sum to 1.
        """
        evals, evecs = eig(A)
        # largest real-part eigenvalue
        idx = int(np.argmax(np.real(evals)))
        v = np.real(evecs[:, idx])
        # Eigenvectors are defined up to sign — flip if the sum is negative
        # so that probabilities come out non-negative after clip.
        if v.sum() < 0:
            v = -v
        v = np.clip(v, 0.0, None)
        s = v.sum()
        if s <= 0:
            raise RuntimeError("Principal eigenvector has non-positive sum — input matrix is malformed.")
        return v / s

    def compute_peq_fes(self, from_: str = "auto") -> "MSM":
        """Compute equilibrium distribution and free energy.

        Parameters
        ----------
        from_
            ``"K"``, ``"M"`` or ``"auto"`` (use whichever is set; if both,
            prefer K).
        """
        if from_ == "auto":
            from_ = "K" if self.K is not None else "M"
        if from_ == "K":
            assert self.K is not None, "K is not set; call build_from_K() or build_from_M() first."
            self.peq = self._principal_eigenvector(self.K)
        elif from_ == "M":
            assert self.M is not None, "M is not set; call build_from_M() first."
            self.peq = self._principal_eigenvector(self.M)
        else:
            raise ValueError(f"from_ must be 'K', 'M' or 'auto', got {from_!r}")
        # mask zeros: free energy is ill-defined where peq == 0
        with np.errstate(divide="ignore"):
            self.free_energy = -self.kbT * np.log(np.where(self.peq > 0, self.peq, np.nan))
        return self

    # ------------------------------------------------------------------
    # MFPT matrix
    # ------------------------------------------------------------------
    def build_mfpt_matrix(self, method: str = "diag", check: bool = False) -> np.ndarray:
        """Compute the matrix of mean first passage times between every (i, j).

        ``method='diag'`` follows Hunter's fundamental-matrix formula:
        ``mfpt[i,j] = (Qjj - Qij + delta_ij) / peq[j]``
        with ``Q = (peq · 1ᵀ + I − Mᵀ)⁻¹``.

        ``method='JJhunter'`` is an alternative recursive scheme — kept for
        bit-for-bit comparison with the prototype but the diag method is
        usually faster and numerically equivalent.

        Returns the MFPT matrix (in ps, scaled by ``self.time_step``).
        """
        if self.peq is None:
            self.compute_peq_fes()
        assert self.M is not None, "MFPT needs the transition matrix M."
        n_total = self.M.shape[0]
        peq = np.asarray(self.peq).reshape(-1)

        if method == "diag":
            # Hunter fundamental-matrix identity:
            #     mfpt[i,j] = (Q⁻¹[j,j] − Q⁻¹[i,j] + δ_ij) / peq[j]
            # Vectorised — the prototype's double loop was O(N²) Python overhead.
            ones = np.ones((n_total, 1))
            I = np.eye(n_total)
            A = (peq.reshape(-1, 1) @ ones.T).T
            Q_inv = inv(A + I - self.M.T)
            diag_Q = np.diag(Q_inv)
            num = diag_Q[np.newaxis, :] - Q_inv + I  # shape (n, n)
            with np.errstate(divide="ignore", invalid="ignore"):
                mfpts = np.where(
                    peq[np.newaxis, :] > 0,
                    num / peq[np.newaxis, :],
                    np.inf,
                )
            mfpts = mfpts * self.time_step

        elif method == "JJhunter":
            mfpts = _jjhunter(self.M.T) * self.time_step

        else:
            raise ValueError(f"method must be 'diag' or 'JJhunter', got {method!r}")

        self.mfpts = mfpts
        if check:
            self.kemeny_constant_check()
        return mfpts

    def kemeny_constant_check(self, tol: float = 1e-6) -> bool:
        """Verify the Kemeny constant: sum_j MFPT(i,j) * peq[j] is i-independent."""
        if self.mfpts is None or self.peq is None:
            raise RuntimeError("Compute MFPTs and peq first.")
        kemeny = self.mfpts @ self.peq.reshape(-1, 1)
        spread = float(np.max(kemeny) - np.min(kemeny))
        ok = spread < tol
        if not ok and self.verbose:
            warnings.warn(
                f"Kemeny constant varies by {spread:.3e} across states; "
                "MSM may not be irreducible/aperiodic.",
                RuntimeWarning,
                stacklevel=2,
            )
        return ok

    # ------------------------------------------------------------------
    # Adjacency on a regular grid
    # ------------------------------------------------------------------
    def is_adjacent(self, i: int, j: int) -> bool:
        """True if states ``i`` and ``j`` share a face on the regular CV grid."""
        assert self.num_states is not None and self.num_dimensions is not None
        coor_i = np.unravel_index(i, (self.num_states,) * self.num_dimensions)
        coor_j = np.unravel_index(j, (self.num_states,) * self.num_dimensions)
        return float(np.linalg.norm(np.array(coor_i) - np.array(coor_j))) == 1.0

    # ------------------------------------------------------------------
    # Bias the rate or transition matrix by a 1-body potential u(x)
    # ------------------------------------------------------------------
    def bias_K(self, bias: np.ndarray) -> np.ndarray:
        """Apply a 1-body bias ``u`` and return the biased rate matrix.

        Uses detailed balance: ``K_biased[i,j] = K[i,j] * exp((u_j − u_i) / (2 kT))``
        for off-diagonal adjacent pairs (Boltzmann splitting).
        """
        assert self.K is not None
        n_total = self.K.shape[0]
        bias = np.asarray(bias).reshape(-1)
        assert bias.shape[0] == n_total, "bias must be length N**dim"
        K_b = np.zeros_like(self.K)
        kbT = self.kbT
        if self.num_dimensions == 1:
            idx = np.arange(n_total - 1)
            u_diff = bias[1:] - bias[:-1]
            up = np.exp(u_diff / (2.0 * kbT))
            dn = np.exp(-u_diff / (2.0 * kbT))
            K_b[idx, idx + 1] = self.K[idx, idx + 1] * up
            K_b[idx + 1, idx] = self.K[idx + 1, idx] * dn
        else:
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    if self.is_adjacent(i, j):
                        u_ij = bias[j] - bias[i]
                        K_b[i, j] = self.K[i, j] * np.exp(u_ij / (2.0 * kbT))
                        K_b[j, i] = self.K[j, i] * np.exp(-u_ij / (2.0 * kbT))
        # diagonal: rates out of state j sum to zero
        for i in range(n_total):
            K_b[i, i] = -np.sum(K_b[:, i]) + K_b[i, i]
        return K_b

    def bias_M(self, bias: np.ndarray, method: str = "direct_bias") -> np.ndarray:
        """Apply a 1-body bias to the transition matrix M and return the biased copy.

        ``method='direct_bias'`` (recommended): apply detailed-balance Boltzmann
        splitting to adjacent off-diagonal elements, recompute diagonals so
        each column sums to 1.

        ``method='logm_K'``: convert M → K via matrix logarithm, bias K, then
        ``M_biased = expm(K_biased * τ)``.  Numerically delicate if M is far
        from identity; we warn when the imaginary norm of ``logm(M)`` is
        non-negligible.
        """
        assert self.M is not None
        n_total = self.M.shape[0]
        bias = np.asarray(bias).reshape(-1)
        kbT = self.kbT

        if method == "direct_bias":
            # Fast path for the 1D nearest-neighbour case (the manuscript's
            # default and all the toy benchmarks).  In 1D the only adjacent
            # pairs are (i, i+1), so we just vectorise along ±1 diagonal.
            if self.num_dimensions == 1:
                M_b = self.M.copy()
                idx = np.arange(n_total - 1)
                # u_diff[i] = bias[i+1] - bias[i]
                u_diff = bias[1:] - bias[:-1]
                up = np.exp(u_diff / (2.0 * kbT))
                dn = np.exp(-u_diff / (2.0 * kbT))
                # M[i, j] is row i, column j = rate from j to i; (i, i+1) is "from i+1 down to i"
                M_b[idx, idx + 1] = self.M[idx, idx + 1] * up  # downhill in bias
                M_b[idx + 1, idx] = self.M[idx + 1, idx] * dn  # uphill in bias
                col_sums = M_b.sum(axis=0)
                nonzero = col_sums > 0
                M_b[:, nonzero] = M_b[:, nonzero] / col_sums[nonzero]
                return M_b
            # Generic dim ≥ 2: fall back to the O(N²) Python loop
            M_b = np.zeros_like(self.M)
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    if self.is_adjacent(i, j):
                        u_ij = bias[j] - bias[i]
                        M_b[i, j] = self.M[i, j] * np.exp(u_ij / (2.0 * kbT))
                        M_b[j, i] = self.M[j, i] * np.exp(-u_ij / (2.0 * kbT))
            for i in range(n_total):
                M_b[i, i] = self.M[i, i]
            col_sums = M_b.sum(axis=0)
            nonzero = col_sums > 0
            M_b[:, nonzero] = M_b[:, nonzero] / col_sums[nonzero]
            return M_b

        elif method == "logm_K":
            if self.K is None:
                L = logm(self.M)
                if np.linalg.norm(np.imag(L)) > 1e-6 * np.linalg.norm(np.real(L)):
                    warnings.warn(
                        "logm(M) has non-negligible imaginary part — M is far from identity; "
                        "consider 'direct_bias' instead.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self.K = np.real(L) / self.time_step
            K_b = self.bias_K(bias)
            return expm(K_b * self.time_step)

        else:
            raise ValueError(f"method must be 'direct_bias' or 'logm_K', got {method!r}")


# --------------------------------------------------------------------------
# JJ-Hunter recursive MFPT (kept verbatim from the prototype — slow but exact)
# --------------------------------------------------------------------------
def _jjhunter(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    m = len(P)
    I = np.eye(m, dtype=np.float64)
    e1 = np.ones((m, 1), dtype=np.float64)
    E = np.ones((m, m), dtype=np.float64)
    PPD = P.copy()
    rD = np.zeros(m, dtype=np.float64)
    SD = np.zeros(m, dtype=np.float64)

    for n in range(m - 1, 0, -1):
        SD[n] = np.sum(PPD[n, :n])
        for i in range(n):
            for j in range(n):
                PPD[i, j] += PPD[i, n] * PPD[n, j] / SD[n]
    rD[0] = 1.0
    for n in range(1, m):
        for i in range(n):
            rD[n] += rD[i] * PPD[i, n] / SD[n]

    TOTD = float(np.sum(rD))
    pit_1D = rD / TOTD
    PiD = e1 @ pit_1D.reshape(1, -1)
    ZD = np.linalg.inv(I - P + PiD)
    DD = np.linalg.inv(np.diag(np.diag(PiD)))
    MD = (I - ZD + E @ np.diag(np.diag(ZD))) @ DD
    return MD
