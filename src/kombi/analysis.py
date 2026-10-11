"""Post-run analysis helpers: plotting, FES recovery, ESS diagnostics.

Kept separate from :mod:`kombi.msm` so that the numerical core has no
matplotlib import — useful for headless / cluster runs and for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from kombi.dham import DHAM
from kombi.io import ensure_parent
from kombi.msm import MSM


def plot_fes_1d(
    cv_grid: np.ndarray,
    fes: np.ndarray,
    save_path: Optional[Path] = None,
    bias: Optional[np.ndarray] = None,
    title: str = "Free energy surface",
) -> plt.Figure:
    """Plot ``fes(cv)`` with optional overlay of the current bias."""
    fig, ax = plt.subplots()
    ax.plot(cv_grid, fes - np.nanmin(fes), "k-", label="FES")
    if bias is not None:
        ax2 = ax.twinx()
        ax2.plot(cv_grid, bias, "r--", label="bias")
        ax2.set_ylabel("bias (kJ/mol)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
    ax.set_xlabel("CV")
    ax.set_ylabel("F (kJ/mol)")
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(ensure_parent(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_mfpt_convergence(
    mfpt_per_iteration: list[float],
    save_path: Optional[Path] = None,
    title: str = "MFPT vs KOMBI iteration",
) -> plt.Figure:
    """One number per iteration: the MFPT from start to target after that bias update."""
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(mfpt_per_iteration)), mfpt_per_iteration, "o-")
    ax.set_xlabel("KOMBI iteration")
    ax.set_ylabel("biased MFPT(start → target) (ps)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if save_path is not None:
        fig.savefig(ensure_parent(save_path), dpi=150, bbox_inches="tight")
    return fig


def recover_unbiased_fes(msm_biased: MSM, bias_array: np.ndarray) -> np.ndarray:
    """Apply DHAM reweighting at the MSM level and return the unbiased FES.

    This is the FES that the reviewer wants to see converge as bias is
    incrementally added — it's the quantity that distinguishes "real
    KOMBI" from "just a funnel toward the target".
    """
    M_b = msm_biased.M
    assert M_b is not None
    u = np.asarray(bias_array).reshape(-1)
    # Use the one DHAM reweight (antisymmetric exp((u_i - u_j)/2kT), column
    # convention); this function previously carried its own copy with the
    # symmetric exp((u_i + u_j)/2kT) factor, which does not invert the tilt.
    dham = DHAM(cv_grid=np.arange(u.size, dtype=float), kbT=msm_biased.kbT)
    M_unb = dham.unbias_transition_matrix(M_b, u)

    msm_unb = MSM(kbT=msm_biased.kbT, time_step=msm_biased.time_step)
    msm_unb.build_from_M(M_unb, dim=msm_biased.num_dimensions, time_step=msm_biased.time_step)
    msm_unb.compute_peq_fes(from_="M")
    return msm_unb.free_energy
