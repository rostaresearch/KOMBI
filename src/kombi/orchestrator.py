"""Top-level KOMBI iteration loop.

For now this handles the **model-potential** systems (1D and 2D analytic
toys).  All-atom systems live in ``examples/*/run.py`` and use the same
algorithm modules but with an OpenMM-based propagator.

The control flow per iteration:

  1.   Run a propagation under the current bias (synthetic for toy
       systems: directly evolve the biased Markov matrix).
  2.   Build counts → biased M → DHAM-corrected unbiased K.
  3.   Optimise Gaussian bias parameters against the unbiased K to minimise
       MFPT(start, target).
  4.   Persist the bias and iteration metrics.

Stops on convergence (target MFPT below threshold or no improvement for
``patience`` iterations) or after ``RunConfig.n_iterations``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kombi.analysis import plot_fes_1d, plot_mfpt_convergence, recover_unbiased_fes
from kombi.bias import GaussianBias, optimize_bias_parameters
from kombi.config import RunConfig
from kombi.io import Paths
from kombi.msm import MSM


@dataclass
class RunResult:
    """Container for the artefacts of one KOMBI run."""

    config: RunConfig
    paths: Paths
    final_mfpt: float
    mfpt_per_iteration: list[float] = field(default_factory=list)
    bias_per_iteration: list[GaussianBias] = field(default_factory=list)
    final_bias: GaussianBias | None = None
    final_fes: np.ndarray | None = None


def _cv_index(cv: float, cv_grid: np.ndarray) -> int:
    """Closest-bin index on a 1D CV grid."""
    return int(np.argmin(np.abs(cv_grid - cv)))


def run_kombi_1d(
    cfg: RunConfig,
    K0: np.ndarray,
    cv_grid: np.ndarray | None = None,
) -> RunResult:
    """Run KOMBI on a 1D system with a known reference rate matrix.

    Used both for toy benchmarking (the manuscript's 1D potential) and as a
    fast smoke test that the algorithm modules wire together cleanly.

    Parameters
    ----------
    cfg
        Run configuration.  ``cfg.dim`` must be 1.
    K0
        Reference unbiased rate matrix of shape ``(N, N)`` — exactly what
        the prototype's ``MSM._init_sample_K(dim=1)`` produced.  Pass in
        explicitly so this function is decoupled from the choice of
        analytical potential.
    cv_grid
        Optional pre-computed CV grid.  Default: ``cfg.cv_grid``.
    """
    if cv_grid is None:
        cv_grid = cfg.cv_grid
    rng = np.random.default_rng(cfg.seed)
    paths = Paths(root=cfg.out_dir)

    n = K0.shape[0]
    assert n == cv_grid.size, "K0 and cv_grid must have the same dimension"

    start_idx = _cv_index(cfg.start_cv, cv_grid)
    target_idx = _cv_index(cfg.target_cv, cv_grid)

    msm = MSM(kbT=cfg.kbT, time_step=cfg.integrator_dt_ps, verbose=cfg.verbose)
    msm.build_from_K(K0, dim=1, time_step=cfg.integrator_dt_ps)

    # Initial bias = zero
    bias = GaussianBias(
        amplitudes=np.zeros(cfg.n_gaussians),
        centres=np.linspace(*cfg.cv_range, cfg.n_gaussians),
        widths=np.full(cfg.n_gaussians, 0.5),
    )

    mfpts_log: list[float] = []
    bias_log: list[GaussianBias] = []

    for k in range(cfg.n_iterations):
        # 1) Apply current bias to K, get biased M
        u = bias.evaluate(cv_grid)
        K_biased = msm.bias_K(u)
        # build M = expm(K * dt); we use the bias_M direct route on a fresh MSM
        msm_b = MSM(kbT=cfg.kbT, time_step=cfg.integrator_dt_ps).build_from_K(
            K_biased, dim=1, time_step=cfg.integrator_dt_ps
        )
        from scipy.linalg import expm

        M_b = expm(K_biased * cfg.integrator_dt_ps)
        # column-normalise (numerical clean-up)
        col_sums = M_b.sum(axis=0)
        M_b[:, col_sums > 0] /= col_sums[col_sums > 0]
        msm_b.build_from_M(M_b, dim=1, time_step=cfg.integrator_dt_ps)
        msm_b.compute_peq_fes(from_="M")

        # 2) Search for new bias minimising MFPT on this MSM
        new_bias, new_mfpt = optimize_bias_parameters(
            msm=msm_b,
            cv_grid=cv_grid,
            start_idx=start_idx,
            target_idx=target_idx,
            n_gaussians=cfg.n_gaussians,
            amp_bounds=cfg.bias_amplitude_bounds,
            width_bounds=cfg.bias_width_bounds,
            tol=cfg.lbfgs_tol,
            n_restarts=max(1, cfg.lbfgs_restarts_per_gaussian),
            rng=rng,
        )
        mfpts_log.append(new_mfpt)
        bias_log.append(new_bias)
        np.save(paths.bias(f"bias_iter_{k:03d}.npy"), new_bias.to_vector())

        if cfg.verbose:
            print(f"[iter {k:3d}]  MFPT(start→target) = {new_mfpt:.3f} ps")

        bias = new_bias

    # Final plots
    final_fes = recover_unbiased_fes(msm_b, bias.evaluate(cv_grid))
    plot_fes_1d(
        cv_grid,
        final_fes,
        save_path=paths.fig("final_fes.png"),
        bias=bias.evaluate(cv_grid),
        title=f"{cfg.name}: recovered FES + final bias",
    )
    plot_mfpt_convergence(
        mfpts_log,
        save_path=paths.fig("mfpt_convergence.png"),
        title=f"{cfg.name}: MFPT vs iteration",
    )

    return RunResult(
        config=cfg,
        paths=paths,
        final_mfpt=mfpts_log[-1] if mfpts_log else float("inf"),
        mfpt_per_iteration=mfpts_log,
        bias_per_iteration=bias_log,
        final_bias=bias,
        final_fes=final_fes,
    )


def run_kombi(cfg: RunConfig, **kwargs) -> RunResult:
    """Dispatch the right backend for ``cfg.system``.

    Currently only ``"model_1d"`` is implemented in this orchestrator;
    all-atom systems live in their example folders, which call the algorithm
    modules directly with an OpenMM-based propagator.
    """
    if cfg.system == "model_1d":
        return run_kombi_1d(cfg, **kwargs)
    raise NotImplementedError(
        f"system={cfg.system!r} is not yet wired into the top-level run_kombi(). "
        "See examples/ras_gap_pmg/run.py for the OpenMM-based pattern."
    )
