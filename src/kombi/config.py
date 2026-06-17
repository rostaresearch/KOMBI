"""Run-time configuration for KOMBI.

Centralises every knob in a single :class:`RunConfig` dataclass so that the
rest of the codebase never has to read global ``kBT = 0.5981`` magic numbers
or hard-coded ``psf_file = 'step3_input.psf'`` paths.

A config can be built in Python, loaded from a YAML file via
:func:`load_config`, or constructed by the CLI from command-line arguments.

Physical constants are computed from the SI ``scipy.constants.k`` so changing
the temperature really changes the simulation; the previous prototype had
``kBT = 0.5981`` hard-coded which silently fixed it at 300 K · 1.987e-3
kcal/(mol·K) ≈ 0.596 kcal/mol — close enough to look right, far enough to bite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from scipy.constants import N_A, k as _k_boltzmann

# k_B in kJ/(mol·K).  scipy gives J/K; multiply by Avogadro and divide by 1000.
_KB_KJ_PER_MOL_K = _k_boltzmann * N_A / 1000.0  # ≈ 8.314462618e-3


@dataclass
class RunConfig:
    """Everything needed to launch one KOMBI run.

    Notes
    -----
    Units convention:
      * length: Å (matches CHARMM topologies and MDTraj)
      * energy: kJ/mol (matches OpenMM)
      * temperature: K (always)
      * time: ps

    Choose the unit system once at the top of your script and stick with it.
    """

    # --- run identity ------------------------------------------------------
    name: str = "kombi_run"
    out_dir: Path = field(default_factory=lambda: Path("runs/kombi_run"))
    seed: int = 42

    # --- system ------------------------------------------------------------
    system: str = "model_1d"  # one of: model_1d, model_2d, openmm
    psf_file: Optional[Path] = None
    pdb_file: Optional[Path] = None
    toppar_str: Optional[Path] = None  # CHARMM toppar list

    # --- thermodynamics ----------------------------------------------------
    temperature: float = 300.0  # Kelvin

    # --- CV / discretisation ----------------------------------------------
    cv_range: tuple[float, float] = (-2.0, 12.0)
    cv_bins: int = 200
    cv_range_2d: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    start_cv: float = 0.0
    target_cv: float = 10.0

    # --- bias --------------------------------------------------------------
    n_gaussians: int = 10
    bias_amplitude_bounds: tuple[float, float] = (-5.0, 5.0)  # in kT
    bias_width_bounds: tuple[float, float] = (0.2, 1.0)  # CV-units
    lbfgs_tol: float = 1e-4
    lbfgs_restarts_per_gaussian: int = 10  # total restarts = restarts * n_gaussians

    # --- DHAM --------------------------------------------------------------
    lag_time_ps: float = 1.0
    dham_max_iter: int = 100
    dham_tol: float = 1e-6

    # --- KOMBI loop --------------------------------------------------------
    n_iterations: int = 10
    propagation_steps: int = 100_000  # MD steps per iteration
    integrator_dt_ps: float = 0.002  # 2 fs default
    convergence_threshold: float = 0.5  # CV-units to declare target reached

    # --- comparison baselines (for the paper figures only) ----------------
    run_wtmetad: bool = False
    wtmetad_bias_factor: float = 10.0
    wtmetad_hill_height_kjmol: float = 1.2
    wtmetad_hill_width_cv: float = 0.05
    wtmetad_pace_steps: int = 500

    # --- misc --------------------------------------------------------------
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------------
    # derived quantities (computed, not stored)
    # ----------------------------------------------------------------------
    @property
    def kbT(self) -> float:
        """Thermal energy in kJ/mol (matches OpenMM unit system)."""
        return _KB_KJ_PER_MOL_K * self.temperature

    @property
    def beta(self) -> float:
        """Inverse temperature in (kJ/mol)⁻¹."""
        return 1.0 / self.kbT

    @property
    def dim(self) -> int:
        return 1 if self.cv_range_2d is None else 2

    @property
    def cv_grid(self) -> np.ndarray:
        """1D CV grid centres."""
        lo, hi = self.cv_range
        return np.linspace(lo, hi, self.cv_bins)

    # ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d


def load_config(path: str | Path) -> RunConfig:
    """Load a :class:`RunConfig` from a YAML file.

    Unknown keys are stored in ``cfg.extra`` rather than raising, so that
    example configs can carry system-specific metadata (e.g. PSF filenames)
    without bloating the dataclass.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    known = {f for f in RunConfig.__dataclass_fields__}
    kwargs = {}
    extra = {}
    for k, v in raw.items():
        if k in known:
            # coerce a couple of types
            if k == "out_dir":
                v = Path(v)
            elif k in ("cv_range",):
                v = tuple(v)
            elif k == "cv_range_2d" and v is not None:
                v = tuple(tuple(p) for p in v)
            kwargs[k] = v
        else:
            extra[k] = v
    if extra:
        kwargs.setdefault("extra", {}).update(extra)
    return RunConfig(**kwargs)
