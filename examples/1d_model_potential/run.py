"""Run KOMBI on the 1D analytic double-well potential.

Equivalent to the manuscript Fig. 1 system.  Identical to:

    python -m kombi run --config examples/1d_model_potential/config.yaml

but kept here as an explicit script so you can step through it in a debugger.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kombi import RunConfig, load_config
from kombi.orchestrator import run_kombi
from kombi.potentials import double_well_1d


def main() -> None:
    cfg: RunConfig = load_config(Path(__file__).parent / "config.yaml")

    # Build the reference unbiased rate matrix from the analytic FES
    x = cfg.cv_grid
    fes = double_well_1d(x)
    n = x.size
    K0 = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K0[j, i] = np.exp((fes[j] - fes[i]) / (2.0 * cfg.kbT))
        K0[i, i] = -np.sum(K0[:, i])

    result = run_kombi(cfg, K0=K0, cv_grid=x)
    print(f"\nFinal biased MFPT(start → target): {result.final_mfpt:.3f} ps")
    print(f"Outputs: {result.paths.root.resolve()}")


if __name__ == "__main__":
    main()
