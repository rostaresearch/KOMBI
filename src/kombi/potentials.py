"""Analytic test potentials for KOMBI benchmarking.

The 1D / 2D toy potentials from TJ's prototype are split out here so they're
easy to import in tests and examples without dragging the full MSM machinery.
"""

from __future__ import annotations

import numpy as np


def double_well_1d(x: np.ndarray, amplitude: float = 10.0, tilt: float = 0.5) -> np.ndarray:
    """Tilted double-well potential used by TJ's prototype.

    ``y(x) = amplitude · [tilt · sin(x − π) + (1 − tilt) · sin((x − π)/2)]``
    on ``x ∈ [0, 2π]``.  Matches ``MSM._init_sample_K(dim=1)`` from RL_MFPT.

    Parameters
    ----------
    x
        1D array of CV positions.
    amplitude
        Overall scale.
    tilt
        Mixing factor between the two sin components (0 to 1).

    Returns
    -------
    y
        Potential energy in the same arbitrary units as ``amplitude``.
    """
    y1 = np.sin(x - np.pi)
    y2 = np.sin((x - np.pi) / 2.0)
    return amplitude * (tilt * y1 + (1.0 - tilt) * y2)


def mueller_brown_2d(
    x: np.ndarray, y: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """Müller-Brown potential — the canonical 2D minima/saddles toy system.

    ``V(x,y) = scale · Σ_i A_i · exp(a_i(x − x_i)² + b_i(x − x_i)(y − y_i) + c_i(y − y_i)²)``

    Parameters
    ----------
    x, y
        Same-shape arrays of CV positions (broadcastable).
    scale
        Multiplier (kJ/mol if you want it in OpenMM units).

    Returns
    -------
    V
        Potential energy on the (x, y) grid.
    """
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])

    x = np.asarray(x); y = np.asarray(y)
    V = np.zeros_like(x, dtype=float)
    for k in range(4):
        V += A[k] * np.exp(
            a[k] * (x - x0[k]) ** 2 + b[k] * (x - x0[k]) * (y - y0[k]) + c[k] * (y - y0[k]) ** 2
        )
    return V * scale
