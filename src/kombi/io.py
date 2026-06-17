"""Output-path bookkeeping.

The single most common source of headache in the previous KOMBI prototype was
that scripts wrote into nested folders (``./figs/explore/``,
``./trajectories/explore_traj/``, ``./params/``, …) without ever creating them.
On a fresh clone this caused immediate ``FileNotFoundError`` crashes.

This module solves that once and for all: instantiate :class:`Paths` once at
the start of a run; every attribute is a :class:`pathlib.Path` whose parent
chain has already been created via ``mkdir(parents=True, exist_ok=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


@dataclass
class Paths:
    """Bundle of output directories for a single KOMBI run.

    All paths are created on construction.  Override any of them via keyword
    argument; otherwise they default to ``<root>/<subdir>/``.

    Parameters
    ----------
    root
        Top-level run directory.  Will be created if missing.
    figures
        Where every plot lives.  Default ``<root>/figures``.
    trajectories
        Raw biased / unbiased DCDs and ``.npy`` trajectory arrays.
    bias_history
        One ``bias_iter_<k>.npy`` per iteration.
    dham
        Reweighted Markov / rate matrices, ESS diagnostics.
    logs
        Stdout / per-iteration ``.json`` logs.
    """

    root: Path
    figures: Path = None  # type: ignore[assignment]
    trajectories: Path = None  # type: ignore[assignment]
    bias_history: Path = None  # type: ignore[assignment]
    dham: Path = None  # type: ignore[assignment]
    logs: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.figures is None:
            self.figures = self.root / "figures"
        if self.trajectories is None:
            self.trajectories = self.root / "trajectories"
        if self.bias_history is None:
            self.bias_history = self.root / "bias_history"
        if self.dham is None:
            self.dham = self.root / "dham"
        if self.logs is None:
            self.logs = self.root / "logs"
        for p in (
            self.root,
            self.figures,
            self.trajectories,
            self.bias_history,
            self.dham,
            self.logs,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)

    # convenience for "<run>/<sub>/whatever.ext"
    def fig(self, name: str) -> Path:
        """Return ``<root>/figures/<name>`` with parents created."""
        p = self.figures / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def traj(self, name: str) -> Path:
        p = self.trajectories / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def bias(self, name: str) -> Path:
        p = self.bias_history / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dham_out(self, name: str) -> Path:
        p = self.dham / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def log(self, name: str) -> Path:
        p = self.logs / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def ensure_parent(path: PathLike) -> Path:
    """``pathlib.Path(path)`` with its parent directory created if missing.

    Handy ad-hoc helper for scripts that don't want to set up :class:`Paths`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
