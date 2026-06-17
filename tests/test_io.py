"""Folders should auto-create — directly answers the user's main code complaint."""

from __future__ import annotations

from pathlib import Path

import pytest

from kombi.io import Paths, ensure_parent


def test_paths_creates_all_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "deeply" / "nested" / "run_dir"
    p = Paths(root=root)
    for d in (p.root, p.figures, p.trajectories, p.bias_history, p.dham, p.logs):
        assert d.exists() and d.is_dir(), f"{d} should exist after Paths()"


def test_paths_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "run"
    Paths(root=root)
    # second construction must not error
    Paths(root=root)


def test_fig_method_creates_parent_for_nested_name(tmp_path: Path) -> None:
    p = Paths(root=tmp_path / "run")
    nested = p.fig("subfolder/another/figure.png")
    assert nested.parent.exists()


def test_ensure_parent_does_not_create_file(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    result = ensure_parent(target)
    assert result == target
    assert result.parent.exists()
    assert not result.exists()


def test_ensure_parent_existing_parent_ok(tmp_path: Path) -> None:
    target = tmp_path / "c.txt"
    ensure_parent(target)
    # second call must not raise
    ensure_parent(target)
