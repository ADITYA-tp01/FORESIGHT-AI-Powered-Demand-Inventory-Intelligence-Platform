"""Atomic Parquet I/O helpers (stubs used from Phase 1)."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
