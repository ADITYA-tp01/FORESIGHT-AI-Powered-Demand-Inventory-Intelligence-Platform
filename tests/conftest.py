"""Shared fixtures for Phase 0 environment tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from foresight.config import load_settings


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings(project_root: Path):
    return load_settings(project_root)
