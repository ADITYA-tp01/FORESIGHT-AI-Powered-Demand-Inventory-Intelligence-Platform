"""Pydantic settings loading YAML configs. No magic numbers in application code."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_FILENAMES = ("config.yaml", "features.yaml", "forecast.yaml", "risk.yaml")


def _project_root() -> Path:
    env = os.environ.get("FORESIGHT_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # src/foresight/config.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


class SystemSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    version: str = "1.0.0"
    seed: int = 42
    environment: str = "development"
    log_level: str = "INFO"
    sales_file_bytes: int = 801086326
    sales_file_mib: float = 763.98


class PathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_root: str
    raw_sales: str
    raw_sku: str
    raw_store: str
    raw_customer: str
    raw_inventory: str
    raw_promotions: str
    raw_flags: str
    clean_sales_control: str
    clean_inventory_control: str
    processed_dir: str
    interim_dir: str
    artifacts_dir: str
    reports_dir: str
    configs_dir: str = "configs"


class IngestionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = 500_000
    parquet_compression: str = "snappy"
    max_peak_ram_mb: int = 500
    deduplication_policy: str = "retain_multiscan"


class AuditExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_skus: int
    n_stores: int
    n_customers: int
    n_promotions: int
    n_flags: int
    n_sales_contaminated: int
    n_sales_clean: int
    n_inventory_contaminated: int


class ApiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_batch_skus: int = 100


class Settings(BaseModel):
    """Runtime settings assembled from configs/*.yaml plus env overrides."""

    model_config = ConfigDict(extra="allow")

    project_root: Path
    system: SystemSettings
    paths: PathSettings
    ingestion: IngestionSettings
    audit_expectations: AuditExpectations
    features: dict[str, Any] = Field(default_factory=dict)
    forecast: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    dashboard: dict[str, Any] = Field(default_factory=dict)
    api: ApiSettings = Field(default_factory=ApiSettings)

    def resolve(self, *parts: str | Path) -> Path:
        path = Path(parts[0]) if parts else Path()
        for extra in parts[1:]:
            path = path / extra
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def dataset_root(self) -> Path:
        env = os.environ.get("FORESIGHT_DATASET_ROOT")
        root = Path(env) if env else Path(self.paths.dataset_root)
        if root.is_absolute():
            return root.resolve()
        return (self.project_root / root).resolve()

    def dataset_file(self, relative_key: str) -> Path:
        relative = getattr(self.paths, relative_key)
        return (self.dataset_root / relative).resolve()

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.paths.processed_dir)

    @property
    def interim_dir(self) -> Path:
        return self.resolve(self.paths.interim_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.paths.artifacts_dir)

    @property
    def serving_dir(self) -> Path:
        """Pre-aggregated dashboard stores (docs/Plan.md 7.1)."""
        return self.artifacts_dir / "serving"

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.paths.reports_dir)


def load_settings(project_root: Path | None = None) -> Settings:
    root = Path(project_root).resolve() if project_root else _project_root()
    config_dir = root / "configs"
    core = load_yaml(config_dir / "config.yaml")
    features = load_yaml(config_dir / "features.yaml")
    forecast = load_yaml(config_dir / "forecast.yaml")
    risk = load_yaml(config_dir / "risk.yaml")

    if os.environ.get("FORESIGHT_ENVIRONMENT"):
        core.setdefault("system", {})["environment"] = os.environ["FORESIGHT_ENVIRONMENT"]
    if os.environ.get("FORESIGHT_LOG_LEVEL"):
        core.setdefault("system", {})["log_level"] = os.environ["FORESIGHT_LOG_LEVEL"]
    if os.environ.get("FORESIGHT_SEED"):
        core.setdefault("system", {})["seed"] = int(os.environ["FORESIGHT_SEED"])
    if os.environ.get("FORESIGHT_CHUNK_SIZE"):
        core.setdefault("ingestion", {})["chunk_size"] = int(os.environ["FORESIGHT_CHUNK_SIZE"])

    return Settings(
        project_root=root,
        system=SystemSettings(**core["system"]),
        paths=PathSettings(**core["paths"]),
        ingestion=IngestionSettings(**core["ingestion"]),
        audit_expectations=AuditExpectations(**core["audit_expectations"]),
        features=features,
        forecast=forecast,
        risk=risk,
        dashboard=core.get("dashboard", {}),
        api=ApiSettings(**core.get("api", {})),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def required_config_files(project_root: Path | None = None) -> list[Path]:
    root = Path(project_root).resolve() if project_root else _project_root()
    return [root / "configs" / name for name in CONFIG_FILENAMES]
