"""FastAPI endpoint integration tests (docs/Plan.md 9.1 item 6)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from tests.test_serving import (
    WEEKS_8,
    mini_cv_summary,
    mini_forecast,
    mini_master,
    mini_promo_weekly,
    mini_risk_scores,
    mini_weekly,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def serving_dir(tmp_path: Path) -> Path:
    from foresight.serving.build import build_serving_layer

    baseline = pd.DataFrame(
        [
            {"sku_id": sku, "year_week": week, "forecast": 10.0, "tier_used": 1}
            for sku in ("SKU00001", "SKU00002", "SKU00003", "SKU00004")
            for week in WEEKS_8
        ]
    )
    build_serving_layer(
        settings=SimpleNamespace(serving_dir=tmp_path),
        weekly=mini_weekly(),
        sku_master=mini_master(),
        promo_weekly=mini_promo_weekly(),
        risk_scores=mini_risk_scores(),
        forecast=mini_forecast(),
        cv_summary=mini_cv_summary(),
        baseline=baseline,
    )
    return tmp_path


@pytest.fixture()
def client(serving_dir: Path):
    from api.main import app
    from api.store import ServingStore, get_store

    store = ServingStore.load(serving_dir)
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_reports_artifact_timestamps(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["phase"] == "6"
    assert payload["version"]
    assert payload["model_artifact_timestamp"]
    assert payload["serving_generated_at"]


def test_forecast_returns_interval_weeks(client: TestClient) -> None:
    response = client.get("/forecast/SKU00001")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"sku_id", "model_version", "generated_at", "forecast"}
    assert payload["sku_id"] == "SKU00001"
    assert len(payload["forecast"]) == 8
    first = payload["forecast"][0]
    assert first["week_start"] == "2026-01-05"  # Monday of 2026-W02
    assert first["lower_bound"] <= first["forecast_units"] <= first["upper_bound"]


def test_forecast_horizon_truncates(client: TestClient) -> None:
    response = client.get("/forecast/SKU00001?horizon_weeks=3")
    assert response.status_code == 200
    assert len(response.json()["forecast"]) == 3


def test_forecast_rejects_out_of_range_horizon(client: TestClient) -> None:
    assert client.get("/forecast/SKU00001?horizon_weeks=13").status_code == 422
    assert client.get("/forecast/SKU00001?horizon_weeks=0").status_code == 422


def test_forecast_unknown_sku_404_structured(client: TestClient) -> None:
    response = client.get("/forecast/SKU99999")
    assert response.status_code == 404
    assert response.json() == {"error": "unknown_sku", "sku_id": "SKU99999"}


def test_risk_scored_sku_contract(client: TestClient) -> None:
    payload = client.get("/risk/SKU00001").json()
    assert payload["quadrant"] == "reorder_now"
    assert payload["recommended_action"]  # sourced from configs/risk.yaml
    assert payload["stockout_risk_score"] == pytest.approx(0.91)
    assert payload["weeks_of_supply"] == pytest.approx(0.42)
    assert payload["sales_at_risk_rupees"] == 60.0
    assert payload["locked_capital_rupees"] == 0.0


def test_risk_unscored_sku_null_risk_fields(client: TestClient) -> None:
    payload = client.get("/risk/SKU00003").json()
    assert payload["quadrant"] == "not_scored"
    assert payload["stockout_risk_score"] is None
    assert payload["weeks_of_supply"] is None
    assert payload["sales_at_risk_rupees"] == 0.0
    assert payload["recommended_action"]


def test_risk_unknown_sku_404_structured(client: TestClient) -> None:
    response = client.get("/risk/SKU99999")
    assert response.status_code == 404
    assert response.json() == {"error": "unknown_sku", "sku_id": "SKU99999"}


def test_batch_risk_deduplicates_and_preserves_order(client: TestClient) -> None:
    response = client.post(
        "/batch/risk", json={"sku_ids": ["SKU00002", "SKU00001", "SKU00002"]}
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["sku_id"] for item in items] == ["SKU00002", "SKU00001"]


def test_batch_risk_unknown_ids_404_lists_missing(client: TestClient) -> None:
    response = client.post(
        "/batch/risk", json={"sku_ids": ["SKU00001", "SKU99998", "SKU99999"]}
    )
    assert response.status_code == 404
    assert response.json() == {
        "error": "unknown_sku_ids",
        "missing": ["SKU99998", "SKU99999"],
    }


def test_batch_risk_empty_payload_rejected(client: TestClient) -> None:
    assert client.post("/batch/risk", json={"sku_ids": []}).status_code == 422


def test_batch_risk_too_large_rejected_with_config_cap(client: TestClient) -> None:
    from foresight.config import get_settings

    cap = get_settings().api.max_batch_skus
    response = client.post(
        "/batch/risk", json={"sku_ids": [f"SKU{i:05d}" for i in range(cap + 1)]}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "error": "batch_too_large",
        "max_batch_skus": cap,
    }


def test_missing_serving_layer_returns_503(tmp_path: Path) -> None:
    from api.main import app
    from api.store import ServingStore, get_store

    app.dependency_overrides[get_store] = lambda: ServingStore.load(tmp_path / "absent")
    try:
        response = TestClient(app).get("/health")
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"] == "serving_unavailable"
        assert "phase 5" in payload["detail"]
    finally:
        app.dependency_overrides.clear()


def test_store_singleton_uses_real_serving_dir() -> None:
    from api.store import get_store

    if not (PROJECT_ROOT / "artifacts" / "serving" / "serving_manifest.json").exists():
        pytest.skip("run `python -m foresight.pipeline --phase 5` first")
    store = get_store()
    assert store is get_store()
    assert len(store.grid) >= 100
    assert store.manifest["forecast_weeks"][0] < store.manifest["forecast_weeks"][1]
