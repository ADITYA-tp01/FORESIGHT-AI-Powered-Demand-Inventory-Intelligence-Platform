"""Streamlit page smoke tests (AppTest) and chart unit tests — Plan 7 acceptance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tests.test_serving import (  # noqa: E402
    mini_forecast,
    mini_master,
    mini_promo_weekly,
    mini_risk_scores,
)

from components import charts  # noqa: E402

SERVING_MANIFEST = ROOT / "artifacts" / "serving" / "serving_manifest.json"
requires_serving = pytest.mark.skipif(
    not SERVING_MANIFEST.exists(), reason="run `python -m foresight.pipeline --phase 5` first"
)

PAGES = [
    "streamlit_app.py",
    "pages/1_Executive_Overview.py",
    "pages/2_Forecast_Explorer.py",
    "pages/3_Inventory_Decision_Grid.py",
    "pages/4_SKU_Drilldown.py",
]


@requires_serving
@pytest.mark.parametrize("script", PAGES)
def test_page_renders_without_exception(script: str) -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(APP_DIR / script), default_timeout=60)
    app_test.run()
    assert not app_test.exception, app_test.exception


@requires_serving
def test_landing_page_shows_title_and_metrics() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=60)
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert "FORESIGHT" in app_test.title[0].proto.body
    assert len(app_test.metric) == 4


@requires_serving
def test_executive_overview_kpi_row() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(APP_DIR / "pages" / "1_Executive_Overview.py"), default_timeout=60
    )
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert len(app_test.metric) >= 5


@requires_serving
def test_explorer_filters_render() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(APP_DIR / "pages" / "2_Forecast_Explorer.py"), default_timeout=60
    )
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert len(app_test.selectbox) >= 1
    assert len(app_test.multiselect) >= 3


@requires_serving
def test_decision_grid_has_triage_tabs() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(APP_DIR / "pages" / "3_Inventory_Decision_Grid.py"), default_timeout=60
    )
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert len(app_test.tabs) == 2


@requires_serving
def test_drilldown_first_sku_metrics() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(APP_DIR / "pages" / "4_SKU_Drilldown.py"), default_timeout=60
    )
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert len(app_test.metric) >= 8


def test_quadrant_pie_builds_single_trace() -> None:
    figure = charts.quadrant_pie({"healthy": 2, "reorder_now": 1, "not_scored": 0})
    assert len(figure.data) == 1
    assert list(figure.data[0].labels) == ["reorder_now", "healthy"]


def test_decision_scatter_uses_log_x_and_risk_range() -> None:
    from foresight.serving.build import build_decision_grid

    grid = build_decision_grid(mini_risk_scores(), mini_master(), mini_forecast())
    figure = charts.decision_scatter(grid)
    assert figure.layout.xaxis.type == "log"
    assert list(figure.layout.yaxis.range) == [0, 1.05]
    plotted_colors = {trace.marker.color for trace in figure.data}
    assert plotted_colors <= set(charts.QUADRANT_COLORS.values())


def test_promo_spans_merge_consecutive_weeks() -> None:
    from foresight.serving.build import build_promo_windows

    spans = charts._promo_spans(build_promo_windows(mini_promo_weekly()))
    assert len(spans) == 2  # active W02 and W04 are 14 days apart -> two spans
    assert (spans[1][0] - spans[0][1]).days == 14


def test_week_monday_roundtrip() -> None:
    assert charts.week_monday("2026-W02").isocalendar()[:2] == (2026, 2)
    assert charts.week_monday("2026-W02").day == 5
