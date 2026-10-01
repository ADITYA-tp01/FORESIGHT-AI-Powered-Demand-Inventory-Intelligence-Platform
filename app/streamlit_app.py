"""Streamlit planning application entrypoint (docs/Plan.md 7).

Multipage navigation lives in the sidebar (Streamlit auto-discovers pages/);
this landing page shows serving-layer health so an empty build is obvious.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT / "src", ROOT / "app"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def main() -> None:
    import pandas as pd
    import streamlit as st

    from components import kpi_cards, loaders

    st.set_page_config(page_title="FORESIGHT", layout="wide")
    st.title("FORESIGHT — Demand & Inventory Intelligence")
    st.markdown(
        "Planning dashboard over the pre-aggregated serving layer "
        "(never raw CSVs). Use the sidebar:\n\n"
        "1. **Executive Overview** — portfolio KPIs, trajectory, quadrant mix\n"
        "2. **Forecast Explorer** — history vs ML forecast with 80% bands & promos\n"
        "3. **Inventory Decision Grid** — quadrant scatter + reorder/markdown lists\n"
        "4. **SKU Drilldown** — telemetry & store-level allocation"
    )

    try:
        meta = loaders.manifest()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    k = loaders.kpi_values()
    row = st.columns(4)
    row[0].metric(
        "Active SKUs", kpi_cards.format_units(k["active_skus"]),
        help=f"{k['scored_skus']:.0f} scored for inventory risk",
    )
    row[1].metric(
        "8-Week Forecast Demand", kpi_cards.format_units(k["forecast_8w_units"])
    )
    row[2].metric(
        "Stockout Rupee Exposure", kpi_cards.format_inr(k["stockout_rupee_exposure"])
    )
    row[3].metric("Locked Capital", kpi_cards.format_inr(k["locked_capital"]))

    files = pd.DataFrame(
        [
            {"store": name, "rows": meta_["rows"], "kb": round(meta_["bytes"] / 1024, 1)}
            for name, meta_ in meta["files"].items()
        ]
    )
    st.subheader("Serving layer")
    st.dataframe(files, hide_index=True)
    st.caption(
        f"Generated {meta['generated_at']} · forecast {meta['forecast_weeks'][0]} .. "
        f"{meta['forecast_weeks'][1]} · {len(meta['history_partitions'])} history "
        "partitions · rebuilt by `python -m foresight.pipeline --phase 5`"
    )


if __name__ == "__main__":
    main()
