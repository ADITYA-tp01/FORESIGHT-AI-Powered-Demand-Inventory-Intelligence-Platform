"""Promotion precedence and SKU-week promo features."""

from __future__ import annotations

import pandas as pd
from tests.helpers import mini_promotions, mini_sku_master


def test_highest_discount_wins_on_overlap():
    from foresight.preprocessing.promotions import resolve_sku_day_promos

    resolved = resolve_sku_day_promos(mini_promotions(), mini_sku_master())
    day = resolved.loc[
        (resolved["sku_id"] == "SKU00001") & (resolved["date"] == pd.Timestamp("2022-01-10"))
    ].iloc[0]
    assert day["promo_id"] == "PROMO001"
    assert float(day["promo_discount_pct"]) == 20.0


def test_weekly_promo_flag_requires_three_active_days():
    from foresight.preprocessing.promotions import (
        resolve_sku_day_promos,
        sku_week_promo_features,
    )

    daily = resolve_sku_day_promos(mini_promotions(), mini_sku_master())
    weekly = sku_week_promo_features(daily, active_min_days=3)
    w02 = weekly.set_index(["year_week", "sku_id"]).loc[("2022-W02", "SKU00001")]
    assert int(w02["is_promo_active"]) == 1
    assert float(w02["promo_discount_pct"]) == 20.0
