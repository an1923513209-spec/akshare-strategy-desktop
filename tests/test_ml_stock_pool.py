from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml_decision.stock_pool import load_all_a_share_universe, merge_universe_into_pool, normalize_a_share_universe


def test_universe_normalization_maps_columns_and_deduplicates() -> None:
    raw = pd.DataFrame({"代码": [1, "000001", "600519"], "名称": ["平安银行旧", "平安银行", "贵州茅台"]})
    table = normalize_a_share_universe(raw)
    assert table.to_dict(orient="records") == [
        {"code": "000001", "name": "平安银行"},
        {"code": "600519", "name": "贵州茅台"},
    ]


def test_merging_all_stocks_preserves_existing_position_fields() -> None:
    existing = {
        "000001": {
            "symbol": "000001",
            "name": "平安银行",
            "shares": "150",
            "cost": "10.25",
            "available_shares": "50",
            "today_bought_shares": "100",
            "buy_date": "2026-07-18",
            "added_at": "old",
        }
    }
    universe = pd.DataFrame({"code": ["000001", "600519"], "name": ["平安银行", "贵州茅台"]})
    merged, added, updated = merge_universe_into_pool(existing, universe, timestamp="now")
    assert (added, updated) == (1, 0)
    assert merged["000001"]["shares"] == "150"
    assert merged["000001"]["available_shares"] == "50"
    assert merged["000001"]["cost"] == "10.25"
    assert merged["600519"]["name"] == "贵州茅台"


def test_universe_loader_uses_fresh_cache_without_network(tmp_path: Path) -> None:
    path = tmp_path / "ml_universe" / "a_share_universe.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"items": [{"code": "000001", "name": "平安银行"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    table, source = load_all_a_share_universe(tmp_path)
    assert source == "cache"
    assert table.iloc[0].to_dict() == {"code": "000001", "name": "平安银行"}
