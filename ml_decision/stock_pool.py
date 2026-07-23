"""Listed A-share universe loading and ML stock-pool merging."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd


UNIVERSE_CACHE_TTL_SECONDS = 24 * 60 * 60


def normalize_a_share_universe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize AKShare's listed A-share code/name table."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("A-share universe source returned no rows")
    code_column = next((column for column in ("code", "代码", "证券代码") if column in raw.columns), None)
    name_column = next((column for column in ("name", "名称", "证券简称") if column in raw.columns), None)
    if code_column is None or name_column is None:
        raise ValueError(f"A-share universe columns changed: {list(raw.columns)}")
    table = raw[[code_column, name_column]].rename(columns={code_column: "code", name_column: "name"}).copy()
    code_text = table["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    table["code"] = code_text.str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    table["name"] = table["name"].fillna("").astype(str).str.strip()
    table = table.dropna(subset=["code"])
    table = table[table["code"].str.fullmatch(r"\d{6}")]
    table = table.drop_duplicates("code", keep="last").sort_values("code", kind="stable").reset_index(drop=True)
    if table.empty:
        raise ValueError("A-share universe contains no valid six-digit codes")
    return table


def load_all_a_share_universe(
    cache_dir: Path,
    *,
    force_refresh: bool = False,
    ttl_seconds: int = UNIVERSE_CACHE_TTL_SECONDS,
) -> tuple[pd.DataFrame, str]:
    """Load all listed A shares, preserving a usable stale cache on source failure."""
    cache_path = Path(cache_dir) / "ml_universe" / "a_share_universe.json"

    def read_cache() -> pd.DataFrame | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            records = payload.get("items", payload) if isinstance(payload, dict) else payload
            return normalize_a_share_universe(pd.DataFrame(records))
        except Exception:
            return None

    cached = read_cache()
    cache_fresh = bool(
        cached is not None
        and cache_path.exists()
        and max(0.0, time.time() - cache_path.stat().st_mtime) <= max(int(ttl_seconds), 0)
    )
    if cache_fresh and not force_refresh:
        return cached.copy(), "cache"

    try:
        import akshare as ak

        table = normalize_a_share_universe(ak.stock_info_a_code_name())
    except Exception as exc:
        if cached is not None:
            return cached.copy(), f"stale_cache:{type(exc).__name__}"
        raise RuntimeError(f"无法取得全部 A 股代码，且没有可用缓存：{type(exc).__name__}: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "ak.stock_info_a_code_name",
        "items": table.to_dict(orient="records"),
    }
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(cache_path)
    return table, "network"


def merge_universe_into_pool(
    pool: dict[str, dict[str, Any]],
    universe: pd.DataFrame,
    *,
    timestamp: str | None = None,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Add the listed universe without overwriting saved position fields."""
    merged = {str(code): dict(item) for code, item in pool.items() if isinstance(item, dict)}
    now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0
    updated = 0
    for row in normalize_a_share_universe(universe).itertuples(index=False):
        code = str(row.code)
        name = str(row.name)
        if code not in merged:
            merged[code] = {
                "symbol": code,
                "name": name,
                "shares": "",
                "cost": "",
                "available_shares": "",
                "today_bought_shares": "",
                "buy_date": "",
                "added_at": now,
                "updated_at": "",
            }
            added += 1
            continue
        item = merged[code]
        item["symbol"] = code
        if not str(item.get("name") or "").strip() and name:
            item["name"] = name
            updated += 1
        item.setdefault("shares", "")
        item.setdefault("cost", "")
        item.setdefault("available_shares", "")
        item.setdefault("today_bought_shares", "")
        item.setdefault("buy_date", "")
        item.setdefault("added_at", now)
        item.setdefault("updated_at", "")
    return merged, added, updated
