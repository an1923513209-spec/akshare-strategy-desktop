"""Immutable production feature/timing schema helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


FEATURE_SCHEMA_VERSION = "2026-07-after-close-v1"


def feature_schema_hash(
    factor_columns: Mapping[str, Sequence[str]],
    *,
    external_factor_lag: int,
    universe_definition: str,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "external_factor_lag": int(external_factor_lag),
        "universe_definition": str(universe_definition),
        "factor_columns": {
            str(group): [str(column) for column in columns]
            for group, columns in sorted(factor_columns.items())
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
