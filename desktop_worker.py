"""Subprocess worker for desktop_strategy_app.

The desktop UI launches this file as a separate process so heavy ML/backtest
work cannot block Tk's event loop.
"""

from __future__ import annotations

import json
import pickle
import sys
import traceback
from pathlib import Path

from desktop_strategy_app import _compute_backtest_payload, _compute_ml_prediction_payload


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            form = json.load(handle)
        if form.get("_job") == "ml_predict":
            payload = _compute_ml_prediction_payload(form)
        else:
            payload = _compute_backtest_payload(form)
        result = {"kind": "ok", "payload": payload}
    except Exception:
        result = {"kind": "error", "error": traceback.format_exc()}
    with output_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
