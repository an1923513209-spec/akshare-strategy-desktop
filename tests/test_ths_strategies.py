import numpy as np
import pandas as pd
import inspect
import queue
import threading
import time
from types import SimpleNamespace

import app
from desktop_strategy_app import (
    StrategyDesktopApp,
    THS_BATCH_PARALLEL_WORKERS,
    _build_ths_monitor_item,
    _ensure_ths_best_selected,
    _set_ths_record_state,
)
from ths_strategy_catalog import build_ths_strategy_card


def _sample_daily_frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = np.linspace(10, 18, rows) + np.sin(np.linspace(0, 18, rows)) * 0.8
    close = pd.Series(base, index=index)
    open_ = close.shift(1).fillna(close.iloc[0]) * 0.998
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.018
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.982
    volume = pd.Series(1_000_000 + np.arange(rows) * 1200, index=index)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def test_ths_strategy_groups_are_available() -> None:
    assert "ths_auto" in app.STRATEGY_TYPES
    assert "ths_hybrid" in app.STRATEGY_TYPES
    assert len(app.THS_ALL_STRATEGIES) >= 18
    assert app.candidate_params("short", "ths_hybrid")


def test_ths_signal_functions_return_boolean_series() -> None:
    data = _sample_daily_frame()
    for strategy_type in app.THS_ALL_STRATEGIES:
        fast, slow = 8, 20
        line_a, line_b, entries, exits = app.strategy_signals(data, fast, slow, "short", strategy_type)
        assert len(line_a) == len(data)
        assert len(line_b) == len(data)
        assert len(entries) == len(data)
        assert len(exits) == len(data)
        assert entries.dtype == bool
        assert exits.dtype == bool


def test_ths_card_exports_selection_trading_and_overlay_formulas() -> None:
    card = build_ths_strategy_card("ths_oversold_reversal", 6, 20)
    assert card.name == "超跌反转共振"
    assert card.buy_condition.startswith("BUYCOND:=")
    assert card.sell_condition.startswith("SELLCOND:=")
    assert "CROSS(RSI6,RSI12)" in card.buy_condition
    assert "WINNER(CLOSE)>0.90" in card.sell_condition
    assert "RSI6" in card.selection_formula
    assert "XG:BUYCOND" in card.selection_formula
    assert "ENTERLONG:BUYCOND" in card.trading_formula
    assert "EXITLONG:SELLCOND" in card.trading_formula
    assert "BUYSIG:=BUYCOND AND REF(BUYCOND,1)=0" in card.backtest_formula
    assert "SELLSIG:=SELLCOND AND REF(SELLCOND,1)=0" in card.backtest_formula
    assert "DRAWICON(BUYSIG" in card.backtest_formula
    assert "DRAWICON(SELLSIG" in card.backtest_formula
    assert "DRAWTEXT(BUYSIG,LOW*0.96,'买入')" in card.backtest_formula
    assert "DRAWTEXT(SELLSIG,HIGH*1.04,'卖出')" in card.backtest_formula
    assert card.overlay_formula == card.backtest_formula


def test_lhb_card_does_not_claim_portable_formula() -> None:
    card = build_ths_strategy_card("ths_lhb_institution", 8, 20)
    assert "无法获得同一数据源" in card.compatibility_note
    assert "ENTERLONG" not in card.trading_formula
    assert "DRAWICON" not in card.backtest_formula


def test_card_rule_text_uses_executable_thresholds_not_vague_prose() -> None:
    card = build_ths_strategy_card("ths_risk_off", 5, 20)
    assert card.buy_condition == (
        "BUYCOND:=CLOSE>MAS AND RSI14>=42 AND RSI14<=68 AND "
        "OBV1>OBVMA AND VR>=0.8 AND VR<=2.3;"
    )
    assert card.sell_condition == "SELLCOND:=CLOSE<MAS OR OBV1<OBVMA OR VR>4 OR RSI14>82;"
    assert "适中" not in card.buy_condition
    assert "转弱" not in card.sell_condition


def test_ths_workflow_does_not_route_results_or_forms_through_traditional_page() -> None:
    apply_source = inspect.getsource(StrategyDesktopApp._apply_ths_backtest_result)
    run_source = inspect.getsource(StrategyDesktopApp.run_ths_backtest)
    batch_source = inspect.getsource(StrategyDesktopApp.run_ths_input_stock_backtests)

    assert "_apply_backtest_result" not in apply_source
    assert "_copy_ths_form_to_backtest" not in run_source
    assert "_copy_ths_form_to_backtest" not in batch_source
    assert '_save_all_strategies"] = "1"' in run_source
    assert '_save_all_strategies"] = "1"' in batch_source
    form_source = inspect.getsource(StrategyDesktopApp._ths_backtest_form)
    assert '"_isolated_workflow": "ths"' in form_source


def test_default_saved_gate_filter_excludes_old_ths_records() -> None:
    ths_gate = {"daily_signal": {"strategy_type": "ths_risk_off"}}
    traditional_gate = {"daily_signal": {"strategy_type": "rsi"}}

    assert not app._daily_gate_matches_filter(ths_gate)
    assert app._daily_gate_matches_filter(ths_gate, strategy_type="ths_risk_off")
    assert app._daily_gate_matches_filter(traditional_gate)


def test_initial_ths_chart_reuses_worker_portfolio(monkeypatch) -> None:
    data = _sample_daily_frame()
    fast_line, slow_line, _entries, _exits = app.strategy_signals(
        data,
        8,
        20,
        "short",
        "ths_platform_breakout",
    )
    result = {
        "symbol": "000001",
        "name": "测试股票",
        "data": data,
        "cash": 100000.0,
        "fee": 0.0003,
        "horizon": "short",
        "best": pd.Series({"fast": 8, "slow": 20, "strategy_type": "ths_platform_breakout"}),
        "fast_line": fast_line,
        "slow_line": slow_line,
        "trades": pd.DataFrame(),
    }
    fake_app = SimpleNamespace(
        ths_backtest_chart_payload=None,
        ths_backtest_zoom=None,
    )
    monkeypatch.setattr(
        app,
        "strategy_portfolio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("portfolio must not rerun")),
    )

    StrategyDesktopApp._draw_backtest_chart(fake_app, result, None, target="ths")

    assert fake_app.ths_backtest_chart_payload
    assert fake_app.ths_backtest_chart_payload["points"]


def test_ths_batch_runs_concurrently_and_keeps_other_results_on_failure(monkeypatch) -> None:
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    messages: queue.Queue = queue.Queue()

    def fake_run(form):
        nonlocal active, max_active
        symbol = form["symbol"]
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.04)
            if symbol == "000002":
                raise RuntimeError("single stock failed")
            return {
                "symbol": symbol,
                "name": symbol,
                "data": _sample_daily_frame(),
                "scan": pd.DataFrame(
                    [{"strategy_type": "ths_platform_breakout", "fast": 8, "slow": 20}]
                ),
                "horizon": "short",
            }
        finally:
            with active_lock:
                active -= 1

    persisted: list[str] = []
    monkeypatch.setattr(
        "desktop_strategy_app._save_all_scan_strategies_from_payload",
        lambda _form, symbol, *_args, **_kwargs: persisted.append(symbol),
    )
    save_calls: list[bool] = []
    monkeypatch.setattr(
        "desktop_strategy_app.engine.save_persistent_strategy_cache",
        lambda: save_calls.append(True),
    )
    fake_app = SimpleNamespace(
        backtest_target="ths",
        pending_backtest_form={
            "symbol": "000001",
            "start": "20200101",
            "adjust": "qfq",
            "cash": "100000",
            "fee": "0.0003",
            "risk": "normal",
            "horizon": "short",
            "strategy_type": "ths_hybrid",
            "_isolated_workflow": "ths",
        },
        backtest_stop_event=threading.Event(),
        queue=messages,
        _run_backtest_process=fake_run,
    )

    StrategyDesktopApp._backtest_batch_worker(
        fake_app,
        ["000001", "000002", "000003"],
    )

    batch_messages = []
    while not messages.empty():
        message = messages.get_nowait()
        if message.kind == "backtest_batch":
            batch_messages.append(message)
    assert THS_BATCH_PARALLEL_WORKERS >= 2
    assert max_active >= 2
    assert len(batch_messages) == 1
    payload = batch_messages[0].payload
    assert {item["symbol"] for item in payload["results"]} == {"000001", "000003"}
    assert len(payload["errors"]) == 1
    assert "000002" in payload["errors"][0]
    assert set(persisted) == {"000001", "000003"}
    assert save_calls == [True]


def test_ths_batch_parallelism_uses_ui_form_value_with_safety_cap() -> None:
    worker_source = inspect.getsource(StrategyDesktopApp._backtest_batch_worker)
    form_source = inspect.getsource(StrategyDesktopApp._ths_backtest_form)

    assert 'base_form.get("_parallel_workers")' in worker_source
    assert "min(max(1, requested_workers), 12, len(symbols))" in worker_source
    assert '"_parallel_workers": self.ths_parallel_workers.get()' in form_source


def test_ths_history_supports_search_and_stock_sorting() -> None:
    build_source = inspect.getsource(StrategyDesktopApp._build_ths_tab)
    render_source = inspect.getsource(StrategyDesktopApp._render_ths_history)

    assert "self.ths_history_search" in build_source
    assert "self.ths_history_sort" in build_source
    assert "代码升序" in build_source
    assert "收益降序" in render_source
    assert "评分降序" in render_source
    assert "display_groups" in render_source


def test_only_starred_ths_strategy_is_available_to_intraday_monitor() -> None:
    desktop = object.__new__(StrategyDesktopApp)
    selected = {
        "ths_selected": True,
        "result": {"daily_signal": {"strategy_type": "ths_quality_pullback"}},
    }
    unselected = {
        "ths_selected": False,
        "result": {"daily_signal": {"strategy_type": "ths_quality_pullback"}},
    }

    assert not desktop._record_matches_strategy_filter(selected)
    assert desktop._record_matches_strategy_filter(selected, include_selected_ths=True)
    assert not desktop._record_matches_strategy_filter(unselected, include_selected_ths=True)


def test_selecting_ths_strategy_preserves_complete_result_and_traditional_state() -> None:
    cache = {
        "ths-old": {
            "symbol": "000001",
            "active_for_trading": True,
            "result": {
                "daily_signal": {"strategy_type": "ths_boll_rsi_break"},
                "best": {"total_return_pct": 12.3, "max_drawdown_pct": -8.2},
            },
        },
        "ths-best": {
            "symbol": "000001",
            "active_for_trading": False,
            "result": {
                "daily_signal": {"strategy_type": "ths_quality_pullback"},
                "best": {"total_return_pct": 18.6, "max_drawdown_pct": -6.1},
            },
        },
        "traditional": {
            "symbol": "000001",
            "active_for_trading": True,
            "selected_for_left": True,
            "result": {"daily_signal": {"strategy_type": "rsi"}},
        },
    }

    _set_ths_record_state(cache, "ths-best", selected=True, active=True)

    assert cache["ths-best"]["ths_selected"] is True
    assert cache["ths-best"]["selected_for_left"] is True
    assert cache["ths-best"]["active_for_trading"] is True
    assert cache["ths-best"]["result"]["best"]["total_return_pct"] == 18.6
    assert cache["ths-old"]["active_for_trading"] is False
    assert cache["traditional"]["active_for_trading"] is True
    assert cache["traditional"]["selected_for_left"] is True


def test_ths_full_backtest_auto_selects_best_strategy() -> None:
    source = inspect.getsource(__import__("desktop_strategy_app")._compute_backtest_payload)
    assert 'isolated_workflow == "ths"' in source
    assert "active_index = scan.index[0]" in source


def test_existing_ths_history_auto_selects_best_and_syncs_old_selection() -> None:
    def record(symbol: str, score: float, **state: object) -> dict[str, object]:
        return {
            "symbol": symbol,
            "result": {
                "daily_signal": {"strategy_type": "ths_hybrid"},
                "best": {"score": score},
            },
            **state,
        }

    cache = {
        "a-low": record("000001", 1.0),
        "a-best": record("000001", 9.0),
        "b-old": record("000002", 3.0, ths_selected=True),
    }

    assert _ensure_ths_best_selected(cache) is True
    assert cache["a-best"]["ths_selected"] is True
    assert cache["a-best"]["selected_for_left"] is True
    assert cache["a-best"]["active_for_trading"] is True
    assert not cache["a-low"].get("ths_selected")
    assert cache["b-old"]["selected_for_left"] is True
    assert cache["b-old"]["active_for_trading"] is True
    assert _ensure_ths_best_selected(cache) is False


def test_ths_monitor_source_contains_only_starred_left_strategies() -> None:
    selected = {
        "saved_at": "2026-07-23",
        "ths_selected": True,
        "selected_for_left": True,
    }
    not_starred = {
        "saved_at": "2026-07-22",
        "ths_selected": False,
        "selected_for_left": False,
    }
    fake_app = SimpleNamespace(
        _ths_saved_records=lambda: [("selected", selected), ("history-only", not_starred)]
    )

    rows = StrategyDesktopApp._ths_monitor_records(fake_app)

    assert rows == [("selected", selected)]


def test_ths_monitor_rejects_traditional_strategy_before_fetching_market_data(monkeypatch) -> None:
    monkeypatch.setattr(
        "desktop_strategy_app.engine.load_persistent_strategy_cache",
        lambda: {
            "traditional": {
                "symbol": "000001",
                "ths_selected": True,
                "result": {"daily_signal": {"strategy_type": "rsi"}},
            }
        },
    )

    try:
        _build_ths_monitor_item(
            {
                "symbol": "000001",
                "period": "5",
                "strategy_key": "traditional",
            }
        )
    except ValueError as exc:
        assert "同花顺监控策略不存在" in str(exc)
    else:
        raise AssertionError("traditional strategy must not enter THS monitor")


def test_monitor_engine_accepts_fresh_daily_override() -> None:
    source = inspect.getsource(app.build_monitor_item)
    assert "daily_override" in source
    assert "ensure_daily_strategy" in source
