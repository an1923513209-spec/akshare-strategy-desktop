"""Desktop intraday monitor for the local A-share strategy workbench."""

from __future__ import annotations

import queue
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from typing import Any

import tkinter as tk
from tkinter import ttk, messagebox

import app as strategy_app


@dataclass
class MonitorResult:
    item: dict[str, Any] | None = None
    error: str | None = None


class DesktopMonitor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("A股盘中监控")
        self.geometry("1260x820")
        self.minsize(980, 680)

        self.results_queue: queue.Queue[list[MonitorResult]] = queue.Queue()
        self.latest_items: dict[str, dict[str, Any]] = {}
        self.running = False
        self.worker: threading.Thread | None = None
        self.selected_symbol: str | None = None

        self._build_ui()
        self.after(300, self._poll_queue)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="股票代码或名称，逗号/换行分隔").grid(row=0, column=0, sticky="w")
        self.symbol_text = tk.Text(top, height=3, width=60)
        self.symbol_text.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.symbol_text.insert("1.0", "002472, 双环传动")

        controls = ttk.Frame(top)
        controls.grid(row=1, column=1, sticky="nsew")

        ttk.Label(controls, text="周期").grid(row=0, column=0, sticky="w")
        self.period_var = tk.StringVar(value="5")
        ttk.Combobox(controls, textvariable=self.period_var, values=("1", "5", "15"), width=8, state="readonly").grid(row=1, column=0, padx=(0, 8))

        ttk.Label(controls, text="刷新秒").grid(row=0, column=1, sticky="w")
        self.interval_var = tk.StringVar(value="30")
        ttk.Entry(controls, textvariable=self.interval_var, width=8).grid(row=1, column=1, padx=(0, 8))

        ttk.Label(controls, text="持股数").grid(row=0, column=2, sticky="w")
        self.shares_var = tk.StringVar(value="")
        ttk.Entry(controls, textvariable=self.shares_var, width=10).grid(row=1, column=2, padx=(0, 8))

        ttk.Label(controls, text="成本价").grid(row=0, column=3, sticky="w")
        self.buy_price_var = tk.StringVar(value="")
        ttk.Entry(controls, textvariable=self.buy_price_var, width=10).grid(row=1, column=3, padx=(0, 8))

        ttk.Button(controls, text="刷新一次", command=self.refresh_once).grid(row=1, column=4, padx=(0, 8))
        self.start_button = ttk.Button(controls, text="开始监控", command=self.toggle_monitor)
        self.start_button.grid(row=1, column=5)

        body = ttk.PanedWindow(self, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        table_frame = ttk.Frame(body)
        chart_frame = ttk.Frame(body)
        body.add(table_frame, weight=3)
        body.add(chart_frame, weight=2)

        columns = ("symbol", "name", "action", "price", "daily_gate", "trend", "volume", "vwap", "stop", "time")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        headings = {
            "symbol": "代码",
            "name": "名称",
            "action": "信号",
            "price": "价格",
            "daily_gate": "日线闸门",
            "trend": "分钟趋势",
            "volume": "量能比",
            "vwap": "VWAP",
            "stop": "风控线",
            "time": "行情时间",
        }
        widths = {
            "symbol": 90,
            "name": 120,
            "action": 130,
            "price": 90,
            "daily_gate": 110,
            "trend": 90,
            "volume": 90,
            "vwap": 90,
            "stop": 90,
            "time": 160,
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._open_selected_xueqiu)

        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(chart_frame, background="#fbfdff", highlightthickness=1, highlightbackground="#d8e0ea")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="打开雪球", command=self._open_selected_xueqiu).grid(row=0, column=1, sticky="e")

    def _symbols_text(self) -> str:
        return self.symbol_text.get("1.0", "end").strip()

    def _interval_seconds(self) -> int:
        try:
            return max(10, int(float(self.interval_var.get() or 30)))
        except ValueError:
            return 30

    def toggle_monitor(self) -> None:
        self.running = not self.running
        self.start_button.configure(text="停止监控" if self.running else "开始监控")
        if self.running:
            self._start_worker(loop=True)

    def refresh_once(self) -> None:
        self._start_worker(loop=False)

    def _start_worker(self, loop: bool) -> None:
        if self.worker and self.worker.is_alive():
            self.status_var.set("上一轮还在刷新中，请稍等")
            return
        self.worker = threading.Thread(target=self._worker_loop, args=(loop,), daemon=True)
        self.worker.start()

    def _worker_loop(self, loop: bool) -> None:
        while True:
            self._fetch_once()
            if not loop or not self.running:
                break
            time.sleep(self._interval_seconds())

    def _fetch_once(self) -> None:
        text = self._symbols_text()
        period = self.period_var.get()
        shares = self.shares_var.get()
        buy_price = self.buy_price_var.get()
        results: list[MonitorResult] = []
        try:
            symbols = strategy_app.parse_symbol_text(text)
        except Exception as exc:
            self.results_queue.put([MonitorResult(error=str(exc))])
            return

        for symbol in symbols:
            try:
                item = strategy_app.build_monitor_item(symbol, period, shares, buy_price)
                results.append(MonitorResult(item=item))
            except Exception as exc:
                results.append(
                    MonitorResult(
                        item={
                            "symbol": symbol,
                            "name": strategy_app.stock_display_name(symbol),
                            "xueqiu_url": strategy_app.xueqiu_url(symbol),
                        },
                        error=f"{symbol}: {exc}",
                    )
                )
        self.results_queue.put(results)

    def _poll_queue(self) -> None:
        try:
            while True:
                results = self.results_queue.get_nowait()
                self._apply_results(results)
        except queue.Empty:
            pass
        self.after(300, self._poll_queue)

    def _apply_results(self, results: list[MonitorResult]) -> None:
        if len(results) == 1 and results[0].error and not results[0].item:
            self.status_var.set(results[0].error)
            return

        for result in results:
            item = result.item or {}
            symbol = str(item.get("symbol", ""))
            if not symbol:
                continue
            if result.error:
                item["action"] = "错误"
                item["action_code"] = "watch"
                item["reasons"] = [result.error]
                item.setdefault("price", "-")
                item.setdefault("daily_gate", "-")
                item.setdefault("minute_trend", "-")
                item.setdefault("volume_ratio", "-")
                item.setdefault("vwap", "-")
                item.setdefault("stop_line", "-")
                item.setdefault("updated", "-")
                item.setdefault("chart_points", [])
            self.latest_items[symbol] = item

        self._render_table()
        if self.selected_symbol in self.latest_items:
            self._draw_chart(self.latest_items[self.selected_symbol])
        elif self.latest_items:
            first_symbol = next(iter(self.latest_items))
            self.selected_symbol = first_symbol
            self.tree.selection_set(first_symbol)
            self._draw_chart(self.latest_items[first_symbol])
        self.status_var.set(f"刷新完成：{time.strftime('%H:%M:%S')}，共 {len(results)} 只")

    def _render_table(self) -> None:
        existing = set(self.tree.get_children())
        current = set(self.latest_items)
        for iid in existing - current:
            self.tree.delete(iid)
        for symbol, item in self.latest_items.items():
            values = (
                item.get("symbol", symbol),
                item.get("name", ""),
                item.get("action", ""),
                item.get("price", ""),
                item.get("daily_gate", ""),
                item.get("minute_trend", ""),
                item.get("volume_ratio", ""),
                item.get("vwap", ""),
                item.get("stop_line", ""),
                item.get("updated", ""),
            )
            if symbol in existing:
                self.tree.item(symbol, values=values)
            else:
                self.tree.insert("", "end", iid=symbol, values=values)

    def _on_select(self, _event: object | None = None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        self.selected_symbol = selection[0]
        item = self.latest_items.get(self.selected_symbol)
        if item:
            self._draw_chart(item)

    def _open_selected_xueqiu(self, _event: object | None = None) -> None:
        selection = self.tree.selection()
        symbol = selection[0] if selection else self.selected_symbol
        if not symbol:
            return
        item = self.latest_items.get(symbol, {})
        url = item.get("xueqiu_url") or strategy_app.xueqiu_url(symbol)
        webbrowser.open(str(url))

    def _draw_chart(self, item: dict[str, Any]) -> None:
        self.canvas.delete("all")
        width = max(400, self.canvas.winfo_width())
        height = max(220, self.canvas.winfo_height())
        pad_left, pad_right, pad_top, pad_bottom = 58, 24, 28, 40
        points = item.get("chart_points") or []
        if not points:
            self.canvas.create_text(width / 2, height / 2, text="暂无分时曲线", fill="#607086", font=("Microsoft YaHei", 14))
            return

        values: list[float] = []
        for point in points:
            if isinstance(point.get("price"), (int, float)):
                values.append(float(point["price"]))
            if isinstance(point.get("vwap"), (int, float)):
                values.append(float(point["vwap"]))
        stop_value = item.get("stop_value")
        if isinstance(stop_value, (int, float)):
            values.append(float(stop_value))
        if not values:
            return

        low = min(values)
        high = max(values)
        if high == low:
            high += 1
            low -= 1
        margin = (high - low) * 0.08
        high += margin
        low -= margin

        def x_at(index: int) -> float:
            if len(points) == 1:
                return pad_left
            return pad_left + index * (width - pad_left - pad_right) / (len(points) - 1)

        def y_at(value: float) -> float:
            return pad_top + (high - value) * (height - pad_top - pad_bottom) / (high - low)

        self.canvas.create_line(pad_left, pad_top, pad_left, height - pad_bottom, fill="#d8e0ea")
        self.canvas.create_line(pad_left, height - pad_bottom, width - pad_right, height - pad_bottom, fill="#d8e0ea")
        self.canvas.create_text(pad_left, 14, text=f"{high:.2f}", fill="#607086", anchor="w")
        self.canvas.create_text(pad_left, height - 16, text=f"{low:.2f}", fill="#607086", anchor="w")

        if isinstance(stop_value, (int, float)):
            y_stop = y_at(float(stop_value))
            self._dashed_line(pad_left, y_stop, width - pad_right, y_stop, fill="#bf2f2f")
            self.canvas.create_text(width - 86, y_stop - 10, text=f"风控 {item.get('stop_line', '')}", fill="#bf2f2f")

        price_coords: list[float] = []
        vwap_coords: list[float] = []
        for idx, point in enumerate(points):
            x = x_at(idx)
            price = point.get("price")
            vwap = point.get("vwap")
            if isinstance(price, (int, float)):
                price_coords.extend([x, y_at(float(price))])
            if isinstance(vwap, (int, float)):
                vwap_coords.extend([x, y_at(float(vwap))])
        if len(vwap_coords) >= 4:
            self.canvas.create_line(*vwap_coords, fill="#f97316", width=2)
        if len(price_coords) >= 4:
            self.canvas.create_line(*price_coords, fill="#1464f4", width=2)

        last = points[-1]
        last_price = float(last.get("price", values[-1]))
        last_x, last_y = x_at(len(points) - 1), y_at(last_price)
        action_code = item.get("action_code")
        color = "#0f8f61" if action_code == "buy" else "#bf2f2f" if action_code == "sell" else "#1464f4"
        self.canvas.create_oval(last_x - 5, last_y - 5, last_x + 5, last_y + 5, fill=color, outline="white", width=2)
        marker_text = "买" if action_code == "buy" else "卖" if action_code == "sell" else ""
        if marker_text:
            self.canvas.create_rectangle(last_x - 18, last_y - 36, last_x + 18, last_y - 12, fill=color, outline=color)
            self.canvas.create_text(last_x, last_y - 24, text=marker_text, fill="white", font=("Microsoft YaHei", 12, "bold"))

        title = f"{item.get('symbol', '')} {item.get('name', '')}  {item.get('action', '')}"
        self.canvas.create_text(pad_left, pad_top - 14, text=title, fill="#14213d", anchor="w", font=("Microsoft YaHei", 12, "bold"))
        self.canvas.create_text(width - pad_right, pad_top - 14, text="蓝=价格  橙=VWAP", fill="#607086", anchor="e")
        self.canvas.create_text(pad_left, height - 18, text=str(points[0].get("time", "")), fill="#607086", anchor="w")
        self.canvas.create_text(width - pad_right, height - 18, text=str(points[-1].get("time", "")), fill="#607086", anchor="e")

    def _dashed_line(self, x1: float, y1: float, x2: float, y2: float, fill: str) -> None:
        dash = 8
        gap = 5
        x = x1
        while x < x2:
            self.canvas.create_line(x, y1, min(x + dash, x2), y2, fill=fill)
            x += dash + gap


def main() -> None:
    try:
        app = DesktopMonitor()
        app.mainloop()
    except Exception:
        messagebox.showerror("程序错误", traceback.format_exc())


if __name__ == "__main__":
    main()
