"""Local A-share strategy cockpit.

Run with:
    .\.venv\Scripts\python.exe app.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt
from flask import Flask, jsonify, request, render_template_string
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
CACHE_DIR = ROOT / "cache"
STRATEGY_CACHE_PATH = CACHE_DIR / "strategy_cache.json"
sys.path.insert(0, str(SCRIPTS))

from data_utils import load_a_share_daily, normalize_symbol  # noqa: E402


app = Flask(__name__)


RESULT_CACHE: dict[str, dict[str, object]] = {}
FORM_CACHE: dict[str, dict[str, str]] = {}
INTRADAY_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
DAILY_GATE_CACHE: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
TEXT_RISK_CACHE: dict[tuple[str, str, int], dict[str, object]] = {}
PERSISTENT_STRATEGY_CACHE: dict[str, dict[str, object]] | None = None
SPOT_CACHE: tuple[float, pd.DataFrame] | None = None
CODE_NAME_CACHE: pd.DataFrame | None = None
CODE_TO_NAME: dict[str, str] = {}
NAME_TO_CODE: dict[str, str] = {}


STRATEGY_GRIDS = {
    "short": {
        "name": "短线",
        "fast": [3, 5, 8, 10, 13],
        "slow": [10, 15, 20, 30, 40],
        "lookback": 10,
        "min_trades": 8,
    },
    "swing": {
        "name": "波段",
        "fast": [5, 10, 15, 20, 25],
        "slow": [30, 40, 50, 60, 90],
        "lookback": 20,
        "min_trades": 5,
    },
    "trend": {
        "name": "趋势",
        "fast": [10, 15, 20, 25, 30, 35],
        "slow": [60, 90, 120, 160],
        "lookback": 30,
        "min_trades": 3,
    },
}

STRATEGY_TYPES = {
    "auto": "Auto",
    "auto_fast": "Auto Fast",
    "sma": "SMA Trend",
    "breakout": "Breakout",
    "rsi": "RSI Pullback",
    "rsi_capital": "RSI Capital",
    "macd": "MACD Momentum",
    "macd_kdj": "MACD KDJ",
    "boll_wr": "BOLL WR",
    "breakout_capital": "Breakout Capital",
    "ml": "ML Stacking",
    "hybrid": "Hybrid Vote",
}


PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A股策略操作台</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --ink: #14213d;
      --muted: #607086;
      --line: #d8e0ea;
      --panel: #ffffff;
      --accent: #1464f4;
      --good: #0f8f61;
      --bad: #bf2f2f;
      --warn: #a46200;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      font-size: 14px;
    }
    header {
      background: #111827;
      color: white;
      padding: 14px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    header h1 { font-size: 18px; margin: 0; font-weight: 650; }
    header span { color: #cbd5e1; font-size: 13px; }
    main { max-width: 1700px; margin: 0 auto; padding: 18px; }
    .layout {
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .sidebar {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      position: sticky;
      top: 12px;
      max-height: calc(100vh - 24px);
      overflow: auto;
    }
    .sidebar h2 { margin: 0 0 10px 0; font-size: 15px; }
    .stock-list { display: grid; gap: 8px; }
    .stock-link {
      display: grid;
      gap: 3px;
      padding: 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      text-decoration: none;
      background: #fbfdff;
    }
    .stock-link.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(20, 100, 244, 0.12); }
    .stock-link span { color: var(--muted); font-size: 12px; }
    .stock-link small { color: var(--muted); line-height: 1.35; }
    .empty-cache { color: var(--muted); line-height: 1.6; }
    .content { min-width: 0; }
    form {
      background: var(--panel);
      border: 1px solid var(--line);
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 12px;
      padding: 14px;
      border-radius: 6px;
      align-items: end;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    input, select, textarea {
      width: 100%;
      border: 1px solid #cbd5e1;
      border-radius: 5px;
      padding: 8px 9px;
      min-height: 36px;
      font-size: 14px;
      background: white;
      color: var(--ink);
    }
    textarea { min-height: 78px; resize: vertical; line-height: 1.45; }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      border-radius: 5px;
      min-height: 36px;
      padding: 0 14px;
      cursor: pointer;
      font-weight: 650;
    }
    .span-2 { grid-column: span 2; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }
    .metric, .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
    }
    .metric .k { color: var(--muted); font-size: 12px; }
    .metric .v { font-size: 22px; font-weight: 700; margin-top: 5px; }
    .section { margin-top: 14px; }
    .section h2 { margin: 0 0 10px 0; font-size: 16px; }
    .advice {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    .advice ul { margin: 8px 0 0 18px; padding: 0; line-height: 1.75; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 650; }
    .tag { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; background: #eef2ff; color: #3049a5; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .plot { overflow: hidden; border-radius: 6px; border: 1px solid var(--line); background: white; }
    .error { margin-top: 14px; color: #9f1239; background: #fff1f2; border: 1px solid #fecdd3; padding: 12px; border-radius: 6px; }
    .batch-status { margin-top: 14px; color: #27548a; background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 6px; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
      form { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .span-2 { grid-column: span 2; }
      .row { grid-template-columns: 1fr 1fr; }
      .advice { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>A股策略操作台</h1>
    <span>数据、回测与信号仅用于研究；实盘下单前请人工复核</span>
    <a href="/monitor" style="color:#bfdbfe;text-decoration:none;">盘中监控</a>
  </header>
  <main>
    <div class="layout">
      <aside class="sidebar">
        <h2>已跑股票</h2>
        {% if cached_symbols %}
          <div class="stock-list">
            {% for item in cached_symbols %}
              <a class="stock-link {% if selected_symbol == item.symbol %}active{% endif %}" href="/view/{{ item.symbol }}">
                <strong>{{ item.symbol }}</strong>
                <span>{{ item.strategy_label }}</span>
                <small>{{ item.signal }}</small>
              </a>
            {% endfor %}
          </div>
        {% else %}
          <div class="empty-cache">还没有缓存结果。先分析一只，或在批量框里输入多个代码。</div>
        {% endif %}
      </aside>
      <div class="content">
    <form method="post" action="/analyze">
      <label>股票代码
        <input name="symbol" value="{{ form.symbol }}" placeholder="002472" required>
      </label>
      <label>回测开始
        <input name="start" value="{{ form.start }}" placeholder="20200101" required>
      </label>
      <label>复权
        <select name="adjust">
          <option value="qfq" {% if form.adjust == "qfq" %}selected{% endif %}>前复权</option>
          <option value="" {% if form.adjust == "" %}selected{% endif %}>不复权</option>
          <option value="hfq" {% if form.adjust == "hfq" %}selected{% endif %}>后复权</option>
        </select>
      </label>
      <label>回测资金
        <input name="cash" type="number" value="{{ form.cash }}" min="1000" step="1000">
      </label>
      <label>手续费率
        <input name="fee" type="number" value="{{ form.fee }}" min="0" step="0.0001">
      </label>
      <label>风险档位
        <select name="risk">
          <option value="normal" {% if form.risk == "normal" %}selected{% endif %}>普通</option>
          <option value="tight" {% if form.risk == "tight" %}selected{% endif %}>保守</option>
          <option value="loose" {% if form.risk == "loose" %}selected{% endif %}>宽松</option>
        </select>
      </label>
      <label>策略周期
        <select name="horizon">
          <option value="short" {% if form.horizon == "short" %}selected{% endif %}>短线</option>
          <option value="swing" {% if form.horizon == "swing" %}selected{% endif %}>波段</option>
          <option value="trend" {% if form.horizon == "trend" %}selected{% endif %}>趋势</option>
        </select>
      </label>
      <label>策略类型
        <select name="strategy_type">
          <option value="auto" {% if form.strategy_type == "auto" %}selected{% endif %}>自动</option>
          <option value="auto_fast" {% if form.strategy_type == "auto_fast" %}selected{% endif %}>自动-传统</option>
          <option value="hybrid" {% if form.strategy_type == "hybrid" %}selected{% endif %}>混合投票</option>
          <option value="rsi_capital" {% if form.strategy_type == "rsi_capital" %}selected{% endif %}>RSI+资金</option>
          <option value="macd_kdj" {% if form.strategy_type == "macd_kdj" %}selected{% endif %}>MACD+KDJ</option>
          <option value="boll_wr" {% if form.strategy_type == "boll_wr" %}selected{% endif %}>BOLL+WR</option>
          <option value="breakout_capital" {% if form.strategy_type == "breakout_capital" %}selected{% endif %}>突破+资金</option>
          <option value="breakout" {% if form.strategy_type == "breakout" %}selected{% endif %}>突破</option>
          <option value="rsi" {% if form.strategy_type == "rsi" %}selected{% endif %}>RSI回踩</option>
          <option value="macd" {% if form.strategy_type == "macd" %}selected{% endif %}>MACD</option>
          <option value="ml" {% if form.strategy_type == "ml" %}selected{% endif %}>ML</option>
          <option value="sma" {% if form.strategy_type == "sma" %}selected{% endif %}>SMA</option>
        </select>
      </label>
      <label class="span-2">批量股票代码
        <textarea name="batch_symbols" placeholder="一行一个或用逗号隔开，例如：002472, 000001, 600519">{{ form.batch_symbols }}</textarea>
      </label>
      <label>当前持股数
        <input name="shares" type="number" value="{{ form.shares }}" min="0" step="100">
      </label>
      <label>买入价格
        <input name="buy_price" type="number" value="{{ form.buy_price }}" min="0" step="0.01">
      </label>
      <label>买入日期
        <input name="buy_date" value="{{ form.buy_date }}" placeholder="20260710">
      </label>
      <div class="span-2">
        <button type="submit">分析</button>
      </div>
    </form>

    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}

    {% if batch_status %}
      <div class="batch-status">{{ batch_status }}</div>
    {% endif %}

    {% if result %}
      <div class="row">
        {% for m in result.metrics %}
          <div class="metric">
            <div class="k">{{ m.k }}</div>
            <div class="v {{ m.cls }}">{{ m.v }}</div>
          </div>
        {% endfor %}
      </div>

      <div class="advice">
        <section class="section">
          <h2>接下来怎么做 <span class="tag">{{ result.position_mode }}</span></h2>
          <ul>
            {% for line in result.action_lines %}
              <li>{{ line }}</li>
            {% endfor %}
          </ul>
        </section>
        <section class="section">
          <h2>当前信号</h2>
          <ul>
            {% for line in result.signal_lines %}
              <li>{{ line }}</li>
            {% endfor %}
          </ul>
        </section>
      </div>

      <section class="section">
        <h2>盘中提醒规则</h2>
        <ul>
          {% for line in result.reminder_lines %}
            <li>{{ line }}</li>
          {% endfor %}
        </ul>
      </section>

      <section class="section">
        <h2>图表</h2>
        <div class="plot">{{ result.chart | safe }}</div>
      </section>

      <section class="section">
        <h2>参数排名</h2>
        <table>
          <thead>
            <tr>
              <th>参数</th><th>收益%</th><th>最大回撤%</th><th>夏普</th><th>交易次数</th><th>最终权益</th><th>评分</th>
            </tr>
          </thead>
          <tbody>
            {% for row in result.table %}
              <tr>
                <td>{{ row.strategy_label }} {{ row.fast }}/{{ row.slow }}</td>
                <td>{{ row.total_return_pct }}</td>
                <td>{{ row.max_drawdown_pct }}</td>
                <td>{{ row.sharpe }}</td>
                <td>{{ row.trades }}</td>
                <td>{{ row.final_value }}</td>
                <td>{{ row.score }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </section>
    {% endif %}
      </div>
    </div>
  </main>
</body>
</html>
"""


MONITOR_PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>盘中监控</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --ink: #14213d;
      --muted: #607086;
      --line: #d8e0ea;
      --panel: #ffffff;
      --accent: #1464f4;
      --good: #0f8f61;
      --bad: #bf2f2f;
      --warn: #a46200;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      font-size: 14px;
    }
    header {
      background: #111827;
      color: white;
      padding: 14px 22px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    header h1 { margin: 0; font-size: 18px; }
    header a { color: #bfdbfe; text-decoration: none; }
    main { max-width: 1280px; margin: 0 auto; padding: 18px; }
    form {
      display: grid;
      grid-template-columns: 2fr repeat(4, minmax(120px, 1fr));
      gap: 12px;
      align-items: end;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    input, select, textarea {
      width: 100%;
      border: 1px solid #cbd5e1;
      border-radius: 5px;
      padding: 8px 9px;
      min-height: 36px;
      font-size: 14px;
      background: white;
      color: var(--ink);
    }
    textarea { min-height: 72px; resize: vertical; line-height: 1.45; }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      border-radius: 5px;
      min-height: 36px;
      padding: 0 14px;
      cursor: pointer;
      font-weight: 650;
    }
    .status { margin: 14px 0; color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .topline { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .symbol { font-size: 18px; font-weight: 750; color: var(--ink); text-decoration: none; }
    .symbol:hover { color: var(--accent); }
    .name { color: var(--muted); margin-left: 6px; font-size: 13px; }
    .price { font-size: 22px; font-weight: 750; text-align: right; }
    .pill { display: inline-block; padding: 5px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .pill.good { background: #dcfce7; }
    .pill.bad { background: #fee2e2; }
    .pill.warn { background: #fef3c7; }
    .facts { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
    .fact { border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfdff; }
    .fact span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .spark {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
      padding: 8px;
      overflow: hidden;
    }
    .spark svg { width: 100%; height: 190px; display: block; }
    .spark text { font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; }
    .reason { line-height: 1.7; margin: 0; padding-left: 18px; }
    .error { color: #9f1239; background: #fff1f2; border: 1px solid #fecdd3; padding: 12px; border-radius: 6px; }
    @media (max-width: 900px) {
      form { grid-template-columns: 1fr 1fr; }
      form label:first-child { grid-column: span 2; }
      .grid { grid-template-columns: 1fr; }
      .facts { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>盘中监控</h1>
    <a href="/">返回策略回测</a>
  </header>
  <main>
    <form id="monitor-form">
      <label>股票代码
        <textarea name="symbols" placeholder="一行一个或逗号隔开，例如：002472, 双环传动, 平安银行">{{ symbols }}</textarea>
      </label>
      <label>分钟周期
        <select name="period">
          <option value="1" {% if period == "1" %}selected{% endif %}>1分钟</option>
          <option value="5" {% if period == "5" %}selected{% endif %}>5分钟</option>
          <option value="15" {% if period == "15" %}selected{% endif %}>15分钟</option>
        </select>
      </label>
      <label>刷新秒数
        <input name="interval" type="number" min="15" step="5" value="{{ interval }}">
      </label>
      <label>持股数
        <input name="shares" type="number" min="0" step="100" value="{{ shares }}">
      </label>
      <label>成本价
        <input name="buy_price" type="number" min="0" step="0.01" value="{{ buy_price }}">
      </label>
      <button type="submit">开始监控</button>
    </form>
    <div class="status" id="status">等待刷新...</div>
    <div class="grid" id="cards"></div>
  </main>
  <script>
    const form = document.getElementById("monitor-form");
    const cards = document.getElementById("cards");
    const statusBox = document.getElementById("status");
    let timer = null;

    function cls(value) {
      if (value === "buy" || value === "hold") return "good";
      if (value === "sell") return "bad";
      return "warn";
    }

    function renderSparkline(item) {
      const points = item.chart_points || [];
      if (!points.length) return `<div class="status">暂无分时曲线</div>`;
      const width = 720;
      const height = 190;
      const pad = { left: 46, right: 18, top: 16, bottom: 30 };
      const values = [];
      points.forEach(p => {
        if (Number.isFinite(p.price)) values.push(p.price);
        if (Number.isFinite(p.vwap)) values.push(p.vwap);
      });
      if (Number.isFinite(item.stop_value)) values.push(item.stop_value);
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (max === min) { max += 1; min -= 1; }
      const extra = (max - min) * 0.08;
      max += extra;
      min -= extra;
      const x = i => pad.left + (points.length === 1 ? 0 : i * (width - pad.left - pad.right) / (points.length - 1));
      const y = v => pad.top + (max - v) * (height - pad.top - pad.bottom) / (max - min);
      const pricePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.price).toFixed(1)}`).join(" ");
      const vwapPath = points.filter(p => Number.isFinite(p.vwap)).map((p, i, arr) => {
        const originalIndex = points.indexOf(p);
        return `${i === 0 ? "M" : "L"}${x(originalIndex).toFixed(1)},${y(p.vwap).toFixed(1)}`;
      }).join(" ");
      const last = points[points.length - 1];
      const lastX = x(points.length - 1);
      const lastY = y(last.price);
      const markerColor = item.action_code === "buy" ? "#0f8f61" : (item.action_code === "sell" ? "#bf2f2f" : "#1464f4");
      const markerText = item.action_code === "buy" ? "买" : (item.action_code === "sell" ? "卖" : "");
      const stopY = Number.isFinite(item.stop_value) ? y(item.stop_value) : null;
      return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="intraday chart">
        <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#d8e0ea"/>
        <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#d8e0ea"/>
        <text x="${pad.left}" y="12" font-size="11" fill="#607086">${max.toFixed(2)}</text>
        <text x="${pad.left}" y="${height - 8}" font-size="11" fill="#607086">${min.toFixed(2)}</text>
        ${stopY === null ? "" : `<line x1="${pad.left}" y1="${stopY.toFixed(1)}" x2="${width - pad.right}" y2="${stopY.toFixed(1)}" stroke="#bf2f2f" stroke-dasharray="5 5" opacity="0.65"/>
        <text x="${width - 86}" y="${(stopY - 5).toFixed(1)}" font-size="11" fill="#bf2f2f">风控 ${item.stop_line}</text>`}
        ${vwapPath ? `<path d="${vwapPath}" fill="none" stroke="#f97316" stroke-width="1.7" opacity="0.9"/>` : ""}
        <path d="${pricePath}" fill="none" stroke="#1464f4" stroke-width="2.2"/>
        <circle cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="5" fill="${markerColor}" stroke="white" stroke-width="2"/>
        ${markerText ? `<g transform="translate(${Math.max(pad.left, lastX - 16).toFixed(1)},${Math.max(18, lastY - 34).toFixed(1)})">
          <rect width="32" height="22" rx="4" fill="${markerColor}"/>
          <text x="16" y="15" text-anchor="middle" font-size="13" font-weight="700" fill="white">${markerText}</text>
        </g>` : ""}
        <text x="${pad.left}" y="${height - 10}" font-size="11" fill="#607086">${points[0].time}</text>
        <text x="${width - pad.right}" y="${height - 10}" text-anchor="end" font-size="11" fill="#607086">${last.time}</text>
        <text x="${width - 150}" y="18" font-size="11" fill="#1464f4">价格</text>
        <text x="${width - 100}" y="18" font-size="11" fill="#f97316">VWAP</text>
      </svg>`;
    }

    function render(items) {
      cards.innerHTML = items.map(item => {
        if (item.error) {
          return `<section class="card"><div class="topline"><a class="symbol" href="${item.xueqiu_url || '#'}" target="_blank" rel="noopener">${item.symbol}</a></div><div class="error">${item.error}</div></section>`;
        }
        const klass = cls(item.action_code);
        return `<section class="card">
          <div class="topline">
            <div>
              <a class="symbol" href="${item.xueqiu_url}" target="_blank" rel="noopener">${item.symbol}<span class="name">${item.name || ""}</span></a>
              <div class="status">${item.updated} · ${item.market_note}</div>
            </div>
            <div>
              <div class="price">${item.price}</div>
              <div class="pill ${klass}">${item.action}</div>
            </div>
          </div>
          <div class="facts">
            <div class="fact"><span>日线闸门</span>${item.daily_gate}</div>
            <div class="fact"><span>K线时间</span>${item.bar_time}</div>
            <div class="fact"><span>VWAP</span>${item.vwap}</div>
            <div class="fact"><span>分钟趋势</span>${item.minute_trend}</div>
            <div class="fact"><span>量能比</span>${item.volume_ratio}</div>
            <div class="fact"><span>风控线</span>${item.stop_line}</div>
          </div>
          <div class="spark">${renderSparkline(item)}</div>
          <ul class="reason">
            ${item.reasons.map(line => `<li>${line}</li>`).join("")}
          </ul>
        </section>`;
      }).join("");
    }

    async function refresh() {
      const params = new URLSearchParams(new FormData(form));
      statusBox.textContent = "刷新中...";
      try {
        const resp = await fetch(`/api/monitor?${params.toString()}`);
        const data = await resp.json();
        render(data.items || []);
        statusBox.textContent = `上次刷新：${data.updated}，共 ${data.items.length} 只。`;
      } catch (err) {
        statusBox.textContent = `刷新失败：${err}`;
      }
    }

    function start() {
      if (timer) clearInterval(timer);
      const interval = Math.max(15, Number(new FormData(form).get("interval") || 30));
      refresh();
      timer = setInterval(refresh, interval * 1000);
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      start();
    });
    start();
  </script>
</body>
</html>
"""


def default_form() -> dict[str, str]:
    return {
        "symbol": "002472",
        "start": "20200101",
        "adjust": "qfq",
        "cash": "100000",
        "fee": "0.0003",
        "risk": "normal",
        "horizon": "short",
        "strategy_type": "auto",
        "batch_symbols": "",
        "shares": "0",
        "buy_price": "",
        "buy_date": "",
    }


@lru_cache(maxsize=64)
def cached_data(symbol: str, start: str, adjust: str) -> pd.DataFrame:
    return load_a_share_daily(symbol, start=start, adjust=adjust, refresh_stale_today=True).copy()


def make_signals(close: pd.Series, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return fast_ma, slow_ma, entries.fillna(False), exits.fillna(False)


def make_short_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    breakout_window = max(5, fast)
    stop_window = max(5, fast)

    prior_high = high.shift(1).rolling(breakout_window).max()
    prior_low = low.shift(1).rolling(stop_window).min()
    volume_ok = volume > volume.rolling(5).mean() * 0.8
    trend_ok = (close > slow_ma) & (fast_ma > slow_ma) & (slow_ma >= slow_ma.shift(3))

    entries = (close > prior_high) & trend_ok & volume_ok
    exits = (close < fast_ma) | (close < prior_low) | ((fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)))
    return fast_ma, slow_ma, entries.fillna(False), exits.fillna(False)


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def money_flow_index(data: pd.DataFrame, window: int = 14) -> pd.Series:
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    raw_flow = typical * data["Volume"]
    positive = raw_flow.where(typical > typical.shift(), 0.0).rolling(window).sum()
    negative = raw_flow.where(typical < typical.shift(), 0.0).rolling(window).sum()
    ratio = positive / negative.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def obv(data: pd.DataFrame) -> pd.Series:
    direction = np.sign(data["Close"].diff()).fillna(0)
    return (direction * data["Volume"]).cumsum()


def kdj(data: pd.DataFrame, window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_min = data["Low"].rolling(window).min()
    high_max = data["High"].rolling(window).max()
    rsv = (data["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def williams_r(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high_max = data["High"].rolling(window).max()
    low_min = data["Low"].rolling(window).min()
    return -100 * (high_max - data["Close"]) / (high_max - low_min).replace(0, np.nan)


def make_rsi_pullback_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    low = data["Low"]
    volume = data["Volume"]
    rsi_line = rsi(close, fast)
    trend_ma = close.rolling(slow).mean()
    recent_low = low.shift(1).rolling(max(5, fast)).min()
    trend_ok = (close > trend_ma) & (trend_ma > trend_ma.shift(3))
    volume_ok = volume > volume.rolling(5).mean() * 0.75
    entries = (rsi_line > 35) & (rsi_line.shift(1) <= 35) & trend_ok & volume_ok
    exits = (rsi_line > 72) | (close < trend_ma) | (close < recent_low)
    return close.rolling(fast).mean(), trend_ma, entries.fillna(False), exits.fillna(False)


def make_rsi_capital_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    open_ = data["Open"]
    low = data["Low"]
    volume = data["Volume"]
    rsi_line = rsi(close, fast)
    trend_ma = close.rolling(slow).mean()
    mfi_line = money_flow_index(data, 14)
    obv_line = obv(data)
    obv_ma = obv_line.rolling(10).mean()
    volume_ratio = volume / volume.rolling(10).mean()
    recent_low = low.shift(1).rolling(max(5, fast)).min()

    trend_ok = (close > trend_ma) & (trend_ma > trend_ma.shift(3))
    rsi_turn = ((rsi_line > 38) & (rsi_line.shift(1) <= 38)) | ((rsi_line > 45) & (rsi_line > rsi_line.shift(2)))
    mfi_turn = ((mfi_line > 42) & (mfi_line > mfi_line.shift(2))) | ((mfi_line > 50) & (mfi_line.shift(1) <= 50))
    obv_ok = (obv_line > obv_ma) & (obv_ma > obv_ma.shift(3))
    volume_ok = (volume_ratio > 0.85) & (volume_ratio < 3.2)
    price_ok = (close > open_) | (close > close.shift(1))

    entries = rsi_turn & trend_ok & mfi_turn & obv_ok & volume_ok & price_ok
    exits = (rsi_line > 76) | (mfi_line > 84) | (close < trend_ma) | (close < recent_low) | ((obv_line < obv_ma) & (close < close.rolling(max(3, fast)).mean()))
    return close.rolling(fast).mean(), trend_ma, entries.fillna(False), exits.fillna(False)


def make_macd_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    golden_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    recent_cross = golden_cross.shift(1).rolling(3).max().fillna(0).astype(bool)
    trend_ok = (close > ema_slow) & (ema_slow > ema_slow.shift(5))
    momentum_ok = (hist > 0) & (hist > hist.shift(1))
    price_confirm = close > high.shift(1)
    volume_ok = volume > volume.rolling(5).mean() * 0.9
    not_chasing = close / ema_fast < 1.075

    entries = recent_cross & trend_ok & momentum_ok & price_confirm & volume_ok & not_chasing

    death_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    recent_low = low.shift(1).rolling(max(5, fast)).min()
    exits = death_cross | (close < ema_fast) | (close < recent_low) | ((hist < 0) & (hist < hist.shift(1)))
    return ema_fast, ema_slow, entries.fillna(False), exits.fillna(False)


def make_macd_kdj_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    low = data["Low"]
    volume = data["Volume"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    k, d, _j = kdj(data, 9)
    trend_ok = (close > ema_slow) & (ema_slow > ema_slow.shift(5))
    macd_ok = (macd_line > signal_line) & (hist > hist.shift(1))
    kdj_cross = (k > d) & (k.shift(1) <= d.shift(1))
    kdj_repair = (k < 68) & (k > k.shift(2))
    volume_ok = volume > volume.rolling(5).mean() * 0.8
    recent_low = low.shift(1).rolling(max(6, fast)).min()
    entries = trend_ok & macd_ok & kdj_cross & kdj_repair & volume_ok
    exits = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))) | ((k < d) & (k > 75)) | (close < ema_fast) | (close < recent_low)
    return ema_fast, ema_slow, entries.fillna(False), exits.fillna(False)


def make_boll_wr_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    low = data["Low"]
    volume = data["Volume"]
    ma_fast = close.rolling(fast).mean()
    mid = close.rolling(slow).mean()
    std = close.rolling(slow).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    wr_line = williams_r(data, max(10, fast))
    recent_low = low.shift(1).rolling(max(6, fast)).min()
    wr_repair = (wr_line > -80) & (wr_line.shift(1) <= -80)
    boll_repair = (close > lower) & (close.shift(1) <= lower.shift(1) * 1.02)
    trend_filter = close > mid * 0.96
    volume_ok = volume > volume.rolling(5).mean() * 0.75
    entries = wr_repair & boll_repair & trend_filter & volume_ok
    exits = (wr_line > -18) | (close > upper) | (close < recent_low) | (close < mid * 0.97)
    return ma_fast, mid, entries.fillna(False), exits.fillna(False)


def make_breakout_capital_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    prior_high = high.shift(1).rolling(max(10, slow)).max()
    recent_low = low.shift(1).rolling(max(6, fast)).min()
    volume_ratio = volume / volume.rolling(20).mean()
    obv_line = obv(data)
    obv_ma = obv_line.rolling(10).mean()
    mfi_line = money_flow_index(data, 14)
    trend_ok = (ma_fast > ma_slow) & (close > ma_slow)
    capital_ok = (obv_line > obv_ma) & (obv_ma > obv_ma.shift(3)) & (mfi_line > 48)
    breakout = close > prior_high
    not_chasing = close / ma_fast < 1.13
    entries = breakout & trend_ok & capital_ok & (volume_ratio > 1.05) & (volume_ratio < 4.0) & not_chasing
    exits = (close < ma_fast) | (close < recent_low) | ((obv_line < obv_ma) & (close < close.shift(1)))
    return ma_fast, ma_slow, entries.fillna(False), exits.fillna(False)


def ml_feature_frame(data: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    open_ = data["Open"]
    volume = data["Volume"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    rsi_line = rsi(close, 6)
    rsi_slow = rsi(close, 14)
    atr_pct = atr(data, 14) / close
    body = (close - open_) / open_
    day_range = (high - low).replace(0, np.nan)
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / day_range
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / day_range
    return pd.DataFrame(
        {
            "qlib_kmid": (close - open_) / open_,
            "qlib_klen": (high - low) / open_,
            "qlib_kup": (high - pd.concat([open_, close], axis=1).max(axis=1)) / open_,
            "qlib_klow": (pd.concat([open_, close], axis=1).min(axis=1) - low) / open_,
            "qlib_ksft": (2 * close - high - low) / open_,
            "qlib_open_close": open_ / close,
            "qlib_high_close": high / close,
            "qlib_low_close": low / close,
            "ret_1": close.pct_change(1),
            "ret_2": close.pct_change(2),
            "ret_3": close.pct_change(3),
            "ret_5": close.pct_change(5),
            "ret_10": close.pct_change(10),
            "ret_20": close.pct_change(20),
            "volatility_5": close.pct_change().rolling(5).std(),
            "volatility_10": close.pct_change().rolling(10).std(),
            "volatility_20": close.pct_change().rolling(20).std(),
            "drawdown_10": close / close.rolling(10).max() - 1,
            "drawdown_20": close / close.rolling(20).max() - 1,
            "breakout_gap": close / high.shift(1).rolling(10).max() - 1,
            "support_gap": close / low.shift(1).rolling(10).min() - 1,
            "ma_gap": fast_ma / slow_ma - 1,
            "close_fast_gap": close / fast_ma - 1,
            "close_slow_gap": close / slow_ma - 1,
            "fast_slope": fast_ma / fast_ma.shift(3) - 1,
            "slow_slope": slow_ma / slow_ma.shift(5) - 1,
            "rsi_6": rsi_line / 100,
            "rsi_14": rsi_slow / 100,
            "atr_pct": atr_pct,
            "vol_ratio": volume / volume.rolling(10).mean() - 1,
            "vol_ratio_3": volume / volume.rolling(3).mean() - 1,
            "body": body,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "ma_5_gap": close / close.rolling(5).mean() - 1,
            "ma_10_gap": close / close.rolling(10).mean() - 1,
            "ma_20_gap": close / close.rolling(20).mean() - 1,
            "ma_60_gap": close / close.rolling(60).mean() - 1,
            "std_5": close.pct_change().rolling(5).std(),
            "std_20": close.pct_change().rolling(20).std(),
            "volume_ma_5": volume / volume.rolling(5).mean() - 1,
            "volume_ma_20": volume / volume.rolling(20).mean() - 1,
        }
    ).replace([np.inf, -np.inf], np.nan)


def _fit_fast_stacking_model(train_x: pd.DataFrame, train_y: pd.Series):
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_specs = []
    has_gpu_xgb = False
    try:
        from xgboost import XGBClassifier

        base_specs.append(
            (
                "xgb_gpu",
                XGBClassifier(
                    n_estimators=130,
                    max_depth=3,
                    learning_rate=0.045,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    device="cuda",
                    random_state=11,
                    n_jobs=1,
                ),
            )
        )
        has_gpu_xgb = True
    except Exception:
        pass

    base_specs.extend([
        (
            "logit",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.6, class_weight="balanced", max_iter=180, solver="lbfgs"),
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=70,
                max_depth=4,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=13,
                n_jobs=1,
            ),
        ),
        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=90,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=17,
                n_jobs=1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=55,
                learning_rate=0.055,
                max_depth=2,
                random_state=19,
            ),
        ),
    ])
    if has_gpu_xgb:
        base_specs = [spec for spec in base_specs if spec[0] in {"xgb_gpu", "logit"}]

    train_y = train_y.astype(int)
    if len(train_x) < 180 or train_y.nunique() < 2:
        model = base_specs[0][1]
        model.fit(train_x, train_y)
        return {"base": [model], "meta": None}

    split = max(120, int(len(train_x) * 0.72))
    split = min(split, len(train_x) - 60)
    base_x, base_y = train_x.iloc[:split], train_y.iloc[:split]
    meta_x, meta_y = train_x.iloc[split:], train_y.iloc[split:]

    meta_base_models = []
    meta_features: list[np.ndarray] = []
    for _name, model in base_specs:
        try:
            fitted = model.fit(base_x, base_y)
        except Exception:
            continue
        meta_base_models.append(fitted)
        meta_features.append(fitted.predict_proba(meta_x)[:, 1])
    if not meta_base_models:
        model = base_specs[-1][1]
        model.fit(train_x, train_y)
        return {"base": [model], "meta": None}

    meta_model = None
    if meta_y.nunique() >= 2:
        meta_train = np.vstack(meta_features).T
        meta_model = LogisticRegression(C=0.8, class_weight="balanced", max_iter=180, solver="lbfgs")
        meta_model.fit(meta_train, meta_y)

    full_base_models = []
    for _name, model in base_specs:
        try:
            full_base_models.append(model.fit(train_x, train_y))
        except Exception:
            continue
    if not full_base_models:
        full_base_models = meta_base_models
    return {"base": full_base_models, "meta": meta_model}


def _predict_fast_stacking(model_pack: dict[str, object], pred_x: pd.DataFrame) -> np.ndarray:
    base_models = list(model_pack.get("base") or [])
    if not base_models:
        return np.full(len(pred_x), np.nan)
    base_probs = np.vstack([model.predict_proba(pred_x)[:, 1] for model in base_models]).T
    meta_model = model_pack.get("meta")
    if meta_model is not None:
        return meta_model.predict_proba(base_probs)[:, 1]
    return base_probs.mean(axis=1)


def _clip_score(value: float) -> float:
    if not np.isfinite(value):
        return 50.0
    return float(np.clip(value, 0, 100))


def factor_score_snapshot(data: pd.DataFrame) -> dict[str, object]:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    latest = float(close.iloc[-1])
    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ret5 = float(close.pct_change(5).iloc[-1])
    ret20 = float(close.pct_change(20).iloc[-1])
    rsi6 = float(rsi(close, 6).iloc[-1])
    atr_pct = float((atr(data, 14) / close).iloc[-1])
    vol_ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if float(volume.rolling(20).mean().iloc[-1] or 0) > 0 else 1.0
    breakout_gap = float(latest / high.shift(1).rolling(20).max().iloc[-1] - 1)
    support_gap = float(latest / low.shift(1).rolling(20).min().iloc[-1] - 1)

    trend_score = _clip_score(50 + (latest / ma20 - 1) * 700 + (ma20 / ma60 - 1) * 500)
    momentum_score = _clip_score(50 + ret5 * 550 + ret20 * 260 + breakout_gap * 300)
    rsi_score = _clip_score(100 - abs(rsi6 - 55) * 2.0)
    volume_score = _clip_score(50 + min(vol_ratio - 1, 1.6) * 22 - max(1 - vol_ratio, 0) * 20)
    risk_penalty = _clip_score(atr_pct * 1700 + max(-support_gap, 0) * 250)
    total = _clip_score(trend_score * 0.30 + momentum_score * 0.25 + rsi_score * 0.18 + volume_score * 0.17 + (100 - risk_penalty) * 0.10)
    return {
        "score": round(total, 2),
        "trend": round(trend_score, 2),
        "momentum": round(momentum_score, 2),
        "rsi": round(rsi_score, 2),
        "volume": round(volume_score, 2),
        "risk_penalty": round(risk_penalty, 2),
        "ret5_pct": round(ret5 * 100, 2),
        "ret20_pct": round(ret20 * 100, 2),
        "rsi6": round(rsi6, 2),
        "atr_pct": round(atr_pct * 100, 2),
        "volume_ratio": round(vol_ratio, 2),
        "breakout_gap_pct": round(breakout_gap * 100, 2),
        "support_gap_pct": round(support_gap * 100, 2),
    }


def predict_ml_horizon(data: pd.DataFrame, days: int, fast: int = 6, slow: int = 20) -> dict[str, object]:
    close = data["Close"]
    features = ml_feature_frame(data, fast, slow)
    future_ret = close.shift(-days) / close - 1
    atr_pct = atr(data, 14) / close
    base_target = {3: 0.008, 5: 0.012, 10: 0.020}.get(days, 0.012)
    dynamic_target = pd.concat(
        [pd.Series(base_target, index=close.index), atr_pct * math.sqrt(days) * 0.32],
        axis=1,
    ).max(axis=1)
    label = (future_ret > dynamic_target).astype(float)
    valid_x = features.loc[future_ret.dropna().index].dropna()
    train_y = label.loc[valid_x.index].astype(int)
    train_ret = future_ret.loc[valid_x.index]
    if len(valid_x) < 220 or train_y.nunique() < 2:
        return {
            "days": days,
            "up_prob": np.nan,
            "expected_return_pct": np.nan,
            "target_return_pct": round(base_target * 100, 2),
            "sample_count": int(len(valid_x)),
            "confidence": "low",
            "detail": "样本不足或标签单一，暂不输出可靠概率",
        }

    train_x = valid_x.tail(900)
    train_y = train_y.loc[train_x.index]
    train_ret = train_ret.loc[train_x.index]
    model = _fit_fast_stacking_model(train_x, train_y)
    latest_x = features.dropna().tail(1)
    if latest_x.empty:
        up_prob = np.nan
    else:
        up_prob = float(_predict_fast_stacking(model, latest_x)[0])

    calib_x = train_x.tail(min(260, len(train_x)))
    calib_probs = pd.Series(_predict_fast_stacking(model, calib_x), index=calib_x.index)
    nearest = (calib_probs - up_prob).abs().sort_values().head(max(20, min(80, len(calib_probs) // 3))).index
    expected_ret = float(train_ret.loc[nearest].mean()) if len(nearest) else float(train_ret.mean())
    hit_rate = float(((calib_probs > calib_probs.median()).astype(int) == train_y.loc[calib_probs.index]).mean())
    if len(train_x) >= 600 and hit_rate >= 0.55:
        confidence = "high"
    elif len(train_x) >= 360 and hit_rate >= 0.51:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "days": days,
        "up_prob": round(up_prob, 3),
        "expected_return_pct": round(expected_ret * 100, 2),
        "target_return_pct": round(float(dynamic_target.dropna().iloc[-1]) * 100, 2),
        "sample_count": int(len(train_x)),
        "hit_rate": round(hit_rate, 3),
        "confidence": confidence,
        "detail": f"{days}日模型：样本 {len(train_x)}，样本内方向命中 {hit_rate:.1%}",
    }


def _text_value(row: pd.Series, candidates: list[str]) -> str:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return ""


def _call_with_timeout(func, seconds: float):
    import queue as queue_lib
    import threading

    result_queue: queue_lib.Queue = queue_lib.Queue(maxsize=1)

    def runner() -> None:
        try:
            result_queue.put(("ok", func()))
        except Exception as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    try:
        kind, payload = result_queue.get(timeout=seconds)
    except queue_lib.Empty as exc:
        raise TimeoutError(f"数据源超过 {seconds:.0f} 秒未返回") from exc
    if kind == "ok":
        return payload
    raise payload


def stock_text_risk_snapshot(symbol: str, as_of: pd.Timestamp | None = None, lookback_days: int = 14) -> dict[str, object]:
    code = normalize_symbol(symbol)
    as_of = pd.Timestamp(as_of or pd.Timestamp.now())
    cache_key = (code, as_of.strftime("%Y%m%d"), int(lookback_days))
    cached = TEXT_RISK_CACHE.get(cache_key)
    if cached:
        return cached
    start = (as_of - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    end = as_of.strftime("%Y%m%d")
    items: list[dict[str, object]] = []
    errors: list[str] = []

    def add_item(source: str, kind: str, row: pd.Series) -> None:
        title = _text_value(row, ["公告标题", "新闻标题", "标题", "报告名称", "title", "内容"])
        date_text = _text_value(row, ["公告时间", "发布时间", "日期", "时间", "datetime", "date"])
        link = _text_value(row, ["公告链接", "新闻链接", "链接", "url", "报告PDF链接"])
        if title:
            items.append({"source": source, "kind": kind, "title": title, "date": date_text, "link": link})

    try:
        import akshare as ak

        notice_df = _call_with_timeout(
            lambda: ak.stock_zh_a_disclosure_report_cninfo(symbol=code, start_date=start, end_date=end),
            4,
        )
        if isinstance(notice_df, pd.DataFrame):
            for _, row in notice_df.head(30).iterrows():
                add_item("巨潮公告", "公告", row)
    except Exception as exc:
        errors.append(f"巨潮公告失败: {exc}")

    errors.append("东方财富新闻源暂未启用：当前环境接口可能阻塞，后续建议通过 RSSHub 异步补充")

    severe_words = ["立案", "行政处罚", "退市", "重大违法", "破产", "债务逾期", "刑事", "暂停上市", "被调查"]
    high_words = ["减持", "监管函", "问询函", "诉讼", "仲裁", "亏损", "预亏", "业绩预减", "业绩下降", "质押", "冻结", "解禁", "风险提示", "停牌"]
    watch_words = ["担保", "关联交易", "重组终止", "延期", "变更", "高管", "会计差错", "商誉", "计提"]
    positive_words = ["回购", "增持", "中标", "订单", "合同", "业绩增长", "预增", "分红", "解除质押", "获批", "突破"]

    negative_score = 0
    positive_score = 0
    hit_details: list[str] = []
    positive_hits: list[str] = []
    for item in items:
        title = str(item.get("title", ""))
        item_hits = [word for word in severe_words if word in title]
        if item_hits:
            negative_score += 8
            hit_details.append(f"{item.get('kind')}: {title}")
            continue
        item_hits = [word for word in high_words if word in title]
        if item_hits:
            negative_score += 4
            hit_details.append(f"{item.get('kind')}: {title}")
            continue
        item_hits = [word for word in watch_words if word in title]
        if item_hits:
            negative_score += 2
            hit_details.append(f"{item.get('kind')}: {title}")
        good_hits = [word for word in positive_words if word in title]
        if good_hits:
            positive_score += 2
            positive_hits.append(f"{item.get('kind')}: {title}")

    if negative_score >= 8:
        level = "high"
    elif negative_score >= 3:
        level = "watch"
    elif items:
        level = "normal"
    else:
        level = "unknown"
    sentiment_score = _clip_score(55 + positive_score * 5 - negative_score * 7)
    detail = "未发现明显新闻/公告风险"
    if hit_details:
        detail = "；".join(hit_details[:3])
    elif not items:
        detail = "最近未拉到可用新闻/公告" if not errors else "新闻/公告源暂不可用"
    result = {
        "status": "ok" if items else "no_data",
        "level": level,
        "score": round(sentiment_score, 2),
        "negative_score": negative_score,
        "positive_score": positive_score,
        "items_count": len(items),
        "detail": detail,
        "positive_detail": "；".join(positive_hits[:3]),
        "items": items[:8],
        "errors": errors[:3],
    }
    TEXT_RISK_CACHE[cache_key] = result
    return result


def build_ml_prediction_snapshot(data: pd.DataFrame, symbol: str = "", fast: int = 6, slow: int = 20) -> dict[str, object]:
    if len(data) < 260:
        raise ValueError("历史数据太少，ML预测至少需要约260个交易日")
    close = data["Close"]
    latest_close = float(close.iloc[-1])
    horizons = [predict_ml_horizon(data, days, fast, slow) for days in (3, 5, 10)]
    factor = factor_score_snapshot(data)
    anomaly = detect_latest_anomaly(data, fast, slow)
    mc = monte_carlo_risk(data, stop_line=None, days=10, simulations=1500)
    text_risk = stock_text_risk_snapshot(symbol, as_of=data.index[-1]) if symbol else {
        "status": "not_connected",
        "level": "unknown",
        "score": None,
        "detail": "未提供股票代码，无法拉取新闻/公告",
    }

    prob3 = float(horizons[0].get("up_prob", np.nan))
    prob5 = float(horizons[1].get("up_prob", np.nan))
    prob10 = float(horizons[2].get("up_prob", np.nan))
    prob_score = np.nanmean([prob3 * 100, prob5 * 100, prob10 * 100])
    if not np.isfinite(prob_score):
        prob_score = 50.0
    anomaly_level = str(anomaly.get("level", "unknown"))
    news_level = str(text_risk.get("level", "unknown"))
    anomaly_penalty = {"normal": 0, "watch": 8, "high": 18, "severe": 32, "unknown": 10}.get(anomaly_level, 10)
    news_penalty = {"normal": 0, "watch": 8, "high": 18, "severe": 32, "unknown": 4}.get(news_level, 4)
    mc_var = abs(float(mc.get("var_95_pct", -8) or -8))
    risk_score = _clip_score(100 - anomaly_penalty - news_penalty - mc_var * 2.2 - float(factor.get("risk_penalty", 50)) * 0.25)
    composite = _clip_score(prob_score * 0.42 + float(factor["score"]) * 0.30 + risk_score * 0.20 + max(float(mc.get("up_prob", 0.5)) * 100, 0) * 0.08)

    if composite >= 72 and prob5 >= 0.58 and risk_score >= 58:
        view = "偏多候选"
    elif composite >= 60 and prob5 >= 0.52:
        view = "观察偏多"
    elif risk_score < 42 or anomaly_level in {"high", "severe"}:
        view = "风险优先"
    else:
        view = "中性观察"

    risk_reasons = []
    if anomaly_level in {"high", "severe"}:
        risk_reasons.append(f"异常等级 {anomaly_level}")
    if news_level in {"watch", "high", "severe"}:
        risk_reasons.append(f"新闻公告风险 {news_level}")
    if float(horizons[2].get("expected_return_pct", 0) or 0) < -1.0:
        risk_reasons.append("10日预期收益偏弱")
    if float(factor.get("atr_pct", 0) or 0) >= 4.0:
        risk_reasons.append("波动率偏高")
    if float(mc.get("var_95_pct", 0) or 0) <= -8.0:
        risk_reasons.append("蒙特卡洛尾部风险偏大")
    if risk_score < 38 or anomaly_level == "severe" or news_level == "severe":
        holding_level = "高风险"
    elif risk_score < 55 or anomaly_level == "high" or news_level == "high":
        holding_level = "风险升高"
    elif risk_score < 68 or anomaly_level == "watch" or news_level == "watch":
        holding_level = "观察"
    else:
        holding_level = "正常"
    if not risk_reasons:
        risk_reasons.append("未发现明显异常")

    return {
        "symbol": normalize_symbol(symbol) if symbol else "",
        "name": stock_display_name(symbol) if symbol else "",
        "date": data.index[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_close,
        "horizons": horizons,
        "factor": factor,
        "anomaly": anomaly,
        "anomaly_components": {
            "price_volume": anomaly,
            "fund_flow": {"status": "not_connected", "detail": "资金流/大单数据源未接入，暂不参与异常评分"},
            "sentiment": text_risk,
        },
        "monte_carlo": mc,
        "news_sentiment": text_risk,
        "holding_risk": {
            "level": holding_level,
            "score": round(100 - risk_score, 2),
            "detail": "；".join(risk_reasons),
        },
        "risk_score": round(risk_score, 2),
        "composite_score": round(composite, 2),
        "view": view,
    }


def make_ml_direction_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    atr_pct = atr(data, 14) / close
    future_days = 3
    future_ret = close.shift(-future_days) / close - 1
    dynamic_target = pd.concat([pd.Series(0.012, index=close.index), atr_pct * 0.65], axis=1).max(axis=1)
    features = ml_feature_frame(data, fast, slow)
    label = (future_ret > dynamic_target).astype(int)

    proba = pd.Series(np.nan, index=close.index)
    min_train = 220
    train_window = 480
    retrain_step = 120
    end_limit = len(close) - future_days
    for i in range(min_train, end_limit, retrain_step):
        start_i = max(0, i - future_days - train_window)
        train_x = features.iloc[start_i : i - future_days].dropna()
        train_y = label.loc[train_x.index]
        if len(train_x) < min_train or train_y.nunique() < 2:
            continue
        model = _fit_fast_stacking_model(train_x, train_y)
        pred_x = features.iloc[i : min(end_limit, i + retrain_step)].dropna()
        if len(pred_x):
            proba.loc[pred_x.index] = _predict_fast_stacking(model, pred_x)

    proba_smooth = proba.rolling(3, min_periods=2).mean()
    exit_line = proba_smooth.rolling(160, min_periods=60).quantile(0.32).clip(lower=0.36, upper=0.48)

    if fast <= 6:
        base_fast, base_slow, proba_threshold = max(6, fast), max(20, slow), 0.45
        price_fast, price_slow, base_entries, base_exits = make_rsi_pullback_signals(data, base_fast, base_slow)
    elif fast <= 8:
        base_fast, base_slow, proba_threshold = 8, max(21, slow), 0.50
        price_fast, price_slow, base_entries, base_exits = make_macd_signals(data, base_fast, base_slow)
    elif fast >= 13:
        base_fast, base_slow, proba_threshold = fast, slow, 0.50
        price_fast, price_slow, base_entries, base_exits = make_signals(close, base_fast, base_slow)
    else:
        base_fast, base_slow, proba_threshold = fast, slow, 0.52
        price_fast, price_slow, base_entries, base_exits = make_hybrid_vote_signals(data, base_fast, base_slow)

    ml_confirm = proba_smooth > proba_threshold
    entries = base_entries & ml_confirm
    probability_break = (proba_smooth < exit_line) & (close < close.rolling(max(3, min(fast, 10))).mean())
    exits = base_exits | probability_break
    return proba_smooth, slow_ma, entries.fillna(False), exits.fillna(False)


def make_hybrid_vote_signals(data: pd.DataFrame, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = data["Close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    _, _, breakout_entries, breakout_exits = make_short_signals(data, fast, slow)
    rsi_fast = 6 if fast <= 8 else 9
    _, _, rsi_entries, rsi_exits = make_rsi_pullback_signals(data, rsi_fast, slow)
    macd_fast = 8 if fast <= 8 else 12
    macd_slow = max(21, slow)
    _, _, macd_entries, macd_exits = make_macd_signals(data, macd_fast, macd_slow)

    sma_bull = (fast_ma > slow_ma) & (close > slow_ma)
    breakout_bull = close > data["High"].shift(1).rolling(max(5, fast)).max()
    rsi_line = rsi(close, rsi_fast)
    rsi_bull = (rsi_line > 42) & (rsi_line < 72)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_bull = macd_line > signal_line

    vote_score = (
        sma_bull.astype(int)
        + breakout_bull.astype(int)
        + rsi_bull.astype(int)
        + macd_bull.astype(int)
    ).astype(float)
    entry_threshold = pd.Series(3.0, index=data.index)
    recent_low = data["Low"].shift(1).rolling(max(5, fast)).min()
    entries = (vote_score >= 3) & (vote_score.shift(1) < 3) & (close > slow_ma)
    exits = (vote_score <= 1) | (close < fast_ma) | (close < recent_low) | breakout_exits | rsi_exits | macd_exits
    return fast_ma, slow_ma, entries.fillna(False), exits.fillna(False)


def detect_latest_anomaly(data: pd.DataFrame, fast: int, slow: int) -> dict[str, object]:
    from sklearn.ensemble import IsolationForest

    features = ml_feature_frame(data, fast, slow).dropna().tail(760)
    if len(features) < 120:
        return {"level": "unknown", "score": 0.0, "detail": "样本不足，暂不做异常检测"}
    latest = features.tail(1)
    train = features.iloc[:-1]
    if len(train) < 100:
        return {"level": "unknown", "score": 0.0, "detail": "样本不足，暂不做异常检测"}
    detector = IsolationForest(n_estimators=120, contamination=0.06, random_state=23)
    detector.fit(train)
    train_scores = -detector.decision_function(train)
    latest_score = float(-detector.decision_function(latest)[0])
    percentile = float((train_scores <= latest_score).mean())

    close = data["Close"]
    volume = data["Volume"]
    daily_ret = close.pct_change()
    latest_ret = float(daily_ret.iloc[-1])
    ret_std = float(daily_ret.tail(60).std() or 0)
    volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if float(volume.tail(20).mean() or 0) > 0 else 1.0
    volatility_ratio = float(daily_ret.tail(5).std() / daily_ret.tail(60).std()) if float(daily_ret.tail(60).std() or 0) > 0 else 1.0

    if percentile >= 0.97 or abs(latest_ret) > ret_std * 3.0 or volume_ratio >= 3.0:
        level = "severe"
    elif percentile >= 0.92 or abs(latest_ret) > ret_std * 2.2 or volume_ratio >= 2.0 or volatility_ratio >= 2.0:
        level = "high"
    elif percentile >= 0.84 or volume_ratio >= 1.5:
        level = "watch"
    else:
        level = "normal"

    return {
        "level": level,
        "score": round(percentile, 3),
        "latest_return_pct": round(latest_ret * 100, 2),
        "volume_ratio": round(volume_ratio, 2),
        "volatility_ratio": round(volatility_ratio, 2),
        "detail": f"异常分位 {percentile:.0%}，量能 {volume_ratio:.2f}x，短波动 {volatility_ratio:.2f}x",
    }


def monte_carlo_risk(data: pd.DataFrame, stop_line: float | None = None, days: int = 10, simulations: int = 2000) -> dict[str, object]:
    close = data["Close"].dropna()
    if len(close) < 80:
        return {"days": days, "detail": "样本不足，暂不做蒙特卡洛"}
    latest = float(close.iloc[-1])
    log_ret = np.log(close / close.shift(1)).dropna().tail(180)
    mu = float(log_ret.mean())
    sigma = float(log_ret.std())
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.018
    date_seed = int(pd.Timestamp(close.index[-1]).strftime("%Y%m%d"))
    price_seed = int(round(latest * 1000))
    seed = int((date_seed * 1000003 + price_seed * 9176 + days * 37) % (2**32))
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=(simulations, days))
    paths = latest * np.exp(np.cumsum(shocks, axis=1))
    terminal = paths[:, -1]
    terminal_ret = terminal / latest - 1
    stop_break_prob = None
    if stop_line and stop_line > 0:
        stop_break_prob = float((paths.min(axis=1) <= stop_line).mean())
    return {
        "days": days,
        "simulations": simulations,
        "expected_return_pct": round(float(np.mean(terminal_ret)) * 100, 2),
        "up_prob": round(float((terminal > latest).mean()), 3),
        "down_prob": round(float((terminal < latest).mean()), 3),
        "var_95_pct": round(float(np.quantile(terminal_ret, 0.05)) * 100, 2),
        "low_price": round(float(np.quantile(terminal, 0.1)), 2),
        "mid_price": round(float(np.quantile(terminal, 0.5)), 2),
        "high_price": round(float(np.quantile(terminal, 0.9)), 2),
        "stop_break_prob": None if stop_break_prob is None else round(stop_break_prob, 3),
    }


def build_ml_risk_snapshot(data: pd.DataFrame, fast: int, slow: int, stop_line: float | None = None) -> dict[str, object]:
    anomaly = detect_latest_anomaly(data, fast, slow)
    mc = monte_carlo_risk(data, stop_line=stop_line, days=10, simulations=2000)
    return {"anomaly": anomaly, "monte_carlo": mc}


def attach_ml_risk_snapshot(
    result: dict[str, object],
    data: pd.DataFrame,
    fast: int,
    slow: int,
    strategy_type: str,
    stop_line: float | None = None,
) -> dict[str, object]:
    if strategy_type != "ml":
        return result
    signal = result.get("daily_signal")
    if not isinstance(signal, dict):
        return result
    risk = build_ml_risk_snapshot(data, fast, slow, stop_line)
    signal["ml_risk"] = risk
    result["ml_risk"] = risk
    return result


def strategy_signals(
    data: pd.DataFrame,
    fast: int,
    slow: int,
    horizon: str,
    strategy_type: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if strategy_type == "breakout":
        return make_short_signals(data, fast, slow)
    if strategy_type == "rsi":
        return make_rsi_pullback_signals(data, fast, slow)
    if strategy_type == "rsi_capital":
        return make_rsi_capital_signals(data, fast, slow)
    if strategy_type == "macd":
        return make_macd_signals(data, fast, slow)
    if strategy_type == "macd_kdj":
        return make_macd_kdj_signals(data, fast, slow)
    if strategy_type == "boll_wr":
        return make_boll_wr_signals(data, fast, slow)
    if strategy_type == "breakout_capital":
        return make_breakout_capital_signals(data, fast, slow)
    if strategy_type == "ml":
        return make_ml_direction_signals(data, fast, slow)
    if strategy_type == "hybrid":
        return make_hybrid_vote_signals(data, fast, slow)
    return make_signals(data["Close"], fast, slow)


def strategy_in_trend(strategy_type: str, latest_fast: float, latest_slow: float, latest_close: float) -> bool:
    if strategy_type == "ml":
        return latest_close > latest_slow and latest_fast >= 0.45
    return latest_fast > latest_slow


def strategy_portfolio(
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    cash: float,
    fee: float,
    horizon: str,
) -> vbt.Portfolio:
    kwargs = {
        "init_cash": cash,
        "fees": fee,
        "freq": "1D",
    }
    if horizon == "short":
        kwargs.update(
            {
                "high": data["High"],
                "low": data["Low"],
                "sl_stop": 0.045,
                "sl_trail": True,
                "tp_stop": 0.095,
            }
        )
    return vbt.Portfolio.from_signals(data["Close"], entries, exits, **kwargs)


def candidate_params(horizon: str, strategy_filter: str) -> list[tuple[str, int, int]]:
    grid = STRATEGY_GRIDS[horizon]
    if strategy_filter == "auto":
        if horizon == "short":
            strategy_types = ["hybrid", "rsi_capital", "macd_kdj", "boll_wr", "breakout_capital", "breakout", "rsi", "macd", "sma"]
        else:
            strategy_types = ["hybrid", "rsi_capital", "macd_kdj", "breakout_capital", "sma", "macd", "breakout"]
    elif strategy_filter == "auto_fast":
        if horizon == "short":
            strategy_types = ["hybrid", "rsi_capital", "macd_kdj", "boll_wr", "breakout_capital", "breakout", "rsi", "macd", "sma"]
        else:
            strategy_types = ["hybrid", "rsi_capital", "macd_kdj", "breakout_capital", "sma", "macd", "breakout"]
    else:
        strategy_types = [strategy_filter]
    candidates: list[tuple[str, int, int]] = []
    for strategy_type in strategy_types:
        if strategy_type == "ml":
            pairs = [(6, 20), (8, 21), (13, 40)]
        elif strategy_type == "rsi":
            pairs = [(6, 20), (6, 30), (9, 30), (9, 40)]
        elif strategy_type == "rsi_capital":
            pairs = [(6, 20), (6, 30), (9, 30), (9, 40), (14, 40)]
        elif strategy_type == "macd":
            pairs = [(5, 20), (8, 21), (12, 26)]
        elif strategy_type == "macd_kdj":
            pairs = [(5, 20), (8, 21), (12, 26)]
        elif strategy_type == "boll_wr":
            pairs = [(10, 20), (14, 20), (14, 30)]
        elif strategy_type == "breakout_capital":
            pairs = [(5, 20), (8, 20), (10, 30), (13, 40)]
        elif strategy_type == "hybrid":
            pairs = [(5, 20), (8, 21), (10, 30), (13, 40)]
        else:
            pairs = [(fast, slow) for fast, slow in product(grid["fast"], grid["slow"]) if fast < slow]
        candidates.extend((strategy_type, fast, slow) for fast, slow in pairs if fast < slow)
    return candidates


def scan_strategies(
    data: pd.DataFrame,
    cash: float,
    fee: float,
    horizon: str,
    strategy_filter: str,
) -> pd.DataFrame:
    grid = STRATEGY_GRIDS[horizon]
    rows = []
    for strategy_type, fast, slow in candidate_params(horizon, strategy_filter):
        _, _, entries, exits = strategy_signals(data, fast, slow, horizon, strategy_type)
        portfolio = strategy_portfolio(data, entries, exits, cash, fee, horizon)
        sharpe = portfolio.sharpe_ratio()
        ret = float(portfolio.total_return()) * 100
        dd = float(portfolio.max_drawdown()) * 100
        trades = int(portfolio.trades.count())
        score = ret - abs(dd) * 0.8 + (float(sharpe) if np.isfinite(sharpe) else 0) * 35
        if horizon == "short":
            score = ret * 0.55 - abs(dd) * 3.2 + (float(sharpe) if np.isfinite(sharpe) else 0) * 35
            score -= max(0, abs(dd) - 16) * 8.0
            score += min(trades, 30) * 1.5
            if abs(dd) > 28:
                score -= 500
            if strategy_type == "ml":
                score -= 15
        if trades < grid["min_trades"]:
            score -= 200
        rows.append(
            {
                "strategy_type": strategy_type,
                "strategy_label": STRATEGY_TYPES[strategy_type],
                "fast": fast,
                "slow": slow,
                "total_return_pct": ret,
                "max_drawdown_pct": dd,
                "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
                "trades": trades,
                "final_value": float(portfolio.final_value()),
                "score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


MIXED_STRATEGY_TYPES = {"hybrid", "rsi_capital", "macd_kdj", "boll_wr", "breakout_capital"}
SINGLE_STRATEGY_TYPES = {"sma", "rsi", "macd", "breakout"}


def prioritized_scan_rows(scan: pd.DataFrame, include_ml: bool = False) -> list[tuple[int, pd.Series, str]]:
    if scan.empty:
        return []
    rows: list[tuple[int, pd.Series, str]] = []
    for original_idx, row in scan.iterrows():
        idx = int(original_idx)
        if not include_ml and str(row.get("strategy_type")) == "ml":
            continue
        rows.append((idx, row, "排名"))
        if len(rows) >= 25:
            break
    return rows


def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def money(value: float) -> str:
    return f"{value:,.2f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def prefixed_symbol(symbol: str) -> str:
    code = normalize_symbol(symbol)
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{code}"


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", value or "").upper()


def load_code_name_table() -> pd.DataFrame:
    global CODE_NAME_CACHE, CODE_TO_NAME, NAME_TO_CODE
    if CODE_NAME_CACHE is not None:
        return CODE_NAME_CACHE.copy()
    import akshare as ak

    table = ak.stock_info_a_code_name().copy()
    table["code"] = table["code"].astype(str).str.zfill(6)
    table["name"] = table["name"].astype(str)
    CODE_TO_NAME = dict(zip(table["code"], table["name"]))
    NAME_TO_CODE = {}
    for _, row in table.iterrows():
        NAME_TO_CODE.setdefault(normalize_name(str(row["name"])), str(row["code"]))
    CODE_NAME_CACHE = table
    return table.copy()


def resolve_stock_identifier(value: str) -> str:
    token = (value or "").strip()
    cleaned = token.upper().replace(".SZSE", "").replace(".SSE", "").replace(".SHSE", "").replace(".SZ", "").replace(".SH", "")
    if cleaned.startswith(("SZ", "SH", "BJ")) and len(cleaned) >= 8:
        cleaned = cleaned[2:]
    if len(cleaned) == 6 and cleaned.isdigit():
        return normalize_symbol(cleaned)

    table = load_code_name_table()
    key = normalize_name(token)
    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]
    matches = table[table["name"].map(normalize_name).str.contains(re.escape(key), na=False)]
    if len(matches) == 1:
        return str(matches.iloc[0]["code"])
    if len(matches) > 1:
        sample = "、".join(f"{row.code} {row.name}" for row in matches.head(5).itertuples())
        raise ValueError(f"{token} 匹配到多只股票：{sample}，请改用股票代码。")
    raise ValueError(f"找不到股票代码或名称：{token}")


def stock_display_name(symbol: str) -> str:
    code = normalize_symbol(symbol)
    if code in CODE_TO_NAME:
        return CODE_TO_NAME[code]
    try:
        load_code_name_table()
        return CODE_TO_NAME.get(code, "")
    except Exception:
        return ""


def xueqiu_symbol(symbol: str) -> str:
    code = normalize_symbol(symbol)
    if code.startswith(("4", "8")):
        prefix = "BJ"
    elif code.startswith(("5", "6", "9")):
        prefix = "SH"
    else:
        prefix = "SZ"
    return f"{prefix}{code}"


def xueqiu_url(symbol: str) -> str:
    return f"https://xueqiu.com/S/{xueqiu_symbol(symbol)}"


def get_intraday_quote(symbol: str) -> dict[str, object] | None:
    try:
        import akshare as ak

        code = prefixed_symbol(symbol)
        spot = ak.stock_zh_a_spot()
        row = spot[spot["代码"].astype(str).eq(code)]
        if row.empty:
            return None
        item = row.iloc[0]
        return {
            "source": "实时行情",
            "name": str(item.get("名称", "")),
            "price": float(item["最新价"]),
            "open": float(item["今开"]),
            "high": float(item["最高"]),
            "low": float(item["最低"]),
            "pct": float(item["涨跌幅"]),
            "volume": float(item["成交量"]),
            "amount": float(item["成交额"]),
            "time": str(item.get("时间戳", "")),
        }
    except Exception:
        return None


def load_spot_table(ttl_seconds: int = 10) -> pd.DataFrame:
    global SPOT_CACHE
    now = time.monotonic()
    if SPOT_CACHE and now - SPOT_CACHE[0] <= ttl_seconds:
        return SPOT_CACHE[1].copy()
    import akshare as ak

    spot = ak.stock_zh_a_spot()
    SPOT_CACHE = (now, spot)
    return spot.copy()


def get_intraday_quote(symbol: str) -> dict[str, object] | None:
    try:
        code = prefixed_symbol(symbol)
        spot = load_spot_table()
        row = spot[spot["代码"].astype(str).str.lower().eq(code)]
        if row.empty:
            return None
        item = row.iloc[0]
        return {
            "source": "实时行情",
            "name": str(item.get("名称", "")),
            "price": float(item["最新价"]),
            "open": float(item["今开"]),
            "high": float(item["最高"]),
            "low": float(item["最低"]),
            "pct": float(item["涨跌幅"]),
            "volume": float(item["成交量"]),
            "amount": float(item["成交额"]),
            "time": str(item.get("时间戳", "")),
        }
    except Exception:
        return None


def normalize_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "day": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "amount": "Amount",
    }
    data = raw.rename(columns=rename_map).copy()
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(data.columns):
        raise RuntimeError(f"Unexpected intraday columns: {list(raw.columns)}")
    if "Amount" not in data.columns:
        data["Amount"] = pd.to_numeric(data["Close"], errors="coerce") * pd.to_numeric(data["Volume"], errors="coerce")
    data = data[required + ["Amount"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    for column in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna().set_index("Date").sort_index()


def load_intraday_minutes(symbol: str, period: str = "5", ttl_seconds: int = 25) -> pd.DataFrame:
    import akshare as ak

    period = period if period in {"1", "5", "15"} else "5"
    code = normalize_symbol(symbol)
    key = (code, period)
    now = time.monotonic()
    cached = INTRADAY_CACHE.get(key)
    if cached and now - cached[0] <= ttl_seconds:
        return cached[1].copy()
    try:
        raw = ak.stock_zh_a_minute(symbol=prefixed_symbol(code), period=period, adjust="")
        data = normalize_intraday(raw)
        if data.empty:
            raise RuntimeError("minute data is empty")
        INTRADAY_CACHE[key] = (now, data)
        return data.copy()
    except Exception:
        if cached:
            return cached[1].copy()
        raise


def parse_symbol_text(text: str) -> list[str]:
    tokens = [token.strip() for token in re.split(r"[\s,\uFF0C;\uFF1B\u3001]+", text or "") if token.strip()]
    symbols: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        symbol = resolve_stock_identifier(token)
        if symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def get_intraday_quote(symbol: str) -> dict[str, object] | None:
    code = prefixed_symbol(symbol)
    for attempt in range(2):
        try:
            spot = load_spot_table(ttl_seconds=8 if attempt == 0 else 0)
            row = spot[spot["代码"].astype(str).str.lower().eq(code)]
            if row.empty:
                continue
            item = row.iloc[0]
            return {
                "source": "实时行情",
                "name": str(item.get("名称", "")),
                "price": float(item["最新价"]),
                "open": float(item["今开"]),
                "high": float(item["最高"]),
                "low": float(item["最低"]),
                "pct": float(item["涨跌幅"]),
                "volume": float(item["成交量"]),
                "amount": float(item["成交额"]),
                "time": str(item.get("时间戳", "")),
            }
        except Exception:
            if attempt == 0:
                time.sleep(0.3)
    return None


def monitor_holding(symbol: str, request_shares: str, request_buy_price: str) -> tuple[int, float | None]:
    saved = FORM_CACHE.get(symbol, {})
    shares_text = request_shares or saved.get("shares", "0")
    price_text = request_buy_price or saved.get("buy_price", "")
    try:
        shares = int(float(shares_text or 0))
    except ValueError:
        shares = 0
    buy_price = parse_float(price_text, None)
    return shares, buy_price


def strategy_cache_key(
    symbol: str,
    start: str,
    adjust: str,
    cash: float,
    fee: float,
    horizon: str,
    strategy_filter: str,
    risk: str,
) -> tuple[str, str, str, str, str, str]:
    return (symbol, start, adjust, str(cash), str(fee), f"{horizon}:{strategy_filter}:{risk}")


def strategy_cache_key_text(cache_key: tuple[str, str, str, str, str, str]) -> str:
    return json.dumps(list(cache_key), ensure_ascii=False, separators=(",", ":"))


def load_persistent_strategy_cache() -> dict[str, dict[str, object]]:
    global PERSISTENT_STRATEGY_CACHE
    if PERSISTENT_STRATEGY_CACHE is not None:
        return PERSISTENT_STRATEGY_CACHE
    if STRATEGY_CACHE_PATH.exists():
        try:
            with STRATEGY_CACHE_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                PERSISTENT_STRATEGY_CACHE = data
                return PERSISTENT_STRATEGY_CACHE
        except Exception:
            pass
    PERSISTENT_STRATEGY_CACHE = {}
    return PERSISTENT_STRATEGY_CACHE


def save_persistent_strategy_cache() -> None:
    cache = load_persistent_strategy_cache()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STRATEGY_CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(STRATEGY_CACHE_PATH)


def load_saved_daily_gate(cache_key: tuple[str, str, str, str, str, str]) -> dict[str, object] | None:
    record = load_persistent_strategy_cache().get(strategy_cache_key_text(cache_key))
    if isinstance(record, dict) and isinstance(record.get("result"), dict):
        result = record["result"]
        DAILY_GATE_CACHE[cache_key] = result
        return result
    return None


def load_saved_daily_gate_by_key_text(key_text: str) -> dict[str, object] | None:
    try:
        parsed = tuple(json.loads(key_text))
    except Exception:
        return None
    if len(parsed) != 6:
        return None
    return load_saved_daily_gate(parsed)  # type: ignore[arg-type]


def daily_gate_strategy_type(result: dict[str, object] | None) -> str:
    if not isinstance(result, dict):
        return ""
    signal = result.get("daily_signal")
    if not isinstance(signal, dict):
        return ""
    return str(signal.get("strategy_type", "")).strip()


def _daily_gate_matches_filter(
    result: dict[str, object] | None,
    strategy_type: str | None = None,
    exclude_strategy_type: str | None = None,
) -> bool:
    result_type = daily_gate_strategy_type(result)
    if strategy_type is not None and result_type != strategy_type:
        return False
    if exclude_strategy_type is not None and result_type == exclude_strategy_type:
        return False
    return True


def load_latest_saved_daily_gate_for_symbol(
    symbol: str,
    strategy_type: str | None = None,
    exclude_strategy_type: str | None = None,
) -> dict[str, object] | None:
    code = normalize_symbol(symbol)
    newest_key: tuple[str, str, str, str, str, str] | None = None
    newest_record: dict[str, object] | None = None
    newest_saved_at = ""
    for key_text, record in load_persistent_strategy_cache().items():
        if not isinstance(record, dict) or normalize_symbol(str(record.get("symbol", ""))) != code:
            continue
        result = record.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("daily_signal"), dict):
            continue
        if not _daily_gate_matches_filter(result, strategy_type, exclude_strategy_type):
            continue
        saved_at = str(record.get("saved_at", ""))
        if newest_record is None or saved_at > newest_saved_at:
            try:
                parsed = tuple(json.loads(key_text))
                if len(parsed) != 6:
                    continue
                newest_key = parsed  # type: ignore[assignment]
            except Exception:
                newest_key = None
            newest_record = record
            newest_saved_at = saved_at
    if newest_record is None:
        return None
    result = newest_record["result"]
    if newest_key is not None:
        DAILY_GATE_CACHE[newest_key] = result
    return result


def save_daily_gate(cache_key: tuple[str, str, str, str, str, str], result: dict[str, object]) -> None:
    cache = load_persistent_strategy_cache()
    symbol = cache_key[0]
    key_text = strategy_cache_key_text(cache_key)
    display_name = str(result.get("name") or "")
    existing_position: dict[str, object] = {}
    existing_record = cache.get(key_text)
    if not display_name and isinstance(existing_record, dict):
        display_name = str(existing_record.get("name") or "")
        existing_result = existing_record.get("result")
        if not display_name and isinstance(existing_result, dict):
            display_name = str(existing_result.get("name") or "")
    if isinstance(existing_record, dict) and isinstance(existing_record.get("position"), dict):
        existing_position = dict(existing_record["position"])
    if not display_name:
        newest_name_saved_at = ""
        for record in cache.values():
            if not isinstance(record, dict):
                continue
            try:
                same_symbol = normalize_symbol(str(record.get("symbol", ""))) == normalize_symbol(symbol)
            except Exception:
                same_symbol = str(record.get("symbol", "")) == symbol
            if not same_symbol:
                continue
            record_name = str(record.get("name") or "")
            if not record_name:
                record_result = record.get("result")
                if isinstance(record_result, dict):
                    record_name = str(record_result.get("name") or "")
            saved_at = str(record.get("saved_at", ""))
            if record_name and saved_at >= newest_name_saved_at:
                newest_name_saved_at = saved_at
                display_name = record_name
    if not display_name:
        display_name = stock_display_name(symbol)
    if display_name:
        result["name"] = display_name
    if not existing_position:
        newest_saved_at = ""
        for record in cache.values():
            if not isinstance(record, dict):
                continue
            try:
                same_symbol = normalize_symbol(str(record.get("symbol", ""))) == normalize_symbol(symbol)
            except Exception:
                same_symbol = str(record.get("symbol", "")) == symbol
            if not same_symbol or not isinstance(record.get("position"), dict):
                continue
            saved_at = str(record.get("saved_at", ""))
            if saved_at >= newest_saved_at:
                newest_saved_at = saved_at
                existing_position = dict(record["position"])
    cache[key_text] = {
        "symbol": symbol,
        "name": display_name,
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "symbol": cache_key[0],
            "start": cache_key[1],
            "adjust": cache_key[2],
            "cash": cache_key[3],
            "fee": cache_key[4],
            "mode": cache_key[5],
        },
        "position": existing_position,
        "result": result,
    }
    save_persistent_strategy_cache()


def compute_daily_gate(form: dict[str, str]) -> dict[str, object]:
    symbol = resolve_stock_identifier(form["symbol"])
    start = form.get("start", "20200101").strip()
    adjust = form.get("adjust", "qfq")
    cash = float(form.get("cash") or 100000)
    fee = float(form.get("fee") or 0.0003)
    risk = form.get("risk", "normal")
    horizon = form.get("horizon", "short")
    if horizon not in STRATEGY_GRIDS:
        horizon = "short"
    strategy_filter = form.get("strategy_type", "auto")
    if strategy_filter not in STRATEGY_TYPES:
        strategy_filter = "auto"

    cache_key = strategy_cache_key(symbol, start, adjust, cash, fee, horizon, strategy_filter, risk)
    cached = DAILY_GATE_CACHE.get(cache_key)
    if cached:
        return cached
    saved = load_saved_daily_gate(cache_key)
    if saved:
        return saved

    data = cached_data(symbol, start, adjust).copy()
    if len(data) < 240:
        raise ValueError("历史数据太少，至少需要约 240 个交易日。")
    scan = scan_strategies(data, cash, fee, horizon, strategy_filter)
    best = scan.iloc[0]
    close = data["Close"]
    latest_date = data.index[-1]
    latest_close = float(close.iloc[-1])
    fast = int(best["fast"])
    slow = int(best["slow"])
    strategy_type = str(best.get("strategy_type", "sma"))
    strategy_label = STRATEGY_TYPES.get(strategy_type, strategy_type)
    fast_ma, slow_ma, entries, exits = strategy_signals(data, fast, slow, horizon, strategy_type)
    latest_fast = float(fast_ma.iloc[-1])
    latest_slow = float(slow_ma.iloc[-1])
    entry_today = bool(entries.iloc[-1])
    exit_today = bool(exits.iloc[-1])
    in_trend = strategy_in_trend(strategy_type, latest_fast, latest_slow, latest_close)
    last_side, last_date = last_signal_date(entries, exits)
    risk_factor = {"tight": 0.8, "normal": 1.0, "loose": 1.3}[risk] if horizon == "short" else {"tight": 1.2, "normal": 1.6, "loose": 2.2}[risk]
    lookback = STRATEGY_GRIDS[horizon]["lookback"]
    atr_value = float(atr(data).iloc[-1])
    recent_low = float(data["Low"].tail(lookback).min())
    recent_high = float(data["High"].tail(lookback).max())
    trend_stop = max(latest_slow, latest_close - risk_factor * atr_value)
    structure_stop = recent_low
    stop_line = min(trend_stop, latest_close * (0.992 if horizon == "short" else 0.985)) if in_trend else max(latest_slow, latest_close * 1.01)
    result = {
        "strategy_label": strategy_label,
        "best_params": f"{fast}/{slow}",
        "signal_lines": [f"Latest daily: {latest_date:%Y-%m-%d}, close {money(latest_close)}."],
        "daily_signal": {
            "date": latest_date.strftime("%Y-%m-%d"),
            "strategy_type": strategy_type,
            "strategy_label": strategy_label,
            "fast": fast,
            "slow": slow,
            "entry_today": entry_today,
            "exit_today": exit_today,
            "in_trend": in_trend,
            "latest_close": latest_close,
            "stop_line": stop_line,
            "structure_stop": structure_stop,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "last_side": last_side,
            "last_date": last_date,
        },
    }
    attach_ml_risk_snapshot(result, data, fast, slow, strategy_type, stop_line)
    DAILY_GATE_CACHE[cache_key] = result
    save_daily_gate(cache_key, result)
    return result


def ensure_daily_strategy(
    symbol: str,
    saved_strategy_key_text: str = "",
    strategy_type: str | None = None,
    exclude_strategy_type: str | None = "ml",
) -> dict[str, object]:
    code = normalize_symbol(symbol)
    if saved_strategy_key_text:
        saved_selected = load_saved_daily_gate_by_key_text(saved_strategy_key_text)
        if saved_selected and _daily_gate_matches_filter(saved_selected, strategy_type, exclude_strategy_type):
            return saved_selected
    saved = load_latest_saved_daily_gate_for_symbol(code, strategy_type=strategy_type, exclude_strategy_type=exclude_strategy_type)
    if saved:
        return saved
    cached = RESULT_CACHE.get(code)
    if cached and cached.get("daily_signal") and _daily_gate_matches_filter(cached, strategy_type, exclude_strategy_type):
        return cached
    form = FORM_CACHE.get(code, default_form()).copy()
    form["symbol"] = code
    form["batch_symbols"] = ""
    if strategy_type is not None:
        form["strategy_type"] = strategy_type
    elif exclude_strategy_type == "ml" and form.get("strategy_type") == "ml":
        form["strategy_type"] = "auto_fast"
    return compute_daily_gate(form)


def build_monitor_item(
    symbol: str,
    period: str,
    request_shares: str = "",
    request_buy_price: str = "",
    saved_strategy_key_text: str = "",
    strategy_type: str | None = None,
    exclude_strategy_type: str | None = "ml",
) -> dict[str, object]:
    code = resolve_stock_identifier(symbol)
    data = load_intraday_minutes(code, period)
    latest_day = data.index[-1].date()
    session = data[data.index.date == latest_day].copy()
    if len(session) < 1:
        raise RuntimeError("分钟线数据太少，暂时无法判断")

    close = session["Close"]
    high = session["High"]
    low = session["Low"]
    volume = session["Volume"]
    amount = session["Amount"]
    vwap = (amount.cumsum() / volume.replace(0, np.nan).cumsum()).ffill()
    fast = close.rolling(5, min_periods=3).mean()
    slow = close.rolling(12, min_periods=5).mean()
    prior_high = high.shift(1).rolling(20, min_periods=5).max()
    prior_low = low.shift(1).rolling(12, min_periods=5).min()
    avg_volume = volume.shift(1).rolling(20, min_periods=5).mean()

    quote = get_intraday_quote(code)
    bar_time = session.index[-1].strftime("%Y-%m-%d %H:%M")
    quote_time = str(quote.get("time", "")).strip() if quote else ""
    price = float(quote["price"]) if quote else float(close.iloc[-1])
    latest_vwap = float(vwap.iloc[-1])
    latest_fast = float(fast.iloc[-1])
    latest_slow = float(slow.iloc[-1])
    latest_prior_high = float(prior_high.iloc[-1]) if not math.isnan(float(prior_high.iloc[-1])) else float(high.max())
    latest_prior_low = float(prior_low.iloc[-1]) if not math.isnan(float(prior_low.iloc[-1])) else float(low.min())
    latest_avg_volume = float(avg_volume.iloc[-1]) if not math.isnan(float(avg_volume.iloc[-1])) else float(volume.mean())
    volume_ratio = float(volume.iloc[-1] / latest_avg_volume) if latest_avg_volume > 0 else 1.0
    day_high = max(float(high.max()), float(quote["high"]) if quote else float(high.max()))
    drop_from_high = price / day_high - 1 if day_high else 0.0

    shares, buy_price = monitor_holding(code, request_shares, request_buy_price)
    cost_stop = buy_price * 0.975 if buy_price else None
    stop_line = max(latest_prior_low, cost_stop) if cost_stop else latest_prior_low
    trend_up = price > latest_vwap and latest_fast >= latest_slow
    trend_down = price < latest_vwap and latest_fast < latest_slow
    breakout = price > latest_prior_high and volume_ratio >= 1.2
    not_chasing = price <= latest_vwap * 1.035

    daily = ensure_daily_strategy(
        code,
        saved_strategy_key_text,
        strategy_type=strategy_type,
        exclude_strategy_type=exclude_strategy_type,
    )
    daily_signal = daily.get("daily_signal", {})
    daily_entry = bool(daily_signal.get("entry_today"))
    daily_exit = bool(daily_signal.get("exit_today"))
    daily_trend = bool(daily_signal.get("in_trend"))
    daily_stop = float(daily_signal.get("stop_line") or stop_line)
    daily_structure_stop = float(daily_signal.get("structure_stop") or latest_prior_low)
    daily_recent_high = float(daily_signal.get("recent_high") or latest_prior_high)
    daily_strategy = f"{daily_signal.get('strategy_label', daily.get('strategy_label', 'Strategy'))} {daily_signal.get('fast', '')}/{daily_signal.get('slow', '')}".strip()
    strategy_type = str(daily_signal.get("strategy_type", "sma"))
    strategy_fast = int(float(daily_signal.get("fast") or 5))
    strategy_slow = int(float(daily_signal.get("slow") or 12))

    if daily_exit:
        daily_gate = "日线卖出"
    elif daily_entry:
        daily_gate = "日线买入"
    elif daily_trend:
        daily_gate = "日线持有"
    else:
        daily_gate = "日线空仓"

    stop_line = max(stop_line, daily_stop if shares > 0 else daily_structure_stop)
    daily_allows_buy = daily_entry
    daily_allows_hold = daily_trend and not daily_exit
    intraday_buy_confirm = breakout and trend_up and not_chasing
    intraday_sell_confirm = price <= stop_line or price <= daily_structure_stop or trend_down or drop_from_high <= -0.025

    reasons: list[str] = [
        f"日线回测策略：{daily_strategy}，闸门状态：{daily_gate}。",
    ]
    if shares > 0:
        if daily_exit or intraday_sell_confirm:
            action_code = "sell"
            action = "卖出/减仓提醒"
            if daily_exit:
                reasons.append("日线策略已经触发卖出闸门，盘中只要不能快速收回风控线，就优先减仓。")
            if intraday_sell_confirm:
                reasons.append("盘中价格跌破 VWAP/分钟趋势或风控线，和日线风控形成共振。")
        elif daily_allows_hold and trend_up:
            action_code = "hold"
            action = "继续持有"
            reasons.append("日线仍允许持有，盘中价格在 VWAP 上方且 5/12 分钟均线维持强势。")
        else:
            action_code = "watch"
            action = "持仓观察"
            reasons.append("日线或分钟线没有形成同向确认，先观察风控线和 VWAP。")
    else:
        if daily_allows_buy and intraday_buy_confirm:
            action_code = "buy"
            action = "买入观察"
            reasons.append("日线回测策略触发买入，盘中也站上 VWAP、分钟趋势向上，并放量突破近 20 根分钟高点。")
        elif daily_allows_buy:
            action_code = "watch"
            action = "等待盘中确认"
            reasons.append("日线有买入资格，但盘中还没同时满足 VWAP、趋势和放量突破，先不追。")
        elif daily_allows_hold and intraday_buy_confirm:
            action_code = "watch"
            action = "趋势内观察"
            reasons.append("日线处在持有区但不是新买点，盘中虽转强，也只作为观察，不当作回测买点。")
        else:
            action_code = "watch"
            action = "暂不买入"
            reasons.append("日线回测没有给买入闸门，盘中信号不单独触发买入。")

    ml_risk = daily_signal.get("ml_risk") if isinstance(daily_signal.get("ml_risk"), dict) else {}
    if strategy_type == "ml" and isinstance(ml_risk, dict):
        anomaly = ml_risk.get("anomaly", {}) if isinstance(ml_risk.get("anomaly"), dict) else {}
        mc = ml_risk.get("monte_carlo", {}) if isinstance(ml_risk.get("monte_carlo"), dict) else {}
        anomaly_level = str(anomaly.get("level", "unknown"))
        stop_prob_raw = mc.get("stop_break_prob")
        stop_prob = float(stop_prob_raw) if isinstance(stop_prob_raw, (int, float)) else 0.0
        reasons.append(f"ML异常检测：{anomaly_level}，{anomaly.get('detail', '暂无详情')}")
        if mc:
            reasons.append(
                f"蒙特卡洛10日：上涨概率 {float(mc.get('up_prob', 0)) * 100:.1f}%，"
                f"VaR95 {mc.get('var_95_pct', '-')}%，跌破风控线概率 {stop_prob * 100:.1f}%。"
            )
        if action_code == "buy" and (anomaly_level in {"high", "severe"} or stop_prob >= 0.35):
            action_code = "watch"
            action = "ML风控拦截"
            reasons.append("虽然买入条件触发，但异常/蒙特卡洛风险偏高，本轮先不自动给买入提醒。")

    if buy_price:
        pnl = price / buy_price - 1
        reasons.append(f"按成本 {money(buy_price)} 估算，当前浮动盈亏 {pnl * 100:.2f}%。")
    reasons.append("A股 T+1，盘中提醒用于辅助执行，最终下单前仍需人工确认。")

    market_note = "最近交易日分钟线"
    if pd.Timestamp.now().date() == latest_day:
        market_note = "今日分钟线"

    market_note = f"K线 {bar_time}"
    updated_text = f"行情 {quote_time}" if quote_time else f"K线 {bar_time}"
    chart_frame = pd.DataFrame({"price": close, "vwap": vwap})
    chart_frame["strat_fast"] = close.rolling(strategy_fast, min_periods=max(2, min(strategy_fast, 5))).mean()
    chart_frame["strat_slow"] = close.rolling(strategy_slow, min_periods=max(2, min(strategy_slow, 8))).mean()
    if strategy_type == "rsi":
        chart_frame["rsi"] = rsi(close, max(2, strategy_fast))
        chart_frame["rsi_low"] = 35.0
        chart_frame["rsi_high"] = 72.0
    elif strategy_type == "macd":
        ema_fast = close.ewm(span=max(2, strategy_fast), adjust=False).mean()
        ema_slow = close.ewm(span=max(strategy_fast + 1, strategy_slow), adjust=False).mean()
        chart_frame["macd"] = ema_fast - ema_slow
        chart_frame["signal"] = chart_frame["macd"].ewm(span=9, adjust=False).mean()
        chart_frame["hist"] = chart_frame["macd"] - chart_frame["signal"]
    chart_frame = chart_frame.dropna(subset=["price", "vwap"]).tail(120)
    chart_points = [
        {
            "time": idx.strftime("%H:%M"),
            "price": round(float(row["price"]), 3),
            "vwap": round(float(row["vwap"]), 3),
            **{
                key: round(float(row[key]), 4)
                for key in ("strat_fast", "strat_slow", "rsi", "rsi_low", "rsi_high", "macd", "signal", "hist")
                if key in row and not pd.isna(row[key])
            },
        }
        for idx, row in chart_frame.iterrows()
    ]
    if quote_time:
        latest_chart = chart_frame.iloc[-1] if not chart_frame.empty else {}
        chart_points.append(
            {
                "time": f"行情 {quote_time}",
                "price": round(price, 3),
                "vwap": round(latest_vwap, 3),
                **{
                    key: round(float(latest_chart[key]), 4)
                    for key in ("strat_fast", "strat_slow", "rsi", "rsi_low", "rsi_high", "macd", "signal", "hist")
                    if key in latest_chart and not pd.isna(latest_chart[key])
                },
            }
        )
    display_name = str(quote.get("name", "")) if quote else stock_display_name(code)

    return {
        "symbol": code,
        "name": display_name,
        "xueqiu_url": xueqiu_url(code),
        "updated": updated_text,
        "market_note": market_note,
        "quote_time": quote_time or "-",
        "bar_time": bar_time,
        "price": money(price),
        "daily_gate": daily_gate,
        "vwap": money(latest_vwap),
        "minute_trend": "强" if trend_up else ("弱" if trend_down else "震荡"),
        "volume_ratio": f"{volume_ratio:.2f}x",
        "stop_line": money(stop_line),
        "stop_value": round(float(stop_line), 3),
        "chart_points": chart_points,
        "chart_strategy_type": strategy_type,
        "chart_strategy_label": daily_strategy,
        "action": action,
        "action_code": action_code,
        "reasons": reasons,
    }


def get_intraday_quote(symbol: str) -> dict[str, object] | None:
    try:
        import requests

        code = normalize_symbol(symbol)
        response = requests.get(f"https://qt.gtimg.cn/q={prefixed_symbol(code)}", timeout=8)
        response.encoding = "gbk"
        text = response.text.strip()
        if '="' not in text:
            return None
        body = text.split('="', 1)[1].rsplit('"', 1)[0]
        parts = body.split("~")
        if len(parts) < 35 or not parts[3]:
            return None
        raw_time = parts[30]
        display_time = raw_time
        if len(raw_time) == 14 and raw_time.isdigit():
            display_time = f"{raw_time[8:10]}:{raw_time[10:12]}:{raw_time[12:14]}"
        return {
            "source": "实时行情",
            "name": parts[1],
            "price": float(parts[3]),
            "open": float(parts[5]),
            "high": float(parts[33]),
            "low": float(parts[34]),
            "pct": float(parts[32]),
            "volume": float(parts[36]) * 100,
            "amount": float(parts[37]) * 10000,
            "time": display_time,
        }
    except Exception:
        return None


def strategy_signal_rules(strategy_type: str) -> list[str]:
    if strategy_type == "sma":
        return [
            "买入信号：短均线上穿长均线，也就是金叉。",
            "卖出信号：短均线下穿长均线，也就是死叉。",
        ]
    if strategy_type == "breakout":
        return [
            "买入信号：价格突破前期高点，同时站在趋势过滤线上方。",
            "卖出信号：跌破短线均线、跌破近期低点，或趋势过滤转弱。",
        ]
    if strategy_type == "rsi":
        return [
            "买入信号：趋势仍向上，RSI 从偏弱区重新上穿，表示回踩后转强。",
            "卖出信号：RSI 过热、跌破趋势线，或跌破近期低点。",
        ]
    if strategy_type == "rsi_capital":
        return [
            "买入信号：RSI 回调转强，同时 MFI 资金流修复、OBV 上行、量能不过热。",
            "卖出信号：RSI/MFI 过热、跌破趋势线，或 OBV 转弱并跌破短线。",
        ]
    if strategy_type == "macd":
        return [
            "买入信号：MACD 金叉只是预警；金叉后 1-3 天内，价格放量突破前一日高点且趋势向上，才给交易买点。",
            "卖出信号：MACD 死叉、跌破快线、跌破近期低点，或 MACD 柱继续走弱。",
        ]
    if strategy_type == "macd_kdj":
        return [
            "买入信号：MACD 动量转强，同时 KDJ 低位/中位金叉，且趋势线向上。",
            "卖出信号：MACD 死叉、KDJ 高位死叉、跌破快线或近期低点。",
        ]
    if strategy_type == "boll_wr":
        return [
            "买入信号：价格从 BOLL 下轨附近修复，WR 从超卖区上穿，且量能配合。",
            "卖出信号：WR 过热、触及 BOLL 上轨、跌破中轨/近期低点。",
        ]
    if strategy_type == "breakout_capital":
        return [
            "买入信号：突破近期高点，同时 OBV/MFI 资金确认，量比放大但不过热。",
            "卖出信号：跌破快线/近期低点，或 OBV 资金线转弱。",
        ]
    if strategy_type == "ml":
        return [
            "买入信号：ML Stacking 上涨概率上穿动态阈值，同时趋势过滤通过，且异常检测/蒙特卡洛风控不过热。",
            "卖出信号：ML Stacking 概率转弱、跌破短线均线，或趋势过滤失败；异常/风控风险升高时会降低买入优先级。",
        ]
    if strategy_type == "hybrid":
        return [
            "买入信号：SMA趋势、突破、RSI、MACD 四个模块中至少 3 个同时转强。",
            "卖出信号：投票分数降到 1 以下，或跌破短线均线/近期低点等风控线。",
        ]
    return ["买卖信号：由当前最佳策略的入场/离场条件决定。"]


def build_chart(data: pd.DataFrame, best: pd.Series, cash: float, fee: float, horizon: str) -> str:
    close = data["Close"]
    fast = int(best["fast"])
    slow = int(best["slow"])
    strategy_type = str(best.get("strategy_type", "sma"))
    fast_ma, slow_ma, entries, exits = strategy_signals(data, fast, slow, horizon, strategy_type)
    plot_fast = close.rolling(fast).mean() if strategy_type in {"ml", "hybrid"} else fast_ma
    plot_slow = close.rolling(slow).mean() if strategy_type in {"ml", "hybrid"} else slow_ma
    portfolio = strategy_portfolio(data, entries, exits, cash, fee, horizon)
    equity = portfolio.value()
    drawdown = (equity / equity.cummax() - 1) * 100
    indicator_title = {
        "macd": "MACD indicator with golden/death crosses",
        "rsi": "RSI indicator",
        "ml": "ML Stacking probability",
        "hybrid": "Hybrid vote score",
        "breakout": "Breakout distance",
        "sma": "SMA spread",
    }.get(strategy_type, "Strategy indicator")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.46, 0.18, 0.13, 0.23],
        subplot_titles=("Price and trade signals", indicator_title, "Volume", "Equity and drawdown"),
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=data.index, y=plot_fast, name=f"{STRATEGY_TYPES.get(strategy_type, strategy_type)} A{fast}", line=dict(width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=plot_slow, name=f"Filter {slow}", line=dict(width=1.6)), row=1, col=1)
    trades = portfolio.trades.records_readable
    entry_x = pd.to_datetime(trades["Entry Timestamp"]) if not trades.empty else pd.Series(dtype="datetime64[ns]")
    entry_y = trades["Avg Entry Price"] if not trades.empty else pd.Series(dtype=float)
    closed_trades = trades[trades["Status"].astype(str).eq("Closed")] if not trades.empty else trades
    exit_x = pd.to_datetime(closed_trades["Exit Timestamp"]) if not closed_trades.empty else pd.Series(dtype="datetime64[ns]")
    exit_y = closed_trades["Avg Exit Price"] if not closed_trades.empty else pd.Series(dtype=float)
    fig.add_trace(
        go.Scatter(
            x=entry_x,
            y=entry_y,
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#0f8f61", line=dict(color="#064e3b", width=1)),
            name="实际买入",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=exit_x,
            y=exit_y,
            mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#bf2f2f", line=dict(color="#7f1d1d", width=1)),
            name="实际卖出",
        ),
        row=1,
        col=1,
    )
    if strategy_type == "macd":
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        gold_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        dead_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        fig.add_trace(go.Scatter(x=data.index, y=macd_line, name="MACD", line=dict(color="#2563eb", width=1.6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=signal_line, name="Signal", line=dict(color="#f97316", width=1.4)), row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=macd_line - signal_line, name="MACD hist", marker_color="#94a3b8", opacity=0.45), row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=data.index[gold_cross.fillna(False)],
                y=macd_line[gold_cross.fillna(False)],
                mode="markers",
                marker=dict(symbol="circle-open", size=10, color="#0f8f61", line=dict(color="#0f8f61", width=2)),
                name="MACD golden cross",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index[dead_cross.fillna(False)],
                y=macd_line[dead_cross.fillna(False)],
                mode="markers",
                marker=dict(symbol="diamond-open", size=10, color="#bf2f2f", line=dict(color="#bf2f2f", width=2)),
                name="MACD death cross",
            ),
            row=2,
            col=1,
        )
    elif strategy_type == "rsi":
        rsi_line = rsi(close, max(2, fast))
        fig.add_trace(go.Scatter(x=data.index, y=rsi_line, name=f"RSI {fast}", line=dict(color="#2563eb", width=1.6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(35, index=data.index), name="RSI buy zone", line=dict(color="#16a34a", dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(72, index=data.index), name="RSI hot zone", line=dict(color="#dc2626", dash="dot")), row=2, col=1)
    elif strategy_type == "ml":
        threshold = fast_ma.rolling(160, min_periods=60).quantile(0.66).clip(lower=0.45, upper=0.58)
        exit_threshold = fast_ma.rolling(160, min_periods=60).quantile(0.32).clip(lower=0.36, upper=0.48)
        fig.add_trace(go.Scatter(x=data.index, y=fast_ma, name="ML Stacking up probability", line=dict(color="#2563eb", width=1.6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=threshold, name="ML buy threshold", line=dict(color="#16a34a", dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=exit_threshold, name="ML sell threshold", line=dict(color="#dc2626", dash="dot")), row=2, col=1)
    elif strategy_type == "hybrid":
        rsi_fast = 6 if fast <= 8 else 9
        macd_fast = 8 if fast <= 8 else 12
        macd_slow = max(21, slow)
        sma_bull = (close.rolling(fast).mean() > close.rolling(slow).mean()) & (close > close.rolling(slow).mean())
        breakout_bull = close > data["High"].shift(1).rolling(max(5, fast)).max()
        rsi_bull = (rsi(close, rsi_fast) > 42) & (rsi(close, rsi_fast) < 72)
        macd_line = close.ewm(span=macd_fast, adjust=False).mean() - close.ewm(span=macd_slow, adjust=False).mean()
        macd_bull = macd_line > macd_line.ewm(span=9, adjust=False).mean()
        vote_score = (sma_bull.astype(int) + breakout_bull.astype(int) + rsi_bull.astype(int) + macd_bull.astype(int)).astype(float)
        fig.add_trace(go.Scatter(x=data.index, y=vote_score, name="Vote score", line=dict(color="#2563eb", width=1.8)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(3.0, index=data.index), name="Buy threshold", line=dict(color="#16a34a", dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(1.0, index=data.index), name="Exit threshold", line=dict(color="#dc2626", dash="dot")), row=2, col=1)
    elif strategy_type == "breakout":
        prior_high = data["High"].shift(1).rolling(max(5, fast)).max()
        prior_low = data["Low"].shift(1).rolling(max(5, fast)).min()
        fig.add_trace(go.Scatter(x=data.index, y=close / prior_high - 1, name="Distance to breakout", line=dict(color="#2563eb", width=1.6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=close / prior_low - 1, name="Distance to stop", line=dict(color="#f97316", width=1.3)), row=2, col=1)
    else:
        spread = fast_ma - slow_ma
        zero = pd.Series(0, index=data.index)
        fig.add_trace(go.Scatter(x=data.index, y=spread, name="Fast - slow MA", line=dict(color="#2563eb", width=1.6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=zero, name="Cross line", line=dict(color="#64748b", dash="dot")), row=2, col=1)

    colors = np.where(data["Close"] >= data["Open"], "#dc2626", "#16a34a")
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], marker_color=colors, name="成交量"), row=3, col=1)
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="资金曲线", line=dict(color="#1464f4", width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="回撤%", line=dict(color="#f97316", width=1.5)), row=4, col=1)
    fig.update_layout(
        height=900,
        template="plotly_white",
        margin=dict(l=45, r=20, t=70, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})


def last_signal_date(entries: pd.Series, exits: pd.Series) -> tuple[str, str]:
    entry_dates = list(entries[entries].index)
    exit_dates = list(exits[exits].index)
    candidates = []
    if entry_dates:
        candidates.append(("买入", entry_dates[-1]))
    if exit_dates:
        candidates.append(("卖出", exit_dates[-1]))
    if not candidates:
        return "无", "-"
    side, date = max(candidates, key=lambda item: item[1])
    return side, pd.Timestamp(date).strftime("%Y-%m-%d")


def generate_advice(
    symbol: str,
    data: pd.DataFrame,
    best: pd.Series,
    shares: int,
    buy_price: float | None,
    buy_date: str,
    risk: str,
    horizon: str,
) -> dict[str, object]:
    close = data["Close"]
    latest_date = data.index[-1]
    latest_close = float(close.iloc[-1])
    fast = int(best["fast"])
    slow = int(best["slow"])
    strategy_type = str(best.get("strategy_type", "sma"))
    strategy_label = STRATEGY_TYPES.get(strategy_type, strategy_type)
    fast_ma, slow_ma, entries, exits = strategy_signals(data, fast, slow, horizon, strategy_type)
    latest_fast = float(fast_ma.iloc[-1])
    latest_slow = float(slow_ma.iloc[-1])
    entry_today = bool(entries.iloc[-1])
    exit_today = bool(exits.iloc[-1])
    in_trend = strategy_in_trend(strategy_type, latest_fast, latest_slow, latest_close)
    last_side, last_date = last_signal_date(entries, exits)
    strategy_active = False
    for entry_flag, exit_flag in zip(entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)):
        if entry_flag:
            strategy_active = True
        if exit_flag:
            strategy_active = False

    if horizon == "short":
        risk_factor = {"tight": 0.8, "normal": 1.0, "loose": 1.3}[risk]
    else:
        risk_factor = {"tight": 1.2, "normal": 1.6, "loose": 2.2}[risk]
    lookback = STRATEGY_GRIDS[horizon]["lookback"]
    atr_value = float(atr(data).iloc[-1])
    recent_low = float(data["Low"].tail(lookback).min())
    recent_high = float(data["High"].tail(lookback).max())
    trend_stop = max(latest_slow, latest_close - risk_factor * atr_value)
    structure_stop = recent_low
    stop_line = min(trend_stop, latest_close * (0.992 if horizon == "short" else 0.985)) if in_trend else max(latest_slow, latest_close * 1.01)
    if strategy_type in {"hybrid", "ml"}:
        rebound_line = recent_high
        protect_line = stop_line if strategy_type == "hybrid" else latest_slow
        entry_watch_line = recent_high
    else:
        rebound_line = max(latest_slow, latest_fast)
        protect_line = latest_fast
        entry_watch_line = latest_fast
    horizon_name = STRATEGY_GRIDS[horizon]["name"]
    if strategy_type == "hybrid":
        indicator_line = f"混合策略价格快线 {money(latest_fast)}，过滤线 {money(latest_slow)}，当前为{'共振偏强' if in_trend else '共振不足'}。"
    elif strategy_type == "ml":
        indicator_line = f"ML上涨概率 {latest_fast:.2f}，价格过滤线 {money(latest_slow)}，当前为{'ML确认区' if in_trend else 'ML未确认'}。"
    else:
        indicator_line = f"短均线 {money(latest_fast)}，长均线 {money(latest_slow)}，当前为{'场内趋势' if in_trend else '场外/弱势'}。"
    quote = get_intraday_quote(symbol)
    live_price = float(quote["price"]) if quote else latest_close
    live_source = f"{quote['source']} {quote['time']}" if quote else "最近收盘价"
    live_high = float(quote["high"]) if quote else float(data["High"].iloc[-1])
    live_low = float(quote["low"]) if quote else float(data["Low"].iloc[-1])

    sell_reasons: list[str] = []
    if exit_today:
        sell_reasons.append("最新交易日触发策略卖出信号")
    if live_price <= stop_line:
        sell_reasons.append(f"参考价 {money(live_price)} 跌破/压在防守线 {money(stop_line)} 下方")
    if live_price <= structure_stop:
        sell_reasons.append(f"参考价 {money(live_price)} 跌破近期结构低点 {money(structure_stop)}")
    if not in_trend:
        sell_reasons.append("趋势过滤转弱，策略不支持继续追买")
    if not strategy_active and last_side == "卖出" and not entry_today:
        sell_reasons.append(f"最近一次策略信号仍是卖出（{last_date}），之后尚未出现新的买入信号")

    buy_reasons: list[str] = []
    if entry_today:
        buy_reasons.append("最新交易日触发策略买入信号")
    if live_price >= recent_high and in_trend:
        buy_reasons.append(f"参考价接近/突破近期高点 {money(recent_high)}，且趋势过滤通过")

    signal_state = "持有/观察"
    signal_class = "warn"
    if sell_reasons and (shares > 0 or exit_today or not strategy_active):
        signal_state = "卖出/减仓提醒"
        signal_class = "bad"
    elif entry_today or (not shares and live_price >= recent_high and in_trend and strategy_active):
        signal_state = "买入观察提醒"
        signal_class = "good"
    elif shares > 0 and in_trend:
        signal_state = "继续持有"
        signal_class = "good"
    elif not strategy_active:
        signal_state = "空仓等待新买点"

    sell_status_line = "当前卖出信号：未触发，继续按防守线跟踪。"
    if sell_reasons:
        sell_status_line = "当前卖出信号：已触发/仍有效，原因：" + "；".join(dict.fromkeys(sell_reasons)) + "。"
    buy_status_line = "当前买入信号：未触发新的策略买点。"
    if buy_reasons and not sell_reasons:
        buy_status_line = "当前买入信号：可观察，原因：" + "；".join(dict.fromkeys(buy_reasons)) + "。"
    elif buy_reasons and sell_reasons:
        buy_status_line = "当前买入信号：有转强迹象，但卖出/风控条件仍优先，暂不追买。"

    signal_rule_lines = strategy_signal_rules(strategy_type)
    buy_rule_lines = [line for line in signal_rule_lines if line.startswith("买入信号")]
    sell_rule_lines = [line for line in signal_rule_lines if line.startswith("卖出信号")]
    if not buy_rule_lines:
        buy_rule_lines = ["买入信号：由当前策略的入场条件决定。"]
    if not sell_rule_lines:
        sell_rule_lines = ["卖出信号：由当前策略的离场条件和风控线决定。"]
    buy_signal_lines = [
        *buy_rule_lines,
        "当前买入是否触发：" + ("是，" + "；".join(dict.fromkeys(buy_reasons)) if buy_reasons and not sell_reasons else "否，未出现新的策略买点或卖出/风控优先"),
        f"买入观察价：重新站上 {money(rebound_line)}，并放量突破近期高点 {money(recent_high)}，才考虑试仓。",
        f"买入后的失效条件：买入后跌回 {money(stop_line)} 或趋势过滤失败，则放弃/退出。",
    ]
    sell_signal_lines = [
        *sell_rule_lines,
        "当前卖出是否触发：" + ("是，" + "；".join(dict.fromkeys(sell_reasons)) if sell_reasons else "否，尚未触发强制卖出条件"),
        f"卖出观察价：跌破防守线 {money(stop_line)} 先减仓；跌破结构低点 {money(structure_stop)} 按破位处理。",
        f"反抽卖出位：反弹到 {money(rebound_line)} 附近但站不回，或站回后很快跌破，继续减仓/退出。",
    ]

    action_lines: list[str] = []
    signal_lines: list[str] = [
        f"最新交易日：{latest_date:%Y-%m-%d}，收盘价 {money(latest_close)}。",
        f"盘中检查：{live_source}，参考价 {money(live_price)}，盘中高低 {money(live_high)} / {money(live_low)}。",
        f"当前操作信号：{signal_state}。",
        f"策略当前状态：{'持仓段' if strategy_active else '空仓/离场段'}。",
        sell_status_line,
        buy_status_line,
        f"自动选择策略：{horizon_name} {strategy_label} {fast}/{slow}，评分 {best['score']:.2f}，历史交易次数 {int(best['trades'])}。",
        indicator_line,
        f"最近一次策略信号：{last_side}，日期 {last_date}。",
    ]
    signal_lines.extend(signal_rule_lines)

    reminder_lines = [
        f"卖出/减仓触发：盘中或收盘跌破 {money(stop_line)}，优先减仓；跌破近期结构低点 {money(structure_stop)}，直接按风控处理。",
        f"买入/加仓触发：空仓时，只有重新触发策略买入信号，或放量突破近期高点 {money(recent_high)} 后，才考虑试仓。",
        f"止盈提醒：冲高后回落跌回 {money(latest_fast)}，说明短线动能减弱，优先保护利润。",
        f"每日复盘：每个交易日收盘后重新点一次分析，看是否出现新买入/卖出信号、金叉/死叉或趋势破位。",
    ]
    first_sell_ratio = {"tight": "70%-100%", "normal": "50%-70%", "loose": "30%-50%"}.get(risk, "50%-70%")
    second_sell_ratio = {"tight": "剩余全部", "normal": "剩余 50%-100%", "loose": "再减 30%-50%"}.get(risk, "剩余 50%-100%")
    sell_plan_lines = [
        f"触发卖：若当前卖出信号已触发，或收盘/盘中跌破防守线 {money(stop_line)}，下一可交易时段先卖出 {first_sell_ratio}；A股 T+1 下，盘中提醒用于提前做委托和风控准备。",
        f"反抽卖：若价格反弹到 {money(rebound_line)} 附近但收盘站不回，或站回后很快跌破，卖出{second_sell_ratio}，不等再次破位。",
        f"破位卖：若跌破近期结构低点 {money(structure_stop)}，说明近 {lookback} 日结构被破坏，按风控退出剩余仓位。",
        f"止盈卖：若冲高接近/突破近期高点 {money(recent_high)} 后回落，并跌回短线保护线 {money(protect_line)}，优先锁定利润，至少减半。",
        f"重新买回条件：卖出后不急着接回，等重新触发策略买入信号，或收盘重新站上 {money(rebound_line)} 且放量突破 {money(recent_high)}。",
    ]
    if sell_reasons:
        sell_plan_lines.insert(0, "当前执行优先级：卖出/减仓优先，因为 " + "；".join(dict.fromkeys(sell_reasons[:3])) + "。")
    else:
        sell_plan_lines.insert(0, "当前未触发强制卖出，以下是预案；只要触发其中任一条件，就按纪律执行。")
    if quote:
        if live_price <= stop_line or live_price <= structure_stop:
            reminder_lines.insert(0, f"现在已经触发风险线：当前 {money(live_price)} <= 防守线 {money(stop_line)}，不要等回测曲线，先按卖出/减仓预案处理。")
        elif live_price >= recent_high and in_trend:
            reminder_lines.insert(0, f"现在接近/突破强势线：当前 {money(live_price)}，近期高点 {money(recent_high)}；若量能配合，可按短线试仓/持有规则跟踪。")
        else:
            reminder_lines.insert(0, f"现在未触发关键线：当前 {money(live_price)}，继续盯 {money(stop_line)} 和 {money(recent_high)} 两条线。")

    if shares > 0:
        cost = buy_price or latest_close
        pnl = (latest_close - cost) * shares
        pnl_pct = (latest_close / cost - 1) * 100 if cost else 0
        action_lines.append(f"你当前按 {shares} 股、成本 {money(cost)} 估算，浮动盈亏 {money(pnl)}，收益率 {pct(pnl_pct)}。")
        if horizon == "short":
            action_lines.append(f"短线先按 {lookback} 日节奏处理，不做长期恋战；盘中/收盘跌破 {money(stop_line)} 就要提高卖出优先级。")
        if sell_reasons:
            action_lines.append(f"卖出信号已经触发或仍有效：{'；'.join(dict.fromkeys(sell_reasons[:3]))}。")
            action_lines.append(f"下一交易日若不能重新站上 {money(rebound_line)}，以减仓或卖出为主；若盘中跌破 {money(stop_line)}，不要等收盘。")
            action_lines.append(f"若盘中跌破 {money(structure_stop)}，说明近 {lookback} 日结构也被破坏，避免继续硬扛。")
        else:
            action_lines.append(f"趋势仍在，持有为主；收盘跌破 {money(stop_line)} 后，下一交易日不能收回就减仓或卖出。")
            action_lines.append(f"若放量突破近 {lookback} 日高点 {money(recent_high)}，可以继续跟踪；冲高后回落跌回 {money(protect_line)}，优先保护利润。")
        if buy_price and latest_close <= buy_price * 0.92:
            action_lines.append(f"价格已接近或低于成本价 8% 防线 {money(buy_price * 0.92)}，先控制亏损。")
        position_mode = "已有持仓"
    else:
        position_mode = "当前空仓"
        if sell_reasons and not entry_today:
            action_lines.append(f"当前是卖出/离场状态，不是买点：{'；'.join(dict.fromkeys(sell_reasons[:3]))}。")
            action_lines.append(f"空仓先继续等，不要因为短线反弹直接追；只有重新触发策略买入信号，或收盘重新站上 {money(rebound_line)} 且放量突破 {money(recent_high)}，才考虑试仓。")
            action_lines.append(f"如果你实际还有持仓，按卖出信号处理：反抽不能站回 {money(rebound_line)} 就减仓，跌破 {money(stop_line)} 或 {money(structure_stop)} 优先离场。")
        elif entry_today:
            action_lines.append(f"最新交易日触发买入信号；下一交易日可考虑在 {money(latest_close)} 附近或回踩/突破 {money(entry_watch_line)} 后分批试仓。")
            action_lines.append(f"首次仓位控制在计划资金的 {'20%-30%' if horizon == 'short' else '30%-50%'}；收盘跌破 {money(stop_line)} 则退出。")
        elif strategy_active and in_trend:
            action_lines.append(f"策略处在持仓段，但不是首个买点；不追高，等待回踩/突破 {money(entry_watch_line)} 附近企稳或再次突破 {money(recent_high)}。")
            action_lines.append(f"买入后防守线看 {money(stop_line)}；没有放量或跌回长均线下方就放弃。")
        else:
            action_lines.append(f"当前策略没有买入信号；继续空仓等待短均线上穿长均线，或收盘重新站上 {money(rebound_line)}。")
            action_lines.append(f"若之后出现买入信号，止损线先参考 {money(structure_stop)}，再按实际成交价调整。")

    metrics = [
        {"k": "参考价", "v": money(live_price), "cls": signal_class},
        {"k": "最佳策略", "v": f"{strategy_label} {fast}/{slow}", "cls": ""},
        {"k": "策略收益", "v": pct(float(best["total_return_pct"])), "cls": "good" if best["total_return_pct"] >= 0 else "bad"},
        {"k": "最大回撤", "v": pct(float(best["max_drawdown_pct"])), "cls": "bad"},
    ]
    ml_risk: dict[str, object] | None = None
    if strategy_type == "ml":
        ml_risk = build_ml_risk_snapshot(data, fast, slow, stop_line)
        anomaly = ml_risk.get("anomaly", {}) if isinstance(ml_risk.get("anomaly"), dict) else {}
        mc = ml_risk.get("monte_carlo", {}) if isinstance(ml_risk.get("monte_carlo"), dict) else {}
        signal_lines.append(f"ML异常检测：{anomaly.get('level', '-')}，{anomaly.get('detail', '-')}")
        signal_lines.append(
            f"蒙特卡洛10日：上涨概率 {float(mc.get('up_prob', 0)) * 100:.1f}%，VaR95 {mc.get('var_95_pct', '-')}%，"
            f"价格区间 {mc.get('low_price', '-')}/{mc.get('mid_price', '-')}/{mc.get('high_price', '-')}"
        )
        if mc.get("stop_break_prob") is not None:
            reminder_lines.insert(0, f"ML风控：未来10日模拟跌破风控线概率 {float(mc.get('stop_break_prob', 0)) * 100:.1f}%，异常等级 {anomaly.get('level', '-')}")

    return {
        "position_mode": position_mode,
        "metrics": metrics,
        "action_lines": action_lines,
        "sell_plan_lines": sell_plan_lines,
        "buy_signal_lines": buy_signal_lines,
        "sell_signal_lines": sell_signal_lines,
        "signal_lines": signal_lines,
        "reminder_lines": reminder_lines,
        "daily_signal": {
            "date": latest_date.strftime("%Y-%m-%d"),
            "strategy_type": strategy_type,
            "strategy_label": strategy_label,
            "fast": fast,
            "slow": slow,
            "entry_today": entry_today,
            "exit_today": exit_today,
            "in_trend": in_trend,
            "strategy_active": strategy_active,
            "sell_reasons": sell_reasons,
            "buy_reasons": buy_reasons,
            "latest_close": latest_close,
            "stop_line": stop_line,
            "structure_stop": structure_stop,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "last_side": last_side,
            "last_date": last_date,
            "ml_risk": ml_risk,
        },
    }


def parse_float(value: str, default: float | None = None) -> float | None:
    value = (value or "").strip()
    if not value:
        return default
    return float(value)


def analyze(form: dict[str, str]) -> dict[str, object]:
    symbol = resolve_stock_identifier(form["symbol"])
    start = form["start"].strip()
    adjust = form.get("adjust", "qfq")
    cash = float(form.get("cash") or 100000)
    fee = float(form.get("fee") or 0.0003)
    shares = int(float(form.get("shares") or 0))
    buy_price = parse_float(form.get("buy_price", ""), None)
    buy_date = form.get("buy_date", "").strip()
    risk = form.get("risk", "normal")
    horizon = form.get("horizon", "short")
    if horizon not in STRATEGY_GRIDS:
        horizon = "short"
    strategy_filter = form.get("strategy_type", "auto")
    if strategy_filter not in STRATEGY_TYPES:
        strategy_filter = "auto"

    data = cached_data(symbol, start, adjust).copy()
    if len(data) < 240:
        raise ValueError("历史数据太少，至少需要约 240 个交易日。")
    scan = scan_strategies(data, cash, fee, horizon, strategy_filter)
    best = scan.iloc[0]
    advice = generate_advice(symbol, data, best, shares, buy_price, buy_date, risk, horizon)
    chart = build_chart(data, best, cash, fee, horizon)
    table = []
    for _source_idx, row in scan.head(12).iterrows():
        table.append(
            {
                "strategy_label": row.get("strategy_label", STRATEGY_TYPES.get(row.get("strategy_type", "sma"), "SMA")),
                "fast": int(row["fast"]),
                "slow": int(row["slow"]),
                "total_return_pct": f"{row['total_return_pct']:.2f}",
                "max_drawdown_pct": f"{row['max_drawdown_pct']:.2f}",
                "sharpe": f"{row['sharpe']:.2f}" if not math.isnan(row["sharpe"]) else "-",
                "trades": int(row["trades"]),
                "final_value": money(row["final_value"]),
                "score": f"{row['score']:.2f}",
            }
        )
    advice["chart"] = chart
    advice["table"] = table
    advice["symbol"] = symbol
    advice["name"] = stock_display_name(symbol)
    advice["strategy_label"] = STRATEGY_TYPES.get(str(best.get("strategy_type", "sma")), "SMA")
    advice["best_params"] = f"{int(best['fast'])}/{int(best['slow'])}"
    return advice


def parse_symbols(form: dict[str, str]) -> list[str]:
    batch_text = form.get("batch_symbols", "").strip()
    text = batch_text if batch_text else form.get("symbol", "")
    tokens = [token.strip() for token in re.split(r"[\s,，;；、]+", text) if token.strip()]
    symbols: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        symbol = resolve_stock_identifier(token)
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def store_result(symbol: str, form: dict[str, str], result: dict[str, object]) -> None:
    symbol = normalize_symbol(symbol)
    saved_form = form.copy()
    saved_form["symbol"] = symbol
    saved_form["batch_symbols"] = ""
    FORM_CACHE[symbol] = saved_form
    RESULT_CACHE[symbol] = result


def cache_sidebar() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for symbol, result in RESULT_CACHE.items():
        signal_lines = result.get("signal_lines", [])
        signal = signal_lines[0] if signal_lines else result.get("position_mode", "")
        items.append(
            {
                "symbol": symbol,
                "strategy_label": f"{result.get('name') or stock_display_name(symbol)} {result.get('strategy_label', '')} {result.get('best_params', '')}".strip(),
                "signal": str(signal)[:70],
            }
        )
    return items


def render_page(
    form: dict[str, str],
    result: dict[str, object] | None = None,
    error: str | None = None,
    batch_status: str | None = None,
):
    try:
        selected_symbol = resolve_stock_identifier(form.get("symbol", "")) if form.get("symbol") else ""
    except Exception:
        selected_symbol = ""
    return render_template_string(
        PAGE,
        form=form,
        result=result,
        error=error,
        batch_status=batch_status,
        cached_symbols=cache_sidebar(),
        selected_symbol=selected_symbol,
    )


@app.route("/", methods=["GET"])
def index():
    return render_page(default_form())


@app.route("/monitor", methods=["GET"])
def monitor_page():
    symbols = request.args.get("symbols", "")
    if not symbols:
        symbols = ", ".join(RESULT_CACHE.keys()) or default_form()["symbol"]
    return render_template_string(
        MONITOR_PAGE,
        symbols=symbols,
        period=request.args.get("period", "5"),
        interval=request.args.get("interval", "30"),
        shares=request.args.get("shares", ""),
        buy_price=request.args.get("buy_price", ""),
    )


@app.route("/api/monitor", methods=["GET"])
def monitor_api():
    symbols_text = request.args.get("symbols", "")
    if not symbols_text:
        symbols_text = ", ".join(RESULT_CACHE.keys()) or default_form()["symbol"]
    try:
        symbols = parse_symbol_text(symbols_text)
    except Exception as exc:
        return jsonify(
            {
                "updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "items": [{"symbol": symbols_text, "xueqiu_url": "", "error": str(exc)}],
            }
        )
    period = request.args.get("period", "5")
    request_shares = request.args.get("shares", "")
    request_buy_price = request.args.get("buy_price", "")
    items = []
    for symbol in symbols:
        try:
            items.append(build_monitor_item(symbol, period, request_shares, request_buy_price))
        except Exception as exc:
            try:
                code = resolve_stock_identifier(symbol)
                url = xueqiu_url(code)
            except Exception:
                code = symbol
                url = ""
            items.append({"symbol": code, "xueqiu_url": url, "error": str(exc)})
    return jsonify(
        {
            "updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze_route():
    form = default_form()
    form.update({key: request.form.get(key, form.get(key, "")) for key in form})
    symbols = parse_symbols(form)
    try:
        if not symbols:
            raise ValueError("请至少输入一个股票代码")
        result = None
        errors: list[str] = []
        for symbol in symbols:
            item_form = form.copy()
            item_form["symbol"] = symbol
            item_form["batch_symbols"] = ""
            try:
                item_result = analyze(item_form)
                store_result(symbol, item_form, item_result)
                if result is None:
                    result = item_result
                    form.update(item_form)
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
        if result is None:
            raise ValueError("；".join(errors) if errors else "没有成功跑出结果")
        batch_status = f"已完成 {len(symbols) - len(errors)}/{len(symbols)} 只股票；左侧可直接切换查看。"
        if errors:
            batch_status += " 失败：" + "；".join(errors[:3])
        error = None
    except Exception as exc:
        result = None
        error = str(exc)
        batch_status = None
    return render_page(form, result=result, error=error, batch_status=batch_status)


@app.route("/view/<symbol>", methods=["GET"])
def view_cached(symbol: str):
    symbol = normalize_symbol(symbol)
    if symbol not in RESULT_CACHE:
        form = default_form()
        form["symbol"] = symbol
        return render_page(form, error=f"{symbol} 还没有缓存结果，请先分析一次。")
    return render_page(FORM_CACHE.get(symbol, default_form()), result=RESULT_CACHE[symbol])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8502, debug=False)
