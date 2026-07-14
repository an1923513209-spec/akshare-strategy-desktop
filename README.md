# A股策略桌面版

这是一个本地 A 股策略研究工具，核心功能包括传统策略回测、保存策略、盘中监控、ML 风控/组合评估。项目只用于研究和辅助决策，不会自动下单。

## 功能概览

- 传统回测：SMA、RSI、MACD、突破、KDJ/WR/资金类混合策略等。
- 策略保存：回测结果可保存，后续盘中监控会读取已保存策略。
- 盘中监控：点击已保存股票即可显示缓存曲线，点击“刷新一次”再拉取最新盘中曲线。
- 持仓记录：在盘中监控左下方填写持股数和成本价，保存后同步到股票表。
- ML 风控/组合：用于持仓风险、异常检测、蒙特卡洛风控、组合权重分配等辅助分析。
- 图表交互：回测图支持框选放大、滚轮缩放、右键平移、全屏查看。

## 安装

建议使用 Python 3.11。

```powershell
cd "D:\08 量化\akshare_strategy_deploy"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

检查环境：

```powershell
.\.venv\Scripts\python.exe .\check_install.py
```

## 启动桌面程序

推荐：

```powershell
.\start_desktop_app.ps1
```

也可以直接运行：

```powershell
.\.venv\Scripts\python.exe .\desktop_strategy_app.py
```

## 启动网页版本

```powershell
.\start_app.ps1
```

然后打开：

```text
http://127.0.0.1:8502
```

## 基本使用流程

1. 打开 `传统回测`。
2. 输入股票代码，例如 `002472`。
3. 设置开始日期、复权、资金、手续费、周期、策略。
4. 点击 `开始回测`。
5. 在排名表里查看策略表现，右键策略行可查看说明或保存策略。
6. 打开 `盘中监控`。
7. 左侧点击已保存股票，右侧会立即显示缓存的盘中监测信息。
8. 点击 `刷新一次` 拉取最新盘中数据。
9. 在左下方填写 `持股数`、`成本价`，点击 `保存持股/成本`。

## 盘中监控说明

- 点选股票只负责快速选中和展示缓存，不会自动联网刷新，避免卡顿。
- `刷新一次` 只刷新当前选中的股票。
- 右键股票行可以刷新这只股票。
- 如果窗口缩得较小，底部和右侧会出现滚动条，功能区保持原比例。

## 回测脚本

单个 SMA 回测：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_backtesting.py --symbol 002472 --start 20200101 --fast 10 --slow 30
```

vectorbt 参数扫描：

```powershell
.\.venv\Scripts\python.exe .\scripts\scan_vectorbt.py --symbol 002472 --start 20200101
```

Optuna 参数搜索：

```powershell
.\.venv\Scripts\python.exe .\scripts\optimize_optuna.py --symbol 002472 --start 20200101 --trials 50
```

生成交互式 HTML 图表：

```powershell
.\.venv\Scripts\python.exe .\scripts\visualize_strategy.py --symbol 002472 --start 20200101
```

## 目录说明

- `desktop_strategy_app.py`：桌面版主程序。
- `app.py`：网页版本和策略引擎。
- `scripts/`：命令行回测、参数扫描、可视化脚本。
- `cache/`：本地缓存，不建议提交。
- `data/`：AKShare 数据缓存，不建议提交。
- `reports/`：生成的报告，不建议提交。

## 风险提示

本项目不是投资建议，也不是自动交易系统。A 股数据源可能延迟、缺失或被调整；策略结果可能过拟合；盘中信号需要人工复核。实盘前请自行检查数据质量、手续费、滑点、涨跌停、T+1、交易单位、仓位控制和最大回撤。
