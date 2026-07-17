# A股策略桌面版

这是一个本地 A 股策略研究工具，核心功能包括传统策略回测、保存策略、盘中监控、ML 持仓次日操作决策。项目只用于研究和辅助决策，不会自动下单。

## 功能概览

- 传统回测：SMA、RSI、MACD、突破、KDJ/WR/资金类混合策略等。
- 策略保存：回测结果可保存，后续盘中监控会读取已保存策略。
- 盘中监控：点击已保存股票即可显示缓存曲线，点击“刷新一次”再拉取最新盘中曲线。
- 持仓记录：在盘中监控左下方填写持股数和成本价，保存后同步到股票表。
- ML 持仓决策：对你已保存/勾选的股票池和当前持仓做次日操作建议，输出清仓、减仓、持有、加仓、目标仓位和理由。
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

## ML 持仓次日决策

`ML持仓决策` 页面已改为持仓次日操作决策逻辑，不再把传统策略买卖点作为 ML 动作判断依据。

当前实现：

- 输入：左侧已保存股票池、勾选股票、持股数、成本价、总资金、目标总仓位。
- 训练数据：默认使用当前评估股票池的历史日线长表；接口保留全市场/同行业训练扩展位置。
- 预测目标：第 t 日收盘后，预测 t+1 开盘到 t+2 开盘的收益分布。
- 模型输出：跳空收益、开盘到再下一开盘收益、上涨概率、超过成本概率、下跌 2% 概率、q10/q50/q90 分位数。
- 动作集合：`SELL_ALL`、`REDUCE_50`、`REDUCE_25`、`HOLD`、`ADD_25`、`ADD_50`。
- 动作评分：统一扣除佣金、最低佣金、印花税、滑点、换手惩罚，并检查 A 股手数、可卖数量、单股仓位、停牌/一字涨跌停等约束。
- 输出：推荐动作、推荐买卖股数、目标仓位、期望净收益、下行风险、置信等级、正负向因子和文字理由。
- 外部数据：默认复用本地缓存；只有勾选 `本次刷新外部数据` 后，本轮评估才会重新拉取资金流、新闻和机构活跃度。刷新成功才覆盖旧缓存，刷新失败不会清空已有缓存。

免费外部因子来源：

- 资金流：AKShare `stock_individual_fund_flow`，东方财富个股资金流，近约 100 个交易日。
- 新闻：AKShare `stock_news_em`，东方财富个股最近新闻；因历史长度有限，目前只作为近期新闻因子。
- 机构活跃度：AKShare 龙虎榜机构席位 `stock_lhb_jgstatistic_em`，以及新浪机构持股 `stock_institute_hold`。
- 外部因子会优先读取 `cache/ml_external/` 本地缓存；评估结果右侧详情会显示 `资金流=缓存/今日拉取/拉取失败`、`新闻=...`、`机构活跃=...`，方便判断本次用了哪些数据。

### ML 卡顿处理

- ML 训练在子进程中运行，单只股票失败或超时会跳过，不影响其他股票继续评估。
- ML 结果表左侧固定股票列、右侧指标列同步滚动；选中联动已做防循环处理，避免左右表格互相触发导致窗口不响应。
- 调试日志位于 `cache/ui_debug.log`，超过 2MB 会自动截断，避免日志过大拖慢界面。

命令行示例：

```powershell
.\.venv\Scripts\python.exe .\daily_holding_decision.py --symbols "002472,603339" --cash 100000 --total-asset 100000
```

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
- `ml_decision/`：ML 持仓次日操作决策引擎。
- `daily_holding_decision.py`：ML 持仓决策命令行示例。
- `config/ml_decision_config.json`：ML 持仓决策默认配置。
- `tests/`：基础测试。
- `scripts/`：命令行回测、参数扫描、可视化脚本。
- `cache/`：本地缓存，不建议提交。
- `data/`：AKShare 数据缓存，不建议提交。
- `reports/`：生成的报告，不建议提交。

## 风险提示

本项目不是投资建议，也不是自动交易系统。A 股数据源可能延迟、缺失或被调整；策略结果可能过拟合；盘中信号需要人工复核。实盘前请自行检查数据质量、手续费、滑点、涨跌停、T+1、交易单位、仓位控制和最大回撤。

## 龙虎榜因子数据

ML 外部因子已接入 AKShare `stock_lhb_detail_em` 和 `stock_lhb_jgmmtj_em`。正常评估复用本地 Parquet，只有勾选“刷新外部数据”或运行更新脚本时才下载缺失日期，不会每天重拉全部历史。

首次全量或日常增量更新：

```powershell
.\.venv\Scripts\python.exe .\scripts\update_lhb_data.py --start 20200101
```

强制重拉指定日期：

```powershell
.\.venv\Scripts\python.exe .\scripts\update_lhb_data.py --start 20260715 --end 20260715 --force
```

主要文件：

- `data/raw/lhb/lhb_detail.parquet`：龙虎榜每日明细。
- `data/raw/lhb/lhb_institution.parquet`：机构席位买卖统计。
- `data/raw/lhb/lhb_availability.parquet`：逐日下载完整性，控制未上榜填 0 或保留缺失。
- `data/quality/lhb/lhb_quality_report.parquet`：每日数据质量报告。
- `config/lhb_factor_config.json`：上榜原因关键词、泄漏字段黑名单和下载参数。
- `logs/lhb_data.log`：下载重试、字段变化和完整性日志。

龙虎榜为收盘后事件，日期 `t` 的因子只用于预测下一交易日；模型交易不得使用当日收盘价成交。任何包含 `future/next/target/label/上榜后/未来/后续` 的字段都会被特征入口拒绝。
# ML 因子治理与生产模型

现有传统回测、盘中监控和 `ml_decision.engine.run_holding_decision` 保持可用。新增治理层不会删除、重命名或重算旧因子，主要入口在 `scripts/ml_governance.py`。

## 三种运行模式

准备一份长表 CSV 或 Parquet。表中可以是已有完整因子，也可以是包含 `date/code/open/high/low/close/volume/amount` 的行情长表。

```powershell
# 每日预测：只加载 production 模型，不训练、不选因子、不更新权重
.\.venv\Scripts\python.exe scripts\ml_governance.py daily-predict --data data\ml_panel.parquet

# 月度训练：36/2/1/1 月滚动 OOS，保存不可覆盖的 candidate 版本
.\.venv\Scripts\python.exe scripts\ml_governance.py monthly-train --data data\ml_panel.parquet --version 2026-07

# 季度审查：单因子、消融、组内置换和测试集 SHAP
.\.venv\Scripts\python.exe scripts\ml_governance.py quarterly-audit --data data\ml_panel.parquet
```

月度训练不会自动替换正式模型。版本保存在 `models/<version>/`，`models/registry.json` 分别记录 `candidate`、`production` 和 `previous_production`。只有连续至少 3 个样本外窗口更好、扣费后收益不恶化、最大回撤不恶化且回归测试通过时，候选版本才能晋升。

## 时间对齐

- 因子时点：`t` 日收盘后可得。
- 执行时点：`t+1` 日开盘。
- 标签：`open[t+2] / open[t+1] - 1`。
- 滚动窗口：训练 36 月、校准 2 月、验证 1 月、测试 1 月，每月前滚。
- 每个边界排除至少 2 个交易日 purge 和 2 个交易日 embargo。
- 特征覆盖率筛选和缺失值拟合只看训练期；概率校准只看校准期；测试期只评价。

## 因子组与门控

`ml_decision/factor_registry.py` 将所有既有候选字段映射到 `technical`、`liquidity`、`fund_flow`、`institution`、`news`、`lhb`、`lhb_institution`、`fundamental`、`market`、`industry` 或 `other_existing`。未识别字段仍被保留；目标和未来字段会直接拒绝入模。

动态权重只作用于因子组子模型，并且只使用生效日前已经完成的 OOS 窗口。确认无新闻时新闻模型权重归零；确认近期无龙虎榜时龙虎榜模型归零。`NaN` 表示数据缺失，不会被误判为“确认无事件”。

## 审计输出

治理任务在 `reports/` 生成：

- `factor_quality.csv`
- `factor_ic_history.csv`
- `factor_group_ablation.csv`
- `factor_group_permutation.csv`
- `factor_group_shap.csv`
- `model_oos_metrics.csv`
- `dynamic_group_weights.csv`
- `factor_status.csv`
- `model_comparison.md`

这些文件只有在对应月度训练或季度审查真实运行后才代表正式样本外结论。不要把单元测试或合成数据结果当成投资结论。

## 桌面 App 的生产模型流程

启动桌面程序：

```powershell
.\start_desktop_app.ps1
```

“评估勾选股票”和“评估全部股票池”现在走同一条生产推理链：批量更新数据、构建最新因子、懒加载一次正式模型、事件门控、组合约束、对排序前 10 只计算轻量级当前样本 SHAP，并保存 `cache/ml_prediction_snapshot.json`。该流程不会调用 `fit`、滚动回测、消融或动态权重训练。正式模型不存在或损坏时会明确提示先训练并晋升候选模型，不会临时训练替代品。

“模型诊断”页读取已有元数据和 `reports/` 文件，不在界面线程现场计算研究报告。页面可查看正式版本、训练截止日、标签、校准状态、样本外指标、因子组权重和离线消融结果。月度训练在独立子进程执行，可以终止；启动 App 时只读取模型元数据，首次评估时才反序列化模型。

桌面股票池月度训练：

```powershell
.\.venv\Scripts\python.exe .\scripts\monthly_train_desktop.py --version 2026-07
```

查看版本指针和回退上一正式模型：

```powershell
.\.venv\Scripts\python.exe .\scripts\ml_governance.py status
.\.venv\Scripts\python.exe .\scripts\ml_governance.py rollback
```

模型版本以不可变目录保存在 `models/<version>/`，`models/registry.json` 原子记录 `candidate`、`production` 和 `previous_production`。晋升前必须通过样本外比较与回归测试；回退只切换指针，不覆盖历史模型文件。

当前正式标签是 `t+1` 开盘到 `t+2` 开盘的收益方向与收益率。界面的“次日上涨概率/次日预期收益”来自独立次日模型；“3日风险投影”和“10日风险投影”是次日模型的风险折算，不是独立训练的 3 日或 10 日模型。
