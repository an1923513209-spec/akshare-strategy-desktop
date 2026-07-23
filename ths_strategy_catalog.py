"""Readable Tonghuashun-style strategy cards and exportable formulas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class THSStrategyCard:
    strategy_type: str
    name: str
    tagline: str
    buy_condition: str
    sell_condition: str
    interpretation: tuple[str, ...]
    optimization: str
    compatibility_note: str
    selection_formula: str
    trading_formula: str
    backtest_formula: str
    overlay_formula: str


_CARDS: dict[str, dict[str, object]] = {
    "ths_oversold_reversal": {
        "name": "超跌反转共振",
        "tagline": "超跌时等反转，不在下跌途中硬接飞刀",
        "buy": "RSI6 上穿 RSI12，或 KDJ 的 J 值低于 10 后 K 线上穿 D。",
        "sell": "MACD 出现近 20 日顶部背离，或获利盘超过 90%，或跌破 20 日均线。",
        "explain": (
            "RSI 金叉：短周期动能重新强于中周期动能，表示超跌后开始修复。",
            "KDJ 极端超卖金叉：J 值低于 10 后再金叉，避免只因低位而过早抄底。",
            "MACD 顶背离：价格仍在阶段高位但 DIF 未创新高，提示上涨动能衰减。",
            "获利盘超过 90%：大量筹码处于盈利状态，回吐压力通常会上升。",
        ),
        "optimization": "可再加入 20 日均线向上、放量反转或大盘趋势过滤，减少弱势行情里的假反弹。",
    },
    "ths_macd_kdj_rsi": {
        "name": "三指标趋势共振",
        "tagline": "MACD 定方向，KDJ 找节奏，RSI 过滤追高",
        "buy": "MACD 金叉、KDJ 多头、RSI 位于 45 至 72，且收盘价站上 20 日均线。",
        "sell": "MACD 死叉，或 KDJ 高位转弱，或 RSI 超过 80，或跌破 20 日均线。",
        "explain": ("MACD 金叉确认中短期动量转强。", "KDJ 多头确认短线节奏同步。", "RSI 不高于 72 用来避开明显过热区。"),
        "optimization": "加入成交量位于 10 日均量的 0.8 至 3.5 倍过滤，避免无量突破和极端爆量。",
    },
    "ths_boll_squeeze": {
        "name": "布林蓄势突破",
        "tagline": "先缩口蓄势，再放量突破",
        "buy": "布林带处于近 60 日窄幅区，收盘突破上轨和前期高点，成交量放大。",
        "sell": "收盘跌回布林中轨，或布林带由极窄快速扩张后转弱。",
        "explain": ("缩口代表波动率压缩。", "突破上轨和前高确认价格离开整理区。", "放量用于验证突破有效性。"),
        "optimization": "RSI 高于 76 时不追，优先选择中轨向上的布林结构。",
    },
    "ths_boll_rsi_break": {
        "name": "布林动量突破",
        "tagline": "突破上轨但不过热",
        "buy": "收盘突破布林上轨、RSI 位于 50 至 76，且成交量高于 20 日均量。",
        "sell": "跌破布林中轨、RSI 超过 82，或跌破布林下轨。",
        "explain": ("上轨突破确认强势。", "RSI 区间限制避免动能不足或严重过热。", "均量过滤剔除无量假突破。"),
        "optimization": "与平台突破或资金净流入组合时更稳健。",
    },
    "ths_wr_rebound": {
        "name": "WR 超卖反弹",
        "tagline": "从超卖区出来才确认反弹",
        "buy": "WR 从 -80 下方向上穿越 -80，价格未明显跌离长期均线。",
        "sell": "WR 进入 -18 以上超买区，或价格继续跌破趋势保护线。",
        "explain": ("WR 低于 -80 表示短线超卖。", "向上穿回 -80 比单纯处于超卖区更有确认意义。"),
        "optimization": "叠加阳线反包或量能恢复，可减少连续下跌中的错误信号。",
    },
    "ths_cci_breakout": {
        "name": "CCI 强势突破",
        "tagline": "动量越过强势门槛后跟随",
        "buy": "CCI 上穿 100、价格在趋势均线上方，且成交量不低于均量。",
        "sell": "CCI 下穿 0，或价格跌破趋势均线。",
        "explain": ("CCI 上穿 100 表示价格进入强势区。", "趋势和成交量过滤降低震荡期假信号。"),
        "optimization": "强趋势股票可将退出门槛放宽到 CCI 下穿 -50。",
    },
    "ths_bias_revert": {
        "name": "乖离修复",
        "tagline": "负乖离收窄时参与均值回归",
        "buy": "BIAS 从 -6 下方向上穿回 -6，且价格没有远离长期均线超过 8%。",
        "sell": "BIAS 高于 8，或价格跌破长期均线的 94%。",
        "explain": ("大幅负乖离说明短期偏离均值。", "乖离开始收窄才确认修复，而不是见跌就买。"),
        "optimization": "只在大盘非单边下跌、个股基本趋势未破坏时使用。",
    },
    "ths_mtm_accel": {
        "name": "动量加速",
        "tagline": "正动量再次加速时跟随",
        "buy": "MTM 上穿自身均线、MTM 大于 0，且价格站上趋势均线。",
        "sell": "MTM 下穿自身均线，或价格跌破趋势均线。",
        "explain": ("MTM 大于 0 表示当前价格高于前期。", "上穿平滑线表示上涨速度重新加快。"),
        "optimization": "叠加量比大于 0.9，过滤流动性不足的信号。",
    },
    "ths_kdj_oversold": {
        "name": "KDJ 超卖金叉",
        "tagline": "低位金叉做反弹，高位死叉及时走",
        "buy": "K 上穿 D 且 J 低于 35，价格没有明显跌破趋势均线。",
        "sell": "KDJ 高位死叉，或价格跌破趋势保护线。",
        "explain": ("低位金叉比普通金叉更强调赔率。", "趋势保护避免在主跌段反复抄底。"),
        "optimization": "J 低于 10 的信号更少但更极端，可与 RSI 金叉做二选一。",
    },
    "ths_volume_breakout": {
        "name": "放量新高突破",
        "tagline": "价格过前高，量能必须跟上",
        "buy": "收盘突破前期高点、短均线在长均线上方，量比位于 1.35 至 5。",
        "sell": "跌破短均线，或跌破近期结构低点。",
        "explain": ("突破前高代表供给区被消化。", "适度放量确认突破，极端爆量则可能是分歧和派发。"),
        "optimization": "排除一字板和无法正常成交的极端行情。",
    },
    "ths_shrink_pullback": {
        "name": "缩量回踩再起",
        "tagline": "上涨趋势中缩量回踩，重新站回短均线",
        "buy": "长趋势向上，前一日缩量回踩，当前收盘重新站上短均线。",
        "sell": "跌破长均线，或跌破近期结构低点。",
        "explain": ("缩量回踩表示主动抛压有限。", "重新站回短均线确认回踩结束。"),
        "optimization": "回踩期间不应出现连续放量阴线。",
    },
    "ths_amount_breakout": {
        "name": "成交额突破",
        "tagline": "价格与成交额同步越过平台",
        "buy": "收盘突破前期高点、站上趋势均线，成交额达到 20 日均值的 1.4 至 6 倍。",
        "sell": "跌破趋势均线，或成交额萎缩至均值的 55% 以下。",
        "explain": ("成交额比单纯成交量更能反映资金承接规模。", "价格和金额共振可过滤部分小量拉升。"),
        "optimization": "结合自由流通市值对成交额进行标准化会更公平。",
    },
    "ths_platform_breakout": {
        "name": "平台蓄势突破",
        "tagline": "窄幅平台整理后放量向上",
        "buy": "整理区宽度小于约 22%，收盘突破平台上沿，成交量放大。",
        "sell": "跌破短均线，或跌回平台下沿。",
        "explain": ("窄幅平台代表多空成本趋于集中。", "放量突破说明新的买盘愿意抬高成本。"),
        "optimization": "平台至少维持 20 个交易日，突破当天避免过度高开。",
    },
    "ths_new_high": {
        "name": "阶段新高趋势",
        "tagline": "强者恒强，但用均线保护利润",
        "buy": "收盘创阶段新高，短均线高于长均线，且成交量不低于均量。",
        "sell": "跌破短均线，或短均线下穿长均线。",
        "explain": ("阶段新高说明上方历史套牢盘较少。", "均线多头过滤无趋势的新高。"),
        "optimization": "可用 ATR 控制追高距离，并限制单次仓位。",
    },
    "ths_trendline_break": {
        "name": "趋势线转强",
        "tagline": "价格上穿上升中的动态趋势线",
        "buy": "收盘上穿动态趋势线，趋势线本身向上，量能正常。",
        "sell": "收盘下穿趋势线，或趋势线转为向下。",
        "explain": ("动态趋势线平滑短期噪声。", "要求趋势线向上可过滤横盘反复交叉。"),
        "optimization": "震荡行情可提高趋势斜率门槛。",
    },
    "ths_obv_mfi": {
        "name": "量能资金共振",
        "tagline": "OBV 方向与资金流强度同步",
        "buy": "OBV 上穿均线、MFI 高于 50，且价格站上趋势均线。",
        "sell": "OBV 下穿均线、MFI 低于 42，或价格跌破趋势均线。",
        "explain": ("OBV 观察量价累积方向。", "MFI 同时使用价格和成交量衡量资金强弱。"),
        "optimization": "若有 Level-2 数据，优先用真实大单净量替代 OBV 代理。",
    },
    "ths_capital_breakout": {
        "name": "主力资金突破",
        "tagline": "资金转强与价格突破同时发生",
        "buy": "主力资金强度持续转正，价格站上趋势线并突破前高或明显放量。",
        "sell": "主力资金明显转负，或价格跌破趋势均线。",
        "explain": ("优先使用程序缓存的真实资金流字段。", "同花顺普通公式无法直接复现本地资金数据。"),
        "optimization": "同花顺 Level-2 用户可手工替换为大单净量或 DDE 指标。",
    },
    "ths_lhb_institution": {
        "name": "龙虎榜机构确认",
        "tagline": "机构净买入只做确认，不单独追榜",
        "buy": "龙虎榜或机构席位净买入转强、价格在趋势均线上方，资金没有明显流出。",
        "sell": "机构席位显著净卖出，或价格跌破趋势均线。",
        "explain": ("龙虎榜数据是收盘后事件因子，最早用于下一交易日。", "未上榜与数据缺失必须严格区分。"),
        "optimization": "该策略依赖程序本地龙虎榜缓存，无法生成完全等价的普通同花顺公式。",
    },
    "ths_alloy_momentum": {
        "name": "六维动量共振",
        "tagline": "趋势、MACD、KDJ、RSI、量能、资金至少五项同向",
        "buy": "六项条件中至少五项转强，并且此前尚未达到五项。",
        "sell": "强势条件降至两项及以下，或价格跌破趋势均线。",
        "explain": ("多数表决降低单一指标误报。", "由未满足到满足的边沿触发，避免每天重复给买点。"),
        "optimization": "导出到同花顺时用 OBV 方向代替本地资金流，属于技术指标兼容版。",
    },
    "ths_quality_pullback": {
        "name": "优质趋势回踩",
        "tagline": "中期趋势向上，短线回踩后重新转强",
        "buy": "20 日均线高于 60 日均线，价格重新上穿 20 日线，RSI 和 WR 位于合理区间。",
        "sell": "跌破 60 日均线、RSI 过热，或资金明显转弱。",
        "explain": ("先确认中期趋势，再寻找短线回踩。", "不过度超买时入场，盈亏比通常更容易控制。"),
        "optimization": "导出公式用 OBV 方向替代本地资金流确认。",
    },
    "ths_risk_off": {
        "name": "稳健趋势守门",
        "tagline": "只做不过热、不过度放量的温和趋势",
        "buy": "价格在趋势均线上方、RSI 位于 42 至 68、资金转强且量比适中。",
        "sell": "跌破趋势均线、资金明显转弱、极端爆量或 RSI 过热。",
        "explain": ("限制 RSI 和量比可以减少追逐情绪顶点。", "趋势破坏时优先退出。"),
        "optimization": "导出公式用 OBV 方向替代本地资金流确认。",
    },
}


def strategy_title_map() -> dict[str, str]:
    return {key: str(value["name"]) for key, value in _CARDS.items()}


def _formula_parts(strategy_type: str, fast: int, slow: int) -> tuple[str, str, str]:
    f = max(3, int(fast))
    s = max(f + 1, int(slow))
    ma = f"F:={f};\nS:={s};\nMAF:=MA(CLOSE,F);\nMAS:=MA(CLOSE,S);\nVR:=VOL/MA(VOL,20);"
    macd = "DIF:=EMA(CLOSE,12)-EMA(CLOSE,26);\nDEA:=EMA(DIF,9);"
    rsi = (
        "LC:=REF(CLOSE,1);\n"
        "RSI6:=SMA(MAX(CLOSE-LC,0),6,1)/SMA(ABS(CLOSE-LC),6,1)*100;\n"
        "RSI12:=SMA(MAX(CLOSE-LC,0),12,1)/SMA(ABS(CLOSE-LC),12,1)*100;\n"
        "RSI14:=SMA(MAX(CLOSE-LC,0),14,1)/SMA(ABS(CLOSE-LC),14,1)*100;"
    )
    kdj = (
        "RSV:=IF(HHV(HIGH,9)=LLV(LOW,9),50,(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100);\n"
        "K:=SMA(RSV,3,1);\nD:=SMA(K,3,1);\nJ:=3*K-2*D;"
    )
    obv = "OBV1:=SUM(IF(CLOSE>REF(CLOSE,1),VOL,IF(CLOSE<REF(CLOSE,1),-VOL,0)),0);\nOBVMA:=MA(OBV1,10);"

    if strategy_type == "ths_oversold_reversal":
        defs = f"{rsi}\n{kdj}\n{macd}\nMA20:=MA(CLOSE,20);\nTOPDIV:=CLOSE>=HHV(CLOSE,20)*0.98 AND DIF<REF(HHV(DIF,20),1);"
        return defs, "CROSS(RSI6,RSI12) OR (J<10 AND CROSS(K,D))", "TOPDIV OR WINNER(CLOSE)>0.90 OR CROSS(MA20,CLOSE)"
    if strategy_type == "ths_macd_kdj_rsi":
        return f"{ma}\n{macd}\n{rsi}\n{kdj}\nMA20:=MA(CLOSE,20);", "CROSS(DIF,DEA) AND K>D AND RSI14>=45 AND RSI14<=72 AND CLOSE>MA20 AND VR>=0.8 AND VR<=3.5", "CROSS(DEA,DIF) OR (K<D AND K>75) OR RSI14>80 OR CLOSE<MA20"
    if strategy_type in {"ths_boll_squeeze", "ths_boll_rsi_break"}:
        defs = f"{ma}\n{rsi}\nMID:=MA(CLOSE,S);\nUPPER:=MID+2*STD(CLOSE,S);\nLOWER:=MID-2*STD(CLOSE,S);\nWIDTH:=(UPPER-LOWER)/MID;\nPREHIGH:=REF(HHV(HIGH,F),1);"
        if strategy_type == "ths_boll_squeeze":
            return defs, "REF(WIDTH,1)<=REF(LLV(WIDTH,60),1)*1.15 AND CLOSE>UPPER AND CLOSE>PREHIGH AND VR>1.15", "CLOSE<MID OR WIDTH>HHV(WIDTH,60)*0.95"
        return defs, "CLOSE>UPPER AND RSI14>=50 AND RSI14<=76 AND VR>1.05", "CLOSE<MID OR RSI14>82 OR CLOSE<LOWER"
    if strategy_type == "ths_wr_rebound":
        return f"{ma}\nWR:=(HHV(HIGH,F)-CLOSE)/(HHV(HIGH,F)-LLV(LOW,F))*-100;", "CROSS(WR,-80) AND CLOSE>MAS*0.96 AND VR>0.75", "WR>-18 OR CLOSE<MAS*0.96"
    if strategy_type == "ths_cci_breakout":
        return f"{ma}\nTYP:=(HIGH+LOW+CLOSE)/3;\nCCI1:=(TYP-MA(TYP,F))/(0.015*AVEDEV(TYP,F));", "CROSS(CCI1,100) AND CLOSE>MAS AND VR>1", "CROSS(0,CCI1) OR CLOSE<MAS"
    if strategy_type == "ths_bias_revert":
        return f"{ma}\nBIAS1:=(CLOSE-MA(CLOSE,F))/MA(CLOSE,F)*100;", "CROSS(BIAS1,-6) AND CLOSE>MAS*0.92", "BIAS1>8 OR CLOSE<MAS*0.94"
    if strategy_type == "ths_mtm_accel":
        return f"{ma}\nMTM1:=CLOSE-REF(CLOSE,F);\nMTMMA:=MA(MTM1,F);", "CROSS(MTM1,MTMMA) AND MTM1>0 AND CLOSE>MAS AND VR>0.9", "CROSS(MTMMA,MTM1) OR CLOSE<MAS"
    if strategy_type == "ths_kdj_oversold":
        return f"{ma}\n{kdj}", "CROSS(K,D) AND J<35 AND CLOSE>MAS*0.94", "(CROSS(D,K) AND J>80) OR CLOSE<MAS*0.95"
    if strategy_type == "ths_volume_breakout":
        return f"{ma}\nPREHIGH:=REF(HHV(HIGH,S),1);\nRECLOW:=REF(LLV(LOW,F),1);", "CLOSE>PREHIGH AND MAF>MAS AND VR>=1.35 AND VR<=5", "CLOSE<MAF OR CLOSE<RECLOW"
    if strategy_type == "ths_shrink_pullback":
        return f"{ma}\nRECLOW:=REF(LLV(LOW,F),1);", "REF(CLOSE,1)<REF(MAF,1) AND REF(CLOSE,1)>REF(MAS,1)*0.97 AND REF(VR,1)<0.85 AND CLOSE>MAF AND MAF>MAS", "CLOSE<MAS OR CLOSE<RECLOW"
    if strategy_type == "ths_amount_breakout":
        return f"{ma}\nAR:=AMOUNT/MA(AMOUNT,20);\nPREHIGH:=REF(HHV(HIGH,S),1);", "CLOSE>PREHIGH AND CLOSE>MAS AND AR>=1.4 AND AR<=6", "CLOSE<MAS OR AR<0.55"
    if strategy_type == "ths_platform_breakout":
        return f"{ma}\nBOXH:=REF(HHV(HIGH,S),1);\nBOXL:=REF(LLV(LOW,S),1);\nBOXW:=(BOXH-BOXL)/CLOSE;", "BOXW<0.22 AND CLOSE>BOXH AND VR>1.15", "CLOSE<MAF OR CLOSE<BOXL"
    if strategy_type == "ths_new_high":
        return f"{ma}\nHIGHLINE:=REF(HHV(HIGH,MAX(S,55)),1);", "CLOSE>HIGHLINE AND MAF>MAS AND VR>1", "CLOSE<MAF OR CROSS(MAS,MAF)"
    if strategy_type == "ths_trendline_break":
        return f"F:={f};\nS:={s};\nFASTEMA:=EMA(CLOSE,F);\nSLOWEMA:=EMA(CLOSE,MAX(S,34));\nTRENDLINE:=FASTEMA*0.35+SLOWEMA*0.65;\nVR:=VOL/MA(VOL,10);", "CROSS(CLOSE,TRENDLINE) AND TRENDLINE>REF(TRENDLINE,5) AND VR>0.9", "CROSS(TRENDLINE,CLOSE) OR TRENDLINE<REF(TRENDLINE,5)"
    if strategy_type == "ths_obv_mfi":
        defs = f"{ma}\n{obv}\nTP:=(HIGH+LOW+CLOSE)/3;\nPMF:=SUM(IF(TP>REF(TP,1),TP*VOL,0),14);\nNMF:=SUM(IF(TP<REF(TP,1),TP*VOL,0),14);\nMFI1:=100-100/(1+PMF/MAX(NMF,1));"
        return defs, "CROSS(OBV1,OBVMA) AND MFI1>50 AND CLOSE>MAS", "CROSS(OBVMA,OBV1) OR MFI1<42 OR CLOSE<MAS"
    if strategy_type in {"ths_capital_breakout", "ths_alloy_momentum", "ths_quality_pullback", "ths_risk_off"}:
        defs = f"{ma}\n{obv}\n{macd}\n{rsi}\n{kdj}\nPREHIGH:=REF(HHV(HIGH,S),1);\nWR:=(HHV(HIGH,14)-CLOSE)/(HHV(HIGH,14)-LLV(LOW,14))*-100;"
        if strategy_type == "ths_capital_breakout":
            return defs, "OBV1>OBVMA AND OBV1>REF(OBV1,2) AND CLOSE>MAS AND (CLOSE>PREHIGH OR VR>1.2)", "OBV1<OBVMA AND CLOSE<MAS"
        if strategy_type == "ths_alloy_momentum":
            return defs + "\nSCORE:=(CLOSE>MAS)+(DIF>DEA)+(K>D)+(RSI14>=45 AND RSI14<=72)+(OBV1>OBVMA)+(VR>1.15);", "SCORE>=5 AND REF(SCORE,1)<5", "SCORE<=2 OR CLOSE<MAS"
        if strategy_type == "ths_quality_pullback":
            return defs + "\nMA20:=MA(CLOSE,20);\nMA60:=MA(CLOSE,60);", "MA20>MA60 AND CROSS(CLOSE,MA20) AND RSI14>=38 AND RSI14<=62 AND WR>-80 AND OBV1>=OBVMA*0.98", "CLOSE<MA60 OR RSI14>78 OR OBV1<OBVMA"
        return defs, "CLOSE>MAS AND RSI14>=42 AND RSI14<=68 AND OBV1>OBVMA AND VR>=0.8 AND VR<=2.3", "CLOSE<MAS OR OBV1<OBVMA OR VR>4 OR RSI14>82"
    return "", "0", "0"


def build_ths_strategy_card(strategy_type: str, fast: int, slow: int) -> THSStrategyCard:
    meta = _CARDS.get(strategy_type)
    if meta is None:
        meta = {
            "name": strategy_type,
            "tagline": "自定义策略",
            "buy": "请以程序回测信号为准。",
            "sell": "请以程序回测信号为准。",
            "explain": ("该策略尚未配置同花顺公式模板。",),
            "optimization": "先完成规则核对后再导出。",
        }
    definitions, buy_expr, sell_expr = _formula_parts(strategy_type, fast, slow)
    unavailable = strategy_type == "ths_lhb_institution"
    if unavailable:
        note = "依赖程序本地龙虎榜/机构席位数据，普通同花顺公式编辑器无法获得同一数据源，因此不生成虚假的等价公式。"
        selection = trading = backtest = overlay = note
        displayed_buy_condition = str(meta["buy"])
        displayed_sell_condition = str(meta["sell"])
    else:
        degraded = strategy_type in {"ths_capital_breakout", "ths_alloy_momentum", "ths_quality_pullback", "ths_risk_off"}
        note = "同花顺兼容技术公式，可直接粘贴后编译。"
        if degraded:
            note = "兼容降级版：导出公式用 OBV 方向代替程序中的真实资金流，信号不会与本地回测完全一致。"
        selection = f"{definitions}\n\nBUYCOND:={buy_expr};\nXG:BUYCOND;"
        trading = f"{definitions}\n\nBUYCOND:={buy_expr};\nSELLCOND:={sell_expr};\nENTERLONG:BUYCOND;\nEXITLONG:SELLCOND;"
        backtest = (
            f"{definitions}\n\nBUYCOND:={buy_expr};\nSELLCOND:={sell_expr};\n"
            "BUYSIG:=BUYCOND AND REF(BUYCOND,1)=0;\n"
            "SELLSIG:=SELLCOND AND REF(SELLCOND,1)=0;\n"
            "DRAWICON(BUYSIG,LOW*0.98,1);\n"
            "DRAWTEXT(BUYSIG,LOW*0.96,'买入');\n"
            "DRAWICON(SELLSIG,HIGH*1.02,2);\n"
            "DRAWTEXT(SELLSIG,HIGH*1.04,'卖出');"
        )
        # 同花顺的策略回测窗口通过 DRAWICON/DRAWTEXT 识别买卖点；
        # K 线标注与回测必须共用同一信号，避免文字和实际触发条件不一致。
        overlay = backtest
        # 卡片顶部展示可执行表达式；自然语言解读仅放在解读区，避免把
        # “适中”“转弱”等无法编译的词误当作同花顺条件。
        displayed_buy_condition = f"BUYCOND:={buy_expr};"
        displayed_sell_condition = f"SELLCOND:={sell_expr};"
    return THSStrategyCard(
        strategy_type=strategy_type,
        name=str(meta["name"]),
        tagline=str(meta["tagline"]),
        buy_condition=displayed_buy_condition,
        sell_condition=displayed_sell_condition,
        interpretation=tuple(str(item) for item in meta["explain"]),
        optimization=str(meta["optimization"]),
        compatibility_note=note,
        selection_formula=selection,
        trading_formula=trading,
        backtest_formula=backtest,
        overlay_formula=overlay,
    )
