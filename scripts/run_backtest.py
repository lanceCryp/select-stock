"""
回测脚本 - 扩展版
运行方式: uv run python scripts/run_backtest.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import json

DATA_DIR = Path("data")
OUTPUT_DIR = Path("backtest_results")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_price_data() -> pd.DataFrame:
    """加载价格数据"""
    log("加载价格数据...")
    csv_path = DATA_DIR / "history_hs300_full.csv" if (DATA_DIR / "history_hs300_full.csv").exists() else DATA_DIR / "history_hs300_sample.csv"
    log(f"  读取文件: {csv_path.name}")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    df = df.pivot_table(index='date', columns='code', values='close')
    df = df.sort_index()
    df = df.dropna(axis=1, how='all')

    log(f"  加载 {len(df.columns)} 只股票, {len(df)} 个交易日")
    log(f"  时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    return df


class SimpleBacktester:
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital

    def run(self, name: str, select_func, price_df: pd.DataFrame,
            rebalance_dates: List, n_stocks: int = 20) -> Dict:
        available_dates = price_df.index.tolist()
        valid_dates = [d for d in rebalance_dates if d in available_dates]
        if not valid_dates:
            return {}

        cash = self.initial_capital
        equity = [self.initial_capital]
        n_trades = 0

        for i, date in enumerate(valid_dates):
            selected = select_func(date, price_df)
            if not selected:
                equity.append(equity[-1])
                continue

            valid_stocks = [s for s in selected if s in price_df.columns and pd.notna(price_df.loc[date, s])]
            if not valid_stocks:
                equity.append(equity[-1])
                continue

            valid_stocks = valid_stocks[:n_stocks]

            if i < len(valid_dates) - 1:
                next_date = valid_dates[i + 1]
            else:
                next_date = available_dates[-1]

            start_prices = price_df.loc[date, valid_stocks]
            end_prices = price_df.loc[next_date, valid_stocks]

            period_return = 0
            for stock in valid_stocks:
                s, e = start_prices[stock], end_prices[stock]
                if pd.notna(s) and pd.notna(e) and s > 0:
                    period_return += (e - s) / s
                    n_trades += 1

            period_return = period_return / len(valid_stocks)
            cash = cash * (1 + period_return)
            equity.append(cash)

        final_value = equity[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        years = len(valid_dates) / 12
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        eq = pd.Series(equity)
        rolling_max = eq.expanding().max()
        drawdowns = (eq - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        returns = eq.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

        return {
            'strategy_name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'n_trades': n_trades,
        }


# ==================== 策略定义 ====================

def get_returns(price_df, lookback):
    """计算过去N日收益率"""
    returns = price_df.pct_change(periods=lookback)
    return returns


def get_volatility(price_df, lookback=20):
    """计算过去N日波动率"""
    returns = price_df.pct_change().rolling(lookback).std()
    return returns


def get_trend_strength(price_df, lookback=20):
    """趋势强度：价格在均线上的比例"""
    ma = price_df.rolling(lookback).mean()
    ratio = (price_df - ma) / ma
    return ratio


# ----- 策略函数 -----

def select_momentum_20(date, df, n=20):
    if date not in df.index: return []
    returns = get_returns(df, 20)
    if date not in returns.index: return []
    r = returns.loc[date].dropna().sort_values(ascending=False)
    return r.head(n).index.tolist()

def select_momentum_60(date, df, n=20):
    if date not in df.index: return []
    returns = get_returns(df, 60)
    if date not in returns.index: return []
    r = returns.loc[date].dropna().sort_values(ascending=False)
    return r.head(n).index.tolist()

def select_momentum_120(date, df, n=20):
    if date not in df.index: return []
    returns = get_returns(df, 120)
    if date not in returns.index: return []
    r = returns.loc[date].dropna().sort_values(ascending=False)
    return r.head(n).index.tolist()

def select_reversal_20(date, df, n=20):
    if date not in df.index: return []
    returns = get_returns(df, 20)
    if date not in returns.index: return []
    r = returns.loc[date].dropna().sort_values(ascending=True)
    return r.head(n).index.tolist()

def select_reversal_60(date, df, n=20):
    if date not in df.index: return []
    returns = get_returns(df, 60)
    if date not in returns.index: return []
    r = returns.loc[date].dropna().sort_values(ascending=True)
    return r.head(n).index.tolist()

def select_low_price(date, df, n=20):
    if date not in df.index: return []
    day_data = df.loc[date].dropna()
    return day_data.sort_values(ascending=True).head(n).index.tolist()

def select_high_price(date, df, n=20):
    if date not in df.index: return []
    day_data = df.loc[date].dropna()
    return day_data.sort_values(ascending=False).head(n).index.tolist()

def select_low_volatility(date, df, n=20):
    """低波动策略：持有最稳定的股票"""
    if date not in df.index: return []
    vol = get_volatility(df, 20)
    if date not in vol.index: return []
    v = vol.loc[date].dropna().sort_values(ascending=True)
    return v.head(n).index.tolist()

def select_high_volatility(date, df, n=20):
    """高波动策略：持有最活跃的股票"""
    if date not in df.index: return []
    vol = get_volatility(df, 20)
    if date not in vol.index: return []
    v = vol.loc[date].dropna().sort_values(ascending=False)
    return v.head(n).index.tolist()

def select_strong_trend(date, df, n=20):
    """强趋势策略：价格在均线上方最多的"""
    if date not in df.index: return []
    ts = get_trend_strength(df, 20)
    if date not in ts.index: return []
    t = ts.loc[date].dropna().sort_values(ascending=False)
    return t.head(n).index.tolist()

def select_weak_trend(date, df, n=20):
    """弱趋势策略：价格在均线下方最多的"""
    if date not in df.index: return []
    ts = get_trend_strength(df, 20)
    if date not in ts.index: return []
    t = ts.loc[date].dropna().sort_values(ascending=True)
    return t.head(n).index.tolist()

def select_momentum_combined(date, df, n=20):
    """动量综合：20日+60日+120日打分"""
    if date not in df.index: return []
    r20 = get_returns(df, 20)
    r60 = get_returns(df, 60)
    r120 = get_returns(df, 120)
    if date not in r20.index: return []

    scores = pd.Series(0.0, index=df.columns)
    for col in df.columns:
        s20 = r20.loc[date, col] if date in r20.index and col in r20.columns else np.nan
        s60 = r60.loc[date, col] if date in r60.index and col in r60.columns else np.nan
        s120 = r120.loc[date, col] if date in r120.index and col in r120.columns else np.nan
        if pd.notna(s20) and pd.notna(s60) and pd.notna(s120):
            scores[col] = s20 * 0.5 + s60 * 0.3 + s120 * 0.2

    scores = scores.dropna().sort_values(ascending=False)
    return scores.head(n).index.tolist()

def select_reversal_combined(date, df, n=20):
    """反转综合：短期跌幅大的"""
    if date not in df.index: return []
    r5 = get_returns(df, 5)
    r10 = get_returns(df, 10)
    r20 = get_returns(df, 20)
    if date not in r5.index: return []

    scores = pd.Series(0.0, index=df.columns)
    for col in df.columns:
        s5 = r5.loc[date, col] if date in r5.index and col in r5.columns else np.nan
        s10 = r10.loc[date, col] if date in r10.index and col in r10.columns else np.nan
        s20 = r20.loc[date, col] if date in r20.index and col in r20.columns else np.nan
        if pd.notna(s5) and pd.notna(s10) and pd.notna(s20):
            scores[col] = s5 * 0.5 + s10 * 0.3 + s20 * 0.2

    scores = scores.dropna().sort_values(ascending=True)
    return scores.head(n).index.tolist()

def select_high_return_low_vol(date, df, n=20):
    """高收益低波动：风险调整后收益最高 (夏普-like)"""
    if date not in df.index: return []
    ret = get_returns(df, 20)
    vol = get_volatility(df, 20)
    if date not in ret.index: return []

    scores = pd.Series(0.0, index=df.columns)
    for col in df.columns:
        r = ret.loc[date, col] if col in ret.columns and date in ret.index else np.nan
        v = vol.loc[date, col] if col in vol.columns and date in vol.index else np.nan
        if pd.notna(r) and pd.notna(v) and v > 0:
            scores[col] = r / v

    scores = scores.dropna().sort_values(ascending=False)
    return scores.head(n).index.tolist()

def select_volume_leverage(date, df, n=20):
    """借力策略：近期涨幅大+波动高的"""
    if date not in df.index: return []
    ret = get_returns(df, 20)
    vol = get_volatility(df, 20)
    if date not in ret.index: return []

    scores = pd.Series(0.0, index=df.columns)
    for col in df.columns:
        r = ret.loc[date, col] if col in ret.columns and date in ret.index else np.nan
        v = vol.loc[date, col] if col in vol.columns and date in vol.index else np.nan
        if pd.notna(r) and pd.notna(v):
            scores[col] = r * v

    scores = scores.dropna().sort_values(ascending=False)
    return scores.head(n).index.tolist()

def select_momentum_reversal_blend(date, df, n=20):
    """动量+反转混合：市场强势时追涨，弱势时抄底"""
    if date not in df.index: return []
    ret20 = get_returns(df, 20)
    if date not in ret20.index: return []

    # 计算市场整体涨跌
    market_return = ret20.loc[date].dropna().mean()

    if market_return > 0:
        # 市场上涨：用动量
        return select_momentum_20(date, df, n)
    else:
        # 市场下跌：用反转
        return select_reversal_20(date, df, n)

def select_breakout_20(date, df, n=20):
    """20日新高策略：突破20日最高价的"""
    if date not in df.index: return []
    rolling_max = df.rolling(20).max().shift(1)  # 昨日最高
    if date not in rolling_max.index: return []

    ratio = (df.loc[date] - rolling_max.loc[date]) / rolling_max.loc[date]
    ratio = ratio.dropna().sort_values(ascending=False)
    return ratio.head(n).index.tolist()

def select_breakdown_20(date, df, n=20):
    """20日新低策略：跌破20日最低价的"""
    if date not in df.index: return []
    rolling_min = df.rolling(20).min().shift(1)
    if date not in rolling_min.index: return []

    ratio = (df.loc[date] - rolling_min.loc[date]) / rolling_min.loc[date]
    ratio = ratio.dropna().sort_values(ascending=True)
    return ratio.head(n).index.tolist()


def run_all_strategies():
    log("="*60)
    log("选股策略回测系统 v3 - 扩展版")
    log("="*60)

    price_df = load_price_data()

    dates = sorted(price_df.index.tolist())
    rebalance_dates = dates[::20]
    log(f"调仓日期数: {len(rebalance_dates)}")

    bt = SimpleBacktester(initial_capital=1000000)

    strategies = [
        # 基础动量/反转
        ("S01_动量20日", lambda d, p: select_momentum_20(d, p)),
        ("S02_动量60日", lambda d, p: select_momentum_60(d, p)),
        ("S03_动量120日", lambda d, p: select_momentum_120(d, p)),
        ("S04_反转20日", lambda d, p: select_reversal_20(d, p)),
        ("S05_反转60日", lambda d, p: select_reversal_60(d, p)),
        # 价格策略
        ("S06_低价股", lambda d, p: select_low_price(d, p)),
        ("S07_高价股", lambda d, p: select_high_price(d, p)),
        # 波动率策略
        ("S08_低波动", lambda d, p: select_low_volatility(d, p)),
        ("S09_高波动", lambda d, p: select_high_volatility(d, p)),
        # 趋势策略
        ("S10_强趋势", lambda d, p: select_strong_trend(d, p)),
        ("S11_弱趋势", lambda d, p: select_weak_trend(d, p)),
        # 综合策略
        ("S12_动量综合", lambda d, p: select_momentum_combined(d, p)),
        ("S13_反转综合", lambda d, p: select_reversal_combined(d, p)),
        ("S14_高夏普", lambda d, p: select_high_return_low_vol(d, p)),
        ("S15_借力策略", lambda d, p: select_volume_leverage(d, p)),
        # 混合策略
        ("S16_动量反转切换", lambda d, p: select_momentum_reversal_blend(d, p)),
        # 突破策略
        ("S17_20日突破", lambda d, p: select_breakout_20(d, p)),
        ("S18_20日破位", lambda d, p: select_breakdown_20(d, p)),
    ]

    results = []
    for name, func in strategies:
        try:
            result = bt.run(name, func, price_df, rebalance_dates)
            if result:
                results.append(result)
        except Exception as e:
            log(f"策略 {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    if results:
        log("\n" + "="*60)
        log("策略对比报告")
        log("="*60)

        df = pd.DataFrame(results)
        df = df.sort_values('annual_return', ascending=False)

        print("\n" + "="*95)
        print(f"{'策略':<22} {'总收益':>10} {'年化':>10} {'夏普':>8} {'最大回撤':>10} {'期末资产':>15}")
        print("-"*95)
        for _, r in df.iterrows():
            rank = "🥇" if r['annual_return'] > 0.15 else "🥈" if r['annual_return'] > 0.08 else "🥉" if r['annual_return'] > 0 else "  "
            print(f"{rank} {r['strategy_name']:<20} {r['total_return']:>10.2%} {r['annual_return']:>10.2%} "
                  f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>10.2%} {r['final_value']:>15,.0f}")
        print("="*95)

        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
        with open(OUTPUT_DIR / "backtest_details.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存详细报告
        with open(OUTPUT_DIR / "ranking_report.md", "w", encoding="utf-8") as f:
            f.write("# 选股策略回测排名报告\n\n")
            f.write(f"**回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**数据范围**: 沪深300全量, 2022-01 ~ 2026-04\n")
            f.write(f"**调仓频率**: 每20个交易日\n")
            f.write(f"**持股数量**: 20只等权\n\n")
            f.write("## 排名表\n\n")
            f.write("| 排名 | 策略 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 |\n")
            f.write("|------|------|--------|----------|----------|----------|\n")
            for i, (_, r) in enumerate(df.iterrows(), 1):
                f.write(f"| {i} | {r['strategy_name']} | {r['total_return']:.2%} | {r['annual_return']:.2%} | {r['sharpe_ratio']:.2f} | {r['max_drawdown']:.2%} |\n")

        log(f"\n结果已保存到 {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    run_all_strategies()
