"""
回测脚本 - 使用本地数据（简化版）
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
    """加载价格数据 - 宽表格式"""
    log("加载价格数据...")
    # 优先读取完整数据，否则用sample
    csv_path = DATA_DIR / "history_hs300_full.csv" if (DATA_DIR / "history_hs300_full.csv").exists() else DATA_DIR / "history_hs300_sample.csv"
    log(f"  读取文件: {csv_path.name}")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # 透视表：每列一只股票，每行一个日期
    df = df.pivot_table(index='date', columns='code', values='close')
    df = df.sort_index()
    df = df.dropna(axis=1, how='all')  # 删除全空的列

    log(f"  加载 {len(df.columns)} 只股票, {len(df)} 个交易日")
    log(f"  时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    return df


class SimpleBacktester:
    """简单回测器"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital

    def run(self, name: str, select_func, price_df: pd.DataFrame,
            rebalance_dates: List, n_stocks: int = 20) -> Dict:
        """
        运行回测
        """
        log(f"\n{'='*50}")
        log(f"运行策略: {name}")
        log(f"{'='*50}")

        # 过滤有效的调仓日期
        available_dates = price_df.index.tolist()
        valid_dates = [d for d in rebalance_dates if d in available_dates]
        if not valid_dates:
            log("  没有有效调仓日期")
            return {}

        log(f"  调仓次数: {len(valid_dates)}, 每期持股: {n_stocks}")

        # 初始资金
        cash = self.initial_capital
        equity = [self.initial_capital]
        n_trades = 0

        for i, date in enumerate(valid_dates):
            # 选股
            selected = select_func(date, price_df)
            if not selected:
                continue

            # 过滤有数据的股票
            valid_stocks = [s for s in selected if s in price_df.columns and pd.notna(price_df.loc[date, s])]
            if not valid_stocks:
                continue

            # 取前 n_stocks 只
            valid_stocks = valid_stocks[:n_stocks]

            # 下一个调仓日（如果最后一天没有下一个，就持有到最后）
            if i < len(valid_dates) - 1:
                next_date = valid_dates[i + 1]
            else:
                next_date = available_dates[-1]  # 持有到最后

            # 等权分配
            per_stock_value = cash / len(valid_stocks)

            # 计算买入持有到下一个调仓日的收益
            start_prices = price_df.loc[date, valid_stocks]
            end_prices = price_df.loc[next_date, valid_stocks]

            # 组合收益
            period_return = 0
            for stock in valid_stocks:
                s, e = start_prices[stock], end_prices[stock]
                if pd.notna(s) and pd.notna(e) and s > 0:
                    period_return += (e - s) / s
                    n_trades += 1

            # 平均收益
            period_return = period_return / len(valid_stocks)

            # 更新资金
            cash = cash * (1 + period_return)
            equity.append(cash)

        # 计算指标
        final_value = equity[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        years = len(valid_dates) / 12
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 最大回撤
        eq = pd.Series(equity)
        rolling_max = eq.expanding().max()
        drawdowns = (eq - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        # 夏普
        returns = eq.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

        result = {
            'strategy_name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'n_trades': n_trades,
        }

        log(f"  总收益率: {total_return:.2%}")
        log(f"  年化收益: {annual_return:.2%}")
        log(f"  夏普比率: {sharpe:.2f}")
        log(f"  最大回撤: {max_drawdown:.2%}")
        log(f"  期末资产: {final_value:,.0f}")

        return result


# ============ 策略定义 ============

def select_momentum(date, price_df, lookback=20, n=20):
    """动量策略：过去N日涨幅最大的"""
    if date not in price_df.index:
        return []
    date_idx = price_df.index.get_loc(date)
    if date_idx < lookback:
        start_idx = 0
    else:
        start_idx = date_idx - lookback

    start_prices = price_df.iloc[start_idx]
    end_prices = price_df.iloc[date_idx]

    returns = (end_prices - start_prices) / start_prices
    returns = returns.dropna().sort_values(ascending=False)
    return returns.head(n).index.tolist()


def select_reversal(date, price_df, lookback=20, n=20):
    """反转策略：过去N日跌幅最大的"""
    if date not in price_df.index:
        return []
    date_idx = price_df.index.get_loc(date)
    if date_idx < lookback:
        start_idx = 0
    else:
        start_idx = date_idx - lookback

    start_prices = price_df.iloc[start_idx]
    end_prices = price_df.iloc[date_idx]

    returns = (end_prices - start_prices) / start_prices
    returns = returns.dropna().sort_values(ascending=True)
    return returns.head(n).index.tolist()


def select_low_price(date, price_df, n=20):
    """持有低价股策略"""
    if date not in price_df.index:
        return []
    day_data = price_df.loc[date].dropna()
    return day_data.sort_values(ascending=True).head(n).index.tolist()


def select_high_price(date, price_df, n=20):
    """持有高价股策略"""
    if date not in price_df.index:
        return []
    day_data = price_df.loc[date].dropna()
    return day_data.sort_values(ascending=False).head(n).index.tolist()


def select_random(date, price_df, n=20):
    """随机选股（作为基准）"""
    if date not in price_df.index:
        return []
    day_data = price_df.loc[date].dropna()
    return day_data.sample(min(n, len(day_data)), random_state=42).index.tolist()


def run_all_strategies():
    """运行所有策略"""
    log("="*60)
    log("选股策略回测系统 v2")
    log("="*60)

    # 加载数据
    price_df = load_price_data()

    # 生成调仓日期（每20个交易日 = 约1个月）
    dates = sorted(price_df.index.tolist())
    rebalance_dates = dates[::20]
    log(f"调仓日期数: {len(rebalance_dates)}")

    # 创建回测器
    bt = SimpleBacktester(initial_capital=1000000)

    # 策略列表
    strategies = [
        ("1.动量(20日)", lambda d, p: select_momentum(d, p, 20)),
        ("2.动量(60日)", lambda d, p: select_momentum(d, p, 60)),
        ("3.动量(120日)", lambda d, p: select_momentum(d, p, 120)),
        ("4.反转(20日)", lambda d, p: select_reversal(d, p, 20)),
        ("5.反转(60日)", lambda d, p: select_reversal(d, p, 60)),
        ("6.持有低价股", lambda d, p: select_low_price(d, p)),
        ("7.持有高价股", lambda d, p: select_high_price(d, p)),
        ("8.随机选股(基准)", lambda d, p: select_random(d, p)),
    ]

    # 运行
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

    # 对比报告
    if results:
        log("\n" + "="*60)
        log("策略对比报告")
        log("="*60)

        df = pd.DataFrame(results)
        df = df.sort_values('annual_return', ascending=False)

        print("\n" + "="*85)
        print(f"{'策略':<22} {'总收益':>10} {'年化':>10} {'夏普':>8} {'最大回撤':>10} {'期末资产':>15}")
        print("-"*85)
        for _, r in df.iterrows():
            print(f"{r['strategy_name']:<22} {r['total_return']:>10.2%} {r['annual_return']:>10.2%} "
                  f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>10.2%} {r['final_value']:>15,.0f}")
        print("="*85)

        # 保存
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
        with open(OUTPUT_DIR / "backtest_details.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log(f"\n结果已保存到 {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    run_all_strategies()
