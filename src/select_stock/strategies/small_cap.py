"""
策略 1: 小市值选股策略
选取市值最小的股票，假设小市值有超额收益
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..data.stock_data import StockData


class SmallCapStrategy:
    """小市值策略"""

    def __init__(self, top_n: int = 50):
        """
        Args:
            top_n: 选取市值最小的 N 只股票
        """
        self.top_n = top_n
        self.name = f"小市值策略(Top{top_n})"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """根据指定日期选股"""
        # 获取所有股票
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            market_cap = stock_data.get_market_cap(code)
            if market_cap and market_cap > 0:
                results.append({
                    'code': code,
                    'market_cap': market_cap
                })

        if not results:
            return []

        df = pd.DataFrame(results)
        df = df.sort_values('market_cap', ascending=True).head(self.top_n)
        return df['code'].tolist()


def get_signal(signal_dates: List[str], stock_data: StockData) -> pd.DataFrame:
    """生成选股信号"""
    strategy = SmallCapStrategy(top_n=50)
    all_selected = {}

    for date in signal_dates:
        selected = strategy.select(date, stock_data)
        for stock in selected:
            if stock not in all_selected:
                all_selected[stock] = {}
            all_selected[stock][date] = 1.0 / len(selected)  # 平均仓位

    if not all_selected:
        return pd.DataFrame()

    df = pd.DataFrame(all_selected).T.fillna(0)
    return df
