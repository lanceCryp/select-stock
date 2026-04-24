"""
策略 4: 动量策略
选取过去 N 月涨幅最高的股票
"""
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime, timedelta
from ..data.stock_data import StockData


class MomentumStrategy:
    """动量策略"""

    def __init__(self, top_n: int = 50, lookback_months: int = 3):
        """
        Args:
            top_n: 选取数量
            lookback_months: 回顾月数
        """
        self.top_n = top_n
        self.lookback_months = lookback_months
        self.name = f"动量策略({lookback_months}月)"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        # 计算回顾期
        current = datetime.strptime(date, '%Y%m%d')
        start = current - timedelta(days=self.lookback_months * 35)
        start_str = start.strftime('%Y%m%d')

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            df = stock_data.get_daily_data(code, start_str, date)

            if len(df) >= 20:  # 需要足够数据
                start_price = df['收盘'].iloc[0]
                end_price = df['收盘'].iloc[-1]
                if start_price > 0:
                    return_rate = (end_price - start_price) / start_price
                    results.append({
                        'code': code,
                        'return': return_rate
                    })

        if not results:
            return []

        df = pd.DataFrame(results)
        df = df.sort_values('return', ascending=False).head(self.top_n)
        return df['code'].tolist()
