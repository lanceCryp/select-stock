"""
策略 6: 高股息策略
选取股息率最高的股票
"""
import pandas as pd
import numpy as np
from typing import List
from ..data.stock_data import StockData


class DividendStrategy:
    """高股息策略"""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.name = "高股息策略"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            dividend = stock_data.get_dividend_data(code)
            if dividend and dividend > 0:
                results.append({
                    'code': code,
                    'dividend': dividend
                })

        if not results:
            return []

        df = pd.DataFrame(results)
        df = df.sort_values('dividend', ascending=False).head(self.top_n)
        return df['code'].tolist()
