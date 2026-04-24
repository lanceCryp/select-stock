"""
策略 3: 成长因子策略
选取营收/净利润增速最高的股票
"""
import pandas as pd
import numpy as np
from typing import List
from ..data.stock_data import StockData


class GrowthStrategy:
    """成长策略"""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.name = "成长策略"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            financial = stock_data.get_financial_data(code)
            rev_growth = financial.get('revenue_growth')
            profit_growth = financial.get('profit_growth')

            if rev_growth is not None or profit_growth is not None:
                results.append({
                    'code': code,
                    'revenue_growth': rev_growth or 0,
                    'profit_growth': profit_growth or 0
                })

        if not results:
            return []

        df = pd.DataFrame(results)
        # 过滤非空
        df = df[(df['revenue_growth'] != 0) | (df['profit_growth'] != 0)]

        if df.empty:
            return []

        # 综合成长得分
        df['score'] = df['revenue_growth'] + df['profit_growth']
        df = df.sort_values('score', ascending=False).head(self.top_n)
        return df['code'].tolist()
