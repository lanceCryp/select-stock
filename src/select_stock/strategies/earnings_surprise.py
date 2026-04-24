"""
策略 8: 净利润惊喜策略
选取业绩超预期的股票（净利润增速超过预期）
"""
import pandas as pd
import numpy as np
from typing import List
from ..data.stock_data import StockData


class EarningsSurpriseStrategy:
    """业绩惊喜策略"""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.name = "业绩惊喜策略"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            financial = stock_data.get_financial_data(code)
            profit_growth = financial.get('profit_growth')
            revenue_growth = financial.get('revenue_growth')

            if profit_growth is not None and revenue_growth is not None:
                # 净利润增速和营收增速都为正，且净利润增速大于营收增速
                if profit_growth > 0 and revenue_growth > 0 and profit_growth > revenue_growth:
                    results.append({
                        'code': code,
                        'profit_growth': profit_growth,
                        'revenue_growth': revenue_growth,
                        'margin_improvement': profit_growth - revenue_growth
                    })

        if not results:
            return []

        df = pd.DataFrame(results)
        df = df.sort_values('margin_improvement', ascending=False).head(self.top_n)
        return df['code'].tolist()
