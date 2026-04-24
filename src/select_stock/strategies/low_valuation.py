"""
策略 2: 低估值因子策略 (PE/PB)
选取 PE、PB 最低的股票
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from ..data.stock_data import StockData


class LowValuationStrategy:
    """低估值策略"""

    def __init__(self, top_n: int = 50, use_pe: bool = True, use_pb: bool = True):
        """
        Args:
            top_n: 选取数量
            use_pe: 使用 PE 因子
            use_pb: 使用 PB 因子
        """
        self.top_n = top_n
        self.use_pe = use_pe
        self.use_pb = use_pb
        self.name = "低估值策略"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            fundamental = stock_data.get_fundamental_data(code)
            pe = fundamental.get('pe', None)
            pb = fundamental.get('pb', None)

            if pe and pe > 0:
                results.append({'code': code, 'pe': pe, 'pb': pb})

        if not results:
            return []

        df = pd.DataFrame(results)

        # 过滤负值
        if self.use_pe:
            df = df[df['pe'] > 0]
        if self.use_pb:
            df = df[df['pb'] > 0]

        if df.empty:
            return []

        # 综合打分（PE 和 PB 各占一半）
        df['score'] = 0
        if self.use_pe and 'pe' in df.columns:
            df['pe_score'] = 1 / df['pe']  # PE 越低越好
            df['score'] += df['pe_score']
        if self.use_pb and 'pb' in df.columns:
            df['pb_score'] = 1 / df['pb']  # PB 越低越好
            df['score'] += df['pb_score']

        df = df.sort_values('score', ascending=False).head(self.top_n)
        return df['code'].tolist()
