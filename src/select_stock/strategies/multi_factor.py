"""
策略 10: 多因子综合策略
综合多个因子打分选股
"""
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime, timedelta
from ..data.stock_data import StockData


class MultiFactorStrategy:
    """多因子综合策略"""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.name = "多因子综合策略"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """综合打分选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        # 计算回顾期
        current = datetime.strptime(date, '%Y%m%d')
        start = current - timedelta(days=90)
        start_str = start.strftime('%Y%m%d')

        results = []
        total = len(all_stocks)

        for idx, (_, row) in enumerate(all_stocks.iterrows()):
            if idx % 100 == 0:
                print(f"  多因子进度: {idx}/{total}")

            code = row['code']

            # 1. 市值因子
            market_cap = stock_data.get_market_cap(code)

            # 2. 估值因子
            fundamental = stock_data.get_fundamental_data(code)

            # 3. 成长因子
            financial = stock_data.get_financial_data(code)

            # 4. 动量因子
            df = stock_data.get_daily_data(code, start_str, date)
            momentum = 0
            if len(df) >= 20:
                start_price = df['收盘'].iloc[0]
                end_price = df['收盘'].iloc[-1]
                if start_price > 0:
                    momentum = (end_price - start_price) / start_price

            # 计算综合得分
            score = 0
            factors = {}

            # 市值因子 (越小越好，log 变换)
            if market_cap and market_cap > 0:
                factors['size'] = 1 / np.log(market_cap + 1)

            # PE 因子 (越小越好)
            pe = fundamental.get('pe')
            if pe and pe > 0 and pe < 100:
                factors['pe'] = 1 / pe

            # PB 因子 (越小越好)
            pb = fundamental.get('pb')
            if pb and pb > 0 and pb < 20:
                factors['pb'] = 1 / pb

            # 成长因子
            profit_growth = financial.get('profit_growth')
            if profit_growth and profit_growth > 0:
                factors['growth'] = min(profit_growth / 100, 2)  # 限制上限

            # 动量因子
            factors['momentum'] = max(-0.3, min(momentum, 0.3)) + 0.3  # 归一化到 0-0.6

            if factors:
                # 标准化得分
                score = sum(factors.values())

            if score > 0:
                results.append({
                    'code': code,
                    'score': score,
                    'market_cap': market_cap,
                    'pe': fundamental.get('pe'),
                    'pb': fundamental.get('pb'),
                    'profit_growth': profit_growth,
                    'momentum': momentum
                })

        if not results:
            return []

        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False).head(self.top_n)
        return df['code'].tolist()
