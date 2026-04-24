"""
策略 7: 成交量异动策略
选取当日成交量突破的股票
"""
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime, timedelta
from ..data.stock_data import StockData


class VolumeBreakoutStrategy:
    """成交量异动策略"""

    def __init__(self, top_n: int = 50, volume_ratio: float = 2.0):
        """
        Args:
            top_n: 选取数量
            volume_ratio: 成交量放大倍数
        """
        self.top_n = top_n
        self.volume_ratio = volume_ratio
        self.name = f"量能异动策略({volume_ratio}x)"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        all_stocks = stock_data.get_stock_list()
        if all_stocks.empty:
            return []

        # 获取前 20 日数据作为基准
        current = datetime.strptime(date, '%Y%m%d')
        start = current - timedelta(days=60)
        start_str = start.strftime('%Y%m%d')

        results = []
        for _, row in all_stocks.iterrows():
            code = row['code']
            df = stock_data.get_daily_data(code, start_str, date)

            if len(df) >= 20:
                recent_volumes = df['成交量'].iloc[-5:].mean()
                avg_volume = df['成交量'].iloc[:-5].mean() if len(df) > 5 else df['成交量'].mean()
                today_volume = df['成交量'].iloc[-1]

                if avg_volume > 0:
                    vol_ratio = today_volume / avg_volume
                    # 涨幅也要为正
                    close_start = df['收盘'].iloc[-5]
                    close_end = df['收盘'].iloc[-1]
                    price_change = (close_end - close_start) / close_start if close_start > 0 else 0

                    if vol_ratio >= self.volume_ratio and price_change > 0:
                        results.append({
                            'code': code,
                            'vol_ratio': vol_ratio,
                            'price_change': price_change
                        })

        if not results:
            return []

        df = pd.DataFrame(results)
        df = df.sort_values('vol_ratio', ascending=False).head(self.top_n)
        return df['code'].tolist()
