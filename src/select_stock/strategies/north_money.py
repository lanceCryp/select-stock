"""
策略 9: 北向资金策略
选取北向资金持股比例最高的股票
"""
import pandas as pd
import numpy as np
from typing import List
from ..data.stock_data import StockData


class NorthMoneyStrategy:
    """北向资金策略"""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.name = "北向资金策略"

    def select(self, date: str, stock_data: StockData) -> List[str]:
        """选股"""
        try:
            # 使用 akshare 获取北向资金持股数据
            import akshare as ak
            df = ak.stock_hsgt_north_hold_stock(symbol="北向资金")
            if df is not None and not df.empty:
                df = df.head(self.top_n)
                return df['代码'].tolist()
        except Exception as e:
            print(f"获取北向资金数据失败: {e}")

        return []


def get_north_money_data() -> pd.DataFrame:
    """获取北向资金持股数据"""
    try:
        import akshare as ak
        df = ak.stock_hsgt_north_hold_stock(symbol="北向资金")
        return df
    except Exception as e:
        print(f"获取北向资金数据失败: {e}")
        return pd.DataFrame()
