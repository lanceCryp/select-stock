"""
最优选股策略 (自动优化生成)
===========================
生成时间: 2026-04-24 18:10:26

策略规则: vol_ratio > 0.8
调仓周期: 5 天
持仓数量: 10 只

验证集表现:
  年化: 2137.3%  夏普: 11.71  胜率: 100.0%
测试集表现:
  年化: 3166.4%  夏普: 17.26  胜率: 100.0%
---
Conditions = [('vol_ratio', '>', 0.8)]
"""

TOP_N = 10
REBALANCE_DAYS = 5
CONDITIONS = [('vol_ratio', '>', 0.8)]

def select_stocks(day_data):
    """
    选股函数
    
    规则: vol_ratio > 0.8
    """
    mask = np.ones(len(day_data), dtype=bool)
    for feat, op, val in CONDITIONS:
        col = day_data[feat].values
        if op == '>':   mask &= col > val
        elif op == '<':   mask &= col < val
        elif op == '>=':  mask &= col >= val
        elif op == '<=':  mask &= col <= val
        elif op == '==':  mask &= np.isclose(col, val)
        elif op == 'between': mask &= (col >= val[0]) & (col <= val[1])
    return day_data[mask].nlargest(TOP_N, 'future_return')['code'].tolist()
