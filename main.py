"""
选股策略回测主程序
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入模块
from src.select_stock.data.stock_data import StockData, get_hs300_stocks, get_zz500_stocks
from src.select_stock.backtest.engine import BacktestEngine, BacktestResult, compare_strategies
from src.select_stock.strategies import (
    SmallCapStrategy,
    LowValuationStrategy,
    GrowthStrategy,
    MomentumStrategy,
    ReversalStrategy,
    DividendStrategy,
    VolumeBreakoutStrategy,
    EarningsSurpriseStrategy,
    NorthMoneyStrategy,
    MultiFactorStrategy,
)


class BacktestRunner:
    """回测运行器"""

    def __init__(self,
                 start_date: str = "20240101",
                 end_date: str = "20240401",
                 stock_pool: str = "hs300",
                 initial_capital: float = 1000000,
                 rebalance_months: int = 1):
        """
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            stock_pool: 股票池 'hs300' 或 'zz500' 或 'all'
            initial_capital: 初始资金
            rebalance_months: 调仓月数
        """
        self.start_date = start_date
        self.end_date = end_date
        self.stock_pool = stock_pool
        self.initial_capital = initial_capital
        self.rebalance_months = rebalance_months

        self.stock_data = StockData()
        self.engine = BacktestEngine(initial_capital=initial_capital)

        # 策略列表
        self.strategies = [
            ("1.小市值", SmallCapStrategy(top_n=30)),
            ("2.低估值", LowValuationStrategy(top_n=30)),
            ("3.成长", GrowthStrategy(top_n=30)),
            ("4.动量", MomentumStrategy(top_n=30, lookback_months=3)),
            ("5.反转", ReversalStrategy(top_n=30, lookback_months=3)),
            ("6.高股息", DividendStrategy(top_n=30)),
            ("7.量能异动", VolumeBreakoutStrategy(top_n=30)),
            ("8.业绩惊喜", EarningsSurpriseStrategy(top_n=30)),
            ("9.北向资金", NorthMoneyStrategy(top_n=30)),
            ("10.多因子", MultiFactorStrategy(top_n=30)),
        ]

        # 生成调仓日期
        self.rebalance_dates = self._generate_rebalance_dates()

    def _generate_rebalance_dates(self) -> List[str]:
        """生成调仓日期"""
        dates = []
        current = datetime.strptime(self.start_date, '%Y%m%d')
        end = datetime.strptime(self.end_date, '%Y%m%d')

        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            # 按月递增
            month = current.month + self.rebalance_months
            year = current.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            current = datetime(year, month, 1)

        return dates

    def get_stock_pool(self) -> List[str]:
        """获取股票池"""
        if self.stock_pool == "hs300":
            return get_hs300_stocks()
        elif self.stock_pool == "zz500":
            return get_zz500_stocks()
        else:
            # 返回所有股票
            df = self.stock_data.get_stock_list()
            return df['code'].tolist() if not df.empty else []

    def run_strategy(self, strategy, rebalance_dates: List[str]) -> BacktestResult:
        """运行单个策略回测"""
        print(f"\n{'='*50}")
        print(f"运行策略: {strategy.name}")
        print(f"{'='*50}")

        # 获取股票池
        stock_pool = self.get_stock_pool()
        print(f"股票池: {len(stock_pool)} 只股票")

        # 收集选股结果和价格数据
        all_selected = {}
        all_prices = {}

        for date in rebalance_dates:
            print(f"\n选股日期: {date}")
            selected = strategy.select(date, self.stock_data)
            print(f"  选中 {len(selected)} 只股票")

            if selected:
                for stock in selected:
                    if stock not in all_selected:
                        all_selected[stock] = {}
                    # 分配仓位
                    all_selected[stock][date] = 1.0 / len(selected)

            # 获取选中股票的价格数据
            for stock in selected[:20]:  # 限制数量避免请求过多
                if stock not in all_prices:
                    df = self.stock_data.get_daily_data(stock, self.start_date, self.end_date)
                    if not df.empty:
                        all_prices[stock] = df['收盘']

        if not all_selected or not all_prices:
            print("没有选股结果")
            return BacktestResult(strategy_name=strategy.name)

        # 构建信号矩阵
        signal_df = pd.DataFrame(all_selected).T.fillna(0)
        signal_df = signal_df.loc[signal_df.sum(axis=1) > 0]

        # 构建价格矩阵
        price_dict = {}
        for stock, series in all_prices.items():
            price_dict[stock] = series

        price_df = pd.DataFrame(price_dict)
        price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d')

        # 运行回测
        result = self.engine.run(signal_df, price_df, strategy.name)

        self._print_result(result)
        return result

    def _print_result(self, result: BacktestResult):
        """打印回测结果"""
        print(f"\n{'='*40}")
        print(f"回测结果: {result.strategy_name}")
        print(f"{'='*40}")
        print(f"总收益率:    {result.total_return:.2%}")
        print(f"年化收益率:  {result.annual_return:.2%}")
        print(f"夏普比率:    {result.sharpe_ratio:.2f}")
        print(f"最大回撤:    {result.max_drawdown:.2%}")
        print(f"胜率:        {result.win_rate:.2%}")
        print(f"交易次数:    {result.total_trades}")
        print(f"盈利次数:    {result.winning_trades}")
        print(f"亏损次数:    {result.losing_trades}")
        if result.avg_win > 0:
            print(f"平均盈利:    {result.avg_win:.2f}")
            print(f"平均亏损:    {result.avg_loss:.2f}")
            print(f"盈利因子:    {result.profit_factor:.2f}")

    def run_all(self) -> pd.DataFrame:
        """运行所有策略"""
        print("\n" + "="*60)
        print("选股策略回测系统")
        print("="*60)
        print(f"回测期间: {self.start_date} - {self.end_date}")
        print(f"股票池: {self.stock_pool}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"调仓周期: 每 {self.rebalance_months} 月")
        print(f"策略数量: {len(self.strategies)}")

        results = []
        for name, strategy in self.strategies:
            try:
                result = self.run_strategy(strategy, self.rebalance_dates)
                results.append(result)
            except Exception as e:
                print(f"策略 {name} 执行失败: {e}")
                import traceback
                traceback.print_exc()

        # 保存结果
        self.save_results(results)

        # 对比报告
        print("\n\n" + "="*80)
        print("策略对比报告")
        print("="*80)

        comparison = compare_strategies(results)
        print(comparison.to_string(index=False))

        return comparison

    def save_results(self, results: List[BacktestResult]):
        """保存回测结果"""
        output_dir = Path("backtest_results")
        output_dir.mkdir(exist_ok=True)

        # 保存详细结果
        for result in results:
            filename = f"{output_dir}/{result.strategy_name.replace(' ', '_')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'strategy_name': result.strategy_name,
                    'total_return': result.total_return,
                    'annual_return': result.annual_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'avg_win': result.avg_win,
                    'avg_loss': result.avg_loss,
                    'profit_factor': result.profit_factor,
                }, f, ensure_ascii=False, indent=2)

        # 保存对比报告
        comparison = compare_strategies(results)
        comparison.to_csv(f"{output_dir}/comparison.csv", index=False, encoding='utf-8-sig')

        print(f"\n结果已保存到 {output_dir}/")


def main():
    """主函数"""
    # 运行回测（使用较短时间范围以加快速度）
    runner = BacktestRunner(
        start_date="20240101",
        end_date="20240401",
        stock_pool="hs300",
        initial_capital=1000000,
        rebalance_months=1
    )

    comparison = runner.run_all()
    return comparison


if __name__ == "__main__":
    main()
