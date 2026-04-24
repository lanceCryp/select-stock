"""
回测引擎模块
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'total_return': f"{self.total_return:.2%}",
            'annual_return': f"{self.annual_return:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'win_rate': f"{self.win_rate:.2%}",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': f"{self.avg_win:.2%}",
            'avg_loss': f"{self.avg_loss:.2%}",
            'profit_factor': f"{self.profit_factor:.2f}",
        }


class BacktestEngine:
    """简单回测引擎"""

    def __init__(self, initial_capital: float = 1000000,
                 commission: float = 0.0003,
                 slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission  # 手续费
        self.slippage = slippage  # 滑点

    def run(self,
            signal_df: pd.DataFrame,
            price_df: pd.DataFrame,
            strategy_name: str = "Strategy") -> BacktestResult:
        """
        运行回测

        Args:
            signal_df: 信号数据，index 为日期，columns 为股票代码，values 为仓位(0-1)
            price_df: 价格数据，index 为日期，columns 为股票代码
            strategy_name: 策略名称

        Returns:
            BacktestResult
        """
        if signal_df.empty or price_df.empty:
            return BacktestResult(strategy_name=strategy_name)

        # 对齐日期
        common_dates = signal_df.index.intersection(price_df.index)
        signal_df = signal_df.loc[common_dates]
        price_df = price_df.loc[common_dates]

        # 初始化
        cash = self.initial_capital
        position = {}  # 持仓 {stock: shares}
        equity_curve = [self.initial_capital]
        daily_returns = []
        trades = []
        wins = 0
        losses = 0
        total_win = 0
        total_loss = 0

        dates = sorted(common_dates)

        for i, date in enumerate(dates):
            # 当日收盘后信号生效，次日开盘交易
            if i >= len(dates) - 1:
                break

            next_date = dates[i + 1]
            next_prices = price_df.loc[next_date]

            # 执行交易信号
            for stock in signal_df.columns:
                if stock not in next_prices.index:
                    continue
                price = next_prices[stock]
                if pd.isna(price) or price <= 0:
                    continue

                signal = signal_df.loc[date, stock] if stock in signal_df.columns else 0

                if signal > 0 and stock not in position:
                    # 买入
                    available_cash = cash
                    if available_cash > 0:
                        shares = int(available_cash * signal / (price * (1 + self.commission + self.slippage)))
                        if shares > 0:
                            cost = shares * price * (1 + self.commission + self.slippage)
                            position[stock] = shares
                            cash -= cost
                            trades.append({
                                'date': date,
                                'action': 'buy',
                                'stock': stock,
                                'price': price,
                                'shares': shares,
                                'cost': cost
                            })

                elif signal == 0 and stock in position:
                    # 卖出
                    shares = position[stock]
                    proceeds = shares * price * (1 - self.commission - self.slippage)
                    cash += proceeds
                    pnl = proceeds - trades[-1]['cost'] if trades else 0
                    if pnl > 0:
                        wins += 1
                        total_win += pnl
                    else:
                        losses += 1
                        total_loss += abs(pnl)
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'stock': stock,
                        'price': price,
                        'shares': shares,
                        'proceeds': proceeds,
                        'pnl': pnl
                    })
                    del position[stock]

            # 计算当日权益
            portfolio_value = cash
            for stock, shares in position.items():
                if stock in next_prices.index:
                    portfolio_value += shares * next_prices[stock]
            equity_curve.append(portfolio_value)

            # 计算每日收益率
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)

        # 平掉所有持仓
        final_date = dates[-1]
        final_prices = price_df.loc[final_date]
        for stock, shares in list(position.items()):
            if stock in final_prices.index:
                price = final_prices[stock]
                if not pd.isna(price) and price > 0:
                    proceeds = shares * price * (1 - self.commission - self.slippage)
                    cash += proceeds
                    pnl = proceeds - trades[-1]['cost'] if trades else 0
                    trades.append({
                        'date': final_date,
                        'action': 'sell',
                        'stock': stock,
                        'price': price,
                        'shares': shares,
                        'proceeds': proceeds,
                        'pnl': pnl,
                        'final': True
                    })

        final_value = cash
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年化收益率
        if len(dates) > 1:
            years = len(dates) / 252
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0

        # 夏普比率
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 最大回撤
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        # 胜率
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0

        # 盈利因子
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=wins,
            losing_trades=losses,
            avg_win=total_win / wins if wins > 0 else 0,
            avg_loss=total_loss / losses if losses > 0 else 0,
            profit_factor=profit_factor,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            trades=trades
        )


def compare_strategies(results: List[BacktestResult]) -> pd.DataFrame:
    """对比多个策略"""
    df = pd.DataFrame([r.to_dict() for r in results])
    df = df.sort_values('annual_return', ascending=False)
    return df
