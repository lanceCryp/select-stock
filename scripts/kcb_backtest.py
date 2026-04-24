"""
科创板选股策略回测系统
运行方式: uv run python scripts/kcb_backtest.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("data")
START_DATE = "2020-01-01"   # 科创板2019年7月开板
END_DATE = "2026-04-22"
TOP_N = 20                   # 每月选出的股票数量
HOLD_DAYS = 20              # 持有天数
INITIAL_CASH = 500000       # 初始资金
COMMISSION = 0.0003         # 佣金（含印花税）

BATCH_SIZE = 100            # 每批下载数量
TEST_MODE = False            # True=只下前20只测试, False=下载全部
TEST_COUNT = 20


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. 数据下载
# ============================================================
def get_last_trading_day():
    d = datetime.now()
    for _ in range(10):
        if d.weekday() < 5:
            return d.strftime('%Y-%m-%d')
        d -= timedelta(days=1)
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


def download_kcb_list():
    """下载科创板股票列表"""
    log("=" * 50)
    log("1. 下载科创板股票列表")
    log("=" * 50)

    last_day = "2026-04-22"  # 固定值，baostock 最新数据日期
    bs.login()
    rs = bs.query_all_stock(day=last_day)
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data, columns=rs.fields)
    kcb = df[df['code'].str.startswith('sh.688')]
    kcb = kcb[~kcb['code_name'].str.contains('ST', na=False)]
    kcb.to_csv(DATA_DIR / "kcb_list.csv", index=False, encoding="utf-8")
    log(f"  科创板股票: {len(kcb)} 只")
    log(f"  保存到: data/kcb_list.csv")
    return kcb['code'].tolist()


def download_kcb_history(codes: list):
    """下载科创板历史行情"""
    log("=" * 50)
    log(f"2. 下载科创板历史行情 ({len(codes)} 只)")
    log("=" * 50)

    if TEST_MODE:
        codes = codes[:TEST_COUNT]
        log(f"  [测试模式] 仅下载前 {TEST_COUNT} 只股票")

    all_data = []
    batch_count = (len(codes) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(batch_count):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(codes))
        batch_codes = codes[start_idx:end_idx]

        log(f"  批次 {batch_idx + 1}/{batch_count}: {start_idx}-{end_idx}")

        bs.login()
        for i, code in enumerate(batch_codes):
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="d",
                adjustflag="3"
            )
            if rs is None or rs.error_code != '0':
                time.sleep(0.02)
                continue

            while rs.next():
                row = rs.get_row_data()
                if row is not None:
                    all_data.append(row)
            time.sleep(0.02)

            if (i + 1) % 50 == 0:
                log(f"    进度: {i + 1}/{len(batch_codes)}")
        bs.logout()

        log(f"  批次完成，累计 {len(all_data)} 条")

    if all_data:
        df = pd.DataFrame(all_data, columns=rs.fields)
        # 强制转换为数值类型，避免字符串比较报错
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df['close'] > 0].copy()
        df = df.sort_values(['code', 'date'])
        df.to_csv(DATA_DIR / "kcb_history.csv", index=False, encoding="utf-8")
        size_mb = len(df) * 50 / 1024 / 1024
        log(f"  保存: data/kcb_history.csv ({len(df)} 条, {size_mb:.1f} MB)")
    return len(all_data)


# ============================================================
# 2. 选股策略
# ============================================================
def calc_returns(df):
    """计算收益率列"""
    df = df.sort_values('date')
    df['ret'] = df.groupby('code')['close'].pct_change()
    df['ret5'] = df.groupby('code')['close'].pct_change(5)
    df['ret20'] = df.groupby('code')['close'].pct_change(20)
    df['vol20'] = df.groupby('code')['ret'].rolling(20).std().reset_index(0, drop=True)
    df['ma5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    df['volume_ma5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean())
    return df


STRATEGIES = {}


def strategy(name):
    """策略注册装饰器"""
    def decorator(func):
        STRATEGIES[name] = func
        return func
    return decorator


@strategy("S01 小市值")
def s01_small_cap(df, date):
    """小市值策略：流通市值最小的股票"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    # 用成交量作为流通市值的代理（价格 * 成交量）
    hist['value'] = hist['close'] * hist['volume']
    hist = hist.sort_values('value')
    scores = pd.Series(1.0 / (hist.index + 1), index=hist.index)
    return scores


@strategy("S02 动量反转")
def s02_momentum(df, date):
    """动量反转策略：近期跌幅大的优先"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    # 过去20日收益率越低（跌得越多）分数越高
    hist['score'] = -hist['ret20'].fillna(0)
    scores = hist['score'].rank()
    return scores


@strategy("S03 趋势动量")
def s03_trend(df, date):
    """趋势动量策略：近期涨幅大的优先"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    hist['score'] = hist['ret20'].fillna(0)
    scores = hist['score'].rank()
    return scores


@strategy("S04 低波动")
def s04_low_vol(df, date):
    """低波动策略：历史波动率最低的"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    hist['score'] = -hist['vol20'].fillna(hist['vol20'].median())
    scores = hist['score'].rank()
    return scores


@strategy("S05 放量突破")
def s05_volume_breakout(df, date):
    """放量突破策略：量能放大 + 站上均线"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    cond1 = hist['volume'] > hist['volume_ma5']
    cond2 = hist['close'] > hist['ma5']
    cond3 = hist['close'] > hist['ma20']
    score = cond1.astype(float) + cond2.astype(float) + cond3.astype(float)
    scores = score.rank()
    return scores


@strategy("S06 均线多头")
def s06_ma_bullish(df, date):
    """均线多头排列策略"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    cond1 = hist['close'] > hist['ma5']
    cond2 = hist['ma5'] > hist['ma20']
    cond3 = hist['ret5'] > 0
    score = cond1.astype(float) * 2 + cond2.astype(float) + cond3.astype(float)
    scores = score.rank()
    return scores


@strategy("S07 RSI超卖")
def s07_rsi(df, date):
    """RSI超卖策略"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)

    def calc_rsi(series, n=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - 100 / (1 + rs)

    hist['rsi'] = df.groupby('code').apply(
        lambda g: calc_rsi(g['close'])
    ).reset_index(0, drop=True)
    hist['score'] = -hist['rsi'].fillna(50)  # RSI越低越好
    scores = hist['score'].rank()
    return scores


@strategy("S08 高弹性")
def s08_elasticity(df, date):
    """高弹性策略：波动大但趋势向上的"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    vol = hist['vol20'].fillna(0.05)
    ret = hist['ret20'].fillna(0)
    hist['score'] = ret / (vol + 0.001)  # 夏普比-like
    scores = hist['score'].rank()
    return scores


@strategy("S09 量价齐升")
def s09_volume_price(df, date):
    """量价齐升策略"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)
    # 价格上涨 + 成交量放大
    ret = hist['ret5'].fillna(0)
    vol_ratio = hist['volume'] / (hist['volume_ma5'] + 1)
    hist['score'] = ret * 0.4 + (vol_ratio - 1) * 0.6
    scores = hist['score'].rank()
    return scores


@strategy("S10 突破20日高点")
def s10_breakout_high(df, date):
    """突破20日新高策略"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)

    def max20(series):
        return series.rolling(20).max().shift(1)

    hist['high20'] = df.groupby('code')['high'].transform(max20)
    score = (hist['close'] - hist['high20']) / (hist['high20'] + 0.001)
    hist['score'] = score
    scores = hist['score'].rank()
    return scores


@strategy("S11 缩量整理")
def s11_volume_shrink(df, date):
    """缩量整理策略：波动降低后选择"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)

    def vol_trend(series):
        return series.rolling(5).mean() / (series.rolling(20).mean() + 1)

    hist['vol_trend'] = df.groupby('code')['volume'].transform(vol_trend)
    hist['price_stable'] = -df.groupby('code')['close'].transform(lambda x: x.pct_change().std())
    hist['score'] = -hist['vol_trend'].fillna(1) + hist['price_stable'].fillna(0)
    scores = hist['score'].rank()
    return scores


@strategy("S12 开盘买入")
def s12_gap_up(df, date):
    """跳空高开策略：昨日涨停或跳空"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)

    def prev_close(code_df):
        return code_df.set_index('date')['close'].shift(1)

    prev = df.groupby('code').apply(prev_close).reset_index(0, drop=True)
    hist = hist.join(prev, on=['code', 'date'], rsuffix='_prev')
    hist['gap'] = (hist['close'] - hist['close_prev']) / (hist['close_prev'] + 0.001)
    hist['score'] = hist['gap'].fillna(0)
    scores = hist['score'].rank()
    return scores


@strategy("S13 综合评分")
def s13_composite(df, date):
    """综合评分策略：多个维度打分"""
    hist = df[df['date'] == date].copy()
    if len(hist) < 5:
        return pd.Series(dtype=float)

    # 多个维度标准化后加总
    hist['mom'] = hist['ret20'].fillna(0).rank(pct=True)
    hist['vol'] = (-hist['vol20'].fillna(0.05)).rank(pct=True)
    hist['vol_r'] = (hist['volume'] / (hist['volume_ma5'] + 1)).rank(pct=True)
    hist['trend'] = (hist['close'] / (hist['ma20'] + 1) - 1).rank(pct=True)
    hist['score'] = hist['mom'] * 0.3 + hist['vol'] * 0.2 + hist['vol_r'] * 0.2 + hist['trend'] * 0.3
    scores = hist['score'].rank()
    return scores


# ============================================================
# 3. 回测引擎
# ============================================================
def run_backtest(df, strategy_name, top_n=TOP_N, hold_days=HOLD_DAYS):
    """月度调仓回测"""
    dates = sorted(df['date'].unique())
    dates = [d for d in dates if d >= pd.Timestamp(START_DATE) and d >= pd.Timestamp("2020-06-01")]

    cash = INITIAL_CASH
    positions = {}   # {code: {'shares': n, 'cost': price}}
    trades = []
    portfolio_values = []

    for i in range(0, len(dates) - hold_days, hold_days):
        rebal_date = dates[i]
        end_date = dates[min(i + hold_days, len(dates) - 1)]

        # 计算评分
        try:
            scores = STRATEGIES[strategy_name](df, rebal_date)
        except Exception:
            continue

        if len(scores) == 0:
            continue

        # 选取评分最高的 top_n 只
        top = scores.nlargest(top_n).index
        target_codes = df.loc[top, 'code'].unique()[:top_n]
        target_codes = [c for c in target_codes if c in df['code'].values]

        # 卖出不在目标中的持仓
        for code, pos in list(positions.items()):
            if code not in target_codes:
                sell_price = df[(df['date'] == rebal_date) & (df['code'] == code)]['close'].values
                if len(sell_price) > 0:
                    proceeds = pos['shares'] * sell_price[0] * (1 - COMMISSION * 2)
                    cash += proceeds
                    trades.append({
                        'date': rebal_date, 'action': 'SELL', 'code': code,
                        'shares': pos['shares'], 'price': sell_price[0]
                    })
                    del positions[code]

        # 等权买入
        if len(target_codes) > 0:
            alloc = cash / len(target_codes)
        else:
            alloc = 0

        for code in target_codes:
            price_row = df[(df['date'] == rebal_date) & (df['code'] == code)]['close'].values
            if len(price_row) == 0:
                continue
            price = price_row[0]
            shares = int(alloc / price / 100) * 100  # 整手
            if shares > 0:
                cost = shares * price * (1 + COMMISSION)
                if cost <= cash:
                    cash -= cost
                    positions[code] = {'shares': shares, 'cost': price}
                    trades.append({
                        'date': rebal_date, 'action': 'BUY', 'code': code,
                        'shares': shares, 'price': price
                    })

        # 记录组合价值
        total_value = cash
        for code, pos in positions.items():
            price_row = df[(df['date'] == end_date) & (df['code'] == code)]['close'].values
            if len(price_row) > 0:
                total_value += pos['shares'] * price_row[0]
            else:
                total_value += pos['shares'] * pos['cost']

        portfolio_values.append({
            'date': end_date, 'value': total_value,
            'positions': len(positions), 'cash': cash
        })

    if not portfolio_values:
        return None

    pv = pd.DataFrame(portfolio_values)
    pv['ret'] = pv['value'].pct_change()

    total_ret = (pv['value'].iloc[-1] / INITIAL_CASH - 1) * 100
    years = len(pv) * hold_days / 252
    annual_ret = ((pv['value'].iloc[-1] / INITIAL_CASH) ** (1 / max(years, 0.1)) - 1) * 100
    sharpe = pv['ret'].mean() / (pv['ret'].std() + 1e-10) * np.sqrt(252 / hold_days)
    max_drawdown = (pv['value'] / pv['value'].cummax() - 1).min() * 100
    win_rate = (pv['ret'] > 0).sum() / max(len(pv['ret'].dropna()), 1) * 100

    return {
        'strategy': strategy_name,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'final_value': pv['value'].iloc[-1],
        'portfolio_values': pv,
        'trades': pd.DataFrame(trades)
    }


# ============================================================
# 4. 主程序
# ============================================================
def main():
    log("=" * 60)
    log("科创板选股策略回测系统")
    log("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)

    # 检查是否有数据
    kcb_history_path = DATA_DIR / "kcb_history.csv"
    need_download = not kcb_history_path.exists()

    if need_download:
        # 1. 下载股票列表
        codes = download_kcb_list()

        # 2. 下载历史数据
        record_count = download_kcb_history(codes)
        if record_count < 1000:
            log("  数据下载失败或数据量太少，退出")
            return
    else:
        log(f"  使用已有数据: {kcb_history_path}")

    # 3. 加载数据
    log("=" * 50)
    log("3. 加载数据")
    log("=" * 50)
    df = pd.read_csv(kcb_history_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['close'] > 0].copy()
    df = calc_returns(df)

    dates = sorted(df['date'].unique())
    log(f"  股票数量: {df['code'].nunique()}")
    log(f"  数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    log(f"  交易日: {len(dates)} 天")

    # 调试：检查数据
    if len(dates) < 5 or len(df) < 100:
        log(f"  [警告] 数据不足！股票:{df['code'].nunique()} 日期:{len(dates)} 行数:{len(df)}")
        log(f"  示例数据: {df.head(3)[['date','code','close']].to_string()}")

    # 4. 运行回测
    log("=" * 50)
    log("4. 运行回测")
    log(f"   持仓周期: {HOLD_DAYS} 个交易日")
    log(f"   每次持仓: TOP {TOP_N} 只")
    log("=" * 50)

    results = []
    for name, func in STRATEGIES.items():
        print(f"  回测: {name}...", end='', flush=True)
        try:
            result = run_backtest(df, name, top_n=TOP_N, hold_days=HOLD_DAYS)
            if result:
                results.append(result)
                log(f" 年化 {result['annual_ret']:.2f}% | 夏普 {result['sharpe']:.2f} | "
                    f"回撤 {result['max_drawdown']:.1f}% | 胜率 {result['win_rate']:.1f}%")
            else:
                log(" 无结果")
        except Exception as e:
            log(f" 错误: {e}")

    # 5. 输出结果
    log("=" * 60)
    log("回测结果汇总")
    log("=" * 60)
    log(f"{'策略':<20} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8} {'总收益':>10} {'最终资金':>12}")
    log("-" * 80)

    results.sort(key=lambda x: x['annual_ret'], reverse=True)
    for r in results:
        log(f"{r['strategy']:<20} {r['annual_ret']:>10.2f}% {r['sharpe']:>8.2f} "
            f"{r['max_drawdown']:>10.1f}% {r['win_rate']:>8.1f}% "
            f"{r['total_ret']:>10.1f}% {r['final_value']:>12,.0f}")

    log("-" * 80)
    best = results[0]
    log(f"最佳策略: {best['strategy']}")
    log(f"  年化收益: {best['annual_ret']:.2f}%")
    log(f"  夏普比率: {best['sharpe']:.2f}")
    log(f"  最大回撤: {best['max_drawdown']:.1f}%")
    log(f"  胜    率: {best['win_rate']:.1f}%")
    log(f"  最终资金: {best['final_value']:,.0f}")

    # 保存结果
    results_df = pd.DataFrame([{k: v for k, v in r.items()
                                  if k not in ['portfolio_values', 'trades']}
                                 for r in results])
    results_df.to_csv(DATA_DIR / "kcb_backtest_results.csv", index=False, encoding="utf-8")
    log(f"\n结果已保存: data/kcb_backtest_results.csv")


if __name__ == "__main__":
    main()
