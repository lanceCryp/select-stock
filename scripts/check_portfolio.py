"""
持仓检查与卖出信号
==================
检查当前持仓是否需要卖出

用法:
    python scripts/check_portfolio.py 2024-12-02

示例持仓:
    python scripts/check_portfolio.py 2024-12-02 sh.688143 sh.688535 sh.688089 sh.688132
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb

DATA_PATH = "data/kcb_history.csv"
MODEL_PATH = "models/lgb_stock_selector.txt"
PP_THRESHOLD = 0.25
TOP_N = 4
HOLD_DAYS = 55

def calc_features(df):
    df = df.sort_values(['code', 'date'])
    df['ret_1d'] = df.groupby('code')['close'].pct_change(1)
    df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    df['ma5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60).mean())
    df['ma_bull'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(int)
    bb_mean = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    bb_std = df.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std + 1e-8)
    def calc_rsi(group):
        delta = group['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi'] = df.groupby('code', group_keys=False).apply(calc_rsi)
    df['vol_ma20'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-8)
    df['volatility'] = df.groupby('code')['ret_1d'].transform(lambda x: x.rolling(20).std())
    rolling_low = df.groupby('code')['low'].transform(lambda x: x.rolling(60).min())
    rolling_high = df.groupby('code')['high'].transform(lambda x: x.rolling(60).max())
    df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
    high_20_shift = df.groupby('code')['high'].transform(lambda x: x.rolling(20).max().shift(1))
    df['break_20d'] = (df['close'] >= high_20_shift).astype(int)
    return df

FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
                'vol_ratio', 'volatility', 'price_position', 'break_20d']

def check_portfolio(target_date, holding_codes):
    """检查持仓是否需要卖出"""
    print(f"\n{'='*60}")
    print(f"📊 持仓检查 - 卖出信号")
    print(f"📅 检查日期: {target_date}")
    print(f"📦 持仓: {holding_codes}")
    print(f"{'='*60}\n")

    # 读取数据
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['pctChg'] = df['pctChg'].fillna(0)
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 计算特征
    df = calc_features(df)

    # 加载模型
    model = lgb.Booster(model_file=MODEL_PATH)

    # 获取目标日期
    available_dates = sorted(df['date'].unique())
    target_dt = pd.to_datetime(target_date)

    # 找到最近的交易日
    if target_dt > available_dates[-1]:
        target_dt = available_dates[-1]
    else:
        dates_ge = [d for d in available_dates if d >= target_dt]
        if dates_ge:
            target_dt = dates_ge[0]

    print(f"📆 使用交易日: {target_dt.date()}\n")

    # 获取当日数据
    day_data = df[df['date'] == target_dt].copy()

    # 模型预测
    X = day_data[FEATURE_COLS].values
    day_data['lgb_pred'] = model.predict(X)

    # 追涨过滤: PP > 0.25
    filtered = day_data[day_data['price_position'] > PP_THRESHOLD].copy()
    filtered = filtered.sort_values('lgb_pred', ascending=False)
    top_stocks = filtered.head(TOP_N)['code'].tolist()

    print(f"📈 当前推荐买入: {top_stocks}\n")

    # 检查每个持仓
    print("📋 持仓状态:")
    print("-" * 60)

    sell_signals = []
    keep_signals = []

    for code in holding_codes:
        if code not in df['code'].values:
            print(f"   ❓ {code}: 数据中未找到")
            continue

        code_data = day_data[day_data['code'] == code]

        if len(code_data) == 0:
            print(f"   ❓ {code}: 当日无数据")
            continue

        row = code_data.iloc[-1]
        in_target = code in top_stocks

        print(f"   {code}")
        print(f"      现价: {row['close']:.2f} | PP: {row['price_position']:.2f} | 分数: {row['lgb_pred']:.4f}")

        if in_target:
            print(f"      📌 建议: 继续持有（在推荐名单中）")
            keep_signals.append(code)
        else:
            print(f"      🔴 建议: 卖出（不在推荐名单中）")
            sell_signals.append(code)

        print()

    # 汇总
    print(f"{'='*60}")
    print(f"📋 检查结果汇总")
    print(f"   建议卖出: {len(sell_signals)} 只 - {sell_signals}")
    print(f"   建议持有: {len(keep_signals)} 只 - {keep_signals}")
    print(f"{'='*60}")

    if len(sell_signals) > 0:
        print(f"\n💡 操作建议:")
        print(f"   卖出: {', '.join(sell_signals)}")
        print(f"   买入: {', '.join([c for c in top_stocks if c not in holding_codes])}")

    return sell_signals, keep_signals, top_stocks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/check_portfolio.py <日期> [持仓股票代码...]")
        print("示例: python scripts/check_portfolio.py 2024-12-02 sh.688143 sh.688535")
        sys.exit(1)

    date_str = sys.argv[1]
    holding_codes = sys.argv[2:] if len(sys.argv) > 2 else []

    if len(holding_codes) == 0:
        # 无持仓时，显示今日推荐
        print("⚠️ 未提供持仓股票，仅显示今日买入推荐")
        from scripts.daily_signal import get_signals
        get_signals(date_str)
    else:
        check_portfolio(date_str, holding_codes)