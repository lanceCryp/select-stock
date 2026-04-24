"""
每日买卖信号生成器 - 进化版
===========================
基于最新进化结果:
- PP阈值: >0.45 (更激进追涨)
- 持仓: 3只 (集中持仓)
- 周期: 55天

用法:
    python scripts/daily_signal_v2.py [date]
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

DATA_PATH = "data/kcb_history.csv"
MODEL_PATH = "models/lgb_stock_selector.txt"

# 进化后的最佳参数
PP_THRESHOLD = 0.45
TOP_N = 3
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

def get_signals(target_date=None):
    """获取指定日期的买卖信号"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"📊 科创板智能选股策略 v2 (进化版)")
    print(f"📅 分析日期: {target_date}")
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

    # 获取目标日期的交易日
    available_dates = sorted(df['date'].unique())
    target_dt = pd.to_datetime(target_date)

    if target_dt > available_dates[-1]:
        target_dt = available_dates[-1]
        print(f"⚠️ 使用最近交易日: {target_dt.date()}")
    else:
        dates_ge = [d for d in available_dates if d >= target_dt]
        if dates_ge:
            target_dt = dates_ge[0]

    print(f"📆 使用交易日: {target_dt.date()}\n")

    # 获取当日数据
    day_data = df[df['date'] == target_dt].copy()

    if len(day_data) == 0:
        print("❌ 当日无数据")
        return None

    # 模型预测
    X = day_data[FEATURE_COLS].values
    day_data['lgb_pred'] = model.predict(X)

    # 追涨过滤: PP > 0.45
    filtered = day_data[day_data['price_position'] > PP_THRESHOLD].copy()

    if len(filtered) == 0:
        print("❌ 没有满足条件的股票（PP>0.45）")
        print("💡 当前市场无强势股，建议观望")
        return None

    # 按预测分数排序
    filtered = filtered.sort_values('lgb_pred', ascending=False)

    # 选取Top 3
    top_stocks = filtered.head(TOP_N).copy()

    print(f"📈 买入信号 (进化版追涨策略)")
    print(f"   选取规则: LGB预测 + PP>{PP_THRESHOLD}")
    print(f"   持仓数量: {TOP_N}只 (集中持仓)")
    print(f"-" * 60)

    signals = []
    for idx, row in top_stocks.iterrows():
        code = row['code']
        name = code

        signal = {
            'date': str(target_dt.date()),
            'code': code,
            'name': name,
            'close': row['close'],
            'price_position': row['price_position'],
            'lgb_score': row['lgb_pred'],
            'rsi': row['rsi'],
            'action': 'BUY'
        }
        signals.append(signal)

        print(f"   🟢 买入 {code}")
        print(f"      现价: {row['close']:.2f}")
        print(f"      PP(价格位置): {row['price_position']:.2f} (越高越强)")
        print(f"      分数: {row['lgb_pred']:.4f}")
        print(f"      RSI: {row['rsi']:.1f}")
        print()

    print(f"\n📋 信号汇总")
    print(f"   买入: {len(signals)} 只")
    print(f"   总仓位: {len(signals) * (100//TOP_N)}% (等权分配)")

    # 交易提示
    print(f"\n💡 交易提示")
    print(f"   - 调仓周期: {HOLD_DAYS}个交易日 (约3个月)")
    print(f"   - 止损: 无 (进化测试证明止损无效)")
    print(f"   - 卖出信号: 不在下次名单时自动卖出")
    print(f"   - 每只股票分配: {100//TOP_N}%资金")

    # 保存信号
    import json
    signal_file = f"output/daily_signal_v2_{target_dt.strftime('%Y%m%d')}.json"
    with open(signal_file, 'w') as f:
        json.dump({
            'date': str(target_dt.date()),
            'signals': signals,
            'strategy': {
                'name': 'KCB_LGB_EVOLVED',
                'version': 'v2',
                'top_n': TOP_N,
                'pp_threshold': PP_THRESHOLD,
                'hold_days': HOLD_DAYS
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 信号已保存: {signal_file}")

    return signals

if __name__ == "__main__":
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')

    get_signals(date_str)