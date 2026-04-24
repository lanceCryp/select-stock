"""
每日买卖信号生成器
==================
基于最优策略 PP>0.25, 持仓4只, 55天调仓

用法:
    python scripts/daily_signal.py [date]

示例:
    python scripts/daily_signal.py 2024-12-02
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta

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

def get_signals(target_date=None):
    """获取指定日期的买卖信号"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"📊 科创板智能选股策略 - 每日信号")
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

    # 获取目标日期的交易日（如果不存在，选择最近的交易日）
    available_dates = sorted(df['date'].unique())
    target_dt = pd.to_datetime(target_date)

    # 找到最近的交易日
    if target_dt > available_dates[-1]:
        target_dt = available_dates[-1]
        print(f"⚠️ 指定日期无数据，使用最近交易日: {target_dt.date()}")
    else:
        # 找到 >= target_date 的第一个交易日
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

    # 追涨过滤: PP > 0.25
    filtered = day_data[day_data['price_position'] > PP_THRESHOLD].copy()

    if len(filtered) == 0:
        print("❌ 没有满足条件的股票（PP>0.25）")
        print("💡 建议: 降低阈值或等待市场机会")
        return None

    # 按预测分数排序
    filtered = filtered.sort_values('lgb_pred', ascending=False)

    # 选取Top 4
    top_stocks = filtered.head(TOP_N).copy()

    print(f"📈 买入信号 (追涨策略 PP>{PP_THRESHOLD})")
    print(f"   选取规则: LGB预测分数 + price_position>{PP_THRESHOLD}")
    print(f"-" * 60)

    signals = []
    for idx, row in top_stocks.iterrows():
        code = row['code']
        name = code  # 数据中无股票名称，用代码代替

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

        print(f"   🟢 买入 {code} {name}")
        print(f"      现价: {row['close']:.2f} | PP: {row['price_position']:.2f} | 分数: {row['lgb_pred']:.4f} | RSI: {row['rsi']:.1f}")

    print(f"\n📋 信号汇总")
    print(f"   买入: {len(signals)} 只")
    print(f"   总仓位: {len(signals) * 25}% (等权分配)")

    # 检查持仓建议
    print(f"\n💡 交易建议")
    print(f"   - 调仓周期: {HOLD_DAYS}个交易日")
    print(f"   - 止损: 无 (回测证明止损无效)")
    print(f"   - 每只股票分配: 25%资金")

    # 保存信号
    import json
    signal_file = f"output/daily_signal_{target_dt.strftime('%Y%m%d')}.json"
    with open(signal_file, 'w') as f:
        json.dump({
            'date': str(target_dt.date()),
            'signals': signals,
            'strategy': {
                'name': 'KCB_LGB_PPV25',
                'top_n': TOP_N,
                'pp_threshold': PP_THRESHOLD,
                'hold_days': HOLD_DAYS
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 信号已保存: {signal_file}")

    return signals

def show_portfolio_tips():
    """显示持仓提示"""
    print(f"\n{'='*60}")
    print("📌 持仓管理提示")
    print("="*60)
    print("""
    买入时点:
    ✅ 当日14:30后确认信号后买入
    ✅ 买入价: 参考close价格
    ✅ 数量: 100股整数倍

    卖出时点:
    ✅ 持有55个交易日后调仓
    ✅ 不在目标名单时次日卖出
    ✅ 不设止损（回测证明无效）

    注意事项:
    ⚠️ 科创板波动大，控制单只仓位≤30%
    ⚠️ 预留约0.1%交易成本
    ⚠️ 避免盘中高频操作
    """)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')

    signals = get_signals(date_str)
    show_portfolio_tips()