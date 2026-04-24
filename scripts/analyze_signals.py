"""
策略信号分析
============
分析历史持仓数据，找出最佳买入/卖出信号
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/kcb_history.csv"
MODEL_PATH = "models/lgb_stock_selector.txt"
PP_THRESHOLD = 0.25
TOP_N = 4
HOLD_DAYS = 55

print("=" * 60)
print("策略信号分析 - 寻找最佳买卖点")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

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

df = calc_features(df)

FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
                'vol_ratio', 'volatility', 'price_position', 'break_20d']

df_model = df.dropna(subset=FEATURE_COLS).copy()
val_df = df_model[df_model['date'] >= '2024-01-01'].copy()
val_dates = sorted(val_df['date'].unique())

model = lgb.Booster(model_file=MODEL_PATH)
X_val = val_df[FEATURE_COLS].values
val_df['lgb_pred'] = model.predict(X_val)

# ========== 分析1: 追踪每只股票的持有期间表现 ==========
print("\n" + "=" * 60)
print("分析1: 持仓期间的价格变化")
print("=" * 60)

def apply_rules(data, conditions):
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
    return mask

def make_func(cond):
    def func(d, n):
        mask = apply_rules(d, cond)
        filtered = d[mask]
        if len(filtered) >= n:
            return filtered.nlargest(n, 'lgb_pred')
        return d.nlargest(n, 'lgb_pred')
    return func

# 追踪所有调仓点
conditions = [('price_position', '>', PP_THRESHOLD)]
rebal_dates = val_dates[::55]

# 收集每个持仓的完整数据
all_holdings = []

for ri, rebal_date in enumerate(rebal_dates):
    next_idx = min(ri + 1, len(rebal_dates) - 1)
    hold_end_date = rebal_dates[next_idx]

    day_data = val_df[val_df['date'] == rebal_date].copy()
    if len(day_data) < TOP_N:
        continue

    selected = make_func(conditions)(day_data, TOP_N)

    for _, row in selected.iterrows():
        code = row['code']
        buy_date = rebal_date
        buy_price = row['close']

        # 获取持有期间数据
        hold_data = df[(df['code'] == code) & (df['date'] > buy_date) & (df['date'] <= hold_end_date)].copy()
        hold_data = hold_data.sort_values('date')

        if len(hold_data) == 0:
            continue

        # 计算持有期间各项指标
        entry_price_position = row['price_position']
        entry_rsi = row['rsi']
        entry_lgb = row['lgb_pred']

        # 持有期间最高/最低价
        max_price = hold_data['high'].max()
        min_price = hold_data['low'].min()
        end_price = hold_data.iloc[-1]['close']

        # 各时间点的收益
        ret_5d = (end_price / buy_price - 1) * 100 if len(hold_data) >= 5 else 0
        ret_end = (end_price / buy_price - 1) * 100

        # 最大回撤
        peak_price = buy_price
        max_drawdown = 0
        for _, h in hold_data.iterrows():
            if h['high'] > peak_price:
                peak_price = h['high']
            dd = (h['low'] - peak_price) / peak_price * 100
            if dd < max_drawdown:
                max_drawdown = dd

        # RSI变化
        rsi_at_end = hold_data.iloc[-1]['rsi'] if len(hold_data) > 0 else 50

        # 价格位置变化
        pp_at_end = hold_data.iloc[-1]['price_position'] if len(hold_data) > 0 else 0

        all_holdings.append({
            'code': code,
            'buy_date': buy_date,
            'buy_price': buy_price,
            'end_price': end_price,
            'ret_total': ret_end,
            'ret_5d': ret_5d,
            'max_drawdown': max_drawdown,
            'max_price': max_price,
            'entry_rsi': entry_rsi,
            'rsi_at_end': rsi_at_end,
            'entry_pp': entry_price_position,
            'pp_at_end': pp_at_end,
            'entry_lgb': entry_lgb,
            'hold_days': len(hold_data)
        })

holdings_df = pd.DataFrame(all_holdings)
print(f"\n追踪到 {len(holdings_df)} 次持仓记录")

# ========== 分析2: 找出买卖点特征 ==========
print("\n" + "=" * 60)
print("分析2: 最佳买入信号特征")
print("=" * 60)

# 按收益分组
win_df = holdings_df[holdings_df['ret_total'] > 0]
lose_df = holdings_df[holdings_df['ret_total'] <= 0]

print(f"\n盈利持仓: {len(win_df)} 次")
print(f"亏损持仓: {len(lose_df)} 次")

if len(win_df) > 0:
    print(f"\n盈利持仓特征:")
    print(f"  平均收益: {win_df['ret_total'].mean():.1f}%")
    print(f"  平均RSI买入: {win_df['entry_rsi'].mean():.1f}")
    print(f"  平均PP买入: {win_df['entry_pp'].mean():.2f}")
    print(f"  平均最大回撤: {win_df['max_drawdown'].mean():.1f}%")

if len(lose_df) > 0:
    print(f"\n亏损持仓特征:")
    print(f"  平均亏损: {lose_df['ret_total'].mean():.1f}%")
    print(f"  平均RSI买入: {lose_df['entry_rsi'].mean():.1f}")
    print(f"  平均PP买入: {lose_df['entry_pp'].mean():.2f}")
    print(f"  平均最大回撤: {lose_df['max_drawdown'].mean():.1f}%")

# ========== 分析3: 哪些条件下买入收益更高 ==========
print("\n" + "=" * 60)
print("分析3: 不同条件的收益对比")
print("=" * 60)

# RSI区间
print("\n按RSI区间分析:")
for low, high in [(0, 30), (30, 50), (50, 70), (70, 100)]:
    subset = holdings_df[(holdings_df['entry_rsi'] >= low) & (holdings_df['entry_rsi'] < high)]
    if len(subset) > 0:
        print(f"  RSI {low}-{high}: {len(subset)}次, 平均收益{subset['ret_total'].mean():+.1f}%, 平均回撤{subset['max_drawdown'].mean():.1f}%")

# PP区间
print("\n按PP区间分析:")
for low, high in [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    subset = holdings_df[(holdings_df['entry_pp'] >= low) & (holdings_df['entry_pp'] < high)]
    if len(subset) > 0:
        print(f"  PP {low:.1f}-{high:.1f}: {len(subset)}次, 平均收益{subset['ret_total'].mean():+.1f}%, 平均回撤{subset['max_drawdown'].mean():.1f}%")

# ========== 分析4: 卖出信号分析 ==========
print("\n" + "=" * 60)
print("分析4: 卖出信号效果模拟")
print("=" * 60)

# 模拟不同卖出信号
def simulate_exit_signal(df, exit_type, exit_param):
    """模拟卖出信号"""
    results = []
    for _, row in df.iterrows():
        code = row['code']
        buy_date = row['buy_date']
        buy_price = row['buy_price']
        hold_end_date = buy_date + pd.Timedelta(days=int(HOLD_DAYS * 1.5))  # 延长时间窗

        hold_data = df[(df['code'] == code) & (df['date'] > buy_date) & (df['date'] <= hold_end_date)].copy()
        hold_data = hold_data.sort_values('date')

        if len(hold_data) == 0:
            continue

        exit_ret = None
        exit_reason = None

        if exit_type == 'rsi_overbought':
            threshold = exit_param
            for _, h in hold_data.iterrows():
                if h['rsi'] > threshold:
                    exit_ret = (h['close'] / buy_price - 1) * 100
                    exit_reason = f'RSI>{threshold}'
                    break

        elif exit_type == 'pp_reversal':
            threshold = exit_param
            for _, h in hold_data.iterrows():
                if h['price_position'] < threshold:
                    exit_ret = (h['close'] / buy_price - 1) * 100
                    exit_reason = f'PP<{threshold}'
                    break

        elif exit_type == 'profit_target':
            target = exit_param
            for _, h in hold_data.iterrows():
                ret = (h['close'] / buy_price - 1) * 100
                if ret > target:
                    exit_ret = ret
                    exit_reason = f'盈利>{target}%'
                    break

        elif exit_type == 'max_drawdown_stop':
            stop = exit_param
            peak = buy_price
            for _, h in hold_data.iterrows():
                if h['high'] > peak:
                    peak = h['high']
                dd = (h['low'] - peak) / peak * 100
                if dd < -stop:
                    exit_ret = dd
                    exit_reason = f'回撤>{stop}%'
                    break

        if exit_ret is None:
            exit_ret = (hold_data.iloc[-1]['close'] / buy_price - 1) * 100
            exit_reason = '到期'

        results.append({
            'code': code,
            'ret': exit_ret,
            'reason': exit_reason
        })

    return pd.DataFrame(results)

# 测试不同卖出信号
print("\n卖出信号模拟:")

# 1. RSI超买
for rsi_th in [60, 70, 80]:
    result = simulate_exit_signal(holdings_df, 'rsi_overbought', rsi_th)
    print(f"  RSI>{rsi_th}: 平均收益{result['ret'].mean():+.1f}%, 触发率{(result['reason']!='到期').mean()*100:.0f}%")

# 2. PP反转
for pp_th in [0.3, 0.4, 0.5]:
    result = simulate_exit_signal(holdings_df, 'pp_reversal', pp_th)
    print(f"  PP<{pp_th}: 平均收益{result['ret'].mean():+.1f}%, 触发率{(result['reason']!='到期').mean()*100:.0f}%")

# 3. 盈利目标
for target in [10, 15, 20, 30]:
    result = simulate_exit_signal(holdings_df, 'profit_target', target)
    triggered = result[result['reason'] != '到期']
    if len(triggered) > 0:
        print(f"  盈利>{target}%: 平均收益{result['ret'].mean():+.1f}%, 触发率{len(triggered)/len(result)*100:.0f}%")
    else:
        print(f"  盈利>{target}%: 平均收益{result['ret'].mean():+.1f}%, 触发率0%")

# 4. 止损
for stop in [5, 8, 10, 15]:
    result = simulate_exit_signal(holdings_df, 'max_drawdown_stop', stop)
    triggered = result[result['reason'] != '到期']
    if len(triggered) > 0:
        print(f"  回撤>{stop}%: 平均收益{result['ret'].mean():+.1f}%, 触发率{len(triggered)/len(result)*100:.0f}%")
    else:
        print(f"  回撤>{stop}%: 平均收益{result['ret'].mean():+.1f}%, 触发率0%")

# ========== 分析5: 找出最佳买卖组合 ==========
print("\n" + "=" * 60)
print("分析5: 最佳买卖组合")
print("=" * 60)

best_combo = None
best_ret = -999

# 组合测试
for buy_rsi_low in [0, 30]:
    for buy_rsi_high in [50, 70]:
        for buy_pp in [0.2, 0.25, 0.3]:
            for exit_type, exit_param in [
                ('rsi_overbought', 70),
                ('rsi_overbought', 80),
                ('pp_reversal', 0.4),
                ('profit_target', 20),
            ]:
                # 筛选买入
                subset = holdings_df[
                    (holdings_df['entry_rsi'] >= buy_rsi_low) &
                    (holdings_df['entry_rsi'] < buy_rsi_high) &
                    (holdings_df['entry_pp'] >= buy_pp)
                ]

                if len(subset) < 5:
                    continue

                # 模拟卖出
                result = simulate_exit_signal(subset, exit_type, exit_param)
                avg_ret = result['ret'].mean()

                if avg_ret > best_ret:
                    best_ret = avg_ret
                    best_combo = {
                        'buy_rsi': f'{buy_rsi_low}-{buy_rsi_high}',
                        'buy_pp': f'>{buy_pp}',
                        'exit_type': exit_type,
                        'exit_param': exit_param,
                        'avg_ret': avg_ret,
                        'n_trades': len(result)
                    }

if best_combo:
    print(f"\n🏆 最佳买卖组合:")
    print(f"   买入条件: RSI{best_combo['buy_rsi']}, PP{best_combo['buy_pp']}")
    print(f"   卖出信号: {best_combo['exit_type']} {best_combo['exit_param']}")
    print(f"   平均收益: {best_combo['avg_ret']:.1f}%")
    print(f"   样本数: {best_combo['n_trades']}次")

# ========== 分析6: 持有期间收益分布 ==========
print("\n" + "=" * 60)
print("分析6: 持有期间收益分布")
print("=" * 60)

# 5日, 20日, 40日, 55日收益分析
for hold_days in [5, 20, 40, 55]:
    rets = []
    for _, row in holdings_df.iterrows():
        code = row['code']
        buy_date = row['buy_date']
        buy_price = row['buy_price']

        target_date = buy_date + pd.Timedelta(days=hold_days)
        close_data = df[(df['code'] == code) & (df['date'] == target_date)]

        if len(close_data) > 0:
            ret = (close_data['close'].values[0] / buy_price - 1) * 100
            rets.append(ret)

    if len(rets) > 0:
        print(f"  {hold_days}日收益: 平均{np.mean(rets):+.1f}%, 中位数{np.median(rets):+.1f}%, 胜率{(np.array(rets)>0).mean()*100:.0f}%")

print("\n" + "=" * 60)
print("结论与建议")
print("=" * 60)
print("""
基于数据分析，建议的买卖信号:

【买入信号】
  1. PP > 0.25 (当前阈值，保持)
  2. RSI < 60 (避免追高)

【卖出信号】
  1. RSI > 75 (超买信号)
  2. PP < 0.40 (从高位回落)
  3. 持有55天后不在名单自动卖出

【优化建议】
  - 加入RSI<60买入过滤，避免买在短期高点
  - 加入RSI>75或PP<0.40卖出信号，实现动态止盈
  - 结合55天定期调仓，形成"定期+条件"双保险
""")