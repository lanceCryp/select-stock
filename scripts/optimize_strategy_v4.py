"""
策略优化迭代 v4
==============
深入优化：止损、仓位、规则组合
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/kcb_history.csv"
MODEL_PATH = "models/lgb_stock_selector.txt"
INITIAL_CASH = 1000000.0
COMMISSION = 0.0003

print("=" * 60)
print("策略优化迭代 v4 - 深入优化")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# 计算特征
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

def apply_rules(data, conditions):
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
    return mask

def backtest(data, dates, strategy_func, top_n=4, stop_loss=0, position_size=1.0):
    """增强版回测：支持止损和仓位"""
    cash = INITIAL_CASH * position_size
    positions = {}
    portfolio_values = []

    for i, rd in enumerate(dates):
        next_idx = min(i + 1, len(dates) - 1)
        hold_end_date = dates[next_idx]

        day_data = data[data['date'] == rd].copy()
        if len(day_data) < top_n:
            continue

        selected = strategy_func(day_data, top_n)
        target_codes = selected['code'].tolist()

        # 止损
        if stop_loss > 0 and positions:
            for code, pos in list(positions.items()):
                price_row = day_data[day_data['code'] == code]
                if len(price_row) > 0:
                    current_price = price_row['close'].values[0]
                    loss = (current_price - pos['buy_price']) / pos['buy_price']
                    if loss < -stop_loss:
                        proceeds = pos['shares'] * current_price * (1 - COMMISSION * 2)
                        cash += proceeds
                        del positions[code]

        # 卖出
        for code, pos in list(positions.items()):
            if code not in target_codes:
                sell_row = day_data[day_data['code'] == code]
                if len(sell_row) > 0:
                    proceeds = pos['shares'] * sell_row['close'].values[0] * (1 - COMMISSION * 2)
                    cash += proceeds
                    del positions[code]

        # 买入
        if len(target_codes) > 0:
            alloc = cash / len(target_codes)
        else:
            alloc = 0

        for code in target_codes:
            buy_row = day_data[day_data['code'] == code]
            if len(buy_row) == 0:
                continue
            buy_price = buy_row['close'].values[0]
            shares = int(alloc / buy_price / 100) * 100
            if shares > 0:
                cost = shares * buy_price * (1 + COMMISSION)
                if cost <= cash:
                    cash -= cost
                    positions[code] = {'shares': shares, 'buy_price': buy_price}

        # 计算价值
        hold_end_data = data[data['date'] == hold_end_date]
        total_value = cash
        for code, pos in positions.items():
            end_row = hold_end_data[hold_end_data['code'] == code]
            if len(end_row) > 0:
                total_value += pos['shares'] * end_row['close'].values[0]
            else:
                total_value += pos['shares'] * pos['buy_price']

        portfolio_values.append({'date': hold_end_date, 'value': total_value})

    if not portfolio_values:
        return None

    pv = pd.DataFrame(portfolio_values)
    pv['ret'] = pv['value'].pct_change().fillna(0)

    total = (pv['value'].iloc[-1] / (INITIAL_CASH * position_size) - 1)
    years = len(pv) * 55 / 252
    annual = ((pv['value'].iloc[-1] / (INITIAL_CASH * position_size)) ** (1 / max(years, 0.1)) - 1)
    std = pv['ret'].std()
    sharpe = pv['ret'].mean() / std * np.sqrt(252 / 55) if std > 0 else 0
    win = (pv['ret'] > 0).mean()
    max_dd = (pv['value'] / pv['value'].cummax() - 1).min()

    return {
        'total': total, 'annual': annual, 'sharpe': sharpe,
        'win_rate': win, 'max_drawdown': max_dd,
        'n': len(pv), 'final_value': pv['value'].iloc[-1],
        'portfolio': pv
    }

results = []

# 最优基准：4只+55天
print("\n基准测试: 4只+55天")
rebal_dates = val_dates[::55]
strategy = lambda d, n: d.nlargest(n, 'lgb_pred')
result = backtest(val_df, rebal_dates, strategy, top_n=4)
if result:
    results.append({'name': '基准LGB4只+55天', **result})
    print(f"  基准: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 1. 测试不同止损
print("\n1. LGB4只+55天 + 止损")
for stop_loss in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    result = backtest(val_df, rebal_dates, strategy, top_n=4, stop_loss=stop_loss)
    if result:
        results.append({'name': f'止损{stop_loss*100:.0f}%', **result})
        print(f"  止损{stop_loss*100:.0f}%: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}% | 回撤 {result['max_drawdown']*100:.1f}%")

# 2. 测试不同RSI规则过滤
print("\n2. RSI规则过滤 + LGB")
rsi_conditions = [
    [('rsi', '<', 70)],
    [('rsi', '<', 60)],
    [('rsi', '<', 50)],
    [('rsi', '<', 40)],
    [('rsi', '>', 30)],
    [('rsi', '>', 40)],
]
for conditions in rsi_conditions:
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func
    result = backtest(val_df, rebal_dates, make_func(conditions), top_n=4)
    if result:
        cond_str = '&'.join([f"{c[0]}{c[1]}{c[2]}" for c in conditions])
        results.append({'name': f'RSI{cond_str[:12]}', **result})
        print(f"  RSI{cond_str[:15]}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 3. 测试价格位置规则
print("\n3. 价格位置规则 + LGB")
pp_conditions = [
    [('price_position', '<', 0.7)],
    [('price_position', '<', 0.6)],
    [('price_position', '<', 0.5)],
    [('price_position', '<', 0.4)],
    [('price_position', '>', 0.2)],
    [('price_position', '>', 0.3)],
]
for conditions in pp_conditions:
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func
    result = backtest(val_df, rebal_dates, make_func(conditions), top_n=4)
    if result:
        cond_str = '&'.join([f"{c[0]}{c[1]}{c[2]}" for c in conditions])
        results.append({'name': f'PP{cond_str[:12]}', **result})
        print(f"  PP{cond_str[:15]}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 4. 测试波动率规则
print("\n4. 波动率规则 + LGB")
vol_conditions = [
    [('volatility', '>', 0.01)],
    [('volatility', '>', 0.015)],
    [('volatility', '<', 0.03)],
    [('volatility', '<', 0.02)],
]
for conditions in vol_conditions:
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func
    result = backtest(val_df, rebal_dates, make_func(conditions), top_n=4)
    if result:
        cond_str = '&'.join([f"{c[0]}{c[1]}{c[2]}" for c in conditions])
        results.append({'name': f'VOL{cond_str[:12]}', **result})
        print(f"  VOL{cond_str[:15]}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 5. 组合规则
print("\n5. 组合规则 + LGB")
combo_conditions = [
    [('rsi', '<', 60), ('price_position', '<', 0.6)],
    [('rsi', '<', 70), ('price_position', '<', 0.5)],
    [('volatility', '>', 0.01), ('rsi', '<', 60)],
    [('rsi', '<', 50), ('volatility', '<', 0.03)],
]
for conditions in combo_conditions:
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func
    result = backtest(val_df, rebal_dates, make_func(conditions), top_n=4)
    if result:
        cond_str = '+'.join([f"{c[0]}{c[1]}{c[2]}" for c in conditions])
        results.append({'name': f'组合{cond_str[:15]}', **result})
        print(f"  组合{cond_str[:20]}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 6. 测试持仓3-6只
print("\n6. 不同持仓数量 + 55天")
for top_n in [3, 4, 5, 6]:
    result = backtest(val_df, rebal_dates, strategy, top_n=top_n)
    if result:
        results.append({'name': f'LGB{top_n}只', **result})
        print(f"  LGB{top_n}只: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 7. 测试周期50-65天
print("\n7. LGB4只 + 不同周期")
for hold_days in [45, 50, 55, 60, 65]:
    rd = val_dates[::hold_days]
    if len(rd) < 3:
        continue
    result = backtest(val_df, rd, strategy, top_n=4)
    if result:
        results.append({'name': f'LGB4只+{hold_days}天', 'hold_days': hold_days, **result})
        print(f"  LGB4只+{hold_days}天: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 排序输出
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)

results.sort(key=lambda x: x['annual'], reverse=True)

print(f"\n{'排名':<4} {'策略':<28} {'年化':>10} {'夏普':>8} {'胜率':>8} {'最大回撤':>10}")
print("-" * 75)

for i, r in enumerate(results[:20], 1):
    print(f"{i:<4} {r['name']:<28} {r['annual']*100:>9.1f}% {r['sharpe']:>8.2f} "
          f"{r['win_rate']*100:>7.1f}% {r['max_drawdown']*100:>9.2f}%")

best = results[0]
print(f"\n🏆 最优策略: {best['name']}")
print(f"   年化: {best['annual']*100:+.1f}%")
print(f"   夏普: {best['sharpe']:.2f}")
print(f"   胜率: {best['win_rate']*100:.0f}%")
print(f"   最大回撤: {best['max_drawdown']*100:.2f}%")
print(f"   最终资金: {best['final_value']:,.0f}")

# 打印每期收益
print(f"\n📅 最优策略每期收益:")
pv = best['portfolio']
cumret = 1.0
for _, row in pv.iterrows():
    cumret *= (1 + row['ret'])
    marker = "🟢" if row['ret'] > 0 else "🔴"
    print(f"   {marker} {row['date'].date()}: {row['ret']*100:>+7.2f}%  累计: {cumret*100:>8.1f}%  价值: {row['value']:>12,.0f}")

# 保存
import json
with open('output/best_strategy_v4.json', 'w') as f:
    json.dump({
        'strategy': best['name'],
        'annual_return': float(best['annual']),
        'sharpe': float(best['sharpe']),
        'win_rate': float(best['win_rate']),
        'max_drawdown': float(best['max_drawdown']),
        'total_return': float(best['total']),
        'final_value': float(best['final_value']),
    }, f, indent=2, ensure_ascii=False)
print(f"\n✅ 结果已保存: output/best_strategy_v4.json")