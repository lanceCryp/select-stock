"""
策略优化迭代 v2
==============
聚焦最优参数：持仓5只 + 各种条件
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
print("策略优化迭代 v2 - 聚焦持仓5只")
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

def backtest(data, dates, strategy_func, top_n=5, stop_loss=0):
    cash = INITIAL_CASH
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

    total = (pv['value'].iloc[-1] / INITIAL_CASH - 1)
    years = len(pv) * 20 / 252
    annual = ((pv['value'].iloc[-1] / INITIAL_CASH) ** (1 / max(years, 0.1)) - 1)
    std = pv['ret'].std()
    sharpe = pv['ret'].mean() / std * np.sqrt(252 / 20) if std > 0 else 0
    win = (pv['ret'] > 0).mean()
    max_dd = (pv['value'] / pv['value'].cummax() - 1).min()

    return {
        'total': total, 'annual': annual, 'sharpe': sharpe,
        'win_rate': win, 'max_drawdown': max_dd,
        'n': len(pv), 'final_value': pv['value'].iloc[-1],
        'portfolio': pv
    }

results = []

# 测试不同持仓周期（持仓5只）
print("\n测试1: 持仓5只 + 不同周期")
for hold_days in [10, 15, 20, 25, 30, 40, 50, 60]:
    rebal_dates = val_dates[::hold_days]
    strategy = lambda d, n: d.nlargest(n, 'lgb_pred')
    result = backtest(val_df, rebal_dates, strategy, top_n=5)
    if result:
        results.append({
            'name': f'LGB5只+{hold_days}天',
            'hold_days': hold_days,
            'top_n': 5,
            **result
        })
        print(f"  持仓5只+{hold_days}天: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 测试3-10只持仓
print("\n测试2: 持仓3-10只 + 20天周期")
rebal_dates = val_dates[::20]
for top_n in [3, 4, 5, 6, 7, 8, 10]:
    strategy = lambda d, n: d.nlargest(n, 'lgb_pred')
    result = backtest(val_df, rebal_dates, strategy, top_n=top_n)
    if result:
        results.append({
            'name': f'LGB{top_n}只+20天',
            'hold_days': 20,
            'top_n': top_n,
            **result
        })
        print(f"  持仓{top_n}只+20天: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 测试持仓5只 + 不同止损
print("\n测试3: 持仓5只 + 止损")
rebal_dates = val_dates[::20]
for stop_loss in [0.05, 0.10, 0.15, 0.20]:
    strategy = lambda d, n: d.nlargest(n, 'lgb_pred')
    result = backtest(val_df, rebal_dates, strategy, top_n=5, stop_loss=stop_loss)
    if result:
        results.append({
            'name': f'LGB5只+止损{stop_loss*100:.0f}%',
            'stop_loss': stop_loss,
            'top_n': 5,
            **result
        })
        print(f"  持仓5只+止损{stop_loss*100:.0f}%: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 测试规则过滤 + LGB
print("\n测试4: 规则过滤 + LGB")
conditions_list = [
    [('price_position', '<', 0.5)],
    [('price_position', '<', 0.6)],
    [('rsi', '<', 50)],
    [('rsi', '<', 60)],
    [('volatility', '>', 0.01)],
    [('volatility', '>', 0.02)],
]

for conditions in conditions_list:
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func

    result = backtest(val_df, rebal_dates, make_func(conditions), top_n=5)
    if result:
        cond_str = '&'.join([f"{c[0]}{c[1]}{c[2]}" for c in conditions])
        results.append({
            'name': f'规则{cond_str[:15]}+LGB',
            'conditions': cond_str,
            **result
        })
        print(f"  规则{cond_str}+LGB: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 结果排序
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)

results.sort(key=lambda x: x['annual'], reverse=True)

print(f"\n{'排名':<4} {'策略':<25} {'年化':>10} {'夏普':>8} {'胜率':>8} {'最大回撤':>10}")
print("-" * 70)

for i, r in enumerate(results[:15], 1):
    print(f"{i:<4} {r['name']:<25} {r['annual']*100:>9.1f}% {r['sharpe']:>8.2f} "
          f"{r['win_rate']*100:>7.1f}% {r['max_drawdown']*100:>9.2f}%")

best = results[0]
print(f"\n🏆 最优策略: {best['name']}")
print(f"   年化: {best['annual']*100:+.1f}%")
print(f"   夏普: {best['sharpe']:.2f}")
print(f"   胜率: {best['win_rate']*100:.0f}%")
print(f"   最大回撤: {best['max_drawdown']*100:.2f}%")

# 打印最优策略的每期收益
print(f"\n📅 最优策略每期收益:")
pv = best['portfolio']
for _, row in pv.iterrows():
    marker = "🟢" if row['ret'] > 0 else "🔴"
    print(f"   {marker} {row['date'].date()}: {row['ret']*100:>+7.2f}%  价值: {row['value']:>14,.0f}")

# 保存
import json
with open('output/optimize_result_v2.json', 'w') as f:
    json.dump({
        'best_strategy': best['name'],
        'annual_return': float(best['annual']),
        'sharpe': float(best['sharpe']),
        'win_rate': float(best['win_rate']),
        'max_drawdown': float(best['max_drawdown']),
        'total_return': float(best['total']),
    }, f, indent=2)
print("\n结果已保存: output/optimize_result_v2.json")