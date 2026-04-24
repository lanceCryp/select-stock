"""
LightGBM 特征分析与策略规则挖掘
=================================
目标：从模型中挖掘可操作的交易规则，逐步验证和优化
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据和模型
# ============================================================
print("=" * 60)
print("LightGBM 特征分析与策略规则挖掘")
print("=" * 60)

DATA_PATH = "data/kcb_history.csv"
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

print(f"\n数据: {df['code'].nunique()} 只股票, {len(df):,} 条记录")

# ============================================================
# 2. 计算特征（和 train_lgb.py 一样）
# ============================================================
print("\n2. 计算特征...")

def calc_indicators(group):
    g = group.sort_values('date')
    g['ret_1d'] = g['close'].pct_change(1)
    g['ret_5d'] = g['close'].pct_change(5)
    g['ret_20d'] = g['close'].pct_change(20)
    g['ma5'] = g['close'].rolling(5).mean()
    g['ma20'] = g['close'].rolling(20).mean()
    g['ma60'] = g['close'].rolling(60).mean()
    g['ma_bull'] = ((g['ma5'] > g['ma20']) & (g['ma20'] > g['ma60'])).astype(int)
    bb_mean = g['close'].rolling(20).mean()
    bb_std = g['close'].rolling(20).std()
    g['bb_upper'] = bb_mean + 2 * bb_std
    g['bb_lower'] = bb_mean - 2 * bb_std
    g['bb_position'] = (g['close'] - g['bb_lower']) / (g['bb_upper'] - g['bb_lower'] + 1e-8)
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    g['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    g['vol_ma20'] = g['volume'].rolling(20).mean()
    g['vol_ratio'] = g['volume'] / (g['vol_ma20'] + 1e-8)
    g['volatility'] = g['ret_1d'].rolling(20).std()
    g['price_position'] = (g['close'] - g['low'].rolling(60).min()) / \
                          (g['high'].rolling(60).max() - g['low'].rolling(60).min() + 1e-8)
    g['high_20d'] = g['high'].rolling(20).max()
    g['break_20d'] = (g['close'] >= g['high_20d'].shift(1)).astype(int)
    # 额外特征
    g['ret_5d_std'] = g['ret_1d'].rolling(5).std()  # 5日波动
    g['volume_std'] = g['volume'].pct_change().rolling(5).std()  # 量能波动
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / (g['high'] - g['low'] + 1e-8)  # 上影线比例
    return g

df_code = df[['code']].copy()
df = df.groupby('code', group_keys=False).apply(calc_indicators)
if 'code' not in df.columns:
    df['code'] = df_code['code'].values

# 标签
df['future_return'] = df.groupby('code')['close'].pct_change(20).shift(-20)
FEATURE_COLS = [
    'ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
    'vol_ratio', 'volatility', 'price_position', 'break_20d',
    'ret_5d_std', 'volume_std', 'upper_shadow'
]

df_model = df.dropna(subset=FEATURE_COLS + ['future_return']).copy()

# 验证集（2024开始）
val_df = df_model[df_model['date'] >= '2024-01-01'].copy()
val_df['pred'] = 0.0

# 加载已训练的模型（预测部分暂不需要，直接分析特征条件）
# model = lgb.Booster(model_file='models/lgb_stock_selector.txt')
# X_val = val_df[FEATURE_COLS[:10]].values  # 用训练时的10个特征
# val_df['pred'] = model.predict(X_val)

print(f"   验证集: {len(val_df):,} 条, {val_df['date'].min().date()} ~ {val_df['date'].max().date()}")

# ============================================================
# 3. 单特征条件分析
# ============================================================
print("\n" + "=" * 60)
print("3. 单特征条件分析（验证集 2024-2026）")
print("=" * 60)

def analyze_condition(data, feature, op, threshold, label=None):
    """分析单个条件下的收益"""
    if op == '>':
        mask = data[feature] > threshold
    elif op == '<':
        mask = data[feature] < threshold
    elif op == '>=':
        mask = data[feature] >= threshold
    elif op == '<=':
        mask = data[feature] <= threshold
    elif op == 'between':
        mask = (data[feature] >= threshold[0]) & (data[feature] <= threshold[1])
    
    if label is None:
        label = f"{feature} {op} {threshold}"
    
    n = mask.sum()
    if n < 100:
        return None
    
    avg_ret = data.loc[mask, 'future_return'].mean()
    std_ret = data.loc[mask, 'future_return'].std()
    win_rate = (data.loc[mask, 'future_return'] > 0).mean()
    
    return {
        'condition': label,
        'n': n,
        'avg_return': avg_ret,
        'std': std_ret,
        'sharpe': avg_ret / std_ret * np.sqrt(252/20) if std_ret > 0 else 0,
        'win_rate': win_rate,
        'edge': avg_ret * 100  # 用 bp 作为衡量指标
    }

results = []

# RSI 分析
print("\n📊 RSI 超卖/超买分析:")
rsi_ranges = [(0, 20), (0, 30), (20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 100), (80, 100)]
for lo, hi in rsi_ranges:
    r = analyze_condition(val_df, 'rsi', 'between', (lo, hi), f"RSI in [{lo},{hi}]")
    if r: results.append(r)
    stars = '⭐' * max(0, int((r['avg_return'] / 0.01) if r['avg_return'] > 0 else r['avg_return'] / 0.005))
    print(f"   RSI [{lo:2d},{hi:2d}]: n={r['n']:4d}, 均收益={r['avg_return']*100:+6.2f}%, 胜率={r['win_rate']*100:.1f}% {stars}")

# 波动率分析
print("\n📊 波动率分析:")
for t in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
    r = analyze_condition(val_df, 'volatility', '<', t, f"volatility < {t:.3f}")
    if r: results.append(r)
    print(f"   volatility < {t:.3f}: n={r['n']:4d}, 均收益={r['avg_return']*100:+6.2f}%, 胜率={r['win_rate']*100:.1f}%")

# 价格位置分析
print("\n📊 价格位置分析:")
for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    r = analyze_condition(val_df, 'price_position', 'between', (lo, hi), f"price_pos in [{lo},{hi}]")
    if r: results.append(r)
    print(f"   price_pos [{lo:.1f},{hi:.1f}]: n={r['n']:4d}, 均收益={r['avg_return']*100:+6.2f}%, 胜率={r['win_rate']*100:.1f}%")

# 布林带位置
print("\n📊 布林带位置分析:")
for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    r = analyze_condition(val_df, 'bb_position', 'between', (lo, hi), f"bb_pos in [{lo},{hi}]")
    if r: results.append(r)
    print(f"   bb_pos [{lo:.1f},{hi:.1f}]: n={r['n']:4d}, 均收益={r['avg_return']*100:+6.2f}%, 胜率={r['win_rate']*100:.1f}%")

# 5日收益分析
print("\n📊 5日收益动量分析:")
for lo, hi in [(-0.15, -0.05), (-0.05, -0.02), (-0.02, 0), (0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]:
    r = analyze_condition(val_df, 'ret_5d', 'between', (lo, hi), f"ret_5d in [{lo:.0%},{hi:.0%}]")
    if r: results.append(r)
    print(f"   ret_5d [{lo:.0%},{hi:.0%}]: n={r['n']:4d}, 均收益={r['avg_return']*100:+6.2f}%, 胜率={r['win_rate']*100:.1f}%")

# 成交量比
print("\n📊 成交量比分析:")
for t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    r1 = analyze_condition(val_df, 'vol_ratio', '<', t, f"vol_ratio < {t:.1f}")
    r2 = analyze_condition(val_df, 'vol_ratio', '>', t, f"vol_ratio > {t:.1f}")
    if r1: results.append(r1)
    if r2: results.append(r2)
    print(f"   vol_ratio < {t:.1f}: n={r1['n']:4d}, 均收益={r1['avg_return']*100:+6.2f}%")
    print(f"   vol_ratio > {t:.1f}: n={r2['n']:4d}, 均收益={r2['avg_return']*100:+6.2f}%")

# ============================================================
# 4. 特征组合分析
# ============================================================
print("\n" + "=" * 60)
print("4. 特征组合分析")
print("=" * 60)

def analyze_combination(data, conditions, label=None):
    """分析多条件组合"""
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
        elif op == '>=': mask &= data[feat] >= val
        elif op == '<=': mask &= data[feat] <= val
        elif op == 'between': mask &= (data[feat] >= val[0]) & (data[feat] <= val[1])
    
    n = mask.sum()
    if n < 50:
        return None
    
    avg_ret = data.loc[mask, 'future_return'].mean()
    std_ret = data.loc[mask, 'future_return'].std()
    win_rate = (data.loc[mask, 'future_return'] > 0).mean()
    
    return {
        'condition': label,
        'n': n,
        'avg_return': avg_ret,
        'std': std_ret,
        'sharpe': avg_ret / std_ret * np.sqrt(252/20) if std_ret > 0 else 0,
        'win_rate': win_rate,
        'edge': avg_ret * 100
    }

print("\n🧩 组合条件测试:")
combos = []

# 组合1：RSI超卖 + 低波动
c = analyze_combination(val_df,
    [('rsi', '<', 30), ('volatility', '<', 0.025)],
    "RSI<30 + 低波动")
if c: combos.append(c); print(f"   RSI<30 + volatility<0.025: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

c = analyze_combination(val_df,
    [('rsi', '<', 20), ('volatility', '<', 0.02)],
    "RSI<20 + 极低波动")
if c: combos.append(c); print(f"   RSI<20 + volatility<0.02: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合2：价格低位 + RSI 超卖
c = analyze_combination(val_df,
    [('price_position', '<', 0.3), ('rsi', '<', 40)],
    "价格低位 + RSI 超卖")
if c: combos.append(c); print(f"   price_pos<0.3 + RSI<40: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合3：连续下跌后反弹
c = analyze_combination(val_df,
    [('ret_5d', 'between', (-0.10, -0.03)), ('rsi', '<', 35)],
    "5日跌3-10% + RSI<35")
if c: combos.append(c); print(f"   ret_5d[-10%,-3%] + RSI<35: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合4：突破20日高点 + 放量
c = analyze_combination(val_df,
    [('break_20d', '>=', 1), ('vol_ratio', '>', 1.2)],
    "突破20日高点 + 放量")
if c: combos.append(c); print(f"   突破20日高点 + vol_ratio>1.2: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合5：布林下轨附近 + 超卖
c = analyze_combination(val_df,
    [('bb_position', '<', 0.2), ('rsi', '<', 30)],
    "布林下轨 + RSI<30")
if c: combos.append(c); print(f"   bb_pos<0.2 + RSI<30: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合6：价格低位 + 放量
c = analyze_combination(val_df,
    [('price_position', '<', 0.25), ('vol_ratio', '>', 1.5)],
    "价格低位 + 放量>1.5x")
if c: combos.append(c); print(f"   price_pos<0.25 + vol_ratio>1.5: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合7：5日均收益为负但不大 + RSI在中间
c = analyze_combination(val_df,
    [('ret_5d', 'between', (-0.03, 0)), ('rsi', 'between', (40, 60))],
    "小幅调整 + 中性RSI")
if c: combos.append(c); print(f"   ret_5d[-3%,0%] + RSI[40,60]: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合8：高弹性（高波动 + 连续下跌）
c = analyze_combination(val_df,
    [('volatility', '>', 0.03), ('ret_5d', 'between', (-0.15, -0.05))],
    "高波动 + 5日跌5-15%")
if c: combos.append(c); print(f"   高波动 + ret_5d[-15%,-5%]: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合9：均线多头 + 缩量回调
c = analyze_combination(val_df,
    [('ma_bull', '>=', 1), ('vol_ratio', '<', 0.8)],
    "均线多头 + 缩量回调")
if c: combos.append(c); print(f"   ma_bull=1 + vol_ratio<0.8: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# 组合10：上影线过长（冲高回落）
c = analyze_combination(val_df,
    [('upper_shadow', '>', 0.5), ('ret_1d', '>', 0.03)],
    "上影线>50% + 当日涨>3%")
if c: combos.append(c); print(f"   上影线>50% + 今日涨>3%: n={c['n']:4d}, 均收益={c['avg_return']*100:+6.2f}%, 胜率={c['win_rate']*100:.1f}%")

# ============================================================
# 5. 找最优条件组合
# ============================================================
print("\n" + "=" * 60)
print("5. 最优条件组合排名")
print("=" * 60)

all_rules = results + combos
valid_rules = [r for r in all_rules if r is not None and r['n'] >= 50]

# 按收益排序
valid_rules.sort(key=lambda x: x['avg_return'], reverse=True)

print(f"\n{'条件':<35} {'样本':>5} {'均收益':>8} {'夏普':>6} {'胜率':>6}")
print("-" * 65)
for r in valid_rules[:15]:
    flag = "✅" if r['avg_return'] > 0 else "❌"
    print(f"   {flag} {r['condition']:<33} {r['n']:>5} {r['avg_return']*100:>+7.2f}% {r['sharpe']:>+6.2f} {r['win_rate']*100:>5.1f}%")

# ============================================================
# 6. 编写策略并回测
# ============================================================
print("\n" + "=" * 60)
print("6. 策略回测（按最优规则）")
print("=" * 60)

REBALANCE_DAYS = 20
TOP_N = 20

val_dates = sorted(val_df['date'].unique())
rebalance_dates = [val_dates[i] for i in range(0, len(val_dates), REBALANCE_DAYS)]

def run_strategy(data, conditions, name):
    """给定条件，运行策略回测"""
    P = 1000000  # 初始资金 100万
    portfolio_value = P
    trades = 0
    wins = 0
    period_returns = []
    
    for rd in rebalance_dates:
        day_data = data[data['date'] == rd].copy()
        if len(day_data) < TOP_N:
            continue
        
        # 应用条件
        mask = pd.Series(True, index=day_data.index)
        for feat, op, val in conditions:
            if op == '>': mask &= day_data[feat] > val
            elif op == '<': mask &= day_data[feat] < val
            elif op == '>=': mask &= day_data[feat] >= val
            elif op == '<=': mask &= day_data[feat] <= val
            elif op == 'between': mask &= (day_data[feat] >= val[0]) & (day_data[feat] <= val[1])
        
        candidates = day_data[mask].sort_values('future_return', ascending=False)
        
        if len(candidates) == 0:
            # 没有符合条件的，换手为空仓
            period_returns.append(0)
            continue
        
        selected = candidates.head(TOP_N)
        ret = selected['future_return'].mean()
        period_returns.append(ret)
        
        if len(selected) > 0:
            wins += (ret > 0)
            trades += 1
    
    if not period_returns:
        return None
    
    period_returns = np.array(period_returns)
    total_ret = (1 + period_returns).prod() - 1
    # 年化：用平均周期收益 × 年化系数（不用几何平均，避免空仓期干扰）
    avg_ret = period_returns.mean()
    periods_per_year = 252 / REBALANCE_DAYS
    annual_ret = avg_ret * periods_per_year
    sharpe = period_returns.mean() / period_returns.std() * np.sqrt(periods_per_year) if period_returns.std() > 0 else 0
    win_rate = (period_returns > 0).mean()
    
    return {
        'name': name,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': trades,
        'periods': len(period_returns)
    }

# 测试最优组合
strategies_to_test = [
    ("RSI<30 + 低波动", [('rsi', '<', 30), ('volatility', '<', 0.025)]),
    ("价格低位 + RSI<40", [('price_position', '<', 0.3), ('rsi', '<', 40)]),
    ("5日跌3-10% + RSI<35", [('ret_5d', 'between', (-0.10, -0.03)), ('rsi', '<', 35)]),
    ("布林下轨 + RSI<30", [('bb_position', '<', 0.2), ('rsi', '<', 30)]),
    ("均线多头 + 缩量", [('ma_bull', '>=', 1), ('vol_ratio', '<', 0.8)]),
    ("价格低位 + 放量", [('price_position', '<', 0.25), ('vol_ratio', '>', 1.5)]),
    ("突破20日高点 + 放量", [('break_20d', '>=', 1), ('vol_ratio', '>', 1.2)]),
]

print(f"\n{'策略名称':<30} {'总收益':>8} {'年化':>8} {'夏普':>6} {'胜率':>6} {'交易次数':>7}")
print("-" * 75)

strategy_results = []
for name, conds in strategies_to_test:
    res = run_strategy(val_df, conds, name)
    if res:
        strategy_results.append(res)
        print(f"   {res['name']:<28} {res['total_ret']*100:>+7.2f}% {res['annual_ret']*100:>+7.2f}% {res['sharpe']:>+6.2f} {res['win_rate']*100:>5.1f}% {res['trades']:>6d}")

# 找最优策略
if strategy_results:
    best = max(strategy_results, key=lambda x: x['annual_ret'])
    print(f"\n🏆 最优策略: {best['name']}")
    print(f"   年化收益: {best['annual_ret']*100:.2f}%")
    print(f"   夏普比率: {best['sharpe']:.2f}")
    print(f"   胜率: {best['win_rate']*100:.1f}%")

print("\n✅ 分析完成！根据上方结果，选择最优条件组合作为策略基础。")
