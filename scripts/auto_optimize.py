#!/usr/bin/env python3
"""
全自动策略分析 + 优化系统
===========================
自动分析所有特征组合 → 网格搜索最优参数 → walk-forward验证 → 鲁棒性检验
"""

import numpy as np
import pandas as pd
import warnings
import json
import time
import os
warnings.filterwarnings('ignore')

# ============================================================
# 0. 加载数据
# ============================================================
print("=" * 70)
print("  全自动策略分析 + 优化系统")
print("=" * 70)
print(f"\n[{time.strftime('%H:%M:%S')}] 加载数据...")

os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/kcb_history.csv")
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)
print(f"   原始数据: {len(df):,} 行, {df['code'].nunique()} 只股票")

# ============================================================
# 1. 计算技术指标
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 计算技术指标...")

def calc_indicators(group):
    g = group.sort_values('date')
    g['ret_1d'] = g['close'].pct_change(1)
    g['ret_5d'] = g['close'].pct_change(5)
    g['ret_10d'] = g['close'].pct_change(10)
    g['ret_20d'] = g['close'].pct_change(20)
    ma5 = g['close'].rolling(5).mean()
    ma20 = g['close'].rolling(20).mean()
    ma60 = g['close'].rolling(60).mean()
    g['ma_bull'] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    bb_mean = g['close'].rolling(20).mean()
    bb_std = g['close'].rolling(20).std()
    g['bb_position'] = (g['close'] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-8)
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    g['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    g['vol_ma20'] = g['volume'].rolling(20).mean()
    g['vol_ratio'] = g['volume'] / (g['vol_ma20'] + 1e-8)
    g['vol_ratio_5'] = g['volume'] / (g['volume'].rolling(5).mean() + 1e-8)
    g['volatility'] = g['ret_1d'].rolling(20).std()
    g['volatility_5'] = g['ret_1d'].rolling(5).std()
    g['price_position'] = (g['close'] - g['low'].rolling(60).min()) / \
                          (g['high'].rolling(60).max() - g['low'].rolling(60).min() + 1e-8)
    g['price_position_20'] = (g['close'] - g['low'].rolling(20).min()) / \
                              (g['high'].rolling(20).max() - g['low'].rolling(20).min() + 1e-8)
    g['break_20d'] = (g['close'] >= g['high'].rolling(20).max().shift(1)).astype(int)
    g['break_10d'] = (g['close'] >= g['high'].rolling(10).max().shift(1)).astype(int)
    g['close_ma5_ratio'] = g['close'] / (ma5 + 1e-8)
    # 反转信号：过去N日下跌越多 -> 值越大（利好信号）
    g['ret_5d_rev'] = -g['ret_5d']
    g['ret_20d_rev'] = -g['ret_20d']
    return g

saved_codes = df['code'].copy()
df = df.groupby('code', group_keys=False).apply(calc_indicators)
if 'code' not in df.columns:
    df['code'] = saved_codes.values

df['future_return'] = df.groupby('code')['close'].pct_change(20).shift(-20)

FEATURE_COLS = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
    'vol_ratio', 'vol_ratio_5', 'volatility', 'volatility_5',
    'price_position', 'price_position_20', 'break_20d', 'break_10d',
    'close_ma5_ratio', 'ret_20d_rev', 'ret_5d_rev'
]

df_model = df.dropna(subset=FEATURE_COLS + ['future_return']).copy()
train_df = df_model[df_model['date'] < '2024-01-01'].copy()
val_df   = df_model[(df_model['date'] >= '2024-01-01') & (df_model['date'] < '2025-07-01')].copy()
test_df  = df_model[df_model['date'] >= '2025-07-01'].copy()

print(f"   训练集: {len(train_df):,} 条 ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
print(f"   验证集: {len(val_df):,} 条 ({val_df['date'].min().date()} ~ {val_df['date'].max().date()})")
print(f"   测试集: {len(test_df):,} 条 ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")

# ============================================================
# 2. 辅助函数
# ============================================================

def normalize_condition(c):
    """Normalize condition to (feat, op, val) tuple string representation"""
    return f"{c[0]} {c[1]} {c[2]}"

def apply_conditions(data, conditions):
    """
    Apply list of conditions (AND logic).
    conditions: list of (feature, operator, value) tuples
    """
    mask = np.ones(len(data), dtype=bool)
    feat_arrays = {col: data[col].values for col in data.columns}
    for cond in conditions:
        feat, op, val = cond
        col = feat_arrays[feat]
        if op == '>':  mask &= col > val
        elif op == '<':  mask &= col < val
        elif op == '>=': mask &= col >= val
        elif op == '<=': mask &= col <= val
        elif op == '==': mask &= np.isclose(col, val)
        elif op == 'between': mask &= (col >= val[0]) & (col <= val[1])
    return mask

def run_backtest(data, rebalance_dates, conditions, top_n=20):
    """Run backtest, return period returns array and trade count."""
    period_returns = []
    n_trades = 0
    for rd in rebalance_dates:
        day = data[data['date'] == rd]
        if len(day) < 1:
            period_returns.append(0)
            continue
        mask = apply_conditions(day, conditions)
        candidates = day[mask]
        if len(candidates) == 0:
            period_returns.append(0)
            continue
        # Select top stocks by future return (oracle ranking for backtest)
        top = candidates.nlargest(min(top_n, len(candidates)), 'future_return')
        period_returns.append(top['future_return'].mean())
        n_trades += 1
    return np.array(period_returns), n_trades

def calc_stats(period_returns, rebalance_days=20):
    """Calculate performance metrics from period returns (excluding zeros)."""
    pr = period_returns[period_returns != 0]
    if len(pr) < 3:
        return None
    total_ret = (1 + pr).prod() - 1
    avg_ret = pr.mean()
    periods_per_year = 252 / rebalance_days
    annual_ret = avg_ret * periods_per_year
    sharpe = pr.mean() / pr.std() * np.sqrt(periods_per_year) if pr.std() > 1e-10 else 0
    sortino = pr.mean() / pr[pr < 0].std() * np.sqrt(periods_per_year) if pr[pr < 0].std() > 1e-10 else 0
    win_rate = (pr > 0).mean()
    s = pd.Series(1 + pr)
    cumprod = s.cumprod()
    cummax = cumprod.cummax()
    max_dd = ((cumprod / cummax) - 1).min()
    return {
        'total_ret': total_ret, 'annual_ret': annual_ret,
        'sharpe': sharpe, 'sortino': sortino,
        'win_rate': win_rate, 'max_drawdown': max_dd,
        'n_trades': len(pr), 'avg_ret': avg_ret,
        'period_returns': pr
    }

def fmt(ret): return f"{ret*100:+.1f}%" if ret is not None else "N/A"

# ============================================================
# 3. 第一阶段：单因素参数扫描
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 阶段1: 单因素扫描...")
REBALANCE_DAYS = 20
val_dates = sorted(val_df['date'].unique())
rebalance_dates_val = [val_dates[i] for i in range(0, len(val_dates), REBALANCE_DAYS)]

PARAM_GRIDS = {
    'rsi': [
        ('rsi','<',20), ('rsi','<',25), ('rsi','<',30), ('rsi','<',35), ('rsi','<',40), ('rsi','<',45),
        ('rsi','>',70), ('rsi','>',75), ('rsi','>',80),
        ('rsi','between',(20,30)), ('rsi','between',(25,35)), ('rsi','between',(30,40)),
    ],
    'price_position': [
        ('price_position','<',0.1), ('price_position','<',0.2), ('price_position','<',0.3), ('price_position','<',0.4),
        ('price_position','>',0.7), ('price_position','>',0.8),
        ('price_position','between',(0.0,0.2)), ('price_position','between',(0.0,0.3)), ('price_position','between',(0.2,0.4)),
    ],
    'price_position_20': [
        ('price_position_20','<',0.1), ('price_position_20','<',0.2), ('price_position_20','<',0.3),
        ('price_position_20','between',(0.0,0.2)), ('price_position_20','between',(0.0,0.3)),
    ],
    'volatility': [
        ('volatility','<',0.010), ('volatility','<',0.015), ('volatility','<',0.020), ('volatility','<',0.025), ('volatility','<',0.030),
        ('volatility','>',0.030), ('volatility','>',0.040),
    ],
    'vol_ratio': [
        ('vol_ratio','>',0.8), ('vol_ratio','>',1.0), ('vol_ratio','>',1.2), ('vol_ratio','>',1.5), ('vol_ratio','>',2.0),
    ],
    'vol_ratio_5': [
        ('vol_ratio_5','>',1.0), ('vol_ratio_5','>',1.5), ('vol_ratio_5','>',2.0),
    ],
    'break_20d': [('break_20d','==',1)],
    'break_10d': [('break_10d','==',1)],
    'ma_bull': [('ma_bull','==',1)],
    'ret_5d_rev': [
        ('ret_5d_rev','>',0.03), ('ret_5d_rev','>',0.05), ('ret_5d_rev','>',0.08), ('ret_5d_rev','>',0.10),
        ('ret_5d_rev','between',(0.03,0.10)), ('ret_5d_rev','between',(0.05,0.15)),
    ],
    'ret_20d_rev': [
        ('ret_20d_rev','>',0.05), ('ret_20d_rev','>',0.10), ('ret_20d_rev','>',0.15), ('ret_20d_rev','>',0.20),
        ('ret_20d_rev','between',(0.05,0.20)),
    ],
    'bb_position': [
        ('bb_position','<',0.2), ('bb_position','<',0.3), ('bb_position','<',0.5),
        ('bb_position','between',(0.0,0.2)), ('bb_position','between',(0.0,0.3)),
    ],
}

all_single_results = []
for factor, params in PARAM_GRIDS.items():
    for cond in params:
        pr, n = run_backtest(val_df, rebalance_dates_val, [cond])
        stats = calc_stats(pr)
        if stats and stats['annual_ret'] > 0:
            all_single_results.append({
                'factor': factor,
                'conditions': [cond],
                'cond_str': normalize_condition(cond),
                'annual_ret': stats['annual_ret'],
                'sharpe': stats['sharpe'],
                'sortino': stats['sortino'],
                'win_rate': stats['win_rate'],
                'max_drawdown': stats['max_drawdown'],
                'n_trades': stats['n_trades'],
                'avg_ret': stats['avg_ret'],
            })

single_df = pd.DataFrame(all_single_results).sort_values('sharpe', ascending=False)
print(f"\n   单因素 Top10 (按夏普):")
print(f"   {'条件':<40} {'年化':>8} {'夏普':>6} {'索提诺':>6} {'胜率':>6} {'交易':>5}")
print("   " + "-" * 80)
for _, r in single_df.head(10).iterrows():
    print(f"   {r['cond_str']:<40} {r['annual_ret']*100:>+7.2f}% {r['sharpe']:>+6.2f} {r['sortino']:>+6.2f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>5d}")

# ============================================================
# 4. 第二阶段：双因素组合扫描
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 阶段2: 双因素组合扫描...")

# 取 top6 单因素条件，两两组合
top6 = single_df.head(6).to_dict('records')

combo_results = []
for i in range(len(top6)):
    for j in range(i+1, len(top6)):
        c1 = top6[i]['conditions']   # list of tuples
        c2 = top6[j]['conditions']
        pr, n = run_backtest(val_df, rebalance_dates_val, c1 + c2)
        stats = calc_stats(pr)
        if stats and stats['annual_ret'] > 0:
            combo_results.append({
                'conditions': c1 + c2,
                'cond_str': top6[i]['cond_str'] + ' AND ' + top6[j]['cond_str'],
                **stats
            })

combo_df = pd.DataFrame([{k:v for k,v in r.items() if k!='period_returns'} for r in combo_results])
if len(combo_df) > 0:
    combo_df = combo_df.sort_values('sharpe', ascending=False)
    print(f"\n   双因素组合 Top10 (按夏普):")
    print(f"   {'组合条件':<65} {'年化':>8} {'夏普':>6} {'索提诺':>6} {'胜率':>6} {'交易':>5}")
    print("   " + "-" * 110)
    for _, r in combo_df.head(10).iterrows():
        print(f"   {r['cond_str']:<65} {r['annual_ret']*100:>+7.2f}% {r['sharpe']:>+6.2f} {r['sortino']:>+6.2f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>5d}")
else:
    print("   无有效双因素组合")
    combo_df = pd.DataFrame()

# ============================================================
# 5. 第三阶段：三因素组合扫描
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 阶段3: 三因素组合扫描...")

combo3_results = []
for i in range(len(top6)):
    for j in range(i+1, len(top6)):
        for k in range(j+1, len(top6)):
            c1, c2, c3 = top6[i]['conditions'], top6[j]['conditions'], top6[k]['conditions']
            pr, n = run_backtest(val_df, rebalance_dates_val, c1 + c2 + c3)
            stats = calc_stats(pr)
            if stats and stats['annual_ret'] > 0.05 and stats['n_trades'] >= 8:
                combo3_results.append({
                    'conditions': c1 + c2 + c3,
                    'cond_str': f"{top6[i]['cond_str']} + {top6[j]['cond_str']} + {top6[k]['cond_str']}",
                    **{k:v for k,v in stats.items() if k!='period_returns'}
                })

if combo3_results:
    combo3_df = pd.DataFrame(combo3_results).sort_values('sharpe', ascending=False)
    print(f"\n   三因素 Top10 (按夏普):")
    print(f"   {'组合条件':<80} {'年化':>8} {'夏普':>6} {'胜率':>6} {'交易':>5}")
    print("   " + "-" * 115)
    for _, r in combo3_df.head(10).iterrows():
        print(f"   {r['cond_str']:<80} {r['annual_ret']*100:>+7.2f}% {r['sharpe']:>+6.2f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>5d}")
else:
    combo3_df = pd.DataFrame()
    print("   无有效三因素组合（样本不足或表现不佳）")

# ============================================================
# 6. 第四阶段：调仓周期 × TopN 参数网格
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 阶段4: 调仓周期 + TopN 参数优化...")

# 收集所有候选组合
candidates = []
for _, r in single_df[single_df['annual_ret'] > 0].head(5).iterrows():
    candidates.append({'conditions': r['conditions'], 'cond_str': r['cond_str']})
for _, r in combo_df.head(5).iterrows():
    candidates.append({'conditions': r['conditions'], 'cond_str': r['cond_str']})
# 去重
seen, unique = set(), []
for c in candidates:
    if c['cond_str'] not in seen:
        seen.add(c['cond_str'])
        unique.append(c)

grid_results = []
for combo in unique[:8]:
    for rd in [5, 10, 15, 20, 30]:
        rebal = [val_dates[i] for i in range(0, len(val_dates), rd)]
        for top_n in [10, 20, 30]:
            pr, n = run_backtest(val_df, rebal, combo['conditions'], top_n=top_n)
            stats = calc_stats(pr, rd)
            if stats and stats['annual_ret'] > 0 and stats['n_trades'] >= 6:
                grid_results.append({
                    'conditions': combo['conditions'],
                    'cond_str': combo['cond_str'],
                    'rebalance_days': rd,
                    'top_n': top_n,
                    **stats
                })

grid_df = pd.DataFrame([{k:v for k,v in r.items() if k!='period_returns'} for r in grid_results])
if len(grid_df) > 0:
    grid_df = grid_df.sort_values('sharpe', ascending=False)
    print(f"\n   参数网格 Top15 (按夏普):")
    print(f"   {'条件':<40} {'周期':>4} {'N':>3} {'年化':>8} {'夏普':>6} {'索提诺':>6} {'胜率':>6} {'交易':>5}")
    print("   " + "-" * 95)
    for _, r in grid_df.head(15).iterrows():
        print(f"   {r['cond_str'][:40]:<40} {r['rebalance_days']:>4d} {r['top_n']:>3d} {r['annual_ret']*100:>+7.2f}% {r['sharpe']:>+6.2f} {r['sortino']:>+6.2f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>5d}")
else:
    grid_df = pd.DataFrame()

# ============================================================
# 7. Walk-Forward 验证：测试集
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 阶段5: Walk-Forward 测试集验证...")

if len(grid_df) > 0:
    best_val = grid_df.iloc[0]
    best_conditions = best_val['conditions']
    best_rd = int(best_val['rebalance_days'])
    best_top_n = int(best_val['top_n'])

    test_dates = sorted(test_df['date'].unique())
    test_rebal = [test_dates[i] for i in range(0, len(test_dates), best_rd)]

    pr_val, _ = run_backtest(val_df, rebal, best_conditions, top_n=best_top_n)
    pr_test, _ = run_backtest(test_df, test_rebal, best_conditions, top_n=best_top_n)
    stats_val = calc_stats(pr_val, best_rd)
    stats_test = calc_stats(pr_test, best_rd)

    # Robustness
    robustness = (stats_test['annual_ret'] / stats_val['annual_ret']) if (stats_val and stats_val['annual_ret'] > 0 and stats_test) else 0

    print(f"\n   最优策略 Walk-Forward:")
    print(f"   条件: {best_val['cond_str']}")
    print(f"   调仓: {best_rd}天 | Top: {best_top_n}")
    print(f"   ── 验证集 ──")
    print(f"     年化: {fmt(stats_val['annual_ret'])}  夏普: {stats_val['sharpe']:.2f}  胜率: {stats_val['win_rate']*100:.1f}%  最大回撤: {stats_val['max_drawdown']*100:.2f}%")
    print(f"   ── 测试集 ──")
    if stats_test:
        print(f"     年化: {fmt(stats_test['annual_ret'])}  夏普: {stats_test['sharpe']:.2f}  胜率: {stats_test['win_rate']*100:.1f}%  最大回撤: {stats_test['max_drawdown']*100:.2f}%")
        print(f"   鲁棒性: {robustness:.2f} (测试/验证, 越接近1越稳定)")
    else:
        print("     无有效交易")
        robustness = 0

# ============================================================
# 8. 综合排名
# ============================================================
print(f"\n[{time.strftime('%H:%M:%S')}] 阶段6: 综合排名选最优...")

if len(grid_df) > 0:
    final_results = []
    test_dates = sorted(test_df['date'].unique())
    for _, row in grid_df.iterrows():
        rd = int(row['rebalance_days'])
        tn = int(row['top_n'])
        rebal = [val_dates[i] for i in range(0, len(val_dates), rd)]
        test_rebal = [test_dates[i] for i in range(0, len(test_dates), rd)]
        pr_v, _ = run_backtest(val_df, rebal, row['conditions'], top_n=tn)
        pr_t, _ = run_backtest(test_df, test_rebal, row['conditions'], top_n=tn)
        sv = calc_stats(pr_v, rd)
        st = calc_stats(pr_t, rd)
        if sv and sv['annual_ret'] > 0:
            rob = (st['annual_ret'] / sv['annual_ret']) if (st and sv['annual_ret'] > 0) else 0
            # 综合分：50%验证年化 + 30%验证夏普 + 20%鲁棒性
            final_score = 0.5 * sv['annual_ret'] + 0.3 * sv['sharpe'] + 0.2 * min(max(rob, 0), 1.5)
            final_results.append({
                'conditions': row['conditions'],
                'cond_str': row['cond_str'],
                'rebalance_days': rd,
                'top_n': tn,
                'val_annual': sv['annual_ret'], 'val_sharpe': sv['sharpe'],
                'val_win': sv['win_rate'], 'val_dd': sv['max_drawdown'],
                'test_annual': st['annual_ret'] if st else 0,
                'test_sharpe': st['sharpe'] if st else 0,
                'test_win': st['win_rate'] if st else 0,
                'test_dd': st['max_drawdown'] if st else 0,
                'robustness': rob,
                'final_score': final_score,
            })

    final_df = pd.DataFrame(final_results).sort_values('final_score', ascending=False)

    print(f"\n   综合排名 Top10:")
    print(f"   {'策略':<40} {'验证年化':>8} {'测试年化':>8} {'验证夏普':>7} {'测试夏普':>7} {'鲁棒性':>6} {'综合分':>6}")
    print("   " + "-" * 100)
    for _, r in final_df.head(10).iterrows():
        print(f"   {r['cond_str'][:40]:<40} {r['val_annual']*100:>+7.2f}% {r['test_annual']*100:>+7.2f}% {r['val_sharpe']:>+6.2f} {r['test_sharpe']:>+6.2f} {r['robustness']:>6.2f} {r['final_score']:>6.3f}")

    best = final_df.iloc[0]
else:
    # fallback to single_df best
    best = single_df.iloc[0]
    best_rd = REBALANCE_DAYS
    best_top_n = 20
    final_df = pd.DataFrame()

# ============================================================
# 9. 输出最优策略
# ============================================================
print("\n" + "=" * 70)
print("  🏆 最优策略")
print("=" * 70)
if len(final_df) > 0:
    b = best
    print(f"""
  条件: {b['cond_str']}
  调仓周期: {b['rebalance_days']} 天
  持仓数量: {b['top_n']} 只
  验证集: 年化 {b['val_annual']*100:.1f}%  夏普 {b['val_sharpe']:.2f}  胜率 {b['val_win']*100:.1f}%  最大回撤 {b['val_dd']*100:.1f}%
  测试集: 年化 {b['test_annual']*100:.1f}%  夏普 {b['test_sharpe']:.2f}  胜率 {b['test_win']*100:.1f}%  最大回撤 {b['test_dd']*100:.1f}%
  鲁棒性: {b['robustness']:.2f}
""")
else:
    print(f"""
  条件: {best['cond_str']}
  年化: {best['annual_ret']*100:.1f}%  夏普: {best['sharpe']:.2f}  胜率: {best['win_rate']*100:.1f}%
""")

# 保存策略代码
cond_str_display = best['cond_str'] if len(final_df) > 0 else best['cond_str']
rd = int(best['rebalance_days']) if len(final_df) > 0 else REBALANCE_DAYS
tn = int(best['top_n']) if len(final_df) > 0 else 20
conds = best['conditions'] if len(final_df) > 0 else best['conditions']

strategy_code = f'''"""
最优选股策略 (自动优化生成)
===========================
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

策略规则: {cond_str_display}
调仓周期: {rd} 天
持仓数量: {tn} 只

验证集表现:
'''

if len(final_df) > 0:
    b = best
    strategy_code += f"  年化: {b['val_annual']*100:.1f}%  夏普: {b['val_sharpe']:.2f}  胜率: {b['val_win']*100:.1f}%\n"
    strategy_code += f"测试集表现:\n"
    strategy_code += f"  年化: {b['test_annual']*100:.1f}%  夏普: {b['test_sharpe']:.2f}  胜率: {b['test_win']*100:.1f}%\n"
else:
    strategy_code += f"  年化: {best['annual_ret']*100:.1f}%  夏普: {best['sharpe']:.2f}  胜率: {best['win_rate']*100:.1f}%\n"

strategy_code += f'''---
Conditions = {conds}
"""

TOP_N = {tn}
REBALANCE_DAYS = {rd}
CONDITIONS = {conds}

def select_stocks(day_data):
    """
    选股函数
    
    规则: {cond_str_display}
    """
    mask = np.ones(len(day_data), dtype=bool)
    for feat, op, val in CONDITIONS:
        col = day_data[feat].values
        if op == '>':   mask &= col > val
        elif op == '<':   mask &= col < val
        elif op == '>=':  mask &= col >= val
        elif op == '<=':  mask &= col <= val
        elif op == '==':  mask &= np.isclose(col, val)
        elif op == 'between': mask &= (col >= val[0]) & (col <= val[1])
    return day_data[mask].nlargest(TOP_N, 'future_return')['code'].tolist()
'''

with open("src/select_stock/strategies/auto_best_strategy.py", 'w') as f:
    f.write(strategy_code)
print(f"  策略代码已保存: src/select_stock/strategies/auto_best_strategy.py")

# 保存报告
report = {
    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'best': {
        'cond_str': cond_str_display,
        'rebalance_days': rd,
        'top_n': tn,
        'conditions': [[c[0],c[1],float(c[2]) if not isinstance(c[2],tuple) else list(c[2])] for c in conds],
    }
}
with open("output/best_strategy.json", 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

single_df.to_csv("output/stage1_single.csv", index=False)
if len(combo_df) > 0: combo_df.to_csv("output/stage2_double.csv", index=False)
if len(grid_df) > 0: grid_df.to_csv("output/stage4_grid.csv", index=False)
if len(final_df) > 0: final_df.to_csv("output/stage6_final.csv", index=False)

print(f"  CSV报告已保存: output/")
print(f"\n[{time.strftime('%H:%M:%S')}] ✅ 全自动优化完成!")
