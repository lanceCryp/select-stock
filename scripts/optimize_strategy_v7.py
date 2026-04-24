"""
策略优化迭代 v7
==============
探索: 1)动量/反转因子组合  2)仓位管理(不等权分配)
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
print("策略优化迭代 v7 - 动量/反转因子 + 仓位管理")
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

    # === 动量/反转因子 ===
    # 短期反转: ret_1d 负相关 → 跌多了反弹
    df['rev_1d'] = -df['ret_1d']
    # 中期反转: ret_5d 负相关
    df['rev_5d'] = -df['ret_5d']
    # 长期动量: ret_20d 正相关 → 强者恒强
    df['mom_20d'] = df['ret_20d']
    # 波动率调整动量
    df['mom_adj'] = df['ret_20d'] / (df['volatility'] + 1e-8)
    # 创20日新高动量
    df['mom_break'] = df['break_20d'] * df['ret_5d']
    # 布林带位置动量
    df['mom_bb'] = df['bb_position'] * df['ret_5d']

    return df

df = calc_features(df)

FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
                'vol_ratio', 'volatility', 'price_position', 'break_20d',
                'rev_1d', 'rev_5d', 'mom_20d', 'mom_adj', 'mom_break', 'mom_bb']

df_model = df.dropna(subset=FEATURE_COLS).copy()
val_df = df_model[df_model['date'] >= '2024-01-01'].copy()
val_dates = sorted(val_df['date'].unique())

model = lgb.Booster(model_file=MODEL_PATH)
# 只用训练时的10个特征做预测
X_val = val_df[FEATURE_COLS[:10]].values
val_df['lgb_pred'] = model.predict(X_val)

# 动量/反转因子用于综合评分(不参与模型预测)
val_df['score_mom'] = val_df['lgb_pred'] + 0.3 * val_df['mom_20d']  # 动量
val_df['score_rev'] = val_df['lgb_pred'] + 0.3 * val_df['rev_5d']   # 反转
val_df['score_comb'] = val_df['lgb_pred'] + 0.15 * val_df['mom_20d'] - 0.15 * val_df['rev_5d']  # 平衡

def apply_rules(data, conditions):
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
    return mask

def backtest_equal(data, dates, strategy_func, top_n=4, stop_loss=0):
    """等权回测"""
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

        for code, pos in list(positions.items()):
            if code not in target_codes:
                sell_row = day_data[day_data['code'] == code]
                if len(sell_row) > 0:
                    proceeds = pos['shares'] * sell_row['close'].values[0] * (1 - COMMISSION * 2)
                    cash += proceeds
                    del positions[code]

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
    years = len(pv) * 55 / 252
    annual = ((pv['value'].iloc[-1] / INITIAL_CASH) ** (1 / max(years, 0.1)) - 1)
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

def backtest_weighted(data, dates, strategy_func, top_n=4, use_momentum=False):
    """不等权回测: 按lgb_pred分数加权分配"""
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
        if len(selected) == 0:
            continue

        # 计算权重: 按预测分数 softmax 权重
        scores = selected['lgb_pred'].values
        # 归一化到概率
        scores_norm = scores - scores.min() + 1e-8
        weights = scores_norm / scores_norm.sum()

        target_codes = selected['code'].tolist()

        # 卖出 不在target的持仓
        for code, pos in list(positions.items()):
            if code not in target_codes:
                sell_row = day_data[day_data['code'] == code]
                if len(sell_row) > 0:
                    proceeds = pos['shares'] * sell_row['close'].values[0] * (1 - COMMISSION * 2)
                    cash += proceeds
                    del positions[code]

        # 买入 按权重分配
        if len(target_codes) > 0:
            total_alloc = cash
        else:
            total_alloc = 0

        for idx, code in enumerate(target_codes):
            buy_row = day_data[day_data['code'] == code]
            if len(buy_row) == 0:
                continue
            buy_price = buy_row['close'].values[0]
            alloc = total_alloc * weights[idx]
            shares = int(alloc / buy_price / 100) * 100
            if shares > 0:
                cost = shares * buy_price * (1 + COMMISSION)
                if cost <= cash:
                    cash -= cost
                    positions[code] = {'shares': shares, 'buy_price': buy_price, 'weight': weights[idx]}

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
    years = len(pv) * 55 / 252
    annual = ((pv['value'].iloc[-1] / INITIAL_CASH) ** (1 / max(years, 0.1)) - 1)
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

# 基准: PP>0.25 等权
print("\n基准: PP>0.25 等权")
conditions = [('price_position', '>', 0.25)]
def make_func(cond):
    def func(d, n):
        mask = apply_rules(d, cond)
        filtered = d[mask]
        if len(filtered) >= n:
            return filtered.nlargest(n, 'lgb_pred')
        return d.nlargest(n, 'lgb_pred')
    return func
result = backtest_equal(val_df, val_dates[::55], make_func(conditions), top_n=4)
if result:
    results.append({'name': 'PP>0.25等权(基准)', **result})
    print(f"  基准: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# ========== 1. 动量/反转因子组合 ==========
print("\n" + "="*50)
print("1. 动量/反转因子组合测试")
print("="*50)

# 1a. 不同评分系统对比
print("\n1a. 评分系统对比 (PP>0.25)")
scorings = [
    ('lgb_pred', '纯LGB'),
    ('score_mom', 'LGB+动量'),
    ('score_rev', 'LGB+反转'),
    ('score_comb', 'LGB+平衡'),
]
for col, name in scorings:
    def make_scored_func(col):
        def func(d, n):
            mask = apply_rules(d, [('price_position', '>', 0.25)])
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, col)
            return d.nlargest(n, col)
        return func
    result = backtest_equal(val_df, val_dates[::55], make_scored_func(col), top_n=4)
    if result:
        results.append({'name': f'{name}+PP>0.25', **result})
        print(f"  {name}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 1b. 动量权重测试
print("\n1b. 动量权重测试")
for mom_w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    val_df['score_w'] = val_df['lgb_pred'] + mom_w * val_df['mom_20d']
    def make_w_func(w):
        def func(d, n):
            mask = apply_rules(d, [('price_position', '>', 0.25)])
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'score_w')
            return d.nlargest(n, 'score_w')
        return func
    result = backtest_equal(val_df, val_dates[::55], make_w_func(mom_w), top_n=4)
    if result:
        results.append({'name': f'动量权重{mom_w}+PP>0.25', **result})
        print(f"  动量权重{mom_w}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 1c. 反转权重测试
print("\n1c. 反转权重测试")
for rev_w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    val_df['score_rev_w'] = val_df['lgb_pred'] + rev_w * val_df['rev_5d']
    def make_rev_func(w):
        def func(d, n):
            mask = apply_rules(d, [('price_position', '>', 0.25)])
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'score_rev_w')
            return d.nlargest(n, 'score_rev_w')
        return func
    result = backtest_equal(val_df, val_dates[::55], make_rev_func(rev_w), top_n=4)
    if result:
        results.append({'name': f'反转权重{rev_w}+PP>0.25', **result})
        print(f"  反转权重{rev_w}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 1d. PP阈值 + 动量
print("\n1d. PP阈值 + 动量")
for pp in [0.20, 0.25, 0.30, 0.35]:
    for mom_w in [0.1, 0.2, 0.3]:
        col_name = f'score_mom_{pp}_{mom_w}'
        val_df[col_name] = val_df['lgb_pred'] + mom_w * val_df['mom_20d']
        def make_pm_func(pp, mw, cn):
            def func(d, n):
                mask = apply_rules(d, [('price_position', '>', pp)])
                filtered = d[mask]
                if len(filtered) >= n:
                    return filtered.nlargest(n, cn)
                return d.nlargest(n, cn)
            return func
        result = backtest_equal(val_df, val_dates[::55], make_pm_func(pp, mom_w, col_name), top_n=4)
        if result:
            results.append({'name': f'PP>{pp}+动量{mom_w}', **result})
            print(f"  PP>{pp}+动量{mom_w}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 1e. 波动率调整动量
print("\n1e. 波动率调整动量")
for mom_w in [0.1, 0.2, 0.3]:
    val_df['score_mom_adj'] = val_df['lgb_pred'] + mom_w * val_df['mom_adj']
    def make_adj_func(w):
        def func(d, n):
            mask = apply_rules(d, [('price_position', '>', 0.25)])
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'score_mom_adj')
            return d.nlargest(n, 'score_mom_adj')
        return func
    result = backtest_equal(val_df, val_dates[::55], make_adj_func(mom_w), top_n=4)
    if result:
        results.append({'name': f'波动调整动量{mom_w}+PP>0.25', **result})
        print(f"  波动调整动量{mom_w}: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# ========== 2. 仓位管理(不等权分配) ==========
print("\n" + "="*50)
print("2. 仓位管理(不等权分配)测试")
print("="*50)

# 2a. 加权 vs 等权基准
print("\n2a. 加权 vs 等权对比")
result_equal = backtest_equal(val_df, val_dates[::55], make_func(conditions), top_n=4)
result_weighted = backtest_weighted(val_df, val_dates[::55], make_func(conditions), top_n=4)
if result_equal:
    results.append({'name': 'PP>0.25等权', **result_equal})
    print(f"  等权: 年化 {result_equal['annual']*100:+.1f}% | 夏普 {result_equal['sharpe']:.2f} | 胜率 {result_equal['win_rate']*100:.0f}%")
if result_weighted:
    results.append({'name': 'PP>0.25加权', **result_weighted})
    print(f"  加权: 年化 {result_weighted['annual']*100:+.1f}% | 夏普 {result_weighted['sharpe']:.2f} | 胜率 {result_weighted['win_rate']*100:.0f}%")

# 2b. 不同PP阈值的加权效果
print("\n2b. 不同PP阈值 + 加权")
for pp in [0.20, 0.25, 0.30, 0.35]:
    conditions_pp = [('price_position', '>', pp)]
    def make_pp_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func
    result = backtest_weighted(val_df, val_dates[::55], make_pp_func(conditions_pp), top_n=4)
    if result:
        results.append({'name': f'PP>{pp}+加权', **result})
        print(f"  PP>{pp}+加权: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 2c. 集中/分散仓位
print("\n2c. 集中/分散仓位")
# top_1 集中
def strategy_top1(d, n):
    return d.nlargest(1, 'lgb_pred')
# top_2 集中
def strategy_top2(d, n):
    return d.nlargest(2, 'lgb_pred')
# top_3 集中
def strategy_top3(d, n):
    return d.nlargest(3, 'lgb_pred')

for pp in [0.25, 0.30]:
    conditions_pp = [('price_position', '>', pp)]
    def make_pp_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func

    for top_n, strat in [(1, strategy_top1), (2, strategy_top2), (3, strategy_top3)]:
        result = backtest_weighted(val_df, val_dates[::55], strat, top_n=top_n)
        if result:
            results.append({'name': f'PP>{pp}+集中{top_n}只', **result})
            print(f"  PP>{pp}+集中{top_n}只: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 2d. 换仓周期 + 加权
print("\n2d. 换仓周期 + 加权")
for hold_days in [45, 50, 55, 60]:
    rd = val_dates[::hold_days]
    if len(rd) < 3:
        continue
    result = backtest_weighted(val_df, rd, make_func(conditions), top_n=4)
    if result:
        results.append({'name': f'PP>0.25+加权+{hold_days}天', **result})
        print(f"  PP>0.25+加权+{hold_days}天: 年化 {result['annual']*100:+.1f}% | 夏普 {result['sharpe']:.2f} | 胜率 {result['win_rate']*100:.0f}%")

# 排序输出
results.sort(key=lambda x: (x['sharpe'], x['annual']), reverse=True)

print("\n" + "=" * 60)
print("结果汇总 (按夏普排序)")
print("=" * 60)

print(f"\n{'排名':<4} {'策略':<28} {'年化':>10} {'夏普':>8} {'胜率':>8} {'最大回撤':>10}")
print("-" * 70)

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
with open('output/best_strategy_v7.json', 'w') as f:
    json.dump({
        'strategy': best['name'],
        'annual_return': float(best['annual']),
        'sharpe': float(best['sharpe']),
        'win_rate': float(best['win_rate']),
        'max_drawdown': float(best['max_drawdown']),
        'total_return': float(best['total']),
        'final_value': float(best['final_value']),
    }, f, indent=2, ensure_ascii=False)
print(f"\n✅ 结果已保存: output/best_strategy_v7.json")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'portfolio'} for r in results])
results_df.to_csv('output/all_strategies_v7.csv', index=False, encoding='utf-8-sig')
print(f"✅ 所有策略已保存: output/all_strategies_v7.csv")