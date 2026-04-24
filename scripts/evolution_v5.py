"""
策略进化 v5 - 参数精细化 + 稳健性验证
======================================
在v4发现PP>0.25可能优于PP>0.45后，进一步精细化测试
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/kcb_history.csv"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")
REPORT_DIR = OUTPUT_DIR / "reports"

INITIAL_CASH = 1000000.0
COMMISSION = 0.0003

print("=" * 70)
print("🧬 策略进化 v5 - 参数精细化 + 稳健性验证")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

print(f"\n📊 数据: {df['code'].nunique()} 只股票")

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

df = calc_features(df)

val_df = df[df['date'] >= '2024-01-01'].copy()
val_dates = sorted(val_df['date'].unique())

model = lgb.Booster(model_file=str(MODEL_DIR / "lgb_stock_selector.txt"))
val_df['lgb_pred'] = model.predict(val_df[FEATURE_COLS].values)

def apply_rules(data, conditions):
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
    return mask

def backtest(data, dates, strategy_func, top_n=4):
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

def make_func(cond, sort_col='lgb_pred'):
    def func(d, n):
        mask = apply_rules(d, cond)
        filtered = d[mask]
        if len(filtered) >= n:
            return filtered.nlargest(n, sort_col)
        return d.nlargest(n, sort_col)
    return func

# ============================================================
# v5.1: PP阈值精细化 (聚焦0.20-0.35)
# ============================================================
print("\n" + "=" * 70)
print("🧬 v5.1: PP阈值精细化 (0.20-0.35)")
print("=" * 70)

results = []

for pp in np.arange(0.20, 0.40, 0.01):
    for top_n in [3, 4, 5]:
        conditions = [('price_position', '>', round(pp, 2))]
        result = backtest(val_df, val_dates[::55], make_func(conditions), top_n=top_n)
        if result:
            results.append({
                'pp': pp, 'top_n': top_n,
                'annual': result['annual'], 'sharpe': result['sharpe'],
                'win_rate': result['win_rate'], 'max_drawdown': result['max_drawdown'],
                'final_value': result['final_value']
            })
            print(f"   PP>{pp:.2f} + {top_n}只: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")

# 按夏普排序
results.sort(key=lambda x: x['sharpe'], reverse=True)

print("\n" + "=" * 70)
print("🏆 Top 10 精细化策略")
print("=" * 70)
print(f"{'排名':<4} {'PP阈值':>8} {'持仓':>6} {'年化':>10} {'夏普':>8} {'胜率':>8} {'最大回撤':>10}")
print("-" * 60)

for i, r in enumerate(results[:10], 1):
    print(f"{i:<4} >{r['pp']:.2f}    {r['top_n']}只    {r['annual']*100:>9.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>7.1f}% {r['max_drawdown']*100:>9.2f}%")

best = results[0]

# ============================================================
# v5.2: 稳健性验证 - 不同周期
# ============================================================
print("\n" + "=" * 70)
print("🧬 v5.2: 稳健性验证 - 不同周期")
print("=" * 70)

for hold_days in [45, 50, 55, 60, 65]:
    rd = val_dates[::hold_days]
    if len(rd) < 3:
        continue
    conditions = [('price_position', '>', round(best['pp'], 2))]
    result = backtest(val_df, rd, make_func(conditions), top_n=best['top_n'])
    if result:
        print(f"   PP>{best['pp']:.2f} + {best['top_n']}只 + {hold_days}天: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f}")

# ============================================================
# v5.3: 验证vs原始v6配置
# ============================================================
print("\n" + "=" * 70)
print("🧬 v5.3: 与历史v6配置对比")
print("=" * 70)

v6_conditions = [('price_position', '>', 0.25)]
result_v6 = backtest(val_df, val_dates[::55], make_func(v6_conditions), top_n=4)

print(f"   v5最优: PP>{best['pp']:.2f} + {best['top_n']}只: 年化{best['annual']*100:+.1f}% | 夏普{best['sharpe']:.2f}")
print(f"   v6原始: PP>0.25 + 4只: 年化{result_v6['annual']*100:+.1f}% | 夏普{result_v6['sharpe']:.2f}")

# ============================================================
# 生成v5报告
# ============================================================
print("\n" + "=" * 70)
print("🏆 v5 最终结论")
print("=" * 70)

print(f"""
📊 精细化测试结果:
   最优PP阈值: {best['pp']:.2f}
   最优持仓: {best['top_n']}只
   年化收益: {best['annual']*100:+.1f}%
   夏普比率: {best['sharpe']:.2f}
   胜率: {best['win_rate']*100:.0f}%
   最大回撤: {best['max_drawdown']*100:.2f}%

📈 版本对比:
   v5精细化: PP>{best['pp']:.2f} + {best['top_n']}只 = 年化{best['annual']*100:+.1f}% | 夏普{best['sharpe']:.2f}
   v6原始: PP>0.25 + 4只 = 年化{result_v6['annual']*100:+.1f}% | 夏普{result_v6['sharpe']:.2f}

✅ 结论:
   {'v5配置更优' if best['sharpe'] > result_v6['sharpe'] else 'v6原始配置仍然最优'}
""")

# 保存
version_data = {
    'version': 'v5',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'best_config': {
        'pp_threshold': float(best['pp']),
        'top_n': int(best['top_n']),
        'hold_days': 55,
        'annual': float(best['annual']),
        'sharpe': float(best['sharpe']),
        'win_rate': float(best['win_rate']),
        'max_drawdown': float(best['max_drawdown'])
    },
    'comparison': {
        'v5_best': float(best['annual']),
        'v6_original': float(result_v6['annual'])
    }
}

with open(OUTPUT_DIR / 'version_v5.json', 'w') as f:
    json.dump(version_data, f, indent=2, ensure_ascii=False)

# 生成报告
report = f"""# v5 进化报告 - 参数精细化

## 进化日期: {datetime.now().strftime('%Y-%m-%d')}

### 进化背景

v4发现PP>0.25可能优于PP>0.45，需要精细化测试确认。

### 精细化测试

测试范围: PP 0.20-0.40, 持仓 3-5只

### Top 10 结果

| 排名 | PP阈值 | 持仓 | 年化 | 夏普 | 胜率 | 回撤 |
|------|--------|------|------|------|------|------|
"""

for i, r in enumerate(results[:10], 1):
    report += f"| {i} | >{r['pp']:.2f} | {r['top_n']}只 | {r['annual']*100:+.1f}% | {r['sharpe']:.2f} | {r['win_rate']*100:.0f}% | {r['max_drawdown']*100:.2f}% |\n"

report += f"""
### 最终配置

| 参数 | v5最优 | v6原始 |
|------|--------|--------|
| PP阈值 | >{best['pp']:.2f} | >0.25 |
| 持仓 | {best['top_n']}只 | 4只 |
| 周期 | 55天 | 55天 |
| 年化 | {best['annual']*100:+.1f}% | {result_v6['annual']*100:+.1f}% |
| 夏普 | {best['sharpe']:.2f} | {result_v6['sharpe']:.2f} |

### 结论

**{'v5配置更优' if best['sharpe'] > result_v6['sharpe'] else 'v6原始配置仍然最优'}**

"""

report_path = REPORT_DIR / f"v5_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ v5报告已保存: {report_path}")
print(f"✅ v5配置已保存: output/version_v5.json")