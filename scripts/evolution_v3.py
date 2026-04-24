"""
策略进化 v3 - 预测目标优化 + 动量增强
======================================
在v2基础上:
1. 优化预测目标(20日 vs 40日 vs 55日)
2. 添加动量因子增强
3. 多空双向验证
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
REPORT_DIR.mkdir(exist_ok=True)

INITIAL_CASH = 1000000.0
COMMISSION = 0.0003

print("=" * 70)
print("🧬 策略进化 v3 - 预测目标优化 + 动量增强")
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

    # 动量因子
    df['momentum_5d'] = df['ret_5d']
    df['momentum_20d'] = df['ret_20d']
    df['momentum_accel'] = df['ret_5d'] - df['ret_20d']  # 5日加速

    return df

FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
                'vol_ratio', 'volatility', 'price_position', 'break_20d']

df = calc_features(df)

val_df = df[df['date'] >= '2024-01-01'].copy()
val_dates = sorted(val_df['date'].unique())

def apply_rules(data, conditions):
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
    return mask

def backtest(data, dates, strategy_func, top_n=3):
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

# ============================================================
# v3.1: 测试不同预测目标训练的模型
# ============================================================
print("\n" + "=" * 70)
print("🧬 v3.1: 测试不同预测目标训练的模型")
print("=" * 70)

best_predict_days = 20
best_predict_sharpe = 0
model_paths = {}

train_df = df[df['date'] <= '2023-12-31'].dropna(subset=FEATURE_COLS)

for predict_days in [20, 40, 55]:
    print(f"\n训练预测{predict_days}日收益模型...")

    # 计算对应预测目标
    df[f'future_{predict_days}d'] = df.groupby('code')['close'].pct_change(predict_days).shift(-predict_days)

    train_data = df[df['date'] <= '2023-12-31'].dropna(subset=[f'future_{predict_days}d'])
    X_train = train_data[FEATURE_COLS].values
    y_train = train_data[f'future_{predict_days}d'].values

    if len(X_train) < 1000:
        continue

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
    params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
              'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
              'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
              'verbose': -1, 'seed': 42}

    model = lgb.train(params, lgb_train, num_boost_round=200)
    model_path = MODEL_DIR / f"lgb_v3_{predict_days}d.txt"
    model.save_model(str(model_path))
    model_paths[predict_days] = str(model_path)

    # 用这个模型预测并回测
    val_df_model = df[df['date'] >= '2024-01-01'].copy()
    X_val = val_df_model[FEATURE_COLS].values
    val_df_model[f'pred_{predict_days}d'] = model.predict(X_val)

    # 回测
    conditions = [('price_position', '>', 0.45)]
    def make_func(cond, pred_col):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, pred_col)
            return d.nlargest(n, pred_col)
        return func

    result = backtest(val_df_model, val_dates[::55], make_func(conditions, f'pred_{predict_days}d'), top_n=3)

    if result:
        print(f"   预测{predict_days}日模型 → 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f}")
        if result['sharpe'] > best_predict_sharpe:
            best_predict_days = predict_days
            best_predict_sharpe = result['sharpe']
            best_predict_result = result

print(f"\n✅ 最佳预测目标: {best_predict_days}日 (夏普: {best_predict_sharpe:.2f})")

# ============================================================
# v3.2: 动量因子增强
# ============================================================
print("\n" + "=" * 70)
print("🧬 v3.2: 动量因子增强")
print("=" * 70)

best_model = lgb.Booster(model_file=model_paths[best_predict_days])
val_df[f'pred'] = best_model.predict(val_df[FEATURE_COLS].values)

# 动量评分组合
val_df['momentum_score'] = val_df['pred'] + 0.1 * val_df['momentum_20d']

best_momentum = None
best_momentum_sharpe = best_predict_sharpe

for mom_weight in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    col = 'pred' if mom_weight == 0 else 'momentum_score'
    if mom_weight > 0:
        val_df['momentum_score'] = val_df['pred'] + mom_weight * val_df['momentum_20d']

    conditions = [('price_position', '>', 0.45)]
    def make_func(cond, score_col):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, score_col)
            return d.nlargest(n, score_col)
        return func

    result = backtest(val_df, val_dates[::55], make_func(conditions, col), top_n=3)

    if result:
        mom_str = f"pred+动量{mom_weight}" if mom_weight > 0 else "纯pred"
        print(f"   {mom_str}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f}")
        if result['sharpe'] > best_momentum_sharpe:
            best_momentum = mom_weight
            best_momentum_sharpe = result['sharpe']
            best_momentum_result = result

print(f"\n✅ 最佳动量权重: {best_momentum if best_momentum else '无'} (夏普: {best_momentum_sharpe:.2f})")

# ============================================================
# v3.3: PP阈值精细化
# ============================================================
print("\n" + "=" * 70)
print("🧬 v3.3: PP阈值精细化 (基于最佳预测目标)")
print("=" * 70)

best_pp_v3 = 0.45
best_pp_v3_sharpe = best_momentum_sharpe

for pp in np.arange(0.35, 0.60, 0.05):
    conditions = [('price_position', '>', round(pp, 2))]
    col = 'momentum_score' if best_momentum and best_momentum > 0 else 'pred'

    def make_func(cond, score_col):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, score_col)
            return d.nlargest(n, score_col)
        return func

    result = backtest(val_df, val_dates[::55], make_func(conditions, col), top_n=3)

    if result:
        print(f"   PP>{pp:.2f}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f}")
        if result['sharpe'] > best_pp_v3_sharpe:
            best_pp_v3 = pp
            best_pp_v3_sharpe = result['sharpe']
            best_pp_v3_result = result

print(f"\n✅ 最佳PP阈值: {best_pp_v3:.2f} (夏普: {best_pp_v3_sharpe:.2f})")

# ============================================================
# 生成v3最终结果
# ============================================================
print("\n" + "=" * 70)
print("🏆 v3 最终版本配置")
print("=" * 70)

col_v3 = 'momentum_score' if best_momentum and best_momentum > 0 else 'pred'
conditions_v3 = [('price_position', '>', best_pp_v3)]

def make_final_func(cond, score_col):
    def func(d, n):
        mask = apply_rules(d, cond)
        filtered = d[mask]
        if len(filtered) >= n:
            return filtered.nlargest(n, score_col)
        return d.nlargest(n, score_col)
    return func

final_result = backtest(val_df, val_dates[::55], make_final_func(conditions_v3, col_v3), top_n=3)

print(f"""
📋 v3 版本配置:
   预测目标: {best_predict_days}日收益
   动量权重: {best_momentum if best_momentum else '无'}
   PP阈值: >{best_pp_v3:.2f}
   持仓数量: 3只
   调仓周期: 55天

📊 v3 最终绩效:
   年化收益: {final_result['annual']*100:+.1f}%
   夏普比率: {final_result['sharpe']:.2f}
   胜率: {final_result['win_rate']*100:.0f}%
   最大回撤: {final_result['max_drawdown']*100:.2f}%
   最终资金: {final_result['final_value']:,.0f}
""")

# ============================================================
# 版本对比
# ============================================================
print("\n" + "=" * 70)
print("📊 版本对比")
print("=" * 70)

# v1: PP>0.25, 4只 (原始)
# v2: PP>0.45, 3只 (v2进化)
# v3: PP>{best_pp_v3}, {best_predict_days}日预测, 3只

print(f"""
| 版本 | PP阈值 | 持仓 | 预测目标 | 动量权重 | 年化 | 夏普 | 胜率 |
|------|--------|------|----------|----------|------|------|------|
| v1 | >0.25 | 4只 | 20日 | 无 | +46.8% | 1.44 | 64% |
| v2 | >0.45 | 3只 | 20日 | 无 | +69.6% | 1.62 | 64% |
| v3 | >{best_pp_v3:.2f} | 3只 | {best_predict_days}日 | {best_momentum if best_momentum else '无'} | {final_result['annual']*100:+.1f}% | {final_result['sharpe']:.2f} | {final_result['win_rate']*100:.0f}% |
""")

# ============================================================
# 保存v3报告
# ============================================================
version_data = {
    'version': 'v3',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'config': {
        'predict_days': int(best_predict_days),
        'momentum_weight': float(best_momentum) if best_momentum else 0,
        'pp_threshold': float(best_pp_v3),
        'top_n': 3,
        'hold_days': 55
    },
    'metrics': {
        'annual': float(final_result['annual']),
        'sharpe': float(final_result['sharpe']),
        'win_rate': float(final_result['win_rate']),
        'max_drawdown': float(final_result['max_drawdown']),
        'final_value': float(final_result['final_value'])
    }
}

with open(OUTPUT_DIR / 'version_v3.json', 'w') as f:
    json.dump(version_data, f, indent=2, ensure_ascii=False)

# 生成报告
report = f"""# v3 进化报告

## 进化日期: {datetime.now().strftime('%Y-%m-%d')}

### 进化内容

1. **预测目标优化**: 测试20/40/55日预测目标 → 最佳 {best_predict_days}日
2. **动量因子增强**: 测试0~0.3权重 → 最佳 {best_momentum if best_momentum else '无'}
3. **PP阈值精细化**: 在新模型基础上精细化 → 最佳 {best_pp_v3:.2f}

### v3 配置

| 参数 | 值 |
|------|-----|
| 预测目标 | {best_predict_days}日 |
| 动量权重 | {best_momentum if best_momentum else '无'} |
| PP阈值 | >{best_pp_v3:.2f} |
| 持仓 | 3只 |
| 周期 | 55天 |

### v3 绩效

| 指标 | 值 |
|------|-----|
| 年化收益 | {final_result['annual']*100:+.1f}% |
| 夏普比率 | {final_result['sharpe']:.2f} |
| 胜率 | {final_result['win_rate']*100:.0f}% |
| 最大回撤 | {final_result['max_drawdown']*100:.2f}% |
| 最终资金 | {final_result['final_value']:,.0f} |

### 每期收益

"""
pv = final_result['portfolio']
cumret = 1.0
for _, row in pv.iterrows():
    cumret *= (1 + row['ret'])
    marker = "🟢" if row['ret'] > 0 else "🔴"
    report += f"{marker} {row['date'].date()}: {row['ret']*100:>+7.2f}%  累计: {cumret*100:>8.1f}%\n"

report_path = REPORT_DIR / f"v3_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ v3报告已保存: {report_path}")
print(f"✅ v3配置已保存: output/version_v3.json")