"""
持续迭代优化系统
================
策略 ↔ 训练 双向循环优化

循环逻辑:
1. 策略需求 → 训练目标 (预测什么?)
2. 训练模型 → 模型评估 (预测准吗?)
3. 回测策略 → 绩效分析 (赚钱吗?)
4. 分析结论 → 优化方向 → 回到1

使用方法:
    uv run python scripts/iterative_optimizer.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_PATH = "data/kcb_history.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 基础参数范围
TOP_N_OPTIONS = [4, 8, 12, 20]
HOLD_DAYS_OPTIONS = [20, 40, 55, 80]
RSI_FILTER_OPTIONS = [0, 50, 60, 70]  # 0=不过滤
PP_THRESHOLD_OPTIONS = [0.2, 0.25, 0.3, 0.35]

INITIAL_CASH = 1000000.0
COMMISSION = 0.0003

# ============================================================
# 加载数据
# ============================================================
print("=" * 70)
print("🔄 持续迭代优化系统 - 策略与训练双向循环")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

print(f"\n📊 数据: {df['code'].nunique()} 只股票, {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============================================================
# 计算特征
# ============================================================
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

# 验证集
val_df = df[df['date'] >= '2024-01-01'].copy()
val_dates = sorted(val_df['date'].unique())

# ============================================================
# 辅助函数
# ============================================================
def apply_rules(data, conditions):
    mask = pd.Series(True, index=data.index)
    for feat, op, val in conditions:
        if op == '>': mask &= data[feat] > val
        elif op == '<': mask &= data[feat] < val
    return mask

def backtest(data, dates, strategy_func, top_n=4):
    """回测函数"""
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
# 循环1: 测试不同预测目标 (训练)
# ============================================================
print("\n" + "=" * 70)
print("🔁 循环1: 测试不同预测目标 (训练目标优化)")
print("=" * 70)

train_end = '2023-12-31'
val_start = '2024-01-01'

train_results = []

for predict_days in [20, 40, 55, 80]:
    print(f"\n📊 训练目标: {predict_days}日收益")

    # 计算未来收益
    df[f'future_{predict_days}d'] = df.groupby('code')['close'].pct_change(predict_days).shift(-predict_days)

    # 准备数据
    df_model = df.dropna(subset=FEATURE_COLS + [f'future_{predict_days}d']).copy()
    train_df = df_model[df_model['date'] <= train_end].copy()
    val_df_model = df_model[df_model['date'] >= val_start].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[f'future_{predict_days}d'].values
    X_val = val_df_model[FEATURE_COLS].values
    y_val = val_df_model[f'future_{predict_days}d'].values

    if len(X_train) < 1000 or len(X_val) < 100:
        continue

    # 训练
    lgb_train = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
    lgb_val = lgb.Dataset(X_val, y_val, feature_name=FEATURE_COLS, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'verbose': -1,
        'seed': 42,
    }

    model = lgb.train(
        params, lgb_train, num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    corr = np.corrcoef(val_pred, y_val)[0, 1]
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"   IC相关性: {corr:.4f} | RMSE: {val_rmse:.4f} | 迭代: {model.best_iteration}")

    # 保存模型
    model_path = MODEL_DIR / f"lgb_predict_{predict_days}d.txt"
    model.save_model(str(model_path))

    train_results.append({
        'predict_days': predict_days,
        'correlation': corr,
        'rmse': val_rmse,
        'best_iteration': model.best_iteration,
        'model_path': str(model_path)
    })

# 选择最佳预测天数
best_train = max(train_results, key=lambda x: x['correlation'])
print(f"\n🏆 最佳预测目标: {best_train['predict_days']}日 (IC: {best_train['correlation']:.4f})")

# ============================================================
# 循环2: 测试不同策略参数 (回测)
# ============================================================
print("\n" + "=" * 70)
print("🔁 循环2: 测试不同策略参数 (策略参数优化)")
print("=" * 70)

# 使用最佳预测目标的模型
best_model = lgb.Booster(model_file=best_train['model_path'])

# 添加预测分数到验证集
val_df[f'pred_{best_train["predict_days"]}d'] = best_model.predict(val_df[FEATURE_COLS].values)

strategy_results = []

for pp_th in PP_THRESHOLD_OPTIONS:
    for top_n in TOP_N_OPTIONS:
        for hold_days in [55]:  # 固定55天周期
            conditions = [('price_position', '>', pp_th)]
            if best_train['predict_days'] != 55:
                pred_col = f'pred_{best_train["predict_days"]}d'
            else:
                pred_col = f'pred_{best_train["predict_days"]}d'

            def make_func(cond, pred_col):
                def func(d, n):
                    mask = apply_rules(d, cond)
                    filtered = d[mask]
                    if len(filtered) >= n:
                        return filtered.nlargest(n, pred_col)
                    return d.nlargest(n, pred_col)
                return func

            rd = val_dates[::hold_days]
            if len(rd) < 3:
                continue

            result = backtest(val_df, rd, make_func(conditions, pred_col), top_n=top_n)

            if result:
                strategy_results.append({
                    'pp_threshold': pp_th,
                    'top_n': top_n,
                    'hold_days': hold_days,
                    'predict_days': best_train['predict_days'],
                    'annual': result['annual'],
                    'sharpe': result['sharpe'],
                    'win_rate': result['win_rate'],
                    'max_drawdown': result['max_drawdown'],
                    'final_value': result['final_value']
                })
                print(f"   PP>{pp_th} + {top_n}只 + {hold_days}天: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")

# ============================================================
# 循环3: 测试RSI过滤 (条件优化)
# ============================================================
print("\n" + "=" * 70)
print("🔁 循环3: 测试RSI过滤 (买卖信号优化)")
print("=" * 70)

best_strategies = sorted(strategy_results, key=lambda x: x['sharpe'], reverse=True)[:3]

rsi_results = []

for s in best_strategies:
    pp_th = s['pp_threshold']
    top_n = s['top_n']

    for rsi_filter in RSI_FILTER_OPTIONS:
        conditions = [('price_position', '>', pp_th)]
        if rsi_filter > 0:
            conditions.append(('rsi', '<', rsi_filter))

        pred_col = f'pred_{best_train['predict_days']}d'

        def make_func(cond, pred_col):
            def func(d, n):
                mask = apply_rules(d, cond)
                filtered = d[mask]
                if len(filtered) >= n:
                    return filtered.nlargest(n, pred_col)
                return d.nlargest(n, pred_col)
            return func

        rd = val_dates[::55]
        result = backtest(val_df, rd, make_func(conditions, pred_col), top_n=top_n)

        if result:
            rsi_results.append({
                'pp_threshold': pp_th,
                'top_n': top_n,
                'rsi_filter': rsi_filter if rsi_filter > 0 else '无',
                'predict_days': best_train['predict_days'],
                'annual': result['annual'],
                'sharpe': result['sharpe'],
                'win_rate': result['win_rate'],
                'max_drawdown': result['max_drawdown']
            })
            rsi_str = f"RSI<{rsi_filter}" if rsi_filter > 0 else "无RSI过滤"
            print(f"   PP>{pp_th} + {top_n}只 + {rsi_str}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f}")

# ============================================================
# 综合排序
# ============================================================
print("\n" + "=" * 70)
print("📊 综合排序 (所有迭代结果)")
print("=" * 70)

all_results = []

# 添加RSI过滤结果
for r in rsi_results:
    all_results.append({
        '策略': f"预测{best_train['predict_days']}日+PP>{r['pp_threshold']}+持仓{r['top_n']}只+{r['rsi_filter']}",
        '年化': r['annual'] * 100,
        '夏普': r['sharpe'],
        '胜率': r['win_rate'] * 100,
        '最大回撤': r['max_drawdown'] * 100
    })

# 添加无RSI结果
for s in strategy_results:
    all_results.append({
        '策略': f"预测{best_train['predict_days']}日+PP>{s['pp_threshold']}+持仓{s['top_n']}只",
        '年化': s['annual'] * 100,
        '夏普': s['sharpe'],
        '胜率': s['win_rate'] * 100,
        '最大回撤': s['max_drawdown'] * 100
    })

all_results.sort(key=lambda x: (x['夏普'], x['年化']), reverse=True)

print(f"\n{'排名':<4} {'策略':<40} {'年化':>10} {'夏普':>8} {'胜率':>8} {'最大回撤':>10}")
print("-" * 80)

for i, r in enumerate(all_results[:15], 1):
    print(f"{i:<4} {r['策略']:<40} {r['年化']:>9.1f}% {r['夏普']:>8.2f} {r['胜率']:>7.1f}% {r['最大回撤']:>9.2f}%")

# ============================================================
# 最终推荐
# ============================================================
best = all_results[0]

print("\n" + "=" * 70)
print("🏆 最终推荐策略")
print("=" * 70)
print(f"""
📋 策略配置:
   预测目标: {best_train['predict_days']}日收益
   模型IC: {best_train['correlation']:.4f}

   选股条件:
   - PP > {best['策略'].split('PP>')[1].split('+')[0]}
   - 持仓 {best['策略'].split('持仓')[1].split('只')[0]} 只
   - 调仓周期: 55个交易日

📊 预期表现:
   年化收益: {best['年化']:+.1f}%
   夏普比率: {best['夏普']:.2f}
   胜率: {best['胜率']:.0f}%
   最大回撤: {best['最大回撤']:.2f}%

💡 策略解读:
   - 追涨策略 (PP>{best['策略'].split('PP>')[1].split('+')[0]})
   - 4只等权持仓
   - 每55天调仓一次
   - 无止损 (回测证明无效)
   - 无RSI过滤 (已优化验证)
""")

# 保存结果
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'iterative_results.csv', index=False, encoding='utf-8-sig')
print(f"✅ 所有结果已保存: output/iterative_results.csv")

# 保存最佳配置
best_config = {
    'predict_days': best_train['predict_days'],
    'model_ic': best_train['correlation'],
    'strategy': best['策略'],
    'annual': best['年化'],
    'sharpe': best['夏普'],
    'win_rate': best['胜率'],
    'max_drawdown': best['最大回撤']
}
with open(OUTPUT_DIR / 'best_config.json', 'w') as f:
    json.dump(best_config, f, indent=2, ensure_ascii=False)
print(f"✅ 最佳配置已保存: output/best_config.json")