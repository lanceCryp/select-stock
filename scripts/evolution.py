"""
策略持续进化系统
===============
循环迭代 + 自动记录 + 版本管理

功能:
1. 多维度参数扫描
2. 分层进化 (PP阈值→持仓→周期→买卖信号)
3. 自动生成进化报告
4. 版本对比分析

使用方法:
    uv run python scripts/evolution.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_PATH = "data/kcb_history.csv"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

INITIAL_CASH = 1000000.0
COMMISSION = 0.0003

# ============================================================
# 初始化日志
# ============================================================
LOG_FILE = OUTPUT_DIR / "evolution_log.md"
VERSION_FILE = OUTPUT_DIR / "version_history.json"

def log(msg):
    """打印并记录日志"""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")

def init_log():
    """初始化日志文件"""
    header = f"""# 科创板智能选股策略 - 进化记录
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 版本历程

| 版本 | 描述 | 年化 | 夏普 | 胜率 | 回撤 |
|------|------|------|------|------|------|
"""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(header)

# ============================================================
# 加载数据
# ============================================================
print("=" * 70)
print("🧬 策略持续进化系统")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

print(f"\n📊 数据: {df['code'].nunique()} 只股票")
print(f"   范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

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
    df['future_return'] = df.groupby('code')['close'].pct_change(20).shift(-20)
    return df

FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
                'vol_ratio', 'volatility', 'price_position', 'break_20d']

df = calc_features(df)

val_df = df[df['date'] >= '2024-01-01'].copy()
val_dates = sorted(val_df['date'].unique())

# 训练模型
train_df = df[df['date'] <= '2023-12-31'].dropna(subset=FEATURE_COLS)
X_train = train_df[FEATURE_COLS].values
y_train = train_df['future_return'].values

lgb_train = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
          'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
          'verbose': -1, 'seed': 42}

model = lgb.train(params, lgb_train, num_boost_round=200)
val_df['lgb_pred'] = model.predict(val_df[FEATURE_COLS].values)

# ============================================================
# 回测函数
# ============================================================
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

# ============================================================
# 进化1: PP阈值优化
# ============================================================
log("\n" + "=" * 70)
log("🧬 进化1: PP阈值优化")
log("=" * 70)

best_pp = 0.25
best_pp_sharpe = 0

for pp in np.arange(0.15, 0.45, 0.05):
    conditions = [('price_position', '>', round(pp, 2))]
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func

    result = backtest(val_df, val_dates[::55], make_func(conditions), top_n=4)
    if result:
        log(f"   PP>{pp:.2f}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")
        if result['sharpe'] > best_pp_sharpe:
            best_pp = pp
            best_pp_sharpe = result['sharpe']
            best_pp_result = result

log(f"\n✅ 最佳PP阈值: {best_pp:.2f} (夏普: {best_pp_sharpe:.2f})")

# ============================================================
# 进化2: 持仓数量优化
# ============================================================
log("\n" + "=" * 70)
log("🧬 进化2: 持仓数量优化")
log("=" * 70)

best_top_n = 4
best_sharpe = best_pp_sharpe

for top_n in [2, 3, 4, 5, 6, 8]:
    conditions = [('price_position', '>', best_pp)]
    def make_func(cond, tn):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= tn:
                return filtered.nlargest(tn, 'lgb_pred')
            return d.nlargest(tn, 'lgb_pred')
        return func

    result = backtest(val_df, val_dates[::55], make_func(conditions, top_n), top_n=top_n)
    if result:
        log(f"   PP>{best_pp:.2f} + 持仓{top_n}只: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")
        if result['sharpe'] > best_sharpe:
            best_top_n = top_n
            best_sharpe = result['sharpe']
            best_top_n_result = result

log(f"\n✅ 最佳持仓: {best_top_n}只 (夏普: {best_sharpe:.2f})")

# ============================================================
# 进化3: 调仓周期优化
# ============================================================
log("\n" + "=" * 70)
log("🧬 进化3: 调仓周期优化")
log("=" * 70)

best_hold_days = 55
best_hold_sharpe = best_sharpe

for hold_days in [30, 40, 45, 50, 55, 60, 70, 80]:
    rd = val_dates[::hold_days]
    if len(rd) < 3:
        continue

    conditions = [('price_position', '>', best_pp)]
    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func

    result = backtest(val_df, rd, make_func(conditions), top_n=best_top_n)
    if result:
        log(f"   PP>{best_pp:.2f} + {best_top_n}只 + {hold_days}天: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")
        if result['sharpe'] > best_hold_sharpe:
            best_hold_days = hold_days
            best_hold_sharpe = result['sharpe']
            best_hold_result = result

log(f"\n✅ 最佳周期: {best_hold_days}天 (夏普: {best_hold_sharpe:.2f})")

# ============================================================
# 进化4: RSI买卖信号优化
# ============================================================
log("\n" + "=" * 70)
log("🧬 进化4: RSI买卖信号优化 (买入过滤)")
log("=" * 70)

best_rsi = 0
best_rsi_sharpe = best_hold_sharpe

for rsi_th in [0, 50, 55, 60, 65, 70, 75]:
    conditions = [('price_position', '>', best_pp)]
    if rsi_th > 0:
        conditions.append(('rsi', '<', rsi_th))

    def make_func(cond):
        def func(d, n):
            mask = apply_rules(d, cond)
            filtered = d[mask]
            if len(filtered) >= n:
                return filtered.nlargest(n, 'lgb_pred')
            return d.nlargest(n, 'lgb_pred')
        return func

    result = backtest(val_df, val_dates[::best_hold_days], make_func(conditions), top_n=best_top_n)
    if result:
        rsi_str = f"RSI<{rsi_th}" if rsi_th > 0 else "无过滤"
        log(f"   PP>{best_pp:.2f} + {best_top_n}只 + {rsi_str}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")
        if result['sharpe'] > best_rsi_sharpe:
            best_rsi = rsi_th
            best_rsi_sharpe = result['sharpe']
            best_rsi_result = result

log(f"\n✅ 最佳RSI买入过滤: {'无' if best_rsi == 0 else f'<{best_rsi}'} (夏普: {best_rsi_sharpe:.2f})")

# ============================================================
# 进化5: 卖出信号优化
# ============================================================
log("\n" + "=" * 70)
log("🧬 进化5: 卖出信号优化 (动态止盈)")
log("=" * 70)

def backtest_with_exit(data, dates, strategy_func, top_n=4, exit_rsi=0, exit_pp=0):
    """带卖出信号的回测"""
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

        # 检查卖出信号
        positions_to_sell = []
        for code, pos in list(positions.items()):
            # RSI超买信号
            if exit_rsi > 0:
                code_data = day_data[day_data['code'] == code]
                if len(code_data) > 0 and code_data['rsi'].values[0] > exit_rsi:
                    positions_to_sell.append(code)
                    continue
            # PP反转信号
            if exit_pp > 0:
                code_data = day_data[day_data['code'] == code]
                if len(code_data) > 0 and code_data['price_position'].values[0] < exit_pp:
                    positions_to_sell.append(code)
                    continue
            # 不在目标名单
            if code not in target_codes:
                positions_to_sell.append(code)

        for code in positions_to_sell:
            pos = positions[code]
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
            if code in positions:
                continue
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

conditions = [('price_position', '>', best_pp)]
if best_rsi > 0:
    conditions.append(('rsi', '<', best_rsi))

def make_func(cond):
    def func(d, n):
        mask = apply_rules(d, cond)
        filtered = d[mask]
        if len(filtered) >= n:
            return filtered.nlargest(n, 'lgb_pred')
        return d.nlargest(n, 'lgb_pred')
    return func

best_exit_rsi = 0
best_exit_pp = 0
best_exit_sharpe = best_rsi_sharpe

# 测试RSI止盈
for exit_rsi in [70, 75, 80, 85]:
    result = backtest_with_exit(val_df, val_dates[::best_hold_days], make_func(conditions),
                                 top_n=best_top_n, exit_rsi=exit_rsi, exit_pp=0)
    if result:
        log(f"   RSI止盈>{exit_rsi}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")
        if result['sharpe'] > best_exit_sharpe:
            best_exit_rsi = exit_rsi
            best_exit_sharpe = result['sharpe']
            best_exit_result = result

# 测试PP止盈
for exit_pp in [0.3, 0.35, 0.4, 0.45]:
    result = backtest_with_exit(val_df, val_dates[::best_hold_days], make_func(conditions),
                                 top_n=best_top_n, exit_rsi=0, exit_pp=exit_pp)
    if result:
        log(f"   PP止盈<{exit_pp}: 年化{result['annual']*100:+.1f}% | 夏普{result['sharpe']:.2f} | 胜率{result['win_rate']*100:.0f}%")
        if result['sharpe'] > best_exit_sharpe:
            best_exit_pp = exit_pp
            best_exit_sharpe = result['sharpe']
            best_exit_result = result

exit_str = ""
if best_exit_rsi > 0:
    exit_str = f"RSI>{best_exit_rsi}"
elif best_exit_pp > 0:
    exit_str = f"PP<{best_exit_pp}"
else:
    exit_str = "无止盈"

log(f"\n✅ 最佳卖出信号: {exit_str} (夏普: {best_exit_sharpe:.2f})")

# ============================================================
# 生成最终版本
# ============================================================
log("\n" + "=" * 70)
log("🏆 最终版本配置")
log("=" * 70)

final_config = {
    'version': 'v_final',
    'pp_threshold': float(best_pp),
    'top_n': int(best_top_n),
    'hold_days': int(best_hold_days),
    'rsi_filter': int(best_rsi) if best_rsi > 0 else None,
    'exit_rsi': int(best_exit_rsi) if best_exit_rsi > 0 else None,
    'exit_pp': float(best_exit_pp) if best_exit_pp > 0 else None
}

# 计算最终绩效
conditions = [('price_position', '>', best_pp)]
if best_rsi > 0:
    conditions.append(('rsi', '<', best_rsi))

def make_final_func(cond):
    def func(d, n):
        mask = apply_rules(d, cond)
        filtered = d[mask]
        if len(filtered) >= n:
            return filtered.nlargest(n, 'lgb_pred')
        return d.nlargest(n, 'lgb_pred')
    return func

if best_exit_rsi > 0:
    final_result = backtest_with_exit(val_df, val_dates[::best_hold_days], make_final_func(conditions),
                                       top_n=best_top_n, exit_rsi=best_exit_rsi, exit_pp=0)
elif best_exit_pp > 0:
    final_result = backtest_with_exit(val_df, val_dates[::best_hold_days], make_final_func(conditions),
                                       top_n=best_top_n, exit_rsi=0, exit_pp=best_exit_pp)
else:
    final_result = backtest(val_df, val_dates[::best_hold_days], make_final_func(conditions), top_n=best_top_n)

log(f"""
📋 版本配置:
   PP阈值: >{best_pp:.2f}
   持仓数量: {best_top_n}只
   调仓周期: {best_hold_days}天
   买入RSI过滤: {'无' if best_rsi == 0 else f'<{best_rsi}'}
   卖出信号: {exit_str}

📊 最终绩效:
   年化收益: {final_result['annual']*100:+.1f}%
   夏普比率: {final_result['sharpe']:.2f}
   胜率: {final_result['win_rate']*100:.0f}%
   最大回撤: {final_result['max_drawdown']*100:.2f}%
   最终资金: {final_result['final_value']:,.0f}
""")

# ============================================================
# 保存版本历史
# ============================================================
version_history = []
if VERSION_FILE.exists():
    with open(VERSION_FILE, 'r') as f:
        version_history = json.load(f)

version_history.append({
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'version': f'v{len(version_history)+1}',
    'config': final_config,
    'metrics': {
        'annual': float(final_result['annual']),
        'sharpe': float(final_result['sharpe']),
        'win_rate': float(final_result['win_rate']),
        'max_drawdown': float(final_result['max_drawdown']),
        'final_value': float(final_result['final_value'])
    }
})

with open(VERSION_FILE, 'w') as f:
    json.dump(version_history, f, indent=2, ensure_ascii=False)

# ============================================================
# 生成报告
# ============================================================
report = f"""# 策略进化报告

## 进化日期: {datetime.now().strftime('%Y-%m-%d')}

### 进化过程

1. **PP阈值优化**: 最佳 {best_pp:.2f}
2. **持仓数量优化**: 最佳 {best_top_n}只
3. **调仓周期优化**: 最佳 {best_hold_days}天
4. **RSI买入过滤**: {'无' if best_rsi == 0 else f'<{best_rsi}'}
5. **卖出信号**: {exit_str}

### 最终配置

| 参数 | 值 |
|------|-----|
| PP阈值 | >{best_pp:.2f} |
| 持仓数量 | {best_top_n}只 |
| 调仓周期 | {best_hold_days}天 |
| 买入RSI过滤 | {'无' if best_rsi == 0 else f'<{best_rsi}'} |
| 卖出信号 | {exit_str} |

### 绩效指标

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

report_path = REPORT_DIR / f"evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 进化报告已保存: {report_path}")
print(f"✅ 版本历史已保存: {VERSION_FILE}")