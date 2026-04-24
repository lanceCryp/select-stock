"""
LightGBM 选股模型训练
====================
功能：
1. 向量化计算特征（快速）
2. 训练 LightGBM 回归模型
3. 分析特征重要性
4. 输出最优参数

使用方法：
    uv run python scripts/train_lgb.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
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

# 训练参数
TRAIN_END = '2023-12-31'  # 训练集截止日期
VAL_START = '2024-01-01'  # 验证集开始日期
TOP_N = 20                 # 持仓数量
HOLD_DAYS = 20             # 持有天数

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("LightGBM 选股模型训练")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = df['pctChg'].fillna(0)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

print(f"\n📊 数据加载完成")
print(f"   股票数量: {df['code'].nunique()}")
print(f"   数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"   总记录数: {len(df):,}")

# ============================================================
# 2. 向量化计算特征（快速）
# ============================================================
print("\n🔧 计算技术指标...")

def calc_features_vectorized(df):
    """向量化计算所有特征，速度比 groupby.apply 快 5-10 倍"""
    # 按 code 和 date 排序
    df = df.sort_values(['code', 'date'])

    # 基础收益率
    df['ret_1d'] = df.groupby('code')['close'].pct_change(1)
    df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    df['ret_60d'] = df.groupby('code')['close'].pct_change(60)

    # 均线
    df['ma5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60).mean())

    # 均线多头排列
    df['ma_bull'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(int)

    # 布林带
    bb_mean = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    bb_std = df.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std + 1e-8)

    # RSI (使用 groupby apply 计算)
    def calc_rsi(group):
        delta = group['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    df['rsi'] = df.groupby('code', group_keys=False).apply(calc_rsi)

    # 成交量指标
    df['vol_ma5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean())
    df['vol_ma20'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-8)

    # 波动率
    df['volatility'] = df.groupby('code')['ret_1d'].transform(lambda x: x.rolling(20).std())

    # 价格相对位置
    rolling_low = df.groupby('code')['low'].transform(lambda x: x.rolling(60).min())
    rolling_high = df.groupby('code')['high'].transform(lambda x: x.rolling(60).max())
    df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)

    # 20日高点突破
    high_20_shift = df.groupby('code')['high'].transform(lambda x: x.rolling(20).max().shift(1))
    df['break_20d'] = (df['close'] >= high_20_shift).astype(int)

    # 持仓期收益（未来20日收益率，作为标签）
    df['future_return'] = df.groupby('code')['close'].pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

    return df

df = calc_features_vectorized(df)
print("   ✅ 特征计算完成")

# ============================================================
# 3. 构建特征矩阵
# ============================================================
print("\n📦 构建特征矩阵...")

FEATURE_COLS = [
    'ret_1d', 'ret_5d', 'ret_20d',
    'ma_bull', 'bb_position', 'rsi',
    'vol_ratio', 'volatility', 'price_position', 'break_20d'
]

# 只保留有完整数据的行
df_model = df.dropna(subset=FEATURE_COLS + ['future_return']).copy()

# 划分训练集和验证集
train_df = df_model[df_model['date'] <= TRAIN_END].copy()
val_df = df_model[df_model['date'] >= VAL_START].copy()

X_train = train_df[FEATURE_COLS].values
y_train = train_df['future_return'].values
X_val = val_df[FEATURE_COLS].values
y_val = val_df['future_return'].values

print(f"   训练集: {len(train_df):,} 条")
print(f"   验证集: {len(val_df):,} 条")
print(f"   特征数: {len(FEATURE_COLS)}")

# ============================================================
# 4. 训练 LightGBM
# ============================================================
print("\n🤖 训练 LightGBM 模型...")

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
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print(f"\n   最佳迭代: {model.best_iteration}")

# ============================================================
# 5. 模型评估
# ============================================================
print("\n📈 模型评估...")

train_pred = model.predict(X_train, num_iteration=model.best_iteration)
val_pred = model.predict(X_val, num_iteration=model.best_iteration)

train_r2 = r2_score(y_train, train_pred)
val_r2 = r2_score(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
corr = np.corrcoef(val_pred, y_val)[0, 1]

print(f"   训练集 R²: {train_r2:.4f}")
print(f"   验证集 R²: {val_r2:.4f}")
print(f"   验证集 RMSE: {val_rmse:.4f}")
print(f"   预测相关性: {corr:.4f}")

# 特征重要性
print("\n📊 特征重要性:")
importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

max_imp = importance['importance'].max()
for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] / max_imp * 20)
    print(f"   {row['feature']:15s} {bar} ({row['importance']})")

# ============================================================
# 6. IC 分析
# ============================================================
print("\n📉 因子 IC 分析...")

val_df = val_df.copy()
val_df['pred'] = val_pred

daily_ic = val_df.groupby('date').apply(
    lambda x: x['pred'].corr(x['future_return']) if len(x) > 5 else np.nan
).dropna()

ic_mean = daily_ic.mean()
ic_std = daily_ic.std()
ic_ir = ic_mean / ic_std if ic_std > 0 else 0
ic_pos_rate = (daily_ic > 0).mean()

print(f"   日均 IC: {ic_mean:.4f}")
print(f"   IC 标准差: {ic_std:.4f}")
print(f"   IC_IR: {ic_ir:.4f}")
print(f"   正 IC 天数占比: {ic_pos_rate*100:.1f}%")

# ============================================================
# 7. 回测
# ============================================================
print("\n💰 回测验证集绩效...")

val_dates = sorted(val_df['date'].unique())
# 每 HOLD_DAYS 个交易日调仓一次
rebalance_dates = val_dates[::HOLD_DAYS]

print(f"   调仓周期: {HOLD_DAYS} 天")
print(f"   调仓次数: {len(rebalance_dates)}")

results = []
for rd in rebalance_dates:
    day_data = val_df[val_df['date'] == rd].copy()
    if len(day_data) < TOP_N:
        continue
    selected = day_data.nlargest(TOP_N, 'pred')
    ret = selected['future_return'].mean()
    results.append({
        'date': rd,
        'return': ret,
        'n_selected': len(selected)
    })

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    total_ret = (1 + results_df['return']).prod() - 1
    avg_ret = results_df['return'].mean()
    std_ret = results_df['return'].std()
    annual_ret = avg_ret * (252 / HOLD_DAYS)
    sharpe = avg_ret / std_ret * np.sqrt(252 / HOLD_DAYS) if std_ret > 0 else 0
    win_rate = (results_df['return'] > 0).mean()

    print(f"\n   ╔══════════════════════════════════════╗")
    print(f"   ║        LightGBM 选股回测结果          ║")
    print(f"   ╠══════════════════════════════════════╣")
    print(f"   ║  调仓次数:    {len(results_df):>5}                ║")
    print(f"   ║  总收益:      {total_ret*100:>8.2f}%             ║")
    print(f"   ║  年化收益:    {annual_ret*100:>8.2f}%             ║")
    print(f"   ║  夏普比率:    {sharpe:>8.2f}               ║")
    print(f"   ║  胜率:        {win_rate*100:>8.1f}%             ║")
    print(f"   ╚══════════════════════════════════════╝")

    # 每期收益明细
    print(f"\n   每期收益:")
    cumret = 1.0
    for _, row in results_df.iterrows():
        cumret *= (1 + row['return'])
        marker = "🟢" if row['return'] > 0 else "🔴"
        print(f"   {marker} {row['date'].date()}: {row['return']*100:>+7.2f}%  累计: {cumret*100:>10.2f}%")

# ============================================================
# 8. 保存模型
# ============================================================
model_path = MODEL_DIR / "lgb_stock_selector.txt"
model.save_model(str(model_path))
print(f"\n   ✅ 模型已保存: {model_path}")

# 保存参数
best_params = {
    'features': FEATURE_COLS,
    'best_iteration': model.best_iteration,
    'train_r2': float(train_r2),
    'val_r2': float(val_r2),
    'val_rmse': float(val_rmse),
    'correlation': float(corr),
    'ic_mean': float(ic_mean),
    'ic_ir': float(ic_ir),
    'ic_pos_rate': float(ic_pos_rate),
    'annual_ret': float(annual_ret) if len(results_df) > 0 else None,
    'sharpe': float(sharpe) if len(results_df) > 0 else None,
    'win_rate': float(win_rate) if len(results_df) > 0 else None,
}

params_path = MODEL_DIR / "lgb_params.json"
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"   ✅ 参数已保存: {params_path}")

# ============================================================
# 9. 结论
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"""
模型训练完成！

🎯 最有效的特征:
{importance.iloc[0]['feature']} (重要性: {importance.iloc[0]['importance']})
{importance.iloc[1]['feature']} (重要性: {importance.iloc[1]['importance']})
{importance.iloc[2]['feature']} (重要性: {importance.iloc[2]['importance']})

📊 模型预测能力:
   IC = {ic_mean:.4f} (IC > 0.02 表示有效)
   相关性 = {corr:.4f}

💰 回测表现:
   年化收益 = {annual_ret*100:.1f}% (如果为负则需要调整策略)
   夏普比率 = {sharpe:.2f}
""")

print("=" * 60)