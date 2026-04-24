# -*- coding: utf-8 -*-
"""
科创板智能选股策略 - QMT版本
============================
基于LightGBM模型 + price_position追涨过滤
适用: 科创板(688开头)股票
周期: 日线，55个交易日调仓

使用方法:
1. 将本文件放入QMT的策略目录
2. 在QMT中加载此策略
3. 设置参数后启动运行

作者: AI策略助手
版本: v1.0
"""

# ========== 全局变量设置 ==========
# 策略名称
G.strategy_name = "KCB_LGB_Strategy"

# 股票池 - 科创板全市场
G.stock_pool = []

# 模型路径 (相对于QMT数据目录)
G.model_path = "models/lgb_stock_selector.txt"

# ========== 策略参数 ==========
class Param:
    # === 选股参数 ===
    top_n = 4                    # 持仓数量
    price_position_threshold = 0.25  # 价格位置阈值 (追涨)

    # === 特征参数 ===
    lookback_60 = 60             # 60日高低计算周期
    lookback_20 = 20             # 20日布林带周期
    rsi_period = 14              # RSI周期

    # === 交易参数 ===
    rebalance_days = 55          # 调仓周期(交易日)
    position_ratio = 1.0        # 仓位比例 (1.0=满仓)
    stop_loss = 0                # 止损比例 (0=不止损)

    # === 风控参数 ===
    max_stocks = 4               # 最大持仓数
    single_stock_limit = 0.30   # 单只股票最大比例

# ========== 持仓管理 ==========
class Position:
    def __init__(self):
        self.holdings = {}      # {code: {shares, buy_price, buy_date, weight}}
        self.last_rebalance_date = None
        self.trade_count = 0     # 交易计数器

# 全局持仓对象
G.position = Position()

# ========== 初始化函数 ==========
def init():
    """策略初始化"""
    # 设置订阅行情
    set_universe(["KCB"])        # 订阅科创板

    # 设置交易时间段
    set_trade_mode(TradeMode.REAL)  # 实盘模式

    # 输出启动信息
    print("=" * 50)
    print("科创板智能选股策略启动")
    print(f"持仓数量: {Param.top_n}")
    print(f"调仓周期: {Param.rebalance_days}交易日")
    print(f"追涨阈值: PP>{Param.price_position_threshold}")
    print("=" * 50)

    # 加载模型
    try:
        import pickle
        with open(G.model_path, 'rb') as f:
            G.model = pickle.load(f)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        G.model = None

    # 初始化持仓
    G.position = Position()

    # 设置定时器 - 每天收盘前30分钟检查调仓
    #run_interval(target=check_rebalance, interval=[14, 30])  # 14:30检查

# ========== 主bar处理函数 ==========
def handle_bar(bars):
    """每个bar执行一次"""
    bar = bars[0]
    current_date = bar.date
    current_time = bar.time

    # 只在每天14:30后执行 (收盘前30分钟)
    if current_time < 143000:
        return

    # 检查是否需要调仓
    check_and_rebalance(current_date)

# ========== 调仓逻辑 ==========
def check_and_rebalance(current_date):
    """检查并执行调仓"""
    pos = G.position

    # 计算自上次调仓以来的交易日数
    if pos.last_rebalance_date is None:
        # 首次运行，需要调仓
        need_rebalance = True
    else:
        days_since = get_trading_days_count(pos.last_rebalance_date, current_date)
        need_rebalance = days_since >= Param.rebalance_days

    if need_rebalance:
        print(f"\n{'='*50}")
        print(f"调仓日: {current_date}")
        print(f"持仓: {list(pos.holdings.keys())}")

        # 执行选股
        selected = select_stocks(current_date)

        if selected is None or len(selected) == 0:
            print("未选出股票，跳过本次调仓")
            return

        print(f"选中: {[s['code'] for s in selected]}")

        # 执行调仓
        rebalance(current_date, selected)

        # 更新调仓日期
        pos.last_rebalance_date = current_date
        pos.trade_count += 1

        print(f"第{pos.trade_count}次调仓完成")

# ========== 选股函数 ==========
def select_stocks(date):
    """
    选股核心逻辑
    1. 获取科创板当日数据
    2. 计算特征
    3. 模型预测
    4. 追涨过滤 + 排序
    """
    stocks = get_current_data(["KCB"])  # 获取当前所有科创板股票

    if not stocks or len(stocks) == 0:
        return None

    candidates = []

    for code in stocks:
        # 获取历史数据
        hist = get_history_data(code, end_date=date, count=Param.lookback_60 + 20)

        if hist is None or len(hist) < Param.lookback_60:
            continue

        try:
            # 计算特征
            features = calc_features(hist)

            # 检查price_position过滤条件
            latest = features.iloc[-1]
            if latest['price_position'] <= Param.price_position_threshold:
                continue

            # 模型预测
            if G.model is not None:
                X = features[FEATURE_COLS].iloc[-1:].values
                pred_score = G.model.predict(X)[0]
            else:
                pred_score = latest['price_position']  # 无模型时用PP

            candidates.append({
                'code': code,
                'name': get_stock_name(code),
                'price': latest['close'],
                'score': pred_score,
                'price_position': latest['price_position'],
                'rsi': latest['rsi'],
                'features': features
            })
        except Exception as e:
            continue

    if len(candidates) < Param.top_n:
        print(f"候选股票不足: {len(candidates)} < {Param.top_n}")
        return None

    # 按预测分数排序，选取top_n
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected = candidates[:Param.top_n]

    return selected

# ========== 特征计算 ==========
def calc_features(hist):
    """计算策略特征"""
    df = hist.copy()

    # 排序
    df = df.sort_values(['date'])

    # 收益率
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)

    # 均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma_bull'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(int)

    # 布林带
    bb_mean = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std + 1e-8)

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # 成交量
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-8)

    # 波动率
    df['volatility'] = df['ret_1d'].rolling(20).std()

    # 价格位置 (核心指标)
    rolling_low = df['low'].rolling(60).min()
    rolling_high = df['high'].rolling(60).max()
    df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)

    # 20日新高
    high_20_shift = df['high'].rolling(20).max().shift(1)
    df['break_20d'] = (df['close'] >= high_20_shift).astype(int)

    return df

# 特征列名
FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_bull', 'bb_position', 'rsi',
                'vol_ratio', 'volatility', 'price_position', 'break_20d']

# ========== 调仓执行 ==========
def rebalance(date, selected):
    """执行调仓买卖"""
    pos = G.position
    target_codes = [s['code'] for s in selected]

    # === 卖出不在目标名单的持仓 ===
    for code in list(pos.holdings.keys()):
        if code not in target_codes:
            # 卖出
            holding = pos.holdings[code]
            order = sell(code, holding['shares'])
            print(f"卖出: {code} {holding['shares']}股")
            del pos.holdings[code]

    # === 买入目标股票 ===
    # 计算总可用资金
    total_value = get_account_value() * Param.position_ratio
    cash_per_stock = total_value / len(selected)

    for stock in selected:
        code = stock['code']
        price = stock['price']

        # 计算可买数量 (100股整数倍)
        shares = int(cash_per_stock / price / 100) * 100

        if shares > 0:
            # 检查是否已持有
            if code in pos.holdings:
                # 增持
                holding = pos.holdings[code]
                new_shares = holding['shares'] + shares
                order = buy(code, shares, price)
                pos.holdings[code]['shares'] = new_shares
                print(f"增持: {code} {shares}股 @ {price}")
            else:
                # 新买入
                order = buy(code, shares, price)
                pos.holdings[code] = {
                    'shares': shares,
                    'buy_price': price,
                    'buy_date': date,
                    'weight': 1.0 / len(selected)
                }
                print(f"买入: {code} {shares}股 @ {price}")

        # 检查止损
        if Param.stop_loss > 0 and code in pos.holdings:
            holding = pos.holdings[code]
            current_price = get_current_price(code)
            loss_ratio = (current_price - holding['buy_price']) / holding['buy_price']

            if loss_ratio < -Param.stop_loss:
                order = sell(code, holding['shares'])
                print(f"止损: {code} 亏损{loss_ratio*100:.1f}%")
                del pos.holdings[code]

# ========== 辅助函数 ==========
def get_current_price(code):
    """获取当前价"""
    tick = get_tick_data(code)
    if tick:
        return tick['last_price']
    return None

def get_stock_name(code):
    """获取股票名称"""
    info = get_stock_info(code)
    if info:
        return info.get('name', code)
    return code

def get_account_value():
    """获取账户总资产"""
    account = get_account_info()
    if account:
        return account['total_value']
    return 1000000  # 默认100万

# ========== 止损监控(可选) ==========
def monitor_stop_loss():
    """每日收盘后监控止损"""
    pos = G.position

    if Param.stop_loss == 0 or len(pos.holdings) == 0:
        return

    for code in list(pos.holdings.keys()):
        holding = pos.holdings[code]
        current_price = get_current_price(code)

        if current_price is None:
            continue

        loss_ratio = (current_price - holding['buy_price']) / holding['buy_price']

        if loss_ratio < -Param.stop_loss:
            print(f"\n触发止损: {code} 当前价{current_price} 买入价{holding['buy_price']} 亏损{loss_ratio*100:.1f}%")
            order = sell(code, holding['shares'])
            del pos.holdings[code]

# ========== 状态输出 ==========
def print_status():
    """输出策略状态"""
    pos = G.position
    print(f"\n===== 策略状态 =====")
    print(f"持仓数: {len(pos.holdings)}")
    print(f"调仓次数: {pos.trade_count}")
    print(f"上次调仓: {pos.last_rebalance_date}")
    print("持仓明细:")
    for code, holding in pos.holdings.items():
        current = get_current_price(code)
        if current:
            pnl = (current - holding['buy_price']) / holding['buy_price'] * 100
            print(f"  {code}: {holding['shares']}股 成本{holding['buy_price']:.2f} 现价{current:.2f} 盈亏{pnl:+.1f}%")
        else:
            print(f"  {code}: {holding['shares']}股 成本{holding['buy_price']:.2f}")

# ========== 结束函数 ==========
def on_trading_end():
    """收盘后处理"""
    monitor_stop_loss()
    print_status()

# ========== 策略结束 ==========
def stop():
    """策略停止"""
    print("策略停止，正在清理...")
    print_status()
    print("策略已停止")