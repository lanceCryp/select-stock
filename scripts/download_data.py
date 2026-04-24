"""
数据下载脚本
运行方式: uv run python scripts/download_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

# 配置
DATA_DIR = Path("data")
START_DATE = "2023-01-01"  # 拉取 1 年数据用于回测
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 股票池
HS300_CODES = []  # 沪深300
ZZ500_CODES = []  # 中证500


def log(msg):
    """打印带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def download_stock_list():
    """下载股票列表"""
    log("=" * 50)
    log("1. 下载 A 股股票列表")
    log("=" * 50)

    bs.login()
    rs = bs.query_all_stock(day='2024-12-31')  # 用历史日期
    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    # 只保留 A 股 (sh.6xxxxx, sz.0xxxxx, sz.3xxxxx)
    df = df[df['code'].str.match(r'^sh\.6|^sz\.0|^sz\.3')]
    df.to_csv(DATA_DIR / "stock_list.csv", index=False, encoding="utf-8")
    log(f"  保存到: data/stock_list.csv, 共 {len(df)} 只股票")
    return df


def download_hs300():
    """下载沪深300成分股"""
    log("=" * 50)
    log("2. 下载沪深300成分股")
    log("=" * 50)

    bs.login()
    rs = bs.query_hs300_stocks()
    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    df.to_csv(DATA_DIR / "hs300.csv", index=False, encoding="utf-8")
    log(f"  保存到: data/hs300.csv, 共 {len(df)} 只股票")

    codes = df['code'].tolist()
    return codes


def download_zz500():
    """下载中证500成分股"""
    log("=" * 50)
    log("3. 下载中证500成分股")
    log("=" * 50)

    bs.login()
    rs = bs.query_zz500_stocks()
    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    df.to_csv(DATA_DIR / "zz500.csv", index=False, encoding="utf-8")
    log(f"  保存到: data/zz500.csv, 共 {len(df)} 只股票")

    codes = df['code'].tolist()
    return codes


def download_history_data(codes: list, stock_type: str = "all"):
    """
    下载历史行情数据

    Args:
        codes: 股票代码列表，如 ['sz.000001', 'sh.600000']
        stock_type: 'hs300', 'zz500', 'all'
    """
    if not codes:
        log("  代码列表为空，跳过")
        return

    log("=" * 50)
    log(f"4. 下载历史行情数据 ({stock_type})")
    log(f"   股票数量: {len(codes)}")
    log(f"   时间范围: {START_DATE} ~ {END_DATE}")
    log("=" * 50)

    bs.login()
    all_data = []
    total = len(codes)

    for i, code in enumerate(codes):
        if i % 20 == 0:
            log(f"  进度: {i}/{total} ({i/total*100:.1f}%)")

        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="d",
            adjustflag="3"  # 不复权
        )

        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            all_data.append(row)

        # 避免请求过快
        time.sleep(0.05)

    bs.logout()

    if all_data:
        df = pd.DataFrame(all_data, columns=rs.fields)
        filename = f"history_{stock_type}.csv"
        df.to_csv(DATA_DIR / filename, index=False, encoding="utf-8")
        log(f"  保存到: data/{filename}, 共 {len(df)} 条记录")
    else:
        log("  没有获取到数据")


def download_financial_data(codes: list):
    """下载财务数据（ROE、PE、PB等）"""
    if not codes:
        log("  代码列表为空，跳过")
        return

    log("=" * 50)
    log("5. 下载财务指标数据")
    log(f"   股票数量: {len(codes)}")
    log("=" * 50)

    bs.login()
    all_data = []
    total = len(codes)

    for i, code in enumerate(codes):
        if i % 20 == 0:
            log(f"  进度: {i}/{total} ({i/total*100:.1f}%)")

        rs = bs.query_profit_data(code=code, year=2024, quarter=4)
        while rs.error_code == '0' and rs.next():
            all_data.append(rs.get_row_data())

        time.sleep(0.05)

    bs.logout()

    if all_data:
        df = pd.DataFrame(all_data, columns=rs.fields)
        df.to_csv(DATA_DIR / "financial_profit.csv", index=False, encoding="utf-8")
        log(f"  保存到: data/financial_profit.csv, 共 {len(df)} 条记录")


def download_balance_sheet(codes: list):
    """下载资产负债表"""
    if not codes:
        log("  代码列表为空，跳过")
        return

    log("=" * 50)
    log("6. 下载资产负债表")
    log(f"   股票数量: {len(codes)}")
    log("=" * 50)

    bs.login()
    all_data = []
    total = len(codes)

    for i, code in enumerate(codes):
        if i % 20 == 0:
            log(f"  进度: {i}/{total} ({i/total*100:.1f}%)")

        rs = bs.query_balance_sheet(code=code, year=2024, quarter=4)
        while rs.error_code == '0' and rs.next():
            all_data.append(rs.get_row_data())

        time.sleep(0.05)

    bs.logout()

    if all_data:
        df = pd.DataFrame(all_data, columns=rs.fields)
        df.to_csv(DATA_DIR / "balance_sheet.csv", index=False, encoding="utf-8")
        log(f"  保存到: data/balance_sheet.csv, 共 {len(df)} 条记录")


def download_dividend_data(codes: list):
    """下载分红数据"""
    if not codes:
        log("  代码列表为空，跳过")
        return

    log("=" * 50)
    log("7. 下载分红数据")
    log(f"   股票数量: {len(codes)}")
    log("=" * 50)

    bs.login()
    all_data = []
    total = len(codes)

    for i, code in enumerate(codes):
        if i % 20 == 0:
            log(f"  进度: {i}/{total} ({i/total*100:.1f}%)")

        rs = bs.query_dividend_data(code=code, year=2024)
        while rs.error_code == '0' and rs.next():
            all_data.append(rs.get_row_data())

        time.sleep(0.05)

    bs.logout()

    if all_data:
        df = pd.DataFrame(all_data, columns=rs.fields)
        df.to_csv(DATA_DIR / "dividend.csv", index=False, encoding="utf-8")
        log(f"  保存到: data/dividend.csv, 共 {len(df)} 条记录")


def main():
    """主函数"""
    log("=" * 60)
    log("A股数据下载脚本")
    log("=" * 60)
    log(f"数据保存目录: {DATA_DIR}")
    log(f"时间范围: {START_DATE} ~ {END_DATE}")
    log("")

    # 创建目录
    DATA_DIR.mkdir(exist_ok=True)

    # 1. 下载股票列表
    stock_list = download_stock_list()
    log("")

    # 2. 下载指数成分股
    hs300_codes = download_hs300()
    log("")

    zz500_codes = download_zz500()
    log("")

    # 合并股票池（沪深300 + 中证500，去重）
    all_codes = list(set(hs300_codes + zz500_codes))
    log(f"合并后股票池大小: {len(all_codes)} 只")
    log("")

    # 3-4. 下载历史行情（先做沪深300，数据量小一点）
    download_history_data(hs300_codes[:100], "hs300_sample")
    log("")

    # 如果你想下载全部，可以取消下面的注释
    # download_history_data(zz500_codes[:100], "zz500_sample")
    # log("")
    # download_history_data(all_codes[:300], "all_sample")
    # log("")

    # 5-7. 财务数据（可选，数据量大，可以先跳过）
    # download_financial_data(hs300_codes[:50])
    # log("")
    # download_balance_sheet(hs300_codes[:50])
    # log("")
    # download_dividend_data(hs300_codes[:50])
    # log("")

    log("=" * 60)
    log("数据下载完成！")
    log("=" * 60)
    log("")
    log("下载的文件:")
    for f in DATA_DIR.glob("*.csv"):
        size = f.stat().st_size / 1024 / 1024  # MB
        log(f"  {f.name} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
