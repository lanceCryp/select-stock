"""
数据下载脚本 - 全市场版
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
START_DATE = "2022-01-01"
END_DATE = "2026-04-22"  # 用最近交易日

BATCH_SIZE = 500


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_last_trading_day():
    """获取最近交易日（向前推）"""
    d = datetime.now()
    for _ in range(10):
        if d.weekday() < 5:  # 周一到周五
            return d.strftime('%Y-%m-%d')
        d -= timedelta(days=1)
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


def download_all_stocks():
    """下载全市场A股列表（排除ST）"""
    log("=" * 50)
    log("1. 下载全市场 A 股列表（非ST）")
    log("=" * 50)

    # 自动获取最近交易日
    last_day = get_last_trading_day()
    log(f"  查询日期: {last_day}")

    bs.login()
    rs = bs.query_all_stock(day=last_day)
    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    log(f"  查询到 {len(df)} 条记录")

    # 只保留 A 股 (sh.6xxxxx, sz.0xxxxx, sz.3xxxxx)
    df = df[df['code'].str.match(r'^sh\.6|^sz\.0|^sz\.3')]
    log(f"  A股数量: {len(df)}")

    # 排除 ST 股（code_name 包含 ST）
    df = df[~df['code_name'].str.contains('ST', na=False)]
    log(f"  排除ST后: {len(df)} 只")

    df.to_csv(DATA_DIR / "stock_list_all.csv", index=False, encoding="utf-8")
    log(f"  保存到: data/stock_list_all.csv")
    log(f"  示例: {df['code'].head(5).tolist()}")

    return df['code'].tolist()


def download_history_data(codes: list, filename: str = "history_all_full"):
    """批量下载历史行情（分批避免超时）"""
    if not codes:
        log("  代码列表为空，跳过")
        return

    total = len(codes)
    log("=" * 50)
    log(f"2. 下载历史行情数据")
    log(f"   股票数量: {total}")
    log(f"   时间范围: {START_DATE} ~ {END_DATE}")
    log("=" * 50)

    all_data = []
    batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(batch_count):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_codes = codes[start_idx:end_idx]

        log(f"  批次 {batch_idx + 1}/{batch_count}: 股票 {start_idx}-{end_idx}")

        bs.login()
        for i, code in enumerate(batch_codes):
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="d",
                adjustflag="3"
            )
            while rs.error_code == '0' and rs.next():
                all_data.append(rs.get_row_data())

            if (i + 1) % 100 == 0:
                log(f"    进度: {i + 1}/{len(batch_codes)}")
            time.sleep(0.03)
        bs.logout()

        log(f"  批次完成，累计 {len(all_data)} 条记录")

    if all_data:
        df = pd.DataFrame(all_data, columns=rs.fields)
        df.to_csv(DATA_DIR / f"{filename}.csv", index=False, encoding="utf-8")
        size_mb = len(df) * 50 / 1024 / 1024
        log(f"  保存到: data/{filename}.csv, {len(df)} 条 ({size_mb:.1f} MB)")
    else:
        log("  没有获取到数据")


def download_index_components():
    """下载指数成分股"""
    log("=" * 50)
    log("3. 下载指数成分股")
    log("=" * 50)

    # 沪深300
    bs.login()
    rs = bs.query_hs300_stocks()
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    bs.logout()
    df_hs300 = pd.DataFrame(data, columns=rs.fields)
    df_hs300.to_csv(DATA_DIR / "hs300.csv", index=False, encoding="utf-8")
    hs300_codes = df_hs300['code'].tolist()
    log(f"  沪深300: {len(hs300_codes)} 只")

    # 中证500
    bs.login()
    rs = bs.query_zz500_stocks()
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    bs.logout()
    df_zz500 = pd.DataFrame(data, columns=rs.fields)
    df_zz500.to_csv(DATA_DIR / "zz500.csv", index=False, encoding="utf-8")
    zz500_codes = df_zz500['code'].tolist()
    log(f"  中证500: {len(zz500_codes)} 只")

    return hs300_codes, zz500_codes


def main():
    log("=" * 60)
    log("A股数据下载脚本 - 全市场版")
    log("=" * 60)
    log(f"数据保存目录: {DATA_DIR}")
    log(f"时间范围: {START_DATE} ~ {END_DATE}")
    log("")

    DATA_DIR.mkdir(exist_ok=True)

    # 1. 获取全市场股票列表
    all_codes = download_all_stocks()
    log("")

    # 2. 下载指数成分股
    hs300_codes, zz500_codes = download_index_components()
    log("")

    # 3. 下载全市场历史行情（预计 30-40 分钟）
    log(f"开始下载全市场 {len(all_codes)} 只股票行情...")
    log(f"预计耗时: {len(all_codes) * 0.035 / 60:.0f} 分钟")
    download_history_data(all_codes, "history_all_full")
    log("")

    # 4. 下载沪深300（更快，用于快速回测）
    log(f"开始下载沪深300 {len(hs300_codes)} 只股票...")
    download_history_data(hs300_codes, "history_hs300_full")
    log("")

    log("=" * 60)
    log("数据下载完成！")
    log("=" * 60)
    for f in sorted(DATA_DIR.glob("*.csv")):
        size = f.stat().st_size / 1024 / 1024
        log(f"  {f.name} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
