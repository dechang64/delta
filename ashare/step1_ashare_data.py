#!/usr/bin/env python3
"""
Step 1: Download A-share data for cross-market comparison.
- CSI 300 constituents (or top 200 by market cap)
- 20 years of monthly returns (2005-01 to 2024-12)
- Save in same format as US data for pipeline compatibility

Author: Siyi / 2026-06-05
"""

import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import time

OUT = "/home/z/my-project/delta_ashare"

# ── Get CSI 300 Constituents ──
print("Fetching CSI 300 constituents...")
cons_df = ak.index_stock_cons_csindex(symbol='000300')
tickers = cons_df['成分券代码'].tolist()
names = cons_df['成分券名称'].tolist()
print(f"CSI 300 stocks: {len(tickers)}")

# ── Download monthly data ──
START = "20050101"
END = "20241231"

all_data = {}
failed = []

for i, (ticker, name) in enumerate(zip(tickers, names)):
    if (i + 1) % 20 == 0 or i == 0:
        print(f"[{i+1}/{len(tickers)}] Downloading {ticker} ({name})...")
    
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=ticker, period='monthly',
                start_date=START, end_date=END, adjust='qfq'
            )
            if df.empty or len(df) < 24:
                failed.append(ticker)
                break
            
            # Parse into our format
            monthly = {}
            for _, row in df.iterrows():
                date_str = str(row['日期'])[:7]  # "2005-01"
                ret = float(row['涨跌幅']) / 100.0  # Convert to decimal
                vol = float(row['成交量']) if pd.notna(row['成交量']) else 0
                monthly[date_str] = {
                    "return": ret,
                    "volume": vol,
                    "close": float(row['收盘']),
                }
            all_data[ticker] = monthly
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  Failed: {ticker} - {e}")
                failed.append(ticker)
    
    time.sleep(0.3)  # Rate limit

print(f"\nDownloaded: {len(all_data)} stocks, Failed: {len(failed)}")

# ── Filter: require at least 120 months of data ──
valid = {t: d for t, d in all_data.items() if len(d) >= 120}
print(f"Valid (≥120 months): {len(valid)} stocks")

# ── Save ──
output = {
    "market": "A-share (CSI 300)",
    "period": "2005-01 to 2024-12",
    "n_stocks": len(valid),
    "data": valid,
}

with open(os.path.join(OUT, "ashare_monthly_returns.json"), "w") as f:
    json.dump(output, f, ensure_ascii=False)

# ── Summary ──
all_months = set()
for t in valid:
    all_months.update(valid[t].keys())
all_months = sorted(all_months)
print(f"Date range: {all_months[0]} to {all_months[-1]}")
print(f"Total months: {len(all_months)}")

# Quarterly months
quarterly = all_months[::3]
print(f"Quarterly months: {len(quarterly)}")
print(f"Expected LLM calls: {len(valid)} × {len(quarterly)} × 3 = {len(valid) * len(quarterly) * 3}")

# Save ticker-name mapping
ticker_names = {}
for t, n in zip(tickers, names):
    if t in valid:
        ticker_names[t] = n
with open(os.path.join(OUT, "ashare_ticker_names.json"), "w") as f:
    json.dump(ticker_names, f, ensure_ascii=False, indent=2)

print(f"\nSaved to: {OUT}/ashare_monthly_returns.json")
print(f"Ticker names: {OUT}/ashare_ticker_names.json")
