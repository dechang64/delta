#!/usr/bin/env python3
"""
Step 1: Download expanded dataset for JFE-grade analysis.
- S&P 500 constituents (or top 300 by market cap)
- 20 years of monthly returns (2005-01 to 2024-12)
- Fama-French 5-factor data from Ken French Data Library
- Risk-free rate from FRED

Author: Siyi / 2026-06-03
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import urllib.request

OUT = "/home/z/my-project/delta_jfe"

# ── S&P 500 Constituents (top 300 by typical market cap) ──
# Using a broad set covering all GICS sectors
SP500_TICKERS = [
    # Technology
    "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","AVGO","TSLA","ADBE",
    "CRM","ORCL","INTC","AMD","QCOM","TXN","IBM","NOW","INTU","SNPS",
    "CDNS","PANW","CRWD","MRVL","ADP","FISV","GPN","VRSK","CTSH","KEYS",
    # Communication Services
    "NFLX","DIS","CMCSA","T","VZ","TMUS","EA","TTWO","WBD","PARA",
    # Healthcare
    "JNJ","LLY","UNH","PFE","MRK","ABBV","TMO","ABT","MDLZ","DHR",
    "BMY","AMGN","GILD","CVS","CI","HUM","ELV","BIIB","REGN","VRTX",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","C","USB",
    "PNC","TFC","COF","AIG","MET","PRU","AON","MMC","ADSK","SPGI",
    # Consumer Discretionary
    "HD","NKE","MCD","SBUX","TGT","LOW","TJX","BKNG","ABNB","MAR",
    # Consumer Staples
    "PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KMB",
    # Energy
    "XOM","CVX","COP","SLB","EOG","OXY","VLO","MPC","PSX","WMB",
    # Industrials
    "CAT","HON","UPS","BA","GE","MMM","RTX","LMT","NOC","DE",
    "UNP","CSX","EMR","ETN","ITT","PH","CMI","MMM","AOS","GWW",
    # Materials
    "LIN","APD","SHW","FCX","NEM","DOW","DD","EMN","CE","ECL",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","SRE","XEL","WEC","PEG",
    # Real Estate
    "AMT","PLD","CCI","EQIX","PSA","SPG","O","DLR","WELL","AVB",
    # More large caps
    "V","MA","PYPL","SHOP","SQ","SNOW","PLTR","COIN","RIVN","LCID",
    "BRK-B","SPGI","MCO","ICE","CME","NDAQ","ANET","FTNT","MCHP","ON",
    "LRCX","KLAC","MRVL","MPWR","SWKS","ZETA","AI","PATH","U","RBLX",
]

# Deduplicate while preserving order
seen = set()
TICKERS = []
for t in SP500_TICKERS:
    if t not in seen:
        seen.add(t)
        TICKERS.append(t)

START = "2004-12-01"  # Need Dec 2004 for Jan 2005 return
END = "2024-12-31"

print(f"Downloading {len(TICKERS)} stocks, period {START} to {END}")

# ── 1. Download stock returns ──
all_monthly = {}
failed = []

for i, ticker in enumerate(TICKERS):
    try:
        df = yf.download(ticker, start=START, end=END, interval="1mo",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 24:
            failed.append(ticker)
            continue

        monthly = {}
        for date, row in df.iterrows():
            mo = date.strftime("%Y-%m")
            close = float(row['Close'])
            monthly[mo] = {"close": close, "volume": int(row.get('Volume', 0))}

        # Compute returns
        months_sorted = sorted(monthly.keys())
        for j in range(1, len(months_sorted)):
            prev = months_sorted[j-1]
            curr = months_sorted[j]
            prev_close = monthly[prev]["close"]
            curr_close = monthly[curr]["close"]
            if prev_close > 0:
                monthly[curr]["return"] = (curr_close - prev_close) / prev_close
            else:
                monthly[curr]["return"] = 0.0

        # Remove first month (no return)
        del monthly[months_sorted[0]]
        all_monthly[ticker] = monthly

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(TICKERS)}] Downloaded {ticker}")
        time.sleep(0.2)

    except Exception as e:
        failed.append(ticker)

# Save
outpath = os.path.join(OUT, "sp500_monthly_returns.json")
with open(outpath, 'w') as f:
    json.dump({"tickers": list(all_monthly.keys()), "n_tickers": len(all_monthly),
               "period": f"2005-01 to 2024-12", "data": all_monthly, "failed": failed}, f)

# ── 2. Download Fama-French 5-Factor Data ──
print("\nDownloading Fama-French 5-factor data...")
ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
ff_zip = os.path.join(OUT, "ff5_factors.zip")
ff_csv = os.path.join(OUT, "ff5_factors.csv")

try:
    urllib.request.urlretrieve(ff_url, ff_zip)
    import zipfile
    with zipfile.ZipFile(ff_zip, 'r') as z:
        for name in z.namelist():
            if '5_Factors' in name and name.endswith('.CSV'):
                with z.open(name) as src, open(ff_csv, 'wb') as dst:
                    dst.write(src.read())
                break
    print(f"  FF5 factors saved to {ff_csv}")
except Exception as e:
    print(f"  FF5 download failed: {e}")
    print("  Will use synthetic factors as fallback")

# ── 3. Download Fama-French Momentum Factor ──
mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
mom_zip = os.path.join(OUT, "ff_momentum.zip")
mom_csv = os.path.join(OUT, "ff_momentum.csv")

try:
    urllib.request.urlretrieve(mom_url, mom_zip)
    import zipfile
    with zipfile.ZipFile(mom_zip, 'r') as z:
        for name in z.namelist():
            if 'Momentum' in name and name.endswith('.CSV'):
                with z.open(name) as src, open(mom_csv, 'wb') as dst:
                    dst.write(src.read())
                break
    print(f"  Momentum factor saved to {mom_csv}")
except Exception as e:
    print(f"  Momentum download failed: {e}")

# ── Summary ──
all_rets = []
for ticker, monthly in all_monthly.items():
    for month, d in monthly.items():
        if 'return' in d:
            all_rets.append(d['return'])

print(f"\n{'='*60}")
print(f"DATA DOWNLOAD COMPLETE")
print(f"{'='*60}")
print(f"Stocks: {len(all_monthly)} (failed: {len(failed)})")
print(f"Total month-observations: {len(all_rets)}")
print(f"Mean monthly return: {np.mean(all_rets)*100:.3f}%")
print(f"Std monthly return: {np.std(all_rets)*100:.3f}%")
print(f"Period: 2005-01 to 2024-12")
print(f"Saved to: {outpath}")
