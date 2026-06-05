#!/usr/bin/env python3
"""
Step 2: Pre-screen stocks by disagreement potential, then LLM-score strategically.

Phase A: Compute disagreement proxy from market data (volatility, turnover, analyst coverage proxy)
Phase B: Classify stocks into High/Medium/Low disagreement tiers
Phase C: LLM scoring — full 3-agent for High, simplified for Low

Author: Siyi / 2026-06-03
"""

import json
import numpy as np
import pandas as pd
import os
import time
import subprocess

OUT = "/home/z/my-project/delta_jfe"
DATA_PATH = os.path.join(OUT, "sp500_monthly_returns.json")

# ── Load data ──
with open(DATA_PATH, "r") as f:
    raw = json.load(f)

all_data = raw["data"]  # {ticker: {month: {close, volume, return}}}
print(f"Loaded {len(all_data)} stocks")

# ── Phase A: Compute disagreement proxy ──
# For each stock-month, compute:
#   1. Volatility = rolling 12-month std of returns
#   2. Turnover = monthly volume / avg volume (relative)
#   3. Return magnitude = |return| (proxy for information events)
#   4. Cross-sectional rank of above

disagreement_proxy = {}

for ticker, monthly in all_data.items():
    # Sort months
    months = sorted(monthly.keys())
    if len(months) < 13:
        continue
    
    returns = []
    for m in months:
        r = monthly[m].get("return")
        if r is not None:
            returns.append(r)
        else:
            returns.append(np.nan)
    
    # Rolling 12-month volatility
    for i in range(12, len(months)):
        month = months[i]
        window = returns[i-12:i]
        valid = [r for r in window if not np.isnan(r)]
        if len(valid) < 6:
            continue
        
        vol = np.std(valid)
        ret_abs = abs(returns[i]) if not np.isnan(returns[i]) else np.nan
        vol_rank = 0  # will be computed cross-sectionally
        
        if ticker not in disagreement_proxy:
            disagreement_proxy[ticker] = {}
        disagreement_proxy[ticker][month] = {
            "vol_12m": vol,
            "ret_abs": ret_abs,
            "volume": monthly[month].get("volume", 0),
        }

# ── Cross-sectional ranking per month ──
all_months = sorted(set(m for t in disagreement_proxy.values() for m in t.keys()))
print(f"Months with proxy data: {len(all_months)}")

for month in all_months:
    # Collect all stocks' proxies for this month
    vols = {}
    rets = {}
    for ticker, months_data in disagreement_proxy.items():
        if month in months_data:
            vols[ticker] = months_data[month]["vol_12m"]
            rets[ticker] = months_data[month].get("ret_abs", 0) or 0
    
    if len(vols) < 10:
        continue
    
    # Rank (0-1)
    vol_series = pd.Series(vols)
    ret_series = pd.Series(rets)
    vol_rank = vol_series.rank(pct=True)
    ret_rank = ret_series.rank(pct=True)
    
    # Composite disagreement score = average of vol rank and |return| rank
    for ticker in vols:
        score = (vol_rank[ticker] + ret_rank[ticker]) / 2
        disagreement_proxy[ticker][month]["disagreement_score"] = score

# ── Phase B: Classify stocks into tiers ──
# For each stock, compute average disagreement score across all months
stock_avg_score = {}
for ticker, months_data in disagreement_proxy.items():
    scores = [m["disagreement_score"] for m in months_data.values() if "disagreement_score" in m]
    if scores:
        stock_avg_score[ticker] = np.mean(scores)

# Classify
score_series = pd.Series(stock_avg_score)
q75 = score_series.quantile(0.75)
q50 = score_series.quantile(0.50)
q25 = score_series.quantile(0.25)

high_disagreement = sorted(score_series[score_series >= q75].index.tolist())
medium_disagreement = sorted(score_series[(score_series >= q25) & (score_series < q75)].index.tolist())
low_disagreement = sorted(score_series[score_series < q25].index.tolist())

print(f"\n{'='*60}")
print(f"DISAGREEMENT CLASSIFICATION")
print(f"{'='*60}")
print(f"High disagreement (top 25%): {len(high_disagreement)} stocks")
print(f"  Score range: {q75:.3f} - {score_series.max():.3f}")
print(f"  Examples: {high_disagreement[:10]}")
print(f"\nMedium disagreement (25-75%): {len(medium_disagreement)} stocks")
print(f"  Score range: {q25:.3f} - {q75:.3f}")
print(f"  Examples: {medium_disagreement[:5]}")
print(f"\nLow disagreement (bottom 25%): {len(low_disagreement)} stocks")
print(f"  Score range: {score_series.min():.3f} - {q25:.3f}")
print(f"  Examples: {low_disagreement[:10]}")

# ── GICS sector check ──
SECTOR_MAP = {
    "AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","AMZN":"Consumer","NVDA":"Tech","META":"Comm",
    "TSLA":"Consumer","AVGO":"Tech","ADBE":"Tech","CRM":"Tech","ORCL":"Tech","INTC":"Tech",
    "AMD":"Tech","QCOM":"Tech","TXN":"Tech","IBM":"Tech","NFLX":"Comm","DIS":"Comm",
    "CMCSA":"Comm","T":"Comm","VZ":"Comm","JNJ":"Health","LLY":"Health","UNH":"Health",
    "PFE":"Health","MRK":"Health","ABBV":"Health","JPM":"Finance","BAC":"Finance","WFC":"Finance",
    "GS":"Finance","MS":"Finance","C":"Finance","BLK":"Finance","SCHW":"Finance","AXP":"Finance",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy","OXY":"Energy",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","DUK":"Utilities","AEP":"Utilities",
    "PG":"Consumer","KO":"Consumer","PEP":"Consumer","WMT":"Consumer","COST":"Consumer",
    "HD":"Consumer","LOW":"Consumer","MCD":"Consumer","SBUX":"Consumer","NKE":"Consumer",
    "CAT":"Industrial","BA":"Industrial","GE":"Industrial","HON":"Industrial","MMM":"Industrial",
    "UPS":"Industrial","FDX":"Industrial","LMT":"Industrial","RTX":"Industrial","DE":"Industrial",
    "AMGN":"Health","GILD":"Health","MRNA":"Health","BIIB":"Health","REGN":"Health",
    "VRTX":"Health","ISRG":"Health","ABNB":"Consumer","COIN":"Finance","PLTR":"Tech",
    "SNOW":"Tech","CRWD":"Tech","PANW":"Tech","MSTR":"Tech","RIVN":"Consumer",
}

print(f"\n{'='*60}")
print(f"SECTOR DISTRIBUTION IN HIGH-DISAGREEMENT TIER")
print(f"{'='*60}")
sector_counts = {}
for t in high_disagreement:
    s = SECTOR_MAP.get(t, "Other")
    sector_counts[s] = sector_counts.get(s, 0) + 1
for s, c in sorted(sector_counts.items(), key=lambda x: -x[1]):
    print(f"  {s:12s}: {c}")

# ── Save classification ──
classification = {
    "high_disagreement": high_disagreement,
    "medium_disagreement": medium_disagreement,
    "low_disagreement": low_disagreement,
    "thresholds": {"q25": q25, "q50": q50, "q75": q75},
    "stock_scores": {k: round(v, 4) for k, v in stock_avg_score.items()},
}

class_path = os.path.join(OUT, "disagreement_classification.json")
with open(class_path, "w") as f:
    json.dump(classification, f, indent=2)
print(f"\nClassification saved to {class_path}")

# ── Estimate LLM call volume ──
# High: 3 agents × all months × all stocks
# Medium: 3 agents × sample months (every 3rd month) × all stocks
# Low: 1 agent × sample months (every 6th month) for confirmation

n_months_high = len(all_months)
n_months_med = len(all_months) // 3
n_months_low = len(all_months) // 6

calls_high = len(high_disagreement) * n_months_high * 3
calls_med = len(medium_disagreement) * n_months_med * 3
calls_low = len(low_disagreement) * n_months_low * 1
total_calls = calls_high + calls_med + calls_low

print(f"\n{'='*60}")
print(f"LLM CALL ESTIMATE (STRATIFIED)")
print(f"{'='*60}")
print(f"High:   {len(high_disagreement):3d} stocks × {n_months_high} months × 3 agents = {calls_high:,} calls")
print(f"Medium: {len(medium_disagreement):3d} stocks × {n_months_med} months × 3 agents = {calls_med:,} calls")
print(f"Low:    {len(low_disagreement):3d} stocks × {n_months_low} months × 1 agent  = {calls_low:,} calls")
print(f"TOTAL: {total_calls:,} calls")
print(f"(vs 132,000 for full-sample brute force — {total_calls/132000*100:.0f}%)")
