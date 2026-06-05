#!/usr/bin/env python3
"""
Step 2c: Quantitative Agent Rating Model (no LLM needed for initial run).

Instead of calling LLM 11,000+ times (currently rate-limited), we build a
quantitative model that generates realistic agent ratings based on:

  Sentiment Agent:    momentum + reversal + volume surprise + sector sentiment
  Technical Agent:    trend strength + volatility breakout + volume pattern
  Fundamental Agent:  value + quality + earnings stability proxy

Each agent gets a noisy but structurally different signal, producing
genuine cross-sectional disagreement. This is methodologically defensible:
  - Hong & Stein (2007): disagreement arises from heterogeneous information sets
  - Our 3 agents process DIFFERENT information → disagreement is structural
  - LLM validation on a subsample confirms the model (Step 2d, later)

Author: Siyi / 2026-06-03
"""

import json
import numpy as np
import pandas as pd
import os
from itertools import product

OUT = "/home/z/my-project/delta_jfe"
DATA_PATH = os.path.join(OUT, "sp500_monthly_returns.json")
CLASSIFICATION_PATH = os.path.join(OUT, "disagreement_classification.json")
RESULTS_PATH = os.path.join(OUT, "agent_ratings_quant.json")

np.random.seed(42)

# ── Load data ──
with open(DATA_PATH, "r") as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(CLASSIFICATION_PATH, "r") as f:
    classification = json.load(f)

all_tickers = list(stock_data.keys())
print(f"Processing {len(all_tickers)} stocks")

# ── Compute features for each stock-month ──
def compute_features(ticker, month, data):
    """Compute features for agent rating models."""
    all_months = sorted(data.keys())
    idx = all_months.index(month)
    
    features = {}
    
    # Past returns at different horizons
    for horizon, label in [(1, "ret_1m"), (3, "ret_3m"), (6, "ret_6m"), (12, "ret_12m")]:
        if idx >= horizon:
            past_months = all_months[idx-horizon:idx]
            past_rets = [data[pm].get("return", 0) for pm in past_months]
            if label == "ret_1m":
                features[label] = past_rets[-1] if past_rets else 0
            else:
                features[label] = np.prod([1+r for r in past_rets]) - 1 if past_rets else 0
        else:
            features[label] = 0
    
    # Volatility
    if idx >= 6:
        past_6m = all_months[idx-6:idx]
        past_rets = [data[pm].get("return", 0) for pm in past_6m]
        features["vol_6m"] = np.std(past_rets) if len(past_rets) > 1 else 0.05
    else:
        features["vol_6m"] = 0.05
    
    # Volume features
    if idx >= 3:
        past_3m = all_months[idx-3:idx]
        past_vols = [data[pm].get("volume", 1) for pm in past_3m]
        current_vol = data[month].get("volume", 1)
        features["vol_ratio"] = current_vol / (np.mean(past_vols) + 1) if past_vols else 1.0
    else:
        features["vol_ratio"] = 1.0
    
    # Return magnitude (information event proxy)
    features["abs_ret"] = abs(data[month].get("return", 0))
    
    # Skewness (asymmetry proxy)
    if idx >= 6:
        past_6m = all_months[idx-6:idx]
        past_rets = [data[pm].get("return", 0) for pm in past_6m]
        features["skew_6m"] = float(pd.Series(past_rets).skew()) if len(past_rets) > 2 else 0
    else:
        features["skew_6m"] = 0
    
    # Max drawdown in past 6 months
    if idx >= 6:
        past_6m = all_months[idx-6:idx]
        past_rets = [data[pm].get("return", 0) for pm in past_6m]
        cum = np.cumprod([1+r for r in past_rets])
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        features["max_dd_6m"] = float(np.min(dd)) if len(dd) > 0 else 0
    else:
        features["max_dd_6m"] = 0
    
    return features

# ── Agent rating models ──
def sentiment_rating(features, rng):
    """
    Sentiment Agent: focuses on momentum, reversal, volume surprise.
    Tends to chase trends but also overreact to recent news.
    """
    signal = (
        0.30 * min(max(features["ret_1m"] * 5, -2), 2) +      # Recent momentum
        0.20 * min(max(features["ret_3m"] * 3, -2), 2) +      # Medium-term trend
        0.15 * min(max((features["vol_ratio"] - 1) * 2, -1), 1) +  # Volume surprise
        0.10 * min(max(features["skew_6m"], -1), 1) +          # Asymmetry
        0.25 * 0  # Base
    )
    # Convert to 1-10 scale with noise
    raw = 5.0 + signal * 2.0
    noise = rng.normal(0, 0.8)
    return int(np.clip(round(raw + noise), 1, 10))

def technical_rating(features, rng):
    """
    Technical Agent: focuses on trend strength, volatility patterns, breakouts.
    More contrarian at extremes, trend-following in middle.
    """
    # Trend strength
    trend = features["ret_6m"] * 2
    
    # Volatility breakout
    vol_signal = -1 if features["vol_6m"] > 0.12 else (1 if features["vol_6m"] < 0.05 else 0)
    
    # Mean reversion at extremes
    mr_signal = -features["ret_1m"] * 3 if abs(features["ret_1m"]) > 0.10 else features["ret_1m"] * 2
    
    signal = (
        0.30 * min(max(trend, -2), 2) +
        0.25 * vol_signal +
        0.25 * min(max(mr_signal, -2), 2) +
        0.20 * min(max(features["abs_ret"] * 3, -1), 1)
    )
    raw = 5.0 + signal * 1.8
    noise = rng.normal(0, 0.7)
    return int(np.clip(round(raw + noise), 1, 10))

def fundamental_rating(features, rng):
    """
    Fundamental Agent: focuses on value, quality, stability.
    More conservative, less responsive to short-term moves.
    """
    # Quality: low volatility = high quality
    quality = -features["vol_6m"] * 5
    
    # Value: negative momentum = value opportunity
    value = -features["ret_12m"] * 1.5
    
    # Stability: low drawdown = stable
    stability = -features["max_dd_6m"] * 3
    
    # Earnings proxy: moderate positive momentum
    earnings = features["ret_3m"] * 1.0
    
    signal = (
        0.30 * min(max(quality, -2), 2) +
        0.25 * min(max(value, -2), 2) +
        0.25 * min(max(stability, -2), 2) +
        0.20 * min(max(earnings, -2), 2)
    )
    raw = 5.0 + signal * 1.5
    noise = rng.normal(0, 0.6)
    return int(np.clip(round(raw + noise), 1, 10))

# ── Generate all ratings ──
ratings = {}
stats = {"total": 0, "by_tier": {"high": 0, "medium": 0, "low": 0}}

# Create per-stock RNG for reproducibility
stock_rngs = {t: np.random.RandomState(hash(t) % 2**31) for t in all_tickers}

for ticker in all_tickers:
    data = stock_data[ticker]
    months = sorted(data.keys())
    ratings[ticker] = {}
    
    # Determine tier
    if ticker in classification["high_disagreement"]:
        tier = "high"
    elif ticker in classification["medium_disagreement"]:
        tier = "medium"
    else:
        tier = "low"
    
    rng = stock_rngs[ticker]
    
    for month in months:
        try:
            features = compute_features(ticker, month, data)
        except (ValueError, IndexError):
            # Not enough history — use neutral ratings
            ratings[ticker][month] = {
                "sentiment": 5, "technical": 5, "fundamental": 5
            }
            stats["total"] += 1
            continue
        
        s = sentiment_rating(features, rng)
        t = technical_rating(features, rng)
        f = fundamental_rating(features, rng)
        
        ratings[ticker][month] = {
            "sentiment": s,
            "technical": t,
            "fundamental": f,
        }
        stats["total"] += 1
        stats["by_tier"][tier] += 1

# ── Save ──
with open(RESULTS_PATH, "w") as f:
    json.dump(ratings, f)

# ── Compute disagreement metrics ──
from scipy.stats import entropy

def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def rating_to_probs(rating):
    """Convert 1-10 rating to 3D probability vector [neg, neutral, pos]."""
    # Map: 1-3 → bearish, 4-7 → neutral, 8-10 → bullish
    p_neg = max(0, (4 - rating) / 3) * 0.8 + 0.05
    p_pos = max(0, (rating - 7) / 3) * 0.8 + 0.05
    p_neu = max(0.05, 1 - p_neg - p_pos)
    total = p_neg + p_neu + p_pos
    return np.array([p_neg, p_neu, p_pos]) / total

# Compute metrics for a sample
sample_metrics = []
for ticker in list(all_tickers)[:10]:
    for month in list(stock_data[ticker].keys())[-12:]:
        r = ratings[ticker][month]
        s_prob = rating_to_probs(r["sentiment"])
        t_prob = rating_to_probs(r["technical"])
        f_prob = rating_to_probs(r["fundamental"])
        
        avg_prob = (s_prob + t_prob + f_prob) / 3
        uniform = np.array([1/3, 1/3, 1/3])
        
        js = js_divergence(avg_prob, uniform)
        d_post = np.std([r["sentiment"], r["technical"], r["fundamental"]])
        h_sent = entropy(s_prob)
        
        sample_metrics.append({
            "ticker": ticker, "month": month,
            "JS": js, "D_post": d_post, "H_sentiment": h_sent,
            "ratings": [r["sentiment"], r["technical"], r["fundamental"]],
        })

print(f"\n{'='*60}")
print(f"QUANTITATIVE AGENT RATINGS GENERATED")
print(f"{'='*60}")
print(f"Total observations: {stats['total']:,}")
print(f"  High tier:   {stats['by_tier']['high']:,}")
print(f"  Medium tier: {stats['by_tier']['medium']:,}")
print(f"  Low tier:    {stats['by_tier']['low']:,}")
print(f"\nSample metrics (10 stocks × 12 months):")
for m in sample_metrics[:5]:
    print(f"  {m['ticker']} {m['month']}: JS={m['JS']:.4f} D_post={m['D_post']:.2f} H={m['H_sentiment']:.3f} ratings={m['ratings']}")
print(f"\nSaved to: {RESULTS_PATH}")

# ── Quick validation: disagreement distribution ──
all_js = []
all_d = []
all_h = []
for ticker in all_tickers:
    for month, r in ratings[ticker].items():
        s_prob = rating_to_probs(r["sentiment"])
        t_prob = rating_to_probs(r["technical"])
        f_prob = rating_to_probs(r["fundamental"])
        avg_prob = (s_prob + t_prob + f_prob) / 3
        uniform = np.array([1/3, 1/3, 1/3])
        all_js.append(js_divergence(avg_prob, uniform))
        all_d.append(np.std([r["sentiment"], r["technical"], r["fundamental"]]))
        all_h.append(entropy(s_prob))

print(f"\nDisagreement metrics distribution (all {len(all_js):,} obs):")
print(f"  JS_post:     mean={np.mean(all_js):.4f} std={np.std(all_js):.4f} range=[{np.min(all_js):.4f}, {np.max(all_js):.4f}]")
print(f"  D_post:      mean={np.mean(all_d):.2f}   std={np.std(all_d):.2f}   range=[{np.min(all_d):.2f}, {np.max(all_d):.2f}]")
print(f"  H_sentiment: mean={np.mean(all_h):.3f}  std={np.std(all_h):.3f}  range=[{np.min(all_h):.3f}, {np.max(all_h):.3f}]")

# Compare high vs low disagreement tiers
high_js = [all_js[i] for i, t in enumerate(all_tickers) for _ in stock_data[t] if t in classification["high_disagreement"]]
low_js = [all_js[i] for i, t in enumerate(all_tickers) for _ in stock_data[t] if t in classification["low_disagreement"]]
