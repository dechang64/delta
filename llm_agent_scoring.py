#!/usr/bin/env python3
"""
Real LLM Agent Scoring using Qwen (DashScope API).

3 agents × 183 stocks × sample months (stratified by disagreement tier):
  High:   46 stocks × 12 months × 3 agents = 1,656 calls
  Medium: 91 stocks × 6 months × 3 agents = 1,638 calls
  Low:    46 stocks × 3 months × 3 agents =   414 calls
  TOTAL: ~3,708 calls (manageable, ~1 hour at 1 call/sec)

Checkpoint every 50 calls. Resume from interruption.

Author: Siyi / 2026-06-03
"""

import json
import numpy as np
import os
import time
from openai import OpenAI
from datetime import datetime

OUT = "/home/z/my-project/delta_jfe"
CHECKPOINT = os.path.join(OUT, "llm_scoring_checkpoint.json")
RATINGS_PATH = os.path.join(OUT, "agent_ratings_llm.json")

# ── Init client ──
import sys
client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ── Load data ──
with open(os.path.join(OUT, "sp500_monthly_returns.json")) as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(os.path.join(OUT, "disagreement_classification.json")) as f:
    classification = json.load(f)

# ── Agent prompts ──
AGENT_CONFIGS = {
    "sentiment": {
        "system": "You are a sentiment-driven stock analyst. You focus on short-term momentum, market sentiment, volume patterns, and investor psychology. Rate the stock's next-month outlook from 1 (very bearish) to 10 (very bullish). Reply with ONLY a single integer number, nothing else.",
        "feature_template": (
            "Stock: {ticker}. "
            "1-month return: {r1m:+.1f}%. "
            "3-month return: {r3m:+.1f}%. "
            "Volume vs 3-month avg: {vrat:.1f}x. "
            "Recent return skewness: {skew:+.2f}. "
            "Rate the next-month outlook."
        ),
    },
    "technical": {
        "system": "You are a technical analyst for stocks. You focus on price trends, volatility patterns, support/resistance levels, and mean-reversion signals. Rate the stock's next-month outlook from 1 (very bearish) to 10 (very bullish). Reply with ONLY a single integer number, nothing else.",
        "feature_template": (
            "Stock: {ticker}. "
            "6-month trend: {r6m:+.1f}%. "
            "6-month volatility: {vol6m:.1f}%. "
            "1-month return: {r1m:+.1f}% (mean-reversion signal). "
            "Rate the next-month outlook."
        ),
    },
    "fundamental": {
        "system": "You are a fundamental analyst for stocks. You focus on valuation (contrarian signals), quality (low volatility), earnings stability, and long-term value. Rate the stock's next-month outlook from 1 (very bearish) to 10 (very bullish). Reply with ONLY a single integer number, nothing else.",
        "feature_template": (
            "Stock: {ticker}. "
            "12-month return: {r12m:+.1f}% (contrarian: negative = cheap). "
            "6-month volatility: {vol6m:.1f}% (low = quality). "
            "Max 6-month drawdown: {maxdd:+.1f}% (small = stable). "
            "3-month earnings momentum: {r3m:+.1f}%. "
            "Rate the next-month outlook."
        ),
    },
}

# ── Feature computation (no look-ahead!) ──
def compute_features(ticker, month):
    """Compute features using ONLY data available at time t (past data)."""
    data = stock_data[ticker]
    all_months = sorted(data.keys())
    if month not in all_months:
        return None
    idx = all_months.index(month)
    
    f = {"ticker": ticker}
    
    # 1-month return (PAST, not current)
    if idx >= 1:
        f["r1m"] = data[all_months[idx-1]].get("return", 0) * 100
    else:
        f["r1m"] = 0.0
    
    # 3-month return
    if idx >= 3:
        pr = [data[all_months[idx-i]].get("return", 0) for i in range(1, 4)]
        f["r3m"] = (np.prod([1+r for r in pr]) - 1) * 100
    else:
        f["r3m"] = 0.0
    
    # 6-month return
    if idx >= 6:
        pr = [data[all_months[idx-i]].get("return", 0) for i in range(1, 7)]
        f["r6m"] = (np.prod([1+r for r in pr]) - 1) * 100
        f["vol6m"] = np.std(pr) * 100
    else:
        f["r6m"] = 0.0
        f["vol6m"] = 5.0
    
    # 12-month return
    if idx >= 12:
        pr = [data[all_months[idx-i]].get("return", 0) for i in range(1, 13)]
        f["r12m"] = (np.prod([1+r for r in pr]) - 1) * 100
    else:
        f["r12m"] = 0.0
    
    # Volume ratio
    if idx >= 4:
        pv = [data[all_months[idx-i]].get("volume", 1) for i in range(1, 4)]
        cv = data[all_months[idx-1]].get("volume", 1)  # Use last month's volume
        f["vrat"] = cv / (np.mean(pv) + 1)
    else:
        f["vrat"] = 1.0
    
    # Skewness
    if idx >= 7:
        pr = [data[all_months[idx-i]].get("return", 0) for i in range(1, 7)]
        from scipy.stats import skew
        f["skew"] = float(skew(pr))
    else:
        f["skew"] = 0.0
    
    # Max drawdown
    if idx >= 7:
        pr = [data[all_months[idx-i]].get("return", 0) for i in range(1, 7)]
        cum = np.cumprod([1+r for r in reversed(pr)])
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        f["maxdd"] = float(np.min(dd)) * 100
    else:
        f["maxdd"] = -5.0
    
    return f

# ── Build task queue (stratified sampling) ──
def get_sample_months(tier):
    """Select months to score based on disagreement tier."""
    all_months = sorted(set(m for t in stock_data for m in stock_data[t]))
    n = len(all_months)
    if tier == "high":
        # Every 20th month = ~12 months
        return [all_months[i] for i in range(0, n, 20)]
    elif tier == "medium":
        # Every 40th month = ~6 months
        return [all_months[i] for i in range(0, n, 40)]
    else:  # low
        # Every 80th month = ~3 months
        return [all_months[i] for i in range(0, n, 80)]

tasks = []
for tier_name, stocks in [("high", classification["high_disagreement"]),
                           ("medium", classification["medium_disagreement"]),
                           ("low", classification["low_disagreement"])]:
    months = get_sample_months(tier_name)
    for ticker in stocks:
        if ticker not in stock_data:
            continue
        for month in months:
            if month not in stock_data[ticker]:
                continue
            for agent in ["sentiment", "technical", "fundamental"]:
                tasks.append({
                    "ticker": ticker,
                    "month": month,
                    "agent": agent,
                    "tier": tier_name,
                })

print(f"Total tasks: {len(tasks)}")
for tier in ["high", "medium", "low"]:
    n = sum(1 for t in tasks if t["tier"] == tier)
    print(f"  {tier}: {n} calls")

# ── Load checkpoint ──
completed = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    completed = ckpt.get("completed", {})
    print(f"Resuming: {len(completed)} tasks already completed")

remaining = [t for t in tasks if f"{t['ticker']}_{t['month']}_{t['agent']}" not in completed]
print(f"Remaining: {len(remaining)} tasks")

# ── Scoring loop ──
def call_llm(agent, features):
    """Call LLM and parse rating."""
    config = AGENT_CONFIGS[agent]
    prompt = config["feature_template"].format(**features)
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        # Parse integer from response
        rating = int(''.join(c for c in text if c.isdigit())[:2])
        return max(1, min(10, rating))
    except Exception as e:
        return None

def save_checkpoint():
    with open(CHECKPOINT, "w") as f:
        json.dump({"completed": completed, "last_update": datetime.now().isoformat()}, f)

error_count = 0
start_time = time.time()
batch_start = time.time()

print(f"\nStarting LLM scoring: {len(remaining)} tasks")
print(f"Estimated time: {len(remaining) * 1.2 / 60:.0f} minutes")

for i, task in enumerate(remaining):
    key = f"{task['ticker']}_{task['month']}_{task['agent']}"
    
    # Compute features
    features = compute_features(task["ticker"], task["month"])
    if features is None:
        completed[key] = {"rating": None, "error": "no_features"}
        error_count += 1
        continue
    
    # Call LLM
    rating = call_llm(task["agent"], features)
    
    if rating is not None:
        completed[key] = {"rating": rating, "agent": task["agent"]}
    else:
        completed[key] = {"rating": None, "error": "api_error"}
        error_count += 1
    
    # Rate limiting: ~1 call/sec
    time.sleep(0.8)
    
    # Progress + checkpoint every 50 calls
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(remaining) - i - 1) / rate / 60
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1}/{len(remaining)} done | "
              f"{rate:.1f} calls/s | ETA: {eta:.0f}min | errors: {error_count}")
        save_checkpoint()

# Final save
save_checkpoint()

# ── Build structured output ──
ratings = {}
for key, val in completed.items():
    if val.get("rating") is None:
        continue
    parts = key.rsplit("_", 2)
    # Handle tickers with hyphens like BRK-B
    # Format: TICKER_YYYY-MM_AGENT
    agent = parts[-1]
    month = parts[-2]
    ticker = "_".join(parts[:-2])
    
    if ticker not in ratings:
        ratings[ticker] = {}
    if month not in ratings[ticker]:
        ratings[ticker][month] = {}
    ratings[ticker][month][agent] = val["rating"]

with open(RATINGS_PATH, "w") as f:
    json.dump(ratings, f, indent=2)

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"LLM SCORING COMPLETE")
print(f"{'='*60}")
print(f"Total tasks: {len(tasks)}")
print(f"Completed: {len(completed)}")
print(f"Errors: {error_count}")
print(f"Time: {elapsed/60:.1f} minutes")
print(f"Ratings saved to: {RATINGS_PATH}")

# Quick validation
from scipy.stats import entropy
all_ratings = []
for ticker in ratings:
    for month in ratings[ticker]:
        r = ratings[ticker][month]
        if len(r) == 3:
            all_ratings.append([r.get("sentiment", 5), r.get("technical", 5), r.get("fundamental", 5)])

if all_ratings:
    all_ratings = np.array(all_ratings)
    d_post = np.std(all_ratings, axis=1)
    print(f"\nLLM Rating Distribution:")
    print(f"  Sentiment:   mean={all_ratings[:,0].mean():.2f} std={all_ratings[:,0].std():.2f}")
    print(f"  Technical:   mean={all_ratings[:,1].mean():.2f} std={all_ratings[:,1].std():.2f}")
    print(f"  Fundamental: mean={all_ratings[:,2].mean():.2f} std={all_ratings[:,2].std():.2f}")
    print(f"  D_post:      mean={d_post.mean():.2f} std={d_post.std():.2f}")
    print(f"  D_post=0:    {(d_post==0).sum()}/{len(d_post)} ({(d_post==0).mean()*100:.1f}%)")
