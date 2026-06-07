#!/usr/bin/env python3
"""
Real LLM Agent Scoring — Qwen/DashScope API.

Stratified sampling:
  High disagreement:   46 stocks × 12 months × 3 agents = 1,656
  Medium disagreement: 91 stocks × 6 months  × 3 agents = 1,638
  Low disagreement:    46 stocks × 3 months  × 3 agents =   414
  TOTAL: ~3,708 calls

Rate: ~1 call/sec with 0.3s buffer → ~1 hour
Checkpoint: every 100 calls

Author: Siyi / 2026-06-03
"""

import json, os, time, sys
import numpy as np
from openai import OpenAI
from datetime import datetime

OUT = "/home/z/my-project/delta_jfe"
CHECKPOINT = os.path.join(OUT, "llm_checkpoint_v2.json")
RATINGS_PATH = os.path.join(OUT, "agent_ratings_llm.json")

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ── Load ──
with open(os.path.join(OUT, "sp500_monthly_returns.json")) as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(os.path.join(OUT, "disagreement_classification.json")) as f:
    classification = json.load(f)

# ── Agent prompts ──
AGENTS = {
    "sentiment": {
        "system": "You are a sentiment-driven stock analyst. You focus on short-term momentum, volume patterns, and market mood. Rate the stock's next-month outlook from 1 (very bearish) to 10 (very bullish). Consider that strong recent returns and high volume suggest positive sentiment. Reply with ONLY a single integer from 1 to 10.",
        "template": "Stock: {ticker} (S&P 500). Recent performance: 1-month return={r1m:+.1f}%, 3-month return={r3m:+.1f}%. Volume ratio vs 3-month average={vrat:.2f}x. Return skewness={skew:+.2f}. Based on sentiment signals, rate the next-month outlook 1-10:"
    },
    "technical": {
        "system": "You are a technical analyst. You focus on price trends, volatility patterns, and mean-reversion signals. Rate the stock's next-month outlook from 1 (very bearish) to 10 (very bullish). Strong trends are bullish; high volatility is bearish; extreme short-term moves may reverse. Reply with ONLY a single integer from 1 to 10.",
        "template": "Stock: {ticker} (S&P 500). Technical indicators: 6-month trend={r6m:+.1f}%, 6-month volatility={vol6:.1f}%, 1-month return={r1m:+.1f}% (extreme moves may reverse). Based on technical signals, rate the next-month outlook 1-10:"
    },
    "fundamental": {
        "system": "You are a fundamental analyst. You focus on valuation, quality, and stability. Rate the stock's next-month outlook from 1 (very bearish) to 10 (very bullish). High past returns suggest overvaluation (bearish); low volatility suggests quality (bullish); small drawdowns suggest stability (bullish). Reply with ONLY a single integer from 1 to 10.",
        "template": "Stock: {ticker} (S&P 500). Fundamental signals: 12-month return={r12m:+.1f}% (high=overvalued, bearish), 6-month volatility={vol6:.1f}% (low=quality, bullish), max drawdown={maxdd:+.1f}% (small=stable, bullish). Based on fundamentals, rate the next-month outlook 1-10:"
    }
}

def get_features(ticker, month):
    """Compute features using ONLY past data (no look-ahead)."""
    data = stock_data[ticker]
    months = sorted(data.keys())
    idx = months.index(month)
    f = {"ticker": ticker}
    f["r1m"] = data[months[idx-1]]["return"] * 100 if idx >= 1 else 0
    f["r3m"] = (np.prod([1+data[months[idx-i]]["return"] for i in range(1,4)])-1)*100 if idx >= 3 else 0
    f["r6m"] = (np.prod([1+data[months[idx-i]]["return"] for i in range(1,7)])-1)*100 if idx >= 6 else 0
    f["r12m"] = (np.prod([1+data[months[idx-i]]["return"] for i in range(1,13)])-1)*100 if idx >= 12 else 0
    if idx >= 3:
        pv = [data[months[idx-i]]["volume"] for i in range(1,4)]
        f["vrat"] = data[month]["volume"] / max(np.mean(pv), 1)
    else:
        f["vrat"] = 1.0
    if idx >= 6:
        pr = [data[months[idx-i]]["return"] for i in range(1,7)]
        f["vol6"] = np.std(pr) * 100
        cum = np.cumprod([1+r for r in reversed(pr)])
        peak = np.maximum.accumulate(cum)
        f["maxdd"] = float(np.min((cum - peak) / peak)) * 100
    else:
        f["vol6"] = 5.0; f["maxdd"] = -5.0
    f["skew"] = 0.0
    return f

def call_llm(agent_name, features):
    """Call Qwen API and parse rating."""
    prompts = AGENTS[agent_name]
    user_msg = prompts["template"].format(**features)
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=10, temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
        # Extract first number
        digits = ''.join(c for c in text if c.isdigit())
        rating = int(digits[:2]) if digits else 5
        rating = max(1, min(10, rating))
        return rating
    except Exception as e:
        return None

# ── Build task list ──
all_months = sorted(stock_data[list(stock_data.keys())[0]].keys())

def sample_months(tier):
    if tier == "high":
        return all_months[::20]  # every 20th month ≈ 12
    elif tier == "medium":
        return all_months[::40]  # every 40th ≈ 6
    else:
        return all_months[::80]  # every 80th ≈ 3

tasks = []
for tier_name, stocks in [("high", classification["high_disagreement"]),
                           ("medium", classification["medium_disagreement"]),
                           ("low", classification["low_disagreement"])]:
    months = sample_months(tier_name)
    for ticker in stocks:
        if ticker not in stock_data:
            continue
        available = sorted(stock_data[ticker].keys())
        for month in months:
            if month in available and available.index(month) >= 12:  # need 12 months history
                for agent in AGENTS:
                    tasks.append((ticker, month, agent))

print(f"Total tasks: {len(tasks)}")

# ── Load checkpoint ──
completed = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    completed = ckpt.get("completed", {})
    print(f"Resuming: {len(completed)} tasks already completed")

remaining = [(t, m, a) for t, m, a in tasks if f"{t}_{m}_{a}" not in completed]
print(f"Remaining: {len(remaining)} tasks")
print(f"Estimated time: {len(remaining) * 1.3 / 60:.0f} minutes")

if not remaining:
    print("All tasks already completed!")
else:
    start_time = time.time()
    error_count = 0
    
    for i, (ticker, month, agent) in enumerate(remaining):
        key = f"{ticker}_{month}_{agent}"
        
        feat = get_features(ticker, month)
        rating = call_llm(agent, feat)
        
        if rating is not None:
            completed[key] = {"rating": rating, "ticker": ticker, "month": month, "agent": agent}
        else:
            completed[key] = {"rating": None, "error": True}
            error_count += 1
        
        # Rate limit: ~1 call/sec
        time.sleep(0.3)
        
        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate / 60
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {i+1}/{len(remaining)} done | {rate:.1f} calls/s | ETA: {eta:.0f}min | errors: {error_count}")
            sys.stdout.flush()
            
            # Checkpoint
            with open(CHECKPOINT, "w") as f:
                json.dump({"completed": completed, "last_update": ts}, f)
    
    # Final save
    with open(CHECKPOINT, "w") as f:
        json.dump({"completed": completed}, f)

# ── Build structured output ──
ratings = {}
for key, val in completed.items():
    if val.get("rating") is None:
        continue
    ticker = val["ticker"]
    month = val["month"]
    agent = val["agent"]
    if ticker not in ratings:
        ratings[ticker] = {}
    if month not in ratings[ticker]:
        ratings[ticker][month] = {}
    ratings[ticker][month][agent] = val["rating"]

with open(RATINGS_PATH, "w") as f:
    json.dump(ratings, f, indent=2)

# ── Validation ──
all_r = []
for ticker in ratings:
    for month in ratings[ticker]:
        r = ratings[ticker][month]
        if len(r) == 3:
            all_r.append([r["sentiment"], r["technical"], r["fundamental"]])

if all_r:
    all_r = np.array(all_r)
    d = np.std(all_r, axis=1)
    print(f"\nLLM Agent Ratings Summary:")
    print(f"  Observations: {len(all_r)}")
    print(f"  Sentiment:   mean={all_r[:,0].mean():.2f} std={all_r[:,0].std():.2f} range=[{all_r[:,0].min()},{all_r[:,0].max()}]")
    print(f"  Technical:   mean={all_r[:,1].mean():.2f} std={all_r[:,1].std():.2f} range=[{all_r[:,1].min()},{all_r[:,1].max()}]")
    print(f"  Fundamental: mean={all_r[:,2].mean():.2f} std={all_r[:,2].std():.2f} range=[{all_r[:,2].min()},{all_r[:,2].max()}]")
    print(f"  D_post:      mean={d.mean():.2f} std={d.std():.2f} range=[{d.min():.2f},{d.max():.2f}]")
    print(f"  D_post=0:    {(d==0).sum()}/{len(d)} ({(d==0).mean()*100:.1f}%)")

elapsed = time.time() - start_time if remaining else 0
print(f"\nTotal time: {elapsed/60:.1f} minutes | Errors: {error_count}")
print(f"Saved to: {RATINGS_PATH}")
