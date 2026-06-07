#!/usr/bin/env python3
"""
Full LLM Agent Scoring: 183 stocks × 12 months × 3 agents = ~5,967 calls.
Uses Qwen/DashScope API. Checkpoint every 100 calls. Resume from interruption.

Author: Siyi / 2026-06-03
"""

import json, os, time, sys
import numpy as np
from openai import OpenAI
from datetime import datetime

OUT = "/home/z/my-project/delta_jfe"
CHECKPOINT = os.path.join(OUT, "llm_full_checkpoint.json")
RATINGS_PATH = os.path.join(OUT, "agent_ratings_llm_full.json")

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ── Load ──
with open(os.path.join(OUT, "sp500_monthly_returns.json")) as f:
    raw = json.load(f)
stock_data = raw["data"]

# 12 evenly-spaced months
all_months = sorted(set(m for t in stock_data for m in stock_data[t]))
step = len(all_months) // 12
sample_months = [all_months[i * step] for i in range(12)]

# ── Agent prompts ──
AGENT_CONFIGS = {
    "sentiment": {
        "system": "You are a sentiment analyst for stocks. Rate the stock outlook from 1 (very bearish) to 10 (very bullish). Reply with ONLY a single number, no explanation.",
        "template": "Stock: {ticker}. 1-month return: {r1m:+.1f}%. 3-month return: {r3m:+.1f}%. Volume ratio vs 3-month avg: {vrat:.1f}x. Return skewness: {skew:+.2f}. Rate the outlook for next month."
    },
    "technical": {
        "system": "You are a technical analyst for stocks. Rate the stock outlook from 1 (very bearish) to 10 (very bullish). Reply with ONLY a single number, no explanation.",
        "template": "Stock: {ticker}. 6-month trend: {r6m:+.1f}%. 6-month volatility: {vol6:.1f}%. 1-month return: {r1m:+.1f}% (mean reversion signal). Rate the outlook for next month."
    },
    "fundamental": {
        "system": "You are a fundamental analyst for stocks. Rate the stock outlook from 1 (very bearish) to 10 (very bullish). Reply with ONLY a single number, no explanation.",
        "template": "Stock: {ticker}. 12-month return: {r12m:+.1f}% (contrarian signal). 6-month volatility: {vol6:.1f}% (quality signal). Max drawdown: {maxdd:+.1f}% (stability). Rate the outlook for next month."
    }
}

def get_features(ticker, month):
    data = stock_data[ticker]
    months = sorted(data.keys())
    if month not in months:
        return None
    idx = months.index(month)
    f = {}
    f["r1m"] = data[months[idx-1]]["return"] * 100 if idx >= 1 else 0
    f["r3m"] = (np.prod([1+data[months[idx-i]]["return"] for i in range(1,4)])-1)*100 if idx >= 3 else f["r1m"]
    f["r6m"] = (np.prod([1+data[months[idx-i]]["return"] for i in range(1,7)])-1)*100 if idx >= 6 else f["r3m"]
    f["r12m"] = (np.prod([1+data[months[idx-i]]["return"] for i in range(1,13)])-1)*100 if idx >= 12 else f["r6m"]
    if idx >= 3:
        pv = [data[months[idx-i]]["volume"] for i in range(1,4)]
        f["vrat"] = data[month]["volume"] / np.mean(pv)
    else:
        f["vrat"] = 1.0
    if idx >= 6:
        pr = [data[months[idx-i]]["return"] for i in range(1,7)]
        f["vol6"] = np.std(pr) * 100
        cum = np.cumprod([1+r for r in reversed(pr)])
        peak = np.maximum.accumulate(cum)
        f["maxdd"] = float(np.min((cum - peak) / peak)) * 100
    else:
        f["vol6"] = 5.0
        f["maxdd"] = -5.0
    f["skew"] = 0.0
    return f

def call_llm(system, user_msg):
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=10, temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
        rating = int(''.join(c for c in text if c.isdigit())[:2]) or 5
        return max(1, min(10, rating))
    except Exception as e:
        return None

# ── Build task list ──
tasks = []
for ticker in sorted(stock_data.keys()):
    for month in sample_months:
        feat = get_features(ticker, month)
        if feat is None:
            continue
        for agent_name, config in AGENT_CONFIGS.items():
            user_msg = config["template"].format(ticker=ticker, **feat)
            tasks.append({
                "ticker": ticker, "month": month, "agent": agent_name,
                "system": config["system"], "user_msg": user_msg,
                "key": f"{ticker}_{month}_{agent_name}"
            })

print(f"Total tasks: {len(tasks)}")

# ── Load checkpoint ──
completed = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    completed = ckpt.get("completed", {})
    print(f"Resuming: {len(completed)} tasks already completed")

remaining = [t for t in tasks if t["key"] not in completed]
print(f"Remaining: {len(remaining)} tasks")
print(f"Estimated time: {len(remaining) / 1.2 / 60:.0f} minutes")

if not remaining:
    print("All tasks already completed!")
else:
    start_time = time.time()
    error_count = 0
    consecutive_errors = 0

    for i, task in enumerate(remaining):
        rating = call_llm(task["system"], task["user_msg"])

        if rating is not None:
            completed[task["key"]] = {
                "ticker": task["ticker"],
                "month": task["month"],
                "agent": task["agent"],
                "rating": rating
            }
            consecutive_errors = 0
        else:
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= 10:
                print(f"\n[ERROR] 10 consecutive failures, stopping.")
                break

        # Rate limit
        time.sleep(0.3)

        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            with open(CHECKPOINT, "w") as f:
                json.dump({"completed": completed, "last_idx": i}, f)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate / 60
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] {i+1}/{len(remaining)} done | {rate:.1f} calls/s | ETA: {eta:.0f}min | errors: {error_count}")
            sys.stdout.flush()

    # Final save
    with open(CHECKPOINT, "w") as f:
        json.dump({"completed": completed}, f)

# ── Build structured ratings ──
ratings = {}
for key, val in completed.items():
    t, m, a = val["ticker"], val["month"], val["agent"]
    if t not in ratings:
        ratings[t] = {}
    if m not in ratings[t]:
        ratings[t][m] = {}
    ratings[t][m][a] = val["rating"]

with open(RATINGS_PATH, "w") as f:
    json.dump(ratings, f, indent=2)

# ── Summary ──
from scipy.stats import entropy as scipy_entropy

def rating_to_probs(rating):
    x = (rating - 5.5) / 2.0
    p_neg = np.exp(-x) / (np.exp(-x) + 1 + np.exp(x))
    p_neu = 1 / (np.exp(-x) + 1 + np.exp(x))
    p_pos = np.exp(x) / (np.exp(-x) + 1 + np.exp(x))
    return np.array([p_neg, p_neu, p_pos])

all_r = []
for ticker in ratings:
    for month in ratings[ticker]:
        r = ratings[ticker][month]
        if len(r) == 3:
            all_r.append([r.get("sentiment", 5), r.get("technical", 5), r.get("fundamental", 5)])

if all_r:
    all_r = np.array(all_r)
    d = np.std(all_r, axis=1)
    print(f"\nLLM Full Rating Summary:")
    print(f"  Observations: {len(all_r)}")
    print(f"  Sentiment:   mean={all_r[:,0].mean():.2f} std={all_r[:,0].std():.2f} range=[{all_r[:,0].min()},{all_r[:,0].max()}]")
    print(f"  Technical:   mean={all_r[:,1].mean():.2f} std={all_r[:,1].std():.2f} range=[{all_r[:,1].min()},{all_r[:,1].max()}]")
    print(f"  Fundamental: mean={all_r[:,2].mean():.2f} std={all_r[:,2].std():.2f} range=[{all_r[:,2].min()},{all_r[:,2].max()}]")
    print(f"  D_post:      mean={d.mean():.2f} std={d.std():.2f} range=[{d.min():.2f},{d.max():.2f}]")
    print(f"  D_post=0:    {(d==0).sum()}/{len(d)} ({(d==0).mean()*100:.1f}%)")

elapsed = time.time() - start_time if remaining else 0
print(f"\nTime: {elapsed/60:.1f} min | Errors: {error_count}")
print(f"Saved to: {RATINGS_PATH}")
