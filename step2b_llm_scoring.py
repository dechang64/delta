#!/usr/bin/env python3
"""
Step 2b: LLM Batch Scoring with checkpoint/resume.

Strategy:
  High disagreement (46 stocks):   All 228 months × 3 agents = 31,464 calls
  Medium disagreement (91 stocks): Quarterly (19 periods) × 3 agents = 5,187 calls
  Low disagreement (46 stocks):    Semi-annual (10 periods) × 1 agent = 460 calls
  TOTAL: ~37,111 calls

Checkpoint: saves after every 50 calls, can resume from interruption.
Rate limit: 2 calls/sec with 0.5s buffer.

Author: Siyi / 2026-06-03
"""

import json
import os
import time
import subprocess
import sys
import signal
import numpy as np
from datetime import datetime

OUT = "/home/z/my-project/delta_jfe"
CHECKPOINT_PATH = os.path.join(OUT, "llm_scoring_checkpoint.json")
RESULTS_PATH = os.path.join(OUT, "llm_agent_ratings.json")
CLASSIFICATION_PATH = os.path.join(OUT, "disagreement_classification.json")
DATA_PATH = os.path.join(OUT, "sp500_monthly_returns.json")

# ── Load data ──
with open(DATA_PATH, "r") as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(CLASSIFICATION_PATH, "r") as f:
    classification = json.load(f)

high_stocks = classification["high_disagreement"]
medium_stocks = classification["medium_disagreement"]
low_stocks = classification["low_disagreement"]

# ── Build task queue ──
def get_months_for_tier(tier):
    """Return list of months to score for each tier."""
    all_months = sorted(list(stock_data[high_stocks[0]].keys()))
    if tier == "high":
        return all_months  # All 228 months
    elif tier == "medium":
        # Quarterly: every 3rd month
        return all_months[::3]
    else:  # low
        # Semi-annual: every 6th month
        return all_months[::6]

def get_agents_for_tier(tier):
    """Return list of agent types to call for each tier."""
    if tier in ("high", "medium"):
        return ["sentiment", "technical", "fundamental"]
    else:  # low — 1 agent only
        return ["sentiment"]

# Build complete task list
tasks = []
for tier, stocks in [("high", high_stocks), ("medium", medium_stocks), ("low", low_stocks)]:
    months = get_months_for_tier(tier)
    agents = get_agents_for_tier(tier)
    for ticker in stocks:
        for month in months:
            for agent in agents:
                tasks.append({
                    "ticker": ticker,
                    "month": month,
                    "agent": agent,
                    "tier": tier,
                })

print(f"Total tasks: {len(tasks)}")
print(f"  High:   {sum(1 for t in tasks if t['tier']=='high'):,}")
print(f"  Medium: {sum(1 for t in tasks if t['tier']=='medium'):,}")
print(f"  Low:    {sum(1 for t in tasks if t['tier']=='low'):,}")

# ── Checkpoint management ──
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"completed": {}, "last_idx": -1, "errors": []}

def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f)

ckpt = load_checkpoint()
completed = ckpt["completed"]
print(f"Resuming from checkpoint: {len(completed)} tasks already completed, {len(tasks) - len(completed)} remaining")

# ── Context builder for LLM ──
def build_context(ticker, month):
    """Build market context for the LLM to make an informed rating."""
    data = stock_data.get(ticker, {})
    if month not in data:
        return None, None
    
    # Get previous 6 months for context
    all_months = sorted(data.keys())
    idx = all_months.index(month)
    prev_months = all_months[max(0, idx-6):idx]
    
    context_parts = []
    for pm in prev_months:
        d = data[pm]
        ret = d.get("return", 0)
        vol = d.get("volume", 0)
        context_parts.append(f"  {pm}: return={ret*100:+.1f}%, volume={vol/1e6:.0f}M")
    
    current = data[month]
    current_ret = current.get("return", 0)
    current_vol = current.get("volume", 0)
    
    context = (
        f"Stock: {ticker}\n"
        f"Rating month: {month}\n"
        f"Previous 6 months performance:\n" + "\n".join(context_parts) + "\n"
        f"Current month return: {current_ret*100:+.1f}% (for reference only — rate based on outlook, not outcome)\n"
        f"Current month volume: {current_vol/1e6:.0f}M"
    )
    
    # Also compute some stats for context
    prev_returns = [data[pm].get("return", 0) for pm in prev_months if pm in data]
    if prev_returns:
        vol_6m = np.std(prev_returns) * 100
        mom_6m = np.prod([1 + r for r in prev_returns]) - 1
        context += f"\n6-month volatility: {vol_6m:.1f}%"
        context += f"\n6-month momentum: {mom_6m*100:+.1f}%"
    
    return context, current_ret

# ── LLM call via z-ai SDK ──
AGENT_PROMPTS = {
    "sentiment": (
        "You are a Sentiment Analyst for equity markets. Based on the stock's recent performance "
        "and market context, rate this stock's outlook on a scale of 1-10 where:\n"
        "1=extremely bearish, 5=neutral, 10=extremely bullish.\n\n"
        "Consider: news sentiment, market mood, investor attention, sector trends.\n"
        "Respond with ONLY a single integer from 1-10. No explanation."
    ),
    "technical": (
        "You are a Technical Analyst for equity markets. Based on the stock's recent price action "
        "and volume patterns, rate this stock's technical outlook on a scale of 1-10 where:\n"
        "1=extremely bearish, 5=neutral, 10=extremely bullish.\n\n"
        "Consider: momentum, trend strength, volume patterns, mean reversion signals.\n"
        "Respond with ONLY a single integer from 1-10. No explanation."
    ),
    "fundamental": (
        "You are a Fundamental Analyst for equity markets. Based on the stock's recent performance "
        "and market context, rate this stock's fundamental outlook on a scale of 1-10 where:\n"
        "1=extremely bearish, 5=neutral, 10=extremely bullish.\n\n"
        "Consider: earnings potential, valuation, growth prospects, competitive position.\n"
        "Respond with ONLY a single integer from 1-10. No explanation."
    ),
}

def call_llm(agent_type, context):
    """Call LLM via z-ai SDK and return rating (1-10)."""
    system_prompt = AGENT_PROMPTS[agent_type]
    user_prompt = f"Market context:\n{context}\n\nYour rating (1-10):"
    
    try:
        result = subprocess.run(
            ["z-ai", "chat", "-s", system_prompt, "-p", user_prompt, "--max-tokens", "10", "--temperature", "0.3"],
            capture_output=True, text=True, timeout=30
        )
        response = result.stdout.strip()
        # Extract integer from response
        for char in response:
            if char.isdigit():
                rating = int(char)
                if 1 <= rating <= 10:
                    return rating
        # Try parsing the whole response
        rating = int(response.strip())
        return max(1, min(10, rating))
    except Exception as e:
        return None

# ── Main scoring loop ──
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.6  # seconds between calls

start_time = time.time()
batch_count = 0
error_count = 0

# Skip already completed tasks
remaining = [(i, t) for i, t in enumerate(tasks) if f"{t['ticker']}_{t['month']}_{t['agent']}" not in completed]
print(f"\nStarting LLM scoring: {len(remaining)} tasks remaining")
print(f"Estimated time: {len(remaining) * RATE_LIMIT_DELAY / 3600:.1f} hours")
print(f"Press Ctrl+C to stop — progress will be saved.\n")

try:
    for task_num, (idx, task) in enumerate(remaining):
        key = f"{task['ticker']}_{task['month']}_{task['agent']}"
        
        if key in completed:
            continue
        
        context, actual_return = build_context(task["ticker"], task["month"])
        if context is None:
            completed[key] = {"rating": None, "error": "no_data"}
            continue
        
        rating = call_llm(task["agent"], context)
        
        if rating is not None:
            completed[key] = {
                "rating": rating,
                "agent": task["agent"],
                "tier": task["tier"],
                "actual_return": actual_return,
            }
        else:
            completed[key] = {"rating": None, "error": "llm_failed"}
            error_count += 1
        
        batch_count += 1
        
        # Checkpoint every BATCH_SIZE calls
        if batch_count % BATCH_SIZE == 0:
            ckpt = {"completed": completed, "last_idx": idx, "errors": ckpt.get("errors", [])}
            save_checkpoint(ckpt)
            elapsed = time.time() - start_time
            rate = batch_count / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - task_num) / rate / 3600 if rate > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"{task_num+1}/{len(remaining)} done | "
                  f"{rate:.1f} calls/s | "
                  f"ETA: {eta:.1f}h | "
                  f"errors: {error_count}")
        
        time.sleep(RATE_LIMIT_DELAY)

except KeyboardInterrupt:
    print(f"\n\nInterrupted! Saving checkpoint...")
    ckpt = {"completed": completed, "last_idx": idx, "errors": []}
    save_checkpoint(ckpt)
    print(f"Saved {len(completed)} completed tasks. Run again to resume.")
    sys.exit(0)

# ── Final save ──
ckpt = {"completed": completed, "last_idx": len(tasks)-1, "errors": []}
save_checkpoint(ckpt)

# Build structured output
ratings = {}
for key, val in completed.items():
    if val.get("rating") is None:
        continue
    parts = key.rsplit("_", 2)
    # key format: TICKER_YYYY-MM_AGENT
    # Need to handle: AAPL_2005-01_sentiment
    ticker = parts[0]
    month = parts[1]
    agent = parts[2]
    
    if ticker not in ratings:
        ratings[ticker] = {}
    if month not in ratings[ticker]:
        ratings[ticker][month] = {}
    ratings[ticker][month][agent] = val["rating"]

with open(RESULTS_PATH, "w") as f:
    json.dump(ratings, f, indent=2)

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"LLM SCORING COMPLETE")
print(f"{'='*60}")
print(f"Total tasks: {len(tasks)}")
print(f"Completed: {len(completed)}")
print(f"Errors: {error_count}")
print(f"Time: {elapsed/3600:.1f} hours")
print(f"Ratings saved to: {RESULTS_PATH}")
