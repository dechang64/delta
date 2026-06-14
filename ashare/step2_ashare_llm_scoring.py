#!/usr/bin/env python3
"""
Step 2: LLM Agent Scoring for A-share stocks.
Same 3-agent framework as US stocks, adapted for A-share context.

Author: Siyi / 2026-06-05
"""

import json, os, time, sys, argparse
import numpy as np
from openai import OpenAI
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=0, help="Max tasks per run (0=all)")
args = parser.parse_args()

OUT = "/home/z/my-project/delta_ashare"
CHECKPOINT = os.path.join(OUT, "ashare_llm_checkpoint.json")
RATINGS_PATH = os.path.join(OUT, "ashare_agent_ratings.json")

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ── Load ──
with open(os.path.join(OUT, "ashare_monthly_returns.json")) as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(os.path.join(OUT, "ashare_ticker_names.json")) as f:
    ticker_names = json.load(f)

# Quarterly months
all_months = sorted(set(m for t in stock_data for m in stock_data[t]))
sample_months = all_months[::3]
print(f"Stocks: {len(stock_data)}, Sample months: {len(sample_months)} (quarterly)")

# ── Agent configs (A-share adapted) ──
AGENT_CONFIGS = {
    "sentiment": {
        "system": "你是一名A股市场情绪分析师。请对股票前景进行1到10的评分（1=极度看空，10=极度看多）。只回复一个数字。",
        "template": "A股: {name}({ticker}). 1个月收益: {r1m:+.1f}%. 3个月收益: {r3m:+.1f}%. 成交量比率: {vrat:.1f}倍. 偏度: {skew:+.2f}. 请评分。"
    },
    "technical": {
        "system": "你是一名A股技术分析师。请对股票前景进行1到10的评分（1=极度看空，10=极度看多）。只回复一个数字。",
        "template": "A股: {name}({ticker}). 6个月趋势: {r6m:+.1f}%. 6个月波动率: {vol6:.1f}%. 1个月收益: {r1m:+.1f}%. 请评分。"
    },
    "fundamental": {
        "system": "你是一名A股基本面分析师。请对股票前景进行1到10的评分（1=极度看空，10=极度看多）。只回复一个数字。",
        "template": "A股: {name}({ticker}). 12个月收益: {r12m:+.1f}%(反转信号). 6个月波动率: {vol6:.1f}%(质量). 最大回撤: {maxdd:+.1f}%. 请评分。"
    }
}

def get_features(ticker, month):
    data = stock_data[ticker]
    months = sorted(data.keys())
    if month not in months:
        return None
    idx = months.index(month)
    f = {"ticker": ticker, "name": ticker_names.get(ticker, ticker)}
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
        with np.errstate(divide="ignore", invalid="ignore"): f["maxdd"] = float(np.min((cum - peak) / np.where(peak != 0, peak, 1))) * 100
    else:
        f["vol6"] = 5.0; f["maxdd"] = -5.0
    f["skew"] = 0.0
    return f

# ── Build task list ──
tickers = sorted(stock_data.keys())
tasks = []
for ticker in tickers:
    for month in sample_months:
        feat = get_features(ticker, month)
        if feat is None:
            continue
        for agent_name in ["sentiment", "technical", "fundamental"]:
            tasks.append((ticker, month, agent_name, feat))

print(f"Total tasks: {len(tasks)}")

# ── Load checkpoint ──
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        checkpoint = json.load(f)
    completed = checkpoint.get("completed", {})
    ratings = checkpoint.get("ratings", {})
else:
    completed = {}
    ratings = {}

remaining = [(t, m, a, f) for t, m, a, f in tasks if f"{t}|{m}|{a}" not in completed]
print(f"Remaining: {len(remaining)} (completed: {len(completed)})")

if remaining:
    start_time = time.time()
    error_count = 0
    batch_count = 0
    
    for i, (ticker, month, agent_name, feat) in enumerate(remaining):
        if args.batch_size > 0 and batch_count >= args.batch_size:
            print(f"[BATCH LIMIT] Reached {args.batch_size} tasks, saving and exiting for resume.")
            break
        key = f"{ticker}|{month}|{agent_name}"
        config = AGENT_CONFIGS[agent_name]
        user_msg = config["template"].format(**feat)
        
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": config["system"]},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=10, temperature=0.3
                )
                rating_text = resp.choices[0].message.content.strip()
                # Extract number from Chinese response
                rating = int(''.join(c for c in rating_text if c.isdigit())[:2]) or 5
                rating = max(1, min(10, rating))
                break
            except Exception as e:
                error_count += 1
                if attempt < 2:
                    time.sleep(2)
                else:
                    rating = 5
        
        completed[key] = rating
        if ticker not in ratings:
            ratings[ticker] = {}
        if month not in ratings[ticker]:
            ratings[ticker][month] = {}
        ratings[ticker][month][agent_name] = rating
        
        time.sleep(0.05)
        batch_count += 1
        
        if batch_count % 200 == 0:
            with open(CHECKPOINT, "w") as f:
                json.dump({"completed": completed, "ratings": ratings}, f)
            elapsed = time.time() - start_time
            rate = batch_count / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - batch_count) / rate / 60 if rate > 0 else 0
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] {batch_count}/{len(remaining)} done | {rate:.1f} calls/s | ETA: {eta:.0f}min | errors: {error_count}")
    
    # Final save
    with open(CHECKPOINT, "w") as f:
        json.dump({"completed": completed, "ratings": ratings}, f)
    with open(RATINGS_PATH, "w") as f:
        json.dump(ratings, f, ensure_ascii=False, indent=2)

# ── Summary ──
all_r = []
for ticker in ratings:
    for month in ratings[ticker]:
        r = ratings[ticker][month]
        if isinstance(r, dict) and len(r) == 3:
            all_r.append([r.get("sentiment", 5), r.get("technical", 5), r.get("fundamental", 5)])

if all_r:
    all_r = np.array(all_r)
    d = np.std(all_r, axis=1)
    print(f"\nA-share LLM Quarterly Rating Summary:")
    print(f"  Observations: {len(all_r)}")
    print(f"  Sentiment:   mean={all_r[:,0].mean():.2f} std={all_r[:,0].std():.2f} range=[{all_r[:,0].min()},{all_r[:,0].max()}]")
    print(f"  Technical:   mean={all_r[:,1].mean():.2f} std={all_r[:,1].std():.2f} range=[{all_r[:,1].min()},{all_r[:,1].max()}]")
    print(f"  Fundamental: mean={all_r[:,2].mean():.2f} std={all_r[:,2].std():.2f} range=[{all_r[:,2].min()},{all_r[:,2].max()}]")
    print(f"  D_post:      mean={d.mean():.2f} std={d.std():.2f} range=[{d.min():.2f},{d.max():.2f}]")
    print(f"  D_post=0:    {(d==0).sum()}/{len(d)} ({(d==0).mean()*100:.1f}%)")

elapsed = time.time() - start_time if remaining else 0
print(f"\nTime: {elapsed/60:.1f} min | Errors: {error_count if remaining else 0}")
print(f"Saved to: {RATINGS_PATH}")
