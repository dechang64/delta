#!/usr/bin/env python3
"""
batch_e3.py — Batch E3 scoring pipeline

Strategy:
  - Download daily data + fundamentals ONCE, cache to disk
  - Score by stratum (High/Medium/Low disagreement)
  - Checkpoint every 200 calls
  - Resume from checkpoint if interrupted

Author: Siyi / 2026-06-13
"""

import os, json, time, sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engine import (
    compute_sentiment_features, compute_technical_features, 
    compute_fundamental_features, download_daily_data
)
from scoring_v2 import call_probability_model

OUT = "/home/z/my-project/delta_jfe_v2"
DATA_DIR = "/home/z/my-project/delta_jfe"
CACHE_DIR = os.path.join(OUT, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)


def load_monthly_data():
    with open(os.path.join(DATA_DIR, "sp500_monthly_returns.json")) as f:
        return json.load(f)["data"]


def load_classification():
    path = os.path.join(DATA_DIR, "disagreement_classification.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        # Normalize keys
        return {
            "high": d.get("high_disagreement", []),
            "medium": d.get("medium_disagreement", []),
            "low": d.get("low_disagreement", []),
        }
    return {"high": [], "medium": [], "low": []}


def cache_daily_data(ticker: str) -> pd.DataFrame:
    """Download and cache daily price data."""
    cache_file = os.path.join(CACHE_DIR, f"daily_{ticker}.parquet")
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    
    df = download_daily_data(ticker)
    if df is not None and len(df) > 0:
        df.to_parquet(cache_file)
    return df


def cache_fundamentals(ticker: str) -> dict:
    """Download and cache fundamental data."""
    cache_file = os.path.join(CACHE_DIR, f"fund_{ticker}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    fund = compute_fundamental_features(ticker)
    if fund:
        # Convert numpy types
        clean = {}
        for k, v in fund.items():
            if isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            else:
                clean[k] = v
        with open(cache_file, 'w') as f:
            json.dump(clean, f)
    return fund


def get_quarter_months(stratum: str, classification: dict) -> list:
    """Get (ticker, month) pairs for a stratum."""
    if stratum == "high":
        tickers = classification["high"]
        # All 80 quarter-end months
        months = [f"{y}-{m:02d}" for y in range(2005, 2025) for m in [3, 6, 9, 12]]
    elif stratum == "medium":
        tickers = classification["medium"]
        # 20 quarter-end months (yearly)
        months = [f"{y}-{m:02d}" for y in range(2010, 2025, 1) for m in [3, 9]]
    else:  # low
        tickers = classification["low"]
        # 10 semi-annual points
        months = [f"{y}-06" for y in range(2015, 2025)]
    
    tasks = []
    for t in tickers:
        for m in months:
            tasks.append((t, m))
    return tasks


def run_batch(tasks, stock_monthly, daily_cache, fund_cache, 
              checkpoint_path, anonymized=False, agents=None):
    """Run batch scoring with checkpoint support."""
    if agents is None:
        agents = ["sentiment", "technical", "fundamental"]
    
    # Load checkpoint
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            completed = json.load(f).get("completed", {})
    
    n_total = len(tasks) * len(agents)
    n_done = 0
    n_errors = 0
    t_start = time.time()
    
    for ticker, month in tasks:
        for agent in agents:
            key = f"{ticker}_{month}_{agent}_{'anon' if anonymized else 'named'}"
            if key in completed:
                n_done += 1
                continue
            
            # Get features
            feat = None
            if agent == "sentiment":
                if ticker in stock_monthly:
                    feat = compute_sentiment_features(stock_monthly[ticker], month)
            elif agent == "technical":
                if ticker in daily_cache:
                    feat = compute_technical_features(daily_cache[ticker], f"{month}-28")
            else:  # fundamental
                if ticker in fund_cache:
                    feat = fund_cache[ticker]
            
            if feat is None:
                completed[key] = {"error": "no_features"}
                n_errors += 1
                continue
            
            # Call model
            try:
                r = call_probability_model(agent, ticker, feat, anonymized=anonymized)
                if "error" in r:
                    completed[key] = {"error": r["error"]}
                    n_errors += 1
                else:
                    completed[key] = r
            except Exception as e:
                completed[key] = {"error": str(e)[:100]}
                n_errors += 1
            
            n_done += 1
            
            # Checkpoint every 200 calls
            if n_done % 200 == 0:
                with open(checkpoint_path, 'w') as f:
                    json.dump({"completed": completed}, f)
                elapsed = time.time() - t_start
                rate = n_done / elapsed * 60
                remaining = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  Checkpoint: {n_done}/{n_total} done, {n_errors} errors, "
                      f"rate={rate:.0f}/min, ETA={remaining:.0f}min")
    
    # Final save
    with open(checkpoint_path, 'w') as f:
        json.dump({"completed": completed}, f)
    
    elapsed = time.time() - t_start
    print(f"  Batch done: {n_done} done, {n_errors} errors, {elapsed/60:.1f}min")
    return completed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["high", "medium", "low", "all"], default="high")
    parser.add_argument("--max-stocks", type=int, default=0, help="Limit stocks (0=all)")
    parser.add_argument("--agents", nargs="+", default=["sentiment", "technical", "fundamental"])
    parser.add_argument("--anonymized", action="store_true")
    parser.add_argument("--skip-cache", action="store_true", help="Re-download data")
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    stock_monthly = load_monthly_data()
    classification = load_classification()
    
    # Determine stocks
    if args.stratum == "all":
        tickers = list(stock_monthly.keys())
    else:
        tickers = classification.get(args.stratum, [])
    
    if args.max_stocks > 0:
        tickers = tickers[:args.max_stocks]
    
    print(f"Stratum: {args.stratum}, Stocks: {len(tickers)}")
    
    # Pre-cache daily and fundamental data
    print("Caching daily data...")
    daily_cache = {}
    for i, t in enumerate(tickers):
        if t in stock_monthly:
            daily_cache[t] = cache_daily_data(t) if not args.skip_cache else download_daily_data(t)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(tickers)} daily cached")
    
    print("Caching fundamental data...")
    fund_cache = {}
    for i, t in enumerate(tickers):
        fund_cache[t] = cache_fundamentals(t) if not args.skip_cache else compute_fundamental_features(t)
        time.sleep(0.1)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(tickers)} fundamentals cached")
    
    # Build task list
    tasks = get_quarter_months(args.stratum, classification) if args.stratum != "all" else \
            [(t, m) for t in tickers for m in [f"{y}-{m:02d}" for y in range(2005,2025) for m in [3,6,9,12]]]
    
    # Filter to only requested tickers
    tasks = [(t, m) for t, m in tasks if t in tickers]
    
    cond = "anon" if args.anonymized else "named"
    ckpt_path = os.path.join(OUT, f"e3_checkpoint_{args.stratum}_{cond}.json")
    
    print(f"\nRunning E3 batch: {len(tasks)} ticker-months × {len(args.agents)} agents "
          f"= {len(tasks)*len(args.agents)} calls")
    
    results = run_batch(tasks, stock_monthly, daily_cache, fund_cache,
                       ckpt_path, anonymized=args.anonymized, agents=args.agents)
