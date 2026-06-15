#!/usr/bin/env python3
"""
run_batch_e3.py — Robust batch E3 scoring with checkpointing

Usage:
  python3 run_batch_e3.py                    # Run default batch
  python3 run_batch_e3.py --phase cache      # Only cache data (no scoring)
  python3 run_batch_e3.py --phase score      # Only score (assumes cached)
  python3 run_batch_e3.py --stocks 10        # Limit to N stocks
  python3 run_batch_e3.py --max-calls 500    # Stop after N new calls
  python3 run_batch_e3.py --agents sentiment # Only run specified agents

Checkpoint: saves every 200 calls. Resume by running again.
"""

import os, sys, json, time, argparse
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

# ── Load base data ──
def load_data():
    with open(os.path.join(DATA_DIR, "sp500_monthly_returns.json")) as f:
        stock_monthly = json.load(f)["data"]
    with open(os.path.join(DATA_DIR, "disagreement_classification.json")) as f:
        cls = json.load(f)
    return stock_monthly, cls

# ── Caching ──
def cache_all_data(tickers):
    """Download and cache daily + fundamental data for all tickers."""
    daily_cache = {}
    fund_cache = {}
    
    # Daily data
    daily_path = os.path.join(CACHE_DIR, "daily_data_cache.json")
    if os.path.exists(daily_path):
        with open(daily_path) as f:
            cached_list = json.load(f)
        print(f"  Daily cache: {len(cached_list)} stocks already cached")
    else:
        cached_list = []
    
    remaining_daily = [t for t in tickers if t not in cached_list]
    if remaining_daily:
        print(f"  Downloading daily data for {len(remaining_daily)} stocks...")
        for i, t in enumerate(remaining_daily):
            try:
                df = download_daily_data(t)
                if df is not None and len(df) > 100:
                    # Save as CSV
                    csv_path = os.path.join(CACHE_DIR, f"daily_{t}.csv")
                    df.to_csv(csv_path)
                    cached_list.append(t)
            except Exception:
                pass
            if (i+1) % 10 == 0 or i == len(remaining_daily)-1:
                with open(daily_path, 'w') as f:
                    json.dump(cached_list, f)
                print(f"    {i+1}/{len(remaining_daily)} done ({len(cached_list)} total)")
            time.sleep(0.05)
    
    # Fundamental data
    fund_path = os.path.join(CACHE_DIR, "fundamental_cache.json")
    if os.path.exists(fund_path):
        with open(fund_path) as f:
            fund_cache = json.load(f)
        print(f"  Fundamental cache: {len(fund_cache)} stocks")
    
    remaining_fund = [t for t in tickers if t not in fund_cache]
    if remaining_fund:
        print(f"  Getting fundamentals for {len(remaining_fund)} stocks...")
        for i, t in enumerate(remaining_fund):
            try:
                feat = compute_fundamental_features(t)
                if feat and len(feat) >= 6:
                    fund_cache[t] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                     for k, v in feat.items()}
            except Exception:
                pass
            if (i+1) % 10 == 0 or i == len(remaining_fund)-1:
                with open(fund_path, 'w') as f:
                    json.dump(fund_cache, f)
                print(f"    {i+1}/{len(remaining_fund)} done ({len(fund_cache)} total)")
            time.sleep(0.15)
    
    return cached_list, fund_cache

# ── Build task list ──
def build_tasks(stock_monthly, cls, tickers, agents, anon=False):
    """Build scoring task list with quarter-end months."""
    quarters = [f"{y}-{m:02d}" for y in range(2005, 2025) for m in [3, 6, 9, 12]]
    
    tasks = []
    for ticker in tickers:
        if ticker not in stock_monthly:
            continue
        available_months = set(stock_monthly[ticker].keys())
        for month in quarters:
            if month not in available_months:
                continue
            for agent in agents:
                tasks.append({
                    'ticker': ticker,
                    'month': month,
                    'agent': agent,
                    'key': f"{ticker}_{month}_{agent}_{('anon' if anon else 'named')}",
                })
    return tasks

# ── Score one task ──
def score_task(task, stock_monthly, daily_cache_list, fund_cache, anon=False):
    ticker = task['ticker']
    month = task['month']
    agent = task['agent']
    
    # Get features
    if agent == 'sentiment':
        feat = compute_sentiment_features(stock_monthly[ticker], month)
    elif agent == 'technical':
        csv_path = os.path.join(CACHE_DIR, f"daily_{ticker}.csv")
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            feat = compute_technical_features(df, f"{month}-28")
        except Exception:
            return None
    else:  # fundamental
        if ticker not in fund_cache:
            return None
        feat = fund_cache[ticker]
    
    if not feat:
        return None
    
    # Call model
    result = call_probability_model(agent, ticker, feat, anonymized=anon)
    if 'error' in result:
        return None
    
    # Add next month return
    if ticker in stock_monthly:
        months_sorted = sorted(stock_monthly[ticker].keys())
        if month in months_sorted:
            idx = months_sorted.index(month)
            if idx + 1 < len(months_sorted):
                result['ret_next'] = stock_monthly[ticker][months_sorted[idx+1]]['return'] * 100
    
    return result

# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', choices=['cache', 'score', 'all'])
    parser.add_argument('--stocks', type=int, default=0, help='Max stocks to process')
    parser.add_argument('--max-calls', type=int, default=0, help='Max new API calls')
    parser.add_argument('--agents', nargs='+', default=['sentiment', 'technical', 'fundamental'])
    parser.add_argument('--anon', action='store_true', help='Run anonymized scoring')
    parser.add_argument('--checkpoint-every', type=int, default=200)
    args = parser.parse_args()
    
    stock_monthly, cls = load_data()
    
    # Stratify tickers
    high = cls.get('high_disagreement', [])
    medium = cls.get('medium_disagreement', [])
    low = cls.get('low_disagreement', [])
    
    # Priority: high > medium > low
    if args.stocks > 0:
        n_h = min(args.stocks // 2, len(high))
        n_m = min(args.stocks // 3, len(medium))
        n_l = min(args.stocks - n_h - n_m, len(low))
        tickers = high[:n_h] + medium[:n_m] + low[:n_l]
    else:
        tickers = high + medium + low
    
    # Filter to available
    tickers = [t for t in tickers if t in stock_monthly]
    print(f"Stocks: {len(tickers)} (high={len([t for t in tickers if t in high])}, "
          f"medium={len([t for t in tickers if t in medium])}, "
          f"low={len([t for t in tickers if t in low])})")
    
    # Phase 1: Cache
    if args.phase in ['cache', 'all']:
        print("\n[Phase 1] Caching data...")
        daily_list, fund_cache = cache_all_data(tickers)
        print(f"  Daily: {len(daily_list)}, Fundamental: {len(fund_cache)}")
    
    if args.phase == 'cache':
        return
    
    # Phase 2: Score
    if args.phase in ['score', 'all']:
        # Load caches
        fund_path = os.path.join(CACHE_DIR, "fundamental_cache.json")
        if os.path.exists(fund_path):
            with open(fund_path) as f:
                fund_cache = json.load(f)
        else:
            fund_cache = {}
        
        daily_path = os.path.join(CACHE_DIR, "daily_data_cache.json")
        if os.path.exists(daily_path):
            with open(daily_path) as f:
                daily_list = json.load(f)
        else:
            daily_list = []
        
        # Build tasks
        tasks = build_tasks(stock_monthly, cls, tickers, args.agents, anon=args.anon)
        print(f"\n[Phase 2] Scoring: {len(tasks)} tasks")
        
        # Load checkpoint
        cond = 'anon' if args.anon else 'named'
        ckpt_path = os.path.join(OUT, f"e3_checkpoint_{cond}.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            completed = ckpt.get('completed', {})
        else:
            completed = {}
        
        remaining = [t for t in tasks if t['key'] not in completed]
        print(f"  Completed: {len(completed)}, Remaining: {len(remaining)}")
        
        if not remaining:
            print("  All tasks completed!")
        else:
            # Limit calls
            if args.max_calls > 0:
                remaining = remaining[:args.max_calls]
                print(f"  Limited to {args.max_calls} calls this run")
            
            n_done = 0
            n_errors = 0
            t_start = time.time()
            
            for i, task in enumerate(remaining):
                result = score_task(task, stock_monthly, daily_list, fund_cache, anon=args.anon)
                
                if result is not None:
                    completed[task['key']] = result
                    n_done += 1
                else:
                    n_errors += 1
                
                # Checkpoint
                if (n_done + n_errors) % args.checkpoint_every == 0:
                    with open(ckpt_path, 'w') as f:
                        json.dump(completed, f, ensure_ascii=False)
                    elapsed = time.time() - t_start
                    rate = (n_done + n_errors) / elapsed * 60
                    print(f"  {n_done + n_errors}/{len(remaining)} done "
                          f"({rate:.0f}/min, {n_errors} errors, {elapsed/60:.1f}min)")
                
                time.sleep(0.12)
            
            # Final save
            with open(ckpt_path, 'w') as f:
                json.dump(completed, f, ensure_ascii=False)
            
            elapsed = time.time() - t_start
            print(f"\n  Batch done: {n_done} scored, {n_errors} errors, {elapsed/60:.1f}min")
        
        # Reorganize into panel format
        print("\n[Phase 3] Building panel...")
        panel = []
        for key, val in completed.items():
            parts = key.rsplit('_', 3)  # ticker_month_agent_condition
            if len(parts) < 4:
                continue
            # Re-parse: ticker may contain underscore, but month is YYYY-MM format
            # Key format: TICKER_YYYY-MM_agent_condition
            # Split from right: condition, agent, month, ticker
            condition = parts[-1]
            agent = parts[-2]
            month = parts[-3]
            ticker = '_'.join(parts[:-3])
            
            if 'probs' not in val:
                continue
            
            row = {
                'ticker': ticker,
                'month': month,
                'agent': agent,
                'condition': condition,
            }
            
            # Add probabilities
            for pk, pv in val['probs'].items():
                row[f'prob_{pk}'] = pv
            
            # Add metrics
            row['H'] = val.get('H', np.nan)
            row['direction'] = val.get('direction', '')
            row['ret_next'] = val.get('ret_next', np.nan)
            row['model'] = val.get('model', '')
            
            panel.append(row)
        
        if panel:
            df = pd.DataFrame(panel)
            panel_path = os.path.join(OUT, f"e3_panel_{cond}.csv")
            df.to_csv(panel_path, index=False)
            print(f"  Panel saved: {len(df)} rows → {panel_path}")
            
            # Quick stats
            print(f"  Stocks: {df['ticker'].nunique()}")
            print(f"  Months: {df['month'].nunique()}")
            print(f"  Agents: {df['agent'].unique()}")
            print(f"  H range: [{df['H'].min():.4f}, {df['H'].max():.4f}]")
            print(f"  H unique: {df['H'].round(4).nunique()}")
            
            # Check granularity
            prob_cols = [c for c in df.columns if c.startswith('prob_') and c != 'prob_trend_strength']
            all_vals = df[prob_cols].values.flatten()
            all_vals = all_vals[~np.isnan(all_vals)]
            mult5 = np.sum(all_vals % 5 == 0) / len(all_vals)
            print(f"  Prob 5x multiples: {mult5:.1%}")
        else:
            print("  No results to build panel from")


if __name__ == '__main__':
    main()
