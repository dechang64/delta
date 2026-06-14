"""
experiments/step2_scoring.py — Multi-Group Agent Scoring for Delta v2.

Scores 183 S&P 500 stocks quarterly using three experiment groups:
  A: Same model, different prompts only (v1 baseline — semantic drift)
  B: Same model, different LoRA + RAG (v2 — professional perspective)
  C: Different models, different LoRA + RAG (full heterogeneity)

Each scoring call respects knowledge base time cutoff (look-ahead control).

Usage:
    # Score Group A only (quick test, uses existing v1 ratings)
    python step2_scoring.py --group A

    # Score Group B (recommended — needs LoRA adapters + RAG KBs)
    python step2_scoring.py --group B --base-model Qwen/Qwen2.5-7B-Instruct

    # Score Group C (full — needs 3 different model endpoints)
    python step2_scoring.py --group C

    # Score all groups
    python step2_scoring.py --group A B C
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import create_agent_group


def load_stock_data(data_dir: str):
    """Load S&P 500 monthly stock data with features for each agent."""
    data_path = os.path.join(data_dir, "sp500_monthly_returns.json")
    with open(data_path) as f:
        raw = json.load(f)
    return raw["data"]


def load_scoring_periods(data_dir: str):
    """Load quarterly scoring periods (Mar/Jun/Sep/Dec)."""
    periods_path = os.path.join(data_dir, "scoring_periods.json")
    if os.path.exists(periods_path):
        with open(periods_path) as f:
            return json.load(f)
    # Default: quarterly from 2005Q1 to 2024Q4
    periods = []
    for year in range(2005, 2025):
        for month in [3, 6, 9, 12]:
            periods.append(f"{year}-{month:02d}")
    return periods


def prepare_agent_input(ticker: str, month: str, stock_data: dict) -> dict:
    """Prepare input info for all three agents.

    CRITICAL: Only include data available AT or BEFORE the scoring month.
    This is the key look-ahead bias control.
    """
    if ticker not in stock_data:
        return {}

    info = {"ticker": ticker, "month": month}

    # Extract available features from stock data
    month_data = stock_data[ticker].get(month, {})
    for key, value in month_data.items():
        info[key] = value

    return info


def score_stock(agents, ticker: str, month: str, stock_data: dict) -> dict:
    """Score a single stock-month with all three agents."""
    info = prepare_agent_input(ticker, month, stock_data)
    if not info:
        return None

    ratings = {}
    for agent in agents:
        rating, reasoning = agent.score(ticker, month, info)
        ratings[agent.domain] = {
            "rating": rating,
            "reasoning": reasoning,
            "agent": agent.name,
        }

    return ratings


def run_scoring(group: str, config: dict, output_dir: str):
    """Run scoring for one experiment group."""
    print(f"\n{'='*60}")
    print(f"  Delta v2 — Group {group} Scoring")
    print(f"{'='*60}")

    # Create agents
    agents = create_agent_group(group, config)
    print(f"  Agents: {[a.name for a in agents]}")

    # Load data
    data_dir = config.get("data_dir", "data")
    stock_data = load_stock_data(data_dir)
    periods = load_scoring_periods(data_dir)
    tickers = sorted(stock_data.keys())
    print(f"  Stocks: {len(tickers)}, Periods: {len(periods)}")
    print(f"  Total scoring calls: {len(tickers) * len(periods) * 3:,}")

    # Score
    results = {}
    total = len(tickers) * len(periods)
    done = 0
    errors = 0

    for period in periods:
        results[period] = {}
        for ticker in tickers:
            ratings = score_stock(agents, ticker, period, stock_data)
            if ratings:
                results[period][ticker] = ratings
                done += 1
            else:
                errors += 1

            if done % 100 == 0:
                print(f"  Progress: {done}/{total} scored, {errors} errors")

    # Save
    out_path = os.path.join(output_dir, f"agent_ratings_group{group}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Total: {done} scored, {errors} errors")

    return results


def main():
    parser = argparse.ArgumentParser(description="Delta v2 Agent Scoring")
    parser.add_argument("--group", nargs="+", default=["B"],
                        help="Experiment group(s): A, B, C")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--api-key", help="OpenAI-compatible API key")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-dir", default="agents/lora")
    parser.add_argument("--kb-dir", default="rag/knowledge_bases")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "data_dir": args.data_dir,
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "api_base": args.api_base or os.environ.get("OPENAI_API_BASE"),
        "base_model": args.base_model,
        "lora_dir": args.lora_dir,
        "kb_dir": args.kb_dir,
    }

    for group in args.group:
        run_scoring(group, config, args.output_dir)


if __name__ == "__main__":
    main()
