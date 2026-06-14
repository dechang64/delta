"""
experiments/prepare_lora_data.py — Prepare LoRA fine-tuning data for 3 Delta Agents.

Downloads public datasets and converts to instruction-following format:
  {"instruction": "...", "input": "...", "output": "RATING: X\\nREASONING: ..."}

Datasets:
  Sentiment Agent:
    - FPB (Financial PhraseBank): 4840 sentences, 3-class sentiment
    - TFNS (Twitter Financial News Sentiment): 11932 tweets, 3-class
    - FiQA Sentiment: financial news with scores

  Technical Agent:
    - Generated from stock price data + technical indicators
    - yfinance → compute RSI, MACD, MA → synthetic rating labels

  Fundamental Agent:
    - Generated from financial statements + valuation metrics
    - SEC EDGAR data or yfinance fundamentals → synthetic rating labels

Usage:
    # Download and prepare all
    python prepare_lora_data.py --all --output-dir data/lora/

    # Individual agents
    python prepare_lora_data.py --sentiment --output-dir data/lora/
    python prepare_lora_data.py --technical --output-dir data/lora/
    python prepare_lora_data.py --fundamental --output-dir data/lora/
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime


# ── Rating mapping ───────────────────────────────────────────────────
# Map 3-class sentiment → 1-10 scale
SENTIMENT_TO_RATING = {
    "negative": 2,
    "Neutral": 4,
    "neutral": 5,
    "Positive": 7,
    "positive": 8,
}

# Add controlled noise to avoid trivial mapping
def add_rating_noise(rating: int, sigma: float = 0.8) -> int:
    """Add Gaussian noise to rating, clip to [1, 10]."""
    noisy = rating + random.gauss(0, sigma)
    return max(1, min(10, round(noisy)))


# ── Sentiment Agent Data ─────────────────────────────────────────────

def prepare_sentiment_data(output_dir: str):
    """Prepare LoRA training data for Sentiment Agent.

    Sources:
      1. Financial PhraseBank (FPB): 4840 sentences
      2. Twitter Financial News Sentiment (TFNS): 11932 tweets
      3. FiQA: financial news with sentiment scores
    """
    print("\n=== Preparing Sentiment Agent LoRA Data ===")
    sentiment_dir = Path(output_dir) / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 1. Financial PhraseBank
    try:
        from datasets import load_dataset
        print("  Loading Financial PhraseBank...")
        fpb = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        for item in fpb["train"]:
            sentence = item["sentence"]
            label = item["label"]  # 0=negative, 1=neutral, 2=positive
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = label_map[label]
            rating = add_rating_noise(SENTIMENT_TO_RATING[sentiment])

            all_samples.append({
                "instruction": "You are a Sentiment Analyst. Rate this financial news on a scale of 1 (very bearish) to 10 (very bullish).",
                "input": f"Financial news: {sentence}",
                "output": f"RATING: {rating}\nREASONING: This news is {sentiment} for the company. {sentence}",
            })
        print(f"  FPB: {len(fpb['train'])} samples loaded")
    except Exception as e:
        print(f"  FPB failed: {e}")
        print("  Install: pip install datasets")

    # 2. Twitter Financial News Sentiment
    try:
        from datasets import load_dataset
        print("  Loading TFNS...")
        tfns = load_dataset("zeroshot/twitter-financial-news-sentiment", trust_remote_code=True)
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        for split in ["train", "validation"]:
            if split in tfns:
                for item in tfns[split]:
                    text = item["text"]
                    sentiment = label_map[item["label"]]
                    rating = add_rating_noise(SENTIMENT_TO_RATING[sentiment])

                    all_samples.append({
                        "instruction": "You are a Sentiment Analyst. Rate this financial tweet on a scale of 1 (very bearish) to 10 (very bullish).",
                        "input": f"Financial tweet: {text}",
                        "output": f"RATING: {rating}\nREASONING: This tweet expresses {sentiment} sentiment. {text}",
                    })
        total = sum(len(tfns[s]) for s in tfns if s in tfns)
        print(f"  TFNS: {total} samples loaded")
    except Exception as e:
        print(f"  TFNS failed: {e}")

    # 3. FiQA Sentiment (if available)
    try:
        from datasets import load_dataset
        print("  Loading FiQA...")
        fiqa = load_dataset("pauri32/fiqa-2018", trust_remote_code=True)
        for split in ["train", "validation", "test"]:
            if split in fiqa:
                for item in fiqa[split]:
                    text = item.get("sentence", item.get("input", ""))
                    score = float(item.get("score", 5))
                    # Map [-1, 1] score to [1, 10]
                    rating = max(1, min(10, round((score + 1) * 5)))

                    all_samples.append({
                        "instruction": "You are a Sentiment Analyst. Rate this financial text on a scale of 1 (very bearish) to 10 (very bullish).",
                        "input": f"Financial text: {text}",
                        "output": f"RATING: {rating}\nREASONING: Market sentiment analysis of this text suggests a rating of {rating}/10.",
                    })
        print(f"  FiQA samples loaded")
    except Exception as e:
        print(f"  FiQA skipped: {e}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)

    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:],
    }

    for split_name, data in splits.items():
        out_path = sentiment_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {split_name}: {len(data)} samples → {out_path}")

    print(f"  Total sentiment samples: {n}")


# ── Technical Agent Data ─────────────────────────────────────────────

def prepare_technical_data(output_dir: str):
    """Prepare LoRA training data for Technical Agent.

    Strategy: Generate synthetic training data from historical stock prices.
    For each stock-date, compute technical indicators and create
    a (indicators → rating) pair using a rule-based labeler.
    """
    print("\n=== Preparing Technical Agent LoRA Data ===")
    tech_dir = Path(output_dir) / "technical"
    tech_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    try:
        import yfinance as yf

        # Download S&P 500 stocks (sample 50)
        print("  Downloading stock data from Yahoo Finance...")
        sp500_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "JNJ", "WMT", "PG", "UNH", "MA", "HD", "DIS", "BAC",
            "XOM", "PFE", "CSCO", "ADBE", "NFLX", "CRM", "INTC", "VZ", "T",
            "CSCO", "PEP", "ABT", "MRK", "TMO", "CVX", "ABBV", "KO", "NKE",
            "MCD", "COST", "AVGO", "TXN", "LLY", "QCOM", "ORCL", "WFC", "PM",
        ]

        for ticker in sp500_tickers[:30]:  # Sample 30 for speed
            try:
                data = yf.download(ticker, period="2y", interval="1d", progress=False)
                if data.empty or len(data) < 60:
                    continue

                # Compute technical indicators
                close = data["Close"].values.flatten()
                volume = data["Volume"].values.flatten()

                for i in range(50, len(close) - 5, 5):  # Every 5 days
                    window = close[i-50:i]
                    vol_window = volume[i-50:i]

                    # Simple indicators
                    ma20 = np.mean(window[-20:])
                    ma50 = np.mean(window)
                    rsi = compute_rsi(window[-14:])
                    price = window[-1]
                    vol_ratio = vol_window[-1] / (np.mean(vol_window[-20:]) + 1e-10)

                    # Future return (for labeling)
                    future_ret = (close[i+5] - price) / price

                    # Rule-based rating
                    rating = rule_based_technical_rating(
                        price, ma20, ma50, rsi, vol_ratio, future_ret
                    )

                    # Create description
                    desc = (
                        f"Stock: {ticker}\n"
                        f"Current Price: ${price:.2f}\n"
                        f"MA(20): ${ma20:.2f}, MA(50): ${ma50:.2f}\n"
                        f"RSI(14): {rsi:.1f}\n"
                        f"Volume Ratio: {vol_ratio:.2f}x\n"
                        f"Price vs MA20: {((price/ma20)-1)*100:+.1f}%\n"
                        f"Price vs MA50: {((price/ma50)-1)*100:+.1f}%"
                    )

                    all_samples.append({
                        "instruction": "You are a Technical Analyst. Rate this stock on a scale of 1 (strong sell) to 10 (strong buy) based on technical indicators.",
                        "input": desc,
                        "output": f"RATING: {rating}\nREASONING: Technical analysis based on price action and momentum indicators.",
                    })
            except Exception as e:
                print(f"  {ticker} failed: {e}")
                continue

    except ImportError:
        print("  yfinance not installed. Generating synthetic technical data...")
        all_samples = generate_synthetic_technical(5000)

    if not all_samples:
        print("  No real data, generating synthetic...")
        all_samples = generate_synthetic_technical(5000)

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)

    for split_name, data in [
        ("train", all_samples[:train_end]),
        ("val", all_samples[train_end:val_end]),
        ("test", all_samples[val_end:]),
    ]:
        out_path = tech_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {split_name}: {len(data)} samples → {out_path}")

    print(f"  Total technical samples: {n}")


def compute_rsi(prices, period=14):
    """Compute RSI from price array."""
    deltas = np.diff(prices[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rule_based_technical_rating(price, ma20, ma50, rsi, vol_ratio, future_ret):
    """Rule-based technical rating for training label generation."""
    score = 5.0  # neutral baseline

    # MA crossover signals
    if price > ma20: score += 0.5
    else: score -= 0.5
    if price > ma50: score += 0.5
    else: score -= 0.5
    if ma20 > ma50: score += 0.5  # golden cross
    else: score -= 0.5

    # RSI signals
    if rsi < 30: score += 1.0  # oversold → buy
    elif rsi > 70: score -= 1.0  # overbought → sell
    elif 40 <= rsi <= 60: score += 0.0  # neutral

    # Volume confirmation
    if vol_ratio > 1.5: score += 0.3  # high volume = conviction
    elif vol_ratio < 0.5: score -= 0.3

    # Use future return as soft signal (with noise to avoid overfitting)
    score += future_ret * 10  # scale return to rating points
    score += random.gauss(0, 0.5)  # add noise

    return max(1, min(10, round(score)))


def generate_synthetic_technical(n=5000):
    """Generate synthetic technical analysis training data."""
    samples = []
    for _ in range(n):
        price = random.uniform(50, 500)
        ma20 = price * random.uniform(0.9, 1.1)
        ma50 = price * random.uniform(0.85, 1.15)
        rsi = random.uniform(20, 80)
        vol_ratio = random.uniform(0.3, 3.0)
        future_ret = random.gauss(0, 0.05)

        rating = rule_based_technical_rating(price, ma20, ma50, rsi, vol_ratio, future_ret)

        desc = (
            f"Current Price: ${price:.2f}\n"
            f"MA(20): ${ma20:.2f}, MA(50): ${ma50:.2f}\n"
            f"RSI(14): {rsi:.1f}\n"
            f"Volume Ratio: {vol_ratio:.2f}x"
        )

        samples.append({
            "instruction": "You are a Technical Analyst. Rate this stock on a scale of 1 (strong sell) to 10 (strong buy) based on technical indicators.",
            "input": desc,
            "output": f"RATING: {rating}\nREASONING: Technical analysis of price action and momentum.",
        })
    return samples


# ── Fundamental Agent Data ───────────────────────────────────────────

def prepare_fundamental_data(output_dir: str):
    """Prepare LoRA training data for Fundamental Agent.

    Strategy: Use yfinance fundamental data to generate
    (financials → rating) pairs.
    """
    print("\n=== Preparing Fundamental Agent LoRA Data ===")
    fund_dir = Path(output_dir) / "fundamental"
    fund_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    try:
        import yfinance as yf

        print("  Downloading fundamental data from Yahoo Finance...")
        sp500_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "JPM", "V", "JNJ", "WMT", "PG", "UNH", "MA", "HD",
            "XOM", "PFE", "CSCO", "ADBE", "NFLX", "CRM", "INTC",
            "PEP", "ABT", "MRK", "CVX", "ABBV", "KO", "NKE", "MCD",
        ]

        for ticker in sp500_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if not info or not info.get("sector"):
                    continue

                # Extract fundamentals
                pe = info.get("trailingPE", info.get("forwardPE", 0))
                pb = info.get("priceToBook", 0)
                roe = info.get("returnOnEquity", 0)
                rev_growth = info.get("revenueGrowth", 0)
                earn_growth = info.get("earningsGrowth", 0)
                de_ratio = info.get("debtToEquity", 0)
                fcf = info.get("freeCashflow", 0)
                div_yield = info.get("dividendYield", 0)
                mcap = info.get("marketCap", 0)
                sector = info.get("sector", "Unknown")

                # Rule-based fundamental rating
                rating = rule_based_fundamental_rating(
                    pe, pb, roe, rev_growth, earn_growth, de_ratio, div_yield
                )

                desc = (
                    f"Stock: {ticker} | Sector: {sector}\n"
                    f"P/E: {pe:.1f if pe else 'N/A'} | P/B: {pb:.1f if pb else 'N/A'}\n"
                    f"ROE: {roe:.1% if roe else 'N/A'}\n"
                    f"Revenue Growth: {rev_growth:.1% if rev_growth else 'N/A'}\n"
                    f"Earnings Growth: {earn_growth:.1% if earn_growth else 'N/A'}\n"
                    f"Debt/Equity: {de_ratio:.1f if de_ratio else 'N/A'}\n"
                    f"Dividend Yield: {div_yield:.2% if div_yield else 'N/A'}\n"
                    f"Market Cap: ${mcap/1e9:.0f}B" if mcap else ""
                )

                all_samples.append({
                    "instruction": "You are a Fundamental Analyst. Rate this stock on a scale of 1 (very overvalued) to 10 (very undervalued) based on financial fundamentals.",
                    "input": desc,
                    "output": f"RATING: {rating}\nREASONING: Fundamental analysis based on valuation, growth, and balance sheet metrics.",
                })
            except Exception as e:
                continue

    except ImportError:
        print("  yfinance not installed. Generating synthetic fundamental data...")
        all_samples = generate_synthetic_fundamental(5000)

    if len(all_samples) < 500:
        print("  Not enough real data, supplementing with synthetic...")
        all_samples.extend(generate_synthetic_fundamental(5000 - len(all_samples)))

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)

    for split_name, data in [
        ("train", all_samples[:train_end]),
        ("val", all_samples[train_end:val_end]),
        ("test", all_samples[val_end:]),
    ]:
        out_path = fund_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {split_name}: {len(data)} samples → {out_path}")

    print(f"  Total fundamental samples: {n}")


def rule_based_fundamental_rating(pe, pb, roe, rev_growth, earn_growth, de_ratio, div_yield):
    """Rule-based fundamental rating for label generation."""
    score = 5.0

    # Valuation (lower PE/PB = more undervalued)
    if pe and pe > 0:
        if pe < 10: score += 1.5
        elif pe < 20: score += 0.5
        elif pe > 40: score -= 1.0
        else: score -= 0.3

    if pb and pb > 0:
        if pb < 1: score += 1.0  # below book value
        elif pb < 3: score += 0.3
        elif pb > 5: score -= 0.5

    # Growth
    if rev_growth and rev_growth > 0:
        score += min(rev_growth * 5, 1.5)
    elif rev_growth:
        score -= 0.5

    if earn_growth and earn_growth > 0:
        score += min(earn_growth * 5, 1.5)
    elif earn_growth:
        score -= 0.5

    # ROE
    if roe and roe > 0:
        if roe > 0.15: score += 0.8
        elif roe > 0.10: score += 0.3
        elif roe < 0: score -= 0.5

    # Debt
    if de_ratio and de_ratio > 0:
        if de_ratio > 200: score -= 1.0
        elif de_ratio > 100: score -= 0.5
        elif de_ratio < 50: score += 0.3

    # Dividend
    if div_yield and div_yield > 0:
        if div_yield > 0.04: score += 0.5
        elif div_yield > 0.02: score += 0.2

    score += random.gauss(0, 0.5)
    return max(1, min(10, round(score)))


def generate_synthetic_fundamental(n=5000):
    """Generate synthetic fundamental analysis training data."""
    samples = []
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial"]

    for _ in range(n):
        pe = random.choice([random.uniform(5, 15), random.uniform(15, 30), random.uniform(30, 80)])
        pb = random.uniform(0.5, 8)
        roe = random.uniform(-0.2, 0.4)
        rev_growth = random.uniform(-0.1, 0.3)
        earn_growth = random.uniform(-0.2, 0.4)
        de_ratio = random.uniform(0, 300)
        div_yield = random.uniform(0, 0.06)
        sector = random.choice(sectors)
        mcap = random.uniform(1, 500)

        rating = rule_based_fundamental_rating(pe, pb, roe, rev_growth, earn_growth, de_ratio, div_yield)

        desc = (
            f"Sector: {sector} | Market Cap: ${mcap:.0f}B\n"
            f"P/E: {pe:.1f} | P/B: {pb:.1f}\n"
            f"ROE: {roe:.1%}\n"
            f"Revenue Growth: {rev_growth:.1%}\n"
            f"Earnings Growth: {earn_growth:.1%}\n"
            f"Debt/Equity: {de_ratio:.0f}\n"
            f"Dividend Yield: {div_yield:.2%}"
        )

        samples.append({
            "instruction": "You are a Fundamental Analyst. Rate this stock on a scale of 1 (very overvalued) to 10 (very undervalued) based on financial fundamentals.",
            "input": desc,
            "output": f"RATING: {rating}\nREASONING: Fundamental analysis of valuation, growth, and balance sheet.",
        })
    return samples


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare LoRA fine-tuning data for Delta Agents")
    parser.add_argument("--all", action="store_true", help="Prepare all agents")
    parser.add_argument("--sentiment", action="store_true")
    parser.add_argument("--technical", action="store_true")
    parser.add_argument("--fundamental", action="store_true")
    parser.add_argument("--output-dir", default="data/lora")
    args = parser.parse_args()

    if not any([args.all, args.sentiment, args.technical, args.fundamental]):
        args.all = True

    if args.all or args.sentiment:
        prepare_sentiment_data(args.output_dir)
    if args.all or args.technical:
        prepare_technical_data(args.output_dir)
    if args.all or args.fundamental:
        prepare_fundamental_data(args.output_dir)

    print("\n✅ Data preparation complete!")
    print(f"   Output: {args.output_dir}/")
    print("   Structure:")
    print("     sentiment/train.json + val.json + test.json")
    print("     technical/train.json + val.json + test.json")
    print("     fundamental/train.json + val.json + test.json")


if __name__ == "__main__":
    main()
