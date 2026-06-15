#!/usr/bin/env python3
"""
feature_engine.py — Agent-specific feature computation for Delta V2

Three feature pipelines:
  1. Sentiment: price momentum + volume patterns + volatility regime
  2. Technical: daily indicator computation (RSI, MACD, BB, ADX, etc.)
  3. Fundamental: financial ratios from yfinance

All features are computed as of quarter-end, NO look-ahead.

Author: Siyi / 2026-06-13
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

OUT = "/home/z/my-project/delta_jfe_v2"
DATA_DIR = "/home/z/my-project/delta_jfe"


# ═══════════════════════════════════════════════════════════
# Agent 1: Sentiment Features (from monthly data)
# ═══════════════════════════════════════════════════════════

def compute_sentiment_features(monthly_data: dict, month: str) -> Optional[dict]:
    """
    Compute sentiment agent features from monthly price data.
    
    Args:
        monthly_data: {month_str: {"close": float, "return": float, "volume": float}}
        month: target month (e.g., "2024-03")
    
    Returns:
        dict of features or None if insufficient data
    """
    months_sorted = sorted(monthly_data.keys())
    if month not in monthly_data:
        return None
    
    idx = months_sorted.index(month)
    if idx < 11:  # need at least 12 months of history
        return None
    
    # ── Returns (multi-horizon) ──
    ret_1m = monthly_data[months_sorted[idx]]["return"] * 100
    ret_3m = (np.prod([1 + monthly_data[months_sorted[idx-i]]["return"] for i in range(3)]) - 1) * 100
    ret_6m = (np.prod([1 + monthly_data[months_sorted[idx-i]]["return"] for i in range(6)]) - 1) * 100
    ret_12m = (np.prod([1 + monthly_data[months_sorted[idx-i]]["return"] for i in range(min(12, idx+1))]) - 1) * 100
    
    # ── Volume ratios ──
    vol_current = monthly_data[months_sorted[idx]].get("volume", 0)
    vol_avg_3m = np.mean([monthly_data[months_sorted[idx-i]].get("volume", vol_current) for i in range(3)])
    vol_avg_12m = np.mean([monthly_data[months_sorted[idx-i]].get("volume", vol_current) for i in range(min(12, idx+1))])
    vol_ratio_1m = vol_current / vol_avg_3m if vol_avg_3m > 0 else 1.0
    vol_ratio_3m = vol_avg_3m / vol_avg_12m if vol_avg_12m > 0 else 1.0
    
    # ── Volatility ──
    rets_3m = [monthly_data[months_sorted[idx-i]]["return"] for i in range(3)]
    rets_6m = [monthly_data[months_sorted[idx-i]]["return"] for i in range(6)]
    vol_1m = abs(ret_1m / 100)  # simplified
    vol_3m = np.std(rets_3m) if len(rets_3m) >= 3 else 0.02
    vol_6m = np.std(rets_6m) if len(rets_6m) >= 6 else 0.02
    
    # ── Drawdown ──
    prices_3m = [100]
    prices_6m = [100]
    for i in range(min(3, idx+1)):
        prices_3m.append(prices_3m[-1] * (1 + monthly_data[months_sorted[idx-i]]["return"]))
    for i in range(min(6, idx+1)):
        prices_6m.append(prices_6m[-1] * (1 + monthly_data[months_sorted[idx-i]]["return"]))
    
    def max_dd(prices):
        peak = prices[0]
        dd = 0
        for p in prices:
            peak = max(peak, p)
            dd = min(dd, (p - peak) / peak)
        return dd * 100
    
    dd_3m = max_dd(prices_3m)
    dd_6m = max_dd(prices_6m)
    
    # ── Up days ratio (approximated from monthly returns) ──
    up_count = sum(1 for i in range(min(3, idx+1)) if monthly_data[months_sorted[idx-i]]["return"] > 0)
    up_ratio = up_count / min(3, idx+1)
    
    # ── Momentum rank (cross-sectional placeholder) ──
    # Will be filled in batch computation
    
    return {
        "ret_1m": round(ret_1m, 2),
        "ret_3m": round(ret_3m, 2),
        "ret_6m": round(ret_6m, 2),
        "ret_12m": round(ret_12m, 2),
        "vol_ratio_1m": round(vol_ratio_1m, 2),
        "vol_ratio_3m": round(vol_ratio_3m, 2),
        "vol_3m": round(vol_3m, 4),
        "vol_6m": round(vol_6m, 4),
        "max_dd_3m": round(dd_3m, 2),
        "max_dd_6m": round(dd_6m, 2),
        "up_ratio": round(up_ratio, 2),
    }


# ═══════════════════════════════════════════════════════════
# Agent 2: Technical Features (from daily data)
# ═══════════════════════════════════════════════════════════

def compute_technical_features(daily_df: pd.DataFrame, as_of_date: str) -> Optional[dict]:
    """
    Compute technical indicators from daily OHLCV data as of a given date.
    NO look-ahead: only uses data up to as_of_date.
    
    Args:
        daily_df: DataFrame with columns [Open, High, Low, Close, Volume]
        as_of_date: YYYY-MM-DD format
    
    Returns:
        dict of technical features or None
    """
    # Filter to data up to as_of_date
    df = daily_df[daily_df.index <= as_of_date].copy()
    if len(df) < 60:  # need at least 60 trading days
        return None
    
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values
    
    # ── RSI(14) ──
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    period = min(14, len(gains))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # ── MACD ──
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    # Signal line (9-period EMA of MACD)
    macd_hist = macd_line[-1]  # simplified: just use current MACD value
    macd_signal = "bullish" if macd_line[-1] > 0 else "bearish"
    
    # ── Bollinger Band %B ──
    sma20 = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    if std20 > 0:
        bb_pct_b = (close[-1] - (sma20 - 2*std20)) / (4 * std20)
    else:
        bb_pct_b = 0.5
    
    # ── ADX(14) ──
    adx = _compute_adx(high, low, close, period=14)
    
    # ── ATR(14) / Close ──
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                              np.abs(low[1:] - close[:-1])))
    atr14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
    atr_pct = atr14 / close[-1] if close[-1] > 0 else 0
    
    # ── Moving average cross ──
    sma50 = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
    sma200 = np.mean(close[-min(200, len(close)):])
    ma_cross = "golden_cross" if sma50 > sma200 else "death_cross"
    
    # ── Volume trend (20d) ──
    vol_20 = np.mean(volume[-20:])
    vol_60 = np.mean(volume[-min(60, len(volume)):])
    vol_trend = "expanding" if vol_20 > vol_60 * 1.1 else ("contracting" if vol_20 < vol_60 * 0.9 else "neutral")
    
    # ── OBV trend ──
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # OBV trend: compare recent 20d vs prior 20d
    if len(obv) >= 40:
        obv_recent = np.mean(obv[-20:])
        obv_prior = np.mean(obv[-40:-20])
        obv_trend = "rising" if obv_recent > obv_prior else "falling"
    else:
        obv_trend = "neutral"
    
    # ── Distribution shape (20d returns) ──
    rets_20d = np.diff(close[-21:]) / close[-21:-1]
    skewness = float(pd.Series(rets_20d).skew()) if len(rets_20d) >= 10 else 0
    kurtosis = float(pd.Series(rets_20d).kurtosis()) if len(rets_20d) >= 10 else 0
    
    # ── Consecutive up/down days ──
    direction = np.sign(np.diff(close[-21:]))
    max_streak = 1
    current_streak = 1
    for i in range(1, len(direction)):
        if direction[i] == direction[i-1] and direction[i] != 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    
    return {
        "RSI_14": round(rsi, 1),
        "MACD_signal": macd_signal,
        "MACD_value": round(float(macd_line[-1]), 2),
        "BB_pct_B": round(bb_pct_b, 3),
        "ADX_14": round(adx, 1),
        "ATR_pct": round(atr_pct, 4),
        "MA_cross": ma_cross,
        "volume_trend": vol_trend,
        "OBV_trend": obv_trend,
        "skewness_20d": round(skewness, 2),
        "kurtosis_20d": round(kurtosis, 2),
        "max_streak": int(max_streak),
    }


# ── Technical helper functions ──

def _ema(data, period):
    """Exponential moving average."""
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * multiplier + ema[i-1] * (1 - multiplier)
    return ema


def _compute_adx(high, low, close, period=14):
    """Average Directional Index."""
    if len(high) < period + 1:
        return 25.0  # neutral default
    
    # True Range
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                              np.abs(low[1:] - close[:-1])))
    
    # +DM and -DM
    plus_dm = np.maximum(high[1:] - high[:-1], 0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0)
    
    # Where -DM > +DM, set +DM to 0 and vice versa
    plus_dm[minus_dm > plus_dm] = 0
    minus_dm[plus_dm > minus_dm] = 0
    
    # Smooth
    atr = np.zeros(len(tr))
    plus_di_smooth = np.zeros(len(plus_dm))
    minus_di_smooth = np.zeros(len(minus_dm))
    
    atr[period] = np.sum(tr[:period])
    plus_di_smooth[period] = np.sum(plus_dm[:period])
    minus_di_smooth[period] = np.sum(minus_dm[:period])
    
    for i in range(period + 1, len(tr)):
        atr[i] = atr[i-1] - atr[i-1]/period + tr[i]
        plus_di_smooth[i] = plus_di_smooth[i-1] - plus_di_smooth[i-1]/period + plus_dm[i]
        minus_di_smooth[i] = minus_di_smooth[i-1] - minus_di_smooth[i-1]/period + minus_dm[i]
    
    # DI
    plus_di = 100 * plus_di_smooth / np.maximum(atr, 1e-10)
    minus_di = 100 * minus_di_smooth / np.maximum(atr, 1e-10)
    
    # DX
    di_sum = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(di_sum, 1e-10)
    
    # ADX (smoothed DX)
    if len(dx) >= 2 * period:
        adx = np.mean(dx[-period:])
    else:
        adx = np.mean(dx[period:]) if len(dx) > period else 25.0
    
    return float(adx)


# ═══════════════════════════════════════════════════════════
# Agent 3: Fundamental Features (from yfinance)
# ═══════════════════════════════════════════════════════════

def compute_fundamental_features(ticker: str, as_of_date: str = None) -> Optional[dict]:
    """
    Compute fundamental features from yfinance.
    
    NOTE: yfinance .info only returns current snapshot.
    For historical analysis, we use .financials + .balance_sheet + price
    to reconstruct historical ratios.
    
    For now, returns current snapshot as a starting point.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got meaningful data
        if not info or info.get("regularMarketPrice") is None:
            return None
        
        features = {}
        
        # Valuation
        features["trailingPE"] = info.get("trailingPE")
        features["forwardPE"] = info.get("forwardPE")
        features["priceToBook"] = info.get("priceToBook")
        features["pegRatio"] = info.get("pegRatio")
        features["enterpriseToEbitda"] = info.get("enterpriseToEbitda")
        features["dividendYield"] = info.get("dividendYield")
        
        # Profitability
        features["returnOnEquity"] = info.get("returnOnEquity")
        features["profitMargins"] = info.get("profitMargins")
        
        # Financial health
        features["debtToEquity"] = info.get("debtToEquity")
        features["currentRatio"] = info.get("currentRatio")
        
        # Growth
        features["earningsGrowth"] = info.get("earningsGrowth")
        features["revenueGrowth"] = info.get("revenueGrowth")
        
        # Risk
        features["beta"] = info.get("beta")
        features["marketCap"] = info.get("marketCap")
        
        # Remove None values
        features = {k: v for k, v in features.items() if v is not None}
        
        if len(features) < 5:  # insufficient data
            return None
        
        return features
        
    except Exception as e:
        return None


def compute_historical_fundamentals(ticker: str, quarters: list[str]) -> dict:
    """
    Compute fundamental features for multiple quarters.
    Uses yfinance financials + balance_sheet to reconstruct historical ratios.
    
    Args:
        ticker: stock symbol
        quarters: list of quarter-end dates ["2024-03", "2024-06", ...]
    
    Returns:
        {quarter_str: {feature: value}}
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    result = {}
    
    try:
        # Get financial statements
        income = stock.quarterly_financials
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        
        if income.empty or balance.empty:
            return result
        
        # Get price history for market cap calculation
        hist = stock.history(period="5y")
        
        for quarter in quarters:
            # Map quarter to approximate date
            q_end = quarter + "-30" if quarter.endswith("03") else \
                    quarter + "-30" if quarter.endswith("06") else \
                    quarter + "-30" if quarter.endswith("09") else \
                    quarter + "-31"
            
            # Find closest financial data before this date
            try:
                q_date = pd.Timestamp(q_end)
                
                # Get price at quarter end
                price_mask = hist.index <= q_date
                if not price_mask.any():
                    continue
                price_at_q = hist.loc[price_mask, "Close"].iloc[-1]
                
                # Get most recent financial data before this date
                fin_mask = income.columns <= q_date
                if not fin_mask.any():
                    continue
                
                fin_col = income.columns[fin_mask][0]  # most recent quarter
                bal_col = balance.columns[balance.columns <= q_date][0] if (balance.columns <= q_date).any() else None
                
                if bal_col is None:
                    continue
                
                # Extract key figures
                net_income = income.loc["Net Income", fin_col] if "Net Income" in income.index else None
                revenue = income.loc["Total Revenue", fin_col] if "Total Revenue" in income.index else None
                
                total_equity = balance.loc["Stockholders Equity", bal_col] if "Stockholders Equity" in balance.index else None
                total_debt = balance.loc["Total Debt", bal_col] if "Total Debt" in balance.index else None
                current_assets = balance.loc["Current Assets", bal_col] if "Current Assets" in balance.index else None
                current_liab = balance.loc["Current Liabilities", bal_col] if "Current Liabilities" in balance.index else None
                
                shares = balance.loc["Common Stock", bal_col] if "Common Stock" in balance.index else None
                
                # Compute ratios
                features = {}
                
                if total_equity and total_equity != 0 and shares and shares != 0:
                    market_cap = price_at_q * abs(shares)  # approximate
                    features["priceToBook"] = round(market_cap / abs(total_equity), 2)
                
                if revenue and revenue != 0 and net_income is not None:
                    features["profitMargins"] = round(float(net_income / revenue), 4)
                
                if total_equity and total_equity != 0 and net_income is not None:
                    features["returnOnEquity"] = round(float(net_income / total_equity), 4)
                
                if total_debt is not None and total_equity and total_equity != 0:
                    features["debtToEquity"] = round(float(total_debt / total_equity), 2)
                
                if current_assets and current_liab and current_liab != 0:
                    features["currentRatio"] = round(float(current_assets / current_liab), 2)
                
                if shares and shares != 0:
                    features["marketCap"] = round(float(price_at_q * abs(shares)), 0)
                
                if len(features) >= 3:
                    result[quarter] = features
                    
            except Exception:
                continue
        
    except Exception as e:
        pass
    
    return result


# ═══════════════════════════════════════════════════════════
# Daily data download
# ═══════════════════════════════════════════════════════════

def download_daily_data(ticker: str, start="2004-01-01", end="2024-12-31") -> Optional[pd.DataFrame]:
    """Download daily OHLCV data from yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # Flatten multi-level columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception:
        return None


def batch_download_daily(tickers: list[str], save_dir: str = None, batch_size: int = 20) -> dict:
    """Download daily data for multiple tickers with caching."""
    if save_dir is None:
        save_dir = os.path.join(OUT, "daily_data")
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        cache_path = os.path.join(save_dir, f"{ticker}.parquet")
        
        # Check cache
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                results[ticker] = df
                continue
            except Exception:
                pass
        
        # Download
        df = download_daily_data(ticker)
        if df is not None and len(df) > 100:
            results[ticker] = df
            # Cache
            try:
                df.to_parquet(cache_path)
            except Exception:
                pass
        else:
            failed.append(ticker)
        
        # Rate limit
        if (i + 1) % batch_size == 0:
            print(f"  Downloaded {i+1}/{len(tickers)} ({len(results)} success, {len(failed)} failed)")
            time.sleep(1)  # yfinance rate limit
        else:
            time.sleep(0.1)
    
    print(f"\nDaily data: {len(results)} stocks downloaded, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed[:20]}")
    
    return results


# ═══════════════════════════════════════════════════════════
# Feature compilation for all stocks × quarters
# ═══════════════════════════════════════════════════════════

def compile_all_features(
    stock_monthly: dict,
    daily_data: dict = None,
    quarters: list[str] = None,
    include_fundamental: bool = False,
) -> dict:
    """
    Compile agent-specific features for all stocks and quarters.
    
    Returns:
        {ticker: {quarter: {"sentiment": {...}, "technical": {...}, "fundamental": {...}}}}
    """
    if quarters is None:
        # Default: quarterly from 2005-Q1 to 2024-Q4
        quarters = []
        for year in range(2005, 2025):
            for q_end in ["03", "06", "09", "12"]:
                quarters.append(f"{year}-{q_end}")
    
    all_features = {}
    
    for ticker in sorted(stock_monthly.keys()):
        monthly = stock_monthly[ticker]
        all_features[ticker] = {}
        
        for quarter in quarters:
            feat = {}
            
            # Sentiment features (from monthly data)
            sent = compute_sentiment_features(monthly, quarter)
            if sent:
                feat["sentiment"] = sent
            
            # Technical features (from daily data)
            if daily_data and ticker in daily_data:
                # Convert quarter to approximate end date
                q_date = f"{quarter}-28"
                tech = compute_technical_features(daily_data[ticker], q_date)
                if tech:
                    feat["technical"] = tech
            
            # Fundamental features
            if include_fundamental:
                fund = compute_fundamental_features(ticker)
                if fund:
                    feat["fundamental"] = fund
            
            if feat:
                all_features[ticker][quarter] = feat
    
    return all_features


if __name__ == "__main__":
    print("=== Feature Engine Test ===\n")
    
    # Load monthly data
    with open(os.path.join(DATA_DIR, "sp500_monthly_returns.json")) as f:
        d = json.load(f)
    stock_monthly = d["data"]
    
    # Test sentiment features
    print("--- Sentiment Features ---")
    for ticker in ["AAPL", "PFE", "XOM"]:
        feat = compute_sentiment_features(stock_monthly[ticker], "2024-03")
        if feat:
            print(f"  {ticker}: {feat}")
    
    # Test daily data download (just 3 stocks)
    print("\n--- Downloading Daily Data (3 stocks) ---")
    daily = batch_download_daily(["AAPL", "PFE", "XOM"])
    
    # Test technical features
    print("\n--- Technical Features ---")
    for ticker in ["AAPL", "PFE", "XOM"]:
        if ticker in daily:
            tech = compute_technical_features(daily[ticker], "2024-03-28")
            if tech:
                print(f"  {ticker}: {tech}")
    
    # Test fundamental features
    print("\n--- Fundamental Features ---")
    for ticker in ["AAPL", "PFE", "XOM"]:
        fund = compute_fundamental_features(ticker)
        if fund:
            print(f"  {ticker}: {fund}")
