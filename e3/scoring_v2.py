#!/usr/bin/env python3
"""
scoring_v2.py — Probability-based scoring for Delta V2 (E2/E3)

E2: Same model (qwen-plus), same features, probability output
E3: Same model (qwen-plus), agent-specific features, probability output

Each agent outputs a probability distribution:
  Sentiment:    {bearish, neutral, bullish}
  Technical:    {breakdown, range, breakout} + trend_strength
  Fundamental:  {undervalued, fair, overvalued}

Author: Siyi / 2026-06-13
"""

import os
import json
import time
import re
import numpy as np
from openai import OpenAI
from typing import Optional
from scipy.stats import entropy as scipy_entropy

sys_path = "/home/z/my-project/delta_jfe_v2"
import sys
sys.path.insert(0, sys_path)

from feature_engine import compute_sentiment_features, compute_technical_features, compute_fundamental_features

OUT = "/home/z/my-project/delta_jfe_v2"
DATA_DIR = "/home/z/my-project/delta_jfe"

API_KEY = os.environ.get("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ═══════════════════════════════════════════════════════════
# Prompt Templates — Probability Output
# ═══════════════════════════════════════════════════════════

SENTIMENT_PROMPT = """你是一位资深市场情绪分析师。基于以下市场行为数据，判断该股票未来1个月的市场情绪方向。

{ticker_line}
近1个月收益: {ret_1m:+.1f}%
近3个月收益: {ret_3m:+.1f}%
近6个月收益: {ret_6m:+.1f}%
近12个月收益: {ret_12m:+.1f}%
成交量比(1月/3月均): {vol_ratio_1m:.2f}
成交量比(3月/12月均): {vol_ratio_3m:.2f}
3个月波动率: {vol_3m:.4f}
6个月波动率: {vol_6m:.4f}
3个月最大回撤: {max_dd_3m:+.1f}%
6个月最大回撤: {max_dd_6m:+.1f}%
近期上涨月份占比: {up_ratio:.0%}

请给出三个概率（0-100的整数，之和=100）：
- bearish: 看跌概率（市场悲观、恐慌抛售）
- neutral: 中性概率（观望、无明显方向）
- bullish: 看涨概率（市场乐观、资金流入）

重要：请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按以下JSON格式输出，不要加任何前缀或说明：
{{"bearish": <整数>, "neutral": <整数>, "bullish": <整数>}}"""

TECHNICAL_PROMPT = """你是一位技术分析师。基于以下技术指标，评估该股票的技术形态和趋势方向。

{ticker_line}
RSI(14): {RSI_14:.1f}  （<30超卖，>70超买）
MACD: {MACD_value:+.2f} （{MACD_signal}）
布林带%B: {BB_pct_B:.3f} （<0下轨外，>1上轨外，0.5中轨）
ADX(14): {ADX_14:.1f} （<20无趋势，>25有趋势，>50强趋势）
ATR/价格: {ATR_pct:.4f}
均线交叉: {MA_cross} （golden_cross=多头，death_cross=空头）
成交量趋势: {volume_trend}
OBV趋势: {OBV_trend}
20日偏度: {skewness_20d:+.2f}
20日峰度: {kurtosis_20d:+.2f}
最大连续同向天数: {max_streak}

请评估：
1. 趋势方向概率（0-100的整数，之和=100）：
   - breakdown: 向下突破概率（跌破支撑位）
   - range: 横盘震荡概率（区间整理）
   - breakout: 向上突破概率（突破阻力位）

2. 趋势强度（1-10整数）：10=极强趋势，1=无趋势

重要：概率请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按以下JSON格式输出：
{{"breakdown": <整数>, "range": <整数>, "breakout": <整数>, "trend_strength": <整数>}}"""

FUNDAMENTAL_PROMPT = """你是一位基本面分析师。基于以下财务数据，评估该股票的估值水平。

{ticker_line}
{fundamental_data}

请评估估值概率（0-100的整数，之和=100%）：
- undervalued: 低估概率（价格低于内在价值）
- fair: 合理估值概率
- overvalued: 高估概率（价格高于内在价值）

重要：概率请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按JSON格式输出：
{{"undervalued": <整数>, "fair": <整数>, "overvalued": <整数>}}"""

# Simplified prompts for E2 (same features, just probability output)
SENTIMENT_PROMPT_E2 = """你是一位资深市场情绪分析师。基于以下数据，判断该股票未来1个月的市场情绪方向。

{ticker_line}
1个月收益: {ret_1m:+.1f}%, 3个月收益: {ret_3m:+.1f}%, 6个月收益: {ret_6m:+.1f}%
成交量比(1月): {vol_ratio_1m:.2f}, 3个月波动率: {vol_3m:.4f}, 最大回撤: {max_dd_3m:+.1f}%

请给出三个概率（0-100的整数，之和=100）：
- bearish: 看跌概率
- neutral: 中性概率
- bullish: 看涨概率

重要：请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按JSON格式输出：{{"bearish": <整数>, "neutral": <整数>, "bullish": <整数>}}"""


# ═══════════════════════════════════════════════════════════
# API Call Functions
# ═══════════════════════════════════════════════════════════

def call_probability_model(agent: str, ticker: str, features: dict, 
                           anonymized: bool = False, model: str = None) -> dict:
    """
    Call LLM to get probability distribution for an agent.
    
    Returns:
        {"probs": {...}, "H": float, "direction": str, "model": str}
        or {"error": str}
    """
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    if model is None:
        model = "qwen-plus"  # Default for E2/E3
    
    ticker_line = f"股票代码：Stock_{hash(ticker) % 10000:04d}" if anonymized else f"股票：{ticker}"
    
    # Convert numpy types to plain Python for format strings
    clean_features = {}
    for k, v in features.items():
        if isinstance(v, np.floating):
            clean_features[k] = float(v)
        elif isinstance(v, np.integer):
            clean_features[k] = int(v)
        elif isinstance(v, np.ndarray):
            clean_features[k] = float(v.item()) if v.size == 1 else str(v)
        else:
            clean_features[k] = v
    
    # Format prompt based on agent and available features
    try:
        if agent == "sentiment":
            if "RSI_14" in clean_features:
                # E3: agent-specific features
                prompt = SENTIMENT_PROMPT.format(ticker_line=ticker_line, **clean_features)
            else:
                # E2: basic features
                prompt = SENTIMENT_PROMPT_E2.format(ticker_line=ticker_line, **clean_features)
        elif agent == "technical":
            prompt = TECHNICAL_PROMPT.format(ticker_line=ticker_line, **clean_features)
        elif agent == "fundamental":
            # Build fundamental_data string dynamically from available features
            label_map = {
                'trailingPE': '市盈率(TTM)', 'forwardPE': '远期PE',
                'priceToBook': '市净率', 'pegRatio': 'PEG比率',
                'enterpriseToEbitda': 'EV/EBITDA', 'dividendYield': '股息率',
                'returnOnEquity': 'ROE', 'profitMargins': '利润率',
                'debtToEquity': '负债权益比', 'currentRatio': '流动比率',
                'earningsGrowth': '盈利增长', 'revenueGrowth': '营收增长',
                'beta': 'Beta', 'marketCap': '市值',
            }
            lines = []
            for k, v in clean_features.items():
                label = label_map.get(k, k)
                if isinstance(v, float):
                    if abs(v) < 1 and k not in ['beta', 'currentRatio', 'dividendYield']:
                        lines.append(f"{label}: {v:.2%}")
                    elif abs(v) > 100:
                        lines.append(f"{label}: {v:.0f}")
                    else:
                        lines.append(f"{label}: {v:.1f}")
                else:
                    lines.append(f"{label}: {v}")
            fundamental_data = "\n".join(lines)
            prompt = FUNDAMENTAL_PROMPT.format(ticker_line=ticker_line, fundamental_data=fundamental_data)
        else:
            return {"error": f"Unknown agent: {agent}"}
    except KeyError as e:
        return {"error": f"Missing feature: {e}"}
    
    # Call API
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        
        # Parse JSON
        clean = raw.replace("```json", "").replace("```", "").strip()
        if "{" in clean:
            clean = clean[clean.index("{"):]
        if "}" in clean:
            clean = clean[:clean.rindex("}") + 1]
        
        probs = json.loads(clean)
        
        # Normalize probabilities to sum to 100
        prob_keys = [k for k in probs if k != "trend_strength"]
        total = sum(probs[k] for k in prob_keys)
        if total > 0 and total != 100:
            for k in prob_keys:
                probs[k] = round(probs[k] * 100 / total)
            # Adjust for rounding
            diff = 100 - sum(probs[k] for k in prob_keys)
            probs[prob_keys[0]] += diff
        
        # Compute entropy
        prob_vals = np.array([probs[k] for k in prob_keys]) / 100.0
        H = float(scipy_entropy(prob_vals, base=2))
        
        # Determine direction
        if "bullish" in probs:
            direction = "bullish" if probs["bullish"] > probs["bearish"] else \
                       "bearish" if probs["bearish"] > probs["bullish"] else "neutral"
        elif "breakout" in probs:
            direction = "breakout" if probs["breakout"] > probs["breakdown"] else \
                       "breakdown" if probs["breakdown"] > probs["breakout"] else "range"
        elif "overvalued" in probs:
            direction = "overvalued" if probs["overvalued"] > probs["undervalued"] else \
                       "undervalued" if probs["undervalued"] > probs["overvalued"] else "fair"
        else:
            direction = "unknown"
        
        result = {
            "probs": {k: probs[k] for k in prob_keys},
            "H": round(H, 4),
            "direction": direction,
            "model": model,
        }
        
        if "trend_strength" in probs:
            result["trend_strength"] = probs["trend_strength"]
        
        return result
        
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}


# ═══════════════════════════════════════════════════════════
# Batch Scoring with Checkpoint
# ═══════════════════════════════════════════════════════════

def batch_probability_score(tasks: list[dict], checkpoint_path: str,
                             checkpoint_every: int = 50,
                             anonymized: bool = False,
                             model: str = "qwen-plus") -> dict:
    """
    Batch score with checkpointing.
    
    Args:
        tasks: [{"ticker": str, "month": str, "agent": str, "features": dict}]
        checkpoint_path: path to save checkpoint JSON
        checkpoint_every: save every N calls
        anonymized: whether to anonymize ticker
        model: model name
    
    Returns:
        {(ticker, month, agent): {probs, H, direction}}
    """
    # Load checkpoint
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
            completed = ckpt.get("completed", {})
        print(f"Resuming from checkpoint: {len(completed)} completed")
    
    total = len(tasks)
    errors = 0
    t0 = time.time()
    
    for i, task in enumerate(tasks):
        key = f"{task['ticker']}_{task['month']}_{task['agent']}"
        
        if key in completed:
            continue
        
        result = call_probability_model(
            agent=task["agent"],
            ticker=task["ticker"],
            features=task["features"],
            anonymized=anonymized,
            model=model,
        )
        
        if "error" in result:
            errors += 1
            if errors <= 5:
                print(f"  Error on {key}: {result['error']}")
        else:
            completed[key] = result
        
        # Checkpoint
        if (len(completed) % checkpoint_every == 0 and len(completed) > 0) or \
           (i == total - 1):
            with open(checkpoint_path, "w") as f:
                json.dump({"completed": completed, "total": total, 
                          "errors": errors, "model": model}, f)
        
        # Rate limit
        time.sleep(0.15)
        
        # Progress
        if (i + 1) % 100 == 0:
            elapsed = (time.time() - t0) / 60
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {elapsed:.1f}min elapsed, {remaining:.1f}min remaining")
    
    elapsed = (time.time() - t0) / 60
    print(f"\nBatch complete: {len(completed)} done, {errors} errors, {elapsed:.1f}min")
    
    return completed


# ═══════════════════════════════════════════════════════════
# Metric Computation
# ═══════════════════════════════════════════════════════════

def compute_metrics(scores: dict) -> dict:
    """
    Compute the new metric system from probability scores.
    
    Args:
        scores: {(ticker, month, agent): {probs, H, direction, ...}}
    
    Returns:
        {(ticker, month): {H_sent, H_tech, H_fund, H_avg, Disagreement, ...}}
    """
    # Group by (ticker, month)
    from collections import defaultdict
    by_stock_month = defaultdict(dict)
    for key, val in scores.items():
        if "error" in val:
            continue
        parts = key.rsplit("_", 1)  # e.g., "AAPL_2024-03_sentiment"
        # More robust parsing
        parts = key.split("_")
        agent = parts[-1]
        month = parts[-2]
        ticker = "_".join(parts[:-2])
        by_stock_month[(ticker, month)][agent] = val
    
    metrics = {}
    for (ticker, month), agents in by_stock_month.items():
        m = {"ticker": ticker, "month": month}
        
        # Per-agent entropy
        H_vals = {}
        for agent in ["sentiment", "technical", "fundamental"]:
            if agent in agents:
                H_vals[agent] = agents[agent]["H"]
        
        if "sentiment" in H_vals:
            m["H_sent"] = H_vals["sentiment"]
        if "technical" in H_vals:
            m["H_tech"] = H_vals["technical"]
        if "fundamental" in H_vals:
            m["H_fund"] = H_vals["fundamental"]
        
        # Average entropy
        if H_vals:
            m["H_avg"] = round(np.mean(list(H_vals.values())), 4)
        
        # ── Disagreement: directional conflict ──
        directions = {}
        for agent in ["sentiment", "technical", "fundamental"]:
            if agent in agents:
                directions[agent] = agents[agent]["direction"]
        
        # Map to bullish/bearish binary
        bullish_agents = set()
        bearish_agents = set()
        
        dir_map = {
            "bullish": "bull", "breakout": "bull", "undervalued": "bull",
            "bearish": "bear", "breakdown": "bear", "overvalued": "bear",
            "neutral": "neutral", "range": "neutral", "fair": "neutral",
        }
        
        for agent, d in directions.items():
            mapped = dir_map.get(d, "neutral")
            if mapped == "bull":
                bullish_agents.add(agent)
            elif mapped == "bear":
                bearish_agents.add(agent)
        
        # Disagreement score: fraction of agents on minority side
        n_total = len(directions)
        n_bull = len(bullish_agents)
        n_bear = len(bearish_agents)
        
        if n_total > 0:
            minority = min(n_bull, n_bear)
            m["Disagreement"] = round(minority / n_total, 4) if n_bull != n_bear else 0.5
        else:
            m["Disagreement"] = 0
        
        # Full agreement
        m["Full_agreement"] = 1 if (n_bull == n_total or n_bear == n_total) else 0
        
        # Conflict type
        if bullish_agents and bearish_agents:
            # Which agents disagree?
            sent_dir = dir_map.get(directions.get("sentiment", ""), "neutral")
            tech_dir = dir_map.get(directions.get("technical", ""), "neutral")
            fund_dir = dir_map.get(directions.get("fundamental", ""), "neutral")
            
            if sent_dir != fund_dir and sent_dir != "neutral" and fund_dir != "neutral":
                m["Conflict_type"] = "sentiment_vs_fundamental"
            elif tech_dir != fund_dir and tech_dir != "neutral" and fund_dir != "neutral":
                m["Conflict_type"] = "technical_vs_fundamental"
            elif sent_dir != tech_dir and sent_dir != "neutral" and tech_dir != "neutral":
                m["Conflict_type"] = "sentiment_vs_technical"
            else:
                m["Conflict_type"] = "mixed"
        else:
            m["Conflict_type"] = "none"
        
        # ── Composite bullish probability ──
        bull_probs = []
        for agent in ["sentiment", "technical", "fundamental"]:
            if agent in agents and "probs" in agents[agent]:
                probs = agents[agent]["probs"]
                if "bullish" in probs:
                    bull_probs.append(probs["bullish"] / 100)
                elif "breakout" in probs:
                    bull_probs.append(probs["breakout"] / 100)
                elif "undervalued" in probs:
                    bull_probs.append(probs["undervalued"] / 100)
        
        if bull_probs:
            m["P_bull_avg"] = round(np.mean(bull_probs), 4)
            m["P_bull_std"] = round(np.std(bull_probs), 4)
        
        metrics[(ticker, month)] = m
    
    return metrics


if __name__ == "__main__":
    print("=== Scoring V2 Test ===\n")
    
    # Load monthly data
    with open(os.path.join(DATA_DIR, "sp500_monthly_returns.json")) as f:
        d = json.load(f)
    stock_monthly = d["data"]
    
    # Test with 3 stocks, one quarter
    test_tickers = ["AAPL", "PFE", "XOM"]
    test_month = "2024-03"
    
    for ticker in test_tickers:
        print(f"\n--- {ticker} ({test_month}) ---")
        
        # Sentiment features (from monthly data)
        sent_feat = compute_sentiment_features(stock_monthly[ticker], test_month)
        if sent_feat:
            r = call_probability_model("sentiment", ticker, sent_feat)
            print(f"  Sentiment: {r.get('probs', {})} H={r.get('H',0):.4f} dir={r.get('direction','')}")
            
            # Anonymized
            r_a = call_probability_model("sentiment", ticker, sent_feat, anonymized=True)
            print(f"  Sent(Anon): {r_a.get('probs', {})} H={r_a.get('H',0):.4f} dir={r_a.get('direction','')}")
        
        time.sleep(0.2)
    
    print("\n--- Metric System Test ---")
    # Quick test of metric computation
    test_scores = {}
    for ticker in test_tickers:
        sent_feat = compute_sentiment_features(stock_monthly[ticker], test_month)
        if sent_feat:
            for agent in ["sentiment"]:
                r = call_probability_model(agent, ticker, sent_feat)
                test_scores[f"{ticker}_{test_month}_{agent}"] = r
                time.sleep(0.15)
    
    metrics = compute_metrics(test_scores)
    for (t, m), v in metrics.items():
        print(f"  {t} {m}: {v}")
