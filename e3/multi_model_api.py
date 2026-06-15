#!/usr/bin/env python3
"""
multi_model_api.py — Unified 3-model API wrapper for Delta V2

Models:
  Agent 1 (Sentiment):   qwen-plus         (Qwen2.5-based, 72B+)
  Agent 2 (Technical):   deepseek-v4-flash  (DeepSeek V4, free flash tier)
  Agent 3 (Fundamental): qwen3.5-flash      (Qwen3.5, newest flash)

All via DashScope OpenAI-compatible API.

Author: Siyi / 2026-06-13
"""

import os
import json
import time
import re
from openai import OpenAI
from typing import Optional

# ── Configuration ──

API_KEY = os.environ.get("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODELS = {
    "sentiment":   "qwen-plus",
    "technical":   "deepseek-v4-flash",
    "fundamental": "qwen-turbo",
}

RATE_LIMITS = {
    "qwen-plus":         0.20,   # seconds between calls
    "deepseek-v4-flash": 0.20,
    "qwen-turbo":        0.10,
}

# ── Prompt Templates ──

SENTIMENT_PROMPT = """你是一位资深市场情绪分析师。请对以下股票给出1-10的情绪评分。

股票：{ticker}
时间窗口：{start_date} 至 {end_date}

评分标准：
  1-3分：整体负面情绪占主导（恐慌、抛售、悲观）
  4-6分：中性或混合情绪
  7-10分：整体正面情绪占主导（乐观、买入、信心）

严格按以下JSON格式输出，不要加任何前缀或说明：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

TECHNICAL_PROMPT = """你是一位技术分析专家。请对以下股票给出1-10的技术评分。

股票：{ticker}

评分标准：
  1-3分：明显下跌趋势（空头排列，RSI超卖但无反弹）
  4-6分：震荡或横盘（无明显趋势）
  7-10分：明显上涨趋势（多头排列，量价配合）

历史数据：
  1个月收益：{ret_1m}%
  3个月收益：{ret_3m}%
  6个月收益：{ret_6m}%
  成交量比：{vol_ratio}
  波动率：{volatility}
  最大回撤：{max_dd}%

严格按以下JSON格式输出：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

FUNDAMENTAL_PROMPT = """你是一位基本面分析师。请对以下股票给出1-10的基本面评分。

股票：{ticker}

评分标准：
  1-3分：基本面恶化（盈利下降，负债上升，行业前景差）
  4-6分：基本面平稳
  7-10分：基本面优秀（盈利增长，现金流健康，行业领先）

市场数据：
  6个月动量：{ret_6m}%
  波动率：{volatility}
  成交量比：{vol_ratio}

严格按以下JSON格式输出：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

# ── Anonymized versions (for familiarity bias test) ──

SENTIMENT_PROMPT_ANON = """你是一位资深市场情绪分析师。请对以下股票给出1-10的情绪评分。

股票：Stock_{ticker_hash}（匿名）
时间窗口：{start_date} 至 {end_date}

评分标准：
  1-3分：整体负面情绪占主导（恐慌、抛售、悲观）
  4-6分：中性或混合情绪
  7-10分：整体正面情绪占主导（乐观、买入、信心）

严格按以下JSON格式输出，不要加任何前缀或说明：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

TECHNICAL_PROMPT_ANON = """你是一位技术分析专家。请对以下股票给出1-10的技术评分。

股票：Stock_{ticker_hash}（匿名，仅提供数值数据）

评分标准：
  1-3分：明显下跌趋势（空头排列，RSI超卖但无反弹）
  4-6分：震荡或横盘（无明显趋势）
  7-10分：明显上涨趋势（多头排列，量价配合）

历史数据：
  1个月收益：{ret_1m}%
  3个月收益：{ret_3m}%
  6个月收益：{ret_6m}%
  成交量比：{vol_ratio}
  波动率：{volatility}
  最大回撤：{max_dd}%

严格按以下JSON格式输出：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

FUNDAMENTAL_PROMPT_ANON = """你是一位基本面分析师。请对以下股票给出1-10的基本面评分。

股票：Stock_{ticker_hash}（匿名，仅提供数值数据）

评分标准：
  1-3分：基本面恶化（盈利下降，负债上升，行业前景差）
  4-6分：基本面平稳
  7-10分：基本面优秀（盈利增长，现金流健康，行业领先）

市场数据：
  6个月动量：{ret_6m}%
  波动率：{volatility}
  成交量比：{vol_ratio}

严格按以下JSON格式输出：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""


# ── API Client ──

def get_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    clean = text.strip()
    # Remove markdown code blocks
    clean = re.sub(r'^```(?:json)?\s*', '', clean)
    clean = re.sub(r'\s*```$', '', clean)
    clean = clean.strip()
    
    # Try direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r'\{[^}]+\}', clean)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    
    # Fallback: try to extract score number
    score_match = re.search(r'"?score"?\s*:\s*(\d+)', clean)
    if score_match:
        return {"score": int(score_match.group(1)), "reason": "parsed from non-JSON response"}
    
    return {"score": None, "reason": f"parse failed: {clean[:80]}"}


def call_model(
    agent: str,
    ticker: str,
    features: dict,
    anonymized: bool = False,
    max_retries: int = 3,
) -> dict:
    """
    Call the appropriate model for the given agent role.
    
    Args:
        agent: "sentiment", "technical", or "fundamental"
        ticker: stock ticker symbol
        features: dict with keys like ret_1m, ret_3m, ret_6m, vol_ratio, volatility, max_dd, etc.
        anonymized: if True, use anonymized prompts (no ticker name)
        max_retries: number of retries on failure
    
    Returns:
        dict with "score" (int 1-10) and "reason" (str)
    """
    model_name = MODELS[agent]
    client = get_client()
    
    # Select prompt template
    if anonymized:
        templates = {
            "sentiment": SENTIMENT_PROMPT_ANON,
            "technical": TECHNICAL_PROMPT_ANON,
            "fundamental": FUNDAMENTAL_PROMPT_ANON,
        }
    else:
        templates = {
            "sentiment": SENTIMENT_PROMPT,
            "technical": TECHNICAL_PROMPT,
            "fundamental": FUNDAMENTAL_PROMPT,
        }
    
    template = templates[agent]
    
    # Build prompt kwargs
    kwargs = {
        "ticker": ticker,
        "start_date": features.get("start_date", "2024-01"),
        "end_date": features.get("end_date", "2024-04"),
        "ret_1m": features.get("ret_1m", 0),
        "ret_3m": features.get("ret_3m", 0),
        "ret_6m": features.get("ret_6m", 0),
        "vol_ratio": features.get("vol_ratio", 1.0),
        "volatility": features.get("volatility", 0.2),
        "max_dd": features.get("max_dd", -10),
    }
    
    if anonymized:
        # Hash the ticker for anonymization
        kwargs["ticker_hash"] = format(abs(hash(ticker)) % 10000, "04d")
    
    prompt = template.format(**kwargs)
    
    # Call with retries
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            text = resp.choices[0].message.content
            result = parse_json_response(text)
            
            # Validate score
            if result.get("score") is not None:
                score = int(result["score"])
                if 1 <= score <= 10:
                    result["score"] = score
                    result["model"] = model_name
                    result["agent"] = agent
                    result["ticker"] = ticker
                    result["anonymized"] = anonymized
                    return result
            
            # Score out of range, retry
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            result = {"score": None, "reason": f"API error: {str(e)[:80]}"}
        
    return result


# ── Batch scoring ──

def batch_score(
    tasks: list[dict],
    checkpoint_path: str = None,
    checkpoint_every: int = 100,
    anonymized: bool = False,
) -> dict:
    """
    Batch score a list of tasks with checkpoint/resume.
    
    Each task: {"ticker": str, "agent": str, "month": str, "features": dict}
    Returns: dict mapping "ticker_month_agent" -> {"score": int, "reason": str, ...}
    """
    # Load checkpoint
    completed = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            completed = json.load(f)
        print(f"Resumed from checkpoint: {len(completed)} completed")
    
    total = len(tasks)
    done = len(completed)
    errors = 0
    start_time = time.time()
    
    for i, task in enumerate(tasks):
        key = f"{task['ticker']}_{task['month']}_{task['agent']}"
        
        if key in completed:
            continue
        
        # Rate limit
        model_name = MODELS[task["agent"]]
        time.sleep(RATE_LIMITS.get(model_name, 0.15))
        
        result = call_model(
            agent=task["agent"],
            ticker=task["ticker"],
            features=task["features"],
            anonymized=anonymized,
        )
        
        if result.get("score") is not None:
            completed[key] = result
            done += 1
        else:
            errors += 1
        
        # Progress
        if (done + errors) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (done + errors) / elapsed if elapsed > 0 else 0
            eta = (total - done - errors) / rate / 60 if rate > 0 else 0
            print(f"  Progress: {done+errors}/{total} (done={done}, err={errors}), "
                  f"rate={rate:.1f}/s, ETA={eta:.0f}min")
        
        # Checkpoint
        if checkpoint_path and (done + errors) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, ensure_ascii=False)
    
    # Final save
    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump(completed, f, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\nBatch complete: {done} done, {errors} errors, {elapsed/60:.1f}min")
    return completed


if __name__ == "__main__":
    # Quick test
    print("=== Multi-Model API Test ===\n")
    
    test_features = {
        "start_date": "2024-01",
        "end_date": "2024-04",
        "ret_1m": 2.3,
        "ret_3m": 5.1,
        "ret_6m": 8.7,
        "vol_ratio": 1.12,
        "volatility": 0.18,
        "max_dd": -12.5,
    }
    
    for agent in ["sentiment", "technical", "fundamental"]:
        result = call_model(agent, "AAPL", test_features)
        print(f"{agent:14s} ({MODELS[agent]}): score={result.get('score')}, "
              f"reason=\"{result.get('reason','')[:50]}\"")
        time.sleep(0.3)
    
    print("\n--- Anonymized test ---")
    for agent in ["sentiment", "technical", "fundamental"]:
        result = call_model(agent, "AAPL", test_features, anonymized=True)
        print(f"{agent:14s} ({MODELS[agent]}): score={result.get('score')}, "
              f"reason=\"{result.get('reason','')[:50]}\"")
        time.sleep(0.3)
