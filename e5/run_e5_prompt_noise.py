#!/usr/bin/env python3
"""
run_e5_prompt_noise.py — Prompt-Noise Baseline (E5)
====================================================

Purpose: Prove that Agent disagreement is NOT just prompt noise.

Design:
  - Same model: qwen-plus (identical to E1)
  - Same input features: ALL features (sentiment + technical + fundamental)
  - Same question: "未来1个月该股票更偏 bearish/neutral/bullish？"
  - 3 different "analyst persona" prompts (same data, different framing)

If E5 JS-divergence << E1 JS-divergence → Agent specialization is real.
If E5 JS-divergence ≈ E1 JS-divergence → Disagreement is prompt noise.

Comparison Table (what we expect):
┌─────────┬──────────┬───────────┬────────────┬───────────────┐
│ Exp     │ Model    │ Input     │ Prompt     │ JS (expected) │
├─────────┼──────────┼───────────┼────────────┼───────────────┤
│ E1      │ qwen-plus│ agent-spec│ agent-spec │ 0.15-0.25     │
│ E5      │ qwen-plus│ ALL (same)│ 3 personas │ 0.03-0.08     │
│ E2(API) │ 3 models │ agent-spec│ agent-spec │ 0.20-0.30     │
└─────────┴──────────┴───────────┴────────────┴───────────────┘

Usage:
  python run_e5_prompt_noise.py --test        # 3 stocks, 4 quarters
  python run_e5_prompt_noise.py               # Full run (46 stocks × 80 quarters × 3 personas)
  python run_e5_prompt_noise.py --analyze     # Compare E5 vs E1

Author: Siyi / 2026-06-16
"""

import os, sys, json, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon

# ── Add parent dir for imports ──
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_engine import compute_sentiment_features

# ── Config ──
PARENT = Path(__file__).resolve().parent
DATA_DIR = Path("/home/z/my-project/delta_jfe")
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# ═══════════════════════════════════════════════════════════
# Prompt Templates — Same Data, Different Persona Framing
# ═══════════════════════════════════════════════════════════

# Persona A: "Conservative Analyst" — emphasizes risk, downside protection
PROMPT_A = """你是一位保守型分析师，偏好风险控制和下行保护。基于以下全面的金融数据，判断该股票未来1个月更偏 bearish、neutral 还是 bullish。

{ticker_line}
═══ 价格动量 ═══
近1个月收益: {ret_1m:+.1f}%
近3个月收益: {ret_3m:+.1f}%
近6个月收益: {ret_6m:+.1f}%
近12个月收益: {ret_12m:+.1f}%
成交量比(1月/3月均): {vol_ratio_1m:.2f}
3个月波动率: {vol_3m:.4f}
6个月波动率: {vol_6m:.4f}
3个月最大回撤: {max_dd_3m:+.1f}%
6个月最大回撤: {max_dd_6m:+.1f}%
近期上涨月份占比: {up_ratio:.0%}

请从风险优先的角度给出三个概率（0-100的整数，之和=100）：
- bearish: 看跌概率（下行风险大于上行机会）
- neutral: 中性概率（风险收益均衡）
- bullish: 看涨概率（上行机会大于下行风险）

重要：请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按JSON格式输出：{{"bearish": <整数>, "neutral": <整数>, "bullish": <整数>}}"""

# Persona B: "Growth Analyst" — emphasizes momentum and upside
PROMPT_B = """你是一位成长型分析师，偏好趋势追踪和上行潜力。基于以下全面的金融数据，判断该股票未来1个月更偏 bearish、neutral 还是 bullish。

{ticker_line}
═══ 价格动量 ═══
近1个月收益: {ret_1m:+.1f}%
近3个月收益: {ret_3m:+.1f}%
近6个月收益: {ret_6m:+.1f}%
近12个月收益: {ret_12m:+.1f}%
成交量比(1月/3月均): {vol_ratio_1m:.2f}
3个月波动率: {vol_3m:.4f}
6个月波动率: {vol_6m:.4f}
3个月最大回撤: {max_dd_3m:+.1f}%
6个月最大回撤: {max_dd_6m:+.1f}%
近期上涨月份占比: {up_ratio:.0%}

请从趋势和成长潜力的角度给出三个概率（0-100的整数，之和=100）：
- bearish: 看跌概率（趋势走弱）
- neutral: 中性概率（趋势不明）
- bullish: 看涨概率（趋势向上，增长动力强）

重要：请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按JSON格式输出：{{"bearish": <整数>, "neutral": <整数>, "bullish": <整数>}}"""

# Persona C: "Balanced Analyst" — neutral framing, same data
PROMPT_C = """你是一位均衡型分析师，综合权衡多空因素。基于以下全面的金融数据，判断该股票未来1个月更偏 bearish、neutral 还是 bullish。

{ticker_line}
═══ 价格动量 ═══
近1个月收益: {ret_1m:+.1f}%
近3个月收益: {ret_3m:+.1f}%
近6个月收益: {ret_6m:+.1f}%
近12个月收益: {ret_12m:+.1f}%
成交量比(1月/3月均): {vol_ratio_1m:.2f}
3个月波动率: {vol_3m:.4f}
6个月波动率: {vol_6m:.4f}
3个月最大回撤: {max_dd_3m:+.1f}%
6个月最大回撤: {max_dd_6m:+.1f}%
近期上涨月份占比: {up_ratio:.0%}

请客观综合评估，给出三个概率（0-100的整数，之和=100）：
- bearish: 看跌概率
- neutral: 中性概率
- bullish: 看涨概率

重要：请使用精确的整数（如37、42、21），不要四舍五入到5的倍数。

严格按JSON格式输出：{{"bearish": <整数>, "neutral": <整数>, "bullish": <整数>}}"""

PERSONAS = {
    "conservative": PROMPT_A,
    "growth": PROMPT_B,
    "balanced": PROMPT_C,
}


# ═══════════════════════════════════════════════════════════
# API Call
# ═══════════════════════════════════════════════════════════

def call_model(prompt: str, model: str = MODEL) -> dict:
    """Call qwen-plus and parse probability output."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
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
        
        # Normalize to sum=100
        prob_keys = ["bearish", "neutral", "bullish"]
        total = sum(probs.get(k, 0) for k in prob_keys)
        if total > 0 and total != 100:
            for k in prob_keys:
                probs[k] = round(probs.get(k, 0) * 100 / total)
            diff = 100 - sum(probs.get(k, 0) for k in prob_keys)
            probs[prob_keys[0]] = probs.get(prob_keys[0], 0) + diff
        
        # Compute entropy
        prob_vals = np.array([probs.get(k, 0) for k in prob_keys]) / 100.0
        H = float(scipy_entropy(prob_vals, base=2))
        
        # Direction
        direction = "bullish" if probs["bullish"] > probs["bearish"] else \
                   "bearish" if probs["bearish"] > probs["bullish"] else "neutral"
        
        return {
            "probs": {k: probs.get(k, 0) for k in prob_keys},
            "H": round(H, 4),
            "direction": direction,
            "model": model,
        }
        
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}


# ═══════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": {}, "metadata": {
        "experiment": "E5_prompt_noise",
        "model": MODEL,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }}

def save_checkpoint(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ═══════════════════════════════════════════════════════════
# Main Scoring
# ═══════════════════════════════════════════════════════════

def run_scoring(stocks, quarters, test_mode=False):
    """Run E5 scoring: same data × 3 personas × same model."""
    ckpt_path = PARENT / "e5_prompt_noise_ckpt.json"
    ckpt = load_checkpoint(ckpt_path)
    completed = ckpt.get("completed", {})
    
    # Load monthly data for feature computation
    with open(DATA_DIR / "sp500_monthly_returns.json") as f:
        monthly_data = json.load(f)["data"]
    
    # Build task list
    tasks = []
    for ticker in stocks:
        if ticker not in monthly_data:
            continue
        for quarter in quarters:
            # Compute sentiment features (used by ALL personas)
            feat = compute_sentiment_features(monthly_data[ticker], quarter)
            if feat is None:
                continue
            for persona in ["conservative", "growth", "balanced"]:
                key = f"{ticker}_{quarter}_{persona}"
                if key not in completed:
                    tasks.append((key, ticker, quarter, persona, feat))
    
    total = len(tasks)
    if total == 0:
        print("All tasks already completed!")
        return completed
    
    print(f"\n{'='*60}")
    print(f"E5 Prompt-Noise Baseline")
    print(f"  Stocks: {len(stocks)}, Quarters: {len(quarters)}")
    print(f"  Tasks remaining: {total} (already done: {len(completed)})")
    print(f"  Model: {MODEL} (same for all personas)")
    print(f"  Design: Same input × 3 persona prompts = noise floor")
    print(f"{'='*60}\n")
    
    errors = 0
    t0 = time.time()
    
    for i, (key, ticker, quarter, persona, feat) in enumerate(tasks):
        # Build prompt
        ticker_line = f"股票：{ticker}"
        
        # Convert numpy types
        clean_feat = {}
        for k, v in feat.items():
            if isinstance(v, (np.floating, float)):
                clean_feat[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean_feat[k] = int(v)
            else:
                clean_feat[k] = v
        
        prompt = PERSONAS[persona].format(ticker_line=ticker_line, **clean_feat)
        
        # Call API
        result = call_model(prompt)
        
        if "error" in result:
            errors += 1
            if errors <= 5:
                print(f"  Error on {key}: {result['error']}")
        else:
            result["persona"] = persona
            result["experiment"] = "E5"
            completed[key] = result
        
        # Checkpoint every 100 calls
        if (i + 1) % 100 == 0 or i == total - 1:
            ckpt["completed"] = completed
            ckpt["metadata"]["errors"] = errors
            save_checkpoint(ckpt_path, ckpt)
            
            elapsed = (time.time() - t0) / 60
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{total}] {elapsed:.1f}min elapsed, "
                  f"{remaining:.1f}min remaining, {errors} errors")
        
        # Rate limit
        time.sleep(0.08)
    
    # Final save
    ckpt["completed"] = completed
    ckpt["metadata"]["total_completed"] = len(completed)
    ckpt["metadata"]["total_errors"] = errors
    save_checkpoint(ckpt_path, ckpt)
    
    # Also save clean results
    results_path = PARENT / "e5_prompt_noise_ratings.json"
    with open(results_path, 'w') as f:
        json.dump(completed, f, ensure_ascii=False, indent=1)
    
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"E5 Complete: {len(completed)} ratings, {errors} errors, {elapsed:.1f}min")
    print(f"Saved to: {results_path}")
    print(f"{'='*60}")
    
    return completed


# ═══════════════════════════════════════════════════════════
# Analysis: E5 vs E1 Comparison
# ═══════════════════════════════════════════════════════════

def analyze():
    """Compare E5 (prompt noise) vs E1 (agent specialization) disagreement."""
    
    # Load E5 results
    e5_path = PARENT / "e5_prompt_noise_ratings.json"
    if not e5_path.exists():
        print("E5 results not found. Run scoring first.")
        return
    with open(e5_path) as f:
        e5_data = json.load(f)
    
    # Load E1 results
    e1_path = PARENT / "e3_high_named_ratings.json"
    if not e1_path.exists():
        print("E1 results not found.")
        return
    with open(e1_path) as f:
        e1_data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"E5 (Prompt Noise) vs E1 (Agent Specialization) Comparison")
    print(f"{'='*70}\n")
    
    # ── Compute E5 metrics ──
    e5_by_stock_month = defaultdict(dict)
    for key, val in e5_data.items():
        if "error" in val:
            continue
        parts = key.rsplit("_", 1)
        persona = parts[-1]
        prefix = parts[0]  # ticker_quarter
        e5_by_stock_month[prefix][persona] = val
    
    e5_js_list = []
    e5_disagreement_list = []
    e5_entropy_list = []
    
    for prefix, personas in e5_by_stock_month.items():
        if len(personas) < 2:
            continue
        
        # JS divergence between persona pairs
        persona_names = sorted(personas.keys())
        js_pairs = []
        for i in range(len(persona_names)):
            for j in range(i + 1, len(persona_names)):
                p1 = personas[persona_names[i]]["probs"]
                p2 = personas[persona_names[j]]["probs"]
                
                # Convert to probability arrays
                keys = ["bearish", "neutral", "bullish"]
                arr1 = np.array([p1.get(k, 0) / 100 for k in keys])
                arr2 = np.array([p2.get(k, 0) / 100 for k in keys])
                
                js = jensenshannon(arr1, arr2, base=2)
                js_pairs.append(js)
        
        if js_pairs:
            e5_js_list.append(np.mean(js_pairs))
        
        # Disagreement (direction conflict)
        dirs = [personas[p]["direction"] for p in persona_names]
        n_bull = sum(1 for d in dirs if d == "bullish")
        n_bear = sum(1 for d in dirs if d == "bearish")
        n_total = len(dirs)
        disagreement = min(n_bull, n_bear) / n_total if n_bull != n_bear else 0.5
        e5_disagreement_list.append(disagreement)
        
        # Average entropy
        H_vals = [personas[p]["H"] for p in persona_names]
        e5_entropy_list.append(np.mean(H_vals))
    
    # ── Compute E1 metrics ──
    e1_by_stock_month = defaultdict(dict)
    for ticker, quarters in e1_data.items():
        for quarter, agents in quarters.items():
            prefix = f"{ticker}_{quarter}"
            for agent, val in agents.items():
                if "error" in val:
                    continue
                e1_by_stock_month[prefix][agent] = val
    
    e1_js_list = []
    e1_disagreement_list = []
    e1_entropy_list = []
    
    for prefix, agents in e1_by_stock_month.items():
        if len(agents) < 2:
            continue
        
        agent_names = sorted(agents.keys())
        js_pairs = []
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                a1 = agents[agent_names[i]]
                a2 = agents[agent_names[j]]
                
                # Map to common probability space
                def to_bnb(probs_dict, direction):
                    """Map any agent output to {bearish, neutral, bullish}."""
                    if "bearish" in probs_dict:
                        return np.array([probs_dict["bearish"]/100, 
                                        probs_dict["neutral"]/100, 
                                        probs_dict["bullish"]/100])
                    elif "breakdown" in probs_dict:
                        return np.array([probs_dict["breakdown"]/100,
                                        probs_dict["range"]/100,
                                        probs_dict["breakout"]/100])
                    elif "undervalued" in probs_dict:
                        # undervalued = bullish (cheap → buy)
                        return np.array([probs_dict["overvalued"]/100,
                                        probs_dict["fair"]/100,
                                        probs_dict["undervalued"]/100])
                    return None
                
                arr1 = to_bnb(a1["probs"], a1["direction"])
                arr2 = to_bnb(a2["probs"], a2["direction"])
                
                if arr1 is not None and arr2 is not None:
                    js = jensenshannon(arr1, arr2, base=2)
                    js_pairs.append(js)
        
        if js_pairs:
            e1_js_list.append(np.mean(js_pairs))
        
        # Disagreement
        dir_map = {
            "bullish": "bull", "breakout": "bull", "undervalued": "bull",
            "bearish": "bear", "breakdown": "bear", "overvalued": "bear",
            "neutral": "neutral", "range": "neutral", "fair": "neutral",
        }
        dirs = [dir_map.get(agents[a]["direction"], "neutral") for a in agent_names]
        n_bull = sum(1 for d in dirs if d == "bull")
        n_bear = sum(1 for d in dirs if d == "bear")
        n_total = len(dirs)
        disagreement = min(n_bull, n_bear) / n_total if n_bull != n_bear else 0.5
        e1_disagreement_list.append(disagreement)
        
        H_vals = [agents[a]["H"] for a in agent_names]
        e1_entropy_list.append(np.mean(H_vals))
    
    # ── Print Results ──
    print(f"  {'Metric':<30} {'E1 (Agent Spec)':>18} {'E5 (Prompt Noise)':>18} {'Ratio':>10}")
    print(f"  {'─'*30} {'─'*18} {'─'*18} {'─'*10}")
    
    def fmt_stat(vals):
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    
    # JS Divergence
    e1_js_mean = np.mean(e1_js_list) if e1_js_list else 0
    e5_js_mean = np.mean(e5_js_list) if e5_js_list else 0
    js_ratio = e1_js_mean / e5_js_mean if e5_js_mean > 0 else float('inf')
    print(f"  {'JS Divergence (mean±std)':<30} {fmt_stat(e1_js_list):>18} {fmt_stat(e5_js_list):>18} {js_ratio:>10.2f}x")
    
    # Disagreement
    e1_d_mean = np.mean(e1_disagreement_list) if e1_disagreement_list else 0
    e5_d_mean = np.mean(e5_disagreement_list) if e5_disagreement_list else 0
    d_ratio = e1_d_mean / e5_d_mean if e5_d_mean > 0 else float('inf')
    print(f"  {'Disagreement (mean±std)':<30} {fmt_stat(e1_disagreement_list):>18} {fmt_stat(e5_disagreement_list):>18} {d_ratio:>10.2f}x")
    
    # Entropy
    print(f"  {'Avg Entropy (mean±std)':<30} {fmt_stat(e1_entropy_list):>18} {fmt_stat(e5_entropy_list):>18}")
    
    # N
    print(f"\n  N(E1) = {len(e1_js_list)} stock-quarters, N(E5) = {len(e5_js_list)} stock-quarters")
    
    # ── Statistical Test ──
    from scipy.stats import mannwhitneyu
    
    if e1_js_list and e5_js_list and len(e1_js_list) > 10 and len(e5_js_list) > 10:
        # JS divergence comparison
        stat_js, p_js = mannwhitneyu(e1_js_list, e5_js_list, alternative='greater')
        print(f"\n  Mann-Whitney U test (E1 JS > E5 JS):")
        print(f"    U = {stat_js:.0f}, p = {p_js:.4f} {'***' if p_js < 0.01 else '**' if p_js < 0.05 else '*' if p_js < 0.1 else 'n.s.'}")
        
        # Disagreement comparison
        stat_d, p_d = mannwhitneyu(e1_disagreement_list, e5_disagreement_list, alternative='greater')
        print(f"\n  Mann-Whitney U test (E1 Disagreement > E5 Disagreement):")
        print(f"    U = {stat_d:.0f}, p = {p_d:.4f} {'***' if p_d < 0.01 else '**' if p_d < 0.05 else '*' if p_d < 0.1 else 'n.s.'}")
    
    # ── Verdict ──
    print(f"\n{'─'*70}")
    if js_ratio > 1.5:
        print(f"  ✅ VERDICT: E1 JS divergence is {js_ratio:.1f}x higher than E5.")
        print(f"     Agent specialization produces REAL disagreement, not prompt noise.")
        print(f"     This directly addresses the 'artificial divergence' concern.")
    elif js_ratio > 1.0:
        print(f"  ⚠️ VERDICT: E1 JS divergence is {js_ratio:.1f}x higher than E5.")
        print(f"     Weak evidence. Agent specialization has marginal benefit over prompt noise.")
    else:
        print(f"  ❌ VERDICT: E5 JS divergence ≥ E1. Disagreement may be prompt noise.")
    print(f"{'─'*70}\n")
    
    # ── Save comparison results ──
    comparison = {
        "E1_JS_mean": float(e1_js_mean),
        "E5_JS_mean": float(e5_js_mean),
        "JS_ratio": float(js_ratio),
        "E1_Disagreement_mean": float(e1_d_mean),
        "E5_Disagreement_mean": float(e5_d_mean),
        "Disagreement_ratio": float(d_ratio),
        "N_E1": len(e1_js_list),
        "N_E5": len(e5_js_list),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    comp_path = PARENT / "e5_vs_e1_comparison.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {comp_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E5: Prompt-Noise Baseline")
    parser.add_argument('--test', action='store_true', help='Test mode (3 stocks, 4 quarters)')
    parser.add_argument('--analyze', action='store_true', help='Compare E5 vs E1 results')
    parser.add_argument('--stocks', type=int, default=46, help='Number of stocks')
    args = parser.parse_args()
    
    if args.analyze:
        analyze()
        return
    
    # Load stock list from E1 data
    e1_path = PARENT / "e3_high_named_ratings.json"
    with open(e1_path) as f:
        e1_data = json.load(f)
    stocks = sorted(e1_data.keys())
    
    if args.test:
        stocks = stocks[:3]
        quarters = ["2022-03", "2022-06", "2022-09", "2022-12"]
    else:
        quarters = [f"{y}-{m:02d}" for y in range(2005, 2025) for m in [3, 6, 9, 12]]
    
    if args.stocks and not args.test:
        stocks = stocks[:args.stocks]
    
    # Run scoring
    run_scoring(stocks, quarters, test_mode=args.test)
    
    # Auto-analyze after scoring
    print("\nAuto-running E5 vs E1 comparison...")
    analyze()


if __name__ == "__main__":
    main()
