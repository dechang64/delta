#!/usr/bin/env python3
"""
Step 3: Full analysis for A-share data — same methodology as US stocks.
Produces FM results, portfolio sorts, sub-sample analysis, and cross-market comparison.

Author: Siyi / 2026-06-05
"""

import json
import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import entropy

OUT = "/home/z/my-project/delta_ashare"
OUT_US = "/home/z/my-project/delta_jfe"

# ── Load A-share data ──
with open(os.path.join(OUT, "ashare_monthly_returns.json"), "r") as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(os.path.join(OUT, "ashare_agent_ratings.json"), "r") as f:
    ratings = json.load(f)

print(f"A-share stocks: {len(stock_data)}, Rated stocks: {len(ratings)}")

# ── Helper functions (same as US pipeline) ──
def rating_to_probs(rating):
    x = (rating - 5.5) / 2.0
    p_neg = np.exp(-x) / (np.exp(-x) + 1 + np.exp(x))
    p_neu = 1 / (np.exp(-x) + 1 + np.exp(x))
    p_pos = np.exp(x) / (np.exp(-x) + 1 + np.exp(x))
    return np.array([p_neg, p_neu, p_pos])

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def compute_all_metrics(r_sent, r_tech, r_fund):
    sp = rating_to_probs(r_sent)
    tp = rating_to_probs(r_tech)
    fp = rating_to_probs(r_fund)
    
    js_st = js_divergence(sp, tp)
    js_sf = js_divergence(sp, fp)
    js_tf = js_divergence(tp, fp)
    js_post = (js_st + js_sf + js_tf) / 3
    
    d_post = np.std([r_sent, r_tech, r_fund])
    
    h_sentiment = float(entropy(sp))
    confidence = float(sp.max())
    
    return {
        "js_post": round(js_post, 6),
        "d_post": round(d_post, 6),
        "h_sentiment": round(h_sentiment, 6),
        "confidence": round(confidence, 6),
    }

# ── Build panel ──
print("\nBuilding A-share panel dataset...")
rows = []
for ticker in sorted(ratings.keys()):
    for month in sorted(ratings[ticker].keys()):
        r = ratings[ticker][month]
        if not isinstance(r, dict) or len(r) != 3:
            continue
        s, t, f_ = r.get("sentiment", 5), r.get("technical", 5), r.get("fundamental", 5)
        m = compute_all_metrics(s, t, f_)
        
        # Get next-month return
        months_sorted = sorted(stock_data.get(ticker, {}).keys())
        if month not in months_sorted:
            continue
        idx = months_sorted.index(month)
        if idx + 1 >= len(months_sorted):
            continue
        next_ret = stock_data[ticker][months_sorted[idx+1]]["return"]
        
        rows.append({
            "ticker": ticker,
            "month": month,
            "r_sentiment": s, "r_technical": t, "r_fundamental": f_,
            "js_post": m["js_post"], "d_post": m["d_post"],
            "h_sentiment": m["h_sentiment"], "confidence": m["confidence"],
            "return": next_ret,
        })

df = pd.DataFrame(rows)
print(f"Panel: {len(df)} observations, {df['ticker'].nunique()} stocks, {df['month'].nunique()} months")

# ── Fama-MacBeth Regressions ──
print("\n--- Fama-MacBeth Univariate Regressions ---")

def fm_regression(df, signal, min_obs=10):
    """Fama-MacBeth cross-sectional regression with Newey-West SE."""
    months = sorted(df['month'].unique())
    betas = []
    
    for month in months:
        sub = df[df['month'] == month]
        if len(sub) < min_obs:
            continue
        x = sub[signal].values
        y = sub['return'].values
        
        # Add constant
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            betas.append(beta[1])  # slope
        except:
            continue
    
    if len(betas) < 10:
        return None
    
    betas = np.array(betas)
    mean_beta = np.mean(betas)
    
    # Newey-West SE
    T = len(betas)
    lag = max(1, int(T ** (1/3)))
    
    gamma0 = np.var(betas, ddof=1)
    gamma_sum = 0
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)
        gamma_j = np.mean((betas[j:] - mean_beta) * (betas[:-j] - mean_beta))
        gamma_sum += 2 * w * gamma_j
    
    nw_var = (gamma0 + gamma_sum) / T
    nw_se = np.sqrt(max(nw_var, 1e-10))
    t_stat = mean_beta / nw_se
    
    # p-value from t-distribution
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=T-1))
    
    # Significance stars
    if p_val < 0.01:
        sig = "***"
    elif p_val < 0.05:
        sig = "**"
    elif p_val < 0.1:
        sig = "*"
    else:
        sig = ""
    
    return {
        "beta": round(mean_beta, 6),
        "t": round(t_stat, 2),
        "sig": sig,
        "p": round(p_val, 4),
        "n_months": len(betas),
    }

signals = ["js_post", "d_post", "h_sentiment", "confidence"]
fm_results = {}
for sig in signals:
    result = fm_regression(df, sig)
    if result:
        fm_results[sig] = result
        print(f"  {sig}: β={result['beta']:+.4f}, t={result['t']:+.2f}{result['sig']}, p={result['p']:.4f}")

# ── Portfolio Sorts ──
print("\n--- Portfolio Sorts ---")

def portfolio_sort(df, signal, n_groups=10):
    """Decile/quintile portfolio sort."""
    months = sorted(df['month'].unique())
    port_rets = {f"Q{i+1}": [] for i in range(n_groups)}
    
    for month in months:
        sub = df[df['month'] == month].copy()
        if len(sub) < n_groups * 3:
            continue
        sub = sub.sort_values(signal)
        n = len(sub)
        qsize = n // n_groups
        
        for q in range(n_groups):
            start = q * qsize
            end = (q + 1) * qsize if q < n_groups - 1 else n
            group = sub.iloc[start:end]
            port_rets[f"Q{q+1}"].append(group['return'].mean())
    
    # Compute L-S
    if port_rets["Q1"] and port_rets[f"Q{n_groups}"]:
        ls_rets = [h - l for h, l in zip(port_rets[f"Q{n_groups}"], port_rets["Q1"])]
        mean_ls = np.mean(ls_rets)
        t_ls = np.mean(ls_rets) / (np.std(ls_rets, ddof=1) / np.sqrt(len(ls_rets)))
        p_ls = 2 * (1 - stats.t.cdf(abs(t_ls), df=len(ls_rets)-1))
        sig = "***" if p_ls < 0.01 else "**" if p_ls < 0.05 else "*" if p_ls < 0.1 else ""
        
        return {
            "groups": n_groups,
            "ls_return": f"{mean_ls*100:+.2f}%/mo",
            "ls_t": round(t_ls, 2),
            "ls_sig": sig,
            "q1_mean": f"{np.mean(port_rets['Q1'])*100:+.2f}%",
            f"q{n_groups}_mean": f"{np.mean(port_rets[f'Q{n_groups}'])*100:+.2f}%",
        }
    return None

port_results = {}
for sig in signals:
    n_groups = 5 if sig == "h_sentiment" else 10
    result = portfolio_sort(df, sig, n_groups)
    if result:
        port_results[sig] = result
        print(f"  {sig} ({n_groups}-group): L-S={result['ls_return']}, t={result['ls_t']:+.2f}{result['ls_sig']}")

# ── Sub-Sample Analysis ──
print("\n--- Sub-Sample Robustness ---")

sub_periods = [
    ("2005-2014", "2005-01", "2014-12"),
    ("2015-2019", "2015-01", "2019-12"),
    ("2020-2024", "2020-01", "2024-12"),
]

sub_results = {}
for label, start, end in sub_periods:
    sub_df = df[(df['month'] >= start) & (df['month'] <= end)]
    result = fm_regression(sub_df, "h_sentiment")
    if result:
        sub_results[label] = {"t": result["t"], "sig": result["sig"]}
        print(f"  {label}: t={result['t']:+.2f}{result['sig']} (n={len(sub_df)})")

# ── Cross-Market Comparison ──
print("\n--- Cross-Market Comparison (A-share vs US) ---")

# Load US results
with open(os.path.join(OUT_US, "analysis_results_llm_corrected.json"), "r") as f:
    us_results = json.load(f)

comparison = {}
for sig in signals:
    a_fm = fm_results.get(sig, {})
    u_fm = us_results.get("fm_univariate", {}).get(sig, {})
    
    comparison[sig] = {
        "a_beta": a_fm.get("beta", 0),
        "a_t": a_fm.get("t", 0),
        "a_sig": a_fm.get("sig", ""),
        "u_beta": u_fm.get("beta", 0),
        "u_t": u_fm.get("t", 0),
        "u_sig": u_fm.get("sig", ""),
    }
    c = comparison[sig]
    print(f"  {sig}: A股 β={c['a_beta']:+.4f} t={c['a_t']:+.2f}{c['a_sig']} | US β={c['u_beta']:+.4f} t={c['u_t']:+.2f}{c['u_sig']}")

# ── Save all results ──
results = {
    "market": "A-share (CSI 300)",
    "fm_univariate": fm_results,
    "portfolio_sorts": port_results,
    "subsample": sub_results,
    "cross_market": comparison,
    "n_stocks": df['ticker'].nunique(),
    "n_obs": len(df),
    "n_months": df['month'].nunique(),
}

with open(os.path.join(OUT, "ashare_analysis_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Also save panel data
df.to_csv(os.path.join(OUT, "ashare_panel_data.csv"), index=False)

print(f"\n✅ Results saved to: {OUT}/ashare_analysis_results.json")
print(f"✅ Panel data saved to: {OUT}/ashare_panel_data.csv")
