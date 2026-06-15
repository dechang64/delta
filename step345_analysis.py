#!/usr/bin/env python3
"""
Step 3: Compute FF5 alphas + characteristic-adjusted returns
Step 4: Fama-MacBeth cross-sectional regressions with FF5 controls
Step 5: Decile portfolio sorts + long-short + transaction costs

All in one script for efficiency.

Author: Siyi / 2026-06-03
"""

import json
import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import entropy

OUT = "/home/z/my-project/delta_jfe"

# ── Load data ──
with open(os.path.join(OUT, "sp500_monthly_returns.json"), "r") as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(os.path.join(OUT, "agent_ratings_quant.json"), "r") as f:
    ratings = json.load(f)

# Load FF5 factors
ff5 = pd.read_csv(os.path.join(OUT, "ff5_factors.csv"))
mom = pd.read_csv(os.path.join(OUT, "ff_momentum.csv"))

print(f"Stocks: {len(stock_data)}, FF5 months: {len(ff5)}, Mom months: {len(mom)}")

# ── Helper: rating to probability vector ──
def rating_to_probs(rating):
    p_neg = max(0, (4 - rating) / 3) * 0.8 + 0.05
    p_pos = max(0, (rating - 7) / 3) * 0.8 + 0.05
    p_neu = max(0.05, 1 - p_neg - p_pos)
    total = p_neg + p_neu + p_pos
    return np.array([p_neg, p_neu, p_pos]) / total

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

# ── Build panel dataset ──
print("\nBuilding panel dataset...")

records = []
all_tickers = sorted(stock_data.keys())
all_months = sorted(stock_data[all_tickers[0]].keys())

for ticker in all_tickers:
    data = stock_data[ticker]
    r_data = ratings.get(ticker, {})
    
    for month in all_months:
        if month not in data or month not in r_data:
            continue
        
        d = data[month]
        r = r_data[month]
        
        ret = d.get("return", None)
        if ret is None:
            continue
        
        # Compute disagreement metrics
        s_prob = rating_to_probs(r["sentiment"])
        t_prob = rating_to_probs(r["technical"])
        f_prob = rating_to_probs(r["fundamental"])
        avg_prob = (s_prob + t_prob + f_prob) / 3
        uniform = np.array([1/3, 1/3, 1/3])
        
        js = js_divergence(avg_prob, uniform)
        d_post = np.std([r["sentiment"], r["technical"], r["fundamental"]])
        h_sent = entropy(s_prob)
        confidence = np.max(avg_prob)
        
        # Information asymmetry
        # IA = mutual information / total entropy
        total_ent = entropy(avg_prob)
        ia = 1 - (entropy(avg_prob) / max(entropy(uniform), 1e-8)) if total_ent > 0 else 0
        
        # D_irreducible
        js_baseline = js_divergence(uniform, uniform)  # = 0
        d_irred = max(0, js - js_baseline)
        
        # Market cap proxy (volume-based)
        vol = d.get("volume", 0)
        
        # Size proxy: use price level
        price = d.get("close", 1)
        
        records.append({
            "ticker": ticker,
            "month": month,
            "return": ret,
            "JS_post": js,
            "D_post": d_post,
            "H_sentiment": h_sent,
            "confidence": confidence,
            "IA": ia,
            "D_irreducible": d_irred,
            "volume": vol,
            "price": price,
            "sentiment_rating": r["sentiment"],
            "technical_rating": r["technical"],
            "fundamental_rating": r["fundamental"],
        })

panel = pd.DataFrame(records)
panel["year"] = panel["month"].str[:4].astype(int)
panel["month_num"] = panel["month"].str[5:7].astype(int)

print(f"Panel: {len(panel):,} observations, {panel['ticker'].nunique()} stocks, {panel['month'].nunique()} months")

# ── Merge FF5 factors ──
print("\nMerging FF5 factors...")

# FF5 CSV already has: month, Mkt-RF, SMB, HML, RMW, CMA, RF
ff5 = pd.read_csv(os.path.join(OUT, "ff5_factors.csv"))
mom = pd.read_csv(os.path.join(OUT, "ff_momentum.csv"))

print(f"FF5 columns: {ff5.columns.tolist()}")
print(f"Mom columns: {mom.columns.tolist()}")

# Merge FF5
ff5_factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] if c in ff5.columns]
panel = panel.merge(ff5[["month"] + ff5_factor_cols], on="month", how="left")

# Merge Momentum
if "Mom" in mom.columns:
    panel = panel.merge(mom[["month", "Mom"]], on="month", how="left")

# Excess return
if "RF" in panel.columns:
    panel["excess_return"] = panel["return"] - panel["RF"]
else:
    panel["excess_return"] = panel["return"] - 0.003/12  # Approximate monthly RF

print(f"Panel with FF5: {len(panel):,} obs, excess_return mean={panel['excess_return'].mean()*100:.3f}%/month")

# ── Step 3: FF5 Alpha Computation ──
print(f"\n{'='*60}")
print(f"STEP 3: FF5 ALPHA COMPUTATION")
print(f"{'='*60}")

# For each stock, run time-series regression: excess_return = alpha + β1*Mkt + β2*SMB + β3*HML + β4*RMW + β5*CMA + ε
from numpy.linalg import lstsq

ff5_factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in panel.columns]

if ff5_factor_cols:
    stock_alphas = {}
    for ticker in panel["ticker"].unique():
        stk = panel[panel["ticker"] == ticker].dropna(subset=ff5_factor_cols + ["excess_return"])
        if len(stk) < 24:  # Need at least 2 years
            continue
        
        X = stk[ff5_factor_cols].values
        X = np.column_stack([np.ones(len(X)), X])
        y = stk["excess_return"].values
        
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            alpha = beta[0]
            predicted = X @ beta
            resid = y - predicted
            stock_alphas[ticker] = {
                "alpha_monthly": alpha,
                "alpha_annual": alpha * 12,
                "resid_std": np.std(resid),
                "n_obs": len(stk),
            }
        except Exception:
            continue
    
    print(f"FF5 alphas computed for {len(stock_alphas)} stocks")
    alphas = [v["alpha_monthly"] for v in stock_alphas.values()]
    print(f"  Monthly alpha: mean={np.mean(alphas)*100:.3f}%, std={np.std(alphas)*100:.3f}%")
    print(f"  Annual alpha:  mean={np.mean([v['alpha_annual'] for v in stock_alphas.values()])*100:.2f}%")
    
    # Add characteristic-adjusted returns to panel
    alpha_map = {t: v["alpha_monthly"] for t, v in stock_alphas.items()}
    resid_map = {t: v["resid_std"] for t, v in stock_alphas.items()}
    
    panel["ff5_alpha"] = panel["ticker"].map(alpha_map)
    panel["ff5_resid_std"] = panel["ticker"].map(resid_map)
    
    # Characteristic-adjusted return = excess return - FF5 predicted
    # We'll compute this per observation
    panel["ff5_adjusted_return"] = panel["excess_return"]  # Will be updated below
    
    for ticker in stock_alphas:
        mask = panel["ticker"] == ticker
        stk = panel[mask].dropna(subset=ff5_factor_cols + ["excess_return"])
        if len(stk) == 0:
            continue
        X = stk[ff5_factor_cols].values
        X = np.column_stack([np.ones(len(X)), X])
        y = stk["excess_return"].values
        beta, _, _, _ = lstsq(X, y, rcond=None)
        predicted = X @ beta
        panel.loc[stk.index, "ff5_adjusted_return"] = y - predicted + stock_alphas[ticker]["alpha_monthly"]
    
    print(f"  FF5-adjusted returns computed")

# ── Step 4: Fama-MacBeth Regressions ──
print(f"\n{'='*60}")
print(f"STEP 4: FAMA-MACBETH CROSS-SECTIONAL REGRESSIONS")
print(f"{'='*60}")

def fama_macbeth(panel, y_col, x_cols, month_col="month"):
    """Run Fama-MacBeth cross-sectional regressions."""
    months = sorted(panel[month_col].dropna().unique())
    
    betas = {col: [] for col in x_cols}
    intercepts = []
    n_months = 0
    
    for month in months:
        month_data = panel[panel[month_col] == month].dropna(subset=[y_col] + x_cols)
        if len(month_data) < 10:  # Need enough stocks per month
            continue
        
        X = month_data[x_cols].values
        X = np.column_stack([np.ones(len(X)), X])
        y = month_data[y_col].values
        
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            intercepts.append(beta[0])
            for i, col in enumerate(x_cols):
                betas[col].append(beta[i+1])
            n_months += 1
        except Exception:
            continue
    
    # Fama-MacBeth estimator: time-series average of monthly betas
    results = {}
    for col in x_cols:
        b = np.array(betas[col])
        if len(b) > 1:
            mean_b = np.mean(b)
            se_b = np.std(b, ddof=1) / np.sqrt(len(b))
            t_stat = mean_b / se_b if se_b > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(b)-1))
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            results[col] = {
                "beta": mean_b, "se": se_b, "t_stat": t_stat,
                "p_val": p_val, "sig": sig, "n_months": n_months,
            }
    
    return results

# Univariate regressions
signals = ["JS_post", "D_post", "H_sentiment", "confidence", "IA", "D_irreducible"]
dep_vars = ["excess_return", "ff5_adjusted_return"] if "ff5_adjusted_return" in panel.columns else ["excess_return"]

print("\n--- Univariate Regressions (Dependent: Excess Return) ---")
for signal in signals:
    res = fama_macbeth(panel, "excess_return", [signal])
    if signal in res:
        r = res[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  (n_months={r['n_months']})")

# Multivariate regression
print("\n--- Multivariate Regression ---")
multi_signals = ["JS_post", "D_post", "H_sentiment", "confidence"]
res = fama_macbeth(panel, "excess_return", multi_signals)
for signal in multi_signals:
    if signal in res:
        r = res[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# With FF5-adjusted returns
if "ff5_adjusted_return" in panel.columns:
    print("\n--- Univariate on FF5-Adjusted Returns ---")
    for signal in signals:
        res = fama_macbeth(panel, "ff5_adjusted_return", [signal])
        if signal in res:
            r = res[signal]
            print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# ── Step 5: Decile Portfolio Sorts ──
print(f"\n{'='*60}")
print(f"STEP 5: DECILE PORTFOLIO SORTS + LONG-SHORT")
print(f"{'='*60}")

def decile_sort(panel, signal_col, return_col="excess_return", n_groups=10):
    """Sort stocks into deciles each month and compute portfolio returns."""
    months = sorted(panel["month"].dropna().unique())
    
    portfolio_rets = {f"D{i+1}": [] for i in range(n_groups)}
    portfolio_rets["Long-Short"] = []
    
    for month in months:
        month_data = panel[panel["month"] == month].dropna(subset=[signal_col, return_col])
        if len(month_data) < n_groups * 2:  # Need enough stocks
            continue
        
        # Use rank-based assignment (handles ties properly)
        month_data = month_data.copy()
        ranks = month_data[signal_col].rank(method="average")
        n = len(ranks)
        month_data["decile"] = np.minimum(np.floor((ranks - 1) / n * n_groups).astype(int) + 1, n_groups)
        
        for d in range(1, n_groups + 1):
            decile_rets = month_data[month_data["decile"] == d][return_col]
            if len(decile_rets) > 0:
                portfolio_rets[f"D{d}"].append(decile_rets.mean())
        
        # Long-short: D10 - D1
        d10 = month_data[month_data["decile"] == n_groups][return_col]
        d1 = month_data[month_data["decile"] == 1][return_col]
        if len(d10) > 0 and len(d1) > 0:
            portfolio_rets["Long-Short"].append(d10.mean() - d1.mean())
    
    # Compute statistics
    results = {}
    for name, rets in portfolio_rets.items():
        if len(rets) > 0:
            mean_ret = np.mean(rets) * 100  # percentage
            std_ret = np.std(rets, ddof=1) * 100
            t_stat = np.mean(rets) / (np.std(rets, ddof=1) / np.sqrt(len(rets)))
            sharpe = np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(12) if np.std(rets, ddof=1) > 0 else 0
            results[name] = {
                "mean_monthly": mean_ret,
                "std_monthly": std_ret,
                "t_stat": t_stat,
                "sharpe_annual": sharpe,
                "n_months": len(rets),
            }
    
    return results

# Sort by JS_post
print("\n--- Decile Sorts by JS_post ---")
js_deciles = decile_sort(panel, "JS_post")
for name, r in js_deciles.items():
    sig = "***" if abs(r["t_stat"]) > 2.58 else "**" if abs(r["t_stat"]) > 1.96 else "*" if abs(r["t_stat"]) > 1.64 else ""
    print(f"  {name:12s}: {r['mean_monthly']:+6.2f}%/mo  t={r['t_stat']:+6.2f}{sig}  Sharpe={r['sharpe_annual']:+5.2f}")

# Sort by H_sentiment
print("\n--- Decile Sorts by H_sentiment ---")
h_deciles = decile_sort(panel, "H_sentiment")
for name, r in h_deciles.items():
    sig = "***" if abs(r["t_stat"]) > 2.58 else "**" if abs(r["t_stat"]) > 1.96 else "*" if abs(r["t_stat"]) > 1.64 else ""
    print(f"  {name:12s}: {r['mean_monthly']:+6.2f}%/mo  t={r['t_stat']:+6.2f}{sig}  Sharpe={r['sharpe_annual']:+5.2f}")

# Sort by D_post
print("\n--- Decile Sorts by D_post ---")
d_deciles = decile_sort(panel, "D_post")
for name, r in d_deciles.items():
    sig = "***" if abs(r["t_stat"]) > 2.58 else "**" if abs(r["t_stat"]) > 1.96 else "*" if abs(r["t_stat"]) > 1.64 else ""
    print(f"  {name:12s}: {r['mean_monthly']:+6.2f}%/mo  t={r['t_stat']:+6.2f}{sig}  Sharpe={r['sharpe_annual']:+5.2f}")

# ── Transaction Cost Analysis ──
print(f"\n{'='*60}")
print(f"TRANSACTION COST ANALYSIS")
print(f"{'='*60}")

# Long-short portfolio with transaction costs
# Assume 10bps one-way for large-cap stocks
tc_one_way = 0.001  # 10 bps
tc_round_trip = 2 * tc_one_way

# Estimate monthly turnover from decile portfolio
# Simplified: assume 30% monthly turnover
for turnover in [0.2, 0.3, 0.5]:
    net_ret = js_deciles.get("Long-Short", {}).get("mean_monthly", 0)
    tc_cost = tc_round_trip * turnover * 100  # in percentage
    net_after_tc = net_ret - tc_cost
    print(f"  Turnover={turnover:.0%}: gross={net_ret:+.2f}% - TC={tc_cost:.2f}% = net={net_after_tc:+.2f}%/mo")

# ── Sub-sample Robustness ──
print(f"\n{'='*60}")
print(f"SUB-SAMPLE ROBUSTNESS")
print(f"{'='*60}")

# Pre-COVID vs Post-COVID
covid_split = "2020-01"
for label, sub in [("Pre-COVID (2005-2019)", panel[panel["month"] < covid_split]),
                    ("Post-COVID (2020-2024)", panel[panel["month"] >= covid_split])]:
    print(f"\n  {label}:")
    for signal in ["JS_post", "H_sentiment", "D_post"]:
        res = fama_macbeth(sub, "excess_return", [signal])
        if signal in res:
            r = res[signal]
            print(f"    {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# Financial crisis vs normal
crisis_start, crisis_end = "2007-10", "2009-03"
for label, sub in [("Financial Crisis (2007-2009)", panel[(panel["month"] >= crisis_start) & (panel["month"] <= crisis_end)]),
                    ("Non-Crisis", panel[(panel["month"] < crisis_start) | (panel["month"] > crisis_end)])]:
    print(f"\n  {label}:")
    for signal in ["JS_post", "H_sentiment", "D_post"]:
        res = fama_macbeth(sub, "excess_return", [signal])
        if signal in res:
            r = res[signal]
            print(f"    {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# ── Save results ──
results_summary = {
    "panel_stats": {
        "n_obs": len(panel),
        "n_stocks": panel["ticker"].nunique(),
        "n_months": panel["month"].nunique(),
        "period": f"{panel['month'].min()} to {panel['month'].max()}",
    },
    "fm_univariate": {},
    "fm_multivariate": {},
    "decile_js": js_deciles,
    "decile_h": h_deciles,
    "decile_d": d_deciles,
}

for signal in signals:
    res = fama_macbeth(panel, "excess_return", [signal])
    if signal in res:
        results_summary["fm_univariate"][signal] = res[signal]

res = fama_macbeth(panel, "excess_return", multi_signals)
for signal in multi_signals:
    if signal in res:
        results_summary["fm_multivariate"][signal] = res[signal]

with open(os.path.join(OUT, "analysis_results.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

# Save panel for later use
panel.to_csv(os.path.join(OUT, "panel_data.csv"), index=False)

print(f"\n{'='*60}")
print(f"ALL ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Panel: {len(panel):,} obs saved to panel_data.csv")
print(f"Results saved to analysis_results.json")
