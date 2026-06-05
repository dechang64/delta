#!/usr/bin/env python3
"""
Step 3-6: Full JFE-grade analysis (revised).

- FF5 alpha + characteristic-adjusted returns
- Fama-MacBeth with Newey-West standard errors (lag=6)
- Decile portfolio sorts with proper rank-based assignment
- Long-short with transaction costs
- Sub-sample robustness (Pre/Post-COVID, Crisis/Non-Crisis, by decade)
- Multiple testing correction (Bonferroni + FDR)

Author: Siyi / 2026-06-03
"""

import json
import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import entropy
from numpy.linalg import lstsq

OUT = "/home/z/my-project/delta_jfe"

# ── Load data ──
with open(os.path.join(OUT, "sp500_monthly_returns.json"), "r") as f:
    raw = json.load(f)
stock_data = raw["data"]

with open(os.path.join(OUT, "agent_ratings_quant.json"), "r") as f:
    ratings = json.load(f)

ff5 = pd.read_csv(os.path.join(OUT, "ff5_factors.csv"))
mom = pd.read_csv(os.path.join(OUT, "ff_momentum.csv"))

print(f"Stocks: {len(stock_data)}, FF5: {len(ff5)} months, Mom: {len(mom)} months")

# ── Helper: rating to probability vector (smooth softmax) ──
def rating_to_probs(rating):
    x = (rating - 5.5) / 2.0
    p_neg = np.exp(-x) / (np.exp(-x) + 1 + np.exp(x))
    p_neu = 1 / (np.exp(-x) + 1 + np.exp(x))
    p_pos = np.exp(x) / (np.exp(-x) + 1 + np.exp(x))
    return np.array([p_neg, p_neu, p_pos])

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

# ── Build panel ──
print("\nBuilding panel dataset...")
rows = []
for ticker in sorted(ratings.keys()):
    for month in sorted(ratings[ticker].keys()):
        r = ratings[ticker][month]
        s, t, f = r["sentiment"], r["technical"], r["fundamental"]
        
        sp = rating_to_probs(s)
        tp = rating_to_probs(t)
        fp = rating_to_probs(f)
        avg_p = (sp + tp + fp) / 3
        uniform = np.array([1/3, 1/3, 1/3])
        
        js = js_divergence(avg_p, uniform)
        d_post = np.std([s, t, f])
        h_sent = entropy(sp)
        conf = float(np.max(avg_p))
        
        # Market data
        ret = stock_data.get(ticker, {}).get(month, {}).get("return", np.nan)
        vol = stock_data.get(ticker, {}).get(month, {}).get("volume", np.nan)
        
        rows.append({
            "ticker": ticker, "month": month,
            "return": ret, "volume": vol,
            "JS_post": js, "D_post": d_post, "H_sentiment": h_sent,
            "confidence": conf,
            "rating_s": s, "rating_t": t, "rating_f": f,
        })

panel = pd.DataFrame(rows)
panel = panel.dropna(subset=["return"])

# Merge FF5 + Momentum
panel = panel.merge(ff5, on="month", how="left")
panel = panel.merge(mom, on="month", how="left")

# Excess return
panel["excess_return"] = panel["return"] - panel["RF"]

# Size proxy: log volume (relative to cross-section median each month)
panel["log_volume"] = np.log1p(panel["volume"])
med_vol = panel.groupby("month")["log_volume"].transform("median")
panel["rel_size"] = panel["log_volume"] / med_vol

# Momentum proxy: will be computed from past returns later
# For now, use Mom factor as control

print(f"Panel: {len(panel):,} obs, {panel['ticker'].nunique()} stocks, {panel['month'].nunique()} months")

# ── STEP 3: FF5 Alpha ──
print(f"\n{'='*60}")
print("STEP 3: FF5 ALPHA + CHARACTERISTIC-ADJUSTED RETURNS")
print(f"{'='*60}")

factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
stock_alphas = {}

for ticker in panel["ticker"].unique():
    stk = panel[panel["ticker"] == ticker].dropna(subset=factor_cols + ["excess_return"])
    if len(stk) < 36:
        continue
    try:
        X = stk[factor_cols].values
        X = np.column_stack([np.ones(len(X)), X])
        y = stk["excess_return"].values
        beta, _, _, _ = lstsq(X, y, rcond=None)
        alpha = beta[0]
        predicted = X @ beta
        resid = y - predicted
        stock_alphas[ticker] = {"alpha": alpha, "resid_std": np.std(resid), "n": len(stk)}
    except:
        continue

# Add FF5-adjusted return
alpha_map = {t: v["alpha"] for t, v in stock_alphas.items()}
panel["ff5_alpha"] = panel["ticker"].map(alpha_map)
panel["ff5_adjusted_return"] = panel["excess_return"] - panel["ff5_alpha"]

alphas = [v["alpha"] for v in stock_alphas.values()]
print(f"FF5 alphas: {len(stock_alphas)} stocks, mean={np.mean(alphas)*100:.3f}%/mo, std={np.std(alphas)*100:.3f}%/mo")

# ── STEP 4: Fama-MacBeth with Newey-West ──
print(f"\n{'='*60}")
print("STEP 4: FAMA-MACBETH REGRESSIONS (Newey-West lag=6)")
print(f"{'='*60}")

def newey_west_se(betas, lag=6):
    """Newey-West standard error for time-series of betas."""
    T = len(betas)
    mean_b = np.mean(betas)
    if T < 2:
        return 0, mean_b
    
    # OLS variance
    gamma0 = np.var(betas, ddof=1)
    
    # Newey-West adjustment
    nw_var = gamma0
    for j in range(1, min(lag + 1, T)):
        w = 1 - j / (lag + 1)  # Bartlett kernel
        gamma_j = np.mean((betas[j:] - mean_b) * (betas[:-j] - mean_b))
        nw_var += 2 * w * gamma_j
    
    nw_var = max(nw_var, 0)  # Ensure non-negative
    se = np.sqrt(nw_var / T)
    t_stat = mean_b / se if se > 0 else 0
    return se, t_stat

def fama_macbeth(panel, y_col, x_cols, nw_lag=6):
    """Fama-MacBeth regression with Newey-West standard errors."""
    months = sorted(panel["month"].dropna().unique())
    betas = {col: [] for col in x_cols}
    intercepts = []
    n_months = 0
    
    for month in months:
        md = panel[panel["month"] == month].dropna(subset=[y_col] + x_cols)
        if len(md) < 10:
            continue
        X = np.column_stack([np.ones(len(md))] + [md[col].values for col in x_cols])
        y = md[y_col].values
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            intercepts.append(beta[0])
            for i, col in enumerate(x_cols):
                betas[col].append(beta[i + 1])
            n_months += 1
        except:
            continue
    
    results = {}
    for col in x_cols:
        b = np.array(betas[col])
        if len(b) > 1:
            se, t_stat = newey_west_se(b, lag=nw_lag)
            mean_b = np.mean(b)
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))  # Use normal for large T
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            results[col] = {"beta": mean_b, "se": se, "t_stat": t_stat, "p_val": p_val, "sig": sig, "n_months": n_months}
    
    return results

signals = ["JS_post", "D_post", "H_sentiment", "confidence"]
dep_vars = ["excess_return", "ff5_adjusted_return"]

print("\n--- Univariate (Dep: Excess Return) ---")
uni_results = {}
for signal in signals:
    res = fama_macbeth(panel, "excess_return", [signal])
    if signal in res:
        r = res[signal]
        uni_results[signal] = r
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  NW-SE={r['se']:.4f}")

print("\n--- Univariate (Dep: FF5-Adjusted Return) ---")
for signal in signals:
    res = fama_macbeth(panel, "ff5_adjusted_return", [signal])
    if signal in res:
        r = res[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  NW-SE={r['se']:.4f}")

print("\n--- Multivariate (Dep: Excess Return) ---")
multi_signals = ["JS_post", "D_post", "H_sentiment", "confidence"]
multi_results = fama_macbeth(panel, "excess_return", multi_signals)
for signal in multi_signals:
    if signal in multi_results:
        r = multi_results[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# With FF5 factor controls
print("\n--- Multivariate + FF5 Factor Loadings ---")
all_controls = multi_signals + ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
full_results = fama_macbeth(panel, "excess_return", all_controls)
for signal in all_controls:
    if signal in full_results:
        r = full_results[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# ── STEP 5: Decile Portfolio Sorts ──
print(f"\n{'='*60}")
print("STEP 5: DECILE PORTFOLIO SORTS + TRANSACTION COSTS")
print(f"{'='*60}")

def decile_sort(panel, signal_col, return_col="excess_return", n_groups=10):
    """Rank-based decile sort with full statistics."""
    months = sorted(panel["month"].dropna().unique())
    portfolio_rets = {f"D{i+1}": [] for i in range(n_groups)}
    portfolio_rets["Long-Short"] = []
    
    for month in months:
        md = panel[panel["month"] == month].dropna(subset=[signal_col, return_col])
        if len(md) < n_groups * 2:
            continue
        md = md.copy()
        ranks = md[signal_col].rank(method="average")
        n = len(ranks)
        md["decile"] = np.minimum(np.floor((ranks - 1) / n * n_groups).astype(int) + 1, n_groups)
        
        for d in range(1, n_groups + 1):
            dr = md[md["decile"] == d][return_col]
            if len(dr) > 0:
                portfolio_rets[f"D{d}"].append(dr.mean())
        
        d10 = md[md["decile"] == n_groups][return_col]
        d1 = md[md["decile"] == 1][return_col]
        if len(d10) > 0 and len(d1) > 0:
            portfolio_rets["Long-Short"].append(d10.mean() - d1.mean())
    
    results = {}
    for name, rets in portfolio_rets.items():
        if len(rets) > 1:
            rets = np.array(rets)
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            se = std_r / np.sqrt(len(rets))
            t_stat = mean_r / se
            sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
            results[name] = {
                "mean_monthly_pct": mean_r * 100,
                "std_monthly_pct": std_r * 100,
                "t_stat": t_stat,
                "sharpe_annual": sharpe,
                "n_months": len(rets),
            }
    return results

for signal_name, signal_col in [("JS Divergence", "JS_post"), ("H Sentiment", "H_sentiment"), ("D_post", "D_post")]:
    print(f"\n--- Decile Sort by {signal_name} ---")
    dec = decile_sort(panel, signal_col)
    for name, r in dec.items():
        sig = "***" if abs(r["t_stat"]) > 2.58 else "**" if abs(r["t_stat"]) > 1.96 else "*" if abs(r["t_stat"]) > 1.64 else ""
        print(f"  {name:12s}: {r['mean_monthly_pct']:+6.2f}%/mo  t={r['t_stat']:+6.2f}{sig}  Sharpe={r['sharpe_annual']:+5.2f}")

# ── Transaction Costs ──
print(f"\n--- Transaction Cost Analysis (JS Long-Short) ---")
js_dec = decile_sort(panel, "JS_post")
ls_gross = js_dec.get("Long-Short", {}).get("mean_monthly_pct", 0)

for tc_bps in [5, 10, 15, 20]:
    for turnover in [0.3, 0.5]:
        tc_cost = tc_bps / 10000 * 2 * turnover * 100  # round-trip × turnover, in %
        net = ls_gross - tc_cost
        print(f"  TC={tc_bps}bps, Turnover={turnover:.0%}: gross={ls_gross:+.2f}% - TC={tc_cost:.2f}% = net={net:+.2f}%/mo")

# ── STEP 6: Sub-sample Robustness ──
print(f"\n{'='*60}")
print("STEP 6: SUB-SAMPLE ROBUSTNESS")
print(f"{'='*60}")

subsamples = {
    "Pre-COVID (2005-2019)": panel[panel["month"] < "2020-01"],
    "Post-COVID (2020-2024)": panel[panel["month"] >= "2020-01"],
    "GFC (2007-2009)": panel[(panel["month"] >= "2007-10") & (panel["month"] <= "2009-03")],
    "Non-Crisis": panel[(panel["month"] < "2007-10") | (panel["month"] > "2009-03")],
    "2005-2009": panel[(panel["month"] >= "2005-01") & (panel["month"] < "2010-01")],
    "2010-2014": panel[(panel["month"] >= "2010-01") & (panel["month"] < "2015-01")],
    "2015-2019": panel[(panel["month"] >= "2015-01") & (panel["month"] < "2020-01")],
    "2020-2024": panel[(panel["month"] >= "2020-01") & (panel["month"] <= "2024-12")],
}

for label, sub in subsamples.items():
    n = len(sub)
    if n < 500:
        continue
    print(f"\n  {label} (n={n:,}):")
    for signal in ["JS_post", "H_sentiment", "D_post"]:
        res = fama_macbeth(sub, "excess_return", [signal])
        if signal in res:
            r = res[signal]
            print(f"    {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# ── Multiple Testing Correction ──
print(f"\n{'='*60}")
print("MULTIPLE TESTING CORRECTION")
print(f"{'='*60}")

all_pvals = []
test_names = []
for signal in signals:
    if signal in uni_results:
        all_pvals.append(uni_results[signal]["p_val"])
        test_names.append(f"Uni: {signal}")

for signal in multi_signals:
    if signal in multi_results:
        all_pvals.append(multi_results[signal]["p_val"])
        test_names.append(f"Multi: {signal}")

all_pvals = np.array(all_pvals)
n_tests = len(all_pvals)

# Bonferroni
bonf_pvals = np.minimum(all_pvals * n_tests, 1.0)

# Benjamini-Hochberg FDR
sorted_idx = np.argsort(all_pvals)
fdr_pvals = np.zeros(n_tests)
for i, idx in enumerate(sorted_idx):
    fdr_pvals[idx] = all_pvals[idx] * n_tests / (i + 1)
fdr_pvals = np.minimum.accumulate(fdr_pvals[sorted_idx[::-1]])[::-1]
fdr_pvals = np.minimum(fdr_pvals, 1.0)

print(f"Number of tests: {n_tests}")
print(f"\n{'Test':25s} {'Raw p':>8s} {'Bonf p':>8s} {'FDR p':>8s} {'Bonf Sig':>8s} {'FDR Sig':>8s}")
for i, name in enumerate(test_names):
    bonf_sig = "***" if bonf_pvals[i] < 0.01 else "**" if bonf_pvals[i] < 0.05 else "*" if bonf_pvals[i] < 0.1 else ""
    fdr_sig = "***" if fdr_pvals[i] < 0.01 else "**" if fdr_pvals[i] < 0.05 else "*" if fdr_pvals[i] < 0.1 else ""
    print(f"  {name:25s} {all_pvals[i]:8.4f} {bonf_pvals[i]:8.4f} {fdr_pvals[i]:8.4f} {bonf_sig:>8s} {fdr_sig:>8s}")

# ── Save everything ──
results_summary = {
    "panel_stats": {
        "n_obs": len(panel),
        "n_stocks": int(panel["ticker"].nunique()),
        "n_months": int(panel["month"].nunique()),
        "period": f"{panel['month'].min()} to {panel['month'].max()}",
    },
    "fm_univariate": {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv)) for kk, vv in v.items()} for k, v in uni_results.items()},
    "fm_multivariate": {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv)) for kk, vv in v.items()} for k, v in multi_results.items()},
    "fm_full_controls": {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv)) for kk, vv in v.items()} for k, v in full_results.items()},
    "decile_js": js_dec,
    "multiple_testing": {
        "n_tests": n_tests,
        "test_names": test_names,
        "raw_pvals": all_pvals.tolist(),
        "bonf_pvals": bonf_pvals.tolist(),
        "fdr_pvals": fdr_pvals.tolist(),
    },
}

with open(os.path.join(OUT, "analysis_results.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

panel.to_csv(os.path.join(OUT, "panel_data.csv"), index=False)

print(f"\n{'='*60}")
print(f"ALL ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Panel: {len(panel):,} obs → panel_data.csv")
print(f"Results → analysis_results.json")
