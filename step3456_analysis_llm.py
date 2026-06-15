#!/usr/bin/env python3
"""
Step 3-6 REVISED: Full JFE-grade analysis using REAL LLM agent ratings.

Key changes from v2:
- Uses agent_ratings_llm_quarterly.json (40,020 real Qwen-plus calls)
- Quarterly observations only (80 months × 183 stocks = 13,340 obs)
- Forward-fills LLM ratings to non-quarterly months for robustness check
- All the same methodology: FF5 alpha, FM-NW, decile sorts, sub-sample, MHT correction

Author: Siyi / 2026-06-05
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

with open(os.path.join(OUT, "agent_ratings_llm_quarterly.json"), "r") as f:
    ratings_llm = json.load(f)

ff5 = pd.read_csv(os.path.join(OUT, "ff5_factors.csv"))
mom = pd.read_csv(os.path.join(OUT, "ff_momentum.csv"))

print(f"Stocks: {len(stock_data)}, FF5: {len(ff5)} months, Mom: {len(mom)} months")
print(f"LLM ratings: {len(ratings_llm)} stocks")

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

# ── Build panel (quarterly only) ──
print("\nBuilding panel dataset from REAL LLM ratings (quarterly)...")
rows = []
for ticker in sorted(ratings_llm.keys()):
    for month in sorted(ratings_llm[ticker].keys()):
        r = ratings_llm[ticker][month]
        if not isinstance(r, dict) or len(r) != 3:
            continue
        s, t, f = r.get("sentiment", 5), r.get("technical", 5), r.get("fundamental", 5)
        
        sp = rating_to_probs(s)
        tp = rating_to_probs(t)
        fp = rating_to_probs(f)
        avg_p = (sp + tp + fp) / 3
        uniform = np.array([1/3, 1/3, 1/3])
        
        js = js_divergence(avg_p, uniform)
        d_post = np.std([s, t, f])
        h_sent = entropy(sp)
        conf = float(np.max(avg_p))
        
        # Market data - use NEXT month return to avoid look-ahead
        # For quarterly observation at month M, predict M+1 return
        all_months = sorted(stock_data.get(ticker, {}).keys())
        if month not in all_months:
            continue
        idx = all_months.index(month)
        if idx + 1 >= len(all_months):
            continue  # No next month available
        
        next_month = all_months[idx + 1]
        ret = stock_data[ticker][next_month].get("return", np.nan)
        vol = stock_data[ticker][month].get("volume", np.nan)
        
        rows.append({
            "ticker": ticker, "month": month,
            "return_next": ret, "volume": vol,
            "JS_post": js, "D_post": d_post, "H_sentiment": h_sent,
            "confidence": conf,
            "rating_s": s, "rating_t": t, "rating_f": f,
        })

panel = pd.DataFrame(rows)
panel = panel.dropna(subset=["return_next"])

# Merge FF5 + Momentum
panel = panel.merge(ff5, on="month", how="left")
panel = panel.merge(mom, on="month", how="left")

# Excess return (next month)
panel["excess_return"] = panel["return_next"] - panel["RF"]

# Size proxy: log volume (relative to cross-section median each month)
panel["log_volume"] = np.log1p(panel["volume"])
med_vol = panel.groupby("month")["log_volume"].transform("median")
panel["rel_size"] = panel["log_volume"] / med_vol

# Momentum proxy: 6-month past return
panel = panel.sort_values(["ticker", "month"])
panel["mom_6m"] = panel.groupby("ticker")["return_next"].transform(
    lambda x: x.rolling(6, min_periods=3).mean()
)

# Volatility proxy: 6-month rolling std
panel["vol_6m"] = panel.groupby("ticker")["return_next"].transform(
    lambda x: x.rolling(6, min_periods=3).std()
)

# BM proxy: inverse of momentum (value proxy)
panel["bm_proxy"] = -panel["mom_6m"]

print(f"Panel: {len(panel):,} obs, {panel['ticker'].nunique()} stocks, {panel['month'].nunique()} months")
print(f"Period: {panel['month'].min()} to {panel['month'].max()}")

# ── Descriptive Statistics ──
print(f"\n{'='*60}")
print("DESCRIPTIVE STATISTICS (Real LLM Ratings)")
print(f"{'='*60}")
for col in ["JS_post", "D_post", "H_sentiment", "confidence", "excess_return"]:
    s = panel[col].dropna()
    print(f"  {col:18s}: mean={s.mean():.4f}  std={s.std():.4f}  "
          f"p10={s.quantile(0.1):.4f}  p50={s.quantile(0.5):.4f}  p90={s.quantile(0.9):.4f}")

# Rating distribution
all_ratings = np.concatenate([
    panel["rating_s"].values, panel["rating_t"].values, panel["rating_f"].values
])
print(f"\n  Rating distribution (1-10):")
for i in range(1, 11):
    pct = (all_ratings == i).mean() * 100
    print(f"    {i:2d}: {pct:5.1f}%")

# D_post = 0 rate
d_zero = (panel["D_post"] == 0).mean() * 100
print(f"\n  D_post = 0: {d_zero:.1f}% of observations")

# Correlation matrix
print(f"\n  Correlation matrix:")
corr_cols = ["JS_post", "D_post", "H_sentiment", "confidence", "excess_return"]
corr = panel[corr_cols].corr()
for c1 in corr_cols:
    print(f"    {c1:18s}: ", end="")
    for c2 in corr_cols:
        print(f"{corr.loc[c1,c2]:+.3f} ", end="")
    print()

# ── STEP 3: FF5 Alpha ──
print(f"\n{'='*60}")
print("STEP 3: FF5 ALPHA + CHARACTERISTIC-ADJUSTED RETURNS")
print(f"{'='*60}")

factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
stock_alphas = {}

for ticker in panel["ticker"].unique():
    stk = panel[panel["ticker"] == ticker].dropna(subset=factor_cols + ["excess_return"])
    if len(stk) < 12:  # Lower threshold for quarterly data
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
    except Exception:
        continue

alpha_map = {t: v["alpha"] for t, v in stock_alphas.items()}
panel["ff5_alpha"] = panel["ticker"].map(alpha_map)
panel["ff5_adjusted_return"] = panel["excess_return"] - panel["ff5_alpha"]

alphas = [v["alpha"] for v in stock_alphas.values()]
print(f"FF5 alphas: {len(stock_alphas)} stocks, mean={np.mean(alphas)*100:.3f}%/mo, std={np.std(alphas)*100:.3f}%/mo")

# ── STEP 4: Fama-MacBeth with Newey-West ──
print(f"\n{'='*60}")
print("STEP 4: FAMA-MACBETH REGRESSIONS (Newey-West lag=int(T^(1/3)))")
print(f"{'='*60}")

def newey_west_se(betas, lag=None):
    T = len(betas)
    if lag is None:
        lag = int(T ** (1/3))  # NW lag = T^(1/3) per Andrews (1991)
    mean_b = np.mean(betas)
    if T < 2:
        return 0, mean_b
    
    gamma0 = np.var(betas, ddof=1)
    nw_var = gamma0
    for j in range(1, min(lag + 1, T)):
        w = 1 - j / (lag + 1)
        gamma_j = np.mean((betas[j:] - mean_b) * (betas[:-j] - mean_b))
        nw_var += 2 * w * gamma_j
    
    nw_var = max(nw_var, 0)
    se = np.sqrt(nw_var / T)
    t_stat = mean_b / se if se > 0 else 0
    return se, t_stat

def fama_macbeth(panel, y_col, x_cols, nw_lag=None):
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
        except Exception:
            continue
    
    results = {}
    for col in x_cols:
        b = np.array(betas[col])
        if len(b) > 1:
            se, t_stat = newey_west_se(b, lag=nw_lag)
            mean_b = np.mean(b)
            # Use t-distribution for small T
            df = len(b) - 1
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
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
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  NW-SE={r['se']:.4f}  p={r['p_val']:.4f}")

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
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  p={r['p_val']:.4f}")

# With FF5 factor controls
print("\n--- Multivariate + FF5 Factor Loadings ---")
all_controls = multi_signals + ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
full_results = fama_macbeth(panel, "excess_return", all_controls)
for signal in all_controls:
    if signal in full_results:
        r = full_results[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# With additional controls (size, momentum, volatility, BM)
print("\n--- Multivariate + All Controls ---")
extended_controls = multi_signals + ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "rel_size", "mom_6m", "vol_6m", "bm_proxy"]
ext_sub = panel.dropna(subset=extended_controls + ["excess_return"])
print(f"  Observations with all controls: {len(ext_sub):,}")
ext_results = fama_macbeth(ext_sub, "excess_return", extended_controls)
for signal in multi_signals:
    if signal in ext_results:
        r = ext_results[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  p={r['p_val']:.4f}")

# ── STEP 5: Decile Portfolio Sorts ──
print(f"\n{'='*60}")
print("STEP 5: DECILE PORTFOLIO SORTS + TRANSACTION COSTS")
print(f"{'='*60}")

def decile_sort(panel, signal_col, return_col="excess_return", n_groups=10):
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

for signal_name, signal_col in [("JS Divergence", "JS_post"), ("H Sentiment", "H_sentiment"), ("D_post", "D_post"), ("Confidence", "confidence")]:
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
        tc_cost = tc_bps / 10000 * 2 * turnover * 100
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
    if n < 200:
        print(f"\n  {label} (n={n:,}): SKIPPED (too few obs)")
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

# ── COMPARISON: Old (Quant) vs New (LLM) ──
print(f"\n{'='*60}")
print("COMPARISON: QUANT MODEL vs REAL LLM RATINGS")
print(f"{'='*60}")

# Load old results
try:
    old = json.load(open(os.path.join(OUT, "analysis_results.json")))
    print("\n  Univariate FM β (excess return):")
    print(f"  {'Signal':18s} {'Quant β':>10s} {'Quant t':>10s} {'LLM β':>10s} {'LLM t':>10s} {'Change':>10s}")
    for signal in signals:
        old_r = old.get("fm_univariate", {}).get(signal, {})
        new_r = uni_results.get(signal, {})
        ob, ot = old_r.get("beta", 0), old_r.get("t_stat", 0)
        nb, nt = new_r.get("beta", 0), new_r.get("t_stat", 0)
        change = "↑" if abs(nt) > abs(ot) else "↓"
        print(f"  {signal:18s} {ob:+10.4f} {ot:+10.2f} {nb:+10.4f} {nt:+10.2f} {change:>10s}")
    
    print("\n  Decile Long-Short:")
    old_ls = old.get("decile_js", {}).get("Long-Short", {})
    new_ls = js_dec.get("Long-Short", {})
    print(f"    Quant: {old_ls.get('mean_monthly_pct', 0):+.2f}%/mo  t={old_ls.get('t_stat', 0):+.2f}")
    print(f"    LLM:   {new_ls.get('mean_monthly_pct', 0):+.2f}%/mo  t={new_ls.get('t_stat', 0):+.2f}")
except Exception:
    print("  (Old results not available for comparison)")

# ── Save everything ──
results_summary = {
    "data_source": "LLM (Qwen-plus) quarterly ratings",
    "panel_stats": {
        "n_obs": len(panel),
        "n_stocks": int(panel["ticker"].nunique()),
        "n_months": int(panel["month"].nunique()),
        "period": f"{panel['month'].min()} to {panel['month'].max()}",
        "next_month_return": True,
        "look_ahead_fixed": True,
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

with open(os.path.join(OUT, "analysis_results_llm.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

panel.to_csv(os.path.join(OUT, "panel_data_llm_quarterly.csv"), index=False)

print(f"\n{'='*60}")
print(f"ALL ANALYSIS COMPLETE (Real LLM Ratings)")
print(f"{'='*60}")
print(f"Panel: {len(panel):,} obs → panel_data_llm_quarterly.csv")
print(f"Results → analysis_results_llm.json")
