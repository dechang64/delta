#!/usr/bin/env python3
"""
AUDIT + FIX: Delta JFE Analysis Pipeline

Issues found and fixed:
  BUG1: rating_to_probs() in step2c uses piecewise linear mapping,
        but step3456 uses softmax mapping → INCONSISTENT JS values
  BUG2: Agent ratings use CURRENT month returns as features → LOOK-AHEAD BIAS
        (ret_1m = data[month]["return"] is the return we're trying to predict!)
  BUG3: FF5 alpha is computed using full-sample OLS → no out-of-sample
  BUG4: Newey-West lag=6 is arbitrary; should use int(T^(1/3))
  BUG5: Decile sort uses same-month return, not next-month → timing wrong
  BUG6: Fama-MacBeth p-values use normal approx; should use t-distribution
  BUG7: No standard controls (size, BM, momentum, volatility) in main regression

Author: Siyi / 2026-06-03 (audit)
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

ff5 = pd.read_csv(os.path.join(OUT, "ff5_factors.csv"))
mom = pd.read_csv(os.path.join(OUT, "ff_momentum.csv"))

print(f"Stocks: {len(stock_data)}, FF5: {len(ff5)} months")

# ══════════════════════════════════════════════════════════════
# FIX 1+2: Regenerate ratings with NO look-ahead bias
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FIX 1+2: Regenerating agent ratings (no look-ahead bias)")
print("="*60)

def rating_to_probs(rating):
    """Softmax mapping: rating 1-10 → [P(neg), P(neu), P(pos)]."""
    x = (rating - 5.5) / 2.0
    p_neg = np.exp(-x) / (np.exp(-x) + 1 + np.exp(x))
    p_neu = 1 / (np.exp(-x) + 1 + np.exp(x))
    p_pos = np.exp(x) / (np.exp(-x) + 1 + np.exp(x))
    return np.array([p_neg, p_neu, p_pos])

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

ratings = {}
for ticker in sorted(stock_data.keys()):
    data = stock_data[ticker]
    months = sorted(data.keys())
    ratings[ticker] = {}
    rng = np.random.RandomState(hash(ticker) % 2**31)
    
    for i, month in enumerate(months):
        # ── FIX 2: Only use PAST data (months 0..i-1) ──
        # Current month return is the DEPENDENT variable, cannot be a feature
        
        # Past returns (LAGGED)
        ret_1m = data[months[i-1]]["return"] if i >= 1 else 0
        ret_3m = np.prod([1 + data[months[i-k]]["return"] for k in range(1, min(4, i+1))]) - 1 if i >= 3 else 0
        ret_6m = np.prod([1 + data[months[i-k]]["return"] for k in range(1, min(7, i+1))]) - 1 if i >= 6 else 0
        ret_12m = np.prod([1 + data[months[i-k]]["return"] for k in range(1, min(13, i+1))]) - 1 if i >= 12 else 0
        
        # Past volatility (LAGGED)
        if i >= 7:
            past_rets_6m = [data[months[i-k]]["return"] for k in range(1, 7)]
            vol_6m = np.std(past_rets_6m)
        else:
            vol_6m = 0.05
        
        # Volume ratio (LAGGED)
        if i >= 4:
            curr_vol = data[months[i-1]].get("volume", 1)
            past_vols = [data[months[i-k]].get("volume", 1) for k in range(2, 5)]
            vol_ratio = curr_vol / (np.mean(past_vols) + 1)
        else:
            vol_ratio = 1.0
        
        # Skewness (LAGGED)
        if i >= 7:
            past_rets_6m = [data[months[i-k]]["return"] for k in range(1, 7)]
            skew_6m = float(pd.Series(past_rets_6m).skew())
        else:
            skew_6m = 0
        
        # Max drawdown (LAGGED)
        if i >= 7:
            past_rets_6m = [data[months[i-k]]["return"] for k in range(1, 7)]
            cum = np.cumprod([1+r for r in past_rets_6m])
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_dd = float(np.min(dd))
        else:
            max_dd = 0
        
        # ── Sentiment Agent: momentum + volume surprise ──
        s_signal = (
            0.35 * np.clip(ret_1m * 5, -2.5, 2.5) +
            0.25 * np.clip(ret_3m * 3, -2, 2) +
            0.20 * np.clip((vol_ratio - 1) * 2, -1.5, 1.5) +
            0.20 * np.clip(skew_6m, -1, 1)
        )
        s_raw = 5.0 + s_signal * 2.5
        s_noise = rng.normal(0, 1.2)
        s_rating = int(np.clip(round(s_raw + s_noise), 1, 10))
        
        # ── Technical Agent: trend + volatility + mean-reversion ──
        trend = ret_6m * 2.5
        vol_sig = -1.5 if vol_6m > 0.12 else (1.0 if vol_6m < 0.04 else 0)
        mr = -ret_1m * 4 if abs(ret_1m) > 0.08 else ret_1m * 1.5
        t_signal = (
            0.35 * np.clip(trend, -2.5, 2.5) +
            0.30 * vol_sig +
            0.35 * np.clip(mr, -2.5, 2.5)
        )
        t_raw = 5.0 + t_signal * 2.0
        t_noise = rng.normal(0, 0.9)
        t_rating = int(np.clip(round(t_raw + t_noise), 1, 10))
        
        # ── Fundamental Agent: value + quality + stability ──
        value = -ret_12m * 2.0
        quality = -vol_6m * 8
        stability = -max_dd * 3
        earnings = ret_3m * 0.8
        f_signal = (
            0.30 * np.clip(value, -2.5, 2.5) +
            0.30 * np.clip(quality, -2.5, 2.5) +
            0.25 * np.clip(stability, -2.5, 2.5) +
            0.15 * np.clip(earnings, -2, 2)
        )
        f_raw = 5.0 + f_signal * 1.8
        f_noise = rng.normal(0, 0.6)
        f_rating = int(np.clip(round(f_raw + f_noise), 1, 10))
        
        ratings[ticker][month] = {
            "sentiment": s_rating,
            "technical": t_rating,
            "fundamental": f_rating,
        }

# Save fixed ratings
with open(os.path.join(OUT, "agent_ratings_quant.json"), "w") as f:
    json.dump(ratings, f)
print(f"Ratings saved: {sum(len(v) for v in ratings.values()):,} obs (no look-ahead)")

# ══════════════════════════════════════════════════════════════
# Build panel with FIXED metrics
# ══════════════════════════════════════════════════════════════
print("\nBuilding panel...")
rows = []
for ticker in sorted(ratings.keys()):
    months_list = sorted(ratings[ticker].keys())
    for i, month in enumerate(months_list):
        r = ratings[ticker][month]
        s, t, f_ = r["sentiment"], r["technical"], r["fundamental"]
        
        # FIX 1: Use consistent softmax mapping
        sp = rating_to_probs(s)
        tp = rating_to_probs(t)
        fp = rating_to_probs(f_)
        avg_p = (sp + tp + fp) / 3
        uniform = np.array([1/3, 1/3, 1/3])
        
        js = js_divergence(avg_p, uniform)
        d_post = np.std([s, t, f_])
        h_sent = entropy(sp)
        conf = float(np.max(avg_p))
        
        # FIX 5: Use NEXT month return as dependent variable
        if i + 1 < len(months_list):
            next_month = months_list[i + 1]
            ret = stock_data.get(ticker, {}).get(next_month, {}).get("return", np.nan)
        else:
            ret = np.nan
        
        vol = stock_data.get(ticker, {}).get(month, {}).get("volume", np.nan)
        
        # FIX 7: Compute standard control variables
        # Size proxy: log volume
        log_vol = np.log1p(vol) if not np.isnan(vol) and vol > 0 else np.nan
        
        # Momentum: past 6-month return
        if i >= 6:
            mom_6m = np.prod([1 + stock_data[ticker][months_list[i-k]]["return"] 
                            for k in range(1, 7)]) - 1
        else:
            mom_6m = np.nan
        
        # Volatility: past 6-month std
        if i >= 6:
            past_rets = [stock_data[ticker][months_list[i-k]]["return"] for k in range(1, 7)]
            vol_6m = np.std(past_rets)
        else:
            vol_6m = np.nan
        
        # Book-to-market proxy: inverse of past 12-month return (value proxy)
        if i >= 12:
            ret_12m = np.prod([1 + stock_data[ticker][months_list[i-k]]["return"] 
                            for k in range(1, 13)]) - 1
            bm_proxy = 1 / (1 + ret_12m) if ret_12m > -0.9 else np.nan
        else:
            bm_proxy = np.nan
        
        rows.append({
            "ticker": ticker, "month": month,
            "return_next": ret,  # NEXT month return (dependent variable)
            "volume": vol, "log_volume": log_vol,
            "JS_post": js, "D_post": d_post, "H_sentiment": h_sent,
            "confidence": conf,
            "rating_s": s, "rating_t": t, "rating_f": f_,
            "mom_6m": mom_6m, "vol_6m": vol_6m, "bm_proxy": bm_proxy,
        })

panel = pd.DataFrame(rows)
panel = panel.dropna(subset=["return_next"])
print(f"Panel: {len(panel):,} obs (after dropping missing next-month returns)")

# Merge FF5
panel = panel.merge(ff5, on="month", how="left")
panel = panel.merge(mom, on="month", how="left")

# Excess return (next month)
panel["excess_return"] = panel["return_next"] - panel["RF"]

# Relative size
med_vol = panel.groupby("month")["log_volume"].transform("median")
panel["rel_size"] = panel["log_volume"] / med_vol

print(f"Panel with FF5: {len(panel):,} obs, excess_return mean={panel['excess_return'].mean()*100:.3f}%/mo")

# ══════════════════════════════════════════════════════════════
# FIX 3: Rolling FF5 alpha (36-month window, out-of-sample)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FIX 3: Rolling FF5 Alpha (36-month window)")
print("="*60)

factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
panel["ff5_alpha"] = np.nan
panel["ff5_adjusted_return"] = np.nan

for ticker in panel["ticker"].unique():
    stk = panel[panel["ticker"] == ticker].sort_values("month").reset_index(drop=True)
    if len(stk) < 48:
        continue
    
    for end_idx in range(36, len(stk)):
        window = stk.iloc[end_idx-36:end_idx]
        X = window[factor_cols].values
        X = np.column_stack([np.ones(len(X)), X])
        y = window["excess_return"].values
        
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            alpha = beta[0]
            
            # Out-of-sample: apply to next observation
            if end_idx < len(stk):
                next_idx = stk.index[end_idx]
                panel.loc[next_idx, "ff5_alpha"] = alpha
                panel.loc[next_idx, "ff5_adjusted_return"] = stk.loc[end_idx, "excess_return"] - alpha
        except Exception:
            continue

valid_alpha = panel["ff5_alpha"].dropna()
print(f"Rolling FF5 alpha: {len(valid_alpha):,} obs, mean={valid_alpha.mean()*100:.3f}%/mo")

# ══════════════════════════════════════════════════════════════
# FIX 4+6: Fama-MacBeth with optimal NW lag + t-distribution p-values
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FIX 4+6: Fama-MacBeth (optimal NW lag + t-dist p-values)")
print("="*60)

def newey_west_se(betas, lag=None):
    """Newey-West SE with automatic lag selection: lag = int(T^(1/3))."""
    T = len(betas)
    if T < 2:
        return 0, np.mean(betas)
    
    if lag is None:
        lag = int(T ** (1/3))  # FIX 4: automatic lag
    lag = min(lag, T - 1)
    
    mean_b = np.mean(betas)
    gamma0 = np.var(betas, ddof=1)
    
    nw_var = gamma0
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)  # Bartlett kernel
        gamma_j = np.mean((betas[j:] - mean_b) * (betas[:-j] - mean_b))
        nw_var += 2 * w * gamma_j
    
    nw_var = max(nw_var, 0)
    se = np.sqrt(nw_var / T)
    t_stat = mean_b / se if se > 0 else 0
    
    # FIX 6: Use t-distribution for p-values (not normal)
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=T-1))
    
    return se, t_stat, p_val

def fama_macbeth(panel, y_col, x_cols, nw_lag=None):
    """Fama-MacBeth with auto NW lag + t-dist p-values."""
    months = sorted(panel["month"].dropna().unique())
    betas = {col: [] for col in x_cols}
    n_months = 0
    
    for month in months:
        md = panel[panel["month"] == month].dropna(subset=[y_col] + x_cols)
        if len(md) < 10:
            continue
        X = np.column_stack([np.ones(len(md))] + [md[col].values for col in x_cols])
        y = md[y_col].values
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            for i, col in enumerate(x_cols):
                betas[col].append(beta[i + 1])
            n_months += 1
        except Exception:
            continue
    
    results = {}
    for col in x_cols:
        b = np.array(betas[col])
        if len(b) > 1:
            se, t_stat, p_val = newey_west_se(b, lag=nw_lag)
            mean_b = np.mean(b)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            results[col] = {"beta": mean_b, "se": se, "t_stat": t_stat, "p_val": p_val, "sig": sig, "n_months": n_months}
    
    return results

signals = ["JS_post", "D_post", "H_sentiment", "confidence"]

# ── Univariate ──
print("\n--- Univariate (Dep: Next-Month Excess Return) ---")
uni_results = {}
for signal in signals:
    res = fama_macbeth(panel, "excess_return", [signal])
    if signal in res:
        r = res[signal]
        uni_results[signal] = r
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  p={r['p_val']:.4f}  NW-SE={r['se']:.4f}")

# ── FF5-Adjusted ──
print("\n--- Univariate (Dep: FF5-Adjusted Return) ---")
for signal in signals:
    sub = panel.dropna(subset=["ff5_adjusted_return"])
    res = fama_macbeth(sub, "ff5_adjusted_return", [signal])
    if signal in res:
        r = res[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}  p={r['p_val']:.4f}")

# ── Multivariate ──
print("\n--- Multivariate ---")
multi_signals = ["JS_post", "D_post", "H_sentiment", "confidence"]
multi_results = fama_macbeth(panel, "excess_return", multi_signals)
for signal in multi_signals:
    if signal in multi_results:
        r = multi_results[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# ── FIX 7: With standard controls ──
print("\n--- Multivariate + Standard Controls (FIX 7) ---")
control_signals = multi_signals + ["rel_size", "bm_proxy", "mom_6m", "vol_6m"]
sub = panel.dropna(subset=control_signals + ["excess_return"])
full_results = fama_macbeth(sub, "excess_return", control_signals)
for signal in control_signals:
    if signal in full_results:
        r = full_results[signal]
        print(f"  {signal:18s}: β={r['beta']:+8.4f}  t={r['t_stat']:+7.2f}{r['sig']}")

# ══════════════════════════════════════════════════════════════
# Decile sorts (with NEXT month return)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DECILE PORTFOLIO SORTS (next-month returns)")
print("="*60)

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
                "mean_pct": mean_r * 100,
                "std_pct": std_r * 100,
                "t_stat": t_stat,
                "sharpe": sharpe,
                "n": len(rets),
            }
    return results

for signal_name, signal_col in [("JS_post", "JS_post"), ("H_sentiment", "H_sentiment"), ("D_post", "D_post")]:
    print(f"\n--- {signal_name} ---")
    dec = decile_sort(panel, signal_col)
    for name, r in dec.items():
        sig = "***" if abs(r["t_stat"]) > 2.58 else "**" if abs(r["t_stat"]) > 1.96 else "*" if abs(r["t_stat"]) > 1.64 else ""
        print(f"  {name:12s}: {r['mean_pct']:+6.2f}%/mo  t={r['t_stat']:+6.2f}{sig}  Sharpe={r['sharpe']:+5.2f}")

# Transaction costs
print("\n--- Transaction Costs ---")
js_dec = decile_sort(panel, "JS_post")
ls_gross = js_dec.get("Long-Short", {}).get("mean_pct", 0)
for tc_bps in [10, 20]:
    for turnover in [0.3, 0.5]:
        tc_cost = tc_bps / 10000 * 2 * turnover * 100
        net = ls_gross - tc_cost
        print(f"  TC={tc_bps}bps, TO={turnover:.0%}: gross={ls_gross:+.2f}% - TC={tc_cost:.2f}% = net={net:+.2f}%/mo")

# ══════════════════════════════════════════════════════════════
# Sub-sample robustness
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUB-SAMPLE ROBUSTNESS")
print("="*60)

subsamples = {
    "Pre-COVID (2005-2019)": panel[panel["month"] < "2020-01"],
    "Post-COVID (2020-2024)": panel[panel["month"] >= "2020-01"],
    "GFC (2007-2009)": panel[(panel["month"] >= "2007-10") & (panel["month"] <= "2009-03")],
    "Non-Crisis": panel[(panel["month"] < "2007-10") | (panel["month"] > "2009-03")],
    "2015-2019": panel[(panel["month"] >= "2015-01") & (panel["month"] < "2020-01")],
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

# ══════════════════════════════════════════════════════════════
# Multiple testing correction
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MULTIPLE TESTING CORRECTION")
print("="*60)

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

print(f"Tests: {n_tests}")
print(f"\n{'Test':25s} {'Raw p':>8s} {'Bonf p':>8s} {'FDR p':>8s} {'Bonf':>6s} {'FDR':>6s}")
for i, name in enumerate(test_names):
    bs = "***" if bonf_pvals[i] < 0.01 else "**" if bonf_pvals[i] < 0.05 else "*" if bonf_pvals[i] < 0.1 else ""
    fs = "***" if fdr_pvals[i] < 0.01 else "**" if fdr_pvals[i] < 0.05 else "*" if fdr_pvals[i] < 0.1 else ""
    print(f"  {name:25s} {all_pvals[i]:8.4f} {bonf_pvals[i]:8.4f} {fdr_pvals[i]:8.4f} {bs:>6s} {fs:>6s}")

# ══════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════
panel.to_csv(os.path.join(OUT, "panel_data.csv"), index=False)

# Convert results for JSON
def to_json_safe(d):
    return {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) 
                for kk, vv in v.items()} for k, v in d.items()}

results_summary = {
    "panel_stats": {
        "n_obs": len(panel),
        "n_stocks": int(panel["ticker"].nunique()),
        "n_months": int(panel["month"].nunique()),
        "period": f"{panel['month'].min()} to {panel['month'].max()}",
        "look_ahead_fixed": True,
        "next_month_return": True,
    },
    "fm_univariate": to_json_safe(uni_results),
    "fm_multivariate": to_json_safe(multi_results),
    "fm_with_controls": to_json_safe(full_results),
    "decile_js": js_dec,
    "multiple_testing": {
        "n_tests": n_tests,
        "raw_pvals": all_pvals.tolist(),
        "bonf_pvals": bonf_pvals.tolist(),
        "fdr_pvals": fdr_pvals.tolist(),
    },
}

with open(os.path.join(OUT, "analysis_results.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"AUDITED ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Panel: {len(panel):,} obs → panel_data.csv")
print(f"Results → analysis_results.json")
print(f"\nKey fixes applied:")
print(f"  ✅ FIX 1: Consistent softmax rating→prob mapping")
print(f"  ✅ FIX 2: No look-ahead bias (only past data as features)")
print(f"  ✅ FIX 3: Rolling 36-month FF5 alpha (out-of-sample)")
print(f"  ✅ FIX 4: Auto NW lag = int(T^(1/3))")
print(f"  ✅ FIX 5: Next-month return as dependent variable")
print(f"  ✅ FIX 6: t-distribution p-values (not normal)")
print(f"  ✅ FIX 7: Standard controls (size, BM, momentum, volatility)")
