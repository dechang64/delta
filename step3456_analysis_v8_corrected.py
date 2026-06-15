#!/usr/bin/env python3
"""
Step 3-6 v8: CORRECTED JFE-grade analysis using REAL LLM agent ratings.

Bug fixes from v7:
  #1 FATAL:   H_sentiment → H_smooth (entropy of avg belief dist, ALL 3 agents)
  #2 CRITICAL: mom_6m/vol_6m look-ahead → use PAST returns (shift 1 month)
  #5 MODERATE: JS divergence → proper JS = H(avg_p) - avg(H(individual_p))
  #7 MODERATE: bm_proxy = -mom → bm_proxy = -past_12m_return
  #8 MODERATE: size = volume → documented as volume-based proxy

Author: Siyi / 2026-06-08
"""

import json
import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import entropy
from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression

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
    """Convert 1-10 rating to 3-bin probability [neg, neu, pos] via softmax."""
    x = (rating - 5.5) / 2.0
    p_neg = np.exp(-x) / (np.exp(-x) + 1 + np.exp(x))
    p_neu = 1 / (np.exp(-x) + 1 + np.exp(x))
    p_pos = np.exp(x) / (np.exp(-x) + 1 + np.exp(x))
    return np.array([p_neg, p_neu, p_pos])

# ── FIX #1: H_smooth = entropy of average belief distribution ──
# OLD (BUG): h_sent = entropy(sp)  — only sentiment agent, binary_entropy(rating/10)
# NEW (FIX): H_smooth = entropy(avg_probs, base=2) — all 3 agents, proper Shannon entropy

# ── FIX #5: JS_correct = H(avg_p) - avg(H(individual_p)) ──
# OLD (BUG): js = js_divergence(avg_p, uniform) — distance from uniform, not agent disagreement
# NEW (FIX): JS = H(avg_p) - mean(H(sp), H(tp), H(fp)) — proper Jensen-Shannon decomposition

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
        
        # FIX #1: H_smooth = Shannon entropy of average belief distribution
        h_smooth = entropy(avg_p, base=2)
        
        # FIX #5: JS_correct = H(avg) - avg(H(individual))
        h_sp = entropy(sp, base=2)
        h_tp = entropy(tp, base=2)
        h_fp = entropy(fp, base=2)
        js_correct = h_smooth - np.mean([h_sp, h_tp, h_fp])
        
        # D_post: standard deviation of ratings (unchanged, was correct)
        d_post = np.std([s, t, f])
        
        # Confidence: max probability of average distribution
        conf = float(np.max(avg_p))
        
        # Market data - use NEXT month return (no look-ahead: rating at M predicts M+1)
        all_months = sorted(stock_data.get(ticker, {}).keys())
        if month not in all_months:
            continue
        idx = all_months.index(month)
        if idx + 1 >= len(all_months):
            continue
        
        next_month = all_months[idx + 1]
        ret = stock_data[ticker][next_month].get("return", np.nan)
        vol = stock_data[ticker][month].get("volume", np.nan)
        
        rows.append({
            "ticker": ticker, "month": month,
            "return_next": ret, "volume": vol,
            "H_smooth": h_smooth, "JS_correct": js_correct, "D_post": d_post,
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

# ── FIX #2: Compute mom/vol from PAST returns (no look-ahead) ──
# Build full monthly return series for each stock
all_ret_rows = []
for ticker, months in stock_data.items():
    for m, data in months.items():
        if isinstance(data, dict) and "return" in data:
            all_ret_rows.append({"ticker": ticker, "month": m, "return": data["return"]})
ret_df = pd.DataFrame(all_ret_rows).sort_values(["ticker", "month"]).reset_index(drop=True)

# Past return = lagged 1 month (to avoid look-ahead)
ret_df["past_return"] = ret_df.groupby("ticker")["return"].shift(1)
ret_df["mom_6m"] = ret_df.groupby("ticker")["past_return"].transform(
    lambda x: x.rolling(6, min_periods=3).mean())
ret_df["vol_6m"] = ret_df.groupby("ticker")["past_return"].transform(
    lambda x: x.rolling(6, min_periods=3).std())

# ── FIX #7: bm_proxy = -past_12m_return (not -mom_6m) ──
ret_df["past_12m_ret"] = ret_df.groupby("ticker")["past_return"].transform(
    lambda x: x.rolling(12, min_periods=6).sum())
ret_df["bm_proxy"] = -ret_df["past_12m_ret"]

# ── FIX #8: Document size = volume proxy ──
ret_df["log_volume"] = np.log1p(ret_df.get("volume", 0))  # may not have volume here
# Use panel's volume for size
panel["log_volume"] = np.log1p(panel["volume"])
med_vol = panel.groupby("month")["log_volume"].transform("median")
panel["rel_size"] = panel["log_volume"] / med_vol

# Merge corrected controls into panel
panel = panel.merge(
    ret_df[["ticker", "month", "mom_6m", "vol_6m", "bm_proxy"]].dropna(subset=["mom_6m"]),
    on=["ticker", "month"], how="left"
)

# Residualize H_smooth on JS_correct and D_post
X_resid = panel[["JS_correct", "D_post"]].values
y_resid = panel["H_smooth"].values
reg_resid = LinearRegression().fit(X_resid, y_resid)
panel["H_smooth_resid"] = y_resid - reg_resid.predict(X_resid)

# Standardize key variables
for col in ["H_smooth", "H_smooth_resid", "JS_correct", "D_post", "confidence",
            "rel_size", "mom_6m", "vol_6m", "bm_proxy"]:
    panel[f"{col}_z"] = (panel[col] - panel[col].mean()) / panel[col].std()

# Interaction terms
panel["HxD"] = panel["H_smooth_z"] * panel["D_post_z"]
panel["year"] = panel["month"].str[:4].astype(int)

# Multi-period returns for horizon analysis
panel = panel.sort_values(["ticker", "month"]).reset_index(drop=True)
panel["ret_2q"] = panel.groupby("ticker")["return_next"].shift(-1)
panel["ret_3q"] = panel.groupby("ticker")["return_next"].shift(-2)
panel["cum_2q"] = panel["return_next"] + panel["ret_2q"].fillna(0)
panel["cum_3q"] = panel["cum_2q"] + panel["ret_3q"].fillna(0)

# Reverse causality control
panel["ret_lag1"] = panel.groupby("ticker")["return_next"].shift(1)
panel["ret_lag1_z"] = (panel["ret_lag1"] - panel["ret_lag1"].mean()) / panel["ret_lag1"].std()

print(f"Panel: {len(panel):,} obs, {panel['ticker'].nunique()} stocks, {panel['month'].nunique()} months")
print(f"Period: {panel['month'].min()} to {panel['month'].max()}")

# ── Descriptive Statistics ──
print(f"\n{'='*60}")
print("DESCRIPTIVE STATISTICS (Corrected Measures)")
print(f"{'='*60}")
for col in ["H_smooth", "H_smooth_resid", "JS_correct", "D_post", "confidence", "excess_return"]:
    s = panel[col].dropna()
    print(f"  {col:20s}: mean={s.mean():.4f}  std={s.std():.4f}  "
          f"p10={s.quantile(0.1):.4f}  p50={s.quantile(0.5):.4f}  p90={s.quantile(0.9):.4f}")

# Correlation matrix
print(f"\n  Correlation matrix:")
corr_cols = ["H_smooth", "JS_correct", "D_post", "confidence", "excess_return"]
corr = panel[corr_cols].corr()
for c1 in corr_cols:
    print(f"    {c1:20s}: ", end="")
    for c2 in corr_cols:
        print(f"{corr.loc[c1,c2]:+.3f} ", end="")
    print()

# ── STEP 3: FF5 Alpha (rolling 36-month window) ──
print(f"\n{'='*60}")
print("STEP 3: FF5 ALPHA (Rolling 36-month window)")
print(f"{'='*60}")

factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
panel = panel.sort_values(["ticker", "month"]).reset_index(drop=True)

# Rolling FF5 alpha
alphas = []
for ticker, grp in panel.groupby("ticker"):
    grp = grp.sort_values("month").reset_index(drop=True)
    ticker_alphas = [np.nan] * len(grp)
    for i in range(35, len(grp)):
        window = grp.iloc[max(0, i-35):i+1].dropna(subset=factor_cols + ["excess_return"])
        if len(window) < 24:
            continue
        try:
            X = window[factor_cols].values
            X = np.column_stack([np.ones(len(X)), X])
            y = window["excess_return"].values
            beta, _, _, _ = lstsq(X, y, rcond=None)
            ticker_alphas[i] = beta[0]
        except Exception:
            continue
    alphas.extend(ticker_alphas)

panel["ff5_alpha_rolling"] = alphas
panel["ff5_adj_return_rolling"] = panel["excess_return"] - panel["ff5_alpha_rolling"]

valid_alpha = panel["ff5_alpha_rolling"].dropna()
print(f"Rolling FF5 alphas: {len(valid_alpha):,} obs, mean={valid_alpha.mean()*100:.3f}%/mo")

# ── STEP 4: Fama-MacBeth with Newey-West ──
print(f"\n{'='*60}")
print("STEP 4: FAMA-MACBETH REGRESSIONS (OLS with NW-SE)")
print(f"{'='*60}")

def newey_west_se(betas, lag=None):
    T = len(betas)
    if lag is None:
        lag = int(T ** (1/3))
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

def fama_macbeth(panel, y_col, x_cols, nw_lag=None, min_obs=10):
    """Standard OLS Fama-MacBeth with Newey-West standard errors."""
    months = sorted(panel["month"].dropna().unique())
    betas = {col: [] for col in x_cols}
    n_months = 0
    
    for month in months:
        md = panel[panel["month"] == month].dropna(subset=[y_col] + x_cols)
        if len(md) < min_obs:
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
            se, t_stat = newey_west_se(b, lag=nw_lag)
            mean_b = np.mean(b)
            df = len(b) - 1
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            results[col] = {"beta": mean_b, "se": se, "t_stat": t_stat, "p_val": p_val, "sig": sig, "n_months": n_months}
    
    return results

def print_fm(results, label=""):
    if label:
        print(f"\n  {label}:")
    for col, r in results.items():
        print(f"    {col:25s}: β={r['beta']:+.5f}  t={r['t_stat']:+.2f}{r['sig']}  p={r['p_val']:.4f}")

# Univariate
print("\n--- Univariate FM (Dep: Excess Return) ---")
uni_results = {}
for signal, name in [("H_smooth_z", "H_smooth"), ("H_smooth_resid_z", "H_resid"),
                      ("JS_correct_z", "JS_correct"), ("D_post_z", "D_post"), 
                      ("confidence_z", "Confidence")]:
    res = fama_macbeth(panel, "excess_return", [signal])
    if signal in res:
        uni_results[name] = res[signal]
        r = res[signal]
        print(f"  {name:20s}: β={r['beta']:+.5f}  t={r['t_stat']:+.2f}{r['sig']}  p={r['p_val']:.4f}")

# Multivariate specifications
print("\n--- Multivariate FM ---")
specs = {
    "(1) H+D+JS": ["H_smooth_z", "D_post_z", "JS_correct_z"],
    "(2) H_resid": ["H_smooth_resid_z"],
    "(3) +Controls": ["H_smooth_resid_z", "D_post_z", "JS_correct_z",
                       "rel_size_z", "bm_proxy_z", "mom_6m_z", "vol_6m_z"],
    "(4) +H×D": ["H_smooth_resid_z", "D_post_z", "JS_correct_z", "HxD",
                  "rel_size_z", "bm_proxy_z", "mom_6m_z", "vol_6m_z"],
}

multi_results = {}
for spec_name, vars_list in specs.items():
    sub = panel.dropna(subset=[v for v in vars_list])
    res = fama_macbeth(sub, "excess_return", vars_list)
    multi_results[spec_name] = res
    print_fm(res, spec_name)

# H×D interaction (univariate)
print("\n--- H×D Interaction (Univariate) ---")
hxd_res = fama_macbeth(panel, "excess_return", ["HxD"])
print_fm(hxd_res)

# ── STEP 5: Horizon Analysis ──
print(f"\n{'='*60}")
print("STEP 5: HORIZON ANALYSIS (1Q, 2Q, 3Q)")
print(f"{'='*60}")

for horizon, col in [("1Q", "return_next"), ("2Q", "cum_2q"), ("3Q", "cum_3q")]:
    sub = panel.dropna(subset=[col])
    print(f"\n  {horizon}:")
    for var, name in [("H_smooth_resid_z", "H_resid"), ("HxD", "H×D")]:
        res = fama_macbeth(sub, col, [var])
        if var in res:
            r = res[var]
            print(f"    {name:10s}: β={r['beta']:+.5f}  t={r['t_stat']:+.2f}{r['sig']}")

# ── STEP 6: Sub-sample Robustness ──
print(f"\n{'='*60}")
print("STEP 6: SUB-SAMPLE ROBUSTNESS")
print(f"{'='*60}")

subsamples = {
    "2005-2009": panel[panel["year"].between(2005, 2009)],
    "2010-2014": panel[panel["year"].between(2010, 2014)],
    "2015-2019": panel[panel["year"].between(2015, 2019)],
    "2020-2024": panel[panel["year"].between(2020, 2024)],
    "Pre-COVID": panel[panel["month"] < "2020-01"],
    "Post-COVID": panel[panel["month"] >= "2020-01"],
}

for label, sub in subsamples.items():
    n = len(sub)
    if n < 200:
        continue
    print(f"\n  {label} (n={n:,}):")
    for signal in ["H_smooth_resid_z", "HxD"]:
        res = fama_macbeth(sub, "excess_return", [signal])
        if signal in res:
            r = res[signal]
            print(f"    {signal.replace('_z',''):20s}: t={r['t_stat']:+.2f}{r['sig']}")

# ── STEP 7: Size Groups ──
print(f"\n{'='*60}")
print("STEP 7: SIZE GROUPS (Volume-based proxy)")
print(f"{'='*60}")

panel["size_group"] = pd.qcut(panel["rel_size"], 3, labels=["Small", "Medium", "Large"], duplicates="drop")
for grp in ["Small", "Medium", "Large"]:
    sub = panel[panel["size_group"] == grp]
    res = fama_macbeth(sub, "excess_return", ["H_smooth_resid_z"])
    if "H_smooth_resid_z" in res:
        r = res["H_smooth_resid_z"]
        print(f"  {grp:8s} (n={len(sub):,}): t={r['t_stat']:+.2f}{r['sig']}")

# ── STEP 8: Arbitrage Constraints ──
print(f"\n{'='*60}")
print("STEP 8: ARBITRAGE CONSTRAINT INTERACTIONS")
print(f"{'='*60}")

for ctrl, name in [("rel_size_z", "Size"), ("vol_6m_z", "Vol"), ("bm_proxy_z", "BM")]:
    panel[f"H_x_{name}"] = panel["H_smooth_resid_z"] * panel[ctrl]
    sub = panel.dropna(subset=[ctrl])
    res = fama_macbeth(sub, "excess_return", ["H_smooth_resid_z", f"H_x_{name}"])
    if f"H_x_{name}" in res:
        r = res[f"H_x_{name}"]
        print(f"  H×{name:6s}: t={r['t_stat']:+.2f}{r['sig']}")

# ── STEP 9: Reverse Causality ──
print(f"\n{'='*60}")
print("STEP 9: REVERSE CAUSALITY DIAGNOSTICS")
print(f"{'='*60}")

sub = panel.dropna(subset=["ret_lag1_z", "mom_6m_z"])
res = fama_macbeth(sub, "excess_return", ["H_smooth_resid_z", "ret_lag1_z", "mom_6m_z"])
if "H_smooth_resid_z" in res:
    r = res["H_smooth_resid_z"]
    print(f"  H_resid (ctrl past ret + mom): t={r['t_stat']:+.2f}{r['sig']}")

# ── STEP 10: Economic Significance ──
print(f"\n{'='*60}")
print("STEP 10: ECONOMIC SIGNIFICANCE")
print(f"{'='*60}")

res = fama_macbeth(panel, "excess_return", ["H_smooth_resid_z"])
if "H_smooth_resid_z" in res:
    beta = res["H_smooth_resid_z"]["beta"]
    print(f"  1σ H_smooth_resid → {beta*100:+.3f}%/month = {beta*100*12:+.3f}%/year")

res = fama_macbeth(panel, "excess_return", ["HxD"])
if "HxD" in res:
    beta = res["HxD"]["beta"]
    print(f"  1σ H×D → {beta*100:+.3f}%/month = {beta*100*12:+.3f}%/year")

# ── STEP 11: Quant vs LLM Comparison ──
print(f"\n{'='*60}")
print("STEP 11: QUANT MODEL PLACEBO")
print(f"{'='*60}")

try:
    quant_panel = pd.read_csv(os.path.join(OUT, "quant_llm_comparison.csv"))
    if "H_smooth_resid_z" in quant_panel.columns:
        res = fama_macbeth(quant_panel, "return_next", ["H_smooth_resid_z"])
        if "H_smooth_resid_z" in res:
            r = res["H_smooth_resid_z"]
            print(f"  Quant H_resid: t={r['t_stat']:+.2f}{r['sig']}")
    else:
        print("  (Quant panel missing H_smooth_resid_z)")
except Exception:
    print("  (Quant comparison data not available)")

# ── Save ──
panel.to_csv(os.path.join(OUT, "panel_v8_corrected.csv"), index=False)

# Summary JSON
summary = {
    "version": "v8_corrected",
    "bugs_fixed": ["#1 H_smooth", "#2 look-ahead mom/vol", "#5 JS_correct", "#7 bm_proxy", "#8 size documented"],
    "univariate": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                       for kk, vv in v.items()} for k, v in uni_results.items()},
    "hxd_univariate": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in hxd_res.get("HxD", {}).items()},
}

with open(os.path.join(OUT, "analysis_results_v8.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"ALL ANALYSIS COMPLETE (v8 Corrected)")
print(f"{'='*60}")
print(f"Panel: {len(panel):,} obs → panel_v8_corrected.csv")
print(f"Results → analysis_results_v8.json")
