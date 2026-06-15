#!/usr/bin/env python3
"""
Step 7: IBES Analyst Dispersion Proxy + Visualization
Step 8: Generate all figures for JFE paper

Since we don't have WRDS/IBES access, we construct a proxy using:
- Cross-sectional return dispersion (CRSP-based)
- Earnings surprise volatility
- Compare with our LLM disagreement measures

Author: Siyi / 2026-06-03
"""

import json
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import entropy

OUT = "/home/z/my-project/delta_jfe"
FIGS = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)

# ── Load panel ──
panel = pd.read_csv(os.path.join(OUT, "panel_data.csv"))
print(f"Panel: {len(panel):,} obs")

# ── IBES Proxy ──
print(f"\n{'='*60}")
print("STEP 7: ANALYST DISPERSION PROXY")
print(f"{'='*60}")

# Cross-sectional return dispersion as proxy for analyst disagreement
# (Diether et al. 2002 use IBES dispersion; we use realized return dispersion)
# For each stock-month, compute rolling std of daily-equivalent returns
# Since we have monthly data, use rolling 6-month return std as proxy

panel = panel.sort_values(["ticker", "month"])
panel["ret_std_6m"] = panel.groupby("ticker")["return"].transform(
    lambda x: x.rolling(6, min_periods=3).std()
)
panel["ret_std_12m"] = panel.groupby("ticker")["return"].transform(
    lambda x: x.rolling(12, min_periods=6).std()
)

# Earnings surprise proxy: |return| / std
panel["earnings_surprise"] = panel["return"].abs() / panel["ret_std_6m"].replace(0, np.nan)

# Analyst dispersion proxy = cross-sectional rank of ret_std_6m
panel["analyst_disp_proxy"] = panel.groupby("month")["ret_std_6m"].transform(
    lambda x: x.rank(pct=True)
)

print(f"Analyst dispersion proxy computed")
print(f"  ret_std_6m: mean={panel['ret_std_6m'].mean()*100:.2f}% std={panel['ret_std_6m'].std()*100:.2f}%")
print(f"  analyst_disp_proxy: mean={panel['analyst_disp_proxy'].mean():.3f} std={panel['analyst_disp_proxy'].std():.3f}")

# Correlation between our measures and the proxy
from numpy.linalg import lstsq

def fama_macbeth_nw(panel, y_col, x_cols, nw_lag=6):
    months = sorted(panel["month"].dropna().unique())
    betas = {col: [] for col in x_cols}
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
        except Exception:
            continue
    
    results = {}
    for col in x_cols:
        b = np.array(betas[col])
        if len(b) > 1:
            mean_b = np.mean(b)
            T = len(b)
            gamma0 = np.var(b, ddof=1)
            nw_var = gamma0
            for j in range(1, min(nw_lag + 1, T)):
                w = 1 - j / (nw_lag + 1)
                gamma_j = np.mean((b[j:] - mean_b) * (b[:-j] - mean_b))
                nw_var += 2 * w * gamma_j
            se = np.sqrt(max(nw_var, 0) / T)
            t_stat = mean_b / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            results[col] = {"beta": mean_b, "t_stat": t_stat, "sig": sig}
    return results

# Compare: does analyst_disp_proxy predict returns?
print("\n--- Analyst Dispersion Proxy → Returns ---")
sub = panel.dropna(subset=["analyst_disp_proxy", "excess_return"])
res = fama_macbeth_nw(sub, "excess_return", ["analyst_disp_proxy"])
for col, r in res.items():
    print(f"  {col}: β={r['beta']:+.4f} t={r['t_stat']:+.2f}{r['sig']}")

# Correlation between JS and analyst proxy
corr_js_proxy = panel[["JS_post", "analyst_disp_proxy"]].dropna().corr().iloc[0, 1]
corr_d_proxy = panel[["D_post", "analyst_disp_proxy"]].dropna().corr().iloc[0, 1]
corr_h_proxy = panel[["H_sentiment", "analyst_disp_proxy"]].dropna().corr().iloc[0, 1]
print(f"\nCorrelation with analyst dispersion proxy:")
print(f"  JS_post:     r={corr_js_proxy:+.3f}")
print(f"  D_post:      r={corr_d_proxy:+.3f}")
print(f"  H_sentiment: r={corr_h_proxy:+.3f}")

# ── STEP 8: Generate Figures ──
print(f"\n{'='*60}")
print("STEP 8: GENERATE FIGURES")
print(f"{'='*60}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Figure 1: Decile Portfolio Returns (3 panels) ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (signal_name, signal_col, color) in [
    (axes[0], ("JS Divergence", "JS_post", "#2166ac")),
    (axes[1], ("H Sentiment", "H_sentiment", "#b2182b")),
    (axes[2], ("D_post (Rating Dispersion)", "D_post", "#4dac26")),
]:
    months = sorted(panel["month"].dropna().unique())
    decile_rets = {f"D{i+1}": [] for i in range(10)}
    ls_rets = []
    
    for month in months:
        md = panel[panel["month"] == month].dropna(subset=[signal_col, "excess_return"])
        if len(md) < 20:
            continue
        md = md.copy()
        ranks = md[signal_col].rank(method="average")
        n = len(ranks)
        md["decile"] = np.minimum(np.floor((ranks - 1) / n * 10).astype(int) + 1, 10)
        
        for d in range(1, 11):
            dr = md[md["decile"] == d]["excess_return"]
            if len(dr) > 0:
                decile_rets[f"D{d}"].append(dr.mean())
        
        d10 = md[md["decile"] == 10]["excess_return"]
        d1 = md[md["decile"] == 1]["excess_return"]
        if len(d10) > 0 and len(d1) > 0:
            ls_rets.append(d10.mean() - d1.mean())
    
    means = [np.mean(decile_rets[f"D{i+1}"]) * 100 for i in range(10)]
    ses = [np.std(decile_rets[f"D{i+1}"], ddof=1) / np.sqrt(len(decile_rets[f"D{i+1}"])) * 100 for i in range(10)]
    
    x = range(1, 11)
    bars = ax.bar(x, means, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, means, yerr=[1.96 * s for s in ses], fmt='none', color='black', capsize=3, linewidth=0.8)
    
    # Long-short
    ls_mean = np.mean(ls_rets) * 100
    ls_se = np.std(ls_rets, ddof=1) / np.sqrt(len(ls_rets)) * 100
    ls_t = np.mean(ls_rets) / (np.std(ls_rets, ddof=1) / np.sqrt(len(ls_rets)))
    ls_sig = "***" if abs(ls_t) > 2.58 else "**" if abs(ls_t) > 1.96 else "*" if abs(ls_t) > 1.64 else ""
    
    ax.bar([11.5], [ls_mean], color='grey', alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
    ax.errorbar([11.5], [ls_mean], yerr=[1.96 * ls_se], fmt='none', color='black', capsize=3, linewidth=0.8)
    ax.text(11.5, ls_mean + 0.15, f"L-S{ls_sig}", ha='center', fontsize=8)
    
    ax.set_xlabel('Decile (Low → High)')
    ax.set_ylabel('Monthly Excess Return (%)')
    ax.set_title(signal_name)
    ax.set_xticks(list(range(1, 11)) + [11.5])
    ax.set_xticklabels([str(i) for i in range(1, 11)] + ['L-S'], fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_decile_returns.png"))
print("  fig1_decile_returns.png")

# ── Figure 2: FM Regression Coefficients ──
fig, ax = plt.subplots(figsize=(8, 5))

# Univariate
signals_plot = ["JS_post", "D_post", "H_sentiment", "confidence"]
labels = ["JS Divergence", "D_post", "H_sentiment", "Confidence"]
uni_betas = []
uni_tstats = []

for signal in signals_plot:
    res = fama_macbeth_nw(panel, "excess_return", [signal])
    if signal in res:
        uni_betas.append(res[signal]["beta"])
        uni_tstats.append(res[signal]["t_stat"])
    else:
        uni_betas.append(0)
        uni_tstats.append(0)

colors = ['#2166ac' if b > 0 else '#b2182b' for b in uni_betas]
x = np.arange(len(signals_plot))
bars = ax.bar(x - 0.15, uni_betas, 0.3, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5, label='Univariate')

# Add significance stars
for i, (b, t) in enumerate(zip(uni_betas, uni_tstats)):
    sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
    ax.text(i - 0.15, b + (0.005 if b > 0 else -0.005), sig, ha='center', fontsize=9)

# Multivariate
multi_betas = []
multi_tstats = []
multi_signals = ["JS_post", "D_post", "H_sentiment", "confidence"]
res = fama_macbeth_nw(panel, "excess_return", multi_signals)
for signal in multi_signals:
    if signal in res:
        multi_betas.append(res[signal]["beta"])
        multi_tstats.append(res[signal]["t_stat"])
    else:
        multi_betas.append(0)
        multi_tstats.append(0)

colors2 = ['#6baed6' if b > 0 else '#fc9272' for b in multi_betas]
bars2 = ax.bar(x + 0.15, multi_betas, 0.3, color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5, label='Multivariate')

for i, (b, t) in enumerate(zip(multi_betas, multi_tstats)):
    sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
    ax.text(i + 0.15, b + (0.005 if b > 0 else -0.005), sig, ha='center', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Fama-MacBeth Coefficient (β)')
ax.set_title('Cross-Sectional Return Predictability of Disagreement Signals')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_fm_coefficients.png"))
print("  fig2_fm_coefficients.png")

# ── Figure 3: Time-Series of JS-return Relationship ──
fig, ax = plt.subplots(figsize=(10, 5))

months_sorted = sorted(panel["month"].dropna().unique())
monthly_js_beta = []
monthly_dates = []

for month in months_sorted:
    md = panel[panel["month"] == month].dropna(subset=["JS_post", "excess_return"])
    if len(md) < 15:
        continue
    try:
        slope, intercept, r, p, se = stats.linregress(md["JS_post"], md["excess_return"])
        monthly_js_beta.append(slope)
        monthly_dates.append(month)
    except Exception:
        continue

# Rolling 24-month average
rolling_beta = pd.Series(monthly_js_beta).rolling(24, min_periods=12).mean()

ax.plot(range(len(monthly_js_beta)), monthly_js_beta, alpha=0.3, color='#2166ac', linewidth=0.5)
ax.plot(range(len(rolling_beta)), rolling_beta, color='#2166ac', linewidth=2, label='24-month rolling avg')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

# Mark key periods
for start, end, label, color in [("2007-10", "2009-03", "GFC", "#fee0d2"), 
                                   ("2020-02", "2020-06", "COVID", "#deebf7")]:
    si = monthly_dates.index(start) if start in monthly_dates else None
    ei = monthly_dates.index(end) if end in monthly_dates else None
    if si and ei:
        ax.axvspan(si, ei, alpha=0.3, color=color, label=label)

ax.set_xlabel('Month')
ax.set_ylabel('JS_post → Return β (monthly)')
ax.set_title('Time-Varying JS Divergence–Return Relationship')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# X-axis labels
tick_idx = list(range(0, len(monthly_dates), 24))
ax.set_xticks(tick_idx)
ax.set_xticklabels([monthly_dates[i] for i in tick_idx], rotation=45, fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_js_timevarying.png"))
print("  fig3_js_timevarying.png")

# ── Figure 4: Correlation Heatmap ──
fig, ax = plt.subplots(figsize=(7, 6))

corr_cols = ["JS_post", "D_post", "H_sentiment", "confidence", "analyst_disp_proxy", "excess_return", "ret_std_6m"]
corr_labels = ["JS Divergence", "D_post", "H Sentiment", "Confidence", "Analyst Disp. Proxy", "Excess Return", "Return Vol"]
corr_matrix = panel[corr_cols].dropna().corr()

im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_labels, fontsize=9)

for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson Correlation')
ax.set_title('Correlation Structure of Disagreement Measures')

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_correlation_heatmap.png"))
print("  fig4_correlation_heatmap.png")

# ── Figure 5: Sub-sample Robustness ──
fig, ax = plt.subplots(figsize=(10, 5))

sub_labels = ["2005-2009", "2010-2014", "2015-2019", "2020-2024"]
sub_ranges = [("2005-01", "2009-12"), ("2010-01", "2014-12"), ("2015-01", "2019-12"), ("2020-01", "2024-12")]
signals_sub = ["JS_post", "H_sentiment", "D_post"]
colors_sub = ["#2166ac", "#b2182b", "#4dac26"]

x = np.arange(len(sub_labels))
width = 0.25

for si, (signal, color) in enumerate(zip(signals_sub, colors_sub)):
    betas = []
    for start, end in sub_ranges:
        sub = panel[(panel["month"] >= start) & (panel["month"] <= end)]
        res = fama_macbeth_nw(sub, "excess_return", [signal])
        betas.append(res[signal]["beta"] if signal in res else 0)
    
    ax.bar(x + si * width - width, betas, width, color=color, alpha=0.7, edgecolor='black', linewidth=0.5, label=signal)

ax.set_xticks(x)
ax.set_xticklabels(sub_labels)
ax.set_ylabel('Fama-MacBeth β')
ax.set_title('Sub-Sample Robustness: Disagreement–Return Relationship by Decade')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig5_subsample_robustness.png"))
print("  fig5_subsample_robustness.png")

# ── Figure 6: JS vs Return Scatter ──
fig, ax = plt.subplots(figsize=(7, 5))

# Sample 2000 points for clarity
sample = panel.dropna(subset=["JS_post", "excess_return"]).sample(min(2000, len(panel)), random_state=42)
ax.scatter(sample["JS_post"], sample["excess_return"] * 100, alpha=0.15, s=8, color='#2166ac')

# Fit line
slope, intercept, r, p, se = stats.linregress(panel["JS_post"].dropna(), panel["excess_return"].dropna() * 100)
x_line = np.linspace(panel["JS_post"].min(), panel["JS_post"].max(), 100)
ax.plot(x_line, slope * x_line + intercept, color='#b2182b', linewidth=2, 
        label=f'β = {slope:.2f} (r = {r:.3f})')

ax.set_xlabel('JS Divergence')
ax.set_ylabel('Monthly Excess Return (%)')
ax.set_title('JS Divergence vs Cross-Sectional Stock Returns')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig6_js_scatter.png"))
print("  fig6_js_scatter.png")

# ── Save updated panel ──
panel.to_csv(os.path.join(OUT, "panel_data.csv"), index=False)

print(f"\nAll figures saved to {FIGS}/")
print(f"Updated panel saved with analyst dispersion proxy")
