#!/usr/bin/env python3
"""
Generate updated figures for the revised paper using real LLM quarterly ratings.
Author: Siyi / 2026-06-05
"""

import json
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from numpy.linalg import lstsq

OUT = "/home/z/my-project/delta_jfe"
FIGS = os.path.join(OUT, "figures_llm_v2")
os.makedirs(FIGS, exist_ok=True)

# ── Load panel ──
panel = pd.read_csv(os.path.join(OUT, "panel_data_llm_quarterly.csv"))
print(f"Panel: {len(panel):,} obs")

# ── FM helper ──
def fama_macbeth_nw(panel, y_col, x_cols, nw_lag=None):
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
            lag = nw_lag if nw_lag else int(T ** (1/3))
            gamma0 = np.var(b, ddof=1)
            nw_var = gamma0
            for j in range(1, min(lag + 1, T)):
                w = 1 - j / (lag + 1)
                gamma_j = np.mean((b[j:] - mean_b) * (b[:-j] - mean_b))
                nw_var += 2 * w * gamma_j
            se = np.sqrt(max(nw_var, 0) / T)
            t_stat = mean_b / se if se > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=T-1))
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            results[col] = {"beta": mean_b, "t_stat": t_stat, "sig": sig, "p_val": p_val}
    return results

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

# ── Figure 1: FM Univariate Coefficients with 95% CI ──
fig, ax = plt.subplots(figsize=(8, 5))

signals_plot = ["JS_post", "D_post", "H_sentiment", "confidence"]
labels = ["JS Divergence", "D_post", "H_sentiment", "Confidence"]
colors = ['#2166ac', '#4dac26', '#b2182b', '#ff7f00']

uni_betas = []
uni_cis = []
uni_sigs = []

for signal in signals_plot:
    res = fama_macbeth_nw(panel, "excess_return", [signal])
    if signal in res:
        uni_betas.append(res[signal]["beta"])
        # 95% CI
        months_data = sorted(panel["month"].dropna().unique())
        betas_ts = []
        for month in months_data:
            md = panel[panel["month"] == month].dropna(subset=["excess_return", signal])
            if len(md) < 10:
                continue
            X = np.column_stack([np.ones(len(md)), md[signal].values])
            y = md["excess_return"].values
            try:
                b, _, _, _ = lstsq(X, y, rcond=None)
                betas_ts.append(b[1])
            except Exception:
                continue
        ci = 1.96 * np.std(betas_ts, ddof=1) / np.sqrt(len(betas_ts))
        uni_cis.append(ci)
        uni_sigs.append(res[signal]["sig"])
    else:
        uni_betas.append(0)
        uni_cis.append(0)
        uni_sigs.append("")

x = np.arange(len(signals_plot))
bars = ax.bar(x, uni_betas, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.errorbar(x, uni_betas, yerr=[1.96 * c for c in uni_cis], fmt='none', color='black', capsize=5, linewidth=1)

for i, (b, s) in enumerate(zip(uni_betas, uni_sigs)):
    offset = 0.005 if b >= 0 else -0.005
    ax.text(i, b + offset, s, ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Fama-MacBeth Coefficient (β)')
ax.set_title('Univariate FM Coefficients (Real LLM Ratings, 13,340 obs)')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_fm_univariate.png"))
print("  fig1_fm_univariate.png")

# ── Figure 2: Decile Portfolio Returns ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (signal_name, signal_col, color) in [
    (axes[0], ("JS Divergence", "JS_post", "#2166ac")),
    (axes[1], ("H Sentiment", "H_sentiment", "#b2182b")),
    (axes[2], ("Confidence", "confidence", "#ff7f00")),
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
    
    ls_mean = np.mean(ls_rets) * 100
    ls_se = np.std(ls_rets, ddof=1) / np.sqrt(len(ls_rets)) * 100
    ls_t = np.mean(ls_rets) / (np.std(ls_rets, ddof=1) / np.sqrt(len(ls_rets)))
    ls_sig = "***" if abs(ls_t) > 2.58 else "**" if abs(ls_t) > 1.96 else "*" if abs(ls_t) > 1.64 else ""
    
    ax.bar([11.5], [ls_mean], color='grey', alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
    ax.errorbar([11.5], [ls_mean], yerr=[1.96 * ls_se], fmt='none', color='black', capsize=3, linewidth=0.8)
    ax.text(11.5, ls_mean + 0.2, f"L-S{ls_sig}", ha='center', fontsize=8)
    
    ax.set_xlabel('Decile (Low → High)')
    ax.set_ylabel('Monthly Excess Return (%)')
    ax.set_title(signal_name)
    ax.set_xticks(list(range(1, 11)) + [11.5])
    ax.set_xticklabels([str(i) for i in range(1, 11)] + ['L-S'], fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_decile_returns.png"))
print("  fig2_decile_returns.png")

# ── Figure 3: Sign Reversal with Controls ──
fig, ax = plt.subplots(figsize=(9, 5))

signals_rev = ["JS_post", "D_post", "H_sentiment"]
labels_rev = ["JS Divergence", "D_post", "H_sentiment"]

# Unconditional
uni_b = []
uni_t = []
for s in signals_rev:
    res = fama_macbeth_nw(panel, "excess_return", [s])
    uni_b.append(res[s]["beta"] if s in res else 0)
    uni_t.append(res[s]["t_stat"] if s in res else 0)

# With controls
ctrl_b = []
ctrl_t = []
ctrl_cols = signals_rev + ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "rel_size", "mom_6m", "vol_6m", "bm_proxy"]
sub = panel.dropna(subset=ctrl_cols + ["excess_return"])
res_ctrl = fama_macbeth_nw(sub, "excess_return", ctrl_cols)
for s in signals_rev:
    ctrl_b.append(res_ctrl[s]["beta"] if s in res_ctrl else 0)
    ctrl_t.append(res_ctrl[s]["t_stat"] if s in res_ctrl else 0)

x = np.arange(len(signals_rev))
width = 0.35

bars1 = ax.bar(x - width/2, uni_b, width, color='#6baed6', alpha=0.8, edgecolor='black', linewidth=0.5, label='Unconditional')
bars2 = ax.bar(x + width/2, ctrl_b, width, color='#b2182b', alpha=0.8, edgecolor='black', linewidth=0.5, label='+ Full Controls')

# Add t-stats
for i, (b, t) in enumerate(zip(uni_b, uni_t)):
    sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
    ax.text(i - width/2, b + (0.02 if b >= 0 else -0.04), f't={t:+.2f}{sig}', ha='center', fontsize=8)

for i, (b, t) in enumerate(zip(ctrl_b, ctrl_t)):
    sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
    ax.text(i + width/2, b + (0.02 if b >= 0 else -0.04), f't={t:+.2f}{sig}', ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels_rev)
ax.set_ylabel('Fama-MacBeth β')
ax.set_title('Sign Reversal: Disagreement Signals Flip Negative with Controls')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_sign_reversal.png"))
print("  fig3_sign_reversal.png")

# ── Figure 4: Sub-sample Robustness ──
fig, ax = plt.subplots(figsize=(10, 5))

sub_labels = ["2005-2009", "2010-2014", "2015-2019", "2020-2024"]
sub_ranges = [("2005-01", "2009-12"), ("2010-01", "2014-12"), ("2015-01", "2019-12"), ("2020-01", "2024-12")]
signals_sub = ["JS_post", "H_sentiment", "D_post"]
colors_sub = ["#2166ac", "#b2182b", "#4dac26"]

x = np.arange(len(sub_labels))
width = 0.25

for si, (signal, color) in enumerate(zip(signals_sub, colors_sub)):
    tstats = []
    for start, end in sub_ranges:
        sub = panel[(panel["month"] >= start) & (panel["month"] <= end)]
        res = fama_macbeth_nw(sub, "excess_return", [signal])
        tstats.append(res[signal]["t_stat"] if signal in res else 0)
    
    ax.bar(x + si * width - width, tstats, width, color=color, alpha=0.7, edgecolor='black', linewidth=0.5, label=signal)

ax.set_xticks(x)
ax.set_xticklabels(sub_labels)
ax.set_ylabel('FM t-statistic')
ax.set_title('Sub-Sample Robustness: Disagreement–Return Relationship by Decade')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.axhline(y=1.96, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.axhline(y=-1.96, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_subsample_robustness.png"))
print("  fig4_subsample_robustness.png")

# ── Figure 5: H_sentiment Scatter ──
fig, ax = plt.subplots(figsize=(7, 5))

sample = panel.dropna(subset=["H_sentiment", "excess_return"]).sample(min(3000, len(panel)), random_state=42)
ax.scatter(sample["H_sentiment"], sample["excess_return"] * 100, alpha=0.12, s=8, color='#b2182b')

slope, intercept, r, p, se = stats.linregress(
    panel["H_sentiment"].dropna(), 
    panel.dropna(subset=["H_sentiment"])["excess_return"] * 100
)
x_line = np.linspace(panel["H_sentiment"].min(), panel["H_sentiment"].max(), 100)
ax.plot(x_line, slope * x_line + intercept, color='#2166ac', linewidth=2, 
        label=f'β = {slope:.2f} (r = {r:.3f}, p = {p:.3f})')

ax.set_xlabel('H_sentiment (Shannon Entropy)')
ax.set_ylabel('Monthly Excess Return (%)')
ax.set_title('H_sentiment vs Next-Month Stock Return (Real LLM Ratings)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig5_h_sentiment_scatter.png"))
print("  fig5_h_sentiment_scatter.png")

# ── Figure 6: LLM vs Quant Comparison ──
fig, ax = plt.subplots(figsize=(9, 5))

# Load old quant results
try:
    old = json.load(open(os.path.join(OUT, "analysis_results.json")))
    old_fm = old.get("fm_univariate", {})
except Exception:
    old_fm = {}

signals_cmp = ["JS_post", "D_post", "H_sentiment", "confidence"]
labels_cmp = ["JS Divergence", "D_post", "H_sentiment", "Confidence"]

# New LLM results
new_fm = {}
for s in signals_cmp:
    res = fama_macbeth_nw(panel, "excess_return", [s])
    new_fm[s] = res.get(s, {})

llm_t = [new_fm.get(s, {}).get("t_stat", 0) for s in signals_cmp]
quant_t = [old_fm.get(s, {}).get("t_stat", 0) for s in signals_cmp]

x = np.arange(len(signals_cmp))
width = 0.35

ax.bar(x - width/2, quant_t, width, color='#4dac26', alpha=0.7, edgecolor='black', linewidth=0.5, label='Quant Model (39,866 obs)')
ax.bar(x + width/2, llm_t, width, color='#b2182b', alpha=0.7, edgecolor='black', linewidth=0.5, label='LLM Agents (13,340 obs)')

ax.axhline(y=1.96, color='grey', linewidth=0.5, linestyle=':', alpha=0.5, label='5% significance')
ax.axhline(y=-1.96, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels_cmp)
ax.set_ylabel('FM t-statistic')
ax.set_title('LLM vs Quantitative Model: Univariate FM t-statistics')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig6_llm_vs_quant.png"))
print("  fig6_llm_vs_quant.png")

print(f"\nAll figures saved to {FIGS}/")
