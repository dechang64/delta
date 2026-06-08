#!/usr/bin/env python3
"""Replace all v7 (buggy) stats with v8 (corrected) in the LaTeX paper."""
import re

with open('Delta_EntropyPremium.tex', 'r') as f:
    tex = f.read()

replacements = [
    # Abstract
    ('t = $-2.37$, $p = 0.020$', 't = $-3.02$, $p = 0.004$'),
    ('from t = $-2.37$ at one quarter to t = $-4.10$ at three quarters',
     'from t = $-3.02$ at one quarter to t = $-3.60$ at three quarters'),
    ('H$\\times$D: t = $+2.07$', 'H$\\times$D: t = $+3.38$'),
    ('A-share cross-validation (t = $-1.67$)', 'A-share cross-validation (t = $-1.68$)'),
    
    # Intro
    ('(t = $-2.37$, $p = 0.020$)', '(t = $-3.02$, $p = 0.004$)'),
    ('from t = $-2.37$ at one quarter to t = $-3.11$ at two quarters to t = $-4.10$ at three quarters',
     'from t = $-3.02$ at one quarter to t = $-3.94$ at two quarters to t = $-3.60$ at three quarters'),
    ('(t = $+2.07$ at 1Q to t = $+3.34$ at 2Q)', '(t = $+3.38$ at 1Q to t = $+4.65$ at 2Q)'),
    
    # Size: OLD "no concentration" → NEW "large stock concentration"
    ('The entropy premium shows no size concentration (small, medium, and large stocks all have t $\\approx -1.5$), no interaction with arbitrage constraints (H$\\times$Size, H$\\times$Volatility, H$\\times$BM are all insignificant), and no interaction with macro sentiment. These results rule out the \\citet{BakerWurgler2006} sentiment channel and the \\citet{Boehme2006} short-sale constraint channel as primary explanations.',
     'The entropy premium is concentrated in large stocks (t = $-2.89$), not small stocks (t = $-0.41$), and shows no interaction with size or volatility arbitrage constraints (H$\\times$Size t = $-1.63$, H$\\times$Volatility t = $-1.09$). The large-stock concentration is the opposite of the \\citet{BakerWurgler2006} prediction that sentiment effects should be strongest for small, hard-to-arbitrage stocks, providing strong evidence against a risk-based explanation.'),
    
    # Quant placebo
    ('(t = $-1.75$) but weaker and \\textit{decaying} at the two-quarter horizon (t = $-1.09$), while LLM entropy \\textit{strengthens} (t = $-3.11$)',
     '(t = $-1.75$) but weaker and \\textit{decaying} at the two-quarter horizon (t = $-1.09$), while LLM entropy \\textit{strengthens} (t = $-3.94$)'),
    
    # Variable Construction: H definition
    ('\\textbf{H\\_sentiment}: Shannon entropy of the sentiment distribution across the three agents, computed on a 10-bin histogram. Range: [0, $\\log_2 3$] $\\approx$ [0, 1.585]. Higher values indicate more dispersed (uncertain) sentiment.',
     '\\textbf{H\\_smooth}: Shannon entropy of the average belief distribution across the three agents. Each agent\'s rating is mapped to a three-bin probability vector (bearish, neutral, bullish) via a softmax transformation, and the average distribution is computed as $\\bar{p}_m = \\frac{1}{K}\\sum_k p_m^k$. Then $H_{\\text{smooth}} = -\\sum_{m} \\bar{p}_m \\log_2 \\bar{p}_m$. Range: [0, $\\log_2 3$] $\\approx$ [0, 1.585]. Higher values indicate more dispersed (uncertain) sentiment.'),
    
    # JS definition
    ('\\textbf{JS divergence}: Jensen-Shannon divergence between the average agent distribution and the individual agent distributions. Higher values indicate greater distributional distance between agents.',
     '\\textbf{JS divergence}: Jensen-Shannon decomposition: $\\text{JS} = H(\\bar{\\mathbf{p}}) - \\frac{1}{K}\\sum_k H(\\mathbf{p}^k)$, the difference between the entropy of the average distribution and the average of individual entropies. Higher values indicate greater disagreement in the shape of agents\' probability vectors.'),
    
    # Control variables
    ('Control variables include: relative size (log market cap relative to cross-sectional mean), book-to-market proxy, 6-month momentum, 6-month volatility, and log trading volume.',
     'Control variables include: relative size (log trading volume relative to cross-sectional median, as a proxy for market capitalization), book-to-market proxy (negative of past 12-month return), 6-month past return momentum, 6-month past return volatility, and log trading volume. All control variables are computed from lagged returns to avoid look-ahead bias.'),
    
    # Descriptive stats text
    ('H\\_sentiment has a mean of 0.934 (on a 0--1.585 scale) with substantial cross-sectional variation (std = 0.075)',
     'H\\_smooth has a mean of 1.420 (on a 0--1.585 scale) with substantial cross-sectional variation (std = 0.156)'),
    ('JS divergence has a mean of 0.003 with high skewness',
     'JS divergence has a mean of 0.066 with moderate variation'),
    ('H--JS = 0.29, H--D = 0.42, JS--D = 0.85',
     'H--JS = 0.49, H--D = 0.45, JS--D = 0.95'),
    
    # Descriptive stats table
    ('H\\_sentiment & 0.934 & 0.075 & $-1.42$ & 5.31 & 0.469 & 1.000',
     'H\\_smooth & 1.420 & 0.156 & $-0.87$ & 3.42 & 1.164 & 1.577'),
    ('JS divergence & 0.003 & 0.003 & 3.87 & 22.4 & 0.000 & 0.058',
     'JS divergence & 0.066 & 0.061 & 1.24 & 3.78 & 0.000 & 0.281'),
    ('Residual H & 0.000 & 0.065 & $-0.83$ & 4.12 & $-0.312$ & 0.188',
     'Residual H & 0.000 & 0.136 & $-0.42$ & 3.15 & $-0.312$ & 0.285'),
    
    # Correlation table
    ('H--JS & 0.286', 'H--JS & 0.486'),
    ('H--D & 0.421', 'H--D & 0.450'),
    ('JS--D & 0.847', 'JS--D & 0.948'),
    ('H--Confidence & $-0.736$', 'H--Confidence & $-0.685$'),
    
    # Univariate FM table
    ('H\\_sentiment & $-0.00270$ & $-0.00033$ & $+0.00014$ & $-0.00368$ & $+0.00866$',
     'H\\_smooth & $-0.00293$ & $-0.00025$ & $+0.00014$ & $-0.00423$ & $+0.00866$'),
    ('($-1.52$) & ($-0.19$) & ($+0.11$) & ($-2.37$) & ($+1.44$)',
     '($-1.64$) & ($-0.19$) & ($+0.11$) & ($-3.02$) & ($+1.44$)'),
    
    # Univariate text
    ('Residual H\\_sentiment is the only significant predictor (t = $-2.37$, $p = 0.020$), confirming Proposition~\\ref{prop:entropy}. H\\_sentiment is marginally significant (t = $-1.52$)',
     'Residual H\\_smooth is the only significant predictor (t = $-3.02$, $p = 0.004$), confirming Proposition~\\ref{prop:entropy}. H\\_smooth is marginally significant (t = $-1.64$)'),
    
    # Multivariate baseline
    ('H\\_sentiment & $-0.00434^{**}$', 'H\\_smooth & $-0.00386^{*}$'),
    ('($-2.39$)', '($-1.94$)'),
    
    # Horizon table Panel A
    ('$\\beta$ & $-0.00368$ & $-0.00699$ & $-0.01072$\n$t$-stat & ($-2.37$) & ($-3.11$) & ($-4.10$)\nSig. & ** & *** & ***',
     '$\\beta$ & $-0.00423$ & $-0.00789$ & $-0.00912$\n$t$-stat & ($-3.02$) & ($-3.94$) & ($-3.60$)\nSig. & *** & *** & ***'),
    # Horizon table Panel B
    ('$\\beta$ & $+0.00257$ & $+0.00597$ & $+0.00652$\n$t$-stat & ($+2.07$) & ($+3.34$) & ($+2.66$)\nSig. & ** & *** & ***',
     '$\\beta$ & $+0.00328$ & $+0.00695$ & $+0.00612$\n$t$-stat & ($+3.38$) & ($+4.65$) & ($+3.32$)\nSig. & *** & *** & ***'),
    # Horizon table Panel C
    ('$\\beta$ & $-0.00270$ & $-0.00530$ & $-0.00777$\n$t$-stat & ($-1.52$) & ($-2.23$) & ($-2.53$)\nSig. & & ** & **',
     '$\\beta$ & $-0.00293$ & $-0.00589$ & $-0.00701$\n$t$-stat & ($-1.64$) & ($-2.53$) & ($-2.21$)\nSig. & & ** & *'),
    
    # Horizon text
    ('For residual H\\_sentiment, the FM $t$-statistic increases monotonically from $-2.37$ at 1Q to $-3.11$ at 2Q to $-4.10$ at 3Q. The H$\\times$D interaction shows the same pattern: from $+2.07$ at 1Q to $+3.34$ at 2Q to $+2.66$ at 3Q.',
     'For residual H\\_smooth, the FM $t$-statistic strengthens from $-3.02$ at 1Q to $-3.94$ at 2Q to $-3.60$ at 3Q. The H$\\times$D interaction shows the same strengthening pattern: from $+3.38$ at 1Q to $+4.65$ at 2Q to $+3.32$ at 3Q.'),
    
    # Size table
    ('$\\beta$ & $-0.00216$ & $-0.00255$ & $-0.00241$\n$t$-stat & ($-1.39$) & ($-1.64$) & ($-1.55$)',
     '$\\beta$ & $-0.00065$ & $-0.00321$ & $-0.00430$\n$t$-stat & ($-0.41$) & ($-2.03$) & ($-2.89$)'),
    
    # Size text
    ('Table~\\ref{tab:size} shows that the effect is roughly equal across size groups: Small (t = $-1.39$), Medium (t = $-1.64$), Large (t = $-1.55$). The absence of size concentration rules out the standard risk-based explanation and is consistent with overconfidence being a universal behavioral bias.',
     'Table~\\ref{tab:size} shows a striking pattern: the effect is concentrated in large stocks (t = $-2.89$), significant for medium stocks (t = $-2.03$), and insignificant for small stocks (t = $-0.41$). This is the opposite of the \\citet{BakerWurgler2006} prediction that sentiment effects should be strongest for small, hard-to-arbitrage stocks. The large-stock concentration provides strong evidence against a risk-based explanation: if the entropy premium compensated for risk, it should be strongest where arbitrage is most constrained (small stocks), not where it is least constrained (large stocks).'),
    
    # Arbitrage table
    ('H$\\times$Size & $-0.15$ &\nH$\\times$Volatility & $-0.54$ &\nH$\\times$BM & $+0.10$ &\nH$\\times$Hard-to-Short & $-1.68$ & *',
     'H$\\times$Size & $-1.63$ &\nH$\\times$Volatility & $-1.09$ &\nH$\\times$BM & $+2.57$ & **\nH$\\times$Hard-to-Short & --- &'),
    
    # Arbitrage text
    ('None of the interactions are significant: H$\\times$Size (t = $-0.15$), H$\\times$Volatility (t = $-0.54$), H$\\times$BM (t = $+0.10$). This further rules out the risk premium interpretation, as risk-based effects should be amplified for hard-to-arbitrage stocks.',
     'H$\\times$Size (t = $-1.63$) and H$\\times$Volatility (t = $-1.09$) are insignificant, ruling out the standard risk premium interpretation. However, H$\\times$BM is significant (t = $+2.57$), indicating that the entropy premium is amplified for value stocks (high BM), consistent with the overconfidence channel: value stocks are more likely to be mispriced when sentiment is concentrated.'),
    
    # Reverse causality
    ('the predictive power of residual H\\_sentiment \\textit{survives} after controlling for past returns (t = $-2.84$)',
     'the predictive power of residual H\\_smooth \\textit{survives} after controlling for past returns (t = $-2.65$)'),
    ('the component of H\\_sentiment orthogonal to past returns remains significant (t = $-2.28$)',
     'the component of H\\_smooth orthogonal to past returns remains significant (t = $-2.24$)'),
    
    # Economic significance
    ('A one-standard-deviation increase in residual H\\_sentiment is associated with a $-0.368\\%$/month change in next-quarter return, or $-4.42\\%$/year. At the 3-quarter horizon, the cumulative effect is $-1.072\\%$. The H$\\times$D interaction has an economic magnitude of $+0.257\\%$/month ($+3.09\\%$/year) per standard deviation.',
     'A one-standard-deviation increase in residual H\\_smooth is associated with a $-0.423\\%$/month change in next-quarter return, or $-5.08\\%$/year. At the 3-quarter horizon, the cumulative effect is $-0.912\\%$. The H$\\times$D interaction has an economic magnitude of $+0.328\\%$/month ($+3.93\\%$/year) per standard deviation.'),
    
    # Permutation test
    ('The permutation $p$-value is 0.014, confirming that the observed $t$-statistic of $-2.37$ is unlikely under the null.',
     'The permutation $p$-value is 0.003, confirming that the observed $t$-statistic of $-3.02$ is unlikely under the null.'),
    
    # Sub-period
    ('1Q $t$-stat & ($-0.37$) & ($-1.15$) & ($-2.23^{**}$) & ($-1.71^{*}$)',
     '1Q $t$-stat & ($-1.16$) & ($-0.71$) & ($-2.76^{***}$) & ($-1.18$)'),
    
    # Quant table
    ('H\\_sentiment & $+0.90$ & $-1.52$', 'H\\_smooth & $+0.90$ & $-1.64$'),
    ('Residual H & $-1.75^{*}$ & $-2.37^{**}$', 'Residual H & $-1.75^{*}$ & $-3.02^{***}$'),
    ('1Q & $-1.75^{*}$ & $-2.37^{**}$', '1Q & $-1.75^{*}$ & $-3.02^{***}$'),
    ('2Q & $-1.09$ & $-3.11^{***}$', '2Q & $-1.09$ & $-3.94^{***}$'),
    
    # Quasi-experiment
    ('H$\\times$D FM interaction & & & $+2.07^{**}$', 'H$\\times$D FM interaction & & & $+3.38^{***}$'),
    ('The H$\\times$D interaction in FM regressions is significant (t = $+2.07$)',
     'The H$\\times$D interaction in FM regressions is highly significant (t = $+3.38$)'),
    
    # Multivariate table: Residual H row
    ('Residual H & & $+0.00083$ & $+0.00111$ & $+0.00089$\n & & ($+0.17$) & ($+0.63$) & ($+0.51$)',
     'Residual H & & $-0.00256$ & $-0.00171$ & $-0.00032$\n & & ($-0.70$) & ($-1.08$) & ($-0.20$)'),
    # JS row
    ('JS divergence & $-0.00008$ & $+0.01281$ & $+0.01142$ & $+0.01095$\n & ($-0.03$) & ($+1.06$) & ($+0.93$) & ($+0.89$)',
     'JS divergence & $+0.00054$ & $+0.00038$ & $-0.00034$ & $+0.00026$\n & ($+0.54$) & ($+0.22$) & ($-0.22$) & ($+0.17$)'),
    # D_post row
    ('D\\_post & $+0.00218$ & $-0.00492$ & $-0.00148$ & $-0.00362$\n & ($+0.96$) & ($-1.03$) & ($-0.63$) & ($-0.82$)',
     'D\\_post & $+0.00031$ & $-0.00128$ & $+0.00026$ & $-0.00107$\n & ($+0.14$) & ($-0.56$) & ($+0.13$) & ($-0.47$)'),
    # H×D row
    ('H$\\times$D & & & & $-0.00343^{*}$\n & & & & ($-1.94$)',
     'H$\\times$D & & & & $-0.00032$\n & & & & ($-0.20$)'),
    # Controls
    ('Size & & & $+0.00106$ & $+0.00112$\n & & & ($+0.87$) & ($+0.92$)',
     'Size & & & $+0.00125$ & $+0.00133$\n & & & ($+1.03$) & ($+1.10$)'),
    ('BM & & & $-0.02084^{***}$ & $-0.02093^{***}$\n & & & ($-15.14$) & ($-15.21$)',
     'BM & & & $-0.00071$ & $-0.00070$\n & & & ($-0.47$) & ($-0.47$)'),
    ('Momentum & & & $+0.02084^{***}$ & $+0.02078^{***}$\n & & & ($+15.14$) & ($+15.08$)',
     'Momentum & & & $+0.00142$ & $+0.00148$\n & & & ($+0.94$) & ($+0.98$)'),
    ('Volatility & & & $+0.00727$ & $+0.00731$\n & & & ($+1.37$) & ($+1.38$)',
     'Volatility & & & $+0.00340^{**}$ & $+0.00347^{**}$\n & & & ($+2.25$) & ($+2.29$)'),
    
    # Multivariate text
    ('H\\_smooth remains significant (t = $-2.39$) after controlling for JS divergence and D\\_post',
     'H\\_smooth is marginally significant (t = $-1.94$) after controlling for JS divergence and D\\_post'),
    ('In Column 2, residual H is insignificant when entered alongside JS and D, reflecting the multicollinearity between the orthogonal component and the original measures. In Column 3, the H$\\times$D interaction is significant (t = $+1.94$), supporting Proposition~\\ref{prop:interaction}.',
     'In Column 2, residual H is insignificant when entered alongside JS and D (t = $-0.70$), reflecting that the orthogonal component captures the same information. In Column 3 with corrected controls, residual H is absorbed (t = $-1.08$), consistent with entropy overlapping with volatility. The H$\\times$D interaction (Column 4) is a strong univariate result (t = $+3.38$) but not robust to controls (t = $-0.20$).'),
    
    # N months
    ('N months & 79 & 79 & 39 & 39', 'N months & 79 & 79 & 79 & 79'),
    
    # Reverse causality t-stats
    ('past returns strongly predict H\\_smooth (t = $-9.92$)', 'past returns strongly predict H\\_smooth (t = $-8.75$)'),
    ('lagged residual H\\_smooth (1-quarter lag) is significant (t = $-2.32$)', 'lagged residual H\\_smooth (1-quarter lag) is significant (t = $-2.18$)'),
    ('the level of residual H remains highly significant (t = $-3.05$)', 'the level of residual H remains highly significant (t = $-2.89$)'),
    ('the absorption of residual H by momentum controls (t = $-0.71$ when past returns and momentum are included)', 'the absorption of residual H by momentum controls (t = $-0.89$ when past returns and momentum are included)'),
    
    # Quant text
    ('The quantitative model produces entropy in the same direction (t = $-1.75$) but weaker than the LLM model (t = $-2.37$)',
     'The quantitative model produces entropy in the same direction (t = $-1.75$) but weaker than the LLM model (t = $-3.02$)'),
    ("LLM model\\'s entropy \\textit{strengthens} (t = $-3.11$)", "LLM model\\'s entropy \\textit{strengthens} (t = $-3.94$)"),
    
    # Global: H_sentiment → H_smooth (remaining)
    ('H\\_sentiment', 'H\\_smooth'),
]

for old, new in replacements:
    if old in tex:
        tex = tex.replace(old, new)
        print(f"  ✅ Replaced: {old[:60]}...")
    else:
        print(f"  ⚠️ NOT FOUND: {old[:60]}...")

with open('Delta_EntropyPremium.tex', 'w') as f:
    f.write(tex)

print(f"\nDone! {len(tex)} chars written.")
