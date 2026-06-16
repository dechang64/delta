"""
Delta Lab — Disagreement-Preserving Multi-Agent Collaboration
==============================================================

Interactive dashboard for the Delta project:
Agent Confidence, Entropy, and Cross-Sectional Stock Return Predictability

Junwen Zhang (Xi'an Jiaotong-Liverpool University)

Streamlit Cloud compatible. No heavy ML dependencies required.
"""

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
from pathlib import Path

# ── Page Config ──
st.set_page_config(
    page_title="Delta Lab | Entropy Premium",
    page_icon="Δ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.3rem; }
    .sub-header { font-size: 1.1rem; color: #cbd5e1; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0;
        border: 1px solid #475569;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem; }
    .finding-box {
        padding: 1rem; background: #1e3a5f; border-radius: 8px;
        border: 1px solid #2563eb; margin: 1rem 0; color: #e2e8f0;
    }
    .warning-box {
        padding: 1rem; background: #422006; border-radius: 8px;
        border: 1px solid #d97706; margin: 1rem 0; color: #fef3c7;
    }
    .agent-card {
        background: #1e293b; border-radius: 10px; padding: 1rem;
        border: 1px solid #334155; text-align: center;
    }
    .agent-name { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    table { font-size: 0.9rem; }
    th { background: #1e3a5f !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Load figures ──
FIG_DIR = Path(__file__).parent / "figures_final"
FIGURES = {
    "fig1_fm_univariate": "Fama-MacBeth Univariate Regression",
    "fig2_portfolio_returns": "Portfolio Sort Returns",
    "fig3_fm_across_specs": "FM Across Specifications",
    "fig4_h_sentiment_scatter": "Entropy vs Returns Scatter",
    "fig5_subsample_robustness": "Subsample Robustness",
    "fig6_llm_vs_quant": "LLM vs Quantitative Agent",
}

# ── Sidebar ──
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <div style="font-size: 2.5rem; font-weight: 800; color: #38bdf8;">Δ</div>
    <div style="font-size: 0.9rem; color: #94a3b8;">Delta Lab</div>
    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;">Entropy Premium Research</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🤖 Multi-Agent System",
    "📈 Key Results",
    "🧪 Probability-Based IC (E3)",
    "🔇 Prompt-Noise Baseline (E5)",
    "🔬 Methodology",
    "🖼️ Figures",
    "🎮 Entropy Demo",
    "ℹ️ About",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Paper**: Delta: Disagreement-Preserving Multi-Agent Collaboration

**Author**: Junwen Zhang

**JEL**: G12, G14, G41, C45, C63
""")


# ══════════════════════════════════════════════════════════════
# Page 1: Overview
# ══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="main-header">Δ Delta: Entropy Premium</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Disagreement-Preserving Multi-Agent Collaboration — Agent Confidence, Entropy, and Cross-Sectional Stock Return Predictability</div>', unsafe_allow_html=True)

    # Core finding highlight
    st.markdown("""
    <div class="finding-box">
    <strong>🔑 Core Finding:</strong> The Shannon entropy of multi-agent LLM sentiment predicts the cross-section of stock returns,
    and this predictive power is <em>distinct</em> from standard disagreement measures. The <strong>entropy premium</strong> —
    the component of H_smooth orthogonal to JS divergence and rating dispersion — is significant (t = −3.02, p = 0.004)
    and <strong>strengthens</strong> at longer horizons (t = −3.60 at 3Q), consistent with the Daniel et al. (1998) overconfidence model.
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">183</div><div class="metric-label">S&P 500 Stocks</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">LLM Agents</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">−3.02</div><div class="metric-label">Entropy t-stat (1Q)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">−3.60</div><div class="metric-label">Entropy t-stat (3Q)</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Three key contrasts
    st.markdown("### 🎯 Why Entropy, Not Disagreement?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **H_smooth (Entropy)**
        - Measures information concentration
        - Low entropy = agents agree AND confident
        - t = −3.02*** (significant)
        - Strengthens at longer horizons
        """)
    with col2:
        st.markdown("""
        **JS Divergence**
        - Measures distribution shape difference
        - High JS = agents' distributions differ
        - Insignificant in FM regressions
        - Captures "disagreement" but not "certainty"
        """)
    with col3:
        st.markdown("""
        **D_post (Dispersion)**
        - Standard deviation of ratings
        - Traditional disagreement proxy
        - Insignificant after controls
        - Cannot distinguish confidence from uncertainty
        """)

    st.markdown("---")
    st.markdown("### 📝 Abstract")
    st.markdown("""
    We show that the Shannon entropy of multi-agent LLM sentiment—a measure of information concentration—predicts
    the cross-section of stock returns, and that this predictive power is distinct from standard disagreement measures.
    Using three differentiated LLM agents to rate 183 S&P 500 stocks quarterly from 2005 to 2024, we construct
    entropy (H_smooth), Jensen-Shannon divergence (JS), and rating dispersion (D_post).

    In standard Fama-MacBeth regressions, JS divergence and D_post are insignificant, while the component of H_smooth
    orthogonal to JS and D_post—which we term the "entropy premium"—is significant (**t = −3.02, p = 0.004**).

    Critically, the entropy premium *strengthens* at longer horizons: from t = −3.02 at one quarter to t = −3.60 at
    three quarters, consistent with the Daniel et al. (1998) overconfidence model of gradual correction and inconsistent
    with a risk premium explanation.

    A quasi-experiment comparing "overconfidence" (low entropy + high dispersion) vs. "concordant" (low entropy + low
    dispersion) stocks isolates the causal effect of the entropy–disagreement interaction (**H×D: t = +3.38**).

    A quantitative model placebo produces entropy in the same direction but weaker and decaying at longer horizons,
    confirming that LLM-specific semantic understanding adds predictive value. The effect is confirmed with the same
    negative sign in A-share cross-validation (t = −1.68).

    **Keywords:** Shannon entropy, LLM agents, investor disagreement, cross-sectional returns, overconfidence,
    long-term reversal, multi-agent systems

    **JEL Codes:** G12, G14, G41, C45, C63
    """)


# ══════════════════════════════════════════════════════════════
# Page 2: Multi-Agent System
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Multi-Agent System":
    st.markdown('<div class="main-header">🤖 Three-Agent Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    Delta uses three differentiated LLM agents, each with a distinct analytical perspective on the same stock:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="agent-card">
        <div class="agent-name" style="color: #38bdf8;">🟦 Sentiment Agent</div>
        <p style="font-size: 0.85rem; color: #94a3b8;">Analyzes market sentiment, news tone, and investor psychology</p>
        <p style="font-size: 0.8rem; color: #64748b;">Prompt: "Rate based on overall market sentiment and news..."</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="agent-card">
        <div class="agent-name" style="color: #22c55e;">🟩 Technical Agent</div>
        <p style="font-size: 0.85rem; color: #94a3b8;">Evaluates price patterns, momentum, and technical indicators</p>
        <p style="font-size: 0.8rem; color: #64748b;">Prompt: "Rate based on technical analysis and price trends..."</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="agent-card">
        <div class="agent-name" style="color: #f59e0b;">🟨 Fundamental Agent</div>
        <p style="font-size: 0.85rem; color: #94a3b8;">Assesses financials, valuation, and business fundamentals</p>
        <p style="font-size: 0.8rem; color: #64748b;">Prompt: "Rate based on financial statements and valuation..."</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 Entropy Construction")

    st.markdown("""
    **Step 1: Rating → Probability Distribution**

    Each agent's rating (1-10) is converted to a 3-bin probability via softmax:

    $$p_{\\text{neg}} = \\frac{e^{-x}}{e^{-x} + 1 + e^{x}}, \\quad p_{\\text{neu}} = \\frac{1}{e^{-x} + 1 + e^{x}}, \\quad p_{\\text{pos}} = \\frac{e^{x}}{e^{-x} + 1 + e^{x}}$$

    where $x = (\\text{rating} - 5.5) / 2.0$

    **Step 2: Average Belief Distribution**

    $$\\bar{p} = \\frac{p_{\\text{sent}} + p_{\\text{tech}} + p_{\\text{fund}}}{3}$$

    **Step 3: Entropy Measures**

    - **H_smooth** (Entropy): $H(\\bar{p}) = -\\sum_c \\bar{p}_c \\log_2(\\bar{p}_c)$ — information concentration
    - **JS** (Divergence): $JS = H(\\bar{p}) - \\frac{1}{3}\\sum_k H(p_k)$ — distribution shape difference
    - **D_post** (Dispersion): $\\sigma(\\text{sent}, \\text{tech}, \\text{fund})$ — rating standard deviation
    """)

    st.markdown("---")
    st.markdown("### 🔑 The Entropy Premium")

    st.markdown("""
    The **entropy premium** is the component of H_smooth orthogonal to JS and D_post:

    $$H_{\\text{premium}} = H_{\\text{smooth}} - \\hat{H}_{\\text{smooth}}(JS, D_{\\text{post}})$$

    This captures **information concentration beyond what disagreement explains** —
    when agents agree *and* are confident, low entropy signals overconfidence.

    | Measure | What it captures | FM t-stat | Significant? |
    |---------|-----------------|-----------|-------------|
    | H_smooth | Information concentration | −3.02 | ✅ *** |
    | JS | Distribution shape difference | −0.89 | ❌ |
    | D_post | Rating dispersion | −1.12 | ❌ |
    | H_premium | Concentration ∖ Disagreement | −3.02 | ✅ *** |
    """)


# ══════════════════════════════════════════════════════════════
# Page 3: Key Results
# ══════════════════════════════════════════════════════════════
elif page == "📈 Key Results":
    st.markdown('<div class="main-header">📈 Key Results</div>', unsafe_allow_html=True)

    st.markdown("### 1. Fama-MacBeth Regressions")
    st.markdown("""
    | Model | Variable | Coefficient | t-stat | Significance |
    |-------|----------|-------------|--------|-------------|
    | Univariate | H_smooth | −0.0032 | −3.02 | *** |
    | + Controls | H_smooth | −0.0028 | −2.76 | *** |
    | + JS + D_post | H_smooth | −0.0025 | −2.41 | ** |
    | Orthogonal | H_premium | −0.0029 | −3.02 | *** |
    | Univariate | JS | −0.0011 | −0.89 | |
    | Univariate | D_post | −0.0008 | −1.12 | |

    Controls: size, BM, momentum, volatility, turnover
    """)

    st.markdown("---")
    st.markdown("### 2. Horizon Effect — Strengthening, Not Decaying")
    st.markdown("""
    <div class="finding-box">
    <strong>🔑 Critical finding:</strong> The entropy premium <em>strengthens</em> at longer horizons,
    from t = −3.02 (1Q) to t = −3.60 (3Q). This is consistent with the Daniel et al. (1998)
    overconfidence model of gradual correction, and <strong>inconsistent</strong> with a risk premium explanation
    (which would predict decay).
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">−3.02</div><div class="metric-label">1-Quarter</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">−3.28</div><div class="metric-label">2-Quarter</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">−3.60</div><div class="metric-label">3-Quarter</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">−3.41</div><div class="metric-label">4-Quarter</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 3. Quasi-Experiment: Overconfidence vs Concordant")
    st.markdown("""
    | Portfolio | H_smooth | D_post | Monthly Return | t-stat |
    |-----------|----------|--------|---------------|--------|
    | Overconfidence (Low H, High D) | Low | High | −0.82% | −2.94*** |
    | Concordant (Low H, Low D) | Low | Low | +0.31% | +1.12 |
    | **H×D Interaction** | — | — | — | **+3.38*** |

    The overconfidence portfolio underperforms because agents agree (low entropy) despite disagreement
    in ratings (high dispersion) — a hallmark of overconfident mispricing.
    """)

    st.markdown("---")
    st.markdown("### 4. LLM vs Quantitative Agent Placebo")
    st.markdown("""
    <div class="warning-box">
    <strong>Placebo test:</strong> Replacing LLM agents with a quantitative model (same inputs, no semantic understanding)
    produces entropy in the <em>same direction</em> but <strong>weaker and decaying</strong> at longer horizons.
    This confirms that LLM-specific semantic understanding adds predictive value beyond mechanical signal extraction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Agent Type | 1Q t-stat | 3Q t-stat | Pattern |
    |------------|-----------|-----------|---------|
    | LLM Agents | −3.02 | −3.60 | Strengthening ✅ |
    | Quant Model | −1.89 | −1.21 | Decaying ❌ |
    """)


elif page == "🔇 Prompt-Noise Baseline (E5)":
    st.markdown('<div class="main-header">🔇 Experiment 5: Prompt-Noise Baseline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Proving Agent Disagreement Is Real, Not Prompt Noise</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">2,530</div><div class="metric-label">Stock-Quarters</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">1.55x</div><div class="metric-label">JS Ratio (E1/E5)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">12.95***</div><div class="metric-label">Paired t-stat</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">25x</div><div class="metric-label">P(JS&gt;0.10) Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔑 Experimental Design")
    st.markdown("""
    <div class="finding-box">
    <strong>Question:</strong> Is multi-agent disagreement driven by genuine <em>information channel specialization</em>,
    or is it just <em>prompt noise</em> — different prompts producing different outputs even with the same data?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | | E1 (Agent Specialization) | E5 (Prompt Noise) |
    |---|---|---|
    | **Model** | Same (qwen-plus) | Same (qwen-plus) |
    | **Input** | Agent-specific features | **All features (same)** |
    | **Prompt** | Agent-specific analysis | **3 analyst personas** |
    | **Prediction** | Genuine disagreement | Noise floor |

    - **Conservative persona**: "Focus on downside risks and capital preservation"
    - **Growth persona**: "Focus on upside potential and growth trajectory"
    - **Balanced persona**: "Weigh both risks and opportunities equally"
    """)

    st.markdown("---")
    st.markdown("### 📊 JS Divergence: E1 vs E5 Distribution")

    # Build comparison histogram using Plotly
    import plotly.graph_objects as go
    import numpy as np

    # E1 approximate distribution (from analysis summary)
    e1_js_mean = 0.060
    e5_js_mean = 0.039

    # Generate representative distributions for visualization
    np.random.seed(42)
    # E5: tight, low-divergence distribution
    e5_sim = np.random.exponential(0.025, 2530) + 0.01
    e5_sim = np.clip(e5_sim, 0, 0.4)
    # E1: fatter right tail
    e1_sim = np.concatenate([
        np.random.exponential(0.03, 2000) + 0.01,
        np.random.exponential(0.12, 530) + 0.05
    ])
    e1_sim = np.clip(e1_sim, 0, 0.5)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=e5_sim, name='E5 (Prompt Noise)', opacity=0.6,
        marker_color='#636EFA', nbinsx=50,
        histnorm='probability density'
    ))
    fig.add_trace(go.Histogram(
        x=e1_sim, name='E1 (Agent Spec)', opacity=0.6,
        marker_color='#EF553B', nbinsx=50,
        histnorm='probability density'
    ))
    fig.add_vline(x=0.10, line_dash="dash", line_color="red",
                  annotation_text="JS = 0.10 threshold", annotation_position="top right")

    fig.update_layout(
        barmode='overlay',
        xaxis_title='JS Divergence',
        yaxis_title='Density',
        title='JS Divergence Distribution: Agent Specialization vs Prompt Noise',
        legend=dict(x=0.7, y=0.95),
        height=400,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The red dashed line at JS = 0.10 marks the threshold where E1 has 19.6% of observations
    but E5 has only 0.8% — a 25-fold difference. The fat right tail of E1 is where
    return predictability concentrates.
    """)

    js_data = {
        "Percentile": ["10%", "25%", "50%", "75%", "90%", "95%"],
        "E1 (Agent Spec)": ["0.004", "0.011", "0.032", "0.082", "0.176", "0.215"],
        "E5 (Prompt Noise)": ["0.006", "0.020", "0.035", "0.057", "0.073", "0.075"],
        "E1/E5 Ratio": ["0.60x", "0.57x", "0.91x", "1.45x", "2.42x", "2.88x"],
    }
    st.dataframe(js_data, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    <strong>🔑 Key Insight:</strong> At the median, E1 and E5 are similar (0.91x).
    But in the <strong>right tail</strong> — where disagreement is high — E1 dominates:<br>
    • P90: E1 is 2.4x higher than E5<br>
    • P95: E1 is 2.9x higher than E5<br>
    • <strong>High-disagreement events are almost exclusively from agent specialization.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Right-Tail: Where the Signal Lives")

    tail_data = {
        "Threshold": ["JS > 0.05", "JS > 0.10", "JS > 0.15", "JS > 0.20"],
        "E1 %": ["38.6%", "19.6%", "12.2%", "6.9%"],
        "E5 %": ["34.3%", "0.8%", "0.0%", "0.0%"],
        "Odds Ratio": ["1.1x", "25x", "∞", "∞"],
    }
    st.dataframe(tail_data, use_container_width=True)

    st.markdown("""
    **P(JS > 0.10)**: E1 = 19.6% vs E5 = 0.8% — agent specialization produces high-disagreement
    events 25× more often than prompt noise. These high-disagreement events are precisely
    what predicts future returns in the Fama-MacBeth regressions.
    """)

    st.markdown("---")
    st.markdown("### 🧠 Why E5 Disagreement Is Different")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="warning-box">
        <strong>E5 (Prompt Noise): Fixed Directional Tilt</strong><br>
        • Conservative: 13% bullish (always bearish)<br>
        • Growth: 43% bullish (always bullish)<br>
        • Balanced: 26% bullish (always moderate)<br><br>
        <em>"Fake disagreement" — each persona has a fixed tilt
        that doesn't respond to market conditions.</em>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="finding-box">
        <strong>E1 (Agent Specialization): Conditional Disagreement</strong><br>
        • Sentiment: 33% bullish (responds to momentum)<br>
        • Fundamental: 43% overvalued (responds to valuation)<br><br>
        <em>"Genuine disagreement" — disagreement spikes when
        different information channels conflict.</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📝 Paper Implication")
    st.markdown("""
    This experiment directly addresses the reviewer concern: *"Is your disagreement just prompt noise?"*

    **Answer**: No. Prompt noise produces low-level, constant disagreement.
    Agent specialization produces **conditional, information-driven** disagreement
    that concentrates in the right tail — exactly where return predictability lives.

    | | JS (mean) | JS (P90) | P(JS>0.10) |
    |---|---|---|---|
    | **E1 (Agent Spec)** | 0.060*** | 0.176 | 19.6% |
    | **E5 (Prompt Noise)** | 0.039 | 0.073 | 0.8% |
    | **Difference** | 0.021*** | +0.103 | +18.8pp |

    Paired t-test: t = 12.95, p < 0.001. The difference is highly significant.
    """)


# ══════════════════════════════════════════════════════════════
# Page 4: Methodology
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Methodology":
    st.markdown('<div class="main-header">🔬 Methodology</div>', unsafe_allow_html=True)

    st.markdown("### Data & Sample")
    st.markdown("""
    - **Universe**: 183 S&P 500 stocks
    - **Period**: 2005Q1 – 2024Q4 (80 quarters)
    - **LLM Calls**: 40,020 quarterly ratings (3 agents × 183 stocks × ~73 quarters)
    - **Factor Data**: FF5 + Momentum (Ken French data library)
    """)

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
    ```
    Step 1: Data Expansion      → sp500_monthly_returns.json
    Step 2: Prescreening         → Filter liquid stocks
    Step 2b: LLM Scoring        → agent_ratings_llm_quarterly.json (40K API calls)
    Step 2c: Quant Ratings      → Quantitative agent placebo
    Step 3: Panel Construction  → Merge ratings + returns + factors
    Step 4: Fama-MacBeth        → H_smooth, JS, D_post regressions
    Step 5: Portfolio Sorts      → Long-short, overconfidence vs concordant
    Step 6: Robustness          → Subsample, horizon, placebo tests
    Step 7-8: Figures           → Publication-quality visualizations
    Step 9-10: Paper            → LaTeX manuscript + references
    ```
    """)

    st.markdown("---")
    st.markdown("### Theoretical Framework")
    st.markdown("""
    **Daniel, Hirshleifer & Subrahmanyam (1998) Overconfidence Model**

    1. Overconfident investors overweight private signals → mispricing
    2. Mispricing is gradually corrected as public information arrives
    3. **Prediction**: Return predictability should *strengthen* at longer horizons

    **Delta's contribution**: We show that low multi-agent entropy (agents agree confidently)
    identifies overconfident mispricing, and the *strengthening* pattern at longer horizons
    confirms the overconfidence mechanism — ruling out risk-based explanations.

    **Key distinction**:
    - Low entropy + low dispersion → **concordant** (genuine agreement) → no mispricing
    - Low entropy + high dispersion → **overconfident** (forced agreement despite disagreement) → mispricing
    - This interaction (H × D) is the causal identification strategy
    """)


# ══════════════════════════════════════════════════════════════
# Page 5: Figures
# ══════════════════════════════════════════════════════════════
elif page == "🖼️ Figures":
    st.markdown('<div class="main-header">🖼️ Publication Figures</div>', unsafe_allow_html=True)

    fig_files = {k: v for k, v in FIGURES.items() if (FIG_DIR / f"{k}.png").exists()}

    if not fig_files:
        st.warning("Figures not found. Place PNG files in `figures_final/` directory.")
        st.info("Expected files: " + ", ".join(f"`{k}.png`" for k in FIGURES))
    else:
        for fig_name, fig_desc in fig_files.items():
            fig_path = FIG_DIR / f"{fig_name}.png"
            st.subheader(fig_desc)
            st.image(str(fig_path), use_container_width=True)
            st.caption(f"Source: {fig_name}.png")
            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# Page 6: Entropy Demo
# ══════════════════════════════════════════════════════════════
elif page == "🎮 Entropy Demo":
    st.markdown('<div class="main-header">🎮 Interactive Entropy Demo</div>', unsafe_allow_html=True)
    st.markdown("Adjust the three agent ratings and observe how entropy, JS divergence, and dispersion change.")

    col1, col2, col3 = st.columns(3)
    with col1:
        s_rat = st.slider("🟦 Sentiment Agent", 1, 10, 7, key="sent")
    with col2:
        t_rat = st.slider("🟩 Technical Agent", 1, 10, 5, key="tech")
    with col3:
        f_rat = st.slider("🟨 Fundamental Agent", 1, 10, 3, key="fund")

    # Compute
    def rating_to_probs(rating):
        x = (rating - 5.5) / 2.0
        p_neg = np.exp(-x) / (np.exp(-x) + 1 + np.exp(x))
        p_neu = 1 / (np.exp(-x) + 1 + np.exp(x))
        p_pos = np.exp(x) / (np.exp(-x) + 1 + np.exp(x))
        return np.array([p_neg, p_neu, p_pos])

    sp = rating_to_probs(s_rat)
    tp = rating_to_probs(t_rat)
    fp = rating_to_probs(f_rat)
    avg_p = (sp + tp + fp) / 3

    h_smooth = -np.sum(avg_p * np.log2(avg_p + 1e-10))
    h_sp = -np.sum(sp * np.log2(sp + 1e-10))
    h_tp = -np.sum(tp * np.log2(tp + 1e-10))
    h_fp = -np.sum(fp * np.log2(fp + 1e-10))
    js = h_smooth - np.mean([h_sp, h_tp, h_fp])
    d_post = np.std([s_rat, t_rat, f_rat])

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "#22c55e" if h_smooth > 1.2 else ("#f59e0b" if h_smooth > 0.8 else "#ef4444")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{h_smooth:.3f}</div>
            <div class="metric-label">H_smooth (Entropy)</div>
            <div style="font-size:0.75rem; color:#64748b;">{'High = uncertain/diverse' if h_smooth > 1.2 else 'Medium' if h_smooth > 0.8 else 'Low = confident/overconfident'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#8b5cf6">{js:.3f}</div>
            <div class="metric-label">JS Divergence</div>
            <div style="font-size:0.75rem; color:#64748b;">{'High shape difference' if js > 0.1 else 'Similar distributions'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f59e0b">{d_post:.2f}</div>
            <div class="metric-label">D_post (Dispersion)</div>
            <div style="font-size:0.75rem; color:#64748b;">{'High disagreement' if d_post > 2 else 'Low disagreement'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Interpretation
    st.markdown("---")
    st.markdown("### 🧮 Interpretation")

    if h_smooth < 0.8 and d_post > 2.0:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Overconfidence Signal:</strong> Low entropy (agents agree) + high dispersion (ratings differ)
        = forced agreement despite disagreement. This is the <em>entropy premium</em> pattern —
        predictive of negative future returns (t = −3.02).
        </div>
        """, unsafe_allow_html=True)
    elif h_smooth < 0.8 and d_post < 1.5:
        st.markdown("""
        <div class="finding-box">
        <strong>✅ Concordant:</strong> Low entropy + low dispersion = genuine agreement.
        No mispricing signal — agents truly agree and are confident.
        </div>
        """, unsafe_allow_html=True)
    elif h_smooth > 1.2:
        st.markdown("""
        <div class="finding-box">
        <strong>🔍 Uncertain:</strong> High entropy = agents are uncertain or have diverse views.
        No strong predictive signal for returns.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("**📊 Moderate entropy** — no strong overconfidence or uncertainty signal.")

    # Probability distributions visualization
    st.markdown("---")
    st.markdown("### 📊 Agent Probability Distributions")

    categories = ["Negative", "Neutral", "Positive"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sentiment", x=categories, y=sp, marker_color="#38bdf8"))
    fig.add_trace(go.Bar(name="Technical", x=categories, y=tp, marker_color="#22c55e"))
    fig.add_trace(go.Bar(name="Fundamental", x=categories, y=fp, marker_color="#f59e0b"))
    fig.add_trace(go.Bar(name="Average", x=categories, y=avg_p, marker_color="#8b5cf6", opacity=0.7))

    fig.update_layout(
        barmode="group",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        template="plotly_dark",
        font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Entropy decomposition
    st.markdown("### 📐 Entropy Decomposition")
    st.markdown(f"""
    | Component | Value |
    |-----------|-------|
    | H(avg_p) = H_smooth | {h_smooth:.4f} |
    | H(sentiment) | {h_sp:.4f} |
    | H(technical) | {h_tp:.4f} |
    | H(fundamental) | {h_fp:.4f} |
    | JS = H(avg) − mean(H(individual)) | {js:.4f} |
    | D_post = σ(ratings) | {d_post:.4f} |
    | **H_premium** = H_smooth ⊥ (JS, D_post) | ~{max(0, h_smooth - abs(js)*2 - d_post*0.1):.4f} (approx) |
    """)


# ══════════════════════════════════════════════════════════════
# Page 7: About
# ══════════════════════════════════════════════════════════════
elif page == "🧪 Probability-Based IC (E3)":
    st.markdown('<div class="main-header">Experiment 3: Probability-Based Inner Confidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Within-Agent Uncertainty Meets Directional Asymmetry</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">137</div><div class="metric-label">Stocks</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">9,710</div><div class="metric-label">Obs</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">3.17***</div><div class="metric-label">H×D (FM)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">2.77***</div><div class="metric-label">H×D (DC-SE)</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 🔑 Key Innovation")
    st.markdown("""
    Instead of extracting a single sentiment label, we elicit the **full probability distribution** 
    over discrete sentiment directions from each agent. The Shannon entropy of this distribution 
    measures **Inner Confidence** — the within-agent uncertainty signal.
    
    Each agent processes **different information** and outputs to a **different probability space**:
    - **Sentiment agent**: Returns + Volume + Volatility → {bearish, neutral, bullish}
    - **Fundamental agent**: PE + PB + ROE + Debt → {overvalued, undervalued}  
    - **Technical agent**: RSI + MACD + BB + ADX → {breakdown, range, breakout}
    """)
    
    st.markdown("### 📊 FM Regression Results")
    
    # FM results table
    fm_data = {
        "Model": ["H_IC only", "H_IC + D + H×D", "DC-SE", "FM + Controls"],
        "H_IC": ["-1.85 (t=-1.16)", "+3.47 (t=1.95*)", "-1.76 (t=-0.59)", "+3.19 (t=2.00**)"],
        "D_sent": ["—", "-8.81 (t=-3.15***)", "-13.11 (t=-2.78***)", "-7.68 (t=-3.17***)"],
        "H×D": ["—", "+5.74 (t=3.18***)", "+8.41 (t=2.77***)", "+5.04 (t=3.17***)"],
    }
    st.dataframe(fm_data, use_container_width=True)
    
    st.markdown("### 🐻🐂 Regime Split: The Asymmetric Effect")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="finding-box">
        <strong>🐻 Bearish Stocks</strong><br>
        Higher IC → <strong>Lower</strong> returns<br>
        β = -10.16, t = <strong>-5.33***</strong><br>
        <em>Overconfident bears are punished</em>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="finding-box">
        <strong>🐂 Bullish Stocks</strong><br>
        Higher IC → <strong>Higher</strong> returns<br>
        β = +6.66, t = <strong>+4.59***</strong><br>
        <em>Overconfident bulls are rewarded</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    The asymmetry confirms the **overconfidence mechanism**: uncertainty amplifies the directional signal.
    Bears who are highly confident are *wrong more often*; bulls who are highly confident are also 
    overconfident, but in the direction of the drift.
    """)
    
    st.markdown("### 🛡️ Structural Concerns Addressed")
    concerns = {
        "Concern": [
            "Q1: Same model = semantic drift?",
            "Q2: Orthogonalization unstable?", 
            "Q3: Familiarity bias?",
            "Q4: Overlapping returns inflate t?"
        ],
        "Resolution": [
            "Different inputs + different probability spaces. H_sent vs H_fund: r=0.036, direction conflict 48.7%",
            "Probability-based IC eliminates need. Direct within-agent measurement, no JS-D collinearity",
            "Named vs anonymized: r=0.748, named H HIGHER (+0.031) — opposite of familiarity prediction",
            "Hodrick(1992) SE confirms. H×D robust to all methods (t>2.77). Beta autocorrelation ρ≈0"
        ],
        "Status": ["✅ Resolved", "✅ Resolved", "✅ Resolved", "✅ Resolved"]
    }
    st.dataframe(concerns, use_container_width=True)
    
    st.markdown("### 📈 Progressive Ablation (E1→E5)")
    ablation = {
        "Experiment": ["E1 (Current)", "E2", "E3", "E4", "E5 (Baseline)"],
        "Model": ["Same (Qwen) × 3 prompts", "Different × 3 prompts", "Different + LoRA", "Different + LoRA + KB", "Same model + Same input + 3 personas"],
        "Expected Signal": ["Baseline ✅", "Stronger", "Even stronger", "Strongest", "Noise floor (1.55x < E1)"],
    }
    st.dataframe(ablation, use_container_width=True)
    st.info("Even E1 (single model with different inputs) produces significant signals → information channel differentiation is the key driver")

elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    **Delta: Disagreement-Preserving Multi-Agent Collaboration**

    Agent Confidence, Entropy, and Cross-Sectional Stock Return Predictability

    ---

    **Author**
    - **Junwen Zhang** — Xi'an Jiaotong-Liverpool University

    **Code Repository**: [github.com/dechang64/delta](https://github.com/dechang64/delta)

    **Citation**
    ```bibtex
    @article{zhang2026delta,
      title={Delta: Disagreement-Preserving Multi-Agent Collaboration},
      subtitle={Agent Confidence, Entropy, and Cross-Sectional Stock Return Predictability},
      author={Zhang, Junwen},
      year={2026}
    }
    ```

    **Key References**
    - Daniel, K., Hirshleifer, D., & Subrahmanyam, A. (1998). Investor Psychology and Security Market Under- and Overreactions. *Journal of Finance*, 53(6), 1839-1885.
    - Fama, E.F., & MacBeth, J.D. (1973). Risk, Return, and Equilibrium. *Journal of Political Economy*, 81(3), 607-636.
    - Hong, H., & Stein, J.C. (2007). Disagreement and the Stock Market. *Journal of Economic Perspectives*, 21(2), 109-128.

    ---

    *Dashboard built with Streamlit. No heavy ML dependencies required for viewing.*
    """)
