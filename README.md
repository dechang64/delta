# Delta: LLM Agent Uncertainty and Stock Returns

## Overview
This repository contains the code and data for **"Delta: Measuring and Pricing LLM Agent Uncertainty in Financial Markets"** (JFE submission).

## Core Finding
LLM agent **Inner Confidence** (token-level probability entropy) predicts future stock returns through an **asymmetric interaction** with sentiment direction:
- Bearish + High Uncertainty → Overconfident bears → Lower returns
- Bullish + High Uncertainty → Overconfident bulls → Higher returns

## Repository Structure

```
delta_jfe/
├── step1_expand_data.py          # S&P 500 data collection
├── step2_prescreen.py            # Analyst disagreement classification
├── step2b_llm_scoring.py         # LLM agent scoring (v1)
├── step3456_analysis_v2.py       # Main analysis pipeline
├── step78_figures_ibes.py        # IBES comparison figures
├── step8_figures_llm_v2.py       # LLM analysis figures
├── paper/                        # LaTeX paper source
├── e3/                           # E3: Probability-based IC (NEW)
│   ├── feature_engine.py         #   Feature computation
│   ├── scoring_v2.py             #   Probability-based scoring
│   ├── e3_combined_panel.csv     #   Final panel data
│   └── e3_final_summary.txt      #   Results summary
└── ashare/                       # A-share cross-market validation
    ├── step1_ashare_data.py      #   A-share data collection
    ├── step2_ashare_llm_scoring.py # A-share LLM scoring
    └── step3_ashare_analysis.py  # A-share analysis
```

## Key Results

### E3: Probability-Based Inner Confidence (137 stocks, 2005-2024)
| Model | H_sent | D_sent | H×D |
|-------|--------|--------|-----|
| FM (NW) | +2.00** | -3.17*** | +3.17*** |
| DC-SE | — | -2.78*** | +2.77*** |

Regime split: Bearish H→returns (t=-5.33***), Bullish H→returns (t=+4.59***)

## Setup
```bash
pip install openai pandas numpy statsmodels matplotlib
export DASHSCOPE_API_KEY=your_key  # For Qwen API
```
