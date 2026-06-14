# E3: Probability-Based Uncertainty and Stock Returns

## Overview
Experiment 3 extends the Delta framework by using LLM probability outputs to construct **Inner Confidence** (IC) measures — the entropy of the agent's token-level probability distribution over discrete sentiment directions.

## Key Innovation
Instead of extracting a single sentiment label (bullish/bearish/neutral), we ask the LLM to:
1. Rate each feature dimension (sentiment, fundamental) on a discrete scale
2. The **probability distribution** over these ratings reveals the agent's **uncertainty**
3. **Shannon entropy H** of this distribution = Inner Confidence measure

## Results (137 stocks, 2005-2024, quarterly)

| Model | H_sent | D_sent | H×D | ln_price | momentum |
|-------|--------|--------|-----|----------|----------|
| FM (NW SE) | +2.00** | -3.17*** | +3.17*** | — | — |
| FM + controls | +2.00** | -3.17*** | +3.17*** | -2.88*** | -1.73* |
| DC-SE | — | -2.78*** | +2.77*** | — | — |

### Regime Split
- **Bearish**: H_sent β=-10.16 (t=-5.33***) → uncertainty hurts bears
- **Bullish**: H_sent β=+6.66 (t=+4.59***) → uncertainty helps bulls

## Files
- `feature_engine.py` — Feature computation (sentiment + fundamental)
- `scoring_v2.py` — Probability-based LLM scoring
- `multi_model_api.py` — Multi-provider API support
- `batch_e3.py` — Batch scoring runner
- `e3_combined_panel.csv` — Final panel data
- `e3_final_summary.txt` — Results summary
- `e3_high_named_ckpt.json` — High disagreement ratings checkpoint
- `e3_med_named_ckpt.json` — Medium disagreement ratings checkpoint
