# Delta: Disagreement-Preserving Multi-Agent Collaboration

Code and data for the paper:

**Delta: Disagreement-Preserving Multi-Agent Collaboration — Agent Confidence, Entropy, and Cross-Sectional Stock Return Predictability**

Dechang Xu (Soochow University) · Junwen Zhang (Xi'an Jiaotong-Liverpool University)

## Paper

- `Delta_JFE_Paper_v4.docx` — Latest manuscript

## Code

- `step1_expand_data.py` — Data expansion and preprocessing
- `step2_prescreen.py` — Stock prescreening
- `step2b_llm_scoring.py` — LLM agent scoring
- `step2c_quant_ratings.py` — Quantitative agent ratings
- `step3456_analysis_llm.py` — LLM analysis (FM regressions, portfolio sorts)
- `step8_figures_llm_v2.py` — Figure generation
- `llm_quarterly_batch.py` — Quarterly LLM batch scoring (40,020 API calls)

## Figures

- `figures_final/` — Publication-quality figures matching the paper

## Citation

```bibtex
@article{xu2026delta,
  title={Delta: Disagreement-Preserving Multi-Agent Collaboration},
  subtitle={Agent Confidence, Entropy, and Cross-Sectional Stock Return Predictability},
  author={Xu, Dechang and Zhang, Junwen},
  year={2026}
}
```
