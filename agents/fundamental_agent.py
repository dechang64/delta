"""
agents/fundamental_agent.py — Fundamental Analysis Agent for Delta.

Specializes in financial statements, valuation, and industry analysis.
Fine-tuned on financial analysis labeled data.
RAG: earnings reports, industry reports, macro indicators.
"""

from .base_agent import BaseAgent


class FundamentalAgent(BaseAgent):
    """Agent that evaluates stocks from a fundamental analysis perspective."""

    @property
    def domain(self) -> str:
        return "fundamental"

    @property
    def system_prompt(self) -> str:
        return """You are a Fundamental Analyst specializing in financial statement analysis and valuation.

Your expertise:
- Earnings quality and growth trajectory
- Valuation metrics (P/E, P/B, EV/EBITDA, DCF)
- Balance sheet strength and cash flow analysis
- Industry positioning and competitive moats

Your perspective focuses on:
1. **Earnings**: Revenue growth, margin trends, earnings surprises
2. **Valuation**: Relative and absolute valuation vs peers
3. **Quality**: ROE, debt levels, free cash flow generation
4. **Industry**: Competitive dynamics, regulatory environment, market share

You are NOT a sentiment analyst or technical analyst. You focus purely on
the FUNDAMENTAL VALUE and QUALITY of this business.

Important: Base your assessment on information available up to the given date.
Do not use any future information."""

    def _format_info(self, info: dict) -> str:
        sections = []

        if "pe_ratio" in info:
            sections.append(f"### P/E Ratio\n{info['pe_ratio']:.1f}x")

        if "pb_ratio" in info:
            sections.append(f"### P/B Ratio\n{info['pb_ratio']:.2f}x")

        if "revenue_growth" in info:
            sections.append(f"### Revenue Growth (YoY)\n{info['revenue_growth']:.1%}")

        if "earnings_growth" in info:
            sections.append(f"### Earnings Growth (YoY)\n{info['earnings_growth']:.1%}")

        if "roe" in info:
            sections.append(f"### ROE\n{info['roe']:.1%}")

        if "debt_to_equity" in info:
            sections.append(f"### Debt/Equity\n{info['debt_to_equity']:.2f}")

        if "free_cash_flow" in info:
            sections.append(f"### Free Cash Flow\n${info['free_cash_flow']:,.0f}M")

        if "dividend_yield" in info:
            sections.append(f"### Dividend Yield\n{info['dividend_yield']:.2%}")

        if "industry" in info:
            sections.append(f"### Industry\n{info['industry']}")

        if "market_cap" in info:
            sections.append(f"### Market Cap\n${info['market_cap']:,.0f}B")

        return "\n\n".join(sections) if sections else "Limited fundamental data available."
