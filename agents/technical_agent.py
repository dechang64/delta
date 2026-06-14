"""
agents/technical_agent.py — Technical Analysis Agent for Delta.

Specializes in price patterns, volume signals, and momentum indicators.
Fine-tuned on technical analysis labeled data.
RAG: technical indicator descriptions, historical pattern cases.
"""

from .base_agent import BaseAgent


class TechnicalAgent(BaseAgent):
    """Agent that evaluates stocks from a technical analysis perspective."""

    @property
    def domain(self) -> str:
        return "technical"

    @property
    def system_prompt(self) -> str:
        return """You are a Technical Analyst specializing in price action and quantitative signals.

Your expertise:
- Chart pattern recognition and trend analysis
- Momentum indicators (RSI, MACD, moving averages)
- Volume analysis and money flow
- Support/resistance levels and breakout signals

Your perspective focuses on:
1. **Price trends**: Moving averages, trend lines, momentum
2. **Volume**: Unusual volume, accumulation/distribution
3. **Oscillators**: RSI, MACD, Stochastic readings
4. **Patterns**: Breakouts, reversals, consolidation phases

You are NOT a sentiment analyst or fundamental analyst. You focus purely on
what the PRICE and VOLUME data tells you about future direction.

Important: Base your assessment on information available up to the given date.
Do not use any future information."""

    def _format_info(self, info: dict) -> str:
        sections = []

        if "price" in info:
            sections.append(f"### Current Price\n{info['price']:.2f}")

        if "price_change_1m" in info:
            sections.append(f"### 1-Month Return\n{info['price_change_1m']:.1%}")

        if "price_change_3m" in info:
            sections.append(f"### 3-Month Return\n{info['price_change_3m']:.1%}")

        if "price_change_6m" in info:
            sections.append(f"### 6-Month Return\n{info['price_change_6m']:.1%}")

        if "volume_ratio" in info:
            sections.append(f"### Volume Ratio (vs 20d avg)\n{info['volume_ratio']:.2f}x")

        if "rsi_14" in info:
            rsi = info['rsi_14']
            signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            sections.append(f"### RSI(14)\n{rsi:.1f} ({signal})")

        if "ma_signal" in info:
            sections.append(f"### Moving Average Signal\n{info['ma_signal']}")

        if "macd_signal" in info:
            sections.append(f"### MACD\n{info['macd_signal']}")

        if "volatility_20d" in info:
            sections.append(f"### 20-Day Volatility\n{info['volatility_20d']:.1%}")

        return "\n\n".join(sections) if sections else "Limited technical data available."
