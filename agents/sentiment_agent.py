"""
agents/sentiment_agent.py — Sentiment Analysis Agent for Delta.

Specializes in market sentiment, news narrative, and social media signals.
Fine-tuned on financial news sentiment data.
RAG: news articles, analyst reports, social media.
"""

from .base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    """Agent that evaluates stocks from a market sentiment perspective."""

    @property
    def domain(self) -> str:
        return "sentiment"

    @property
    def system_prompt(self) -> str:
        return """You are a Sentiment Analyst specializing in market psychology and narrative analysis.

Your expertise:
- Analyzing market sentiment from news, social media, and analyst commentary
- Identifying momentum shifts in market narrative
- Assessing how public perception affects stock prices
- Evaluating the emotional tone of market discourse

Your perspective focuses on:
1. **News flow**: Recent positive/negative news coverage
2. **Social sentiment**: Reddit, Twitter/X, StockTwits buzz
3. **Analyst consensus**: Buy/sell/hold recommendations and tone
4. **Market narrative**: Dominant themes and stories around the stock

You are NOT a technical analyst or fundamental analyst. You focus purely on
how market participants FEEL about and REACT to this stock.

Important: Base your assessment on information available up to the given date.
Do not use any future information."""

    def _format_info(self, info: dict) -> str:
        sections = []

        if "news" in info:
            sections.append(f"### Recent News\n{info['news']}")

        if "analyst_rating" in info:
            sections.append(f"### Analyst Consensus\n{info['analyst_rating']}")

        if "social_sentiment" in info:
            sections.append(f"### Social Media Sentiment\n{info['social_sentiment']}")

        if "price_change_1m" in info:
            sections.append(f"### Recent Price Action (sentiment driver)\n1-month return: {info['price_change_1m']:.1%}")

        if "volume_ratio" in info:
            sections.append(f"### Volume\nVolume ratio vs 20-day avg: {info['volume_ratio']:.2f}x")

        return "\n\n".join(sections) if sections else "Limited sentiment data available."
