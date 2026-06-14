"""
agents/base_agent.py — Base class for Delta multi-agent scoring system.

All agents share the same base model (Qwen2.5-7B-Instruct) but differ in:
  1. LoRA adapter (domain-specific fine-tuning)
  2. RAG knowledge base (domain-specific retrieval)
  3. System prompt (domain-specific persona)

This design ensures disagreement comes from professional perspective differences,
not model heterogeneity or semantic drift.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Optional


class BaseAgent(ABC):
    """Base class for Delta scoring agents."""

    def __init__(
        self,
        name: str,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_path: Optional[str] = None,
        kb_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_local: bool = False,
    ):
        self.name = name
        self.model_name = model_name
        self.lora_path = lora_path
        self.kb_dir = kb_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.use_local = use_local

        # RAG retriever (lazy init)
        self._retriever = None

        # Scoring history
        self.history = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Domain-specific system prompt."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier: sentiment / technical / fundamental."""
        pass

    def retrieve_context(self, ticker: str, date: str, top_k: int = 5) -> str:
        """Retrieve relevant context from domain knowledge base.

        CRITICAL: Only returns documents dated BEFORE the scoring date
        to prevent look-ahead bias.
        """
        if self.kb_dir is None:
            return ""

        # Lazy init retriever
        if self._retriever is None:
            from rag.retriever import KnowledgeRetriever
            self._retriever = KnowledgeRetriever(
                kb_dir=self.kb_dir,
                cutoff_date=date,  # No look-ahead!
            )

        # Update cutoff for this query
        self._retriever.cutoff_date = date

        query = f"{ticker} {date} {self.domain}"
        docs = self._retriever.search(query, top_k=top_k)
        return "\n".join(docs) if docs else ""

    def build_prompt(self, ticker: str, info: dict, date: str) -> str:
        """Build the full scoring prompt with RAG context.

        Args:
            ticker: Stock ticker symbol
            info: Dict with stock data (price, volume, financials, news, etc.)
            date: Scoring date (YYYY-MM-DD)
        """
        # Retrieve domain-specific context
        rag_context = self.retrieve_context(ticker, date)

        # Build info section based on domain
        info_section = self._format_info(info)

        prompt = f"""{self.system_prompt}

## Context (knowledge base retrieval, dated before {date})
{rag_context if rag_context else "No additional context available."}

## Stock Information for {ticker} as of {date}
{info_section}

## Task
Rate {ticker} on a 1-10 scale where:
- 1-3: Strongly bearish
- 4-5: Mildly bearish
- 6-7: Mildly bullish
- 8-10: Strongly bullish

Provide your rating as a single integer. Then briefly explain your reasoning (2-3 sentences).

Output format:
RATING: <integer>
REASONING: <text>"""
        return prompt

    @abstractmethod
    def _format_info(self, info: dict) -> str:
        """Format stock info for this agent's domain perspective."""
        pass

    def score(self, ticker: str, info: dict, date: str) -> dict:
        """Score a stock from this agent's perspective.

        Returns:
            dict with keys: rating (int), reasoning (str), agent (str), date (str)
        """
        prompt = self.build_prompt(ticker, info, date)

        if self.use_local:
            rating, reasoning = self._score_local(prompt)
        else:
            rating, reasoning = self._score_api(prompt)

        result = {
            "agent": self.name,
            "domain": self.domain,
            "ticker": ticker,
            "date": date,
            "rating": rating,
            "reasoning": reasoning,
            "model": self.model_name,
            "lora": self.lora_path is not None,
            "rag": self.kb_dir is not None,
        }
        self.history.append(result)
        return result

    def _score_api(self, prompt: str) -> tuple:
        """Score via OpenAI-compatible API (vLLM / OpenAI / DeepSeek)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            text = response.choices[0].message.content.strip()
            rating, reasoning = self._parse_response(text)
            return rating, reasoning

        except Exception as e:
            print(f"  [ERROR] {self.name} API call failed: {e}")
            return 5, f"API error: {e}"

    def _score_local(self, prompt: str) -> tuple:
        """Score via local model (vLLM or transformers)."""
        try:
            import requests
            # vLLM OpenAI-compatible server
            resp = requests.post(
                f"{self.api_base}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
                timeout=30,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
            rating, reasoning = self._parse_response(text)
            return rating, reasoning

        except Exception as e:
            print(f"  [ERROR] {self.name} local inference failed: {e}")
            return 5, f"Local error: {e}"

    @staticmethod
    def _parse_response(text: str) -> tuple:
        """Parse model response into (rating, reasoning)."""
        rating = 5  # default
        reasoning = text

        # Try to extract "RATING: <int>"
        import re
        match = re.search(r'RATING:\s*(\d+)', text)
        if match:
            rating = max(1, min(10, int(match.group(1))))

        # Try to extract "REASONING: <text>"
        match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()

        return rating, reasoning
