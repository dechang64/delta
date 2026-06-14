"""
agents/__init__.py — Delta Agent Factory.

Creates agents for different experiment groups:
  - Group A: Same model, same prompt variants (v1 baseline)
  - Group B: Same model, different LoRA + different RAG (v2 recommended)
  - Group C: Different models, different LoRA + RAG (full heterogeneity)
"""

from .base_agent import BaseAgent
from .sentiment_agent import SentimentAgent
from .technical_agent import TechnicalAgent
from .fundamental_agent import FundamentalAgent


AGENT_CLASSES = {
    "sentiment": SentimentAgent,
    "technical": TechnicalAgent,
    "fundamental": FundamentalAgent,
}


def create_agent_group(group: str, config: dict = None):
    """Create a group of 3 agents for the specified experiment.

    Args:
        group: "A" (prompt-only), "B" (LoRA+RAG), "C" (different models)
        config: Configuration dict with API keys, model paths, etc.

    Returns:
        List of 3 Agent instances (sentiment, technical, fundamental)
    """
    config = config or {}
    api_key = config.get("api_key")
    api_base = config.get("api_base")

    if group == "A":
        # Group A: Same model, different prompts only (v1 baseline)
        # NO LoRA, NO RAG — disagreement is pure semantic drift
        return [
            SentimentAgent("Sentiment-v1", api_key=api_key, api_base=api_base),
            TechnicalAgent("Technical-v1", api_key=api_key, api_base=api_base),
            FundamentalAgent("Fundamental-v1", api_key=api_key, api_base=api_base),
        ]

    elif group == "B":
        # Group B: Same base model, different LoRA + different RAG (v2 recommended)
        # Disagreement comes from professional perspective differences
        model = config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        lora_dir = config.get("lora_dir", "agents/lora")
        kb_dir = config.get("kb_dir", "rag/knowledge_bases")

        return [
            SentimentAgent(
                "Sentiment-LoRA",
                model_name=model,
                lora_path=f"{lora_dir}/sentiment",
                kb_dir=f"{kb_dir}/sentiment",
                api_key=api_key,
                api_base=api_base,
            ),
            TechnicalAgent(
                "Technical-LoRA",
                model_name=model,
                lora_path=f"{lora_dir}/technical",
                kb_dir=f"{kb_dir}/technical",
                api_key=api_key,
                api_base=api_base,
            ),
            FundamentalAgent(
                "Fundamental-LoRA",
                model_name=model,
                lora_path=f"{lora_dir}/fundamental",
                kb_dir=f"{kb_dir}/fundamental",
                api_key=api_key,
                api_base=api_base,
            ),
        ]

    elif group == "C":
        # Group C: Different base models + different LoRA + RAG
        # Disagreement from both model heterogeneity AND perspective differences
        models = config.get("models", {
            "sentiment": "Qwen/Qwen2.5-7B-Instruct",
            "technical": "microsoft/Phi-3.5-mini-instruct",
            "fundamental": "google/gemma-2-9b-it",
        })
        lora_dir = config.get("lora_dir", "agents/lora")
        kb_dir = config.get("kb_dir", "rag/knowledge_bases")

        return [
            SentimentAgent(
                "Sentiment-Qwen",
                model_name=models["sentiment"],
                lora_path=f"{lora_dir}/sentiment",
                kb_dir=f"{kb_dir}/sentiment",
                api_key=api_key,
                api_base=api_base,
            ),
            TechnicalAgent(
                "Technical-Phi",
                model_name=models["technical"],
                lora_path=f"{lora_dir}/technical",
                kb_dir=f"{kb_dir}/technical",
                api_key=api_key,
                api_base=api_base,
            ),
            FundamentalAgent(
                "Fundamental-Gemma",
                model_name=models["fundamental"],
                lora_path=f"{lora_dir}/fundamental",
                kb_dir=f"{kb_dir}/fundamental",
                api_key=api_key,
                api_base=api_base,
            ),
        ]

    else:
        raise ValueError(f"Unknown group: {group}. Must be A, B, or C.")
