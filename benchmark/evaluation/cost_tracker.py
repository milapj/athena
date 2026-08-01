from dataclasses import dataclass, field
from typing import List, Dict


# Approximate OpenRouter pricing per 1M tokens (as of early 2026)
# Update these as needed
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using known pricing."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def aggregate_costs(results: list) -> dict:
    """Summarize costs from a list of BenchmarkResult objects."""
    total_cost = sum(r.cost_usd + r.tree_build_cost_usd for r in results)
    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_completion_tokens = sum(r.completion_tokens for r in results)
    return {
        "total_cost_usd": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "num_queries": len(results),
        "avg_cost_per_query": total_cost / len(results) if results else 0.0,
    }
