import os
from typing import List, Dict

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ContextPrecision, AnswerCorrectness

from benchmark.config import OPENROUTER_BASE_URL


def _get_ragas_llm() -> ChatOpenAI:
    """Create an LLM judge routed through OpenRouter (Gemini Flash 2.0 — cheap)."""
    return ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/grayscope/athena",
            "X-Title": "Athena Benchmark RAGAS",
        },
    )


def _get_ragas_embeddings() -> OpenAIEmbeddings:
    """Create embeddings using OpenAI directly (OpenRouter doesn't serve embeddings)."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


class RagasEvaluator:
    """Wraps RAGAS evaluation, using OpenRouter as the LLM judge."""

    def __init__(self):
        self.llm = _get_ragas_llm()
        self.embeddings = _get_ragas_embeddings()
        self.metrics = [Faithfulness(), ContextPrecision(), AnswerCorrectness()]

    def evaluate_batch(self, results: List[Dict]) -> pd.DataFrame:
        """
        Evaluate a list of result dicts with keys:
        - query: str
        - answer: str
        - contexts: List[str]
        - ground_truth: str
        """
        samples = []
        for r in results:
            samples.append(
                SingleTurnSample(
                    user_input=r["query"],
                    response=r["answer"],
                    retrieved_contexts=r["contexts"],
                    reference=r["ground_truth"],
                )
            )
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )
        return result.to_pandas()
