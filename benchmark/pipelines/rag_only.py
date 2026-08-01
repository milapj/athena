import time
from typing import Optional

from benchmark.datasets.base import BenchmarkSample
from benchmark.evaluation.cost_tracker import estimate_cost
from benchmark.evaluation.metrics import parse_choice_letter
from benchmark.models.openrouter import OpenRouterModel
from benchmark.pipelines.base import Pipeline, PipelineResult
from benchmark.pipelines.retrieval_cache import RetrievalCache
from benchmark.vectorstore.qdrant_store import QdrantVectorStore


SYSTEM_PROMPT = """You are a precise reasoning assistant. Answer questions based strictly on the provided context. If the context does not contain enough information, say so. Do not fabricate information."""

MC_PROMPT_TEMPLATE = """Given the following context, answer the question by selecting the best choice.

Context:
{context}

Question: {question}

Choices:
{choices_str}

Respond with ONLY the letter of the correct answer (e.g., A, B, C, etc.)."""

OPEN_PROMPT_TEMPLATE = """Given the following context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""


class RAGOnlyPipeline(Pipeline):
    def __init__(self, vectorstore: QdrantVectorStore, generator: OpenRouterModel, top_k: int = 5,
                 retrieval_cache: RetrievalCache = None):
        self.vectorstore = vectorstore
        self.generator = generator
        self.top_k = top_k
        self.retrieval_cache = retrieval_cache

    def run(self, sample: BenchmarkSample) -> PipelineResult:
        start = time.time()

        # Use direct context if sample provides it (e.g., MuSR), else retrieve from vectorstore
        if sample.context_documents:
            contexts = sample.context_documents
        elif self.retrieval_cache and self.retrieval_cache.get(sample.id) is not None:
            contexts = self.retrieval_cache.get(sample.id)
        else:
            results = self.vectorstore.search(sample.query, top_k=self.top_k)
            contexts = [r.text for r in results]
            if self.retrieval_cache:
                self.retrieval_cache.put(sample.id, contexts)
        context_str = "\n\n---\n\n".join(contexts)

        # Format prompt
        if sample.is_multiple_choice:
            letters = "ABCDEFGHIJ"
            choices_str = "\n".join(
                f"{letters[i]}) {c}" for i, c in enumerate(sample.choices)
            )
            prompt = MC_PROMPT_TEMPLATE.format(
                context=context_str,
                question=sample.query,
                choices_str=choices_str,
            )
        else:
            prompt = OPEN_PROMPT_TEMPLATE.format(
                context=context_str,
                question=sample.query,
            )

        # Generate
        response = self.generator.inference(prompt, system_prompt=SYSTEM_PROMPT)
        answer = response["choices"][0]["message"]["content"]
        latency = time.time() - start

        # Extract token usage and cost
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = estimate_cost(self.generator.engine, prompt_tokens, completion_tokens)

        # Parse choice for MC
        choice_idx = None
        if sample.is_multiple_choice:
            choice_idx = parse_choice_letter(answer, len(sample.choices))

        return PipelineResult(
            answer=answer,
            selected_choice_index=choice_idx,
            retrieved_contexts=contexts,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
