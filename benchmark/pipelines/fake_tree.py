"""
Fake tree pipeline — generator sees chunks + a RANDOM tree from a different sample.

Same prompt template as RAGPlusTreePipeline, but the tree is from a random
different sample. This is an ablation study to test whether the tree content
matters or if the improvement is just from prompt engineering.
"""

import random
import time
import os
import json
from typing import Optional

from benchmark.datasets.base import BenchmarkSample
from benchmark.evaluation.cost_tracker import estimate_cost
from benchmark.evaluation.metrics import parse_choice_letter
from benchmark.models.openrouter import OpenRouterModel
from benchmark.pipelines.base import Pipeline, PipelineResult
from benchmark.pipelines.retrieval_cache import RetrievalCache
from benchmark.vectorstore.qdrant_store import QdrantVectorStore


SYSTEM_PROMPT = """You are a precise reasoning assistant. Answer questions based strictly on the provided context and reasoning tree. The reasoning tree shows step-by-step logical deductions from the evidence. Use it to guide your answer. Do not fabricate information."""

MC_PROMPT_TEMPLATE = """Given the following context and reasoning tree, answer the question by selecting the best choice.

Context:
{context}

Reasoning Tree:
{tree}

Question: {question}

Choices:
{choices_str}

Respond with ONLY the letter of the correct answer (e.g., A, B, C, etc.)."""

OPEN_PROMPT_TEMPLATE = """Given the following context and reasoning tree, answer the question accurately and concisely.

Context:
{context}

Reasoning Tree:
{tree}

Question: {question}

Answer:"""


class FakeTreePipeline(Pipeline):
    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        generator: OpenRouterModel,
        tree_cache_dir: str,
        top_k: int = 5,
        retrieval_cache: RetrievalCache = None,
        seed: int = 42,
    ):
        self.vectorstore = vectorstore
        self.generator = generator
        self.tree_cache_dir = tree_cache_dir
        self.top_k = top_k
        self.retrieval_cache = retrieval_cache
        self.rng = random.Random(seed)
        self._all_trees = None

    def _load_all_trees(self):
        """Load all cached trees and index by sample ID."""
        if self._all_trees is not None:
            return self._all_trees
        self._all_trees = {}
        if not os.path.exists(self.tree_cache_dir):
            return self._all_trees
        for fname in os.listdir(self.tree_cache_dir):
            if not fname.endswith('.json'):
                continue
            sample_id = fname.split('_')[0]
            with open(os.path.join(self.tree_cache_dir, fname)) as f:
                data = json.load(f)
            tree_str = data.get('tree_str', '')
            if tree_str and len(tree_str) > 40:  # Skip empty trees
                self._all_trees[sample_id] = tree_str
        return self._all_trees

    def _get_random_tree(self, exclude_sample_id: str) -> str:
        """Get a random tree from a DIFFERENT sample."""
        trees = self._load_all_trees()
        candidates = [sid for sid in trees if sid != exclude_sample_id]
        if not candidates:
            return "root | Deduced Root Conclusion"
        chosen = self.rng.choice(candidates)
        return trees[chosen]

    def run(self, sample: BenchmarkSample) -> PipelineResult:
        start = time.time()

        # Retrieve chunks (same as A and B configs)
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

        # Get a RANDOM tree from a different sample
        tree_str = self._get_random_tree(sample.id)

        # Format prompt (same template as B configs)
        if sample.is_multiple_choice:
            letters = "ABCDEFGHIJ"
            choices_str = "\n".join(
                f"{letters[i]}) {c}" for i, c in enumerate(sample.choices)
            )
            prompt = MC_PROMPT_TEMPLATE.format(
                context=context_str,
                tree=tree_str,
                question=sample.query,
                choices_str=choices_str,
            )
        else:
            prompt = OPEN_PROMPT_TEMPLATE.format(
                context=context_str,
                tree=tree_str,
                question=sample.query,
            )

        # Generate
        response = self.generator.inference(prompt, system_prompt=SYSTEM_PROMPT)
        answer = response["choices"][0]["message"]["content"]
        latency = time.time() - start

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = estimate_cost(self.generator.engine, prompt_tokens, completion_tokens)

        choice_idx = None
        if sample.is_multiple_choice:
            choice_idx = parse_choice_letter(answer, len(sample.choices))

        return PipelineResult(
            answer=answer,
            selected_choice_index=choice_idx,
            retrieved_contexts=contexts,
            logic_tree_str=f"[FAKE TREE from random sample]\n{tree_str}",
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
