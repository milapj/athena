"""
Tree-only pipeline — generator sees ONLY the logic tree, no raw chunks.

Uses the same retrieval + tree building as RAGPlusTreePipeline,
but strips out the raw chunks from the generator prompt.
"""

import time
from typing import Optional

from benchmark.datasets.base import BenchmarkSample
from benchmark.evaluation.cost_tracker import estimate_cost
from benchmark.evaluation.metrics import parse_choice_letter
from benchmark.models.openrouter import OpenRouterModel
from benchmark.pipelines.base import Pipeline, PipelineResult
from benchmark.pipelines.retrieval_cache import RetrievalCache
from benchmark.pipelines.tree_cache import TreeCache
from benchmark.vectorstore.qdrant_store import QdrantVectorStore

from dataset_builder import DatasetBuilder
from logic_tree.tree import LogicTree
from validators.types.structure_validator import StructureValidator


TREE_BUILDER_INTRO = """We are constructing a reasoning guide from a collection of retrieved evidence passages. To do this, we use an entailment tree — a tree structure where each intermediate node is entailed by its children. Together, they form a natural language reasoning proof.

To fill out this tree we need to complete an entailment. Completing an entailment means filling in one subtree of the entailment tree by providing supporting facts.

Facts From Story are facts explicitly stated in the retrieved evidence passages.
Commonsense Knowledge are facts that most people would agree are true and don't need to be explicitly stated.

All facts for the step must combine to entail the root parent fact.
No facts may contradict the current tree.
Always match the exact structure of the entailment step given. Give the same number of Facts From Story and Commonsense Knowledge facts, in the same order."""


SYSTEM_PROMPT = """You are a precise reasoning assistant. Answer questions based strictly on the provided reasoning tree. The reasoning tree contains step-by-step logical deductions extracted from evidence. Use it to guide your answer. Do not fabricate information beyond what the reasoning tree contains."""

MC_PROMPT_TEMPLATE = """Given the following reasoning tree, answer the question by selecting the best choice.

Reasoning Tree:
{tree}

Question: {question}

Choices:
{choices_str}

Respond with ONLY the letter of the correct answer (e.g., A, B, C, etc.)."""

OPEN_PROMPT_TEMPLATE = """Given the following reasoning tree, answer the question accurately and concisely.

Reasoning Tree:
{tree}

Question: {question}

Answer:"""


class TreeOnlyPipeline(Pipeline):
    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        generator: OpenRouterModel,
        tree_builder: OpenRouterModel,
        tree_cache: TreeCache,
        top_k: int = 5,
        tree_depth: int = 3,
        retrieval_cache: RetrievalCache = None,
    ):
        self.vectorstore = vectorstore
        self.generator = generator
        self.tree_builder = tree_builder
        self.tree_cache = tree_cache
        self.top_k = top_k
        self.tree_depth = tree_depth
        self.retrieval_cache = retrieval_cache
        self.builder = DatasetBuilder()

    def _build_logic_tree(self, sample_id: str, context_str: str, contexts: list) -> tuple:
        """Build or retrieve cached logic tree. Returns (tree_str, build_cost)."""
        cached = self.tree_cache.get(sample_id, contexts)
        if cached is not None:
            tree = LogicTree.from_json(cached["tree_json"])
            return cached["tree_str"], 0.0

        # Build skeleton
        tree = self.builder.build_structure(
            depth=self.tree_depth,
            bf_factor={1: 2.0, 2: 1.0},
            chance_to_prune=0.3,
            chance_to_prune_all=0.3,
        )

        completion_prompt_fn = self.builder.create_completion_prompt(
            example_trees=[],
            example_node_completions=[],
            example_descriptions=[],
            intro=TREE_BUILDER_INTRO,
        )

        cost_before = getattr(self.tree_builder, "total_cost", 0.0)

        completed_tree = self.builder.complete_structure(
            _tree=tree,
            model=self.tree_builder,
            description=context_str,
            completion_prompt_fn=completion_prompt_fn,
            use_iterative_complete_v2=True,
            validators=[StructureValidator()],
            max_retries_on_error=2,
        )

        cost_after = getattr(self.tree_builder, "total_cost", 0.0)
        build_cost = cost_after - cost_before

        tree_str = completed_tree.print_for_gpt(pad_space=1, pad_char="> ")

        self.tree_cache.put(sample_id, contexts, {
            "tree_json": completed_tree.to_json(),
            "tree_str": tree_str,
        })

        return tree_str, build_cost

    def run(self, sample: BenchmarkSample) -> PipelineResult:
        start = time.time()

        # Retrieve chunks (needed for tree building, but NOT passed to generator)
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

        # Build logic tree from chunks
        tree_str, tree_cost = self._build_logic_tree(sample.id, context_str, contexts)

        # Format prompt with ONLY the tree (no raw chunks)
        if sample.is_multiple_choice:
            letters = "ABCDEFGHIJ"
            choices_str = "\n".join(
                f"{letters[i]}) {c}" for i, c in enumerate(sample.choices)
            )
            prompt = MC_PROMPT_TEMPLATE.format(
                tree=tree_str,
                question=sample.query,
                choices_str=choices_str,
            )
        else:
            prompt = OPEN_PROMPT_TEMPLATE.format(
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
            retrieved_contexts=contexts,  # Store for evaluation, even though not passed to generator
            logic_tree_str=tree_str,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            tree_build_cost_usd=tree_cost,
        )
