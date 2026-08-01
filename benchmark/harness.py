import os
import traceback
from typing import List, Optional

from tqdm import tqdm

from benchmark.config import PipelineConfig
from benchmark.datasets.base import BenchmarkDataset, BenchmarkSample
from benchmark.evaluation.metrics import accuracy, string_accuracy
from benchmark.models.openrouter import OpenRouterModel
from benchmark.pipelines.base import Pipeline, PipelineResult
from benchmark.pipelines.rag_only import RAGOnlyPipeline
from benchmark.pipelines.rag_plus_tree import RAGPlusTreePipeline
from benchmark.pipelines.retrieval_cache import RetrievalCache
from benchmark.pipelines.tree_cache import TreeCache
from benchmark.pipelines.fake_tree import FakeTreePipeline
from benchmark.pipelines.tree_only import TreeOnlyPipeline
from benchmark.results import (
    BenchmarkResult,
    append_result_jsonl,
    get_completed_keys,
)
from benchmark.vectorstore.qdrant_store import QdrantVectorStore


class BenchmarkHarness:
    def __init__(
        self,
        configs: List[PipelineConfig],
        datasets: List[BenchmarkDataset],
        vectorstore: Optional[QdrantVectorStore] = None,
        vectorstores: Optional[dict] = None,
        output_dir: str = "results",
    ):
        self.configs = configs
        self.datasets = datasets
        self.vectorstore = vectorstore
        self.vectorstores = vectorstores or {}
        self.output_dir = output_dir
        self.results_path = os.path.join(output_dir, "raw_results.jsonl")

    def _get_vectorstore(self, dataset_name: str) -> Optional[QdrantVectorStore]:
        """Get the right vectorstore for a dataset."""
        if dataset_name in self.vectorstores:
            return self.vectorstores[dataset_name]
        return self.vectorstore

    def _get_retrieval_cache(self, dataset_name: str) -> RetrievalCache:
        """Get or create retrieval cache for a dataset (shared across all configs)."""
        if not hasattr(self, "_retrieval_caches"):
            self._retrieval_caches = {}
        if dataset_name not in self._retrieval_caches:
            cache_dir = os.path.join(self.output_dir, "retrieval_cache", dataset_name)
            self._retrieval_caches[dataset_name] = RetrievalCache(cache_dir)
        return self._retrieval_caches[dataset_name]

    def _build_pipeline(self, config: PipelineConfig, vectorstore: Optional[QdrantVectorStore] = None,
                        retrieval_cache: RetrievalCache = None) -> Pipeline:
        vs = vectorstore or self.vectorstore
        generator = OpenRouterModel(
            engine=config.generator_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        if config.pipeline_type == "fake_tree":
            return FakeTreePipeline(
                vectorstore=vs,
                generator=generator,
                tree_cache_dir=os.path.join(self.output_dir, "tree_cache"),
                top_k=config.top_k,
                retrieval_cache=retrieval_cache,
            )
        if config.pipeline_type in ("rag_plus_tree", "tree_only"):
            tree_builder = OpenRouterModel(
                engine=config.tree_builder_model,
                temperature=0.7,
                max_tokens=4096,
            )
            tree_cache = TreeCache(
                cache_dir=os.path.join(self.output_dir, "tree_cache")
            )
            pipeline_cls = TreeOnlyPipeline if config.pipeline_type == "tree_only" else RAGPlusTreePipeline
            return pipeline_cls(
                vectorstore=vs,
                generator=generator,
                tree_builder=tree_builder,
                tree_cache=tree_cache,
                top_k=config.top_k,
                retrieval_cache=retrieval_cache,
            )
        return RAGOnlyPipeline(
            vectorstore=vs,
            generator=generator,
            top_k=config.top_k,
            retrieval_cache=retrieval_cache,
        )

    def _compute_metrics(
        self, sample: BenchmarkSample, result: PipelineResult
    ) -> dict:
        acc = None
        if sample.is_multiple_choice:
            acc = accuracy(result.selected_choice_index, sample.ground_truth_index)
        else:
            acc = string_accuracy(result.answer, sample.ground_truth_answer)
        return {"accurate": acc}

    def run(self, max_samples: Optional[int] = None):
        os.makedirs(self.output_dir, exist_ok=True)
        completed = get_completed_keys(self.results_path)
        print(f"Resuming: {len(completed)} results already completed")

        for dataset in self.datasets:
            samples = dataset.load()
            if max_samples:
                samples = samples[:max_samples]

            vs = self._get_vectorstore(dataset.name)

            # Create retrieval cache per dataset (shared across all configs)
            retrieval_cache = self._get_retrieval_cache(dataset.name)

            for config in self.configs:
                pipeline = self._build_pipeline(config, vectorstore=vs, retrieval_cache=retrieval_cache)
                desc = f"{config.name}/{dataset.name}"
                skipped = 0

                for sample in tqdm(samples, desc=desc):
                    key = (config.name, dataset.name, sample.id)
                    if key in completed:
                        skipped += 1
                        continue

                    try:
                        result = pipeline.run(sample)
                        metrics = self._compute_metrics(sample, result)

                        br = BenchmarkResult(
                            config_name=config.name,
                            dataset_name=dataset.name,
                            sample_id=sample.id,
                            query=sample.query,
                            ground_truth=sample.ground_truth_answer,
                            predicted_answer=result.answer,
                            selected_choice_index=result.selected_choice_index,
                            ground_truth_index=sample.ground_truth_index,
                            accurate=metrics.get("accurate"),
                            latency_seconds=result.latency_seconds,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                            cost_usd=result.cost_usd,
                            tree_build_cost_usd=result.tree_build_cost_usd,
                            retrieved_contexts=result.retrieved_contexts,
                            logic_tree_str=result.logic_tree_str,
                        )
                        append_result_jsonl(br, self.results_path)
                        completed.add(key)
                    except Exception as e:
                        print(f"Error on {key}: {e}")
                        traceback.print_exc()
                        continue

                if skipped:
                    print(f"  Skipped {skipped} already-completed samples")
