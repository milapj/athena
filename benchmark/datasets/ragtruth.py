"""
RAGTruth dataset loader for Track 2 benchmarking.

Loads the QA subset from HuggingFace (wandb/RAGTruth-processed) or local JSONL files.
Unlike MuSR (direct context), RAGTruth uses actual retrieval from Qdrant.
"""

import json
import os
from typing import List, Optional

from benchmark.datasets.base import BenchmarkDataset, BenchmarkSample, Document


class RAGTruthDataset(BenchmarkDataset):
    """
    RAGTruth benchmark for hallucination detection in RAG.

    Two loading modes:
      1. HuggingFace (default): auto-downloads wandb/RAGTruth-processed
      2. Local JSONL: provide data_dir with response.jsonl + source_info.jsonl

    Focuses on QA task type for RAG benchmarking.
    context_documents is left EMPTY so the pipeline retrieves from Qdrant.
    """

    name = "ragtruth"

    def __init__(self, data_dir: Optional[str] = None, task_types: Optional[List[str]] = None,
                 split: str = "test"):
        self.data_dir = data_dir
        self.task_types = task_types or ["QA"]  # Default: QA only for RAG relevance
        self.split = split
        self._hf_dataset = None
        self._source_info = None
        self._responses = None

    # ── HuggingFace loading ──────────────────────────────────────────

    def _load_hf(self):
        """Load from HuggingFace datasets hub."""
        if self._hf_dataset is not None:
            return self._hf_dataset

        from datasets import load_dataset
        ds = load_dataset("wandb/RAGTruth-processed", split=self.split)
        self._hf_dataset = ds
        return ds

    def _load_from_hf(self) -> List[BenchmarkSample]:
        ds = self._load_hf()
        samples = []

        for i, row in enumerate(ds):
            task_type = row.get("task_type", "")
            if task_type not in self.task_types:
                continue

            # HuggingFace wandb/RAGTruth-processed column mapping:
            # query = question, context = source passages, output = model response
            query = row.get("query", "")
            context = row.get("context", "")
            output = row.get("output", "")
            quality = row.get("quality", "")  # "good" or "poor"

            # Hallucination labels
            hallucination_labels = row.get("hallucination_labels", [])
            hallucination_processed = row.get("hallucination_labels_processed", {})

            samples.append(
                BenchmarkSample(
                    id=f"ragtruth-{i}",
                    query=query,
                    context_documents=[],  # EMPTY — forces Qdrant retrieval
                    ground_truth_answer=output,
                    metadata={
                        "dataset": "ragtruth",
                        "task_type": task_type,
                        "source_context": context,  # Original context for reference
                        "model": row.get("model", ""),
                        "quality": quality,
                        "hallucination_labels": hallucination_labels,
                        "hallucination_processed": hallucination_processed,
                        "split": self.split,
                    },
                )
            )

        return samples

    def _docs_from_hf(self) -> List[Document]:
        ds = self._load_hf()
        documents = {}  # Dedupe by context text hash

        for i, row in enumerate(ds):
            task_type = row.get("task_type", "")
            if task_type not in self.task_types:
                continue

            context = row.get("context", "")
            if not context:
                continue

            # Dedupe by query (same query = same source context)
            query = row.get("query", "")
            if query in documents:
                continue

            documents[query] = Document(
                id=f"ragtruth-ctx-{i}",
                text=context,
                metadata={
                    "task_type": task_type,
                    "dataset": "ragtruth",
                    "query": query,
                },
            )

        return list(documents.values())

    # ── Local JSONL loading (fallback) ───────────────────────────────

    def _load_source_info(self) -> dict:
        if self._source_info is None:
            path = os.path.join(self.data_dir, "source_info.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"RAGTruth source_info.jsonl not found at {path}")
            self._source_info = {}
            with open(path) as f:
                for line in f:
                    row = json.loads(line.strip())
                    self._source_info[row["source_id"]] = row
        return self._source_info

    def _load_responses(self) -> list:
        if self._responses is None:
            path = os.path.join(self.data_dir, "response.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"RAGTruth response.jsonl not found at {path}")
            self._responses = []
            with open(path) as f:
                for line in f:
                    self._responses.append(json.loads(line.strip()))
        return self._responses

    def _load_from_local(self) -> List[BenchmarkSample]:
        source_info = self._load_source_info()
        responses = self._load_responses()
        samples = []

        for resp in responses:
            source_id = resp.get("source_id", "")
            src = source_info.get(source_id, {})
            task_type = src.get("task_type", "")

            if task_type not in self.task_types:
                continue

            prompt = src.get("prompt", "")
            source_text = src.get("source_info", "")
            if isinstance(source_text, dict):
                source_text = json.dumps(source_text)

            labels = resp.get("labels", {})
            hallucinated_spans = labels.get("hallucinated_spans", [])

            samples.append(
                BenchmarkSample(
                    id=f"ragtruth-{resp.get('id', '')}",
                    query=prompt,
                    context_documents=[],  # EMPTY — forces Qdrant retrieval
                    ground_truth_answer=resp.get("response", ""),
                    metadata={
                        "dataset": "ragtruth",
                        "task_type": task_type,
                        "source_id": source_id,
                        "source_text": source_text,
                        "model": resp.get("model", ""),
                        "hallucinated_spans": hallucinated_spans,
                        "split": resp.get("split", ""),
                    },
                )
            )
        return samples

    def _docs_from_local(self) -> List[Document]:
        source_info = self._load_source_info()
        documents = []
        for source_id, src in source_info.items():
            task_type = src.get("task_type", "")
            if task_type not in self.task_types:
                continue

            source_text = src.get("source_info", "")
            if isinstance(source_text, dict):
                source_text = json.dumps(source_text)
            if source_text:
                documents.append(
                    Document(
                        id=f"ragtruth-{source_id}",
                        text=source_text,
                        metadata={
                            "task_type": task_type,
                            "dataset": "ragtruth",
                        },
                    )
                )
        return documents

    # ── Public interface ─────────────────────────────────────────────

    def load(self) -> List[BenchmarkSample]:
        if self.data_dir:
            return self._load_from_local()
        return self._load_from_hf()

    def documents_for_indexing(self) -> List[Document]:
        if self.data_dir:
            return self._docs_from_local()
        return self._docs_from_hf()
