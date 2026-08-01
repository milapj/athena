"""
MultiHop-RAG dataset loader for Track 3 benchmarking.

Loads multi-hop QA from HuggingFace (yixuantt/MultiHopRAG).
Questions require reasoning across 2-4 documents from a news corpus.
Uses Qdrant retrieval (context_documents is empty).
"""

from typing import List, Optional

from benchmark.datasets.base import BenchmarkDataset, BenchmarkSample, Document


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into word-level chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]


class MultiHopRAGDataset(BenchmarkDataset):
    """
    MultiHop-RAG benchmark for multi-hop retrieval-augmented generation.

    Two configs from HuggingFace (yixuantt/MultiHopRAG):
      1. "MultiHopRAG" — 2,556 multi-hop QA pairs with evidence
      2. "corpus" — 609 news articles (knowledge base)

    Questions require chaining facts from 2-4 documents.
    context_documents is EMPTY — forces Qdrant retrieval.
    Ground truth includes answer + supporting evidence for evaluation.
    """

    name = "multihoprag"

    def __init__(self, question_types: Optional[List[str]] = None):
        self.question_types = question_types  # None = all types
        self._qa_dataset = None
        self._corpus_dataset = None

    def _load_qa(self):
        if self._qa_dataset is not None:
            return self._qa_dataset
        from datasets import load_dataset
        ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
        self._qa_dataset = ds
        return ds

    def _load_corpus(self):
        if self._corpus_dataset is not None:
            return self._corpus_dataset
        from datasets import load_dataset
        ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
        self._corpus_dataset = ds
        return ds

    def load(self) -> List[BenchmarkSample]:
        ds = self._load_qa()
        samples = []

        for i, row in enumerate(ds):
            query = row.get("query", "")
            answer = row.get("answer", "")
            question_type = row.get("question_type", "")
            evidence_list = row.get("evidence_list", [])

            # Filter by question type if specified
            if self.question_types and question_type not in self.question_types:
                continue

            # Extract supporting evidence metadata for evaluation
            evidence_facts = []
            evidence_sources = []
            for ev in evidence_list:
                if isinstance(ev, dict):
                    evidence_facts.append(ev.get("fact", ""))
                    evidence_sources.append({
                        "source": ev.get("source", ""),
                        "title": ev.get("title", ""),
                    })

            samples.append(
                BenchmarkSample(
                    id=f"multihop-{i}",
                    query=query,
                    context_documents=[],  # EMPTY — forces Qdrant retrieval
                    ground_truth_answer=answer,
                    metadata={
                        "dataset": "multihoprag",
                        "question_type": question_type,
                        "evidence_facts": evidence_facts,
                        "evidence_sources": evidence_sources,
                        "num_hops": len(evidence_list),
                    },
                )
            )

        return samples

    def documents_for_indexing(self) -> List[Document]:
        """
        Index the 609 news articles from the corpus config.
        Articles are chunked (avg 1,746 words) to fit embedding models.
        """
        corpus = self._load_corpus()
        documents = []

        for i, row in enumerate(corpus):
            title = row.get("title", "")
            body = row.get("body", "")
            source = row.get("source", "")
            category = row.get("category", "")
            published_at = row.get("published_at", "")

            if not body:
                continue

            # Prepend title to body for better retrieval
            full_text = f"{title}\n\n{body}" if title else body

            # Chunk long articles
            chunks = _chunk_text(full_text, chunk_size=512, overlap=50)

            for chunk_idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        id=f"multihop-corpus-{i}-chunk-{chunk_idx}",
                        text=chunk,
                        metadata={
                            "dataset": "multihoprag",
                            "doc_idx": i,
                            "chunk_idx": chunk_idx,
                            "title": title,
                            "source": source,
                            "category": category,
                            "published_at": published_at,
                        },
                    )
                )

        return documents
