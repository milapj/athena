import ast
import json
from typing import List

from datasets import load_dataset

from benchmark.datasets.base import BenchmarkDataset, BenchmarkSample, Document
from benchmark.config import CHUNK_SIZE, CHUNK_OVERLAP


def _ensure_list(choices) -> List[str]:
    """Handle choices that may be a JSON string, list, or other format."""
    if isinstance(choices, list):
        return choices
    if isinstance(choices, str):
        # Try JSON first (double quotes)
        try:
            parsed = json.loads(choices)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # Try Python repr format (single quotes like "['A', 'B']")
        try:
            parsed = ast.literal_eval(choices)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        # Try splitting by newline
        if "\n" in choices:
            return [c.strip() for c in choices.split("\n") if c.strip()]
    return []


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple word-level chunking with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if end >= len(words):
            break
    return chunks


class MuSRDataset(BenchmarkDataset):
    """Loads TAUR-Lab/MuSR from HuggingFace (756 samples, 3 splits)."""

    name = "musr"

    SPLITS = ["murder_mysteries", "object_placements", "team_allocation"]

    def __init__(self, splits: List[str] = None):
        self.splits = splits or self.SPLITS
        self._dataset = None

    def _load_hf(self):
        if self._dataset is None:
            self._dataset = load_dataset("TAUR-Lab/MuSR")
        return self._dataset

    def load(self) -> List[BenchmarkSample]:
        ds = self._load_hf()
        samples = []
        for split in self.splits:
            if split not in ds:
                continue
            for idx, row in enumerate(ds[split]):
                choices = _ensure_list(row.get("choices", []))
                answer_idx = row.get("answer_index", 0)
                samples.append(
                    BenchmarkSample(
                        id=f"musr-{split}-{idx}",
                        query=row["question"],
                        context_documents=[row["narrative"]],
                        ground_truth_answer=choices[answer_idx] if choices else "",
                        ground_truth_index=answer_idx,
                        choices=choices,
                        metadata={
                            "split": split,
                            "dataset": "musr",
                        },
                    )
                )
        return samples

    def documents_for_indexing(self) -> List[Document]:
        ds = self._load_hf()
        documents = []
        for split in self.splits:
            if split not in ds:
                continue
            for idx, row in enumerate(ds[split]):
                narrative = row["narrative"]
                chunks = chunk_text(narrative)
                for chunk_idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            id=f"musr-{split}-{idx}-chunk-{chunk_idx}",
                            text=chunk,
                            metadata={
                                "split": split,
                                "sample_idx": idx,
                                "chunk_idx": chunk_idx,
                                "dataset": "musr",
                            },
                        )
                    )
        return documents
