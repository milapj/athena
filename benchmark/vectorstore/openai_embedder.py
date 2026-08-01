"""
OpenAI embedder for querying the existing ragtruth-benchmark collection
which was indexed with text-embedding-3-large (3072-dim).
"""

import os
from typing import List, Tuple

from openai import OpenAI

from benchmark.config import SPARSE_MAX_CHARS


class OpenAIEmbedder:
    """Hybrid embedder: dense (OpenAI text-embedding-3-large) + sparse (SPLADE local)."""

    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._sparse_model = None

    def _get_sparse_model(self):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            self._sparse_model = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1"
            )
        return self._sparse_model

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        model = self._get_sparse_model()
        truncated = [t[:SPARSE_MAX_CHARS] for t in texts]
        embeddings = list(model.embed(truncated))
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in embeddings
        ]

    def embed_hybrid(self, texts: List[str]) -> List[Tuple[List[float], dict]]:
        dense = self.embed_dense(texts)
        sparse = self.embed_sparse(texts)
        return list(zip(dense, sparse))

    def embed_single_hybrid(self, text: str) -> Tuple[List[float], dict]:
        return self.embed_hybrid([text])[0]
