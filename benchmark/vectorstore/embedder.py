from typing import List, Tuple

from fastembed import TextEmbedding, SparseTextEmbedding

from benchmark.config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL, SPARSE_MAX_CHARS


class Embedder:
    """Hybrid embedder: dense (BGE) + sparse (SPLADE), both local via fastembed."""

    def __init__(
        self,
        dense_model: str = DENSE_EMBEDDING_MODEL,
        sparse_model: str = SPARSE_EMBEDDING_MODEL,
    ):
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self._dense_model = None
        self._sparse_model = None

    def _get_dense_model(self) -> TextEmbedding:
        if self._dense_model is None:
            self._dense_model = TextEmbedding(model_name=self.dense_model_name)
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    def embed_dense(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        model = self._get_dense_model()
        embeddings = list(model.embed(texts, batch_size=batch_size))
        return [e.tolist() for e in embeddings]

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        """Returns list of {indices: [...], values: [...]} dicts."""
        model = self._get_sparse_model()
        # Truncate for SPLADE's token limit
        truncated = [t[:SPARSE_MAX_CHARS] for t in texts]
        embeddings = list(model.embed(truncated))
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in embeddings
        ]

    def embed_hybrid(self, texts: List[str]) -> List[Tuple[List[float], dict]]:
        """Returns list of (dense_vector, sparse_dict) tuples."""
        dense = self.embed_dense(texts)
        sparse = self.embed_sparse(texts)
        return list(zip(dense, sparse))

    def embed_single_hybrid(self, text: str) -> Tuple[List[float], dict]:
        return self.embed_hybrid([text])[0]
