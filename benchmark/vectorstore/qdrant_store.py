import os
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from benchmark.config import (
    QDRANT_COLLECTION,
    DENSE_EMBEDDING_DIM,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
)
from benchmark.datasets.base import Document
from benchmark.vectorstore.embedder import Embedder


@dataclass
class RetrievedDocument:
    text: str
    score: float
    metadata: Dict[str, Any]


class QdrantVectorStore:
    def __init__(
        self,
        embedder: Embedder,
        url: str = None,
        api_key: str = None,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=url or os.getenv("QDRANT_URL"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
            timeout=300,
        )

    def ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=DENSE_EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                    ),
                },
            )
            print(f"Created hybrid collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def collection_count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def index(self, documents: List[Document], batch_size: int = 64):
        self.ensure_collection()
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.text for doc in batch]
            hybrid_embeddings = self.embedder.embed_hybrid(texts)

            points = []
            for doc, (dense_vec, sparse_dict) in zip(batch, hybrid_embeddings):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id)),
                        vector={
                            "dense": dense_vec,
                            "sparse": SparseVector(
                                indices=sparse_dict["indices"],
                                values=sparse_dict["values"],
                            ),
                        },
                        payload={"text": doc.text, "doc_id": doc.id, **doc.metadata},
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Indexed {min(i + batch_size, total)}/{total} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
    ) -> List[RetrievedDocument]:
        dense_vec, sparse_dict = self.embedder.embed_single_hybrid(query)

        # Query dense
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points

        # Query sparse
        sparse_query = SparseVector(
            indices=sparse_dict["indices"],
            values=sparse_dict["values"],
        )
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_query,
            using="sparse",
            limit=top_k,
            with_payload=True,
        ).points

        # Fuse with score normalization
        return self._fuse_results(dense_results, sparse_results, dense_weight, sparse_weight, top_k)

    def _fuse_results(
        self,
        dense_results: list,
        sparse_results: list,
        dense_weight: float,
        sparse_weight: float,
        k: int,
    ) -> List[RetrievedDocument]:
        """Min-max normalization fusion matching production legal indexer."""
        total_w = dense_weight + sparse_weight
        dense_w = dense_weight / total_w if total_w > 0 else 0.7
        sparse_w = sparse_weight / total_w if total_w > 0 else 0.3

        all_points = {}
        scores = {}

        # Normalize dense scores to [0, 1]
        if dense_results:
            d_max = max(p.score for p in dense_results)
            d_min = min(p.score for p in dense_results)
            d_range = d_max - d_min if d_max != d_min else 1.0

            for p in dense_results:
                all_points[p.id] = p
                normalized = (p.score - d_min) / d_range
                scores[p.id] = normalized * dense_w

        # Normalize sparse scores to [0, 1]
        if sparse_results:
            s_max = max(p.score for p in sparse_results)
            s_min = min(p.score for p in sparse_results)
            s_range = s_max - s_min if s_max != s_min else 1.0

            for p in sparse_results:
                all_points[p.id] = p
                normalized = (p.score - s_min) / s_range
                scores[p.id] = scores.get(p.id, 0.0) + normalized * sparse_w

        # Sort by fused score, take top k
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            RetrievedDocument(
                text=all_points[pid].payload.get("text", "") or all_points[pid].payload.get("page_content", ""),
                score=score,
                metadata={
                    k: v for k, v in all_points[pid].payload.items()
                    if k not in ("text", "page_content")
                },
            )
            for pid, score in sorted_ids
        ]
