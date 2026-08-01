"""
Retrieval cache — ensures A and B configs get identical retrieved chunks.

Caches the retrieved chunks for each (sample_id, dataset) pair on the first
retrieval, then serves the cached chunks for all subsequent configs.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple


class RetrievalCache:
    """File-backed cache for retrieved chunks per sample."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._memory: Dict[str, List[str]] = {}

    def _key(self, sample_id: str) -> str:
        return sample_id

    def _path(self, sample_id: str) -> str:
        safe = sample_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def get(self, sample_id: str) -> Optional[List[str]]:
        """Return cached chunks or None."""
        key = self._key(sample_id)

        # Check memory first
        if key in self._memory:
            return self._memory[key]

        # Check disk
        path = self._path(sample_id)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            self._memory[key] = chunks
            return chunks

        return None

    def put(self, sample_id: str, chunks: List[str]):
        """Cache retrieved chunks for a sample."""
        key = self._key(sample_id)
        self._memory[key] = chunks

        path = self._path(sample_id)
        with open(path, "w") as f:
            json.dump({"sample_id": sample_id, "chunks": chunks}, f)
