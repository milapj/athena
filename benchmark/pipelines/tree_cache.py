import hashlib
import json
import os
from typing import Optional


class TreeCache:
    """
    Disk-based cache for logic trees. Since the tree builder model is fixed
    and retrieval is identical across B configs, each tree is built once per
    (sample_id, retrieved_context_hash) and shared across B1/B2/B3.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, sample_id: str, contexts: list) -> str:
        ctx_hash = hashlib.md5(json.dumps(contexts, sort_keys=True).encode()).hexdigest()[:12]
        safe_id = sample_id.replace("/", "_")
        return f"{safe_id}_{ctx_hash}"

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, sample_id: str, contexts: list) -> Optional[dict]:
        key = self._cache_key(sample_id, contexts)
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def put(self, sample_id: str, contexts: list, tree_data: dict):
        key = self._cache_key(sample_id, contexts)
        path = self._cache_path(key)
        with open(path, "w") as f:
            json.dump(tree_data, f)
