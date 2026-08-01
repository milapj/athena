import os
from dataclasses import dataclass, field
from typing import Optional, List

from dotenv import load_dotenv

load_dotenv()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QDRANT_COLLECTION = "athena-benchmark"
QDRANT_RAGTRUTH_COLLECTION = "ragtruth-benchmark"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIM = 3072
DENSE_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # local via fastembed
DENSE_EMBEDDING_DIM = 1024
SPARSE_EMBEDDING_MODEL = "prithivida/Splade_PP_en_v1"  # local via fastembed
SPARSE_MAX_CHARS = 500  # SPLADE input truncation
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
TREE_BUILDER_MODEL = "google/gemini-2.0-flash-001"
RETRIEVAL_TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def get_env(key: str, required: bool = True) -> Optional[str]:
    val = os.getenv(key)
    if required and not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val


@dataclass
class PipelineConfig:
    name: str
    pipeline_type: str  # "rag_only" or "rag_plus_tree"
    generator_model: str  # OpenRouter model slug
    tree_builder_model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1024
    top_k: int = RETRIEVAL_TOP_K

    @property
    def uses_tree(self) -> bool:
        return self.pipeline_type in ("rag_plus_tree", "tree_only")


BENCHMARK_CONFIGS: List[PipelineConfig] = [
    # A configs: RAG only
    PipelineConfig(
        name="A1",
        pipeline_type="rag_only",
        generator_model="meta-llama/llama-3.1-8b-instruct",
    ),
    PipelineConfig(
        name="A2",
        pipeline_type="rag_only",
        generator_model="meta-llama/llama-3.1-70b-instruct",
    ),
    PipelineConfig(
        name="A3",
        pipeline_type="rag_only",
        generator_model="openai/gpt-5.4",
    ),
    # B configs: RAG + Logic Tree
    PipelineConfig(
        name="B1",
        pipeline_type="rag_plus_tree",
        generator_model="meta-llama/llama-3.1-8b-instruct",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    PipelineConfig(
        name="B2",
        pipeline_type="rag_plus_tree",
        generator_model="meta-llama/llama-3.1-70b-instruct",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    PipelineConfig(
        name="B3",
        pipeline_type="rag_plus_tree",
        generator_model="openai/gpt-5.4",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    # E configs: Ablation — chunks + FAKE tree (random tree from different sample)
    PipelineConfig(
        name="E1",
        pipeline_type="fake_tree",
        generator_model="meta-llama/llama-3.1-8b-instruct",
    ),
    # A4/B4/C4: Mid-tier model (Claude Sonnet 4.6)
    PipelineConfig(
        name="A4",
        pipeline_type="rag_only",
        generator_model="anthropic/claude-sonnet-4.6",
    ),
    PipelineConfig(
        name="B4",
        pipeline_type="rag_plus_tree",
        generator_model="anthropic/claude-sonnet-4.6",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    PipelineConfig(
        name="C4",
        pipeline_type="tree_only",
        generator_model="anthropic/claude-sonnet-4.6",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    # C configs: Tree only (no raw chunks to generator)
    PipelineConfig(
        name="C1",
        pipeline_type="tree_only",
        generator_model="meta-llama/llama-3.1-8b-instruct",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    PipelineConfig(
        name="C2",
        pipeline_type="tree_only",
        generator_model="meta-llama/llama-3.1-70b-instruct",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
    PipelineConfig(
        name="C3",
        pipeline_type="tree_only",
        generator_model="openai/gpt-5.4",
        tree_builder_model=TREE_BUILDER_MODEL,
    ),
]


def get_configs_by_names(names: List[str]) -> List[PipelineConfig]:
    config_map = {c.name: c for c in BENCHMARK_CONFIGS}
    result = []
    for name in names:
        if name not in config_map:
            raise ValueError(f"Unknown config: {name}. Available: {list(config_map.keys())}")
        result.append(config_map[name])
    return result
