from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from benchmark.datasets.base import BenchmarkSample


@dataclass
class PipelineResult:
    answer: str
    selected_choice_index: Optional[int] = None
    retrieved_contexts: List[str] = field(default_factory=list)
    logic_tree_str: Optional[str] = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tree_build_cost_usd: float = 0.0


class Pipeline(ABC):
    @abstractmethod
    def run(self, sample: BenchmarkSample) -> PipelineResult:
        ...
