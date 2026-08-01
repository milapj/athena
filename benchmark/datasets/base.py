from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    id: str
    query: str
    context_documents: List[str]
    ground_truth_answer: str
    ground_truth_index: Optional[int] = None
    choices: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return self.choices is not None and len(self.choices) > 0


class BenchmarkDataset(ABC):
    name: str

    @abstractmethod
    def load(self) -> List[BenchmarkSample]:
        """Load benchmark samples."""
        ...

    @abstractmethod
    def documents_for_indexing(self) -> List[Document]:
        """Return documents to index into the vector store."""
        ...
