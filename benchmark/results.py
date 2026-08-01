import csv
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd


@dataclass
class BenchmarkResult:
    config_name: str
    dataset_name: str
    sample_id: str
    query: str
    ground_truth: str
    predicted_answer: str
    selected_choice_index: Optional[int] = None
    ground_truth_index: Optional[int] = None
    # Metrics
    accurate: Optional[float] = None
    span_f1: Optional[float] = None
    hallucinations_per_100_tokens: Optional[float] = None
    # RAGAS (filled post-hoc)
    ragas_faithfulness: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    ragas_answer_correctness: Optional[float] = None
    # Cost & performance
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tree_build_cost_usd: float = 0.0
    # Context
    retrieved_contexts: List[str] = field(default_factory=list)
    logic_tree_str: Optional[str] = None


def append_result_jsonl(result: BenchmarkResult, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def load_results_jsonl(path: str) -> List[BenchmarkResult]:
    results = []
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                results.append(BenchmarkResult(**d))
    return results


def get_completed_keys(path: str) -> set:
    results = load_results_jsonl(path)
    return {(r.config_name, r.dataset_name, r.sample_id) for r in results}


def export_summary_csv(results: List[BenchmarkResult], path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])

    metric_cols = [
        "accurate", "span_f1", "hallucinations_per_100_tokens",
        "ragas_faithfulness", "ragas_context_precision", "ragas_answer_correctness",
        "latency_seconds", "cost_usd", "tree_build_cost_usd",
    ]
    existing_cols = [c for c in metric_cols if c in df.columns]

    summary = df.groupby(["config_name", "dataset_name"])[existing_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.to_csv(path)
    return summary
