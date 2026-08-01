import os
import sys

import click

# Add project root to path so we can import athena modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.config import get_configs_by_names, BENCHMARK_CONFIGS, QDRANT_RAGTRUTH_COLLECTION
from benchmark.datasets.musr import MuSRDataset
from benchmark.datasets.multihoprag import MultiHopRAGDataset
from benchmark.datasets.ragtruth import RAGTruthDataset
from benchmark.vectorstore.embedder import Embedder
from benchmark.vectorstore.qdrant_store import QdrantVectorStore


def _get_vectorstore(dataset_name: str = None) -> QdrantVectorStore:
    """Get vectorstore configured for the target dataset."""
    if dataset_name == "ragtruth":
        # Use OpenAI embeddings + ragtruth-benchmark collection (already indexed)
        from benchmark.vectorstore.openai_embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        return QdrantVectorStore(embedder=embedder, collection_name=QDRANT_RAGTRUTH_COLLECTION)
    if dataset_name == "multihoprag":
        # Use local BGE+SPLADE + dedicated collection
        embedder = Embedder()
        return QdrantVectorStore(embedder=embedder, collection_name="multihoprag-benchmark")
    # Default: local BGE+SPLADE + athena-benchmark collection
    embedder = Embedder()
    return QdrantVectorStore(embedder=embedder)


def _get_datasets(names: list, ragtruth_dir: str = None) -> list:
    datasets = []
    for name in names:
        if name == "musr":
            datasets.append(MuSRDataset())
        elif name == "ragtruth":
            # Use local dir if provided, otherwise auto-download from HuggingFace
            datasets.append(RAGTruthDataset(data_dir=ragtruth_dir))
        elif name == "multihoprag":
            datasets.append(MultiHopRAGDataset())
        else:
            raise click.BadParameter(f"Unknown dataset: {name}")
    return datasets


@click.group()
def cli():
    """Athena Benchmark Harness — RAG vs RAG+LogicTree evaluation."""
    pass


@cli.command()
@click.option("--dataset", "-d", multiple=True, required=True, help="Dataset(s) to index: musr, ragtruth, multihoprag")
@click.option("--ragtruth-dir", default=None, help="Path to RAGTruth data directory")
def index(dataset, ragtruth_dir):
    """Index dataset documents into Qdrant."""
    datasets = _get_datasets(list(dataset), ragtruth_dir)

    for ds in datasets:
        vs = _get_vectorstore(ds.name)
        docs = ds.documents_for_indexing()
        click.echo(f"Indexing {len(docs)} documents from {ds.name} into collection...")
        vs.index(docs)
        click.echo(f"Done. Collection has {vs.collection_count()} points.")


@cli.command()
@click.option("--configs", "-c", required=True, help="Comma-separated config names: A1,A2,B1,...")
@click.option("--datasets", "-d", required=True, help="Comma-separated dataset names: musr,ragtruth")
@click.option("--max-samples", "-n", type=int, default=None, help="Max samples per dataset (for dev)")
@click.option("--output-dir", "-o", required=True, help="Output directory for results")
@click.option("--ragtruth-dir", default=None, help="Path to RAGTruth data directory")
def run(configs, datasets, max_samples, output_dir, ragtruth_dir):
    """Run benchmark across specified configs and datasets."""
    from benchmark.harness import BenchmarkHarness

    config_names = [c.strip() for c in configs.split(",")]
    dataset_names = [d.strip() for d in datasets.split(",")]

    pipeline_configs = get_configs_by_names(config_names)
    benchmark_datasets = _get_datasets(dataset_names, ragtruth_dir)

    # Build vectorstore per dataset that needs retrieval
    # MuSR uses direct context (no vectorstore), RAGTruth uses Qdrant retrieval
    # MuSR uses direct context (no vectorstore), others use Qdrant retrieval
    direct_context_datasets = {"musr"}
    retrieval_datasets = [d.name for d in benchmark_datasets if d.name not in direct_context_datasets]
    vectorstores = {}
    for ds_name in retrieval_datasets:
        vectorstores[ds_name] = _get_vectorstore(ds_name)

    click.echo(f"Configs: {config_names}")
    click.echo(f"Datasets: {dataset_names}")
    if retrieval_datasets:
        click.echo(f"Retrieval datasets: {retrieval_datasets}")
    if max_samples:
        click.echo(f"Max samples: {max_samples}")

    harness = BenchmarkHarness(
        configs=pipeline_configs,
        datasets=benchmark_datasets,
        vectorstores=vectorstores,
        output_dir=output_dir,
    )
    harness.run(max_samples=max_samples)
    click.echo(f"Results saved to {output_dir}/raw_results.jsonl")


@cli.command()
@click.option("--results-dir", "-r", required=True, help="Directory containing raw_results.jsonl")
@click.option("--output-dir", "-o", required=True, help="Output directory for analysis")
def evaluate(results_dir, output_dir):
    """Run RAGAS evaluation and generate summary statistics on existing results."""
    from benchmark.results import load_results_jsonl, export_summary_csv
    from benchmark.evaluation.ragas_eval import RagasEvaluator

    results_path = os.path.join(results_dir, "raw_results.jsonl")
    results = load_results_jsonl(results_path)
    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Loaded {len(results)} results")

    # Export summary CSV
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.csv")
    export_summary_csv(results, summary_path)
    click.echo(f"Summary exported to {summary_path}")

    # Run RAGAS evaluation
    click.echo("Running RAGAS evaluation...")
    evaluator = RagasEvaluator()
    ragas_input = [
        {
            "query": r.query,
            "answer": r.predicted_answer,
            "contexts": r.retrieved_contexts,
            "ground_truth": r.ground_truth,
        }
        for r in results
    ]
    ragas_df = evaluator.evaluate_batch(ragas_input)
    ragas_path = os.path.join(output_dir, "ragas.csv")
    ragas_df.to_csv(ragas_path, index=False)
    click.echo(f"RAGAS results exported to {ragas_path}")

    # Generate charts
    click.echo("Generating charts...")
    from benchmark.visualization import generate_all_charts
    generate_all_charts(results, output_dir)


@cli.command()
@click.option("--results-dir", "-r", required=True, help="Directory containing raw_results.jsonl")
@click.option("--output-dir", "-o", required=True, help="Output directory for charts")
def charts(results_dir, output_dir):
    """Generate visualization charts from existing results (no RAGAS re-run)."""
    from benchmark.results import load_results_jsonl
    from benchmark.visualization import generate_all_charts

    results_path = os.path.join(results_dir, "raw_results.jsonl")
    results = load_results_jsonl(results_path)
    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Loaded {len(results)} results")
    generate_all_charts(results, output_dir)


if __name__ == "__main__":
    cli()
