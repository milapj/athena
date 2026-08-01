import os
from typing import List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from benchmark.results import BenchmarkResult


# Consistent styling
COLORS = {
    "A1": "#4C72B0", "A2": "#55A868", "A3": "#C44E52",
    "B1": "#8172B2", "B2": "#CCB974", "B3": "#64B5CD",
}
RAG_ONLY_CONFIGS = ["A1", "A2", "A3"]
RAG_TREE_CONFIGS = ["B1", "B2", "B3"]
CONFIG_LABELS = {
    "A1": "RAG\nLlama-8B", "A2": "RAG\nLlama-70B", "A3": "RAG\nGPT-5.4",
    "B1": "RAG+Tree\nLlama-8B", "B2": "RAG+Tree\nLlama-70B", "B3": "RAG+Tree\nGPT-5.4",
}


def _to_df(results: List[BenchmarkResult]) -> pd.DataFrame:
    from dataclasses import asdict
    return pd.DataFrame([asdict(r) for r in results])


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy_by_config(results: List[BenchmarkResult], output_dir: str):
    """Bar chart: accuracy per config, grouped by dataset."""
    df = _to_df(results)
    fig, ax = plt.subplots(figsize=(12, 6))

    grouped = df.groupby(["config_name", "dataset_name"])["accurate"].mean().unstack(fill_value=0)
    configs = [c for c in list(COLORS.keys()) if c in grouped.index]
    grouped = grouped.loc[configs]

    x = np.arange(len(configs))
    width = 0.35
    datasets = grouped.columns.tolist()

    for i, dataset in enumerate(datasets):
        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            grouped[dataset].values,
            width,
            label=dataset,
            color=[COLORS.get(c, "#999999") for c in configs],
            alpha=0.7 + 0.3 * i,
            edgecolor="white",
        )
        for bar, val in zip(bars, grouped[dataset].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Pipeline Configuration", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(title="Dataset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, os.path.join(output_dir, "accuracy_by_config.png"))


def plot_rag_vs_tree_comparison(results: List[BenchmarkResult], output_dir: str):
    """Side-by-side comparison: RAG-only vs RAG+Tree for each model tier."""
    df = _to_df(results)
    pairs = [("A1", "B1"), ("A2", "B2"), ("A3", "B3")]
    pair_labels = ["Llama-8B", "Llama-70B", "GPT-5.4"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, (a_cfg, b_cfg), label in zip(axes, pairs, pair_labels):
        a_acc = df[df["config_name"] == a_cfg]["accurate"].mean()
        b_acc = df[df["config_name"] == b_cfg]["accurate"].mean()

        bars = ax.bar(
            ["RAG Only", "RAG + Tree"],
            [a_acc, b_acc],
            color=[COLORS[a_cfg], COLORS[b_cfg]],
            edgecolor="white",
            width=0.6,
        )
        for bar, val in zip(bars, [a_acc, b_acc]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        delta = b_acc - a_acc
        color = "green" if delta > 0 else "red"
        ax.set_title(f"{label}\n({delta:+.1%} delta)", fontsize=12, fontweight="bold", color=color)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=12)
    fig.suptitle("RAG vs RAG + Logic Tree: Accuracy by Model Tier", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "rag_vs_tree_comparison.png"))


def plot_cost_vs_accuracy(results: List[BenchmarkResult], output_dir: str):
    """Scatter plot: cost per query vs accuracy for each config."""
    df = _to_df(results)
    fig, ax = plt.subplots(figsize=(10, 6))

    for config_name in COLORS:
        subset = df[df["config_name"] == config_name]
        if subset.empty:
            continue
        total_cost = subset["cost_usd"].sum() + subset["tree_build_cost_usd"].sum()
        avg_cost = total_cost / len(subset) if len(subset) > 0 else 0
        avg_acc = subset["accurate"].mean()

        ax.scatter(
            avg_cost, avg_acc,
            s=200, c=COLORS[config_name],
            edgecolors="black", linewidth=1, zorder=5,
        )
        ax.annotate(
            config_name, (avg_cost, avg_acc),
            textcoords="offset points", xytext=(10, 5),
            fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("Avg Cost per Query (USD)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cost-Accuracy Tradeoff", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    _save(fig, os.path.join(output_dir, "cost_vs_accuracy.png"))


def plot_latency_distribution(results: List[BenchmarkResult], output_dir: str):
    """Box plot: latency distribution per config."""
    df = _to_df(results)
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = [c for c in COLORS if c in df["config_name"].values]
    data = [df[df["config_name"] == c]["latency_seconds"].values for c in configs]

    bp = ax.boxplot(
        data, labels=[CONFIG_LABELS.get(c, c) for c in configs],
        patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )
    for patch, config in zip(bp["boxes"], configs):
        patch.set_facecolor(COLORS.get(config, "#999999"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (seconds)", fontsize=12)
    ax.set_title("Response Latency by Configuration", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, os.path.join(output_dir, "latency_distribution.png"))


def plot_metrics_heatmap(results: List[BenchmarkResult], output_dir: str):
    """Heatmap: all metrics across configs."""
    df = _to_df(results)
    metrics = ["accurate", "ragas_faithfulness", "ragas_context_precision", "ragas_answer_correctness"]
    metric_labels = ["Accuracy", "Faithfulness", "Context Precision", "Answer Correctness"]

    # Filter to metrics that have data
    available = [(m, l) for m, l in zip(metrics, metric_labels) if m in df.columns and df[m].notna().any()]
    if not available:
        return

    metrics, metric_labels = zip(*available)
    configs = [c for c in COLORS if c in df["config_name"].values]

    matrix = []
    for config in configs:
        row = []
        for metric in metrics:
            val = df[df["config_name"] == config][metric].mean()
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([CONFIG_LABELS.get(c, c) for c in configs], fontsize=9)

    for i in range(len(configs)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0), shrink=0.8)
    ax.set_title("Benchmark Metrics Heatmap", fontsize=14, fontweight="bold")

    _save(fig, os.path.join(output_dir, "metrics_heatmap.png"))


def plot_accuracy_by_split(results: List[BenchmarkResult], output_dir: str):
    """Grouped bar chart: accuracy per MuSR split (murder_mysteries, object_placements, team_allocation)."""
    df = _to_df(results)
    # Extract split from metadata if available, else from sample_id
    if "split" not in df.columns:
        df["split"] = df["sample_id"].apply(
            lambda x: x.split("-")[1] if x.startswith("musr-") and len(x.split("-")) > 1 else "unknown"
        )

    splits = sorted(df["split"].unique())
    if len(splits) <= 1:
        return

    configs = [c for c in COLORS if c in df["config_name"].values]
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(splits))
    width = 0.12
    n_configs = len(configs)

    for i, config in enumerate(configs):
        offset = (i - n_configs / 2 + 0.5) * width
        vals = [df[(df["config_name"] == config) & (df["split"] == s)]["accurate"].mean() for s in splits]
        ax.bar(x + offset, vals, width, label=config, color=COLORS[config], edgecolor="white")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by MuSR Task Split", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in splits], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(ncol=3, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, os.path.join(output_dir, "accuracy_by_split.png"))


def generate_all_charts(results: List[BenchmarkResult], output_dir: str):
    """Generate all visualization charts."""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    plot_accuracy_by_config(results, charts_dir)
    plot_rag_vs_tree_comparison(results, charts_dir)
    plot_cost_vs_accuracy(results, charts_dir)
    plot_latency_distribution(results, charts_dir)
    plot_metrics_heatmap(results, charts_dir)
    plot_accuracy_by_split(results, charts_dir)

    print(f"\nAll charts saved to {charts_dir}/")
