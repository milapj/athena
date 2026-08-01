# Athena: an external reasoning engine for retrieval-augmented generation

Evaluation harness and results for the dissertation *Optimizing Retrieval-Augmented
Generation (RAG) with an External Reasoning Engine for Enhanced Performance*
(Milap Jhumkhawala, University of the Cumberlands, 2026).

## What this is

A retrieval-augmented generation pipeline that interposes a Toulmin-structured
logic tree between retrieval and generation, evaluated against baselines across
three model-cost tiers on the MultiHop-RAG benchmark (Tang & Yang, 2024).

## Configurations

| Config | Pipeline | Generator | Tree builder |
|---|---|---|---|
| A1 | RAG only | Llama-3.1-8B-Instruct | — |
| A2 | RAG only | Claude 3 Haiku | — |
| A3 | RAG only | GPT-5.4 | — |
| A4 | RAG only | Claude Sonnet 4.6 | — |
| B1 | RAG + logic tree | Llama-3.1-8B-Instruct | Gemini 2.0 Flash-Lite |
| C1 | Tree only, no passages | Llama-3.1-8B-Instruct | Gemini 2.0 Flash-Lite |
| E1 | RAG + content-mismatched tree | Llama-3.1-8B-Instruct | — |

B2/B3/B4 and C2/C3/C4 repeat the tree and tree-only pipelines on the remaining
generators. All models were served through OpenRouter.

## Settings

Retrieval is hybrid into Qdrant: BAAI/bge-large-en-v1.5 dense (1024-d, weight
0.7) with SPLADE PP en v1 sparse (weight 0.3), both local via fastembed. Chunks
of 512 tokens with 50 overlap, top-k 5, no reranker, passages supplied in
retrieval order. Retrieved passages are cached and reused across every
configuration, so all conditions see identical context.

Generation uses temperature 0.3, top-p 1.0, and a 1024-token limit for every
configuration. No seed is set: OpenRouter does not expose one for these models,
which is why all reported figures come from a single execution. API calls retry
up to ten times with a ten-second wait; tree construction retries twice against
a structure validator. Execution is sequential.

## Reproducing the reported tables

    uv sync
    cp .env.example .env        # add OPENROUTER_API_KEY and QDRANT_URL
    python -m benchmark.cli --dataset multihoprag --configs A1,A2,A3,A4
    python -m benchmark.cli --dataset multihoprag --configs B1,B2,B3,B4,C1,C2,C3,C4,E1
    python results/dissertation/make_table2.py

Re-running will not reproduce the stored outputs exactly. Generation is
stochastic and unseeded, and the commercial endpoints are updated by their
providers. What is reproducible from this repository is the analysis:
`results/dissertation/per_query_scores.csv` regenerates every published figure.

## What is included

Everything reported in the dissertation is here. The harness, the logic-tree
builder, all pipeline implementations, the 1,788 cached logic trees, the
per-query scores, the aggregate results tables, and the raw model outputs.

Raw outputs are gzipped. The four baseline runs are one file each at
`results/mh-a{1,2,3,4}/raw_results.jsonl.gz`. The treatment and ablation run
covered nine configurations in a single pass and is split by configuration at
`results/mh-b/by_config/{B1,B2,B3,B4,C1,C2,C3,C4,E1}.jsonl.gz`. Each record
carries the query, the gold answer, the model response, the Exact Match score,
latency, token counts, cost, the retrieved passages, and for tree
configurations the serialized logic tree that was supplied to the generator.

    import gzip, json
    with gzip.open("results/mh-b/by_config/B1.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]

Excluded: the retrieval cache, which duplicates passages from the public
MultiHop-RAG corpus and is regenerable, and pilot runs on RAGTruth and MuSR
that are not reported in the dissertation.

## Known issues

The tree builder returned a structurally valid but contentless tree
(`root | Deduced Root Conclusion`, no child nodes) for 632 of the 1,788 cached
queries. 601 of those fall inside the 1,701-query analysis set, so roughly a
third of the treatment condition received an empty scaffold rather than a
populated logic tree. This is documented in the dissertation and the cached
trees are included here so the claim can be checked.

`benchmark/config.py` was edited after the reported runs and its model
assignments no longer match them. The authoritative record of which model ran
in which configuration is `results/dissertation/main_results.csv`.

## Citation

Jhumkhawala, M. (2026). *Optimizing retrieval-augmented generation (RAG) with
an external reasoning engine for enhanced performance* [Doctoral dissertation,
University of the Cumberlands].
