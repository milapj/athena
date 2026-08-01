# TAG: Thought Augmented Generation

## Overview

**TAG (Thought Augmented Generation)** is an evolution of the traditional RAG (Retrieval Augmented Generation) pipeline. Instead of retrieving flat text chunks based on semantic similarity, TAG retrieves **structured logic trees** that capture complete reasoning chains.

The key insight: **Logic trees don't just store WHAT a document says - they store WHY and HOW conclusions are reached.**

---

## RAG vs TAG Comparison

See `rag_vs_tag.drawio` for the visual diagram.

### Traditional RAG Pipeline

```
Document → Chunk → Embed → Vector DB → Retrieve → LLM → Response
```

1. **Document Processing**: Split document into fixed-size chunks (512-2000 tokens)
2. **Embedding**: Convert chunks to vectors using embedding model
3. **Storage**: Store vectors in vector database (Pinecone, Chroma, etc.)
4. **Retrieval**: Find top-K chunks by cosine similarity to query
5. **Generation**: Pass chunks to LLM as context

**Limitations:**
- No logical structure preserved
- Lost reasoning chains
- Semantic similarity ≠ logical relevance
- Cannot trace deductions
- Chunks lack context of WHY

### TAG Pipeline

```
Document → Chunk → Logic Tree → Tree DB → Retrieve Trees → LLM → Response
```

1. **Document Processing**: Same chunking as RAG
2. **Tree Generation**: Build logic trees for each chunk (using Athena)
3. **Storage**: Store trees with metadata (can still embed root conclusions)
4. **Retrieval**: Match query to relevant trees (by conclusion or embedded search)
5. **Generation**: Pass complete reasoning trees to LLM

**Advantages:**
- Structured reasoning chains preserved
- Traceable deductions with evidence
- Explicit vs commonsense facts distinguished
- Logical entailment captured
- WHY is captured, not just WHAT

---

## How TAG Works

### 1. Preprocessing Phase

Using Athena, transform documents into logic trees:

```python
# For each document chunk
logic_tree = builder.build_structure(depth=3, ...)
completed_tree = builder.complete_structure(logic_tree, model, chunk_text, ...)
```

**Output per chunk:**
```
Root Conclusion: "Company violated Section 34"
├── Deduction: "Failed to file returns"
│   ├── Explicit: "No filing since 2019"
│   └── Commonsense: "Filing is mandatory"
├── Deduction: "Had legal obligation"
│   ├── Explicit: "Section 34 applies"
│   └── Explicit: "Company is registered"
└── Commonsense: "Non-compliance = violation"
```

### 2. Storage Phase

Store logic trees with indexing capabilities:

```python
{
    "chunk_id": "doc_001_chunk_005",
    "source_document": "legal_case.pdf",
    "root_conclusion": "Company violated Section 34",
    "root_embedding": [0.12, -0.34, ...],  # Optional: for similarity search
    "full_tree": { ... },
    "explicit_facts": ["No filing since 2019", ...],
    "commonsense_facts": ["Filing is mandatory", ...]
}
```

**Indexing Options:**
- Embed root conclusions for semantic search
- Full-text index on explicit facts
- Tag-based indexing on commonsense categories
- Hierarchical indexing by tree structure

### 3. Retrieval Phase

When a query comes in, retrieve relevant trees:

**Query:** "Why was the company found in violation?"

**Retrieval Methods:**
1. **Semantic Match**: Embed query, find similar root conclusions
2. **Keyword Match**: Search explicit facts for relevant terms
3. **Structured Query**: Match against tree patterns

**Retrieved Tree:**
```
Conclusion: "Violated Section 34"
├── "Failed to file returns"
│   └── FACT: "No filing since 2019"
├── "Had legal obligation"
│   └── FACT: "Company registered"
└── CS: "Non-compliance = violation"
```

### 4. Generation Phase

Pass the complete reasoning tree to the LLM:

**Prompt:**
```
You have access to the following reasoning tree from the document:

[Tree in structured format]

Question: Why was the company found in violation?

Using the reasoning tree above, provide a clear answer that traces
the logical chain from facts to conclusion.
```

**LLM Response (with TAG):**
```
The company was found in violation of Section 34 because:

1. They failed to file required annual returns
   - Evidence: No filing records found since 2019
   - This matters because annual filing is mandatory

2. They had a legal obligation to file
   - Evidence: Section 34 applies to all registered companies
   - Evidence: OTTOMAN TUBES is a registered Pvt. Ltd. company

3. Non-compliance with legal requirements constitutes a violation

Therefore, the combination of having an obligation (registered company
under Section 34) and failing to meet it (no filings since 2019)
resulted in the violation finding.
```

---

## Implementation Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      TAG Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Athena    │───▶│  Tree Store │───▶│  Retriever  │     │
│  │  (Builder)  │    │  (Database) │    │   (Query)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                                     │             │
│         ▼                                     ▼             │
│  ┌─────────────┐                       ┌─────────────┐     │
│  │    LLM      │                       │    LLM      │     │
│  │ (Generate)  │                       │  (Respond)  │     │
│  └─────────────┘                       └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Tree Store Options

1. **Document Database** (MongoDB, CouchDB)
   - Store trees as JSON documents
   - Query by metadata fields
   - Good for structured queries

2. **Vector Database** (Pinecone, Chroma)
   - Embed root conclusions
   - Semantic similarity search
   - Combine with metadata filtering

3. **Graph Database** (Neo4j)
   - Native tree structure support
   - Pattern matching queries
   - Traverse reasoning chains

4. **Hybrid Approach**
   - Vector DB for semantic retrieval
   - Document DB for tree storage
   - Link by IDs

---

## Query Strategies

### 1. Conclusion Matching
Find trees whose root conclusions are semantically similar to the query.

```python
query_embedding = embed(user_query)
similar_trees = vector_db.search(query_embedding, top_k=5)
```

### 2. Fact-Based Retrieval
Search for trees containing specific facts mentioned in the query.

```python
keywords = extract_keywords(user_query)
matching_trees = tree_db.search_facts(keywords)
```

### 3. Multi-Hop Reasoning
Chain multiple trees together for complex questions.

```python
# Find trees that answer sub-questions
tree1 = retrieve("What obligation did the company have?")
tree2 = retrieve("Did the company meet its obligations?")
tree3 = retrieve("What are the consequences?")
combined_context = merge_trees([tree1, tree2, tree3])
```

### 4. Evidence-First Retrieval
Start from explicit facts and trace up to conclusions.

```python
# Find trees containing specific evidence
trees = tree_db.search_explicit_facts("no filing since 2019")
# Return full reasoning chain for each
```

---

## Prompt Templates for TAG

### Basic Tree-Grounded Response
```
Given the following reasoning tree from the source document:

{tree_formatted}

Answer the question: {user_query}

Base your answer on the reasoning chain shown above. Cite specific
facts when making claims.
```

### Multi-Tree Synthesis
```
I have retrieved the following reasoning trees relevant to your question:

Tree 1 (Conclusion: {conclusion_1}):
{tree_1}

Tree 2 (Conclusion: {conclusion_2}):
{tree_2}

Question: {user_query}

Synthesize information from both trees to provide a complete answer.
```

### Evidence Tracing
```
The following reasoning tree explains {topic}:

{tree_formatted}

For each claim in the tree marked as "Explicit Fact", this information
was directly stated in the source document.

For claims marked "Commonsense", these are logical inferences.

Answer: {user_query}

Include citations to explicit facts in your response.
```

---

## Benefits of TAG over RAG

| Aspect | RAG | TAG |
|--------|-----|-----|
| **Context Structure** | Flat text chunks | Hierarchical reasoning trees |
| **Reasoning** | LLM must infer | Pre-computed reasoning chains |
| **Traceability** | Limited | Full evidence chain |
| **Fact Types** | All text equal | Explicit vs commonsense |
| **Logical Relations** | Implicit | Explicit (AND/OR operators) |
| **Retrieval Unit** | Text chunk | Reasoning tree |
| **Answer Quality** | Variable | Structured, logical |

---

## Use Cases

### Legal Document Analysis
- Retrieve reasoning for court decisions
- Trace legal arguments to source citations
- Distinguish legal rules from case-specific facts

### Scientific Literature
- Capture hypothesis → evidence → conclusion chains
- Distinguish experimental findings from theoretical background
- Enable multi-paper reasoning synthesis

### Financial Analysis
- Structure investment theses
- Trace conclusions to supporting data
- Separate market data from analyst assumptions

### Educational Content
- Generate explanations with clear reasoning steps
- Adapt depth of explanation by tree pruning
- Identify prerequisite knowledge (commonsense nodes)

---

## Future Directions

1. **Dynamic Tree Construction**: Build trees at query time for novel document combinations

2. **Tree Summarization**: Compress trees for constrained context windows

3. **Interactive Exploration**: Allow users to traverse trees and ask follow-up questions

4. **Cross-Document Trees**: Link trees from multiple documents for complex reasoning

5. **Tree Validation**: Use LLMs to verify logical consistency of retrieved trees

6. **Confidence Scoring**: Weight facts and deductions by source reliability
