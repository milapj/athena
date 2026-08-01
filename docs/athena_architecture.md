# Athena: Logic Tree Generation System

## Overview

Athena is a system that transforms unstructured text documents into structured **logic trees** (also known as entailment trees). Unlike traditional text processing that treats documents as flat chunks, Athena extracts the underlying reasoning structure - capturing not just *what* a document says, but *how* conclusions are reached.

## Architecture Diagram

See `athena_architecture.drawio` for the visual representation.

---

## 1. Generation Pipeline

### Step 1: Input Document
- **Source**: PDFs, Text etc
- **Extraction**: PyPDF2 library extracts raw text from PDF files
- **Function**: `extract_text_from_pdf()` in `main.py`

### Step 2: Text Chunking
- **Method**: Fixed character-based splitting (2000 characters per chunk)
- **Purpose**: Makes processing manageable while preserving context
- **Function**: `chunk_text_by_chars()` in `main.py`

```python
def chunk_text_by_chars(text: str, chunk_size: int = 2000) -> list:
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
```

### Step 3: Tree Skeleton Generation
- **Class**: `LogicTree` in `logic_tree/tree.py`
- **Method**: `build_structure()` in `DatasetBuilder`
- **Process**: Creates an empty tree structure with placeholders for facts

**Key Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `depth` | Maximum levels in the tree | `3` |
| `bf_factor` | Branching factor distribution | `{1: 0.5, 2: 0.5}` |
| `chance_to_prune` | Probability of removing a node | `0.3` |
| `chance_to_prune_all` | Probability of removing all children | `0.3` |
| `enforce_cs_fact_per_level` | Ensure commonsense fact at each level | `True` |

### Step 4: LLM Completion
- **Class**: `OpenAIModel` in `model/openai.py`
- **Method**: `iteratively_complete_v2()` in `DatasetBuilder`
- **Model**: GPT-4o (configurable)
- **Process**: Recursively fills empty nodes with generated content

**How it works:**
1. For each node with empty children, create a prompt
2. Prompt includes current tree state and expected structure
3. LLM generates deductions and facts
4. Parse output for "Fact From Story" (explicit) and "Commonsense Knowledge"
5. Fill child nodes with parsed content
6. Recurse to children

### Step 5: Structure Validation
- **Class**: `StructureValidator` in `validators/types/structure_validator.py`
- **Purpose**: Ensures LLM output matches the expected template
- **Checks**:
  - Correct number of explicit facts
  - Correct number of commonsense facts
  - No duplicate facts
- **On Failure**: Retry with error feedback appended to prompt

### Step 6: Output
- **JSON**: `logic_trees_for_chunks.json` - Complete tree structures
- **Text**: `tree_string-*.txt` - Human-readable format

---

## 2. Logic Tree Structure

### Node Types

#### Root Conclusion (Green)
The top-level conclusion that all child facts support. This is the main deduction derived from the source text.

#### Intermediate Deduction (Yellow)
Mid-level nodes that have children. They represent intermediate reasoning steps that combine facts into higher-level conclusions.

#### Explicit Fact (Blue, Square)
Leaf nodes containing facts **directly stated** in the source document. These are extractable quotes or paraphrases from the original text.

```python
fact_type = LogicNodeFactType.EXPLICIT  # "explicit"
```

#### Commonsense Knowledge (Gray, Dashed)
Leaf nodes containing **implied knowledge** that doesn't need to be stated. These represent reasoning steps that most people would agree are true.

```python
fact_type = LogicNodeFactType.COMMONSENSE  # "commonsense"
```

### Operators

#### AND Operator
All children must be true for the parent to be true.

```
Parent Conclusion
├── Fact A  ─┐
├── Fact B  ─┼─► ALL required
└── Fact C  ─┘
```

#### OR Operator
Any child being true makes the parent true.

```
Parent Conclusion
├── Fact A  ─┐
├── Fact B  ─┼─► ANY sufficient
└── Fact C  ─┘
```

### Example Tree

```
> "OTTOMAN TUBES violated Section 34" | Deduced Root Conclusion
> > "Failed to file required returns" | Deduced Fact
> > > "No filing records since 2019" | Fact From Story
> > > "Annual filing is mandatory" | Commonsense Knowledge
> > "Had legal obligation to file" | Deduced Fact
> > > "Section 34 applies to registered companies" | Fact From Story
> > > "OTTOMAN TUBES is registered" | Fact From Story
> > "Non-compliance constitutes violation" | Commonsense Knowledge
```

---

## 3. Key Code Modules

### `logic_tree/tree.py`
Core data structures for the logic tree.

- **LogicNode**: Tree primitive representing a fact or deduction
  - `value`: Content of the node
  - `children`: List of child nodes
  - `fact_type`: "explicit" or "commonsense"
  - `operator`: "and" or "or"
  - `prunable`: Whether node can be removed
  - `frozen`: Whether structure is locked

- **LogicTree**: Main tree structure
  - `populate()`: Build tree structure
  - `prune()`: Remove nodes based on probability
  - `get_facts()`: Extract leaf facts
  - `print_for_gpt()`: Format for LLM prompts
  - `to_json()` / `from_json()`: Serialization

### `dataset_builder.py`
Orchestrates tree construction and completion.

- **DatasetBuilder**: Main builder class
  - `build_structure()`: Create empty tree skeleton
  - `complete_structure()`: Fill tree with LLM
  - `create_completion_prompt()`: Generate prompts
  - `iteratively_complete_v2()`: Recursive completion algorithm

### `model/openai.py`
LLM integration layer.

- **OpenAIModel**: OpenAI API wrapper
  - `inference()`: Call the API
  - Built-in retry logic
  - Rate limiting handling
  - Support for chat and completion endpoints

### `validators/`
Output validation system.

- **Validator**: Abstract base class
- **StructureValidator**: Verifies fact counts match template

### `main.py`
Entry point and orchestration.

- PDF text extraction
- Chunking
- Processing loop
- Output generation

---

## 4. Prompt Engineering

The system uses carefully crafted prompts to guide the LLM in generating valid tree structures.

### Base Prompt Structure
```
We are making a reasoning guide for [domain]. To do this, we are using
an entailment tree. An entailment tree is a tree structure where
intermediate nodes are entailed by their children.

Facts From Story are facts that will be explicitly stated.
Commonsense Knowledge are facts that most people would agree are true.

All facts for the step must combine to entail the root parent fact.
```

### In-Context Learning
The system uses example trees to guide the LLM:
1. Show example scenario
2. Show current tree state
3. Show entailment step to complete
4. Show expected output format

### Because Clause
Adding "Because, " after parent nodes helps the LLM generate facts that logically flow:
```
> Parent conclusion Because,
> > [Child fact that explains WHY]
```

---

## 5. Data Flow

```
PDF File
    ↓
extract_text_from_pdf()
    ↓
Raw Text String
    ↓
chunk_text_by_chars(2000)
    ↓
List of Chunks
    ↓
For each chunk:
    ├── build_structure() → Empty LogicTree
    ├── complete_structure() → Filled LogicTree
    │   ├── create_completion_prompt()
    │   ├── OpenAIModel.inference()
    │   ├── Parse output
    │   ├── StructureValidator.validate()
    │   └── Retry if invalid
    └── to_json()
    ↓
logic_trees_for_chunks.json
```

---

## 6. Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
```

### Tree Parameters (in main.py)
```python
logic_tree = builder.build_structure(
    depth=3,
    bf_factor={1: 2.0, 2: 1.0},
    chance_to_prune_all=0.3,
    chance_to_prune=0.3,
    root_nodes=[]
)
```

### Model Configuration
```python
llm = OpenAIModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    engine="gpt-4o",
    api_endpoint="chat",
    temperature=0.7,
    max_tokens=16000
)
```

---

## 7. Output Formats

### JSON Structure
```json
{
  "filename": "document.pdf",
  "total_chunks": 25,
  "trees": [
    {
      "chunk_index": 1,
      "chunk_text": "The court held that...",
      "logic_tree": {
        "nodes": [{
          "value": "Company violated Section 34",
          "operator": "and",
          "fact_type": "explicit",
          "children": [...]
        }]
      }
    }
  ]
}
```

### Text Representation
```
> Root Conclusion | Deduced Root Conclusion
> > Intermediate Deduction | Deduced Fact
> > > Leaf Fact | Fact From Story
> > > Implied Knowledge | Commonsense Knowledge
```
