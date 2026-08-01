# Athena Codebase Documentation

## Overview
Athena is a sophisticated text processing and logic tree generation system that processes text input and creates structured logical representations of the content. The system is designed to break down complex text into manageable chunks and generate hierarchical logic trees that represent the relationships and structure within the text.

## Core Components

### 1. Main Processing Pipeline (`main.py`)
The main entry point of the system that orchestrates the entire process:
- Reads and chunks input text files
- Initializes the dataset builder and model
- Processes text chunks to generate logic trees
- Saves the results in JSON format

### 2. Dataset Builder (`dataset_builder.py`)
A comprehensive module responsible for:
- Creating and managing dataset structures
- Building logic tree skeletons
- Completing tree structures with content
- Handling prompt generation and validation

### 3. Model Integration (`model/`)
Contains implementations for different language models:
- `openai.py`: Integration with OpenAI's GPT models
- `hf.py`: Integration with Hugging Face models
- `model.py`: Base model interface and common functionality

### 4. Logic Tree System (`logic_tree/`)
Implements the core tree data structure:
- `tree.py`: Defines the logic tree structure and operations
- Handles tree construction, traversal, and manipulation
- Manages node relationships and tree validation

### 5. Validators (`validators/`)
Ensures the integrity and correctness of generated structures:
- Structure validation
- Type checking
- Content verification

### 6. Utilities (`utils/`)
Supporting tools and helper functions used throughout the codebase

### 7. Dataset Types (`dataset_types/`)
Defines various data structures and types used in the system

## How It Works

1. **Text Processing**
   - Input text is read and split into manageable chunks
   - Each chunk is processed independently

2. **Tree Generation**
   - For each text chunk:
     - A logic tree skeleton is created
     - The tree is completed with relevant content
     - The structure is validated

3. **Model Integration**
   - Uses language models (OpenAI GPT or Hugging Face) to:
     - Generate tree completions
     - Process and understand text content
     - Create structured representations

4. **Output Generation**
   - Results are saved in two formats:
     - JSON file containing all logic trees
     - Text file with tree string representations

## Key Features

- **Modular Design**: Components are loosely coupled and can be extended
- **Flexible Model Integration**: Supports multiple language model backends
- **Robust Validation**: Multiple validation layers ensure output quality
- **Scalable Processing**: Handles large texts through chunking
- **Structured Output**: Generates well-organized, hierarchical representations

## Usage

The system is primarily used through the `main.py` script, which:
1. Takes an input text file
2. Processes it through the pipeline
3. Generates logic trees
4. Saves the results

## Dependencies

- Python 3.x
- OpenAI API (for GPT model integration)
- Environment variables for API keys and configuration

## Configuration

The system uses environment variables for configuration:
- `OPENAI_API_KEY`: For OpenAI model access
- Other configuration parameters can be set in the environment

## Output Format

The system generates two main output files:
1. `logic_trees_for_chunks.json`: Contains the complete logic tree structures
2. `tree_string-1.txt`: Contains string representations of the trees

Each logic tree represents the hierarchical structure and relationships within the processed text chunks.

## Logic Tree Creation Process

The logic tree creation process is a sophisticated system that breaks down text into hierarchical logical structures. Here's how it works:

### 1. Tree Structure Components

#### LogicNode
- The fundamental building block of the tree
- Contains:
  - `value`: The content/deduction for the node
  - `children`: List of child nodes
  - `fact_type`: Either 'explicit' (mentioned in text) or 'commonsense' (implied knowledge)
  - `operator`: How child nodes combine ('and', 'or', or 'choose')
  - `prunable`: Whether the node can be removed
  - `can_be_leaf`: Whether the node can be a leaf node

#### LogicTree
- The main data structure that manages the tree
- Key parameters:
  - `depth`: Maximum depth of the tree
  - `bf_factor`: Branching factor (controls number of children per node)
  - `chance_to_prune`: Probability of removing nodes
  - `chance_to_prune_all`: Probability of removing all children of a node

### 2. Tree Creation Process

1. **Initial Structure Building**
   - Creates a root node
   - Recursively builds the tree structure based on:
     - Specified depth
     - Branching factor
     - Pruning probabilities

2. **Tree Population**
   - For each node:
     - Determines number of children based on branching factor
     - Creates child nodes with appropriate operators
     - May enforce commonsense facts at each level

3. **Tree Pruning**
   - Removes nodes based on pruning probabilities
   - Can remove individual nodes or entire subtrees
   - Maintains tree integrity during pruning

### 3. Tree Completion Process

1. **Prompt Generation**
   - Creates structured prompts for the language model
   - Includes:
     - Example trees for in-context learning
     - Current tree state
     - Entailment steps to complete

2. **Iterative Completion**
   - For each node:
     - Generates appropriate deductions
     - Creates child nodes with facts
     - Validates the structure
     - May retry on errors

3. **Validation**
   - Ensures tree structure integrity
   - Validates node relationships
   - Checks for contradictions
   - Maintains logical consistency

### 4. Output Generation

The system produces two main outputs:
1. **JSON Structure**
   - Complete tree representation
   - Node relationships
   - Fact types and operators
   - Hierarchical structure

2. **Text Representation**
   - Human-readable tree format
   - Indented structure
   - Node values and relationships
   - Fact types and operators

### 5. Key Features

- **Flexible Structure**: Adaptable to different types of content
- **Validation**: Multiple validation layers ensure output quality
- **Iterative Refinement**: Can retry and refine tree completions
- **Commonsense Integration**: Incorporates both explicit and implicit knowledge
- **Customizable Parameters**: Adjustable depth, branching, and pruning

### 6. Usage Example

```python
# Create a tree structure
tree = LogicTree(
    depth=3,
    bf_factor={2: 0.8, 3: 0.2},
    chance_to_prune=0.3,
    chance_to_prune_all=0.2
)

# Complete the tree with content
completed_tree = builder.complete_structure(
    _tree=tree,
    model=llm,
    description=text_content,
    completion_prompt_fn=prompt_function
)
```

This system is particularly useful for:
- Breaking down complex text into logical components
- Creating structured representations of content
- Generating hierarchical reasoning guides
- Analyzing relationships between facts and deductions

## Node Generation Process

### 1. Node Creation Parameters

Each `LogicNode` is created with the following key parameters:
- `value`: The content/deduction for the node (string)
- `operator`: How child nodes combine ('and', 'or', or 'choose')
- `fact_type`: Either 'explicit' (mentioned in text) or 'commonsense' (implied knowledge)
- `prunable`: Whether the node can be removed (boolean)
- `can_be_leaf`: Whether the node can be a leaf node (boolean)
- `frozen`: Whether the node's structure can be modified (boolean)

### 2. Node Population Process

The `populate` method in `LogicTree` handles node generation with the following steps:

1. **Operator Determination**
   ```python
   if node.operator == LogicNodeOperatorType.CHOOSE:
       node.operator = LogicNodeOperatorType.OR if random.random() < self.chance_of_or else LogicNodeOperatorType.AND
   ```
   - If operator is 'choose', randomly selects between 'and' or 'or'
   - Uses `chance_of_or` parameter to determine probability

2. **Branching Factor Calculation**
   ```python
   bf = max(0, random.choices(list(self.bf_factor.keys()), 
                            list(self.bf_factor.values()), k=1)[0] - len(node.children))
   ```
   - Determines how many children to create
   - Uses `bf_factor` dictionary to weight different branching possibilities
   - Subtracts existing children from the target number

3. **Child Node Generation**
   For each new child node:
   - **Fact Type Selection**
     ```python
     fact_type = LogicNodeFactType.COMMONSENSE if random.random() < self.chance_of_cs_fact 
                 else LogicNodeFactType.EXPLICIT
     ```
     - Randomly determines if node is commonsense or explicit fact
     - Uses `chance_of_cs_fact` parameter
     - Ensures only one commonsense fact per level if `enforce_cs_fact_per_level` is true

   - **Operator Assignment**
     ```python
     if roll_for_or > self.chance_of_or and current_depth < self.depth:
         # Create AND node
     else:
         # Create OR node
     ```
     - Determines operator based on depth and probability
     - AND nodes can have children, OR nodes are typically leaf nodes

4. **Recursive Population**
   ```python
   if current_depth < self.depth:
       for node in node.children:
           if node.fact_type == LogicNodeFactType.EXPLICIT:
               self.populate(node, current_depth+1)
   ```
   - Recursively populates child nodes
   - Stops at specified depth
   - Skips commonsense nodes

### 3. Node Pruning

After population, nodes may be pruned based on:
```python
if random.random() < self.chance_to_prune_all:
    node.children = []
```
- `chance_to_prune`: Probability of removing individual nodes
- `chance_to_prune_all`: Probability of removing all children
- Maintains minimum required children (1 for OR, 2 for AND)

### 4. Example Node Creation

```python
# Creating a basic node
node = LogicNode(
    value="Eruptions block sunlight",
    operator=LogicNodeOperatorType.AND,
    fact_type=LogicNodeFactType.EXPLICIT,
    prunable=True,
    can_be_leaf=False
)

# Creating a commonsense node
cs_node = LogicNode(
    value="Ash blocks sunlight",
    operator=LogicNodeOperatorType.OR,
    fact_type=LogicNodeFactType.COMMONSENSE,
    prunable=False,
    can_be_leaf=True
)
```

### 5. Node Properties

- **Parent-Child Relationship**: Each node maintains a reference to its parent
- **Value Inheritance**: Child nodes combine to form parent node's value
- **Type Constraints**: Commonsense nodes are typically leaf nodes
- **Structural Rules**: 
  - AND nodes require at least 2 children
  - OR nodes can have any number of children
  - Commonsense facts are typically combined with AND operators
