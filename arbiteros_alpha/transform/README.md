## Introduction

The transform module automates the process of adding ArbiterOS governance to existing agent code. Instead of manually adding `@os.instruction()` decorators and OS initialization, you can run a single command that:

1. **Parses** your Python file to detect agent type (LangGraph or vanilla)
2. **Extracts** all node functions that need classification
3. **Classifies** each function into an appropriate instruction type (using LLM or manual selection)
4. **Transforms** the code by adding imports, OS initialization, decorators, and registration calls
5. **Creates** a backup of your original file

This makes migration to ArbiterOS seamless and reduces the chance of errors.

## Quick Start

### Basic Usage

Transform a LangGraph agent with automatic LLM classification:

```bash
uv run -m arbiteros_alpha.transform path/to/agent.py
```

Transform with manual classification (interactive prompts):

```bash
uv run -m arbiteros_alpha.transform path/to/agent.py --manual
```

Preview changes without modifying files:

```bash
uv run -m arbiteros_alpha.transform path/to/agent.py --dry-run
```

Non-interactive mode (skip confirmations):

```bash
uv run -m arbiteros_alpha.transform path/to/agent.py --yes
```

### Programmatic Usage

```python
from arbiteros_alpha.transform.parser import AgentParser
from arbiteros_alpha.transform.classifier import InstructionClassifier, ClassificationConfig
from arbiteros_alpha.transform.generator import CodeGenerator

# Parse the agent
parser = AgentParser()
parsed = parser.parse_file("my_agent.py")

# Classify functions
config = ClassificationConfig(api_key="your-key")
classifier = InstructionClassifier(config=config)
classifications = {}
for func in parsed.functions:
    if func.is_node_function:
        classifications[func.name] = classifier.classify(func)

# Transform the code
generator = CodeGenerator()
result = generator.transform(
    file_path="my_agent.py",
    parsed_agent=parsed,
    classifications=classifications,
    dry_run=False,
)

if result.success:
    print(f"Transformed! Backup: {result.backup_file}")
```

## Usage Examples

### Example 1: LangGraph Agent Transformation

**Before:**

```python
from langgraph.graph import StateGraph, END, START

def generate(state):
    return {"response": "Hello"}

def verify(state):
    return state

builder = StateGraph(dict)
builder.add_node("generate", generate)
builder.add_node("verify", verify)
graph = builder.compile()
```

**After transformation:**

```python
from langgraph.graph import StateGraph, END, START
from arbiteros_alpha import ArbiterOSAlpha
import arbiteros_alpha.instructions as Instr

os = ArbiterOSAlpha(backend="langgraph")

@os.instruction(Instr.GENERATE)
def generate(state):
    return {"response": "Hello"}

@os.instruction(Instr.VERIFY)
def verify(state):
    return state

builder = StateGraph(dict)
builder.add_node("generate", generate)
builder.add_node("verify", verify)
graph = builder.compile()
os.register_compiled_graph(graph)
```

### Example 2: Vanilla Agent Transformation

**Before:**

```python
def generate(state):
    return {"response": "Hello"}

def verify(state):
    return state

state = {}
state.update(generate(state))
state.update(verify(state))
```

**After transformation:**

```python
from arbiteros_alpha import ArbiterOSAlpha
import arbiteros_alpha.instructions as Instr

os = ArbiterOSAlpha(backend="vanilla")

@os.instruction(Instr.GENERATE)
def generate(state):
    return {"response": "Hello"}

@os.instruction(Instr.VERIFY)
def verify(state):
    return state

state = {}
state.update(generate(state))
state.update(verify(state))
```

### Example 3: Manual Classification

When using `--manual` flag, you'll be prompted for each function:

```
Select instruction type for 'generate':
  1. GENERATE
  2. DECOMPOSE
  3. REFLECT
  ...
Enter number: 1
```

### Example 4: Dry Run Preview

Use `--dry-run` to see what changes would be made:

```bash
$ uv run -m arbiteros_alpha.transform agent.py --dry-run

Dry run complete - no files modified
Changes that would be made:
  • Added import: from arbiteros_alpha import ArbiterOSAlpha
  • Added import: import arbiteros_alpha.instructions as Instr
  • Added OS initialization at line 5
  • Added @os.instruction(Instr.GENERATE) to generate
  • Added @os.instruction(Instr.VERIFY) to verify
  • Added os.register_compiled_graph(graph)
```
