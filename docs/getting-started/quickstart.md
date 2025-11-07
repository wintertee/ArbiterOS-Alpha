# Quick Start

This guide will help you get started with ArbiterOS-alpha in minutes.

## Basic Workflow

1. Create an `ArbiterOSAlpha` instance
2. Define your policies (checkers and routers)
3. Decorate your functions with `@instruction`
4. Use LangGraph normally with added governance

## Step-by-Step Example

### 1. Define Your State

```python
from typing import TypedDict

class State(TypedDict):
    query: str
    response: str
    confidence: float
```

### 2. Create ArbiterOS Instance and Policies

```python
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter
from arbiteros_alpha.instructions import GENERATE, EVALUATE

# Create instance
os = ArbiterOSAlpha()

# Add policy checker to prevent direct toolcall after generate
os.add_policy_checker(
    HistoryPolicyChecker(
        name="no_direct_toolcall",
        bad_sequence=[GENERATE, TOOL_CALL]
    )
)

# Add policy router to regenerate when confidence is low
os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="regenerate_on_low_confidence",
        key="confidence",
        threshold=0.6,
        target="generate"
    )
)
```

### 3. Define and Decorate Functions

```python
@os.instruction(GENERATE)
def generate(state: State) -> dict:
    """Generate AI response."""
    response = "AI generated response"
    return {"response": response}

@os.instruction(EVALUATE)
def evaluate(state: State) -> dict:
    """Evaluate response quality."""
    confidence = 0.8  # Calculate confidence
    return {"confidence": confidence}
```

### 4. Use with LangGraph

```python
from langgraph.graph import StateGraph, END
from arbiteros_alpha import print_history

# Create graph
graph = StateGraph(State)

# Add nodes (decorated functions)
graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)

# Define edges
graph.set_entry_point("generate")
graph.add_edge("generate", "evaluate")
graph.add_edge("evaluate", END)

# Compile and run
app = graph.compile()
result = app.invoke({"query": "What is AI?", "response": "", "confidence": 0.0})

# Print execution history
print_history(os.history)
```

## Understanding the Output

When you run the code, you'll see:

1. **Debug logs**: Showing policy checks and routing decisions
2. **Execution history**: Formatted display with:
   - Timestamps
   - Input/Output states (in YAML format)
   - Policy check results (✓ pass, ✗ fail)
   - Policy routing decisions (→ arrow indicates routing)

## Policy Types

### HistoryPolicyChecker

Validates instruction sequences to prevent forbidden patterns:

```python
checker = HistoryPolicyChecker(
    name="my_checker",
    bad_sequence=[GENERATE, TOOL_CALL]  # Prevents GENERATE→TOOL_CALL
)
```

### MetricThresholdPolicyRouter

Routes based on metric thresholds:

```python
router = MetricThresholdPolicyRouter(
    name="my_router",
    key="confidence",      # Metric key in output
    threshold=0.6,         # Minimum acceptable value
    target="retry_node"    # Where to route if below threshold
)
```

## Next Steps

- Explore the [API Reference](../api/core.md) for detailed documentation
