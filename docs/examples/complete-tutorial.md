# Complete Tutorial: AI Assistant with Policy Governance

This tutorial walks through a complete working example that demonstrates all key features of ArbiterOS-alpha. We'll build an AI assistant with quality control through policy-driven governance.

## Overview

The example implements an AI assistant workflow with:

- **Policy Checking**: Prevents forbidden instruction sequences
- **Dynamic Routing**: Automatically retries when quality is low
- **Execution Tracking**: Full history with timestamps and I/O
- **LangGraph Integration**: Standard LangGraph patterns with governance

## Complete Code

Here's the full `examples/main.py`:

```python
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from rich.logging import RichHandler

from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter
from arbiteros_alpha.instructions import GENERATE, TOOL_CALL, EVALUATE

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[RichHandler()],
)

# 1. Setup OS
os = ArbiterOSAlpha()

# Policy: Prevent direct generate->toolcall without proper flow
history_checker = HistoryPolicyChecker(
    name="no_direct_toolcall", bad_sequence=[GENERATE, TOOL_CALL]
)

# Policy: If confidence is low, regenerate the response
confidence_router = MetricThresholdPolicyRouter(
    name="regenerate_on_low_confidence",
    key="confidence",
    threshold=0.6,
    target="generate",
)

os.add_policy_checker(history_checker)
os.add_policy_router(confidence_router)

# 2. Define State
class State(TypedDict):
    """State for a simple AI assistant with tool usage and self-evaluation."""
    query: str
    response: str
    tool_result: str
    confidence: float

# 3. Define Instructions
@os.instruction(GENERATE)
def generate(state: State) -> dict:
    """Generate a response to the user query."""
    is_retry = bool(state.get("response"))
    
    if is_retry:
        response = "Here is my comprehensive and detailed response with much more content and explanation."
    else:
        response = "Short reply."
    
    return {"response": response}

@os.instruction("toolcall")
def tool_call(state: State) -> dict:
    """Call external tools to enhance the response."""
    return {"tool_result": "ok"}

@os.instruction("evaluate")
def evaluate(state: State) -> dict:
    """Evaluate confidence in the response quality."""
    response_length = len(state["response"])
    confidence = min(response_length / 100.0, 1.0)
    return {"confidence": confidence}

# 4. Build LangGraph
builder = StateGraph(State)
builder.add_node(GENERATE, generate)
builder.add_node(TOOL_CALL, tool_call)
builder.add_node(EVALUATE, evaluate)

builder.add_edge(START, GENERATE)
builder.add_edge(GENERATE, TOOL_CALL)
builder.add_edge(TOOL_CALL, EVALUATE)
builder.add_edge(EVALUATE, END)

graph = builder.compile()

# 5. Run
initial_state: State = {
    "query": "What is AI?",
    "response": "",
    "tool_result": "",
    "confidence": 0.0,
}

for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
    logger.info(f"Current state: {chunk}\n")

# 6. View History
from arbiteros_alpha import print_history
print_history(os.history)
```

## Step-by-Step Explanation

### Step 1: Setup ArbiterOS and Policies

```python
os = ArbiterOSAlpha()

# Policy 1: Sequence Validation
history_checker = HistoryPolicyChecker(
    name="no_direct_toolcall", 
    bad_sequence=["generate", "toolcall"]
)

# Policy 2: Quality-Based Routing
confidence_router = MetricThresholdPolicyRouter(
    name="regenerate_on_low_confidence",
    key="confidence",
    threshold=0.6,
    target="generate",
)

os.add_policy_checker(history_checker)
os.add_policy_router(confidence_router)
```

**What's happening:**

- **`ArbiterOSAlpha()`**: Creates the governance coordinator
- **`HistoryPolicyChecker`**: Monitors instruction sequences and flags when `generate→toolcall` pattern occurs (though it allows execution to continue)
- **`MetricThresholdPolicyRouter`**: Watches the `confidence` metric; if it's below 0.6, routes back to `generate` for a retry
- These policies are registered with the OS instance

### Step 2: Define State Schema

```python
class State(TypedDict):
    """State for a simple AI assistant with tool usage and self-evaluation."""
    query: str
    response: str
    tool_result: str
    confidence: float
```

**What's happening:**

Standard LangGraph state definition. The state flows through all nodes and accumulates information:

- `query`: User's question
- `response`: Generated answer
- `tool_result`: Result from tool execution
- `confidence`: Quality score (0.0 to 1.0)

### Step 3: Define Instruction Functions

#### Generate Instruction

```python
@os.instruction("generate")
def generate(state: State) -> dict:
    """Generate a response to the user query."""
    is_retry = bool(state.get("response"))
    
    if is_retry:
        response = "Here is my comprehensive and detailed response with much more content and explanation."
    else:
        response = "Short reply."
    
    return {"response": response}
```

**What's happening:**

- The `@os.instruction("generate")` decorator wraps the function with governance
- **First call**: Returns a short response (simulating low-quality output)
- **Retry call**: Detects existing response and generates a longer, better one
- Returns only the fields that changed (partial state update)

#### Tool Call Instruction

```python
@os.instruction(TOOL_CALL)
def tool_call(state: State) -> dict:
    """Call external tools to enhance the response."""
    return {"tool_result": "ok"}
```

**What's happening:**

- Simulates calling external tools (APIs, databases, etc.)
- The decorator tracks this execution and checks policies
- Returns the tool result

#### Evaluate Instruction

```python
@os.instruction(EVALUATE)
def evaluate(state: State) -> dict:
    """Evaluate confidence in the response quality."""
    response_length = len(state["response"])
    confidence = min(response_length / 100.0, 1.0)
    return {"confidence": confidence}
```

**What's happening:**

- Calculates quality metric based on response length
- Short responses (< 60 chars) get confidence < 0.6 (triggers retry)
- Longer responses (>= 60 chars) get confidence >= 0.6 (passes)
- This is where the routing decision is made

### Step 4: Build LangGraph

```python
builder = StateGraph(State)
builder.add_node(generate)
builder.add_node(tool_call)
builder.add_node(evaluate)

builder.add_edge(START, "generate")
builder.add_edge("generate", "tool_call")
builder.add_edge("tool_call", "evaluate")
builder.add_edge("evaluate", END)

graph = builder.compile()
```

**What's happening:**

Standard LangGraph construction:

```
START → generate → tool_call → evaluate → END
           ↑__________________________|
           (routes back if confidence < 0.6)
```

The routing from `evaluate` back to `generate` happens dynamically through the `MetricThresholdPolicyRouter`, not through static edges.

### Step 5: Execute

```python
initial_state: State = {
    "query": "What is AI?",
    "response": "",
    "tool_result": "",
    "confidence": 0.0,
}

for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
    logger.info(f"Current state: {chunk}\n")
```

**What's happening:**

- Starts with empty response and zero confidence
- Streams through the graph, printing state at each step
- ArbiterOS governance runs automatically at each instruction

### Step 6: View History

```python
from arbiteros_alpha import print_history
print_history(os.history)
```

Displays formatted execution history with all decisions and state changes.

## Execution Flow

The actual execution follows this path:

### First Iteration (Low Quality)

1. **`GENERATE` (attempt #1)**
   - Input: `{query: "What is AI?", response: "", ...}`
   - Output: `{response: "Short reply."}`
   - Policy Check: ✓ No violations (first call)

2. **`TOOL_CALL`**
   - Input: `{..., response: "Short reply.", ...}`
   - Output: `{tool_result: "ok"}`
   - Policy Check: ✗ Detects `GENERATE→TOOL_CALL` sequence (flagged but continues)

3. **`EVALUATE`**
   - Input: `{..., response: "Short reply.", tool_result: "ok", ...}`
   - Output: `{confidence: 0.12}` (13 chars / 100 = 0.13)
   - Policy Check: ✗ Still has `GENERATE→TOOL_CALL` in history
   - **Policy Route**: ⚡ **`confidence < 0.6` → Routes to `generate`**

### Second Iteration (High Quality)

4. **`GENERATE` (attempt #2 - retry)**
   - Input: `{..., response: "Short reply.", ...}` (response exists)
   - Output: `{response: "Here is my comprehensive and detailed response with much more content and explanation."}`
   - Policy Check: ✗ Still has old `GENERATE→TOOL_CALL` in history

5. **`TOOL_CALL`**
   - Input: `{..., response: "Here is my comprehensive...", ...}`
   - Output: `{tool_result: "ok"}`
   - Policy Check: ✗ Multiple `GENERATE→TOOL_CALL` sequences now

6. **`EVALUATE`**
   - Input: `{..., response: "Here is my comprehensive...", ...}`
   - Output: `{confidence: 0.86}` (86 chars / 100 = 0.86)
   - Policy Check: ✗ Multiple violations in history
   - **Policy Route**: ✓ **`confidence >= 0.6` → No routing, continues to END**

### Final State

```python
{
    "query": "What is AI?",
    "response": "Here is my comprehensive and detailed response with much more content and explanation.",
    "tool_result": "ok",
    "confidence": 0.86
}
```

## Example Output

When you run `uv run -m examples.main`, you'll see:

### Console Logs

```
[DEBUG] Adding policy checker: HistoryPolicyChecker(name='no_direct_toolcall', bad_sequence='GENERATE->TOOL_CALL')
[DEBUG] Adding policy router: MetricThresholdPolicyRouter(name='regenerate_on_low_confidence', key='confidence', threshold=0.6, target='generate')

[DEBUG] Executing instruction: GENERATE
[DEBUG] Running 1 policy checkers (before)
[DEBUG] Instruction GENERATE returned: {'response': 'Short reply.'}
[DEBUG] Checking 1 policy routers

[DEBUG] Executing instruction: TOOL_CALL
[DEBUG] Running 1 policy checkers (before)
[ERROR] Blacklisted sequence detected: no_direct_toolcall:[GENERATE->TOOL_CALL] in [GENERATE->TOOL_CALL]
[ERROR] Policy checker HistoryPolicyChecker(...) failed validation.
[DEBUG] Instruction TOOL_CALL returned: {'tool_result': 'ok'}

[DEBUG] Executing instruction: EVALUATE
[DEBUG] Instruction EVALUATE returned: {'confidence': 0.12}
[WARNING] Routing decision made to: generate
[INFO] Routing from evaluate to generate

[DEBUG] Executing instruction: GENERATE
[DEBUG] Instruction GENERATE returned: {'response': 'Here is my comprehensive...'}
[DEBUG] Instruction EVALUATE returned: {'confidence': 0.86}
```

### Execution History

```
📋 Arbiter OS Execution History
================================================================================

[1] GENERATE
  Timestamp: 2025-11-05 10:12:24.659058
  Input:
    query: What is AI?
    response: ''
    tool_result: ''
    confidence: 0.0
  Output:
    response: Short reply.
  Policy Checks:
    (none)
  Policy Routes:
    (none)

[2] TOOL_CALL
  Timestamp: 2025-11-05 10:12:24.662379
  Input:
    query: What is AI?
    response: Short reply.
    tool_result: ''
    confidence: 0.0
  Output:
    tool_result: ok
  Policy Checks:
    ✗ no_direct_toolcall
  Policy Routes:
    (none)

[3] EVALUATE
  Timestamp: 2025-11-05 10:12:24.666841
  Input:
    query: What is AI?
    response: Short reply.
    tool_result: ok
    confidence: 0.0
  Output:
    confidence: 0.12
  Policy Checks:
    ✗ no_direct_toolcall
  Policy Routes:
    → regenerate_on_low_confidence ⇒ generate

[4] GENERATE
  Timestamp: 2025-11-05 10:12:24.673606
  Input:
    query: What is AI?
    response: Short reply.
    tool_result: ok
    confidence: 0.12
  Output:
    response: Here is my comprehensive and detailed response with much more content and
      explanation.
  Policy Checks:
    ✗ no_direct_toolcall
  Policy Routes:
    (none)

[5] TOOL_CALL
  Timestamp: 2025-11-05 10:12:24.679333
  Input:
    query: What is AI?
    response: Here is my comprehensive and detailed response with much more content and
      explanation.
    tool_result: ok
    confidence: 0.12
  Output:
    tool_result: ok
  Policy Checks:
    ✗ no_direct_toolcall
  Policy Routes:
    (none)

[6] EVALUATE
  Timestamp: 2025-11-05 10:12:24.683659
  Input:
    query: What is AI?
    response: Here is my comprehensive and detailed response with much more content and
      explanation.
    tool_result: ok
    confidence: 0.12
  Output:
    confidence: 0.86
  Policy Checks:
    ✗ no_direct_toolcall
  Policy Routes:
    (none)

================================================================================
```

## Key Insights

### 1. Policy Checking vs Routing

- **Policy Checkers** (HistoryPolicyChecker): **Detect and flag** violations but don't block execution
- **Policy Routers** (MetricThresholdPolicyRouter): **Actively redirect** execution flow

### 2. Automatic Retry Pattern

The router implements an automatic quality control loop:

```
Low Quality (confidence < 0.6) → Retry
High Quality (confidence >= 0.6) → Continue
```

No manual retry logic needed in your code!

### 3. Full Observability

Every execution is tracked:

- ✅ What ran, when, and with what inputs/outputs
- ✅ Which policies triggered
- ✅ Why routing decisions were made
- ✅ Complete audit trail

### 4. LangGraph Compatibility

Notice how the LangGraph code is completely standard:

```python
builder = StateGraph(State)
builder.add_node(generate)  # Decorated function works seamlessly
builder.add_edge(START, "generate")
graph = builder.compile()
```

Governance is added through decorators, not by changing the graph structure.

## Running the Example

```bash
# From project root
uv run -m examples.main
```

## Customization Ideas

Try modifying the example:

1. **Change the threshold**: Set `threshold=0.8` for stricter quality control
2. **Add more policies**: Create a `MaxRetriesPolicyRouter` to prevent infinite loops
3. **Different metrics**: Monitor response length, keyword presence, or API scores
4. **Multiple routers**: Add routing for different failure modes

## Next Steps

- Explore the [API Reference](../api/core.md) for all available options
- Read the [Quick Start Guide](../getting-started/quickstart.md) for more examples
- Check the [Installation Guide](../getting-started/installation.md) for setup details
