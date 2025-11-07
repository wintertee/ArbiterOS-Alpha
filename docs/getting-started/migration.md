# Migration Guide

## Overview

ArbiterOS is designed to wrap LangGraph with minimal migration cost. This guide shows how to migrate existing LangGraph applications to ArbiterOS step by step.

## Design Principle

**ArbiterOS wraps LangGraph, not replaces it.**

- ✅ Keep your existing LangGraph code structure
- ✅ Use standard LangGraph API (`add_node()`, `add_edge()`, `compile()`)
- ✅ Add governance incrementally with decorators
- ✅ Remove ArbiterOS anytime by removing decorators

## Migration Steps

### Step 1: Original LangGraph Code

Let's start with a typical LangGraph application:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    query: str
    response: str
    confidence: float

def generate(state: State) -> State:
    """Generate AI response."""
    return {
        "response": "AI generated response",
        "confidence": 0.85
    }

def verify(state: State) -> State:
    """Verify response quality."""
    # Verification logic
    return state

def execute(state: State) -> State:
    """Execute the action."""
    # Execution logic
    return state

# Build LangGraph
graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("verify", verify)
graph.add_node("execute", execute)

graph.set_entry_point("generate")
graph.add_edge("generate", "verify")
graph.add_edge("verify", "execute")
graph.add_edge("execute", END)

app = graph.compile()

# Run
result = app.invoke({"query": "user question"})
```

### Step 2: Add ArbiterOS (Minimal Changes)

To add governance, you only need **3 changes**:

```python
from langgraph.graph import StateGraph, END
from langgraph.types import Command  # LangGraph's native Command
from typing import TypedDict

# Change 1: Import ArbiterOS
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter
from arbiteros_alpha.instructions import GENERATE, VERIFY, EXECUTE

class State(TypedDict):
    query: str
    response: str
    confidence: float

# Change 2: Create ArbiterOS instance and add policies
os = ArbiterOSAlpha()

# Add policy: prevent direct generate→execute (require verification)
os.add_policy_checker(
    HistoryPolicyChecker(
        name="require_verification",
        bad_sequence=[GENERATE, EXECUTE]
    )
)

# Add policy: retry on low confidence
os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="retry_on_low_confidence",
        key="confidence",
        threshold=0.7,
        target="generate"
    )
)

# Change 3: Add @os.instruction decorator to functions
@os.instruction(GENERATE)
def generate(state: State) -> State:
    """Generate AI response."""
    return {
        "response": "AI generated response",
        "confidence": 0.85
    }

@os.instruction(VERIFY)
def verify(state: State) -> State:
    """Verify response quality."""
    return state

@os.instruction(EXECUTE)
def execute(state: State) -> State:
    """Execute the action."""
    return state

# Same LangGraph API - no changes!
graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("verify", verify)
graph.add_node("execute", execute)

graph.set_entry_point("generate")
graph.add_edge("generate", "verify")
graph.add_edge("verify", "execute")
graph.add_edge("execute", END)

app = graph.compile()

# Run - governance is now automatic!
result = app.invoke({"query": "user question"})

# Optional: View execution history
from arbiteros_alpha import print_history
print_history(os.history)
```

That's it! Your LangGraph code now has:
- ✅ Policy enforcement (no generate→execute without verify)
- ✅ Dynamic routing (retry if confidence < 0.7)
- ✅ Execution history tracking
- ✅ Auditable logs

## Step 3: Progressive Governance

You can add more governance incrementally as needed:

### Add Custom Policy Checkers

```python
from dataclasses import dataclass
from arbiteros_alpha.policy import PolicyChecker, History

@dataclass
class RateLimitPolicyChecker(PolicyChecker):
    """Prevent API calls if rate limit exceeded."""
    max_calls_per_minute: int = 10
    
    def check(self, history: list[History], current_instruction: str) -> bool:
        if current_instruction != "execute":
            return True
        
        # Count recent executions in last minute
        import time
        one_minute_ago = time.time() - 60
        recent_calls = sum(
            1 for h in history
            if h.instruction == "execute" and h.timestamp > one_minute_ago
        )
        
        if recent_calls >= self.max_calls_per_minute:
            return False
        
        return True

# Add to ArbiterOS
os.add_policy_checker(
    RateLimitPolicyChecker(
        name="api_rate_limit",
        max_calls_per_minute=10
    )
)
```

### Add Custom Policy Routers

```python
from arbiteros_alpha.policy import PolicyRouter

@dataclass
class ErrorFallbackRouter(PolicyRouter):
    """Route to fallback on errors."""
    fallback_target: str = "fallback_node"
    
    def route(self, history: list[History]) -> str | None:
        if not history:
            return None
        
        latest = history[-1]
        if latest.output_state.get("error"):
            return self.fallback_target
        
        return None

# Add to ArbiterOS
os.add_policy_router(
    ErrorFallbackRouter(
        name="error_fallback",
        fallback_target="fallback_generate"
    )
)
```

## Migration Patterns

### Pattern 1: Conditional Edges → Policy Routers

**Before (LangGraph conditional edges):**
```python
def should_continue(state: State) -> str:
    if state["confidence"] < 0.7:
        return "generate"
    return "execute"

graph.add_conditional_edges("verify", should_continue)
```

**After (ArbiterOS policy router):**
```python
# Keep LangGraph edges simple
graph.add_edge("verify", "execute")

# Move logic to policy
os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="retry_logic",
        key="confidence",
        threshold=0.7,
        target="generate"
    )
)
```

### Pattern 2: Guard Functions → Policy Checkers

**Before (Manual guards in functions):**
```python
def execute(state: State) -> State:
    # Manual guard logic
    if state.get("is_high_risk_user"):
        raise ValueError("High risk user - action blocked")
    
    # Actual logic
    return perform_action(state)
```

**After (ArbiterOS policy checker):**
```python
@dataclass
class RiskPolicyChecker(PolicyChecker):
    def check(self, history: list[History], current_instruction: str) -> bool:
        if current_instruction != "execute":
            return True
        
        latest_state = history[-1].output_state if history else {}
        if latest_state.get("is_high_risk_user"):
            return False
        
        return True

os.add_policy_checker(RiskPolicyChecker(name="risk_check"))

# Function is now cleaner
@os.instruction(EXECUTE)
def execute(state: State) -> State:
    # Only business logic - no guards needed
    return perform_action(state)
```

### Pattern 3: Multi-Agent Systems

**Before (LangGraph multi-agent):**
```python
from langgraph.graph import StateGraph

# Multiple agents
agent1_graph = StateGraph(State)
agent2_graph = StateGraph(State)

# Compose them
main_graph = StateGraph(State)
main_graph.add_node("agent1", agent1_graph.compile())
main_graph.add_node("agent2", agent2_graph.compile())
```

**After (Each agent governed by ArbiterOS):**
```python
# Create separate ArbiterOS instances for each agent
os1 = ArbiterOSAlpha()
os2 = ArbiterOSAlpha()

# Each agent has its own policies
os1.add_policy_checker(...)
os2.add_policy_checker(...)

# Decorate agent functions
@os1.instruction("process")
def agent1_process(state): ...

@os2.instruction("process")
def agent2_process(state): ...

# Compose as normal
main_graph = StateGraph(State)
main_graph.add_node("agent1", agent1_process)
main_graph.add_node("agent2", agent2_process)
```

## Rollback Strategy

If you need to remove ArbiterOS, simply:

1. Remove the `@os.instruction()` decorators
2. Remove policy definitions
3. Remove ArbiterOS import

Your original LangGraph code will work exactly as before:

```python
# Remove decorators
# @os.instruction(GENERATE)  ← Remove this
def generate(state: State) -> State:
    return {"response": "AI response"}

# Keep LangGraph code unchanged
graph = StateGraph(State)
graph.add_node("generate", generate)
# ... rest of code stays the same
```

## Best Practices

### 1. Start Small
- Begin with just execution history tracking
- Add one policy at a time
- Test each policy independently

### 2. Separate Concerns
- **Business logic**: Keep in your functions
- **Governance logic**: Move to policies
- **Workflow structure**: Keep in LangGraph

### 3. Test Policies
```python
import pytest

def test_rate_limit_policy():
    """Test that rate limiting policy works correctly."""
    os = ArbiterOSAlpha()
    os.add_policy_checker(RateLimitPolicyChecker(max_calls_per_minute=2))
    
    # First call should pass
    # Second call should pass
    # Third call should be blocked
    # ... test implementation
```

## Common Questions

### Q: Does ArbiterOS change LangGraph's execution?

**A:** No. ArbiterOS wraps your functions with governance logic but doesn't modify LangGraph's core execution. Your graph runs the same way.

### Q: Can I use ArbiterOS with existing LangGraph features?

**A:** Yes. ArbiterOS is compatible with:
- ✅ Conditional edges
- ✅ Checkpointers (MemorySaver, etc.)
- ✅ Streaming
- ✅ Async execution
- ✅ All LangGraph primitives

### Q: Can I use both LangGraph conditional edges and ArbiterOS routers?

**A:** Yes, but we recommend choosing one pattern for clarity:
- Use **LangGraph conditional edges** for core workflow logic
- Use **ArbiterOS policy routers** for governance-based routing (fallbacks, retries, escalations)

## Next Steps

- See [Policy Architecture](../concepts/policies.md) for detailed policy design
- Check [Complete Tutorial](../examples/complete-tutorial.md) for a full example
- Explore [API Reference](../api/core.md) for all available features
