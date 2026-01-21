# Migration Guide

Two ways to add ArbiterOS to your existing LangGraph code: **manual** or **automated**.

---

## Manual Migration

Only **3 changes** needed to add ArbiterOS governance:

**Before:**
```python
from langgraph.graph import StateGraph, END

def generate(state):
    return {"response": "Hello"}

def verify(state):
    return state

graph = StateGraph(dict)
graph.add_node("generate", generate)
graph.add_node("verify", verify)
app = graph.compile()
```

**After (3 changes):**

```python
from langgraph.graph import StateGraph, END

# ============ Change 1: Import ArbiterOS ============
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.instructions import GENERATE, VERIFY

# ============ Change 2: Create ArbiterOS instance ============
arbiter_os = ArbiterOSAlpha()

# ============ Change 3: Add @arbiter_os.instruction() decorators ============
@arbiter_os.instruction(GENERATE)
def generate(state):
    return {"response": "Hello"}

@arbiter_os.instruction(VERIFY)
def verify(state):
    return state

# Everything else stays exactly the same
graph = StateGraph(dict)
graph.add_node("generate", generate)
graph.add_node("verify", verify)
app = graph.compile()

# Register the compiled graph (required for LangGraph)
arbiter_os.register_compiled_graph(app)
```

**Done.** Your LangGraph code now has:

- ✅ Execution history tracking
- ✅ Policy enforcement (add later)
- ✅ Dynamic routing (add later)

**LangGraph API unchanged** - `add_node()`, `add_edge()`, `compile()`, `invoke()` all work the same.

---

## Automated Migration

For quick migration, use the automated tool:

```bash
uv run -m arbiteros_alpha.migrator path/to/agent.py
```

**What it does:**

1. Parses your file (detects LangGraph or native)
2. Classifies each function using LLM (e.g., `generate` → `GENERATE`)
3. Adds imports, decorators, and OS initialization
4. Creates a backup

**Options:**
```bash
# Preview changes without modifying
uv run -m arbiteros_alpha.migrator agent.py --dry-run

# Manual classification (interactive)
uv run -m arbiteros_alpha.migrator agent.py --manual

# Skip confirmations
uv run -m arbiteros_alpha.migrator agent.py --yes
```

**Before/After Example:**

Before:
```python
from langgraph.graph import StateGraph

def generate(state):
    return {"response": "Hello"}

builder = StateGraph(dict)
builder.add_node("generate", generate)
graph = builder.compile()
```

After (automatic):
```python
from langgraph.graph import StateGraph
from arbiteros_alpha import ArbiterOSAlpha
import arbiteros_alpha.instructions as Instr

arbiter_os = ArbiterOSAlpha(backend="langgraph")

@arbiter_os.instruction(Instr.GENERATE)
def generate(state):
    return {"response": "Hello"}

builder = StateGraph(dict)
builder.add_node("generate", generate)
graph = builder.compile()
arbiter_os.register_compiled_graph(graph)
```

---

## Adding Policies

After migration, add governance policies:

```python
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter

# Prevent skipping verification
arbiter_os.add_policy_checker(
    HistoryPolicyChecker(
        name="require_verification",
        bad_sequence=["generate", "execute"]
    )
)

# Retry on low confidence
arbiter_os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="retry_low_confidence",
        key="confidence",
        threshold=0.7,
        target="generate"
    )
)
```

---

## Rollback

Remove ArbiterOS by deleting imports and decorators. Your original LangGraph code works as before.

---

## Next Steps

- [Policy Architecture](../concepts/policies.md) - Understand policy design
- [Complete Tutorial](../examples/complete-tutorial.md) - Full example
- [API Reference](../api/core.md) - Detailed docs
