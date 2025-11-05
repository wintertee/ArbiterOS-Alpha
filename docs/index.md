# ArbiterOS-alpha

**Policy-driven governance layer for LangGraph**

ArbiterOS-alpha is a lightweight governance framework that wraps LangGraph, enabling policy-based validation and dynamic routing without modifying the underlying graph structure.

## Key Features

- 🔒 **Policy-Driven Execution**: Validate execution constraints before and after instruction execution
- 🔀 **Dynamic Routing**: Route execution flow based on policy conditions
- 📊 **Execution History**: Track all instruction executions with timestamps and I/O
- 🎯 **LangGraph-Native**: Minimal migration cost from existing LangGraph code
- 🧩 **Decorator-Based**: Use `@instruction` decorator for lightweight governance
- 🔓 **Zero Lock-In**: Remove ArbiterOS by removing decorators and policies

## Quick Example

```python
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter

# Create ArbiterOS instance
os = ArbiterOSAlpha()

# Add policies
os.add_policy_checker(
    HistoryPolicyChecker(
        name="no_direct_toolcall",
        bad_sequence=["generate", "toolcall"]
    )
)

os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="regenerate_on_low_confidence",
        key="confidence",
        threshold=0.6,
        target="generate"
    )
)

# Decorate your functions
@os.instruction("generate")
def generate(state):
    return {"response": "AI response"}

@os.instruction("evaluate")
def evaluate(state):
    return {"confidence": 0.8}
```

## Design Philosophy

**ArbiterOS wraps LangGraph, not replaces it.**

- **Minimal Migration Cost**: Existing LangGraph code works with minimal changes
- **Native LangGraph API**: Standard `add_node()`, `add_edge()`, `compile()`, `invoke()`, `stream()`
- **Governance is Additive**: Not replacing core functionality

## Getting Started

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Migration Guide](getting-started/migration.md)
- [API Reference](api/core.md)

## Core Concepts

- [Policy Architecture](concepts/policies.md)

## Documentation Structure

- **Getting Started**: Installation and quick start guides
- **Concepts**: Core architectural concepts and design patterns
- **API Reference**: Auto-generated API documentation from code
- **Examples**: Practical examples and use cases
- **Development**: Contributing guidelines and testing documentation
