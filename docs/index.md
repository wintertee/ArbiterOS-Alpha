# ArbiterOS-alpha

**Policy-driven governance layer for LangGraph**

ArbiterOS-alpha is a lightweight governance framework that wraps LangGraph, enabling policy-based validation and dynamic routing without modifying the underlying graph structure.

## Key Features

- 🔒 **Policy-Driven Execution**: Validate execution constraints before and after instruction execution
- 🔀 **Dynamic Routing**: Route execution flow based on policy conditions
- 📊 **Evaluation & Feedback**: Assess node quality with non-blocking evaluators (RL-style rewards)
- 📈 **Execution History**: Track all instruction executions with timestamps and I/O
- 🎯 **LangGraph-Native**: Minimal migration cost from existing LangGraph code
- 🧩 **Decorator-Based**: Use `@instruction` decorator for lightweight governance
- 🔓 **Zero Lock-In**: Remove ArbiterOS by removing decorators and policies

## Quick Example

```python
from arbiteros_alpha import ArbiterOSAlpha, ThresholdEvaluator
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter
from arbiteros_alpha.instructions import CognitiveCore

# Create ArbiterOS instance
os = ArbiterOSAlpha()

# Add policy checker (pre-execution validation)
os.add_policy_checker(
    HistoryPolicyChecker(
        name="no_direct_toolcall",
        bad_sequence=["generate", "toolcall"]
    )
)

# Add policy router (post-execution routing)
os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="regenerate_on_low_confidence",
        key="confidence",
        threshold=0.6,
        target="generate"
    )
)

# Add evaluator (quality assessment)
os.add_evaluator(
    ThresholdEvaluator(
        name="confidence_check",
        key="confidence",
        threshold=0.7,
        target_instructions=[CognitiveCore.GENERATE]
    )
)

# Decorate your functions
@os.instruction(CognitiveCore.GENERATE)
def generate(state):
    return {"response": "AI response", "confidence": 0.85}

@os.instruction(CognitiveCore.EVALUATE)
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

- [Policy Architecture](concepts/policies.md) - PolicyChecker (pre-execution) and PolicyRouter (post-execution)
- [Evaluators](concepts/evaluators.md) - Quality assessment and RL-style reward signals

## Documentation Structure

- **Getting Started**: Installation and quick start guides
- **Concepts**: Core architectural concepts and design patterns
- **API Reference**: Auto-generated API documentation from code
- **Examples**: Practical examples and use cases
- **Development**: Contributing guidelines and testing documentation
