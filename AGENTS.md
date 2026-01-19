# AI Development Guidelines for ArbiterOS

This document outlines the standards and best practices that AI agents must follow when contributing to the ArbiterOS project.

## Core Principles

### 1. Always Use `uv run`

All Python commands must be executed using `uv run` to ensure consistent environment management.

For running examples, use `uv run -m example.<example_name>`.

### 2. Google Style Docstrings

All code comments and docstrings must follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

**Important: Use Code Blocks for Examples**

In docstrings, always use triple backticks for code examples instead of doctest format:

```python
# ✅ Good: Use code blocks
def example_function():
    """Short description.
    
    Example:
        ```python
        result = example_function()
        print(result)
        ```
    """
    pass

# ❌ Bad: Don't use doctest format
def example_function():
    """Short description.
    
    Example:
        >>> result = example_function()
        >>> print(result)
    """
    pass
```

**Reason:** Code blocks work better with mkdocstrings and avoid cross-reference warnings during documentation builds.

### 3. Test-Driven Development

**Every new feature MUST include:**
- Unit tests in the `tests/` directory
- Test coverage for all major code paths
- Tests must pass before committing

### 4. Documentation Requirements

Focus on code quality, tests, and Google-style docstrings.

### 5. LangGraph-Native Interface

**ArbiterOS wraps LangGraph, not replaces it.**

**Design Goals:**
- **Minimal Migration Cost:** Existing LangGraph code should work with minimal changes
- **Decorator-Based:** Use `@governed` decorator for lightweight governance
- **Native LangGraph API:** Standard `add_node()`, `add_edge()`, `compile()`, `invoke()`, `stream()`
- **Zero Lock-In:** Users can remove ArbiterOS by removing decorators and policies

**API Principles:**
- Import from LangGraph directly when possible (e.g., `from langgraph.types import Command`)
- Use LangGraph's native classes and types (`StateGraph`, `Command`, `END`)
- ArbiterGraph should be a thin wrapper around `StateGraph`
- Governance is additive, not replacing core functionality

**Before Implementing New Classes:**
1. Check if LangGraph already provides the functionality
2. Reuse LangGraph's implementation when possible
3. Only create new classes when adding governance-specific features
4. Document why LangGraph's class wasn't sufficient

## Development Workflow

1. Write code following Google style with comprehensive docstrings
2. Write unit tests in `tests/`
3. Run tests: `uv run pytest tests/`
4. Verify all tests pass

## Code Quality Standards

- **Type Hints**: All functions must include type hints
- **Error Handling**: Provide clear error messages
- **Naming Conventions**:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

## LangGraph Best Practices

### 1. Reuse LangGraph Classes

**Always check LangGraph's codebase before creating new classes:**

```python
# ✅ Good: Use LangGraph's native types
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint import MemorySaver

# ❌ Bad: Reinventing LangGraph's types
class ArbiterCommand:  # Don't do this - use langgraph.types.Command
    pass
```

### 2. Wrap, Don't Replace

**ArbiterGraph should delegate to StateGraph:**

```python
# ✅ Good: Thin wrapper
class ArbiterGraph:
    def __init__(self, state_schema, policy_config):
        self._state_graph = StateGraph(state_schema)  # Delegate to LangGraph
        self._policy_config = policy_config
    
    def add_node(self, node_id: str, node_fn: Callable, **kwargs):
        # Add governance wrapper, then delegate
        wrapped_fn = self._wrap_if_governed(node_fn)
        self._state_graph.add_node(node_id, wrapped_fn, **kwargs)
    
    def compile(self, **kwargs):
        # Validate policies, then delegate
        self._validate_policy_targets()
        return self._state_graph.compile(**kwargs)

# ❌ Bad: Reimplementing StateGraph from scratch
class ArbiterGraph:
    def __init__(self):
        self._nodes = {}  # Don't reimplement - use StateGraph!
```

### 3. Check LangGraph Source

**Before implementing features, consult LangGraph's codebase:**

- **StateGraph API:** `langgraph/graph/state.py`
- **Command API:** `langgraph/types.py`
- **Checkpoint API:** `langgraph/checkpoint/`
- **Repository:** https://github.com/langchain-ai/langgraph


## Testing Standards

- Unit tests for individual functions/methods
- Integration tests for component interactions
- Test edge cases and error handling
- Use Arrange-Act-Assert pattern
